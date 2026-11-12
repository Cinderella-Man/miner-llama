// GGML_CUDA_KEEPALIVE's parts that run without a GPU (ggml/src/ggml-cuda/keepalive-policy.h): the settings, a block's
// decision at each poll, and the controller that starts the cards' blocks after graphs and ends them before captures,
// frees and device-wide synchronizes, and (GGML_CUDA_KEEPALIVE_EAGER, C811) before evaluations that run kernel by kernel
// and work outside a graph at new sizes. A thread stands in for each card's block: the kernel's loop over the same
// decision function, on the host's clock, reading the time and then the word as the kernel does. A block whose start is
// held waits for its stream (mock_stream: when the card's queued work is done), and a card's blocks run in order, so a
// stop at a held start can be timed against the evaluation it is held behind (C817, with C811's controller beside it).
// Usage: test-cuda-keepalive       (exit status 0 when every check passes)
//        test-cuda-keepalive gpu   on a CUDA card (C811's and C817's check on a rig): the first evaluation of kernels
//                                  never launched before, right after a graph whose block runs, timed; a memset and a
//                                  set at sizes new to the card; and (C817) a set at a new size right after a long graph
//                                  is queued whose first call runs kernel by kernel, the card's next start held behind
//                                  it. It reads GGML_CUDA_KEEPALIVE (default decode), GGML_CUDA_KEEPALIVE_MS (default
//                                  2000) and GGML_CUDA_KEEPALIVE_EAGER from the environment; with EAGER at 1 or 2 none of
//                                  the first three may take a quarter of the timeout, and the last set not a quarter of
//                                  the long graph's time. Without it the times are printed (a first launch under lazy
//                                  module loading waits for the block: about the timeout)

#include "../ggml/src/ggml-cuda/keepalive-policy.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static unsigned long long now_ns() {
    return (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// a card's compute stream, as a held start sees it (launch's after): the work queued on it so far is done at busy_until
struct mock_stream {
    std::atomic<unsigned long long> busy_until{0}; // now_ns() time; 0: idle

    // an evaluation of ms milliseconds queued now, after the work already queued (the stream's order)
    void queue_ms(int ms) {
        const unsigned long long t = now_ns();
        busy_until = std::max(t, busy_until.load()) + (unsigned long long) ms * 1000000ull;
    }
};

// the cards: a word, a report and a thread each. A card's blocks run in order, as on its keep-alive stream: a block
// begins after the one launched before it has ended, and a held one (after) also after its stream's work so far
// (C817: a block the controller ended before it began is not waited for, and the next one queues behind it)
struct mock_ops {
    static constexpr int N = 4;
    struct card {
        std::mutex                 m;
        uint64_t                   word = 0;
        ggml_cuda_keepalive_report report{};
        std::thread                th;
        std::atomic<bool>          ended{true};      // the newest block has ended
        std::atomic<bool>          force_end{false}; // the test ends the block as its timeout would
        std::atomic<uint32_t>      beat_seen{0};
        int                        launches = 0;
        int                        writes   = 0;
        int                        launches_after = 0;       // launches that waited for a stream (EAGER's held starts)
        void *                     last_after     = nullptr; // the last launch's stream
        unsigned long long         gate   = 0;               // the newest block begins no earlier (ns): a held start's
        int                        newest = 0;               // the newest block's number (launches so far)
        std::atomic<int>           ran{0};                   // the number of the last block that ran (polled its run on)
    };
    card cards[N];
    std::atomic<int> bad_launches{0}; // a launch while the card's last block was not waited for and had begun

    void write(int dev, uint64_t w) {
        std::lock_guard<std::mutex> lock(cards[dev].m);
        cards[dev].word = w;
        cards[dev].writes++;
    }

    bool launch(int dev, uint32_t gen, uint32_t beat, uint32_t idle_ms, uint32_t poll_us, void * after) {
        card & c = cards[dev];
        // a block not waited for is allowed only if it never ran: one ended before it began (C817), which ends at its
        // first poll; a block that ran must have been waited for
        if (c.th.joinable() && c.ran.load() == c.newest) {
            bad_launches++;
        }
        c.launches++;
        c.launches_after += after != nullptr;
        c.last_after      = after;
        if (after != nullptr) {
            c.gate = std::max(c.gate, ((mock_stream *) after)->busy_until.load());
        }
        const unsigned long long gate = c.gate;
        int id;
        {
            std::lock_guard<std::mutex> lock(c.m);
            id           = ++c.newest;
            c.ended      = false;
            c.report.gen = 0;
        }
        c.force_end = false;
        std::thread prev = std::move(c.th);
        c.th = std::thread([&c, gen, beat, idle_ms, poll_us, gate, id, prev = std::move(prev)]() mutable {
            if (prev.joinable()) {
                prev.join(); // the block launched before this one, on the same stream
            }
            while (now_ns() < gate && !c.force_end) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            const unsigned long long t0 = now_ns();
            ggml_cuda_keepalive_watch w = { beat, t0 };
            unsigned long long polls = 0;
            int r;
            for (;;) {
                unsigned long long t;
                uint64_t v;
                {
                    std::lock_guard<std::mutex> lock(c.m);
                    t = now_ns();
                    v = c.word;
                }
                polls++;
                r = ggml_cuda_keepalive_poll(v, gen, t, (unsigned long long) idle_ms * 1000000ull, w);
                if (r == GGML_CUDA_KEEPALIVE_RUN && c.force_end) {
                    r = GGML_CUDA_KEEPALIVE_TIMEOUT;
                }
                if (r != GGML_CUDA_KEEPALIVE_RUN) {
                    break;
                }
                c.ran       = id;
                c.beat_seen = w.beat;
                std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
            }
            std::lock_guard<std::mutex> lock(c.m);
            c.report.reason = (uint32_t) r;
            c.report.polls  = polls;
            c.report.ns     = now_ns() - t0;
            for (uint32_t & h : c.report.hist) {
                h = 0;
            }
            c.report.hist[1885 / 50] = 1; // one sample at a P106-090's boost clock
            c.report.gen = gen;
            if (c.newest == id) {
                c.ended = true;
            }
        });
        return true;
    }

    bool done(int dev) {
        return cards[dev].ended.load();
    }

    void wait(int dev) {
        if (cards[dev].th.joinable()) {
            cards[dev].th.join();
        }
    }

    // the newest block has not begun: the stream it (or a block it queues behind) was held for is still busy
    std::atomic<int> pending_calls{0};
    bool pending(int dev) {
        pending_calls++;
        return now_ns() < cards[dev].gate;
    }

    const ggml_cuda_keepalive_report & report(int dev) {
        return cards[dev].report;
    }

    ~mock_ops() {
        for (card & c : cards) {
            c.force_end = true;
            if (c.th.joinable()) {
                c.th.join();
            }
        }
    }
};

using ctl_t = ggml_cuda_keepalive_ctl<mock_ops, mock_ops::N>;

static ggml_cuda_keepalive_config cfg_of(const char * mode, const char * ms = nullptr, const char * poll = "50") {
    return ggml_cuda_keepalive_parse(mode, ms, poll);
}

static void test_parse() {
    ggml_cuda_keepalive_config c = ggml_cuda_keepalive_parse(nullptr, nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_OFF && c.idle_ms == 1000 && c.poll_us == 50);
    CHECK(ggml_cuda_keepalive_parse("", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    CHECK(ggml_cuda_keepalive_parse("0", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    CHECK(ggml_cuda_keepalive_parse("on", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    CHECK(ggml_cuda_keepalive_parse("decodes", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_OFF);
    c = ggml_cuda_keepalive_parse("1", nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_DECODE && c.idle_ms == 1000);
    CHECK(ggml_cuda_keepalive_parse("decode", nullptr, nullptr).mode == GGML_CUDA_KEEPALIVE_DECODE);
    c = ggml_cuda_keepalive_parse("request", nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_REQUEST && c.idle_ms == 1000);
    c = ggml_cuda_keepalive_parse("idle", nullptr, nullptr);
    CHECK(c.mode == GGML_CUDA_KEEPALIVE_IDLE && c.idle_ms == 120000);
    c = ggml_cuda_keepalive_parse("idle", "5000", "200");
    CHECK(c.idle_ms == 5000 && c.poll_us == 200);
    CHECK(ggml_cuda_keepalive_parse("decode", "3", nullptr).idle_ms == 10);
    CHECK(ggml_cuda_keepalive_parse("decode", "99999999", nullptr).idle_ms == 3600000);
    CHECK(ggml_cuda_keepalive_parse("decode", "", nullptr).idle_ms == 1000);
    CHECK(ggml_cuda_keepalive_parse("decode", nullptr, "0").poll_us == 1);
    CHECK(ggml_cuda_keepalive_parse("decode", nullptr, "1000000").poll_us == 100000);
    CHECK(strcmp(ggml_cuda_keepalive_mode_name(GGML_CUDA_KEEPALIVE_REQUEST), "request") == 0);
}

static void test_poll() {
    ggml_cuda_keepalive_watch w = { 5, 1000 };
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 5), 7, 1500, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 5), 7, 1999, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 5), 7, 2000, 1000, w) == GGML_CUDA_KEEPALIVE_TIMEOUT);
    // a new beat resets the timeout, however late
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 6), 7, 2500, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    CHECK(w.beat == 6 && w.t_beat == 2500);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(7, 6), 7, 3499, 1000, w) == GGML_CUDA_KEEPALIVE_RUN);
    // the host's stop wins over a beat and over a timeout
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(0, 7), 7, 2600, 1000, w) == GGML_CUDA_KEEPALIVE_STOPPED);
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(0, 6), 7, 9999, 1000, w) == GGML_CUDA_KEEPALIVE_STOPPED);
    // another run's generation stops this one
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(8, 6), 7, 2600, 1000, w) == GGML_CUDA_KEEPALIVE_STOPPED);
    // the beat wraps
    ggml_cuda_keepalive_watch v = { 0xffffffffu, 0 };
    CHECK(ggml_cuda_keepalive_poll(ggml_cuda_keepalive_word(1, 0), 1, 5000, 1000, v) == GGML_CUDA_KEEPALIVE_RUN && v.beat == 0);
}

static void test_stats() {
    ggml_cuda_keepalive_stats s;
    ggml_cuda_keepalive_report r{};
    r.reason = GGML_CUDA_KEEPALIVE_STOPPED;
    r.ns     = 2000000000ull;
    r.hist[1354 / 50] = 3; // 1350-1399: base
    r.hist[1885 / 50] = 7; // 1850-1899: boost
    s.add(r);
    r.reason = GGML_CUDA_KEEPALIVE_TIMEOUT;
    r.hist[1354 / 50] = 0;
    r.hist[1885 / 50] = 10;
    s.add(r);
    CHECK(s.stopped == 1 && s.timeouts == 1 && s.samples() == 20);
    CHECK(s.seconds > 3.99 && s.seconds < 4.01);
    CHECK(s.share(0, 1400) == 3.0 / 20 && s.share(1800, 100000) == 17.0 / 20);
    CHECK(s.mhz_quantile(0.1) == 1375 && s.mhz_quantile(0.5) == 1875 && s.mhz_quantile(0.9) == 1875);
}

static void test_off() {
    mock_ops ops;
    ctl_t ka(cfg_of(nullptr), ops);
    CHECK(!ka.on());
    ka.graph_begin(0, true);
    ka.graph_end(0);
    ka.graph_end(1);
    ka.kick();
    ka.quiesce();
    for (const auto & c : ops.cards) {
        CHECK(c.launches == 0 && c.writes == 0);
    }
    CHECK(ka.take_stats(0).starts == 0);
}

static void test_decode() {
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "1000"), ops);
    ka.graph_end(0);
    CHECK(ops.cards[0].launches == 1 && ka.running(0));
    const uint32_t gen0 = ka.devs[0].gen;
    for (int i = 0; i < 20; ++i) {
        ka.graph_end(0);
        sleep_ms(2);
    }
    CHECK(ops.cards[0].launches == 1);
    sleep_ms(20);
    CHECK(ops.cards[0].beat_seen == ka.devs[0].beat); // the block follows the beats
    ka.graph_begin(0, false);
    CHECK(ka.running(0) && !ops.cards[0].ended);
    // a capture ends the block, and it reports that the host ended it
    ka.graph_begin(0, true);
    CHECK(!ka.running(0) && ops.cards[0].ended);
    CHECK(ops.cards[0].report.reason == GGML_CUDA_KEEPALIVE_STOPPED && ops.cards[0].report.gen == gen0);
    // the next graph starts a new run
    ka.graph_end(0);
    CHECK(ops.cards[0].launches == 2 && ka.running(0) && ka.devs[0].gen != gen0);
    ka.graph_end(1);
    CHECK(ops.cards[1].launches == 1 && ops.cards[2].launches == 0);
    // a free ends every block
    ka.quiesce();
    CHECK(!ka.running(0) && !ka.running(1) && ops.cards[0].ended && ops.cards[1].ended);
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 2 && s.stopped == 2 && s.timeouts == 0 && s.lost == 0 && s.samples() == 2);
    CHECK(ka.take_stats(0).starts == 0); // taken stats are reset
    CHECK(ka.take_stats(1).starts == 1);
    CHECK(ops.bad_launches == 0);
}

static void test_timeout() {
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "50"), ops);
    ka.graph_end(0);
    sleep_ms(300);
    CHECK(ops.cards[0].ended); // the block ended itself
    CHECK(ka.running(0));      // the host has not looked yet
    // a beat after a gap of the timeout ends the old run (already over: its report says timeout) and starts one
    ka.graph_end(0);
    CHECK(ops.cards[0].launches == 2 && ka.running(0));
    ka.quiesce();
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 2 && s.timeouts == 1 && s.stopped == 1 && s.lost == 0);
    CHECK(ops.bad_launches == 0);
}

static void test_beats_hold() {
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "1000"), ops);
    // beats every 5 ms for 1.5 s, past the 1 s timeout: one run throughout
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(1500)) {
        ka.graph_end(0);
        sleep_ms(5);
    }
    CHECK(ops.cards[0].launches == 1 && !ops.cards[0].ended);
    ka.quiesce();
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 1 && s.stopped == 1 && s.timeouts == 0);
}

static void test_check() {
    // a block that ended while the host thought it running (as after a held-up host thread) is found by the beats'
    // check within GGML_CUDA_KEEPALIVE_CHECK_MS, and a new one started
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "60000"), ops);
    ka.graph_end(0);
    ops.cards[0].force_end = true;
    while (!ops.cards[0].ended) {
        sleep_ms(1);
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (ops.cards[0].launches < 2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
        ka.graph_end(0);
        sleep_ms(10);
    }
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ops.cards[0].launches == 2 && ka.running(0));
    CHECK(took <= GGML_CUDA_KEEPALIVE_CHECK_MS + 200);
    ka.quiesce();
    ggml_cuda_keepalive_stats s = ka.take_stats(0);
    CHECK(s.starts == 2 && s.timeouts == 1 && s.stopped == 1);
    CHECK(ops.bad_launches == 0);
}

static void test_request() {
    {
        mock_ops ops;
        ctl_t ka(cfg_of("request"), ops);
        ka.kick(); // no card has computed a graph yet: nothing starts
        for (const auto & c : ops.cards) {
            CHECK(c.launches == 0);
        }
        ka.graph_end(0);
        ka.graph_end(2);
        ka.quiesce();
        // a request starts the blocks of the cards that have computed graphs, ahead of the prompt
        ka.kick();
        CHECK(ops.cards[0].launches == 2 && ops.cards[2].launches == 2);
        CHECK(ops.cards[1].launches == 0 && ops.cards[3].launches == 0);
        CHECK(ka.running(0) && ka.running(2));
        ka.kick(); // a second kick while they run: beats only
        CHECK(ops.cards[0].launches == 2 && ops.cards[2].launches == 2);
        ka.quiesce();
    }
    {
        mock_ops ops;
        ctl_t ka(cfg_of("decode"), ops);
        ka.graph_end(0);
        ka.quiesce();
        ka.kick(); // only request mode kicks
        CHECK(ops.cards[0].launches == 1 && !ka.running(0));
    }
}

static void test_idle() {
    mock_ops ops;
    ctl_t ka(cfg_of("idle"), ops);
    CHECK(ka.cfg.idle_ms == 120000);
    ka.graph_end(1);
    sleep_ms(200);
    CHECK(ka.running(1) && !ops.cards[1].ended); // the block outlives the gap between requests
    ka.quiesce();
    CHECK(ops.cards[1].ended && ops.cards[1].report.reason == GGML_CUDA_KEEPALIVE_STOPPED);
}

static void test_threads() {
    // graphs, captures and frees from three threads at once: every run is accounted for, none is lost
    mock_ops ops;
    ctl_t ka(cfg_of("decode", "20", "20"), ops);
    std::atomic<bool> stop{false};
    std::thread graphs([&] {
        for (int i = 0; i < 3000; ++i) {
            ka.graph_end(i % 3);
            if (i % 50 == 0) {
                sleep_ms(30); // past the timeout now and then
            }
        }
        stop = true;
    });
    std::thread captures([&] {
        while (!stop) {
            ka.graph_begin(1, true);
            sleep_ms(1);
        }
    });
    std::thread frees([&] {
        while (!stop) {
            ka.quiesce();
            sleep_ms(3);
        }
    });
    graphs.join();
    captures.join();
    frees.join();
    ka.quiesce();
    uint64_t starts = 0;
    for (int dev = 0; dev < mock_ops::N; ++dev) {
        CHECK(!ka.running(dev));
        const ggml_cuda_keepalive_stats s = ka.take_stats(dev);
        CHECK(s.lost == 0 && s.starts == s.stopped + s.timeouts);
        CHECK(s.starts == (uint64_t) ops.cards[dev].launches);
        starts += s.starts;
    }
    CHECK(starts > 3);
    CHECK(ops.bad_launches == 0);
}

static ggml_cuda_keepalive_config cfg_eager(const char * mode, const char * eager, const char * ms = "1000") {
    ggml_cuda_keepalive_config c = ggml_cuda_keepalive_parse(mode, ms, "50");
    c.eager = ggml_cuda_keepalive_parse_eager(eager);
    return c;
}

static void test_eager_parse() {
    CHECK(ggml_cuda_keepalive_parse_eager(nullptr) == -1);
    CHECK(ggml_cuda_keepalive_parse_eager("") == -1);
    CHECK(ggml_cuda_keepalive_parse_eager("0") == 0);
    CHECK(ggml_cuda_keepalive_parse_eager("1") == 1);
    CHECK(ggml_cuda_keepalive_parse_eager("2") == 2);
    CHECK(ggml_cuda_keepalive_parse_eager("3") == 0);
    CHECK(ggml_cuda_keepalive_parse_eager("on") == 0);
    CHECK(ggml_cuda_keepalive_parse(nullptr, nullptr, nullptr).eager == -1); // the plain settings leave it unset
    CHECK(ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_SET, 4096) !=
          ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_GET, 4096));
}

// the switch unset or at 0: the controller does what it did before (no stop, no held start), and at 0 only counts
static void test_eager_unset() {
    for (const char * e : { (const char *) nullptr, "0" }) {
        mock_ops ops;
        mock_stream dummy;
        ctl_t ka(cfg_eager("decode", e), ops);
        ka.graph_end(0, &dummy);
        CHECK(ops.cards[0].launches == 1 && ops.cards[0].launches_after == 0 && ka.running(0));
        CHECK(!ka.eager_begin(0, 7));
        CHECK(!ka.outside(0, ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_MEMSET, 64)));
        CHECK(ka.running(0) && !ops.cards[0].ended);
        ka.graph_end(0, &dummy);
        CHECK(ops.cards[0].launches == 1);
        const ggml_cuda_keepalive_counts c = ka.counts();
        CHECK(c.eager == (e == nullptr ? 0u : 1u) && c.eager_new == 0 && c.out_new == 0 && c.held == 0);
        ka.quiesce();
        CHECK(ops.pending_calls == 0 && ka.counts().cancelled == 0); // C817's check is not reached
    }
    // the keep-alive off: nothing, whatever the switch says
    mock_ops ops;
    ctl_t ka(cfg_eager(nullptr, "1"), ops);
    CHECK(!ka.eager_begin(0, 7) && !ka.outside(0, 1));
    ka.graph_end(0, &ops);
    CHECK(ops.cards[0].launches == 0 && ops.cards[0].writes == 0);
}

// level 1: a block is ended before an evaluation run kernel by kernel at shapes new to the card, and its next start
// waits for that evaluation's stream; shapes seen before change nothing
static void test_eager_new_shapes() {
    mock_ops ops;
    mock_stream stream_a, stream_b;
    ctl_t ka(cfg_eager("decode", "1"), ops);
    // the first evaluation on a card: no block runs, the start after it is held
    CHECK(ka.eager_begin(0, 100));
    CHECK(ops.cards[0].launches == 0 && !ka.running(0));
    ka.graph_end(0, &stream_a);
    CHECK(ops.cards[0].launches == 1 && ops.cards[0].launches_after == 1 && ops.cards[0].last_after == &stream_a);
    CHECK(ka.running(0));
    // replays (no eager_begin) keep the block
    for (int i = 0; i < 5; ++i) {
        ka.graph_end(0, &stream_a);
    }
    CHECK(ops.cards[0].launches == 1);
    const uint32_t gen0 = ka.devs[0].gen;
    // the same shapes again kernel by kernel: nothing
    CHECK(!ka.eager_begin(0, 100));
    CHECK(ka.running(0) && !ops.cards[0].ended);
    ka.graph_end(0, &stream_a);
    CHECK(ops.cards[0].launches == 1);
    // new shapes: the running block is ended (its report says the host ended it) and the next start waits
    CHECK(ka.eager_begin(0, 200));
    CHECK(!ka.running(0) && ops.cards[0].ended);
    CHECK(ops.cards[0].report.reason == GGML_CUDA_KEEPALIVE_STOPPED && ops.cards[0].report.gen == gen0);
    ka.graph_end(0, &stream_b);
    CHECK(ops.cards[0].launches == 2 && ops.cards[0].launches_after == 2 && ops.cards[0].last_after == &stream_b);
    // the hold is spent: a block that ends later starts again at once
    ka.quiesce();
    ka.graph_end(0, &stream_b);
    CHECK(ops.cards[0].launches == 3 && ops.cards[0].launches_after == 2 && ops.cards[0].last_after == nullptr);
    // shapes are per card: card 1 has not run shape 100
    ka.graph_end(1, &stream_a);
    CHECK(ka.eager_begin(1, 100) && !ka.running(1));
    // a graph end without a stream (the old call) starts a held block at once and keeps the hold
    ka.graph_end(1);
    CHECK(ops.cards[1].launches == 2 && ops.cards[1].launches_after == 0 && ka.devs[1].hold);
    ka.quiesce();
    ka.graph_end(1, &stream_a);
    CHECK(ops.cards[1].launches == 3 && ops.cards[1].launches_after == 1 && !ka.devs[1].hold);
    const ggml_cuda_keepalive_counts c = ka.counts();
    CHECK(c.eager == 4 && c.eager_new == 3 && c.eager_stops == 2 && c.held == 3 && c.out_new == 0);
    CHECK(ka.counts(0).eager_new == 2 && ka.counts(1).eager_new == 1);
    ka.quiesce();
    const ggml_cuda_keepalive_counts e = ka.counts();
    CHECK(e.starts == 6 && e.stopped == 6 && e.timeouts == 0 && e.lost == 0 && e.running == 0);
    CHECK(ops.bad_launches == 0);
}

// level 2: every evaluation run kernel by kernel ends the block
static void test_eager_every() {
    mock_ops ops;
    mock_stream s;
    ctl_t ka(cfg_eager("decode", "2"), ops);
    ka.graph_end(0, &s);
    for (int i = 0; i < 3; ++i) {
        CHECK(ka.eager_begin(0, 100));
        CHECK(!ka.running(0));
        ka.graph_end(0, &s);
        CHECK(ka.running(0));
    }
    CHECK(ops.cards[0].launches == 4 && ops.cards[0].launches_after == 3);
    const ggml_cuda_keepalive_counts c = ka.counts();
    CHECK(c.eager == 3 && c.eager_new == 3 && c.eager_stops == 3);
    // work outside a graph still acts on new sizes only
    CHECK(ka.outside(0, ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_SET, 8)));
    CHECK(!ka.outside(0, ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_SET, 8)));
    ka.quiesce();
}

// work outside a graph: a size new to the card ends its block and holds the next start until its next evaluation
static void test_outside() {
    mock_ops ops;
    mock_stream s;
    ctl_t ka(cfg_eager("decode", "1"), ops);
    ka.graph_end(2, &s);
    CHECK(ka.running(2) && ops.cards[2].launches_after == 0);
    const uint64_t k1 = ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_MEMSET, 4096);
    const uint64_t k2 = ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_MEMSET, 8192);
    CHECK(ka.outside(2, k1));
    CHECK(!ka.running(2) && ops.cards[2].ended);
    CHECK(!ka.outside(2, k1)); // the same size again
    ka.graph_end(2, &s);
    CHECK(ops.cards[2].launches == 2 && ops.cards[2].launches_after == 1);
    CHECK(!ka.outside(2, k1) && ka.running(2));
    // a new size while no block runs: no stop, still a held start
    ka.quiesce();
    CHECK(ka.outside(2, k2));
    ka.graph_end(2, &s);
    CHECK(ops.cards[2].launches == 3 && ops.cards[2].launches_after == 2);
    ka.quiesce();
    const ggml_cuda_keepalive_counts c = ka.counts(2);
    CHECK(c.out_new == 2 && c.out_stops == 1 && c.held == 2 && c.eager == 0);
    CHECK(ops.bad_launches == 0);
}

// request mode's kick starts a card without waiting for a stream, and keeps a hold for the next start after a graph
static void test_eager_request_kick() {
    mock_ops ops;
    mock_stream s;
    ctl_t ka(cfg_eager("request", "1"), ops);
    ka.graph_end(0, &s);
    ka.quiesce();
    CHECK(ka.outside(0, 5));
    ka.kick();
    CHECK(ka.running(0) && ops.cards[0].launches == 2 && ops.cards[0].launches_after == 0 && ka.devs[0].hold);
    ka.graph_end(0, &s); // the block runs: a beat; the hold waits for the next start
    CHECK(ops.cards[0].launches == 2 && ka.devs[0].hold);
    ka.quiesce();
}

// a block ended by its timeout is counted by counts(), which reads the card's state as the beats' check does
static void test_eager_counts_timeout() {
    mock_ops ops;
    ctl_t ka(cfg_eager("decode", "0", "30"), ops);
    ka.graph_end(1);
    sleep_ms(200);
    const ggml_cuda_keepalive_counts c = ka.counts();
    CHECK(c.starts == 1 && c.timeouts == 1 && c.running == 0 && !ka.running(1));
    ka.graph_end(1);
    ka.quiesce();
    const ggml_cuda_keepalive_counts d = ka.counts();
    CHECK(d.starts == 2 && d.timeouts == 1 && d.stopped == 1);
    // take_stats resets the per-card stats, never the counts
    CHECK(ka.take_stats(1).starts == 2 && ka.counts().starts == 2);
}

// graphs, evaluations kernel by kernel, work outside graphs, captures and frees from five threads at once
static void test_eager_threads() {
    mock_ops ops;
    mock_stream s;
    ctl_t ka(cfg_eager("decode", "1", "20"), ops);
    std::atomic<bool> stop{false};
    std::thread graphs([&] {
        for (int i = 0; i < 3000; ++i) {
            if (i % 7 == 0) {
                ka.eager_begin(i % 3, (uint64_t) (i % 40));
            }
            ka.graph_end(i % 3, &s);
            if (i % 50 == 0) {
                sleep_ms(30);
            }
        }
        stop = true;
    });
    std::thread outs([&] {
        uint64_t k = 0;
        while (!stop) {
            ka.outside((int) (k % 3), ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_SET, k % 50));
            k++;
            sleep_ms(1);
        }
    });
    std::thread captures([&] {
        while (!stop) {
            ka.graph_begin(1, true);
            sleep_ms(2);
        }
    });
    std::thread frees([&] {
        while (!stop) {
            ka.quiesce();
            ka.counts();
            sleep_ms(3);
        }
    });
    graphs.join();
    outs.join();
    captures.join();
    frees.join();
    ka.quiesce();
    const ggml_cuda_keepalive_counts c = ka.counts();
    CHECK(c.lost == 0 && c.running == 0 && c.starts == c.stopped + c.timeouts);
    CHECK(c.eager_new <= 3 * 40 && c.out_new <= 3 * 50 && c.held > 0);
    uint64_t launches = 0;
    for (const auto & card : ops.cards) {
        launches += card.launches;
    }
    CHECK(c.starts == launches);
    CHECK(ops.bad_launches == 0);
}

static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// until every stream's work is done and the blocks held behind it have begun
static void wait_streams(const mock_stream * s, int n) {
    unsigned long long until = 0;
    for (int i = 0; i < n; ++i) {
        until = std::max(until, s[i].busy_until.load());
    }
    while (now_ns() < until + 20000000ull) {
        sleep_ms(5);
    }
}

// C817: a stop on a card whose block has not begun. An evaluation kernel by kernel at a new shape ends the card's
// block and holds the next start behind the evaluation (300 ms of the card's time, just queued); the copy of its
// output to the next card, at a size new to this card, then stops that held block. C811's controller (cancel_held
// off) waited for the block, and so for the evaluation it is held behind; C817's leaves it to end by itself at its
// first poll, once the evaluation is done
static void test_held_stop() {
    for (const bool cancel : { false, true }) {
        mock_ops ops;
        mock_stream ev;
        ctl_t ka(cfg_eager("decode", "1"), ops);
        ka.cancel_held = cancel;
        ka.graph_end(0, &ev); // an earlier graph: the card's block runs
        CHECK(ka.running(0) && !ops.pending(0));
        CHECK(ka.eager_begin(0, 100));
        CHECK(!ka.running(0));
        ev.queue_ms(300);
        ka.graph_end(0, &ev);
        const uint32_t gen1 = ka.devs[0].gen;
        CHECK(ka.running(0) && ops.pending(0) && ops.cards[0].launches == 2 && ops.cards[0].launches_after == 1);
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(ka.outside(0, ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_SEND, 4096)));
        const double took = elapsed_ms(t0);
        CHECK(!ka.running(0));
        printf("test_held_stop: %s: the stop at a held start took %.1f ms (the evaluation it is held behind: 300 ms)\n",
               cancel ? "C817" : "C811's controller", took);
        ggml_cuda_keepalive_counts c = ka.counts();
        CHECK(c.starts == 2 && c.stopped == 2 && c.out_stops == 1 && c.lost == 0);
        if (cancel) {
            CHECK(took < 50);
            CHECK(c.cancelled == 1);
            CHECK(ops.pending(0) && !ops.cards[0].ended); // the block has not begun
        } else {
            CHECK(took > 200); // it waited for the evaluation
            CHECK(c.cancelled == 0);
        }
        // either way the block began after its evaluation and ended at its first poll, by the host, without running
        wait_streams(&ev, 1);
        CHECK(ops.cards[0].ended && ops.cards[0].report.gen == gen1 && ops.cards[0].report.polls == 1);
        CHECK(ops.cards[0].report.reason == GGML_CUDA_KEEPALIVE_STOPPED && ops.cards[0].ran == 1);
        // the next evaluation's start is held again (the copy's size was new) and runs once that evaluation is done
        ev.queue_ms(20);
        ka.graph_end(0, &ev);
        CHECK(ka.running(0) && ops.cards[0].launches == 3 && ops.cards[0].launches_after == 2);
        wait_streams(&ev, 1);
        sleep_ms(5);
        CHECK(ops.cards[0].ran == 3 && !ops.cards[0].ended);
        ka.quiesce();
        c = ka.counts();
        CHECK(c.starts == 3 && c.stopped == 3 && c.timeouts == 0 && c.lost == 0 && c.running == 0);
        CHECK(c.cancelled == (cancel ? 1u : 0u));
        CHECK(ops.bad_launches == 0);
    }
}

// C817: a prompt's first chunk at sizes new to the cards, as R131 queued it: on each card in turn an evaluation kernel
// by kernel at a new shape (its next start held), then the copy of its output to the next card, a size new to both.
// Each evaluation takes 60 ms of its card's time and begins when the previous card's ends. C811's controller queued a
// card only once the card before it was done (the source card's stop at the copy waited for its held block); C817's
// queues the four cards at once
static void test_held_chunk() {
    const int N = mock_ops::N;
    for (const bool cancel : { false, true }) {
        mock_ops ops;
        mock_stream ev[mock_ops::N];
        ctl_t ka(cfg_eager("decode", "1"), ops);
        ka.cancel_held = cancel;
        for (int k = 0; k < N; ++k) {
            ka.graph_end(k, &ev[k]); // the blocks run, as at a request's start
        }
        const uint64_t send = ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_SEND, 3 * 1024 * 1024 + 4);
        const uint64_t recv = ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_RECV, 3 * 1024 * 1024 + 4);
        double entry[mock_ops::N];
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < N; ++k) {
            entry[k] = elapsed_ms(t0);
            CHECK(ka.eager_begin(k, 1000 + k));
            const unsigned long long prev_end = k > 0 ? ev[k - 1].busy_until.load() : 0;
            ev[k].busy_until = std::max(now_ns(), prev_end) + 60000000ull;
            ka.graph_end(k, &ev[k]);
            if (k + 1 < N) {
                CHECK(ka.outside(k, send));
                CHECK(ka.outside(k + 1, recv));
            }
        }
        const double all = elapsed_ms(t0);
        printf("test_held_chunk: %s: each card queued at %.1f, %.1f, %.1f, %.1f ms (all in %.1f ms; 60 ms a card)\n",
               cancel ? "C817" : "C811's controller", entry[0], entry[1], entry[2], entry[3], all);
        ggml_cuda_keepalive_counts c = ka.counts();
        if (cancel) {
            CHECK(all < 50);
            CHECK(c.cancelled == (uint64_t) (N - 1));
        } else {
            CHECK(entry[N - 1] > 150); // one evaluation a card
            CHECK(c.cancelled == 0);
        }
        wait_streams(ev, N);
        ka.quiesce();
        c = ka.counts();
        CHECK(c.lost == 0 && c.running == 0 && c.timeouts == 0 && c.starts == c.stopped);
        CHECK(c.eager_new == (uint64_t) N && c.out_new == (uint64_t) (2 * (N - 1)) && c.held == (uint64_t) N);
        CHECK(ops.bad_launches == 0);
    }
}

// C817: test_eager_threads with evaluations that keep the cards busy (0 to 3 ms each, queued in order), so that
// stops, captures and frees from other threads meet starts still held: every start is accounted for, none is lost,
// and no block that ran was launched over without a wait
static void test_held_threads() {
    mock_ops ops;
    mock_stream s[3];
    ctl_t ka(cfg_eager("decode", "1", "20"), ops);
    std::atomic<bool> stop{false};
    std::thread graphs([&] {
        for (int i = 0; i < 3000; ++i) {
            const int dev = i % 3;
            if (i % 4 == 0) {
                ka.eager_begin(dev, (uint64_t) (i % 400));
            }
            s[dev].queue_ms(i % 4);
            ka.graph_end(dev, &s[dev]);
            if (i % 50 == 0) {
                sleep_ms(30);
            }
        }
        stop = true;
    });
    std::thread outs([&] {
        uint64_t k = 0;
        while (!stop) {
            ka.outside((int) (k % 3), ggml_cuda_keepalive_outside_key(GGML_CUDA_KEEPALIVE_OUT_SEND, k % 90));
            k++;
            sleep_ms(1);
        }
    });
    std::thread captures([&] {
        while (!stop) {
            ka.graph_begin(1, true);
            sleep_ms(2);
        }
    });
    std::thread frees([&] {
        while (!stop) {
            ka.quiesce();
            ka.counts();
            sleep_ms(3);
        }
    });
    graphs.join();
    outs.join();
    captures.join();
    frees.join();
    ka.quiesce();
    wait_streams(s, 3);
    const ggml_cuda_keepalive_counts c = ka.counts();
    printf("test_held_threads: %llu runs, %llu ended before they began, %llu ended by the timeout\n",
           (unsigned long long) c.starts, (unsigned long long) c.cancelled, (unsigned long long) c.timeouts);
    CHECK(c.lost == 0 && c.running == 0 && c.starts == c.stopped + c.timeouts);
    CHECK(c.cancelled > 0 && c.cancelled <= c.stopped);
    uint64_t launches = 0;
    for (auto & card : ops.cards) {
        launches += card.launches;
        // a block left without a wait (ended before it began) never ran
        if (card.th.joinable()) {
            card.th.join();
            CHECK(card.ran != card.newest);
        }
        CHECK(card.ended);
    }
    CHECK(c.starts == launches);
    CHECK(ops.bad_launches == 0);
}

// ---- on a CUDA card -----------------------------------------------------------------------------------------------

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void env_default(const char * name, const char * value) {
    if (getenv(name) == nullptr) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 0);
#endif
    }
}

static int test_gpu() {
    env_default("GGML_CUDA_KEEPALIVE", "decode");
    env_default("GGML_CUDA_KEEPALIVE_MS", "2000");
    const char * ka_ms_s = getenv("GGML_CUDA_KEEPALIVE_MS");
    const double ka_ms   = atof(ka_ms_s);
    const char * eager_s = getenv("GGML_CUDA_KEEPALIVE_EAGER");
    const int    eager   = ggml_cuda_keepalive_parse_eager(eager_s);
    const char * loading = getenv("CUDA_MODULE_LOADING");

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr || strstr(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") == nullptr) {
        fprintf(stderr, "gpu: no CUDA device\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "gpu: the device did not start\n");
        return 1;
    }
    size_t (*counts)(char *, size_t) = (size_t (*)(char *, size_t))
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_keepalive_counts");

    const int n = 1 << 16;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c  = ggml_add(ctx, a, b);                               // graph A: kernels its first call launches
    ggml_tensor * d  = ggml_sin(ctx, ggml_sqr(ctx, a));                   // graph B: kernels not launched before
    ggml_tensor * m1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * n + 5); // a memset at a new size
    ggml_tensor * s1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7 * n + 3); // a set at a new size
    ggml_cgraph * ga = ggml_new_graph(ctx);
    ggml_build_forward_expand(ga, c);
    ggml_cgraph * gb = ggml_new_graph(ctx);
    ggml_build_forward_expand(gb, d);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "gpu: no buffer\n");
        return 1;
    }
    std::vector<float> va(n), vb(n), vd(n), vs(7 * n + 3, 1.0f);
    for (int i = 0; i < n; ++i) {
        va[i] = 0.001f * (float) (i % 1000);
        vb[i] = 1.0f;
    }
    ggml_backend_tensor_set(a, va.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(b, vb.data(), 0, n * sizeof(float));

    // graph A: its first call runs kernel by kernel, then it is captured and replayed; the card's block runs after each
    double t_a[4];
    for (double & t : t_a) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, ga);
        ggml_backend_synchronize(backend);
        t = ms_since(t0);
        sleep_ms(20);
    }
    // graph B right after a beat: its first call launches sqr and sin for the first time
    ggml_backend_graph_compute(backend, ga);
    auto t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);
    const double t_b = ms_since(t0);
    ggml_backend_tensor_get(d, vd.data(), 0, n * sizeof(float));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        bad += fabsf(vd[i] - sinf(va[i] * va[i])) > 1e-5f;
    }
    // a memset and a set at sizes new to the card, each right after a beat
    ggml_backend_graph_compute(backend, ga);
    t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_memset(m1, 0, 0, ggml_nbytes(m1));
    const double t_m = ms_since(t0);
    ggml_backend_graph_compute(backend, ga);
    t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(s1, vs.data(), 0, ggml_nbytes(s1));
    const double t_s = ms_since(t0);
    // graph B's second call (a capture: its kernels are loaded now), as a control
    ggml_backend_graph_compute(backend, ga);
    t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);
    const double t_b2 = ms_since(t0);

    // C817: a stop at a held start that has not begun. Graph C (cos in place, n_cos times on 8 Mi floats) runs kernel by
    // kernel on its first call, so the switch ends the card's block before it and holds the next start behind it. Right
    // after graph C is queued, a set at a size new to the card stops that held block. At 1 or 2 the set must not wait
    // for graph C (C811's controller waited for it: R131's long prompt)
    const int nc = 1 << 23, n_cos = 400;
    ggml_init_params ip2 = { (size_t) (n_cos + 8) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx2 = ggml_init(ip2);
    ggml_tensor * x  = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, nc);
    ggml_tensor * s2 = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 1000 + 7); // a set at a new size
    ggml_tensor * y  = x;
    for (int i = 0; i < n_cos; ++i) {
        y = ggml_cos_inplace(ctx2, y);
    }
    ggml_cgraph * gc = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gc, y);
    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, backend);
    if (buf2 == nullptr) {
        fprintf(stderr, "gpu: no buffer for graph C\n");
        return 1;
    }
    ggml_backend_tensor_memset(x, 0, 0, ggml_nbytes(x));
    ggml_backend_graph_compute(backend, ga); // a beat: the card's block runs
    t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute_async(backend, gc);
    const double t_cq = ms_since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(s2, vs.data(), 0, ggml_nbytes(s2));
    const double t_hs = ms_since(t1);
    ggml_backend_synchronize(backend);
    const double t_c = ms_since(t0);
    // graph C's second and third calls (a capture and a replay, or eager again): its time on the card
    double t_cn[2];
    for (double & t : t_cn) {
        t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gc);
        ggml_backend_synchronize(backend);
        t = ms_since(t0);
    }
    // cos applied again and again from 0 reaches the fixed point of cos (0.7390851...)
    std::vector<float> vx(1024);
    ggml_backend_tensor_get(x, vx.data(), 0, vx.size() * sizeof(float));
    int bad_c = 0;
    for (float v : vx) {
        bad_c += !(fabsf(v - 0.7390851f) < 1e-4f);
    }

    char line[640] = "";
    if (counts != nullptr) {
        counts(line, sizeof(line));
    }
    printf("gpu: %s, CUDA_MODULE_LOADING=%s, GGML_CUDA_KEEPALIVE=%s, GGML_CUDA_KEEPALIVE_MS=%s, GGML_CUDA_KEEPALIVE_EAGER=%s\n",
           ggml_backend_dev_description(dev), loading ? loading : "(unset)", getenv("GGML_CUDA_KEEPALIVE"), ka_ms_s,
           eager_s ? eager_s : "(unset)");
    printf("gpu: graph A (add) %.2f, %.2f, %.2f, %.2f ms; graph B (sqr, sin) first call %.2f ms, %d of %d values wrong; "
           "memset at a new size %.2f ms; set at a new size %.2f ms; graph B's second call %.2f ms\n",
           t_a[0], t_a[1], t_a[2], t_a[3], t_b, bad, n, t_m, t_s, t_b2);
    printf("gpu: keep-alive counts: %s\n", line[0] ? line : "(none: the switch is unset)");
    printf("gpu: graph B's first call %s for the block (%.0f ms against a %.0f ms timeout); the memset %s\n",
           t_b > ka_ms / 2 ? "waited" : "did not wait", t_b, ka_ms, t_m > ka_ms / 2 ? "waited" : "did not wait");
    unsigned long long unbegun = 0;
    const char * held_at = strstr(line, "starts held, ");
    if (held_at != nullptr) {
        sscanf(held_at, "starts held, %llu", &unbegun);
    }
    printf("gpu: C817: graph C (cos in place %d times on %d floats) queued in %.2f ms; a set at a new size right after it, "
           "the card's next start held behind graph C: %.2f ms; graph C done %.2f ms after it was queued, %d of %d values "
           "wrong; its second and third calls %.2f and %.2f ms; %llu blocks ended before they began\n",
           n_cos, nc, t_cq, t_hs, t_c, bad_c, (int) vx.size(), t_cn[0], t_cn[1], unbegun);
    printf("gpu: C817: the set %s for graph C (%.1f ms against its %.1f ms)\n",
           t_hs > t_cn[1] / 2 ? "waited" : "did not wait", t_hs, t_cn[1]);
    CHECK(bad == 0);
    CHECK(bad_c == 0);
    if (eager >= 1) {
        CHECK(t_b < ka_ms / 4);
        CHECK(t_m < ka_ms / 4);
        CHECK(t_s < ka_ms / 4);
        CHECK(strstr(line, "acted on") != nullptr);
        CHECK(t_hs < t_cn[1] / 4);
        CHECK(unbegun >= 1);
    }
    ggml_backend_buffer_free(buf2);
    ggml_free(ctx2);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("test-cuda-keepalive gpu: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "gpu") == 0) {
        return test_gpu();
    }
    test_parse();
    test_poll();
    test_stats();
    test_off();
    test_decode();
    test_timeout();
    test_beats_hold();
    test_check();
    test_request();
    test_idle();
    test_threads();
    test_eager_parse();
    test_eager_unset();
    test_eager_new_shapes();
    test_eager_every();
    test_outside();
    test_eager_request_kick();
    test_eager_counts_timeout();
    test_eager_threads();
    test_held_stop();
    test_held_chunk();
    test_held_threads();
    printf("test-cuda-keepalive: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
