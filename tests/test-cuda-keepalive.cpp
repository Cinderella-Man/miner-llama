// GGML_CUDA_KEEPALIVE's parts that run without a GPU (ggml/src/ggml-cuda/keepalive-policy.h): the settings, a block's
// decision at each poll, and the controller that starts the cards' blocks after graphs and ends them before captures,
// frees and device-wide synchronizes, and (GGML_CUDA_KEEPALIVE_EAGER, C811) before evaluations that run kernel by kernel
// and work outside a graph at new sizes. A thread stands in for each card's block: the kernel's loop over the same
// decision function, on the host's clock, reading the time and then the word as the kernel does.
// Usage: test-cuda-keepalive       (exit status 0 when every check passes)
//        test-cuda-keepalive gpu   on a CUDA card (C811's check on a rig): the first evaluation of kernels never
//                                  launched before, right after a graph whose block runs, timed; and a memset and a set
//                                  at sizes new to the card. It reads GGML_CUDA_KEEPALIVE (default decode),
//                                  GGML_CUDA_KEEPALIVE_MS (default 2000) and GGML_CUDA_KEEPALIVE_EAGER from the
//                                  environment; with EAGER at 1 or 2 none of them may take a quarter of the timeout.
//                                  Without it the times are printed (a first launch under lazy module loading waits for
//                                  the block: about the timeout)

#include "../ggml/src/ggml-cuda/keepalive-policy.h"

#include "ggml.h"
#include "ggml-backend.h"

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

// the cards: a word, a report and a thread each
struct mock_ops {
    static constexpr int N = 4;
    struct card {
        std::mutex                 m;
        uint64_t                   word = 0;
        ggml_cuda_keepalive_report report{};
        std::thread                th;
        std::atomic<bool>          ended{true};
        std::atomic<bool>          force_end{false}; // the test ends the block as its timeout would
        std::atomic<uint32_t>      beat_seen{0};
        int                        launches = 0;
        int                        writes   = 0;
        int                        launches_after = 0;       // launches that waited for a stream (EAGER's held starts)
        void *                     last_after     = nullptr; // the last launch's stream
    };
    card cards[N];
    std::atomic<int> bad_launches{0}; // a launch while the card's last block was not waited for

    void write(int dev, uint64_t w) {
        std::lock_guard<std::mutex> lock(cards[dev].m);
        cards[dev].word = w;
        cards[dev].writes++;
    }

    bool launch(int dev, uint32_t gen, uint32_t beat, uint32_t idle_ms, uint32_t poll_us, void * after) {
        card & c = cards[dev];
        if (c.th.joinable()) {
            bad_launches++;
            c.th.join();
        }
        c.launches++;
        c.launches_after += after != nullptr;
        c.last_after      = after;
        c.ended      = false;
        c.force_end  = false;
        c.report.gen = 0;
        c.th = std::thread([&c, gen, beat, idle_ms, poll_us] {
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
                c.beat_seen = w.beat;
                std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
            }
            c.report.reason = (uint32_t) r;
            c.report.polls  = polls;
            c.report.ns     = now_ns() - t0;
            for (uint32_t & h : c.report.hist) {
                h = 0;
            }
            c.report.hist[1885 / 50] = 1; // one sample at a P106-090's boost clock
            c.report.gen = gen;
            c.ended      = true;
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
        int dummy = 0;
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
    int stream_a = 0, stream_b = 0;
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
    int s = 0;
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
    int s = 0;
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
    int s = 0;
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
    int s = 0;
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
    CHECK(bad == 0);
    if (eager >= 1) {
        CHECK(t_b < ka_ms / 4);
        CHECK(t_m < ka_ms / 4);
        CHECK(t_s < ka_ms / 4);
        CHECK(strstr(line, "acted on") != nullptr);
    }
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
    printf("test-cuda-keepalive: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
