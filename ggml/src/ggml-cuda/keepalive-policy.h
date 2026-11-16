#pragma once

// GGML_CUDA_KEEPALIVE (keepalive.cu): one small block per card, on a low-priority stream of its own, polling a word in
// mapped host memory with a clock64() back-off, so that the driver counts the card as busy and keeps its boost clock.
// A layer split over many cards leaves each card idle most of a token (on nine P106 cards ~8/9 of it), and the driver
// then runs a decoding card at its base clock (1,354 against up to 1,911 MHz on a P106-090).
//
// The parts that need no GPU live here: the settings, the word the host writes and each block polls, a block's decision
// at each poll (the same function runs in the kernel), the report a block leaves when it ends, and the controller that
// starts the blocks after graph computes and stops them before a capture, a free, a host allocation or a device-wide
// synchronize, any of which would otherwise wait for a block that only the host can end. tests/test-cuda-keepalive.cpp
// checks them on a CPU, with a thread standing in for each card's block.
//
//   GGML_CUDA_KEEPALIVE          unset, empty or 0: off (nothing is launched or allocated)
//                                decode (or 1): a card's block starts after a graph is computed on it, and ends
//                                  itself when no graph has been computed on the card for GGML_CUDA_KEEPALIVE_MS
//                                request: decode, and every card's block also starts when llama-server takes a request,
//                                  so that the cards leave their idle state while the prompt is prepared
//                                idle: decode with a timeout long enough to cover the time the server idles between an
//                                  agent's turns (default 120 s), so the cards stay at their clock between requests
//   GGML_CUDA_KEEPALIVE_MS       how long a block keeps running after its card's last graph, 10..3600000 (default
//                                1000; 120000 for idle). The timeout also bounds the wait of any synchronizing call the
//                                controller is not told about.
//   GGML_CUDA_KEEPALIVE_POLL_US  the block's back-off between two reads of the word, 1..100000 us (default 50): the
//                                longest a stop waits, and one PCIe read per back-off
//   GGML_CUDA_KEEPALIVE_EAGER    (C811) unset or empty: as before. Under CUDA 12's lazy module loading the first launch
//                                of a kernel on a card waits until every kernel running on the card has ended, so a
//                                first launch while the card's block runs waits for the block's timeout (FN85: about
//                                1 s at a decode or verify graph's first call, which runs kernel by kernel).
//                                1: the card's block is ended before an evaluation that runs kernel by kernel (not a
//                                  capture, not a replay) whose shapes the card has not run before, and before a copy,
//                                  set, get or memset the backend runs on the card outside a graph at a size new to the
//                                  card; the block's next start is ordered on the card after the next evaluation's work
//                                2: as 1, and before every evaluation that runs kernel by kernel, new shapes or not
//                                0 (or any other value): no stop, only the counts
//                                At 1 and 2 a stop on a card whose block has not begun yet (its start held behind an
//                                  evaluation the card is still running) does not wait for it (C817): the word the stop
//                                  writes ends the block at its first poll, once the evaluation is done, and the host goes
//                                  on at once. Otherwise the copy to the next card at a new size, right after a card's
//                                  evaluation is queued, waited for that evaluation (R131: a long prompt's first chunk
//                                  queued one card at a time)
//                                Whenever it is set, llama-server prints the counts after each reply (keep-alive = ...)
// When a context is freed, each card's runs are logged with the SM clock the block measured while it ran.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

#if defined(__CUDACC__)
#define GGML_CUDA_KEEPALIVE_HD __host__ __device__
#else
#define GGML_CUDA_KEEPALIVE_HD
#endif

enum ggml_cuda_keepalive_mode {
    GGML_CUDA_KEEPALIVE_OFF,
    GGML_CUDA_KEEPALIVE_DECODE,
    GGML_CUDA_KEEPALIVE_REQUEST,
    GGML_CUDA_KEEPALIVE_IDLE,
};

struct ggml_cuda_keepalive_config {
    ggml_cuda_keepalive_mode mode    = GGML_CUDA_KEEPALIVE_OFF;
    uint32_t                 idle_ms = 1000;
    uint32_t                 poll_us = 50;
    int                      eager   = -1; // GGML_CUDA_KEEPALIVE_EAGER: -1 unset, 0 counts only, 1 new shapes, 2 every
};

// GGML_CUDA_KEEPALIVE_EAGER's value (nullptr: unset): -1 unset or empty, 1 and 2 as written, any other value 0
static inline int ggml_cuda_keepalive_parse_eager(const char * v) {
    if (v == nullptr || v[0] == '\0') {
        return -1;
    }
    if (strcmp(v, "1") == 0) {
        return 1;
    }
    if (strcmp(v, "2") == 0) {
        return 2;
    }
    return 0;
}

// what the backend runs on a card outside a graph (GGML_CUDA_KEEPALIVE_EAGER): a kind and a size, as one key
enum ggml_cuda_keepalive_outside_kind {
    GGML_CUDA_KEEPALIVE_OUT_SET     = 1, // host to card (a tensor set, sync or async, 1D or 2D, or a batch of sets)
    GGML_CUDA_KEEPALIVE_OUT_GET     = 2, // card to host
    GGML_CUDA_KEEPALIVE_OUT_MEMSET  = 3, // a memset or a buffer clear
    GGML_CUDA_KEEPALIVE_OUT_COPY    = 4, // card to card on one card
    GGML_CUDA_KEEPALIVE_OUT_SEND    = 5, // the source card of a copy between cards
    GGML_CUDA_KEEPALIVE_OUT_RECV    = 6, // the destination card of a copy between cards
};

static inline uint64_t ggml_cuda_keepalive_outside_key(int kind, uint64_t size) {
    return ((uint64_t) kind << 56) ^ size;
}

// The three variables' values (nullptr: unset). An unknown mode is off; out-of-range numbers are clamped.
static inline ggml_cuda_keepalive_config ggml_cuda_keepalive_parse(const char * mode, const char * ms, const char * poll_us) {
    ggml_cuda_keepalive_config c;
    if (mode == nullptr || mode[0] == '\0' || strcmp(mode, "0") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_OFF;
    } else if (strcmp(mode, "1") == 0 || strcmp(mode, "decode") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_DECODE;
    } else if (strcmp(mode, "request") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_REQUEST;
    } else if (strcmp(mode, "idle") == 0) {
        c.mode = GGML_CUDA_KEEPALIVE_IDLE;
    } else {
        c.mode = GGML_CUDA_KEEPALIVE_OFF;
    }
    c.idle_ms = c.mode == GGML_CUDA_KEEPALIVE_IDLE ? 120000 : 1000;
    if (ms != nullptr && ms[0] != '\0') {
        const long v = strtol(ms, nullptr, 10);
        c.idle_ms = (uint32_t) (v < 10 ? 10 : v > 3600000 ? 3600000 : v);
    }
    if (poll_us != nullptr && poll_us[0] != '\0') {
        const long v = strtol(poll_us, nullptr, 10);
        c.poll_us = (uint32_t) (v < 1 ? 1 : v > 100000 ? 100000 : v);
    }
    return c;
}

static inline const char * ggml_cuda_keepalive_mode_name(ggml_cuda_keepalive_mode m) {
    switch (m) {
        case GGML_CUDA_KEEPALIVE_DECODE:  return "decode";
        case GGML_CUDA_KEEPALIVE_REQUEST: return "request";
        case GGML_CUDA_KEEPALIVE_IDLE:    return "idle";
        default:                          return "off";
    }
}

// The word the host writes and a block polls: the run's generation in the low 32 bits (a block runs while it equals
// its own; the host ends a run by writing 0), and the card's beat in the high 32 bits (one more after every graph).
static inline uint64_t ggml_cuda_keepalive_word(uint32_t gen, uint32_t beat) {
    return ((uint64_t) beat << 32) | gen;
}

enum ggml_cuda_keepalive_state {
    GGML_CUDA_KEEPALIVE_RUN     = 0,
    GGML_CUDA_KEEPALIVE_STOPPED = 1, // the host ended the run
    GGML_CUDA_KEEPALIVE_TIMEOUT = 2, // no beat for the timeout
};

// what a block has seen of the beat
struct ggml_cuda_keepalive_watch {
    uint32_t           beat;
    unsigned long long t_beat; // ns, the block's clock, when the beat last changed (or the run started)
};

// A block's decision at one poll of the word, at time now_ns.
GGML_CUDA_KEEPALIVE_HD static inline int ggml_cuda_keepalive_poll(uint64_t word, uint32_t gen, unsigned long long now_ns,
                                                                  unsigned long long idle_ns, ggml_cuda_keepalive_watch & w) {
    if ((uint32_t) word != gen) {
        return GGML_CUDA_KEEPALIVE_STOPPED;
    }
    const uint32_t beat = (uint32_t) (word >> 32);
    if (beat != w.beat) {
        w.beat   = beat;
        w.t_beat = now_ns;
        return GGML_CUDA_KEEPALIVE_RUN;
    }
    return now_ns - w.t_beat >= idle_ns ? GGML_CUDA_KEEPALIVE_TIMEOUT : GGML_CUDA_KEEPALIVE_RUN;
}

// The SM clock the block measured while it ran (its clock64() against the global timer, every
// GGML_CUDA_KEEPALIVE_SAMPLE_NS), in bins of 50 MHz.
#define GGML_CUDA_KEEPALIVE_SAMPLE_NS 100000000ull
#define GGML_CUDA_KEEPALIVE_BINS      48

// how often a beat also asks whether a running block has ended (a stream query)
#define GGML_CUDA_KEEPALIVE_CHECK_MS  250

// What a block leaves in mapped host memory when it ends; gen is written last.
struct ggml_cuda_keepalive_report {
    uint32_t           reason; // GGML_CUDA_KEEPALIVE_STOPPED or _TIMEOUT
    uint32_t           pad;
    unsigned long long polls;
    unsigned long long ns;     // from the start to the end
    uint32_t           hist[GGML_CUDA_KEEPALIVE_BINS];
    volatile uint32_t  gen;
};

// A card's runs, summed over the reports the host has read.
struct ggml_cuda_keepalive_stats {
    uint64_t starts   = 0;
    uint64_t stopped  = 0; // ended by the host: a capture, a free, a host allocation, a device-wide synchronize
    uint64_t timeouts = 0; // ended themselves: no graph for the timeout
    uint64_t lost     = 0; // ended with no report of this run (a bug)
    double   seconds  = 0;
    uint64_t hist[GGML_CUDA_KEEPALIVE_BINS] = {};

    void add(const ggml_cuda_keepalive_report & r) {
        stopped  += r.reason == GGML_CUDA_KEEPALIVE_STOPPED;
        timeouts += r.reason == GGML_CUDA_KEEPALIVE_TIMEOUT;
        seconds  += r.ns * 1e-9;
        for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
            hist[i] += r.hist[i];
        }
    }

    uint64_t samples() const {
        uint64_t n = 0;
        for (uint64_t h : hist) {
            n += h;
        }
        return n;
    }

    // the clock below which a share q of the samples lie, MHz (the bin's middle)
    int mhz_quantile(double q) const {
        const uint64_t n = samples();
        uint64_t c = 0;
        for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
            c += hist[i];
            if (n > 0 && c >= q * n) {
                return i * 50 + 25;
            }
        }
        return 0;
    }

    // the share of samples in [lo, hi) MHz, at the bins' resolution
    double share(int lo_mhz, int hi_mhz) const {
        const uint64_t n = samples();
        uint64_t c = 0;
        for (int i = 0; i < GGML_CUDA_KEEPALIVE_BINS; ++i) {
            c += (i * 50 >= lo_mhz && i * 50 < hi_mhz) ? hist[i] : 0;
        }
        return n ? (double) c / n : 0.0;
    }
};

// What GGML_CUDA_KEEPALIVE_EAGER has done since the start, and every run, over the cards (never reset).
struct ggml_cuda_keepalive_counts {
    uint64_t starts   = 0;
    uint64_t stopped  = 0;
    uint64_t timeouts = 0;
    uint64_t lost     = 0;
    uint64_t running  = 0; // blocks launched and not yet seen to end
    uint64_t eager       = 0; // evaluations that ran kernel by kernel
    uint64_t eager_new   = 0; // of them, the ones the switch acted on (new shapes; at 2 every one)
    uint64_t eager_stops = 0; // of those, the ones before which a running block was ended
    uint64_t out_new     = 0; // sets, gets, memsets and copies outside a graph at a size new to the card
    uint64_t out_stops   = 0; // of them, the ones before which a running block was ended
    uint64_t held        = 0; // blocks started after the card's evaluation, not at once
    uint64_t cancelled   = 0; // blocks ended by the host before they began, without a wait (C817; in stopped too)
};

// Starts and stops the cards' blocks. Ops is what a card does (keepalive.cu on CUDA, a thread in the test):
//   void     write(int dev, uint64_t word)          the word into the card's mapped host memory
//   bool     launch(int dev, uint32_t gen, uint32_t beat, uint32_t idle_ms, uint32_t poll_us, void * after)
//                                                   a block on the card's keep-alive stream; false if it failed. after:
//                                                   nullptr, or the stream (cudaStream_t) of the card whose work so far
//                                                   the block must not start before (GGML_CUDA_KEEPALIVE_EAGER)
//   bool     done(int dev)                          the block has ended (does not wait)
//   void     wait(int dev)                          until the block has ended
//   bool     pending(int dev)                       the card's newest block has not begun: the work of the stream a
//                                                   start was held for last is not done, and the block waits behind it
//                                                   (does not wait; asked only at GGML_CUDA_KEEPALIVE_EAGER 1 and 2)
//   const ggml_cuda_keepalive_report & report(int dev)
// Every method takes the controller's lock, so a free on one thread and a graph on another are safe.
template <typename Ops, int MAX_DEVICES>
struct ggml_cuda_keepalive_ctl {
    using clock = std::chrono::steady_clock;

    ggml_cuda_keepalive_config cfg;
    Ops &                      ops;
    std::mutex                 mutex;
    // C817: a stop at a held start that has not begun leaves the block to end by itself (false: C811's controller,
    // which waits for it, and so for the evaluation it is held behind; the test compares the two)
    bool                       cancel_held = true;

    struct dev_state {
        bool              seen    = false; // a graph has been computed on the card
        bool              running = false; // launched and not yet seen to end
        uint32_t          gen     = 0;
        uint32_t          beat    = 0;
        clock::time_point last_beat;       // when the host last wrote a beat to a running block
        clock::time_point last_check;      // when the host last asked whether the block has ended
        ggml_cuda_keepalive_stats stats;
        // GGML_CUDA_KEEPALIVE_EAGER
        bool                         hold = false; // the next start waits for the card's evaluation (its stream)
        std::unordered_set<uint64_t> shapes;       // the shapes of the evaluations run kernel by kernel on the card
        std::unordered_set<uint64_t> sizes;        // the kinds and sizes of the work outside a graph on the card
        ggml_cuda_keepalive_counts   counts;       // since the start (stopped, timeouts and lost by the reports)
    };
    dev_state devs[MAX_DEVICES];
    uint32_t  next_gen = 1;

    ggml_cuda_keepalive_ctl(const ggml_cuda_keepalive_config & cfg, Ops & ops) : cfg(cfg), ops(ops) {}

    bool on() const { return cfg.mode != GGML_CUDA_KEEPALIVE_OFF; }

    // before a graph's work is enqueued on the card: a capture must not begin while its block runs
    void graph_begin(int dev, bool capture) {
        if (!on() || !capture) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        stop_locked(dev);
    }

    // after a graph's work is enqueued on the card: a beat for its block, which is started if it is not running.
    // after: the stream the graph ran on (cudaStream_t), or nullptr; a start the switch holds waits for its work so far
    void graph_end(int dev, void * after = nullptr) {
        if (!on() || dev < 0 || dev >= MAX_DEVICES) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        devs[dev].seen = true;
        beat_locked(dev, after);
    }

    // GGML_CUDA_KEEPALIVE_EAGER: before an evaluation that runs kernel by kernel on the card, shape: its shapes as one
    // number. When the switch acts on it (a shape new to the card at 1, every one at 2), the card's block is ended and
    // its next start held until this evaluation's work is done (graph_end's stream), so that no kernel launched by the
    // evaluation finds the block running. Returns whether the switch acted.
    bool eager_begin(int dev, uint64_t shape) {
        if (!on() || cfg.eager < 0 || dev < 0 || dev >= MAX_DEVICES) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        dev_state & d = devs[dev];
        d.counts.eager++;
        if (cfg.eager < 1) {
            return false;
        }
        const bool fresh = d.shapes.insert(shape).second;
        if (!fresh && cfg.eager < 2) {
            return false;
        }
        d.counts.eager_new++;
        if (d.running) {
            d.counts.eager_stops++;
            stop_locked(dev);
        }
        d.hold = true;
        return true;
    }

    // GGML_CUDA_KEEPALIVE_EAGER at 1 or 2: before a set, get, memset or copy the backend runs on the card outside a
    // graph, key: its kind and size (ggml_cuda_keepalive_outside_key). At a key new to the card, the card's block is
    // ended and its next start held until the card's next evaluation. Returns whether the switch acted.
    bool outside(int dev, uint64_t key) {
        if (!on() || cfg.eager < 1 || dev < 0 || dev >= MAX_DEVICES) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        dev_state & d = devs[dev];
        if (!d.sizes.insert(key).second) {
            return false;
        }
        d.counts.out_new++;
        if (d.running) {
            d.counts.out_stops++;
            stop_locked(dev);
        }
        d.hold = true;
        return true;
    }

    // every card's counts since the start, summed; a block found ended is read first (as the beats' check does)
    ggml_cuda_keepalive_counts counts(int dev_only = -1) {
        std::lock_guard<std::mutex> lock(mutex);
        ggml_cuda_keepalive_counts c;
        for (int dev = 0; dev < MAX_DEVICES; ++dev) {
            if (dev_only >= 0 && dev != dev_only) {
                continue;
            }
            dev_state & d = devs[dev];
            if (d.running && ops.done(dev)) {
                finish_locked(dev);
            }
            c.starts      += d.counts.starts;
            c.stopped     += d.counts.stopped;
            c.timeouts    += d.counts.timeouts;
            c.lost        += d.counts.lost;
            c.running     += d.running;
            c.eager       += d.counts.eager;
            c.eager_new   += d.counts.eager_new;
            c.eager_stops += d.counts.eager_stops;
            c.out_new     += d.counts.out_new;
            c.out_stops   += d.counts.out_stops;
            c.held        += d.counts.held;
            c.cancelled   += d.counts.cancelled;
        }
        return c;
    }

    // llama-server took a request (request mode): the block of every card that has computed a graph (the warm-up's
    // cards: the request's), ahead of the prompt
    void kick() {
        if (cfg.mode != GGML_CUDA_KEEPALIVE_REQUEST) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        for (int dev = 0; dev < MAX_DEVICES; ++dev) {
            if (devs[dev].seen) {
                beat_locked(dev, nullptr);
            }
        }
    }

    // before a free, a host allocation or a device-wide synchronize: every block ended
    void quiesce() {
        if (!on()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        // every stop first, then every wait: the stops overlap
        for (int dev = 0; dev < MAX_DEVICES; ++dev) {
            if (devs[dev].running) {
                ops.write(dev, ggml_cuda_keepalive_word(0, devs[dev].beat));
            }
        }
        for (int dev = 0; dev < MAX_DEVICES; ++dev) {
            if (devs[dev].running && !cancel_locked(dev)) {
                finish_locked(dev);
            }
        }
    }

    // the card's stats so far (ending its block first, so that its report counts); the stats are reset
    ggml_cuda_keepalive_stats take_stats(int dev) {
        std::lock_guard<std::mutex> lock(mutex);
        stop_locked(dev);
        ggml_cuda_keepalive_stats s = devs[dev].stats;
        devs[dev].stats = ggml_cuda_keepalive_stats();
        return s;
    }

    bool running(int dev) {
        std::lock_guard<std::mutex> lock(mutex);
        return devs[dev].running;
    }

  private:
    void stop_locked(int dev) {
        if (dev < 0 || dev >= MAX_DEVICES || !devs[dev].running) {
            return;
        }
        ops.write(dev, ggml_cuda_keepalive_word(0, devs[dev].beat));
        if (!cancel_locked(dev)) {
            finish_locked(dev);
        }
    }

    // GGML_CUDA_KEEPALIVE_EAGER at 1 or 2 (C817), after a stop's word is written: a block that has not begun is left
    // to end by itself. It waits behind an evaluation the card is still running; when that is done, it reads a word
    // without its generation (this stop's 0, or a later run's) and ends at its first poll, without running. Waiting
    // for it would wait for that evaluation. The word is written first, so a block found not begun is sure to read it.
    // Its report is not read (a later block on the card writes its own after it); it counts as ended by the host.
    bool cancel_locked(int dev) {
        if (!cancel_held || cfg.eager < 1 || !ops.pending(dev)) {
            return false;
        }
        dev_state & d = devs[dev];
        d.stats.stopped++;
        d.counts.stopped++;
        d.counts.cancelled++;
        d.running = false;
        return true;
    }

    // wait for the card's block to end and take its report
    void finish_locked(int dev) {
        dev_state & d = devs[dev];
        ops.wait(dev);
        const ggml_cuda_keepalive_report & r = ops.report(dev);
        if (r.gen == d.gen) {
            d.stats.add(r);
            d.counts.stopped  += r.reason == GGML_CUDA_KEEPALIVE_STOPPED;
            d.counts.timeouts += r.reason == GGML_CUDA_KEEPALIVE_TIMEOUT;
        } else {
            d.stats.lost++;
            d.counts.lost++;
        }
        d.running = false;
    }

    void beat_locked(int dev, void * after) {
        if (dev < 0 || dev >= MAX_DEVICES) {
            return;
        }
        dev_state & d = devs[dev];
        const clock::time_point now = clock::now();
        d.beat++;
        if (d.running) {
            // A block whose last beat is less than its timeout ago (by a margin far above a PCIe read) cannot end by
            // itself before it reads this one. Any other may be ending as the beat is written: it is ended and a
            // fresh one started, which costs one stop after each gap of about the timeout. A block that ended while
            // this thread was held up between taking a beat's time and writing it is found by the check every
            // GGML_CUDA_KEEPALIVE_CHECK_MS.
            const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(now - d.last_beat).count();
            if (since + 5 >= (long long) cfg.idle_ms) {
                stop_locked(dev);
            } else if (now - d.last_check >= std::chrono::milliseconds(GGML_CUDA_KEEPALIVE_CHECK_MS)) {
                d.last_check = now;
                if (ops.done(dev)) {
                    finish_locked(dev);
                }
            }
        }
        if (d.running) {
            ops.write(dev, ggml_cuda_keepalive_word(d.gen, d.beat));
        } else {
            const uint32_t gen = next_gen++;
            if (next_gen == 0) {
                next_gen = 1;
            }
            ops.write(dev, ggml_cuda_keepalive_word(gen, d.beat));
            // GGML_CUDA_KEEPALIVE_EAGER: a held start waits for the card's work so far (only a graph's end has a stream)
            void * wait_for = d.hold ? after : nullptr;
            if (ops.launch(dev, gen, d.beat, cfg.idle_ms, cfg.poll_us, wait_for)) {
                d.gen        = gen;
                d.running    = true;
                d.last_check = now;
                d.stats.starts++;
                d.counts.starts++;
                if (wait_for != nullptr) {
                    d.hold = false;
                    d.counts.held++;
                }
            }
        }
        d.last_beat = now;
    }
};
