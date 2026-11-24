// Q771: the micro-batch taper (LLAMA_UBATCH_TAPER, llama_set_ubatch_taper).
//
// Unit part (always): the spec parser, the plan of a call (llama_ubatch_taper_plan) on fixed cases and on a sweep of
// sizes, flags and schedules (every plan covers the call in order, no micro-batch above n_ubatch, empty or under min,
// and the plan is empty exactly when it equals the split without one).
//
// Model part (-m <gguf>): prompts of the --lens sizes on the model, decoded in the calls llama-server makes (n_batch
// tokens at most, the batch cut 4 + n_ubatch and 4 tokens before the prompt's end, a checkpoint read before each of
// those two calls), once with the taper off and once per taper arm, then 8 greedy tokens. Each arm against off:
//   logits of every prompt position (outputs "all") or of the last one ("last"): values that differ, the largest
//     absolute difference and the nmse (pass: nmse <= 1e-8, the cross-build tolerance of test-llama-archs);
//   the recurrent-state checkpoints read before the two tail calls, and the full sequence state after the prompt:
//     byte for byte (the arm tiny1, micro-batches of one token: as floats within 1e-5, see below);
//   the 8 greedy tokens;
//   the micro-batch sizes the context logged for each call against llama_ubatch_taper_plan for that call, and the
//     head and tail it used against the server's rule (head when a checkpoint read or the start precedes the call,
//     tail when it stops short of n_batch or has outputs).
// "base:" lines show the base's own rounding at a one-token batch with the taper off (n tokens in one call against
// n - 1 and 1): the reason the plans keep micro-batches at 9 tokens or more (min).
// C905 (cuts=1): the unit part plans calls with cuts (llama-server's in-call checkpoints, 4 + n_ubatch and 4 tokens
// before a call's end) and checks that every segment between them holds no piece under min unless the cut itself makes
// the segment shorter; the model part adds arms that decode each call whole with those cuts inside it
// (--ctx-checkpoints-incall: llama_set_ubatch_cuts, the checkpoints filled at the cuts by llama_state_seq_store_at),
// against the same calls with the taper off.
// --plan-check <N,N,...> [--plan-specs <spec;spec;...>] [--plan-rs <n_rs_seq>]: no comparisons; per spec and call size
// N, one call of N tokens decoded on the model as llama-server's prompt call with in-call checkpoints (head and tail,
// cuts at N - 4 - n_ubatch and N - 4, n_batch 4096, n_ubatch 512), and the micro-batches the context logged, as TSV.
// Usage: test-ubatch-taper [--unit-only] [-m <gguf>] [-t <threads>] [--lens 100,600,3000,9000] [--tsv <file>]
//        [--plan-check <sizes> [--plan-specs <specs>] [--plan-rs <n>]]
#include "llama.h"
#include "../src/llama-batch.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

static std::vector<uint32_t> sizes_of(const std::vector<uint32_t> & ends) {
    std::vector<uint32_t> res;
    uint32_t prev = 0;
    for (uint32_t e : ends) {
        res.push_back(e - prev);
        prev = e;
    }
    return res;
}

static std::string str_of(const std::vector<uint32_t> & v) {
    std::string s;
    for (uint32_t x : v) {
        s += (s.empty() ? "" : ",") + std::to_string(x);
    }
    return s;
}

// the micro-batch sizes of a call: the plan's, or n_ubatch at a time without one
static std::vector<uint32_t> plan_sizes(const llama_ubatch_taper & t, uint32_t n, uint32_t n_ubatch, bool head, bool tail) {
    std::vector<uint32_t> ends = llama_ubatch_taper_plan(t, n, n_ubatch, head, tail);
    if (ends.empty()) {
        for (uint32_t pos = 0; pos < n; ) {
            pos = std::min(n, pos + n_ubatch);
            ends.push_back(pos);
        }
    }
    return sizes_of(ends);
}

// C905: llama-server's cuts in a prompt call of n tokens with --ctx-checkpoints-incall: 4 + n_ubatch and 4 tokens
// before its end, where they fall inside it
static std::vector<uint32_t> server_cuts(uint32_t n, uint32_t n_ubatch) {
    std::vector<uint32_t> cuts;
    for (uint32_t off : { 4 + n_ubatch, 4u }) {
        if (n > off) {
            cuts.push_back(n - off);
        }
    }
    return cuts;
}

// the micro-batch sizes the splits make of a call of n tokens with plan ends and cuts, as split_simple takes them (at
// most n_ubatch, up to the next planned end, up to the next cut; no recurrent n_keep_tail rule)
static std::vector<uint32_t> split_sizes(const std::vector<uint32_t> & plan, uint32_t n, uint32_t n_ubatch,
                                         std::vector<uint32_t> cuts) {
    std::sort(cuts.begin(), cuts.end());
    std::vector<uint32_t> res;
    uint32_t used = 0;
    while (used < n) {
        uint32_t cap = n_ubatch;
        for (uint32_t e : plan) {
            if (e > used) {
                cap = std::max<uint32_t>(1, std::min(n_ubatch, e - used));
                break;
            }
        }
        for (uint32_t c : cuts) {
            if (c > used) {
                cap = std::min(cap, c - used);
                break;
            }
        }
        cap = std::min(cap, n - used);
        res.push_back(cap);
        used += cap;
    }
    return res;
}

// the sizes of a call with cuts under the taper (C905 when t.cuts): the plan with the cuts, as the splits take it
static std::vector<uint32_t> plan_sizes_cut(const llama_ubatch_taper & t, uint32_t n, uint32_t n_ubatch, bool head,
                                            bool tail, const std::vector<uint32_t> & cuts) {
    return split_sizes(llama_ubatch_taper_plan(t, n, n_ubatch, head, tail, cuts), n, n_ubatch, cuts);
}

// "64x4 512 4" (the context's log form) from sizes
static std::string compact_of(const std::vector<uint32_t> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ) {
        size_t j = i + 1;
        while (j < v.size() && v[j] == v[i]) {
            ++j;
        }
        s += (s.empty() ? "" : " ") + std::to_string(v[i]) + (j - i > 1 ? "x" + std::to_string(j - i) : "");
        i = j;
    }
    return s;
}

// C905's floor: every piece of a call with cuts holds min_s tokens or more, unless it is a whole segment between two
// cuts (or a cut and an end) that is shorter than min_s; false with the first piece that breaks it in why
static bool floor_holds(const std::vector<uint32_t> & sizes, uint32_t n, std::vector<uint32_t> cuts, uint32_t min_s,
                        std::string & why) {
    std::vector<uint32_t> bounds = { 0, n };
    for (uint32_t c : cuts) {
        if (c > 0 && c < n) {
            bounds.push_back(c);
        }
    }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    auto is_bound = [&](uint32_t p) { return std::binary_search(bounds.begin(), bounds.end(), p); };
    uint32_t pos = 0;
    for (uint32_t s : sizes) {
        if (s < min_s) {
            // a whole segment: it starts and ends at bounds with no bound inside
            bool whole = is_bound(pos) && is_bound(pos + s);
            for (uint32_t p = pos + 1; whole && p < pos + s; ++p) {
                whole = !is_bound(p);
            }
            if (!whole) {
                why = "a piece of " + std::to_string(s) + " tokens at " + std::to_string(pos);
                return false;
            }
        }
        pos += s;
    }
    return pos == n;
}

// C905: cuts=1, the plan of a call with cuts made per segment between them
static void unit_cuts() {
    const int fail0 = g_fail;
    llama_ubatch_taper t;
    std::string err;

    // parser
    CHECK(t.parse("short=1400:64,cuts=1", err) && t.cuts && t.str() == "short=1400:64,cuts=1", "-> %s", t.str().c_str());
    CHECK(t.parse("cuts=1, short=1400:64", err) && t.cuts && t.str() == "short=1400:64,cuts=1", "-> %s", t.str().c_str());
    CHECK(t.parse("short=1400:64,cuts=0", err) && !t.cuts && t.str() == "short=1400:64", "-> %s", t.str().c_str());
    CHECK(t.parse("short=1400:64", err) && !t.cuts, "cuts is not 0 by default");
    CHECK(t.parse("cuts=1", err) && !t.enabled() && t.str() == "off", "cuts alone -> %s", t.str().c_str());
    CHECK(t.parse("head=256:64,tail=512:128,short=600:64,min=12,cuts=1", err) && t.str() == "head=256:64,tail=512:128,short=600:64,min=12,cuts=1",
          "-> %s", t.str().c_str());
    for (const char * bad : { "cuts=2", "cuts=", "cuts=1:2", "cuts=yes", "short=1400:64,cuts", "on,cuts=1", "cut=1" }) {
        llama_ubatch_taper keep;
        keep.parse("short=1400:64", err);
        CHECK(!keep.parse(bad, err) && keep.str() == "short=1400:64" && !keep.cuts, "'%s' accepted or changed the schedule", bad);
    }

    // fixed calls with llama-server's in-call cuts (n_ubatch 512, head and tail): the v5 preset's short=1400:64 on
    // C222's 577- and 583-token calls, short=2048:64 on its 1,534-token call, and "on" on a 3,000-token call, without
    // and with cuts=1
    struct ccase { const char * spec; uint32_t n; const char * want_old; const char * want_new; };
    const ccase cases[] = {
        { "short=1400:64",  577, "61 3 64x7 61 4",        "61 64x8 4" },
        { "short=1400:64",  583, "64 3 61 64x6 67 4",     "67 64x8 4" },
        { "short=2048:64", 1534, "64x15 58 6 64x7 58 4",  "64x15 58 64x8 4" },
        { "short=1400:64",  610, "64 30 34 64x7 30 4",    "64 30 64x8 4" },
        { "short=1400:64",  516, "64x8 4",                "64x8 4" },
        { "short=1400:64",  520, "4 60 64x6 68 4",        "4 64x8 4" },
        { "short=1400:64",   30, "26 4",                  "26 4" },
        { "short=1400:64", 2000, "512x2 460 512 4",       "512x2 460 512 4" },
        { "head=256:64,tail=512:128,short=600:64", 3000, "64x4 512x4 180 4 128x3 124 4", "64x4 512x4 180 132 128x2 124 4" },
    };
    for (const auto & c : cases) {
        const std::vector<uint32_t> cuts = server_cuts(c.n, 512);
        for (int k = 0; k < 2; ++k) {
            const std::string spec = std::string(c.spec) + (k ? ",cuts=1" : "");
            CHECK(t.parse(spec.c_str(), err), "bad spec %s", spec.c_str());
            const std::string got = compact_of(plan_sizes_cut(t, c.n, 512, true, true, cuts));
            const char * want = k ? c.want_new : c.want_old;
            CHECK(got == want, "%s, %u tokens, cuts %s: %s, want %s", spec.c_str(), c.n, str_of(cuts).c_str(), got.c_str(), want);
        }
    }
    // cuts=1: the cuts are ends of the plan, a segment under min is one piece
    t.parse("short=1400:64,cuts=1", err);
    {
        const auto ends = llama_ubatch_taper_plan(t, 577, 512, true, true, server_cuts(577, 512));
        CHECK(std::find(ends.begin(), ends.end(), 61u) != ends.end() && std::find(ends.begin(), ends.end(), 573u) != ends.end(),
              "577: the cuts are not ends of the plan: %s", str_of(ends).c_str());
    }
    // without cuts, cuts=1 is the old plan; with cuts=0 the cuts are not read
    {
        llama_ubatch_taper t0;
        t0.parse("short=1400:64", err);
        for (uint32_t n : { 100u, 577u, 1400u, 1534u }) {
            CHECK(llama_ubatch_taper_plan(t, n, 512, true, true) == llama_ubatch_taper_plan(t0, n, 512, true, true),
                  "cuts=1 without cuts changes the plan of %u", n);
            CHECK(llama_ubatch_taper_plan(t0, n, 512, true, true, server_cuts(n, 512)) == llama_ubatch_taper_plan(t0, n, 512, true, true),
                  "cuts=0 reads the cuts of %u", n);
        }
    }
    // a plan equal to the plain split with the cuts is dropped: short=2000:512 on a 1,000-token call cut at 300 is
    // 300 | 512 188, the plain split with that cut
    t.parse("short=2000:512,cuts=1", err);
    CHECK(llama_ubatch_taper_plan(t, 1000, 512, true, true, { 300 }).empty(), "short=2000:512, 1000 with a cut at 300 has a plan");

    // sweep: random schedules, calls and cuts (the server's two or one to three anywhere); with cuts=1 every plan covers
    // the call in order, holds every cut as an end, no piece above n_ubatch, none under min unless a whole segment is
    // shorter, empty exactly when it equals the plain split with the cuts; the splits then take it as it is; cuts=0
    // reads no cut
    std::mt19937 rng(905);
    int n_plans = 0;
    int n_fixed = 0; // calls whose old plan leaves a piece under min beside a cut that cuts=1 removes
    const int n_sweep = 20000;
    for (int k = 0; k < n_sweep; ++k) {
        llama_ubatch_taper s;
        s.head_n  = rng() % 3 ? rng() % 1500 : 0; s.head_s  = s.head_n  ? 1 + rng() % 700 : 0;
        s.tail_n  = rng() % 3 ? rng() % 1500 : 0; s.tail_s  = s.tail_n  ? 1 + rng() % 700 : 0;
        s.short_n = rng() % 3 ? rng() % 1500 : 0; s.short_s = s.short_n ? 1 + rng() % 700 : 0;
        if (rng() % 4 == 0) {
            s.min_s = 1 + rng() % 20;
        }
        const uint32_t n_ubatch = rng() % 2 ? 512 : 1 + rng() % 1024;
        const uint32_t n = 1 + rng() % 5000;
        const bool head = rng() & 1;
        const bool tail = rng() & 1;
        std::vector<uint32_t> cuts;
        if (rng() % 2) {
            cuts = server_cuts(n, n_ubatch);
        } else {
            for (uint32_t i = 0, nc = 1 + rng() % 3; i < nc; ++i) {
                cuts.push_back(rng() % (n + 2)); // 0, n and n + 1 are outside and ignored
            }
        }
        std::vector<uint32_t> in;
        for (uint32_t c : cuts) {
            if (c > 0 && c < n) {
                in.push_back(c);
            }
        }
        std::sort(in.begin(), in.end());
        in.erase(std::unique(in.begin(), in.end()), in.end());

        llama_ubatch_taper s0 = s;
        s0.cuts = false;
        s.cuts = true;
        CHECK(llama_ubatch_taper_plan(s0, n, n_ubatch, head, tail, cuts) == llama_ubatch_taper_plan(s0, n, n_ubatch, head, tail),
              "cuts=0 read the cuts: %s, n %u", s0.str().c_str(), n);
        const auto ends = llama_ubatch_taper_plan(s, n, n_ubatch, head, tail, cuts);
        if (in.empty()) {
            CHECK(ends == llama_ubatch_taper_plan(s0, n, n_ubatch, head, tail), "cuts=1 without cuts in the call changed the plan: %s, n %u",
                  s.str().c_str(), n);
            continue;
        }
        n_plans += !ends.empty();
        uint32_t prev = 0;
        bool ok = true;
        for (uint32_t e : ends) {
            ok = ok && e > prev && e - prev <= n_ubatch;
            prev = e;
        }
        ok = ok && (ends.empty() || ends.back() == n);
        for (uint32_t c : in) {
            ok = ok && (ends.empty() || std::binary_search(ends.begin(), ends.end(), c));
        }
        std::vector<uint32_t> plain;
        {
            uint32_t sb = 0;
            for (size_t i = 0; i <= in.size(); ++i) {
                const uint32_t se = i < in.size() ? in[i] : n;
                for (uint32_t pos = sb; pos < se; ) {
                    pos = std::min(se, pos + n_ubatch);
                    plain.push_back(pos);
                }
                sb = se;
            }
        }
        CHECK(ends.empty() || ends != plain, "a plan equal to the plain split with the cuts was returned: %s, n %u", s.str().c_str(), n);
        const std::vector<uint32_t> got = split_sizes(ends, n, n_ubatch, cuts);
        CHECK(got == sizes_of(ends.empty() ? plain : ends), "the splits do not take the plan as it is: %s, n %u, cuts %s: %s",
              s.str().c_str(), n, str_of(in).c_str(), str_of(got).c_str());
        std::string why;
        if (!ends.empty() && s.min_s > 1 && n_ubatch >= 2*s.min_s) {
            ok = ok && floor_holds(sizes_of(ends), n, in, s.min_s, why);
            std::string why0;
            const auto old = llama_ubatch_taper_plan(s0, n, n_ubatch, head, tail);
            if (!old.empty() && !floor_holds(split_sizes(old, n, n_ubatch, in), n, in, s.min_s, why0)) {
                n_fixed++;
            }
        }
        CHECK(ok, "cuts=1 plan of %s, n %u, n_ubatch %u, head %d, tail %d, cuts %s: %s %s", s.str().c_str(), n, n_ubatch, head, tail,
              str_of(in).c_str(), str_of(ends).c_str(), why.c_str());
    }
    printf("unit, cuts=1: parser, %zu fixed calls with the server's cuts, %d random calls with cuts (%d with a plan, %d whose "
           "cuts=0 split leaves a piece under min beside a cut): %s\n",
           sizeof(cases)/sizeof(cases[0]), n_sweep, n_plans, n_fixed, g_fail > fail0 ? "FAIL" : "OK");
}

static void unit() {
    llama_ubatch_taper t;
    std::string err;

    // parser
    CHECK(t.parse("on", err) && t.str() == "head=256:64,tail=512:128,short=600:64", "on -> %s", t.str().c_str());
    CHECK(t.parse("1", err) && t.str() == "head=256:64,tail=512:128,short=600:64", "1 -> %s", t.str().c_str());
    for (const char * off : { "", "0", "off", " off " }) {
        CHECK(t.parse(off, err) && !t.enabled() && t.str() == "off", "'%s' is not off", off);
    }
    CHECK(t.parse("tail=50:13, head=100:7", err) && t.str() == "head=100:7,tail=50:13", "-> %s", t.str().c_str());
    CHECK(t.parse("short=600:64,head=0:0", err) && t.str() == "short=600:64", "-> %s", t.str().c_str());
    CHECK(t.parse("short=600:4,min=1", err) && t.str() == "short=600:4,min=1" && t.min_s == 1, "-> %s", t.str().c_str());
    CHECK(t.parse("min=9,head=10:2", err) && t.str() == "head=10:2", "-> %s", t.str().c_str());
    for (const char * bad : { "head=1", "head=1:0", "head=0:5", "foo=1:2", "head=a:2", "head=1:2,tail", "head=-1:2",
                              "head=1:2:3", "head=1234567890:2", "min=0", "min=", "min=2:3", "on,min=1" }) {
        llama_ubatch_taper keep;
        keep.parse("on", err);
        CHECK(!keep.parse(bad, err) && keep.str() == "head=256:64,tail=512:128,short=600:64",
              "'%s' accepted or changed the schedule", bad);
    }

    // fixed plans (n_ubatch 512): llama-server's calls of a 450-, 1,500- and 3,000-token prompt with "on"
    t.parse("on", err);
    struct pcase { uint32_t n; bool head, tail; const char * want; };
    const pcase cases[] = {
        {  446, true,  true,  "64,64,64,64,64,64,62" },
        {  984, true,  true,  "64,64,64,64,216,128,128,128,128" },
        {  512, true,  true,  "64,64,64,64,64,64,64,64" },
        { 2048, true,  false, "64,64,64,64,512,512,512,256" },
        {  436, false, true,  "128,128,128,52" },
        {    4, true,  true,  "4" },
        { 2048, false, false, "512,512,512,512" },
        {  700, true,  false, "64,64,64,64,444" },
        // no micro-batch under 9 tokens: a short remainder joins its neighbour
        {  200, true,  false, "72,64,64" },
        {  129, true,  true,  "64,65" },
        {  516, true,  false, "64,64,64,64,260" },
        {  261, true,  true,  "64,64,64,69" },
        {  262, false, true,  "128,134" },
    };
    for (const auto & c : cases) {
        const std::string got = str_of(plan_sizes(t, c.n, 512, c.head, c.tail));
        CHECK(got == c.want, "on, n %u head %d tail %d: %s, want %s", c.n, c.head, c.tail, got.c_str(), c.want);
    }
    CHECK(llama_ubatch_taper_plan(t, 2048, 512, false, false).empty(), "a plain call has a plan");
    CHECK(llama_ubatch_taper_plan(t, 4, 512, true, true).empty(), "a 4-token call has a plan");
    CHECK(llama_ubatch_taper_plan(t, 65, 512, true, true).empty(), "64 + 1 is not one micro-batch of 65");
    // a remainder beside a full n_ubatch micro-batch: the pair is halved
    t.parse("head=4:4", err);
    CHECK(str_of(plan_sizes(t, 516, 512, true, false)) == "258,258", "head=4:4, 516: %s", str_of(plan_sizes(t, 516, 512, true, false)).c_str());
    // min=1 keeps the remainders
    t.parse("head=256:64,tail=512:128,short=600:64,min=1", err);
    CHECK(str_of(plan_sizes(t, 200, 512, true, false)) == "8,64,64,64", "min=1, 200: %s", str_of(plan_sizes(t, 200, 512, true, false)).c_str());
    CHECK(str_of(plan_sizes(t, 129, 512, true, true)) == "64,64,1", "min=1, 129: %s", str_of(plan_sizes(t, 129, 512, true, true)).c_str());
    t.parse("on", err);
    // sizes above n_ubatch are clamped: at n_ubatch 32, short=600:64 is the plain split up to its short remainder;
    // sizes under min are raised to it
    t.parse("short=600:64", err);
    CHECK(str_of(plan_sizes(t, 100, 32, true, true)) == "32,32,18,18", "clamped: %s", str_of(plan_sizes(t, 100, 32, true, true)).c_str());
    t.parse("head=333:1", err);
    {
        std::vector<uint32_t> want(37, 9); // 333 = 37 x 9, then the body's 365
        want.push_back(365);
        CHECK(plan_sizes(t, 698, 512, true, false) == want, "raised: %s", str_of(plan_sizes(t, 698, 512, true, false)).c_str());
    }
    t.parse("off", err);
    CHECK(llama_ubatch_taper_plan(t, 1000, 512, true, true).empty(), "off has a plan");

    // sweep: every plan covers [0, n) in order, with micro-batches of 1 .. n_ubatch tokens, and is empty exactly when it
    // equals the plain split
    std::mt19937 rng(7);
    int n_plans = 0;
    for (int k = 0; k < 4000; ++k) {
        llama_ubatch_taper s;
        s.head_n  = rng() % 3 ? rng() % 1500 : 0; s.head_s  = s.head_n  ? 1 + rng() % 700 : 0;
        s.tail_n  = rng() % 3 ? rng() % 1500 : 0; s.tail_s  = s.tail_n  ? 1 + rng() % 700 : 0;
        s.short_n = rng() % 3 ? rng() % 1500 : 0; s.short_s = s.short_n ? 1 + rng() % 700 : 0;
        const uint32_t n_ubatch = 1 + rng() % 1024;
        const uint32_t n = 1 + rng() % 5000;
        const bool head = rng() & 1;
        const bool tail = rng() & 1;
        const auto ends = llama_ubatch_taper_plan(s, n, n_ubatch, head, tail);
        n_plans += !ends.empty();
        uint32_t prev = 0;
        bool ok = true;
        for (uint32_t e : ends) {
            ok = ok && e > prev && e - prev <= n_ubatch;
            prev = e;
        }
        ok = ok && (ends.empty() || ends.back() == n);
        // and none under min_s tokens where the rule applies
        if (n >= s.min_s && n_ubatch >= 2*s.min_s) {
            for (uint32_t x : sizes_of(ends)) {
                ok = ok && x >= s.min_s;
            }
        }
        CHECK(ok, "plan of %s, n %u, n_ubatch %u, head %d, tail %d: %s", s.str().c_str(), n, n_ubatch, head, tail,
              str_of(ends).c_str());
        if (!ends.empty()) {
            std::vector<uint32_t> plain;
            for (uint32_t pos = 0; pos < n; ) {
                pos = std::min(n, pos + n_ubatch);
                plain.push_back(pos);
            }
            CHECK(ends != plain, "a plan equal to the plain split was returned");
        }
    }
    printf("unit: parser, %zu fixed plans, %d random schedules (%d with a plan): %s\n",
           sizeof(cases)/sizeof(cases[0]), 4000, n_plans, g_fail ? "FAIL" : "OK");

    unit_cuts();
}

//
// model part
//

// the "ubatch taper:" lines the context logs, one per call
static std::vector<std::string> g_taper_log;

static void log_cb(ggml_log_level level, const char * text, void *) {
    const char * p = strstr(text, "ubatch taper: ");
    if (p) {
        g_taper_log.emplace_back(p);
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct call_log {
    uint32_t n = 0;
    int head = -1, tail = -1;
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> cuts; // C905: the cuts inside the call (in-call checkpoints)
};

// "ubatch taper: 2048 tokens, head 1, tail 0, 8 ubatches: 64x4 512x3 256"
static bool parse_log(const std::string & line, call_log & out) {
    unsigned n = 0, nu = 0;
    int h = 0, tl = 0;
    int off = 0;
    if (sscanf(line.c_str(), "ubatch taper: %u tokens, head %d, tail %d, %u ubatches: %n", &n, &h, &tl, &nu, &off) < 4 || off == 0) {
        return false;
    }
    out.n = n; out.head = h; out.tail = tl; out.sizes.clear();
    const char * s = line.c_str() + off;
    while (*s && *s != '\n') {
        unsigned sz = 0, cnt = 1;
        int used = 0;
        if (sscanf(s, "%u%n", &sz, &used) != 1) {
            return false;
        }
        s += used;
        if (*s == 'x') {
            ++s;
            if (sscanf(s, "%u%n", &cnt, &used) != 1) {
                return false;
            }
            s += used;
        }
        out.sizes.insert(out.sizes.end(), cnt, sz);
        while (*s == ' ') {
            ++s;
        }
    }
    return out.sizes.size() == nu;
}

struct arm {
    std::string name;
    std::string spec;   // "off" for the reference
    int hint_head;      // -1: the server's (auto), 1: always
    int hint_tail;      // -1: the server's rule, 1: always
    uint32_t n_rs_seq;
    bool incall = false; // C905: llama-server with --ctx-checkpoints-incall (each call whole, cut inside at the checkpoints)
};

struct run_out {
    std::vector<float> logits;                 // per output, n_vocab each
    std::vector<std::vector<uint8_t>> ckpts;   // the checkpoints read before the tail calls
    std::vector<uint8_t> state;                // the full sequence state after the prompt
    std::vector<llama_token> gen;              // greedy tokens after the prompt
    std::vector<call_log> calls;               // what the context logged
    std::vector<call_log> expect;              // the server's head/tail and the plan, per call
    int n_ubatches = 0;
};

static bool run(llama_context * ctx, const llama_ubatch_taper & t, const arm & a, const std::vector<llama_token> & tokens,
                bool all_outputs, run_out & out) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_batch  = llama_n_batch(ctx);
    const uint32_t n_ubatch = llama_n_ubatch(ctx);
    const uint32_t L = tokens.size();

    llama_memory_clear(llama_get_memory(ctx), true);
    if (!llama_set_ubatch_taper(ctx, a.spec.c_str())) {
        return false;
    }
    g_taper_log.clear();

    // llama-server's cuts: 4 + n_ubatch and 4 tokens before the end (where they fall inside the prompt)
    std::vector<uint32_t> cuts;
    for (uint32_t off : { 4 + n_ubatch, 4u }) {
        if (L > off) {
            cuts.push_back(L - off);
        }
    }

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    uint32_t pos = 0;
    bool after_wait = true; // a fresh memory: nothing in flight
    while (pos < L) {
        uint32_t end = std::min(L, pos + n_batch);
        for (uint32_t c : cuts) {
            if (!a.incall && c > pos && c < end) {
                end = c;
            }
        }
        // C905, in-call checkpoints: the cuts inside this call, each a checkpoint filled at its cut into store k (as
        // the server's ckpt_incall_begin), read after the call in the host format
        std::vector<uint32_t> rel;
        std::vector<int32_t>  stores;
        for (size_t k = 0; a.incall && k < cuts.size(); ++k) {
            if (cuts[k] > pos && cuts[k] < end) {
                rel.push_back(cuts[k] - pos);
                stores.push_back((int32_t) k);
                llama_state_seq_store_at(ctx, 0, (int32_t) k, cuts[k] - pos, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }
        // a checkpoint before each tail call, as the server takes it (the recurrent part; a host wait)
        if (!a.incall && std::find(cuts.begin(), cuts.end(), pos) != cuts.end()) {
            const size_t n = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            std::vector<uint8_t> ck(n);
            llama_state_seq_get_data_ext(ctx, ck.data(), n, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            out.ckpts.push_back(std::move(ck));
            after_wait = true;
        }
        llama_batch_ext_clear(batch);
        bool has_output = false;
        for (uint32_t i = pos; i < end; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            const llama_pos p[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch, idx, p);
            const bool o = all_outputs || i == L - 1;
            llama_batch_ext_set_output_logits(batch, idx, o);
            has_output = has_output || o;
        }
        const uint32_t n = end - pos;
        // the server's rule (decode() in server-context.cpp)
        const int tail = a.hint_tail >= 0 ? a.hint_tail : ((has_output || n < n_batch) ? 1 : 0);
        llama_set_ubatch_taper_next(ctx, a.hint_head, tail);

        call_log e;
        e.n = n;
        e.head = a.hint_head >= 0 ? a.hint_head : (after_wait ? 1 : 0);
        e.tail = tail;
        e.sizes = a.incall ? plan_sizes_cut(t, n, n_ubatch, e.head, e.tail, rel) : plan_sizes(t, n, n_ubatch, e.head, e.tail);
        e.cuts = rel;
        out.expect.push_back(e);

        llama_set_ubatch_cuts(ctx, rel.data(), rel.size());
        const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        llama_set_ubatch_cuts(ctx, nullptr, 0);
        if (rc != 0) {
            fprintf(stderr, "decode failed at %u\n", pos);
            llama_batch_ext_free(batch);
            return false;
        }
        after_wait = false;
        for (const int32_t st : stores) {
            std::vector<uint8_t> blob(llama_state_seq_store_at_get(ctx, nullptr, 0, 0, st, nullptr, nullptr));
            std::vector<uint8_t> host(blob.empty() ? 0 : llama_state_seq_store_get_size_host(ctx, 0, st));
            if (blob.empty() || llama_state_seq_store_at_get(ctx, blob.data(), blob.size(), 0, st, nullptr, nullptr) != blob.size() ||
                host.empty() || llama_state_seq_store_get_data_host(ctx, blob.data(), blob.size(), host.data(), host.size(), 0, st) != host.size()) {
                CHECK(false, "%s: the checkpoint filled at %u inside the call at %u was not taken", a.name.c_str(), cuts[st], pos);
                host.clear();
            }
            out.ckpts.push_back(std::move(host));
            llama_state_seq_store_free(ctx, 0, st);
        }
        if (has_output) {
            for (uint32_t i = pos; i < end; ++i) {
                if (all_outputs || i == L - 1) {
                    const float * l = llama_get_logits_ith(ctx, i - pos);
                    out.logits.insert(out.logits.end(), l, l + n_vocab);
                }
            }
            after_wait = true;
        }
        pos = end;
    }

    {
        const size_t n = llama_state_seq_get_size(ctx, 0);
        out.state.resize(n);
        llama_state_seq_get_data(ctx, out.state.data(), n, 0);
    }

    // 8 greedy tokens
    const float * last = out.logits.data() + out.logits.size() - n_vocab;
    llama_token tok = (llama_token) (std::max_element(last, last + n_vocab) - last);
    for (int g = 0; g < 8; ++g) {
        out.gen.push_back(tok);
        llama_batch_ext_clear(batch);
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tok);
        const llama_pos p[4] = { (llama_pos) (L + g), (llama_pos) (L + g), (llama_pos) (L + g), 0 };
        llama_batch_ext_set_pos(batch, idx, p);
        llama_batch_ext_set_output_logits(batch, idx, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            llama_batch_ext_free(batch);
            return false;
        }
        const float * l = llama_get_logits_ith(ctx, 0);
        tok = (llama_token) (std::max_element(l, l + n_vocab) - l);
    }
    llama_batch_ext_free(batch);

    for (const auto & line : g_taper_log) {
        call_log c;
        if (parse_log(line, c) && c.n > 8) {
            out.calls.push_back(c);
            out.n_ubatches += (int) c.sizes.size();
        }
    }
    return true;
}

// the largest difference of two state blobs read as 4-byte floats (0 when equal, -1 when their sizes differ); prints
// where they differ
static double blob_maxf(const char * what, const std::vector<uint8_t> & x, const std::vector<uint8_t> & y) {
    if (x == y) {
        return 0.0;
    }
    if (x.size() != y.size()) {
        fprintf(stderr, "  %s: %zu and %zu bytes\n", what, x.size(), y.size());
        return -1.0;
    }
    size_t n_b = 0, first = SIZE_MAX, n_w = 0;
    double max_f = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) {
            n_b++;
            first = std::min(first, i);
        }
    }
    for (size_t i = 0; i + 4 <= x.size(); i += 4) {
        float fx, fy;
        memcpy(&fx, x.data() + i, 4);
        memcpy(&fy, y.data() + i, 4);
        if (memcmp(&fx, &fy, 4) != 0) {
            n_w++;
            if (std::isfinite(fx) && std::isfinite(fy)) {
                max_f = std::max(max_f, (double) std::fabs(fx - fy));
            }
        }
    }
    fprintf(stderr, "  %s: %zu bytes, %zu differ from offset %zu, %zu words, largest float difference %.3e\n",
            what, x.size(), n_b, first, n_w, max_f);
    return max_f;
}

// the base's own rounding at a one-token batch, with the taper off: n tokens in one call against the first n - 1 in
// one call and the last in a call of its own; the largest float difference of the recurrent states after them
static double base_one_token(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t n) {
    std::vector<uint8_t> st[2];
    llama_set_ubatch_taper(ctx, "off");
    for (int k = 0; k < 2; ++k) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        const std::vector<std::pair<uint32_t, uint32_t>> calls = k == 0
            ? std::vector<std::pair<uint32_t, uint32_t>>{ { 0, n } }
            : std::vector<std::pair<uint32_t, uint32_t>>{ { 0, n - 1 }, { n - 1, n } };
        for (const auto & c : calls) {
            llama_batch_ext_clear(batch);
            for (uint32_t i = c.first; i < c.second; ++i) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
                const llama_pos p[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
                llama_batch_ext_set_pos(batch, idx, p);
                llama_batch_ext_set_output_logits(batch, idx, i + 1 == c.second);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
                llama_batch_ext_free(batch);
                return -2.0;
            }
        }
        llama_batch_ext_free(batch);
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        st[k].resize(sz);
        llama_state_seq_get_data_ext(ctx, st[k].data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }
    return blob_maxf("base, one call against n - 1 and 1", st[0], st[1]);
}

// C905, --plan-check: per spec and call size N, one call of N tokens decoded as llama-server's prompt call with in-call
// checkpoints (head and tail, the cuts at N - 4 - n_ubatch and N - 4), and the micro-batches the context logged; the
// planned column is the plan with the cuts as split_simple takes it (without the recurrent rule that n_rs_seq adds)
static int plan_check(llama_model * model, const std::vector<uint32_t> & sizes, const std::vector<std::string> & specs,
                      uint32_t rs, int n_threads, FILE * out) {
    const uint32_t n_max = *std::max_element(sizes.begin(), sizes.end());
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = GGML_PAD(n_max + 64, 256);
    cp.n_batch = std::max<uint32_t>(4096, n_max);
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.n_rs_seq = rs;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context (n_rs_seq %u)\n", rs);
        return 1;
    }
    const uint32_t n_ubatch = llama_n_ubatch(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(out, "spec\tn_rs_seq\tn_ubatch\tcall_tokens\tcuts\tlogged\tplanned\tlogged_as_planned\tpieces_under_min\tfloor\n");
    int n_bad = 0;
    for (const auto & spec : specs) {
        llama_ubatch_taper t;
        std::string err;
        if (!t.parse(spec.c_str(), err) || !llama_set_ubatch_taper(ctx, spec.c_str())) {
            fprintf(stderr, "bad spec '%s': %s\n", spec.c_str(), err.c_str());
            llama_free(ctx);
            return 1;
        }
        for (uint32_t n : sizes) {
            std::mt19937 rng(5000 + n);
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch_ext * batch = llama_batch_ext_init(ctx);
            for (uint32_t i = 0; i < n; ++i) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, (llama_token) (rng() % n_vocab));
                const llama_pos p[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
                llama_batch_ext_set_pos(batch, idx, p);
                llama_batch_ext_set_output_logits(batch, idx, i + 1 == n);
            }
            const std::vector<uint32_t> cuts = server_cuts(n, n_ubatch);
            g_taper_log.clear();
            llama_set_ubatch_cuts(ctx, cuts.data(), cuts.size());
            llama_set_ubatch_taper_next(ctx, 1, 1);
            const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
            llama_set_ubatch_cuts(ctx, nullptr, 0);
            llama_batch_ext_free(batch);
            if (rc != 0) {
                fprintf(stderr, "decode of %u tokens failed\n", n);
                llama_free(ctx);
                return 1;
            }
            call_log c;
            const bool logged = !g_taper_log.empty() && parse_log(g_taper_log.back(), c) && c.n == n;
            const std::vector<uint32_t> planned = plan_sizes_cut(t, n, n_ubatch, true, true, cuts);
            std::string why = "holds";
            int n_small = 0;
            if (logged) {
                std::string w;
                if (!floor_holds(c.sizes, n, cuts, t.min_s, w)) {
                    why = w;
                }
                for (uint32_t x : c.sizes) {
                    n_small += x < t.min_s;
                }
            }
            n_bad += !logged;
            fprintf(out, "%s\t%u\t%u\t%u\t%s\t%s\t%s\t%s\t%d\t%s\n", spec.c_str(), llama_n_rs_seq(ctx), n_ubatch, n,
                    str_of(cuts).c_str(), logged ? compact_of(c.sizes).c_str() : "-", compact_of(planned).c_str(),
                    logged ? (c.sizes == planned ? "yes" : "no") : "-", n_small, logged ? why.c_str() : "not logged");
        }
    }
    llama_free(ctx);
    return n_bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tsv_path;
    int n_threads = 4;
    bool unit_only = false;
    std::vector<uint32_t> lens = { 100, 600, 3000, 9000 };
    std::vector<uint32_t> plan_lens;
    std::vector<std::string> plan_specs = { "short=1400:64", "short=1400:64,cuts=1" };
    uint32_t plan_rs = 3;
    auto split_list = [](const std::string & s, char sep) {
        std::vector<std::string> res;
        for (size_t p = 0; p < s.size(); ) {
            const size_t q = s.find(sep, p);
            res.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
            p = q == std::string::npos ? s.size() : q + 1;
        }
        return res;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--plan-check" && i + 1 < argc) {
            plan_lens.clear();
            for (const auto & x : split_list(argv[++i], ',')) {
                plan_lens.push_back((uint32_t) atoi(x.c_str()));
            }
        } else if (a == "--plan-specs" && i + 1 < argc) {
            plan_specs = split_list(argv[++i], ';');
        } else if (a == "--plan-rs" && i + 1 < argc) {
            plan_rs = (uint32_t) atoi(argv[++i]);
        } else if (a == "--unit-only") {
            unit_only = true;
        } else if (a == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "-t" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (a == "--tsv" && i + 1 < argc) {
            tsv_path = argv[++i];
        } else if (a == "--lens" && i + 1 < argc) {
            lens.clear();
            std::string s = argv[++i];
            for (size_t p = 0; p < s.size(); ) {
                const size_t q = s.find(',', p);
                lens.push_back((uint32_t) atoi(s.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str()));
                p = q == std::string::npos ? s.size() : q + 1;
            }
        } else {
            fprintf(stderr, "usage: %s [--unit-only] [-m <gguf>] [-t <threads>] [--lens 100,600,3000,9000] [--tsv <file>]\n"
                            "       [--plan-check <N,N,...> [--plan-specs <spec;spec;...>] [--plan-rs <n_rs_seq>]]\n", argv[0]);
            return 2;
        }
    }

    unit();
    if (unit_only || model_path.empty()) {
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", model_path.c_str());
        return 1;
    }

    if (!plan_lens.empty()) {
        FILE * out = tsv_path.empty() ? stdout : fopen(tsv_path.c_str(), "w");
        const int rc = out ? plan_check(model, plan_lens, plan_specs, plan_rs, n_threads, out) : 1;
        if (out && out != stdout) {
            fclose(out);
        }
        llama_model_free(model);
        llama_backend_free();
        printf("plan check: %s\n", rc == 0 && !g_fail ? "PASS" : "FAIL");
        return rc == 0 && !g_fail ? 0 : 1;
    }
    const uint32_t L_max = *std::max_element(lens.begin(), lens.end());

    // tiny1 (min=1, micro-batches of one token) goes through the one-token path that the others avoid: its states may
    // differ from off's in the last bits, as the base's own one-token batches do (see "base:" below)
    const std::vector<arm> arms = {
        { "off",       "off",                                        -1, -1, 0 },
        { "on",        "on",                                         -1, -1, 0 },
        { "small",     "head=512:16,tail=512:16,short=600:16",        -1, -1, 0 },
        { "odd",       "head=100:7,tail=50:13,short=300:5",           -1, -1, 0 },
        { "uniform",   "short=1000000:32",                            1,  1, 0 },
        { "tiny1",     "head=1000000:1,tail=1000000:1,short=1000000:1,min=1", 1, 1, 0 },
        { "off-rs3",   "off",                                        -1, -1, 3 },
        { "on-rs3",    "on",                                         -1, -1, 3 },
        { "odd-rs3",   "head=100:7,tail=50:13,short=300:5",           -1, -1, 3 },
        // C905: llama-server's calls with in-call checkpoints (--ctx-checkpoints-incall), against them with the taper off;
        // without cuts=1 (the plans of whole calls, as before) and with it. n_rs_seq 3 is Flash-Next's line (an MTP draft
        // of three tokens: split_equal keeps a segment's last four tokens together)
        { "off-in",    "off",                                        -1, -1, 0, true },
        { "on-in",     "on",                                         -1, -1, 0, true },
        { "onc-in",    "head=256:64,tail=512:128,short=600:64,cuts=1", -1, -1, 0, true },
        { "v5-in",     "short=1400:64",                              -1, -1, 0, true },
        { "v5c-in",    "short=1400:64,cuts=1",                       -1, -1, 0, true },
        { "smallc-in", "head=512:16,tail=512:16,short=600:16,cuts=1", -1, -1, 0, true },
        { "oddc-in",   "head=100:7,tail=50:13,short=300:5,cuts=1",    -1, -1, 0, true },
        { "off-in-rs3",  "off",                                      -1, -1, 3, true },
        { "v5-in-rs3",   "short=1400:64",                            -1, -1, 3, true },
        { "v5c-in-rs3",  "short=1400:64,cuts=1",                     -1, -1, 3, true },
        { "onc-in-rs3",  "head=256:64,tail=512:128,short=600:64,cuts=1", -1, -1, 3, true },
        { "oddc-in-rs3", "head=100:7,tail=50:13,short=300:5,cuts=1",  -1, -1, 3, true },
    };

    FILE * tsv = tsv_path.empty() ? nullptr : fopen(tsv_path.c_str(), "w");
    if (tsv) {
        fprintf(tsv, "arm\tspec\tn_rs_seq\tprompt\toutputs\tcalls\tubatches\tlogits\tdiffer\tmax_abs_diff\tnmse\tckpts_same\tckpts_max_float_diff\tstate_same\tstate_max_float_diff\tgen_same\tsplits_as_planned\thead_tail_as_server\tverdict\n");
    }

    int n_rows = 0;
    int n_bad = 0;
    for (uint32_t rs : { 0u, 3u }) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = GGML_PAD(L_max + 64, 256);
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;
        cp.n_rs_seq = rs;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            fprintf(stderr, "cannot create a context (n_rs_seq %u)\n", rs);
            return 1;
        }
        const uint32_t rs_eff = llama_n_rs_seq(ctx);
        if (rs == 0) {
            std::mt19937 rng(77);
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            std::vector<llama_token> tokens(512);
            for (auto & x : tokens) {
                x = (llama_token) (rng() % n_vocab);
            }
            for (uint32_t n : { 96u, 512u }) {
                const double f = base_one_token(ctx, tokens, n);
                printf("base: %u tokens in one call against %u + 1, taper off: recurrent states %s (largest float difference %.3e)\n",
                       n, n - 1, f == 0.0 ? "the same" : "differ", f);
            }
        }
        for (uint32_t L : lens) {
            std::mt19937 rng(1000 + L);
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            std::vector<llama_token> tokens(L);
            for (auto & x : tokens) {
                x = (llama_token) (rng() % n_vocab);
            }
            for (bool all : { true, false }) {
                run_out ref;
                const arm * ref_arm = nullptr;
                for (const auto & a : arms) {
                    if (a.n_rs_seq != rs) {
                        continue;
                    }
                    llama_ubatch_taper t;
                    std::string err;
                    t.parse(a.spec.c_str(), err);
                    run_out o;
                    if (!run(ctx, t, a, tokens, all, o)) {
                        CHECK(false, "%s: run failed at %u tokens", a.name.c_str(), L);
                        continue;
                    }
                    if (a.spec == "off") {
                        ref = std::move(o);
                        ref_arm = &a;
                        CHECK(ref.calls.empty(), "%s logged %zu taper lines", a.name.c_str(), ref.calls.size());
                        continue;
                    }
                    GGML_ASSERT(ref_arm);
                    // logits
                    size_t n_diff = 0;
                    double max_abs = 0.0, mse = 0.0, msa = 0.0;
                    const bool same_n = o.logits.size() == ref.logits.size();
                    for (size_t i = 0; same_n && i < o.logits.size(); ++i) {
                        const double d = (double) o.logits[i] - ref.logits[i];
                        n_diff += o.logits[i] != ref.logits[i];
                        max_abs = std::max(max_abs, std::fabs(d));
                        mse += d*d;
                        msa += (double) ref.logits[i]*ref.logits[i];
                    }
                    const double nmse = msa > 0 ? mse/msa : 0.0;
                    const bool ck_same = o.ckpts == ref.ckpts;
                    const bool st_same = o.state == ref.state;
                    double ck_f = o.ckpts.size() == ref.ckpts.size() ? 0.0 : -1.0;
                    for (size_t c = 0; ck_f >= 0 && c < o.ckpts.size(); ++c) {
                        const double f = blob_maxf(("checkpoint " + std::to_string(c)).c_str(), ref.ckpts[c], o.ckpts[c]);
                        ck_f = f < 0 ? -1.0 : std::max(ck_f, f);
                    }
                    const double st_f = blob_maxf("state", ref.state, o.state);
                    // an arm that may run micro-batches of 1-8 tokens (min under 9) takes the one-token path, whose
                    // rounding differs as the base's own one-token batches do: its states are compared as floats. So
                    // does an in-call arm without cuts=1 (C905): its plan leaves such pieces beside the cuts
                    const bool one_token = t.min_s < 9 || (a.incall && !t.cuts);
                    const bool states_ok = one_token ? (ck_f >= 0 && ck_f <= 1e-5 && st_f >= 0 && st_f <= 1e-5) : (ck_same && st_same);
                    const bool gen_same = o.gen == ref.gen;
                    // splits: logged against planned, one logged line per call of more than 8 tokens; with n_rs_seq the
                    // last n_rs_seq + 1 tokens stay in one micro-batch, which may move the last boundary of a call
                    std::vector<call_log> exp8;
                    for (const auto & e : o.expect) {
                        if (e.n > 8) {
                            exp8.push_back(e);
                        }
                    }
                    bool splits = exp8.size() == o.calls.size();
                    bool ht = splits;
                    for (size_t c = 0; splits && c < exp8.size(); ++c) {
                        ht = ht && exp8[c].head == o.calls[c].head && exp8[c].tail == o.calls[c].tail;
                        uint32_t sum = 0;
                        bool cap = true;
                        for (uint32_t s : o.calls[c].sizes) {
                            sum += s;
                            cap = cap && s >= 1 && s <= 512;
                        }
                        if (rs_eff == 0 || (a.incall && t.cuts)) {
                            // with cuts=1 every piece of a segment of min or more tokens holds min or more, so the
                            // recurrent rule (n_rs_seq + 1 trailing tokens together) leaves the plan as it is
                            splits = splits && o.calls[c].sizes == exp8[c].sizes;
                        } else {
                            const auto & sz = o.calls[c].sizes;
                            splits = splits && sum == exp8[c].n && cap && (sz.back() >= rs_eff + 1 || sz.size() == 1);
                        }
                        // C905: with cuts=1, no logged piece under min unless the cuts make a whole segment shorter
                        if (a.incall && t.cuts) {
                            std::string why;
                            const bool fl = floor_holds(o.calls[c].sizes, o.calls[c].n, exp8[c].cuts, t.min_s, why);
                            if (!fl) {
                                fprintf(stderr, "%s, %u tokens, call %zu: %s\n", a.name.c_str(), L, c, why.c_str());
                            }
                            splits = splits && fl;
                        }
                        if (!splits || !ht) {
                            fprintf(stderr, "%s, %u tokens, call %zu: logged %u tokens head %d tail %d [%s], planned head %d tail %d [%s]\n",
                                    a.name.c_str(), L, c, o.calls[c].n, o.calls[c].head, o.calls[c].tail, str_of(o.calls[c].sizes).c_str(),
                                    exp8[c].head, exp8[c].tail, str_of(exp8[c].sizes).c_str());
                        }
                    }
                    const bool pass = same_n && nmse <= 1e-8 && states_ok && gen_same && splits && ht;
                    n_rows++;
                    n_bad += !pass;
                    CHECK(pass, "%s (n_rs_seq %u), %u tokens, outputs %s: differ %zu nmse %.3e ckpts %d (%.3e) state %d (%.3e) gen %d splits %d head/tail %d",
                          a.name.c_str(), rs_eff, L, all ? "all" : "last", n_diff, nmse, ck_same, ck_f, st_same, st_f, gen_same, splits, ht);
                    auto blob_word = [](bool same, double f) {
                        char buf[32];
                        snprintf(buf, sizeof(buf), "within %.1e", f);
                        return same ? std::string("same") : f < 0 ? std::string("SIZE") : std::string(buf);
                    };
                    printf("%-8s rs %u %5u tokens, outputs %-4s: %2zu calls, %4d ubatches, %zu logits, %zu differ (max %.3e, nmse %.3e), "
                           "ckpts %s, state %s, tokens %s, splits %s, head/tail %s: %s\n",
                           a.name.c_str(), rs_eff, L, all ? "all" : "last", o.expect.size(), o.n_ubatches, o.logits.size(), n_diff,
                           max_abs, nmse, blob_word(ck_same, ck_f).c_str(), blob_word(st_same, st_f).c_str(), gen_same ? "same" : "DIFFER",
                           splits ? "as planned" : "OFF PLAN", ht ? "as server" : "WRONG", pass ? "OK" : "FAIL");
                    if (tsv) {
                        fprintf(tsv, "%s\t%s\t%u\t%u\t%s\t%zu\t%d\t%zu\t%zu\t%.3e\t%.3e\t%d\t%.3e\t%d\t%.3e\t%d\t%d\t%d\t%s\n", a.name.c_str(),
                                t.str().c_str(), rs_eff, L, all ? "all" : "last", o.expect.size(), o.n_ubatches, o.logits.size(),
                                n_diff, max_abs, nmse, ck_same, ck_f, st_same, st_f, gen_same, splits, ht, pass ? "OK" : "FAIL");
                    }
                }
            }
        }
        llama_free(ctx);
    }
    if (tsv) {
        fclose(tsv);
    }
    llama_model_free(model);
    llama_backend_free();

    printf("model: %d comparisons against off, %d failed\n", n_rows, n_bad);
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
