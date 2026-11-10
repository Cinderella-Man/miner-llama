// Block verification (common_spec_block_compute / common_spec_block_residual, the arithmetic of
// common_sampler_sample_and_accept_n_block; Sun et al., "Block Verification Accelerates Speculative Decoding",
// arXiv:2403.10444, Algorithm 2) and the grammar's forced text (llama_grammar_forced_text):
//   1. the paper's worked example (its Section 2: vocabulary {a, b}, M_b = (1/3, 2/3), M_s = (2/3, 1/3), two drafted
//      tokens): the acceptance of each draft as the paper states it, and 11/9 expected accepted tokens against token
//      verification's 10/9
//   2. exactness by exact enumeration on random context-dependent targets and drafts: the distribution of the first
//      g + 1 output tokens (accepted tokens, the next token, then the target's own continuation) equals the target's,
//      also with drafted tokens the target cannot produce (a grammar's zeros), drafts of top-k support, draft lengths
//      that depend on the drafted prefix and point-mass proposals (llama-server's grammar-forced tokens), and with a
//      block cut at an end-of-generation token that takes no next token when it is accepted whole
//      (LLAMA_SPEC_REJECT_STOP, Q1263)
//   3. block verification accepts no fewer tokens in expectation than token-by-token rejection, on every instance
//   4. llama_grammar_forced_text on small grammars
// Exits 0 when every check passes.

#include "sampling.h"
#include "llama.h"

#include "../src/llama-grammar.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_failed = 0;

static void check(bool ok, const char * what, double got = 0.0, double want = 0.0) {
    n_checks++;
    if (!ok) {
        n_failed++;
        fprintf(stderr, "FAIL: %s (got %.12f, want %.12f)\n", what, got, want);
    }
}

// the distributions are stored as llama_token_data, in float: about 1e-7 of rounding
static bool near(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

using dist_t = std::vector<llama_token_data>;

static dist_t make_dist(const std::vector<double> & p) {
    dist_t d;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] > 0.0) {
            d.push_back({ (llama_token) i, 0.0f, (float) p[i] });
        }
    }
    return d;
}

static double prob(const dist_t & d, llama_token id) {
    for (const auto & e : d) {
        if (e.id == id) {
            return e.p;
        }
    }
    return 0.0;
}

// what the server's walk computes for one drafted block: the target's distributions up to the first position whose
// weight is 0 (or the last drafted position), then the weights
struct block_case {
    std::vector<dist_t> pb;
    common_spec_block_weights w;
};

static block_case walk(const std::function<dist_t(const llama_tokens &)> & target, const llama_tokens & prefix,
        const llama_tokens & draft, const std::vector<dist_t> & q) {
    block_case c;
    double p = 1.0;
    llama_tokens ctx = prefix;
    for (size_t i = 0; i < draft.size(); ++i) {
        c.pb.push_back(target(ctx));
        const double px = prob(c.pb.back(), draft[i]);
        const double qx = prob(q[i], draft[i]);
        p = qx > 0.0 ? std::min(1.0, p * px / qx) : 0.0;
        if (p <= 0.0) {
            break;
        }
        ctx.push_back(draft[i]);
    }
    c.w = common_spec_block_compute(c.pb, q, draft);
    return c;
}

// P(tau = t) under the backward scan: the largest i with eta_i < h_i
static std::vector<double> tau_probs(const common_spec_block_weights & w) {
    const size_t g = w.h.size() - 1;
    std::vector<double> pt(g + 1, 0.0);
    double rest = 1.0; // none of the positions above passed
    for (size_t i = g; i >= 1; --i) {
        const double h = std::min(1.0, std::max(0.0, w.h[i]));
        pt[i] = rest * h;
        rest *= 1.0 - h;
    }
    pt[0] = rest;
    return pt;
}

//
// 1. the paper's example
//

static void test_paper_example() {
    const dist_t mb = make_dist({ 1.0/3, 2.0/3 });
    const dist_t ms = make_dist({ 2.0/3, 1.0/3 });
    auto target = [&](const llama_tokens &) { return mb; };
    const std::vector<dist_t> q = { ms, ms };

    const llama_token a = 0;
    const llama_token b = 1;

    double e_block = 0.0;
    double e_token = 0.0;

    for (llama_token x1 : { a, b }) {
        for (llama_token x2 : { a, b }) {
            const llama_tokens draft = { x1, x2 };
            const double pd = prob(ms, x1) * prob(ms, x2);

            const auto c  = walk(target, {}, draft, q);
            const auto pt = tau_probs(c.w);

            e_block += pd * (pt[1] * 1 + pt[2] * 2);

            // token verification: accept X_i with min(1, M_b/M_s) while all before it were accepted
            const double r1 = std::min(1.0, prob(mb, x1) / prob(ms, x1));
            const double r2 = std::min(1.0, prob(mb, x2) / prob(ms, x2));
            e_token += pd * (r1 + r1 * r2);

            if (x1 == a && x2 == a) {
                // "Pr(Accept X1X2) = 1/4, and else the algorithm rejects both tokens and only corrects the first token to b"
                check(near(pt[2], 0.25), "paper aa: both accepted with probability 1/4", pt[2], 0.25);
                check(near(pt[0], 0.75), "paper aa: else both rejected", pt[0], 0.75);
                const auto res = common_spec_block_residual(c.pb[0], q[0], c.w.p[0]);
                check(near(prob(res, b), 1.0), "paper aa: the first token corrected to b", prob(res, b), 1.0);
            } else if (x2 == b) {
                // "When the draft tokens X1X2 = ab or bb, Pr(Accept X1X2) = 1"
                check(near(pt[2], 1.0), "paper ab/bb: both accepted", pt[2], 1.0);
            } else {
                // ba: "it always accepts b, and then accepts a with probability 1/2 (else it corrects the second token to b)"
                check(near(pt[1] + pt[2], 1.0), "paper ba: b always accepted", pt[1] + pt[2], 1.0);
                check(near(pt[2], 0.5), "paper ba: a accepted with probability 1/2", pt[2], 0.5);
                const auto res = common_spec_block_residual(c.pb[1], q[1], c.w.p[1]);
                check(near(prob(res, b), 1.0), "paper ba: the second token corrected to b", prob(res, b), 1.0);
            }
        }
    }

    check(near(e_block, 11.0/9), "paper: block verification accepts 11/9 tokens in expectation", e_block, 11.0/9);
    check(near(e_token, 10.0/9), "paper: token verification accepts 10/9", e_token, 10.0/9);
    printf("paper example: E[accepted] block %.6f (11/9 = %.6f), token %.6f (10/9 = %.6f)\n",
            e_block, 11.0/9, e_token, 10.0/9);
}

//
// 2. and 3. exact enumeration
//

struct toy {
    int    n_vocab;
    int    top_k;      // the draft's support (0: all)
    double p_zero;     // a target token is 0 with this probability (a grammar's mask), never all of them
    double p_point;    // a draft position is a point mass on a token chosen from the prefix (a forced token)
    double p_stop;     // the drafter stops after a prefix with this probability (a function of the prefix)
    int    eog;        // >= 0: an end-of-generation token. The block ends after it (the cut of LLAMA_SPEC_REJECT_STOP),
                       // and a block it ends before g_max takes no next token when it is accepted whole
    uint64_t seed;

    std::map<llama_tokens, dist_t> mb_cache, ms_cache;
    std::map<llama_tokens, bool>   stop_cache;

    uint64_t key(const llama_tokens & ctx, uint64_t salt) const {
        uint64_t h = seed * 0x9e3779b97f4a7c15ull + salt;
        for (llama_token t : ctx) {
            h = (h ^ (uint64_t) (t + 1)) * 0x100000001b3ull;
        }
        return h;
    }

    const dist_t & mb(const llama_tokens & ctx) {
        auto it = mb_cache.find(ctx);
        if (it != mb_cache.end()) {
            return it->second;
        }
        std::mt19937_64 rng(key(ctx, 1));
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> p(n_vocab);
        double s = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            p[i] = u(rng) < p_zero ? 0.0 : std::pow(u(rng), 3.0) + 1e-3;
            s += p[i];
        }
        if (s == 0.0) {
            p[rng() % n_vocab] = 1.0;
            s = 1.0;
        }
        for (auto & x : p) {
            x /= s;
        }
        return mb_cache[ctx] = make_dist(p);
    }

    const dist_t & ms(const llama_tokens & ctx) {
        auto it = ms_cache.find(ctx);
        if (it != ms_cache.end()) {
            return it->second;
        }
        std::mt19937_64 rng(key(ctx, 2));
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> p(n_vocab, 0.0);
        if (u(rng) < p_point) {
            p[rng() % n_vocab] = 1.0;
        } else {
            for (int i = 0; i < n_vocab; ++i) {
                p[i] = std::pow(u(rng), 2.0) + 1e-3;
            }
            if (top_k > 0 && top_k < n_vocab) {
                std::vector<double> s = p;
                std::sort(s.begin(), s.end(), std::greater<double>());
                const double cut = s[top_k - 1];
                for (auto & x : p) {
                    if (x < cut) {
                        x = 0.0;
                    }
                }
            }
            double s = 0.0;
            for (auto x : p) {
                s += x;
            }
            for (auto & x : p) {
                x /= s;
            }
        }
        return ms_cache[ctx] = make_dist(p);
    }

    bool stop(const llama_tokens & ctx) {
        auto it = stop_cache.find(ctx);
        if (it != stop_cache.end()) {
            return it->second;
        }
        std::mt19937_64 rng(key(ctx, 3));
        return stop_cache[ctx] = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p_stop;
    }
};

// all sequences of n tokens over the vocabulary
static void for_each_seq(int n_vocab, int n, const std::function<void(const llama_tokens &)> & fn) {
    llama_tokens s(n, 0);
    while (true) {
        fn(s);
        int k = n - 1;
        while (k >= 0 && ++s[k] == n_vocab) {
            s[k] = 0;
            --k;
        }
        if (k < 0) {
            return;
        }
    }
}

// the target's own continuation: add P(sequence) * the target's probability of extending `out` to length L
static void continue_target(toy & t, const llama_tokens & out, double p, int L, std::map<llama_tokens, double> & acc) {
    if ((int) out.size() == L) {
        acc[out] += p;
        return;
    }
    for (const auto & e : t.mb(out)) {
        llama_tokens next = out;
        next.push_back(e.id);
        continue_target(t, next, p * e.p, L, acc);
    }
}

// one instance: exact output distribution of block verification over the first g_max + 1 tokens, against the
// target's; E[accepted] of block and token verification
static void test_instance(toy & t, int g_max, double & e_block, double & e_token, double & max_err) {
    const int L = g_max + 1;
    std::map<llama_tokens, double> out_dist;

    e_block = 0.0;
    e_token = 0.0;

    // the drafter's process: draft token by token from M_s until g_max or its stop rule (a function of the prefix)
    std::function<void(llama_tokens, double)> draft_rec = [&](llama_tokens draft, double pd) {
        const bool done = (int) draft.size() == g_max || (!draft.empty() && (t.stop(draft) || draft.back() == t.eog));
        if (!done) {
            for (const auto & e : t.ms(draft)) {
                llama_tokens next = draft;
                next.push_back(e.id);
                draft_rec(next, pd * e.p);
            }
            return;
        }
        if (draft.empty()) {
            return; // never: the stop rule applies after one token at least
        }

        const size_t g = draft.size();
        std::vector<dist_t> q(g);
        for (size_t i = 0; i < g; ++i) {
            q[i] = t.ms(llama_tokens(draft.begin(), draft.begin() + i));
        }

        const auto c  = walk([&](const llama_tokens & ctx) { return t.mb(ctx); }, {}, draft, q);
        const auto pt = tau_probs(c.w);

        const bool cut = (int) g < g_max && draft.back() == t.eog;

        for (size_t tau = 0; tau <= g; ++tau) {
            if (pt[tau] <= 0.0) {
                continue;
            }
            e_block += pd * pt[tau] * tau;

            const llama_tokens acc(draft.begin(), draft.begin() + tau);

            // a cut block accepted whole: no next token, and the target's own continuation from there
            if (tau == g && cut) {
                continue_target(t, acc, pd * pt[tau], L, out_dist);
                continue;
            }

            dist_t next;
            if (tau == g) {
                next = t.mb(acc);
            } else {
                next = common_spec_block_residual(c.pb[tau], q[tau], c.w.p[tau]);
                if (next.empty()) {
                    next = c.pb[tau];
                }
            }
            for (const auto & e : next) {
                llama_tokens out = acc;
                out.push_back(e.id);
                continue_target(t, out, pd * pt[tau] * e.p, L, out_dist);
            }
        }

        // token verification on the same draft
        double run = 1.0;
        for (size_t i = 0; i < g; ++i) {
            const double qx = prob(q[i], draft[i]);
            const double px = prob(t.mb(llama_tokens(draft.begin(), draft.begin() + i)), draft[i]);
            run *= qx > 0.0 ? std::min(1.0, px / qx) : 0.0;
            e_token += pd * run;
        }
    };
    draft_rec({}, 1.0);

    // the target's joint distribution of L tokens
    max_err = 0.0;
    double total = 0.0;
    for_each_seq(t.n_vocab, L, [&](const llama_tokens & s) {
        double p = 1.0;
        for (int k = 0; k < L && p > 0.0; ++k) {
            p *= prob(t.mb(llama_tokens(s.begin(), s.begin() + k)), s[k]);
        }
        const auto it = out_dist.find(s);
        const double got = it == out_dist.end() ? 0.0 : it->second;
        max_err = std::max(max_err, std::fabs(got - p));
        total += got;
    });
    max_err = std::max(max_err, std::fabs(total - 1.0));
}

static void test_enumeration() {
    struct config { const char * name; int n_vocab, top_k, g_max; double p_zero, p_point, p_stop; int n; int eog = -1; };
    const config cfgs[] = {
        { "dense, g = 1",                         3, 0, 1, 0.0,  0.0, 0.0, 300 },
        { "dense, g = 2",                         3, 0, 2, 0.0,  0.0, 0.0, 300 },
        { "dense, g = 3",                         3, 0, 3, 0.0,  0.0, 0.0, 300 },
        { "dense, g = 4",                         3, 0, 4, 0.0,  0.0, 0.0, 100 },
        { "4 tokens, g = 3",                      4, 0, 3, 0.0,  0.0, 0.0, 100 },
        { "target zeros (a grammar), g = 3",      4, 0, 3, 0.35, 0.0, 0.0, 200 },
        { "draft top-2, g = 3",                   4, 2, 3, 0.0,  0.0, 0.0, 200 },
        { "zeros + top-2, g = 3",                 4, 2, 3, 0.3,  0.0, 0.0, 200 },
        { "point masses (forced tokens), g = 3",  3, 0, 3, 0.2,  0.4, 0.0, 200 },
        { "length set by the prefix, g <= 4",     3, 0, 4, 0.0,  0.0, 0.4, 200 },
        { "all of them, g <= 4",                  3, 2, 4, 0.25, 0.3, 0.3, 200 },
        { "cut at an EOG, g <= 4",                3, 0, 4, 0.0,  0.0, 0.0, 200, 0 },
        { "cut at an EOG + all of them, g <= 4",  3, 2, 4, 0.25, 0.3, 0.3, 200, 0 },
    };

    for (const auto & cfg : cfgs) {
        double worst_err   = 0.0;
        double worst_gap   = 1e9; // min over instances of E_block - E_token
        double sum_block   = 0.0;
        double sum_token   = 0.0;
        for (int k = 0; k < cfg.n; ++k) {
            toy t;
            t.n_vocab = cfg.n_vocab;
            t.top_k   = cfg.top_k;
            t.p_zero  = cfg.p_zero;
            t.p_point = cfg.p_point;
            t.p_stop  = cfg.p_stop;
            t.eog     = cfg.eog;
            t.seed    = 1000003ull * (uint64_t) (&cfg - cfgs) + (uint64_t) k + 1;

            double e_block = 0.0, e_token = 0.0, err = 0.0;
            test_instance(t, cfg.g_max, e_block, e_token, err);

            worst_err  = std::max(worst_err, err);
            worst_gap  = std::min(worst_gap, e_block - e_token);
            sum_block += e_block;
            sum_token += e_token;
        }
        check(worst_err <= 1e-6, (std::string("exact output distribution: ") + cfg.name).c_str(), worst_err, 0.0);
        check(worst_gap >= -1e-6, (std::string("block >= token verification: ") + cfg.name).c_str(), worst_gap, 0.0);
        printf("%-40s %4d instances: max |P_out - P_target| %.2e, E[accepted] block %.4f, token %.4f (+%.2f %%), "
               "min gap %.2e\n", cfg.name, cfg.n, worst_err, sum_block / cfg.n, sum_token / cfg.n,
               100.0 * (sum_block / sum_token - 1.0), worst_gap);
    }
}

//
// limits of the weights
//

static void test_limits() {
    // target = draft at every position: every weight 1, every acceptance 1
    const dist_t d = make_dist({ 0.5, 0.3, 0.2 });
    const std::vector<dist_t> q = { d, d, d };
    const auto w = common_spec_block_compute({ d, d, d }, q, { 0, 1, 2 });
    check(near(w.p[3], 1.0) && near(w.h[3], 1.0), "equal target and draft: the block accepted", w.h[3], 1.0);
    check(near(w.h[1], 1.0) && near(w.h[2], 1.0), "equal target and draft: the 0/0 limit is 1", w.h[1], 1.0);

    // a drafted token the target cannot produce: no position from it on is accepted
    const dist_t t0 = make_dist({ 0.0, 0.6, 0.4 });
    const auto w2 = common_spec_block_compute({ d, t0 }, q, { 0, 0, 1 });
    check(w2.p[2] == 0.0 && w2.p[3] == 0.0 && w2.h[2] == 0.0 && w2.h[3] == 0.0,
            "a token the target cannot produce: nothing from it on accepted", w2.h[3], 0.0);

    // the residual of equal distributions has no mass
    check(common_spec_block_residual(d, d, 1.0).empty(), "no residual mass when target = draft");
}

//
// 4. the grammar's forced text
//

static llama_grammar * grammar_init(const std::string & str, bool lazy = false) {
    const char * trigger = "<tool_call>";
    return llama_grammar_init_impl(nullptr, str.c_str(), "root", lazy, lazy ? &trigger : nullptr, lazy ? 1 : 0, nullptr, 0);
}

static void check_forced(llama_grammar * g, const std::string & want, const char * what) {
    const std::string got = llama_grammar_forced_text(*g, 64);
    n_checks++;
    if (got != want) {
        n_failed++;
        fprintf(stderr, "FAIL: %s: forced text '%s', want '%s'\n", what, got.c_str(), want.c_str());
    }
}

static void test_forced_text() {
    {
        llama_grammar * g = grammar_init(
                "root ::= \"{\\\"name\\\": \\\"\" name \"\\\", \\\"n\\\": \" [0-9]+ \"}\"\n"
                "name ::= \"read_file\" | \"write_file\" | \"read_dir\"\n");
        check_forced(g, "{\"name\": \"", "the start");
        llama_grammar_accept_str(*g, "{\"name\": \"w");
        check_forced(g, "rite_file\", \"n\": ", "after the name's first letter");
        llama_grammar_accept_str(*g, "rite_file\", \"n\": 1");
        check_forced(g, "", "inside a number");
        llama_grammar_free_impl(g);
    }
    {
        llama_grammar * g = grammar_init(
                "root ::= \"{\\\"name\\\": \\\"\" name \"\\\"}\"\n"
                "name ::= \"read_file\" | \"write_file\" | \"read_dir\"\n");
        llama_grammar_accept_str(*g, "{\"name\": \"r");
        check_forced(g, "ead_", "two names left");
        llama_grammar_accept_str(*g, "ead_d");
        check_forced(g, "ir\"}", "one name left, to the end");
        llama_grammar_free_impl(g);
    }
    {
        llama_grammar * g = grammar_init("root ::= \"ab\" | \"abc\"\n");
        check_forced(g, "ab", "up to where the grammar may end");
        llama_grammar_free_impl(g);
    }
    {
        llama_grammar * g = grammar_init("root ::= \"x\" [a-c] \"yz\"\n");
        check_forced(g, "x", "up to a character class");
        llama_grammar_accept_str(*g, "xb");
        check_forced(g, "yz", "after the class");
        llama_grammar_free_impl(g);
    }
    {
        llama_grammar * g = grammar_init("root ::= \"\xc3\xa9t\xc3\xa9\" [0-9]\n");
        check_forced(g, "\xc3\xa9t\xc3\xa9", "multibyte characters");
        llama_grammar_accept_str(*g, "\xc3");
        check_forced(g, "", "mid-way through a character");
        llama_grammar_free_impl(g);
    }
    {
        llama_grammar * g = grammar_init("root ::= \"<tool_call>\\n<function=\" [a-z]+\n", /* lazy = */ true);
        check_forced(g, "", "a lazy grammar before its trigger");
        llama_grammar_free_impl(g);
    }
}

int main() {
    test_paper_example();
    test_limits();
    test_enumeration();
    test_forced_text();

    printf("%d/%d checks passed\n", n_checks - n_failed, n_checks);
    return n_failed == 0 ? 0 : 1;
}
