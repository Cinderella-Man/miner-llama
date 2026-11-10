#include "server-context.h"
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include "base64.hpp"
#include "server-chat.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-realtime.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "pipe-guess.h"
#include "pipe-store.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "base64.hpp"

#include "../../src/llama-ext.h" // staging API: llama_set_head_bound (Q619)

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <deque>
#include <exception>
#include <memory>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <random>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

// GGML_CUDA_KEEPALIVE=request: each card's keep-alive block starts when a request is taken, ahead of its prompt, so that
// the cards leave their idle state while it is prepared (the CUDA backend's kick does nothing in the other modes)
static void server_keepalive_kick() {
    static void (*kick)(void) = [] {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
        return reg ? (void (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_keepalive_kick") : nullptr;
    }();
    if (kick != nullptr) {
        kick();
    }
}

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (!params.mmproj.path.empty()) {
        const auto mcaps = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        if (mcaps.gen_audio) {
            // some TTS models decode with embeddings output for the whole batch
            return { params.n_batch, params.n_batch };
        }
    }

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// Speculative verification switches (Q797, Q802), read once at the server's start; all off by default, and then
// nothing below changes what the server does:
//   LLAMA_SPEC_BLOCK_VERIFY=1  sampled drafts (--spec-draft-sampling probabilistic at temperature > 0) are verified as
//                              one block, by block verification (common_sampler_sample_and_accept_n_block; Sun et al.,
//                              arXiv:2403.10444) instead of token by token. Greedy requests (sample-and-match) and
//                              requests with backend sampling keep their verification
//   LLAMA_SPEC_FORCE=1         while a grammar applies (a tool call), the tokens of the text it forces after the draft
//                              are appended to the draft, up to the drafter's width (--spec-draft-n-max), also when the
//                              drafter drafted fewer tokens or none. Exact: the verify pass decides them as it does
//                              drafted ones (a rejected guess costs a verify column, no output)
//   LLAMA_SPEC_FORCE_CHARS=<n> the forced text read per step, in characters (default 64)
//   LLAMA_SPEC_FORCE_MAX=<n>   at most n forced tokens per step (default: no limit but the width)
//   LLAMA_SPEC_DRAFT_KEEP=<n>  every draft is cut to its first n drafted tokens (default: no cut), so that forced
//                              tokens get the rest of the width: --spec-draft-n-max 4 with n = 2 verifies 3 tokens
//                              outside tool calls and up to 5 inside. A fixed cut, so exact
//   LLAMA_SPEC_KEEP_DRAWS=1    token-by-token rejection keeps its random draws used across a checkpoint replay, as
//                              block verification always does (Q808): without it the step after a replay (a draft
//                              longer than a recurrent target's rollback snapshots, or a target that cannot remove a
//                              partial sequence) draws the uniforms of the step it replays, conditioned on its outcome
struct server_spec_verify {
    bool    block       = false;
    bool    force       = false;
    int32_t force_chars = 64;
    int32_t force_max   = -1;
    int32_t keep        = -1;
    bool    keep_draws  = false;

    static server_spec_verify & get() {
        static server_spec_verify v;
        return v;
    }

    bool any_draft() const {
        return force || keep >= 0;
    }

    // throws std::invalid_argument on a malformed value
    static void init() {
        auto & v = get();
        v = server_spec_verify();

        auto read_int = [](const char * name, int32_t lo, int32_t & out) {
            const char * s = getenv(name);
            if (s == nullptr || s[0] == 0) {
                return false;
            }
            char * end = nullptr;
            const long x = strtol(s, &end, 10);
            if (end == s || *end != 0 || x < lo || x > 1000000) {
                throw std::invalid_argument(string_format("%s=%s: expected an integer >= %d", name, s, lo));
            }
            out = (int32_t) x;
            return true;
        };

        int32_t on = 0;
        if (read_int("LLAMA_SPEC_BLOCK_VERIFY", 0, on)) {
            v.block = on != 0;
        }
        on = 0;
        if (read_int("LLAMA_SPEC_FORCE", 0, on)) {
            v.force = on != 0;
        }
        read_int("LLAMA_SPEC_FORCE_CHARS", 1, v.force_chars);
        read_int("LLAMA_SPEC_FORCE_MAX",   0, v.force_max);
        read_int("LLAMA_SPEC_DRAFT_KEEP",  0, v.keep);
        on = 0;
        if (read_int("LLAMA_SPEC_KEEP_DRAWS", 0, on)) {
            v.keep_draws = on != 0;
        }

        if (v.block || v.any_draft() || v.keep_draws) {
            LOG_INF("speculative verification: block verification %s, grammar-forced tokens %s (%d characters, at most "
                    "%d a step), drafts cut to %d tokens, draws kept across a replay %s\n", v.block ? "on" : "off",
                    v.force ? "on" : "off", v.force_chars, v.force_max, v.keep, v.block || v.keep_draws ? "on" : "off");
        }
    }
};

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// LLAMA_SPEC_DUMP=<file>: one JSON line per speculative verify step, for offline analysis of token trees and draft
// lengths (bench/tree_replay in the Miner Llama repository). Off by default; when unset nothing below runs.
//   LLAMA_SPEC_DUMP_K=<n>        candidates kept per drafted position (default 8)
//   LLAMA_SPEC_DUMP_TGT_K=<n>    the target's most probable tokens per sampled position (default 8, 0: none)
//   LLAMA_SPEC_DUMP_NGRAM=<n>    the longest context match looked for and the tokens kept after it (default 16, 0: none)
//   LLAMA_SPEC_DUMP_FULL_BLOCK=<n> DFlash2 decodes a second block for the record ("fb": the trained block for 1, a block
//                                of n tokens for n >= 2, at most the trained block), after the draft is taken from the
//                                block the line decodes anyway: drafting, verifying and the output are the same with it
//                                on or off, the step costs one more drafter pass (default 0: off)
// Replayed steps (after a checkpoint restore) and synthetic acceptance are not written.
struct server_spec_dump {
    FILE *   f          = nullptr;
    int32_t  k          = 8;
    int32_t  tgt_k      = 8;
    int32_t  ngram      = 16;
    int32_t  full_block = 0;
    uint64_t n_steps    = 0;
    int64_t  run        = 0; // the server start in microseconds: tells apart runs appended to one file

    static server_spec_dump & get() {
        static server_spec_dump d = [] {
            server_spec_dump r;
            const char * path = getenv("LLAMA_SPEC_DUMP");
            if (path == nullptr || *path == '\0') {
                return r;
            }
            r.f = fopen(path, "a");
            if (r.f == nullptr) {
                LOG_ERR("LLAMA_SPEC_DUMP: cannot open '%s' for appending\n", path);
                return r;
            }
            auto env_int = [](const char * name, int32_t def) {
                const char * v = getenv(name);
                return v && *v ? std::max(0, atoi(v)) : def;
            };
            r.k          = std::max(1, env_int("LLAMA_SPEC_DUMP_K", 8));
            r.tgt_k      = env_int("LLAMA_SPEC_DUMP_TGT_K", 8);
            r.ngram      = env_int("LLAMA_SPEC_DUMP_NGRAM", 16);
            r.full_block = env_int("LLAMA_SPEC_DUMP_FULL_BLOCK", 0);
            r.run        = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            LOG_INF("LLAMA_SPEC_DUMP: verify steps to '%s' (k %d, target k %d, n-gram %d, full block %d)\n",
                    path, r.k, r.tgt_k, r.ngram, (int) r.full_block);
            return r;
        }();
        return d;
    }

    static bool on() { return get().f != nullptr; }
};

static double spec_dump_round(double x) {
    return std::round(x * 1e5) / 1e5;
}

// the k most probable tokens of a logits row at temperature 1, and the probability of `sampled`
static void spec_dump_top(const float * logits, int32_t n_vocab, int32_t k, llama_token sampled,
        json & ids, json & probs, double & p_sampled) {
    float mx = -INFINITY;
    for (int32_t i = 0; i < n_vocab; ++i) {
        mx = std::max(mx, logits[i]);
    }
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp((double) logits[i] - mx);
    }
    // a min-heap of the k largest
    std::vector<std::pair<float, llama_token>> top;
    top.reserve(k + 1);
    auto cmp = [](const std::pair<float, llama_token> & a, const std::pair<float, llama_token> & b) { return a.first > b.first; };
    for (int32_t i = 0; i < n_vocab && k > 0; ++i) {
        if ((int32_t) top.size() < k) {
            top.emplace_back(logits[i], i);
            std::push_heap(top.begin(), top.end(), cmp);
        } else if (logits[i] > top.front().first) {
            std::pop_heap(top.begin(), top.end(), cmp);
            top.back() = { logits[i], i };
            std::push_heap(top.begin(), top.end(), cmp);
        }
    }
    std::sort(top.begin(), top.end(), cmp);
    ids   = json::array();
    probs = json::array();
    for (const auto & t : top) {
        ids.push_back(t.second);
        probs.push_back(spec_dump_round(std::exp((double) t.first - mx) / sum));
    }
    p_sampled = sampled >= 0 && sampled < n_vocab ? std::exp((double) logits[sampled] - mx) / sum : 0.0;
}

// the longest suffix of ctx + [last] (up to n_max tokens) that occurs earlier, its most recent occurrence, and up to
// n_max tokens that followed it there: what a copy (n-gram) drafter would propose
static int32_t spec_dump_ngram(const llama_tokens & ctx, llama_token last, int32_t n_max, llama_tokens & next) {
    next.clear();
    const int32_t N = (int32_t) ctx.size(); // last sits at index N
    auto at = [&](int32_t i) { return i == N ? last : ctx[i]; };
    int32_t best = 0, best_j = -1;
    for (int32_t j = N - 1; j >= 0 && best < n_max; --j) {
        if (ctx[j] != last) {
            continue;
        }
        int32_t len = 1;
        while (len < n_max && j - len >= 0 && ctx[j - len] == at(N - len)) {
            ++len;
        }
        if (len > best) {
            best   = len;
            best_j = j;
        }
    }
    for (int32_t i = best_j + 1; best_j >= 0 && i <= N && (int32_t) next.size() < n_max; ++i) {
        next.push_back(at(i));
    }
    return best;
}

// LLAMA_SPEC_ADAPT=1: the draft length chosen per step from the drafter's own probabilities and a table of verify
// costs (common_spec_adapt_choose in common/speculative.h, whose comment gives the rule; the other LLAMA_SPEC_ADAPT_*
// variables are listed at common_spec_adapt_params_from_env). After each draft the longest prefix that pays for its
// verification is kept, at one of a few verify widths. Speculation stays exact: only the number of drafted tokens
// verified changes.
//   LLAMA_SPEC_ADAPT_LOG=<file>  one JSON line per drafted step, after a header line per server start; it works with
//                                the switch off too (the rule's choice is then recorded, not applied)
// Off by default: with neither variable set nothing below runs and the drafters compute no probabilities.
struct server_spec_adapt {
    common_spec_adapt_params params;
    std::vector<int32_t>     lengths;     // the drafted-token counts allowed
    int32_t                  n_max   = 0;
    bool                     active  = false; // switched on, or logging
    FILE *                   f       = nullptr;
    int64_t                  run     = 0; // the server start in microseconds: tells apart runs appended to one file
    uint64_t                 n_steps = 0;

    static server_spec_adapt & get() {
        static server_spec_adapt a;
        return a;
    }

    static bool on() { return get().active; }

    // at the server's start, once the drafters are known; throws on a malformed variable or an unwritable log
    static void init(int32_t n_max) {
        auto & a = get();
        if (a.f) {
            fclose(a.f);
            a.f = nullptr;
        }
        a.params  = common_spec_adapt_params_from_env();
        a.n_max   = n_max;
        a.lengths = common_spec_adapt_lengths(a.params, n_max);
        a.active  = a.params.enabled || !a.params.log.empty();
        if (!a.active) {
            return;
        }
        a.run = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        std::vector<double> t;
        std::vector<int32_t> widths;
        for (const int32_t d : a.lengths) {
            widths.push_back(d + 1);
        }
        for (int32_t k = 1; k <= std::max(1, n_max) + 1; ++k) {
            t.push_back(common_spec_adapt_cost(a.params, k));
        }
        const char * s_mode = a.params.s_mode == common_spec_adapt_params::S_RUN ? "run" :
                              a.params.s_mode == common_spec_adapt_params::S_STEP ? "step" : "fixed";

        if (!a.params.log.empty()) {
            a.f = fopen(a.params.log.c_str(), "a");
            if (a.f == nullptr) {
                throw std::runtime_error(string_format("cannot open '%s' for appending", a.params.log.c_str()));
            }
            const json hdr = {
                {"hdr",      1},
                {"run",      a.run},
                {"enabled",  a.params.enabled},
                {"n_max",    n_max},
                {"widths",   widths},
                {"cost_ms",  t},
                {"draft_ms", a.params.draft_ms},
                {"s_mode",   s_mode},
                {"s_fixed",  a.params.s_fixed},
                {"s0",       a.params.s0},
                {"ema",      a.params.ema},
                {"scale",    a.params.scale},
            };
            const std::string s = hdr.dump();
            fputs(s.c_str(), a.f);
            fputc('\n', a.f);
            fflush(a.f);
        }

        std::string w;
        for (const int32_t k : widths) {
            w += string_format("%s%d", w.empty() ? "" : ",", k);
        }
        LOG_INF("LLAMA_SPEC_ADAPT: %s; verify widths %s (n_max %d), T(1) %.1f ms, T(%d) %.1f ms, draft %.1f ms, S %s, "
                "scale %.3f%s%s\n", a.params.enabled ? "on" : "off (log only)", w.c_str(), n_max, t.front(),
                (int) t.size(), t.back(), a.params.draft_ms, s_mode, a.params.scale,
                a.f ? ", log " : "", a.f ? a.params.log.c_str() : "");
    }
};

static double spec_adapt_round(double x) {
    return std::round(x * 1e5) / 1e5;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()

    struct token {
        int32_t id_slot;
        llama_token token;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        bool output;
        bool is_prompt; // for stats tracking
        int32_t decision_order = 0;
        int32_t i_embd = -1; // row in embd, -1 if this is a token
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // embedding entries, they can be mixed with tokens if the context supports it
    std::vector<float> embd; // n_embd per row

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, { pos, 0, 0, 0 }, output, is_prompt });
        return true;
    }

    // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
    bool add_embd(int32_t id_slot, const float * embd_in, const llama_pos * pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, { pos[0], pos[1], pos[2], pos[3] }, output, is_prompt });
        tokens.back().i_embd = (int32_t) (embd.size() / n_embd);
        embd.insert(embd.end(), embd_in, embd_in + n_embd);
        return true;
    }

    bool has_embd() const {
        return !embd.empty();
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // remove the entries after the first n
    void truncate(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= size());
        for (int32_t i = n; i < size(); i++) {
            if (tokens[i].i_embd >= 0) {
                embd.resize((size_t) tokens[i].i_embd * n_embd);
                break;
            }
        }
        tokens.resize(n);
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_decision_order(int32_t idx, int32_t order) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].decision_order = order;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (t.i_embd >= 0) {
                view.add_embd({ embd.data() + (size_t) t.i_embd * n_embd, 1, (size_t) n_embd }, t.pos.data(), t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            view.tokens.back().decision_order = t.decision_order;
        }
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    struct tts_ctx {
        mtmd_helper::gen_audio ctx;
        std::vector<float> h_state_prompt; // the first h_state after step_prompt done
        const float * h_state;
        llama_token   sampled;
        int32_t       n_decoded;
        bool          stop;
        bool is_supported() const {
            return ctx.valid();
        }
        void reset() {
            ctx.reset();
            h_state_prompt.clear();
            h_state   = nullptr;
            sampled   = LLAMA_TOKEN_NULL;
            n_decoded = 0;
            stop      = false;
        }
    };
    tts_ctx tts;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;

    // LLAMA_SPEC_FORCE (server_spec_verify): where the grammar's forced tokens begin in spec_draft (-1: none), and
    // this task's steps with forced tokens, forced tokens verified and forced tokens accepted
    int32_t spec_force_from  = -1;
    int32_t spec_force_steps = 0;
    int32_t spec_force_n     = 0;
    int32_t spec_force_acc   = 0;

    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;
    common_speculative_draft_info spec_info; // the drafter's candidates, for LLAMA_SPEC_DUMP
    bool spec_gram = false; // LLAMA_SPEC_DUMP: the sampler's grammar (a tool call's) was active when the draft was made

    // LLAMA_SPEC_ADAPT (see server_spec_adapt): the drafter's probability of each token of spec_draft, the running
    // speed-up (kept across tasks), the drafted step waiting for its accepted length, and the steps of this task by
    // the drafted tokens kept
    std::vector<float> spec_draft_p;
    common_spec_adapt_state spec_adapt_state;
    struct {
        bool    pending = false;
        int32_t n_draft = 0; // drafted
        int32_t n_rule  = 0; // what the rule keeps (the drafted count when the drafter gave no probabilities)
        int32_t n_keep  = 0; // verified
        double  s       = 0.0;
        double  e_tau   = 1.0;
        double  cost_ms = 0.0;
        int64_t pos     = 0;
        std::vector<float> p;
    } spec_adapt_step;
    std::vector<int32_t> spec_adapt_hist;

    // --ctx-checkpoints-spread: where the last prompt batch stopped for a checkpoint inside the prompt, which the next
    // batch takes before it is decoded
    int64_t ckpt_spread_pos = -1;

    // pipelined speculation (--pipe-depth): the slot's tokens in flight in the context's pipe slots (llama_pipe_*),
    // [0] its newest sampled token, then guesses of the tokens after it
    struct pipe_entry {
        int32_t     pslot;
        llama_token tok;
        llama_pos   pos;
        bool        drafted = false; // guessed by the staircase drafter
    };
    std::deque<pipe_entry> pipe_win;
    common_pipe_guesser    pipe_guesser;
    bool    pipe_draws   = false; // every sampled token's random draw is keyed by its position (llama_sampler_set_pos)
    int32_t pipe_n_guess = 0;     // this request's guesses, right guesses and wrong guesses
    int32_t pipe_n_hit   = 0;
    int32_t pipe_n_miss  = 0;
    int32_t pipe_n_draft     = 0; // of them, the staircase drafter's guesses and right ones
    int32_t pipe_n_draft_hit = 0;
    llama_pos pipe_draft_pending = -1; // the position the drafter's pending launch guesses

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        // checkpoints on the devices (--ctx-checkpoints-vram) go into the cache as host copies: the slot reuses its
        // stores; a store's checkpoint is copied before its move, so the slot's own one keeps its store
        server_prompt prompt_host;
        const bool on_dev = std::any_of(prompt.checkpoints.begin(), prompt.checkpoints.end(),
                [](const server_prompt_checkpoint_ptr & ck) { return ck->store >= 0; });
        if (on_dev) {
            prompt_host = prompt.clone(prompt_cache.share_checkpoints);
            for (auto it = prompt_host.checkpoints.begin(); it != prompt_host.checkpoints.end(); ) {
                if ((*it)->store < 0) {
                    ++it;
                    continue;
                }
                auto host = std::make_shared<common_prompt_checkpoint>(**it);
                if (host->tgt_to_host(ctx_tgt, id)) {
                    *it = std::move(host);
                    ++it;
                } else {
                    it = prompt_host.checkpoints.erase(it);
                }
            }
        }

        auto * cur = prompt_cache.alloc(on_dev ? prompt_host : prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        // the state goes into the cache whole, or not at all: its buffers are not zero-filled (#30137)
        const size_t n_tgt =           llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t n_dft = ctx_dft ? llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        if (n_tgt != cur_size_tgt || n_dft != cur_size_dft) {
            SLT_ERR(*this, "%s", "failed to save complete prompt state\n");
            prompt_cache.states.pop_back();
            return false;
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    // for realtime transcription, these are the rows from the last audio stream step (size = n_rows * n_embd)
    std::vector<float> inp_embd;

    // realtime transcription, see SERVER_TASK_TYPE_TRANSCRIBE_RT
    mtmd::audio_stream_ptr audio_stream;
    bool audio_wait = false; // the stream needs more audio before it can produce rows
    std::deque<int64_t> audio_commit_rows; // end row of each non-final commit not yet completed

    // realtime conversation, see SERVER_TASK_TYPE_REALTIME
    std::unique_ptr<server_realtime_slot> rt;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_draft_p.clear();
            spec_adapt_step.pending = false;
            spec_adapt_hist.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        spec_force_from  = -1;
        spec_force_steps = 0;
        spec_force_n     = 0;
        spec_force_acc   = 0;

        pipe_n_guess = 0;
        pipe_n_hit   = 0;
        pipe_n_miss  = 0;
        pipe_n_draft     = 0;
        pipe_n_draft_hit = 0;
        pipe_draft_pending = -1;
        pipe_guesser.n_store = 0;

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();

        tts.reset();

        audio_stream.reset();
        audio_wait = false;
        audio_commit_rows.clear();
        inp_embd.clear();
        rt.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // a decision over a non-causal context reads the whole prompt in one batch
        if (task->type == SERVER_TASK_TYPE_DECISION && !llama_get_causal_attn(ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (task->type == SERVER_TASK_TYPE_DECISION) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        // a joint decision head reads the whole batch
        if (!task->decision.order.empty() || !other_slot.task->decision.order.empty()) {
            return false;
        }

        // lfm3 d1 has no memory, its prompt is read as one flat ubatch and the short convs would mix the sequences
        // TODO: batch them along the sequence dimension, this needs ubatches of sequences with unequal lengths
        if (task->decision.no_batching || other_slot.task->decision.no_batching) {
            return false;
        }

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        // the audio stream has no token before the first step, track the position with a dummy token
        const llama_token tok_track = sampled == LLAMA_TOKEN_NULL ? 0 : sampled;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                const int32_t n_rows = (int32_t) inp_embd.size() / batch.n_embd;
                GGML_ASSERT((int32_t) inp_embd.size() == n_rows * batch.n_embd);
                i_batch = batch.size() + n_rows - 1;
                for (int32_t i = 0; i < n_rows; i++) {
                    // text embeddings broadcast the same position across the M-RoPE sections
                    const llama_pos p = prompt.tokens.pos_next();
                    const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                    add_ok &= batch.add_embd(id, inp_embd.data() + (size_t) i * batch.n_embd, pos, i == n_rows - 1, false);
                    if (i < n_rows - 1) {
                        prompt.tokens.push_back(tok_track);
                    }
                }
                if (audio_stream) {
                    inp_embd.clear(); // rows are consumed, the next step produces new ones
                }
            } else {
                if (pipe_draws) {
                    llama_sampler_set_pos(common_sampler_get(smpl.get()), prompt.tokens.pos_next());
                }
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(tok_track);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            // pipelined tokens still in flight: the newest sampled one stays (its KV cells are being written), the
            // guesses behind it go
            if (!pipe_win.empty()) {
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), id, pipe_win.front().pos + 1, -1);
                llama_pipe_drain(ctx_tgt);
                pipe_win.clear();
            }

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless() || audio_stream || task->type == SERVER_TASK_TYPE_REALTIME) {
                prompt_clear();
            }

            // mtmd_helper always re-eval the whole prompt
            if (task->type == SERVER_TASK_TYPE_TTS) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    // the sampler's counters into stats, for requests with a grammar
    void update_sampling_stats() {
        if (smpl && !task->params.sampling.grammar.empty()) {
            const auto & s = common_sampler_get_stats(smpl.get());
            stats.smpl_grammar     = true;
            stats.smpl_device      = s.n_device;
            stats.smpl_constrained = s.n_constrained;
            stats.smpl_rejected    = s.n_rejected;
        }
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (pipe_n_guess > 0 || pipe_draws) {
            SLT_INF(*this,
                    "       pipelined = %10d guesses, %d right, %d wrong, %d from the store; drafter %d guesses, %d right\n",
                    pipe_n_guess, pipe_n_hit, pipe_n_miss, pipe_guesser.n_store, pipe_n_draft, pipe_n_draft_hit);
        }

        if (smpl && (common_sampler_get_stats(smpl.get()).n_head_short + common_sampler_get_stats(smpl.get()).n_head_full) > 0) {
            const auto & s = common_sampler_get_stats(smpl.get());
            SLT_INF(*this,
                    "  shortlist head = %10d tokens from the shortlist's logits, %d from the full head's (the bound could "
                    "not exclude every token left out)\n", s.n_head_short, s.n_head_full);
        }

        if (smpl && !task->params.sampling.grammar.empty()) {
            const auto & s = common_sampler_get_stats(smpl.get());
            SLT_INF(*this,
                    "        sampling = %10d tokens, %d drawn on the device, %d while the grammar applied, %d rejected "
                    "(%d after a device draw), %d forced\n",
                    s.n_sampled, s.n_device, s.n_constrained, s.n_rejected, s.n_rejected_device, s.n_forced);
        }

        if (spec_force_steps > 0) {
            SLT_INF(*this,
                    "   forced tokens = %10d steps, %d tokens the grammar forced appended to drafts, %d accepted\n",
                    spec_force_steps, spec_force_n, spec_force_acc);
        }

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        if (server_spec_adapt::on() && !spec_adapt_hist.empty()) {
            const auto & a = server_spec_adapt::get();
            std::string widths;
            int64_t n_steps = 0;
            int64_t n_kept  = 0;
            for (size_t d = 0; d < spec_adapt_hist.size(); ++d) {
                if (spec_adapt_hist[d] > 0) {
                    widths  += string_format("%s%zu: %d", widths.empty() ? "" : ", ", d + 1, spec_adapt_hist[d]);
                    n_steps += spec_adapt_hist[d];
                    n_kept  += (int64_t) d * spec_adapt_hist[d];
                }
            }
            SLT_INF(*this, "adaptive draft (%s) = %" PRId64 " drafted steps, %.2f drafted tokens verified per step, "
                    "steps by verify width (%s), running S %.3f\n", a.params.enabled ? "on" : "log only", n_steps,
                    n_steps > 0 ? (double) n_kept / n_steps : 0.0, widths.c_str(), spec_adapt_state.s(a.params));
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    // the other slot continues from the tokens processed so far; share: it shares the host checkpoints
    // (--ctx-checkpoints-share), else it gets copies
    void copy_prompt_to(server_slot & other, bool share) const {
        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.prompt = prompt.clone(share);
        drop_device_checkpoints(other.prompt);
    }

    void copy_state_to(server_slot & other, bool share) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone(share);
        drop_device_checkpoints(other.prompt);
        other.init_sampler();
    }

    // a checkpoint on the devices (--ctx-checkpoints-vram) lives in a store of this slot's sequence, which this slot
    // reuses: a copy for another slot leaves it out
    static void drop_device_checkpoints(server_prompt & p) {
        p.checkpoints.remove_if([](const server_prompt_checkpoint_ptr & ck) { return ck->store >= 0; });
    }
};

// one verify step to LLAMA_SPEC_DUMP's file: where it starts, what the drafter proposed, what the target sampled
// (accepted: the target's tokens, the accepted drafts and then its own), and what a copy drafter would have proposed
static void spec_dump_step(const server_slot & slot, llama_context * ctx_tgt, const std::vector<int32_t> & spec_i_batch,
        const llama_tokens & accepted, bool rejection) {
    auto & d = server_spec_dump::get();

    const size_t n_draft = slot.spec_draft.size();
    const auto & info    = slot.spec_info;

    json line = {
        {"v",        1},
        {"run",      d.run},
        {"step",     d.n_steps++},
        {"task",     slot.task->id},
        {"slot",     slot.id},
        {"pos",      (int64_t) slot.prompt.tokens.pos_next() - (int64_t) n_draft - 1}, // the position of "last"
        {"n_prompt", slot.task->n_tokens()},
        {"temp",     spec_dump_round(slot.task->params.sampling.temp)},
        {"verify",   rejection ? "reject" : "match"},
        {"type",     info.type},
        {"last",     slot.sampled},
        {"draft",    slot.spec_draft},
        {"sampled",  accepted},
        {"n_acc",    accepted.size() - 1},
        {"gram",     slot.spec_gram ? 1 : 0},
    };

    if (!info.ids.empty()) {
        json p = json::array();
        for (const auto & row : info.p) {
            json r = json::array();
            for (const float x : row) {
                r.push_back(spec_dump_round(x));
            }
            p.push_back(std::move(r));
        }
        json cand = json::array();
        for (const auto & row : info.ids) {
            cand.push_back(row);
        }
        line["lattice"] = info.lattice;
        line["k"]       = info.k_max;
        line["cand"]    = std::move(cand);
        line["p"]       = std::move(p);
        line["pick"]    = info.pick;
        if (info.lattice) {
            json kept = json::array();
            for (const float x : info.kept) {
                kept.push_back(spec_dump_round(x));
            }
            line["kept"] = std::move(kept);
        }
    }

    // the second block of LLAMA_SPEC_DUMP_FULL_BLOCK: its lattice in the same form, along the selector's greedy walk
    if (info.fb_width > 0 && !info.fb_ids.empty()) {
        json p = json::array();
        for (const auto & row : info.fb_p) {
            json r = json::array();
            for (const float x : row) {
                r.push_back(spec_dump_round(x));
            }
            p.push_back(std::move(r));
        }
        json cand = json::array();
        for (const auto & row : info.fb_ids) {
            cand.push_back(row);
        }
        json kept = json::array();
        for (const float x : info.fb_kept) {
            kept.push_back(spec_dump_round(x));
        }
        line["fb"] = {
            {"w",    info.fb_width},
            {"cand", std::move(cand)},
            {"p",    std::move(p)},
            {"pick", info.fb_pick},
            {"kept", std::move(kept)},
        };
    }

    if (d.tgt_k > 0) {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_tgt)));
        json t_ids = json::array(), t_p = json::array(), t_ps = json::array();
        for (size_t i = 0; i < accepted.size() && i < spec_i_batch.size(); ++i) {
            const float * logits = llama_get_logits_ith_fetch(ctx_tgt, spec_i_batch[i]);
            if (logits == nullptr) {
                break;
            }
            json ids, probs;
            double ps = 0.0;
            spec_dump_top(logits, n_vocab, d.tgt_k, accepted[i], ids, probs, ps);
            t_ids.push_back(std::move(ids));
            t_p.push_back(std::move(probs));
            t_ps.push_back(spec_dump_round(ps));
        }
        line["tgt_ids"] = std::move(t_ids);
        line["tgt_p"]   = std::move(t_p);
        line["tgt_ps"]  = std::move(t_ps);
    }

    if (d.ngram > 0) {
        llama_tokens next;
        line["ng_len"] = spec_dump_ngram(slot.spec_prompt, slot.sampled, d.ngram, next);
        line["ng"]     = next;
    }

    const std::string s = line.dump(); // numbers and fixed names only
    fputs(s.c_str(), d.f);
    fputc('\n', d.f);
    fflush(d.f);
}

// LLAMA_SPEC_ADAPT: a drafted step is done, with n_acc of its verified drafted tokens accepted: the running speed-up,
// this task's counts and the log line
static void spec_adapt_finish(server_slot & slot, int32_t n_acc) {
    auto & a  = server_spec_adapt::get();
    auto & st = slot.spec_adapt_step;
    if (!st.pending) {
        return;
    }
    st.pending = false;

    slot.spec_adapt_state.update(a.params, st.n_keep, n_acc);
    if ((int32_t) slot.spec_adapt_hist.size() <= st.n_keep) {
        slot.spec_adapt_hist.resize(st.n_keep + 1, 0);
    }
    slot.spec_adapt_hist[st.n_keep]++;

    if (a.f == nullptr) {
        return;
    }
    const json line = {
        {"run",     a.run},
        {"step",    a.n_steps++},
        {"task",    slot.task ? slot.task->id : -1},
        {"slot",    slot.id},
        {"pos",     st.pos},     // the position of the token the draft continues
        {"on",      a.params.enabled},
        {"n_draft", st.n_draft}, // drafted
        {"n_rule",  st.n_rule},  // what the rule keeps
        {"n_keep",  st.n_keep},  // verified: n_rule when on, n_draft when off
        {"p",       st.p},       // the drafter's probability of each drafted token given the drafted prefix
        {"s",       st.s},       // the S the rule used (its last test)
        {"e_tau",   st.e_tau},   // E(n_rule)
        {"cost_ms", st.cost_ms}, // C(n_rule)
        {"n_acc",   n_acc},      // accepted of the n_keep verified
        {"s_next",  slot.spec_adapt_state.s(a.params)},
    };
    const std::string s = line.dump(); // numbers and fixed names only
    fputs(s.c_str(), a.f);
    fputc('\n', a.f);
    fflush(a.f);
}

// LLAMA_SPEC_ADAPT: what the rule keeps of a new draft; with the switch on the draft (and its candidates for
// rejection sampling) is cut to it. A draft cut to nothing ends its step here, as a one-token step.
static void spec_adapt_choose(server_slot & slot) {
    const auto & a  = server_spec_adapt::get();
    auto       & st = slot.spec_adapt_step;

    st.pending = false;

    const int32_t n_draft = (int32_t) slot.spec_draft.size();
    if (n_draft == 0) {
        return;
    }

    st.pending = true;
    st.n_draft = n_draft;
    st.pos     = slot.prompt.tokens.pos_next();
    st.p       = slot.spec_draft_p;

    const double s_run = slot.spec_adapt_state.s(a.params);
    if ((int32_t) slot.spec_draft_p.size() == n_draft) {
        const auto ch = common_spec_adapt_choose(a.params, a.lengths, slot.spec_draft_p, s_run);
        st.n_rule  = ch.n_keep;
        st.s       = ch.s;
        st.e_tau   = ch.e_tau;
        st.cost_ms = ch.cost_ms;
    } else {
        // a drafter without probabilities (the n-gram ones): its draft stays as it is
        st.n_rule  = n_draft;
        st.s       = s_run;
        st.e_tau   = 0.0;
        st.cost_ms = 0.0;
    }
    st.n_keep = a.params.enabled ? st.n_rule : n_draft;

    if (st.n_keep < n_draft) {
        slot.spec_draft.resize(st.n_keep);
        if (!slot.spec_draft_q.empty()) {
            slot.spec_draft_q.resize(st.n_keep);
        }
    }

    if (st.n_keep == 0) {
        spec_adapt_finish(slot, 0);
    }
}

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int encode_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx) {
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);

    mbatch.reset(mtmd_batch_init(mctx));
    int32_t res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return 0;
}

// LLAMA_SPEC_DRAFT_KEEP and LLAMA_SPEC_FORCE (server_spec_verify) on a new draft: cut it, then append the tokens of
// the text the grammar forces after it, up to n_width drafted tokens in all. Returns the change in the draft's length.
//
// Exact under every verification: whether and what is appended depends only on the text before the appended position
// (the history and the drafted tokens kept), so each appended token is a proposal with all its probability on one
// token (its entry in spec_draft_q, when the drafter filled them). Nothing is appended after a drafted token the grammar
// rejects: that token is certain to be rejected, and the draft is left as it was.
//
// The forced text is tokenized as plain text would be. Its last token is held back unless the text ends at a newline
// or with a token that is not plain text (a special token): the text after it may join it into a longer token.
static int32_t spec_force_extend(server_slot & slot, int32_t n_width) {
    const auto & v = server_spec_verify::get();

    auto & draft = slot.spec_draft;
    auto & dq    = slot.spec_draft_q;

    const int32_t n0 = (int32_t) draft.size();

    slot.spec_force_from = -1;

    if (v.keep >= 0 && (int32_t) draft.size() > v.keep) {
        draft.resize(v.keep);
        if (!dq.empty()) {
            dq.resize(v.keep);
        }
    }

    // only while the grammar applies after the history: before a lazy grammar's trigger (most of the output) the
    // per-step copy of the grammar is not paid; a trigger inside the draft is caught on the next step
    if (!v.force || !slot.smpl || !common_sampler_grammar_active(slot.smpl.get()) || (int32_t) draft.size() >= n_width) {
        return (int32_t) draft.size() - n0;
    }

    size_t n_valid = 0;
    const std::string text = common_sampler_grammar_forced_text(slot.smpl.get(), draft, v.force_chars, &n_valid);
    if (text.empty() || n_valid != draft.size()) {
        return (int32_t) draft.size() - n0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(slot.ctx_tgt));

    llama_tokens toks = common_tokenize(vocab, text, false, true);

    // the pieces must spell the text, or the tokenizer normalized it and nothing is appended
    {
        std::string spelled;
        for (const llama_token t : toks) {
            spelled += common_token_to_piece(slot.ctx_tgt, t, true);
        }
        if (spelled != text) {
            return (int32_t) draft.size() - n0;
        }
    }

    if (!toks.empty() && text.back() != '\n' && (llama_vocab_get_attr(vocab, toks.back()) & LLAMA_TOKEN_ATTR_NORMAL)) {
        toks.pop_back();
    }

    int32_t n_add = std::min<int32_t>((int32_t) toks.size(), n_width - (int32_t) draft.size());
    if (v.force_max >= 0) {
        n_add = std::min(n_add, v.force_max);
    }
    if (n_add <= 0) {
        return (int32_t) draft.size() - n0;
    }

    // a draft verified by rejection (a drafter that gave its candidates, or forced tokens alone on a sampled request)
    // carries the forced tokens' proposals, point masses; a drafter without candidates (the n-gram ones) is matched
    const bool with_q = dq.size() == draft.size() && (!dq.empty() || slot.use_spec_rejection());

    slot.spec_force_from = (int32_t) draft.size();
    for (int32_t i = 0; i < n_add; ++i) {
        draft.push_back(toks[i]);
        if (with_q) {
            dq.push_back({ llama_token_data{ toks[i], 0.0f, 1.0f } });
        }
    }

    slot.spec_force_steps += 1;
    slot.spec_force_n     += n_add;

    return (int32_t) draft.size() - n0;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (encode_mtmd_chunk(slot, mbatch, idx) != 0) {
        return -1;
    }

    return try_decode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool add_mtmd_chunk(const server_slot & slot, const mtmd_input_chunk * chunk, const float * embd, server_batch & batch) {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const auto * image    = mtmd_input_chunk_get_tokens_image(chunk);
    const bool   is_mrope = mtmd_decode_use_mrope(slot.mctx);
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd   = batch.n_embd;
    const llama_pos pos_0 = slot.prompt.tokens.pos_next();

    for (size_t i = 0; i < n_tokens; i++) {
        const llama_pos p = pos_0 + (llama_pos) i;
        llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
        if (is_mrope && image) {
            const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = rel.t;
            pos[1] = rel.y;
            pos[2] = rel.x;
            pos[3] = rel.z;
        }
        if (!batch.add_embd(slot.id, embd + i * n_embd, pos, slot.need_embd(), /* is_prompt */ true)) {
            return false;
        }
    }
    return true;
}

//
// --prefix-cache: saved states of shared prompt prefixes
//
// One file per prefix, written by llama_state_seq_save_file (magic, version, token count, the tokens, then the sequence
// state: every layer's cells of the prefix), found by its tokens. A prefix is saved when a slot processes it from the
// first token and the batch stops at its end, so a restore gives exactly the cells a cold server computes for it.
//

struct server_prefix_store {
    struct entry {
        std::string  path;
        llama_tokens tokens;
    };

    std::string dir;
    std::string fp;      // the model and the settings that shape its cells: a file of another one is never used
    int32_t     n_min   = 0;
    int32_t     n_files = 0;

    std::vector<entry> entries; // the least recently used first

    bool enabled() const {
        return !dir.empty();
    }

    static uint64_t hash_bytes(uint64_t h, const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        for (size_t i = 0; i < n; i++) {
            h ^= p[i]; // FNV-1a
            h *= 0x100000001b3ULL;
        }
        return h;
    }

    static uint64_t hash(const llama_tokens & tokens) {
        return hash_bytes(0xcbf29ce484222325ULL, tokens.data(), tokens.size()*sizeof(llama_token));
    }

    // the model file (description, size, parameters) and the context settings that change the cells' values: the batch
    // and micro-batch sizes (where a prompt's evaluations split), flash attention and the cache types
    static std::string fingerprint(const llama_model * model, const llama_context * ctx, const common_params & params) {
        char desc[256] = { 0 };
        llama_model_desc(model, desc, sizeof(desc));
        const std::string s = string_format("%s|%" PRIu64 "|%" PRIu64 "|%u|%u|%d|%d|%d", desc, llama_model_size(model),
                llama_model_n_params(model), llama_n_batch(ctx), llama_n_ubatch(ctx), (int) params.flash_attn_type,
                (int) params.cache_type_k, (int) params.cache_type_v);
        return string_format("%016" PRIx64, hash_bytes(0xcbf29ce484222325ULL, s.data(), s.size()));
    }

    std::string path_of(const llama_tokens & tokens) const {
        return dir + string_format("%s-%016" PRIx64 "-%zu.prefix", fp.c_str(), hash(tokens), tokens.size());
    }

    // the token list of a state file, empty if it is not one
    static llama_tokens read_tokens(const std::string & path) {
        std::ifstream ifs(std::filesystem::u8path(path), std::ios::binary);
        uint32_t hdr[3] = { 0, 0, 0 };
        if (!ifs.read((char *) hdr, sizeof(hdr)) || hdr[0] != LLAMA_STATE_SEQ_MAGIC || hdr[1] != LLAMA_STATE_SEQ_VERSION ||
            hdr[2] == 0 || hdr[2] > (1u << 24)) {
            return {};
        }
        llama_tokens tokens(hdr[2]);
        if (!ifs.read((char *) tokens.data(), tokens.size()*sizeof(llama_token))) {
            return {};
        }
        return tokens;
    }

    void init(const std::string & path, const std::string & fingerprint, int32_t min_tokens, int32_t max_files) {
        dir     = path;
        fp      = fingerprint;
        n_min   = min_tokens;
        n_files = std::max(1, max_files);
        entries.clear();

        std::vector<std::pair<std::filesystem::file_time_type, entry>> found;
        std::error_code ec;
        for (const auto & de : std::filesystem::directory_iterator(std::filesystem::u8path(dir), ec)) {
            if (!de.is_regular_file() || de.path().extension() != ".prefix" ||
                de.path().filename().string().rfind(fp + "-", 0) != 0) {
                continue;
            }
            entry e;
            e.path   = de.path().string();
            e.tokens = read_tokens(e.path);
            if (e.tokens.empty()) {
                SRV_WRN("prefix cache: '%s' is not a state file - ignored\n", e.path.c_str());
                continue;
            }
            found.emplace_back(de.last_write_time(ec), std::move(e));
        }
        std::sort(found.begin(), found.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        for (auto & f : found) {
            entries.push_back(std::move(f.second));
        }
        trim();
    }

    // the longest saved prefix that the prompt starts with and continues after
    const entry * find(const server_tokens & prompt) const {
        const entry * best = nullptr;
        for (const auto & e : entries) {
            if (e.tokens.size() >= prompt.size() || (best && best->tokens.size() >= e.tokens.size())) {
                continue;
            }
            bool match = true;
            for (size_t i = 0; i < e.tokens.size() && match; i++) {
                match = prompt[i] == e.tokens[i];
            }
            if (match) {
                best = &e;
            }
        }
        return best;
    }

    bool has(const llama_tokens & tokens) const {
        for (const auto & e : entries) {
            if (e.tokens == tokens) {
                return true;
            }
        }
        return false;
    }

    // the entry becomes the most recently used, in memory and by its file's time
    void touch(const std::string & path) {
        for (size_t i = 0; i < entries.size(); i++) {
            if (entries[i].path == path) {
                entry e = std::move(entries[i]);
                entries.erase(entries.begin() + i);
                entries.push_back(std::move(e));
                std::error_code ec;
                std::filesystem::last_write_time(std::filesystem::u8path(path), std::filesystem::file_time_type::clock::now(), ec);
                return;
            }
        }
    }

    void add(entry e) {
        entries.push_back(std::move(e));
        trim();
    }

    void remove(const std::string & path) {
        for (size_t i = 0; i < entries.size(); i++) {
            if (entries[i].path == path) {
                entries.erase(entries.begin() + i);
                break;
            }
        }
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(path), ec);
    }

    void trim() {
        while ((int32_t) entries.size() > n_files) {
            SRV_INF("prefix cache: deleting the least recently used '%s' (%zu tokens)\n", entries.front().path.c_str(), entries.front().tokens.size());
            std::error_code ec;
            std::filesystem::remove(std::filesystem::u8path(entries.front().path), ec);
            entries.erase(entries.begin());
        }
    }
};

// LLAMA_SERVER_TEXT_REUSE=1: keep the cached ids where a new prompt's text continues the cached tokens' text.
// A model sampling at temperature > 0 sometimes writes a word in other ids than its tokenizer would (Flint: "file" +
// "type" where the tokenizer writes "filetype"; 6 % of agent turns, 45 % of those over 4k tokens). The client sends
// the reply back as text, the template renders it again and the tokenizer splits it its own way, so the token match
// with the cache stops inside the reply; with sliding-window layers the server then restores the checkpoint at the
// previous prompt's end and processes the whole reply again. Off by default.
static bool server_text_reuse_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_SERVER_TEXT_REUSE");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

// LLAMA_SERVER_TAIL_CKPT_MIN=<n>: take the end-of-prompt checkpoint only when the prompt ends at least n tokens past
// the slot's newest checkpoint; checkpoints at user messages are unchanged. Each tail checkpoint waits for the pipeline
// to drain, copies the sliding-window ring to the host (Flint, nine P106 cards: 52.5 MiB, ~0.3 s) and processes the
// prompt's last 4 tokens as a micro-batch of their own. With LLAMA_SERVER_TEXT_REUSE a reply sent back unchanged needs
// no checkpoint; one that comes back changed re-reads up to n tokens more. 0 (the default): every prompt.
static uint32_t server_tail_ckpt_min() {
    static const uint32_t n_min = [] {
        const char * env = getenv("LLAMA_SERVER_TAIL_CKPT_MIN");
        return env ? (uint32_t) std::max(0, atoi(env)) : 0u;
    }();
    return n_min;
}

// After the first token that differs (n), the longest stretch over which the cached tokens and the prompt spell the
// same bytes, cut at the last byte offset where a token ends in both: the prompt's tokens up to there are replaced by
// the cached ones, so the prompt's text is unchanged. Returns the number of cached tokens taken past n.
static size_t server_reuse_cached_ids(const llama_vocab * vocab, const llama_tokens & cache, llama_tokens & prompt, size_t & n) {
    n = 0;
    while (n < cache.size() && n < prompt.size() && cache[n] == prompt[n]) {
        n++;
    }
    if (n == cache.size() || n == prompt.size()) {
        return 0;
    }

    // the bytes after n and the offsets at which each token ends; the prompt only as far as the cache reaches
    std::string tc, tp;
    std::vector<size_t> ec, ep;
    for (size_t i = n; i < cache.size(); i++) {
        tc += common_token_to_piece(vocab, cache[i], true);
        ec.push_back(tc.size());
    }
    for (size_t i = n; i < prompt.size() && tp.size() < tc.size(); i++) {
        tp += common_token_to_piece(vocab, prompt[i], true);
        ep.push_back(tp.size());
    }
    size_t n_same = 0;
    while (n_same < tc.size() && n_same < tp.size() && tc[n_same] == tp[n_same]) {
        n_same++;
    }

    size_t kc = 0, kp = 0;
    for (size_t ic = 0, ip = 0; ic < ec.size() && ip < ep.size(); ) {
        if (std::min(ec[ic], ep[ip]) > n_same) {
            break;
        }
        if (ec[ic] == ep[ip]) {
            kc = ++ic;
            kp = ++ip;
        } else if (ec[ic] < ep[ip]) {
            ic++;
        } else {
            ip++;
        }
    }
    if (kc == 0) {
        return 0;
    }

    llama_tokens out(prompt.begin(), prompt.begin() + n);
    out.insert(out.end(), cache.begin() + n, cache.begin() + n + kc);
    out.insert(out.end(), prompt.begin() + n + kp, prompt.end());
    prompt = std::move(out);
    return kc;
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    bool head_bound_on = false; // the shortlist head with a bound is set up on ctx_tgt (LLAMA_HEAD_BOUND)
    uint32_t screen_head_m = 0; // > 0: the screened output head may be used (LLAMA_SCREEN_HEAD=1 and the model has one)
    int screen_head_state  = -1; // last decision (-1 = none yet), logged when it changes

    mtmd_context * mctx = nullptr;
    mtmd_helper_cap mtmd_cap = {};
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    // prompt of the "asr" chat template, for live ASR
    llama_tokens asr_prompt;

    bool has_cap_tts() const {
        return mtmd_cap.audio_gen;
    }

    bool has_cap_chat() const {
        return mtmd_cap.chat;
    }

    server_decision_context decision;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    int32_t pipe_next = 0; // the next pipe slot to try (--pipe-depth)

    // --pipe-guess staircase: the drafter is loaded, and the slot whose sequence it follows (-1: none). With
    // LLAMA_PIPE_DRAFT_TAPS_ONLY=1 it follows the sequence (the hidden-state taps of every token, the prompt's final
    // states) but never guesses, and the guesses are cont's: a measurement of what following costs
    bool         pipe_draft_on  = false;
    bool         pipe_draft_taps_only = false;
    llama_seq_id pipe_draft_seq = -1;

    // the continuation store of --pipe-store, shared by the slots' guessers
    std::unique_ptr<common_pipe_store, void (*)(common_pipe_store *)> pipe_store { nullptr, common_pipe_store_free };

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    // agent sessions (Q111): saved shared prefixes, generation checkpoints, and whether a ring checkpoint taken after
    // the divergence still serves it (a sliding-window model whose cells can be removed from the end: the restore
    // removes the cells past the divergence, and the ring holds a complete window before it)
    server_prefix_store      prefix_store;
    std::vector<llama_token> ckpt_gen_tokens;
    bool                     ckpt_trim = false;
    // each part of a sliding-window model's ckpt_trim checkpoints on unless its switch is 0 (then stock's rule):
    // LLAMA_SERVER_CKPT_RING (the whole ring saved), LLAMA_SERVER_CKPT_RING_PAST (a ring checkpoint past the
    // divergence restored), LLAMA_SERVER_CKPT_RESUME_NEXT (resume after pos_max, not at it)
    bool                     ckpt_ring        = true;
    bool                     ckpt_ring_past   = true;
    bool                     ckpt_resume_next = true;

    // --ctx-checkpoints-vram (C22): a slot's newest ckpt_vram_n checkpoints live in device stores of its sequence, with
    // ids 0 .. ckpt_vram_stores - 1 (one more than ckpt_vram_n while older ones move to the host: the moving one keeps
    // its store until its host copy is done, at most one per slot)
    struct ckpt_spill_job {
        int32_t store = -1;
        std::future<common_state_data> data;
    };
    int32_t ckpt_vram_n      = 0;
    int32_t ckpt_vram_stores = 0;
    std::map<int, ckpt_spill_job> ckpt_spills; // by slot id

    // --ctx-checkpoints-incall (C217): checkpoints taken inside a prompt's decode call. While a batch is built and run,
    // ckpt_incall holds its checkpoints until the call has filled their stores, incall_cuts the batch rows where the
    // calls it joins would have ended, and ckpt_incall_slot the slot whose batch it is (no other slot joins it)
    struct ckpt_incall_job {
        std::shared_ptr<common_prompt_checkpoint> ck;
        int     slot_id;
        int32_t i_batch;
        bool    done = false;
    };
    bool    ckpt_incall_on   = false;
    int     ckpt_incall_slot = -1;
    std::vector<ckpt_incall_job> ckpt_incall;
    std::vector<int32_t>         incall_cuts;

    // Q771: the target context's micro-batch taper: the server's (--ubatch-taper, else LLAMA_UBATCH_TAPER) and the one
    // set now (a request's "ubatch_taper" sets it for the batches of its prompt)
    std::string taper_default = "off";
    std::string taper_cur     = "off";

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        // the background copies of checkpoints to the host read the context's stores: let them finish first
        ckpt_spills.clear();

        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        // on resume params_base already holds the margin from the first load
        if (has_mmproj && params_base.fit_params && !is_resume) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        // screened output head: asked for with LLAMA_SCREEN_HEAD=1, used for the batches whose slots all sample
        // in a way it is exact for (decode())
        screen_head_m = common_screen_head_requested() && model_tgt ? llama_model_screen_head_m(model_tgt) : 0;
        if (screen_head_m > 0) {
            SRV_INF("screened output head: %u candidates, used for requests whose sampling it is exact for\n", screen_head_m);
        }

        // the shortlist head with a bound: set up by common_init_from_params (LLAMA_HEAD_BOUND), turned on and off
        // per batch (decode())
        head_bound_on = common_head_bound_requested() && ctx_tgt && llama_head_shortlist_size(ctx_tgt) > 0;

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        try {
            decision.init(model_tgt);
        } catch (const std::exception & e) {
            SRV_ERR("failed to init decision model: %s\n", e.what());
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                const bool is_mixed = use_mixed_batch();
                if (is_mixed || mmproj_usage.use_non_causal) {
                    const int n_max = is_mixed ? n_ubatch / 2 : n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        SRV_WRN("cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", mmproj_usage.image_max_tokens, n_max, n_ubatch,
                                is_mixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
                        SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        mtmd_cap = mtmd_helper_get_capabilities(model_tgt, mctx);

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        // LLAMA_SPEC_BLOCK_VERIFY, LLAMA_SPEC_FORCE*, LLAMA_SPEC_DRAFT_KEEP: read once here, so that a malformed value
        // stops the server at its start
        try {
            server_spec_verify::init();
        } catch (const std::exception & e) {
            SRV_ERR("%s\n", e.what());
            return false;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        // LLAMA_SPEC_ADAPT: read once here, so that a malformed value stops the server at its start
        if (spec) {
            try {
                server_spec_adapt::init(common_speculative_n_max(spec.get()));
            } catch (const std::exception & e) {
                SRV_ERR("LLAMA_SPEC_ADAPT: %s\n", e.what());
                return false;
            }
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            if (has_cap_tts()) {
                slot.tts.ctx.init(ctx_tgt, mctx);
            }

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        // pipelined speculation: pipe slots for the newest token and up to pipe_depth guesses, with room to refill
        if (params_base.pipe_depth > 0) {
            const int32_t n_pipe = 2*(params_base.pipe_depth + 1);
            if (params_base.sampling.backend_sampling && params_base.pipe_guess == "staircase") {
                llama_pipe_draft_params dp = llama_pipe_draft_default_params();
                dp.path   = params_base.pipe_draft.c_str();
                dp.device = params_base.pipe_draft_dev;
                dp.n_ctx  = params_base.pipe_draft_ctx;
                dp.n_rows = std::max(8, ((params_base.pipe_depth + 2 + 7)/8)*8);
                dp.alpha  = params_base.pipe_draft_alpha;
                dp.stair  = params_base.pipe_draft_stair.empty() ? nullptr : params_base.pipe_draft_stair.c_str();
                dp.kind   = params_base.pipe_draft_kind.empty() ? nullptr : params_base.pipe_draft_kind.c_str();
                if (params_base.pipe_draft.empty() || llama_pipe_draft_init(ctx_tgt, dp) != 0) {
                    SRV_WRN("%s", "--pipe-guess staircase needs a drafter (--pipe-draft): guessing with cont\n");
                    params_base.pipe_guess = "cont";
                } else {
                    pipe_draft_on = true;
                    const char * env = getenv("LLAMA_PIPE_DRAFT_TAPS_ONLY");
                    pipe_draft_taps_only = env && atoi(env) != 0;
                    if (pipe_draft_taps_only) {
                        SRV_WRN("%s", "LLAMA_PIPE_DRAFT_TAPS_ONLY: the drafter follows the sequence and never guesses; "
                                "guessing with cont\n");
                    }
                }
            }
            if (!params_base.sampling.backend_sampling) {
                SRV_WRN("%s", "--pipe-depth needs backend sampling (-bs): pipelined speculation off\n");
                params_base.pipe_depth = 0;
            } else if (llama_pipe_init(ctx_tgt, n_pipe) != 0) {
                SRV_WRN("%s", "failed to set up the pipe slots: pipelined speculation off\n");
                params_base.pipe_depth = 0;
            } else {
                SRV_INF("pipelined speculation: up to %d guesses (%s, copy >= %d, floor %d), %d pipe slots\n",
                        params_base.pipe_depth, params_base.pipe_guess.c_str(), params_base.pipe_copy_min,
                        params_base.pipe_floor, n_pipe);
                pipe_store.reset();
                if (!params_base.pipe_store.empty()) {
                    pipe_store.reset(common_pipe_store_load(params_base.pipe_store));
                    if (pipe_store) {
                        SRV_INF("continuation store %s: %lld tokens\n", params_base.pipe_store.c_str(),
                                (long long) common_pipe_store_size(pipe_store.get()));
                    } else {
                        SRV_WRN("cannot read the continuation store %s: guessing without it\n",
                                params_base.pipe_store.c_str());
                    }
                }
                for (auto & slot : slots) {
                    slot.pipe_guesser.store = pipe_store.get();
                    slot.pipe_draws = true;
                    // staircase: the copy lookup first (with a store also the request's continuations of 2-token or
                    // longer contexts and the store), the drafter for the rest. LLAMA_PIPE_STAIR_CONT=1: the copy
                    // lookup, the request's continuations of 2-token or longer contexts (on agent text they are right
                    // more often than the drafter), the store if there is one, then the drafter instead of the
                    // continuations of a 1-token context
                    static const bool stair_cont = [] {
                        const char * env = getenv("LLAMA_PIPE_STAIR_CONT");
                        return env != nullptr && atoi(env) != 0;
                    }();
                    const bool stair = params_base.pipe_guess == "staircase" && !pipe_draft_taps_only;
                    slot.pipe_guesser.kind = params_base.pipe_guess == "none" ? common_pipe_guesser::NONE :
                                             params_base.pipe_guess == "copy" ? common_pipe_guesser::COPY :
                                             stair && !pipe_store && !stair_cont ? common_pipe_guesser::COPY :
                                                                                   common_pipe_guesser::CONT;
                    slot.pipe_guesser.cont_any = !stair;
                    slot.pipe_guesser.min_match = params_base.pipe_copy_min;
                }
            }
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        {
            const auto env_on = [](const char * name) {
                const char * env = getenv(name);
                return env == nullptr || atoi(env) != 0;
            };
            ckpt_ring        = env_on("LLAMA_SERVER_CKPT_RING");
            ckpt_ring_past   = env_on("LLAMA_SERVER_CKPT_RING_PAST");
            ckpt_resume_next = env_on("LLAMA_SERVER_CKPT_RESUME_NEXT");
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (decision.type != COMMON_DECISION_TYPE_NONE && !params_base.cache_ram_explicit) {
            SRV_INF("%s", "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n");
            params_base.cache_ram_mib    = 0;
            params_base.cache_idle_slots = false;
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
            prompt_cache->share_checkpoints = params_base.checkpoint_share;
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        ckpt_trim = n_swa > 0 && ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && ctx_dft == nullptr;

        ckpt_vram_n      = std::min(params_base.checkpoint_vram, std::max(params_base.n_ctx_checkpoints, 0));
        ckpt_vram_stores = ckpt_vram_n + (params_base.n_ctx_checkpoints > ckpt_vram_n ? 1 : 0);
        if (ckpt_vram_n > 0) {
            SRV_INF("context checkpoints on the devices: the newest %d of each slot, older ones move to the host %s\n",
                    ckpt_vram_n, params_base.checkpoint_vram_sync ? "at once" : "in the background");
        }
        // C217: checkpoints inside a prompt's calls go into device stores, need a model without sliding-window layers
        // (whose checkpoints hold a ring of cells), and a speculative decoder whose pass over a call follows its cuts
        ckpt_incall_on = false;
        if (params_base.checkpoint_incall) {
            const char * why = nullptr;
            if (ckpt_vram_n <= 0) {
                why = "it needs --ctx-checkpoints-vram";
            } else if (n_swa > 0) {
                why = "not with sliding-window layers";
            } else if (!common_speculative_incall_ok(spec.get())) {
                why = "not with this speculative decoding";
            }
            if (why != nullptr) {
                SRV_WRN("--ctx-checkpoints-incall: %s - off\n", why);
            } else {
                ckpt_incall_on = true;
                SRV_INF("%s", "context checkpoints inside a prompt's decode calls, into their device stores without a wait\n");
            }
        }

        if (params_base.checkpoint_spread > 0 && params_base.n_ctx_checkpoints > 0) {
            SRV_INF("context checkpoints inside prompts every %d tokens, evicted by the smallest gap\n", params_base.checkpoint_spread);
        }
        if (params_base.checkpoint_share && params_base.n_ctx_checkpoints > 0) {
            SRV_INF("%s", "context checkpoints shared by a slot, its prompt-cache entries and child slots, not copied\n");
        }

        if (!params_base.prefix_cache_dir.empty()) {
            if (ctx_dft != nullptr || mctx != nullptr || ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                SRV_WRN("%s", "prefix cache: not with a draft model, a multimodal projector or a context without memory - off\n");
            } else {
                prefix_store.init(params_base.prefix_cache_dir, server_prefix_store::fingerprint(model_tgt, ctx_tgt, params_base),
                        params_base.prefix_cache_min, params_base.prefix_cache_max);
                SRV_INF("prefix cache: '%s' holds %zu saved prefixes for this model and settings (%s-*); prefixes of %d tokens or more are saved, up to %d files\n",
                        params_base.prefix_cache_dir.c_str(), prefix_store.entries.size(), prefix_store.fp.c_str(), prefix_store.n_min, prefix_store.n_files);
            }
        }

        ckpt_gen_tokens.clear();
        if (params_base.checkpoint_gen) {
            if (!ckpt_trim || params_base.n_ctx_checkpoints <= 0) {
                SRV_WRN("%s", "--checkpoint-gen needs a sliding-window model and context checkpoints - off\n");
            } else {
                for (const auto & text : string_split<std::string>(params_base.checkpoint_gen_at, ',')) {
                    const auto toks = common_tokenize(vocab, text, false, true);
                    if (toks.size() == 1) {
                        ckpt_gen_tokens.push_back(toks[0]);
                    } else {
                        SRV_WRN("--checkpoint-gen-at: '%s' is %zu tokens, not one - ignored\n", text.c_str(), toks.size());
                    }
                }
                SRV_INF("generation checkpoints at %zu token(s) of '%s', %d tokens or more after the last checkpoint\n",
                        ckpt_gen_tokens.size(), params_base.checkpoint_gen_at.c_str(), params_base.checkpoint_gen_min);
            }
        }

        // Q771: the micro-batch taper of the target context's prompt calls: --ubatch-taper, else what LLAMA_UBATCH_TAPER set
        if (!params_base.ubatch_taper.empty() && !llama_set_ubatch_taper(ctx_tgt, params_base.ubatch_taper.c_str())) {
            SRV_ERR("bad --ubatch-taper '%s'\n", params_base.ubatch_taper.c_str());
            return false;
        }
        taper_default = llama_get_ubatch_taper(ctx_tgt);
        taper_cur     = taper_default;
        if (taper_default != "off") {
            SRV_INF("micro-batch taper of prompt calls: %s (n_ubatch %d)\n", taper_default.c_str(), (int) llama_n_ubatch(ctx_tgt));
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            try {
                asr_prompt = server_chat_get_asr_prompt(model_tgt);
            } catch (const std::exception & e) {
                SRV_ERR("%s: \"asr\" chat template error: %s\n", __func__, e.what());
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mtmd_cap.vision,
                /* allow_audio           */ mtmd_cap.audio,
                /* allow_video           */ mtmd_cap.video,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning").get<bool>();
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                // a busy slot is returned untouched, the caller defers the task
                if (ret->is_processing()) {
                    return ret;
                }

                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            const auto free_slots = get_free_slots(1, -1);

            if (!free_slots.empty()) {
                ret = free_slots[0];

                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache && !task.is_stateless();

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        // LLAMA_SERVER_TEXT_REUSE=1: the cached ids wherever the prompt spells the same text (server_reuse_cached_ids)
        if (server_text_reuse_enabled() && task.type == SERVER_TASK_TYPE_COMPLETION && task.params.cache_prompt &&
            !task.is_stateless() && task.n_tokens_shared == 0 && !lora_all_alora(slot.lora) &&
            !task.tokens.has_mtmd && !slot.prompt.tokens.has_mtmd) {
            llama_tokens ids = task.tokens.get_tokens();
            size_t n_match = 0;
            const size_t n_kept = server_reuse_cached_ids(vocab, slot.prompt.tokens.get_tokens(), ids, n_match);
            if (n_kept > 0) {
                SLT_INF(slot, "prompt ids differ from the cache at %zu, same text: %zu more cached tokens kept (%zu -> %zu prompt tokens)\n",
                        n_match, n_kept, task.tokens.size(), ids.size());
                task.tokens = server_tokens(ids, false);
            }
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        if (task.type == SERVER_TASK_TYPE_TTS) {
            GGML_ASSERT(has_cap_tts()); // should already checked in route handler
            if (!slot.tts.is_supported()) {
                slot.tts.ctx.init(ctx_tgt, slot.mctx);
            }
            // mtmd_helper always re-eval the whole prompt
            slot.prompt_clear();
            task.tts_inp.data.seq_id = slot.id;
            if (slot.tts.ctx.set_input(task.tts_inp.get()) != 0) {
                send_error(task, "failed to process TTS prompt", ERROR_TYPE_SERVER);
                return false;
            }
        }

        // realtime transcription: the rows come from the audio stream, the helper may bias some tokens
        if (task.type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            auto sparams = mtmd_audio_stream_params_default();
            sparams.prompt   = asr_prompt.data();
            sparams.n_prompt = asr_prompt.size();
            slot.audio_stream.reset(mtmd_audio_stream_init(mctx, sparams));
            if (!slot.audio_stream) {
                send_error(task, "failed to initialize the audio stream", ERROR_TYPE_SERVER);
                return false;
            }
            if (!asr_prompt.empty()) {
                for (const auto & b : mtmd_helper::audio_stream_logit_bias(mctx, model_tgt)) {
                    task.params.sampling.logit_bias.push_back(b);
                }
            }
        }

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: check verify if this actually works with TTS
            use_backend_sampling &= task.type != SERVER_TASK_TYPE_TTS;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        if (slot.task->type == SERVER_TASK_TYPE_TTS) {
            slot.state = SLOT_STATE_PROCESSING_PROMPT;
        } else {
            slot.state = slot.task->is_child()
                ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
                : SLOT_STATE_STARTED;
        }

        // no prompt, the rows come from the audio stream, which audio controls reach from now on
        if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            slot.prompt_clear();
            slot.init_sampler();
            slot.sampled = LLAMA_TOKEN_NULL;
            slot.i_batch = -1;
            slot.stats.update_prompt_start();
            slot.state = SLOT_STATE_GENERATING;
            send_partial_response(slot, {}, false, true);
        }

        // no prompt either, the session appends text and audio and asks for responses with its controls
        if (slot.task->type == SERVER_TASK_TYPE_REALTIME) {
            slot.rt = std::make_unique<server_realtime_slot>(ctx_tgt, mctx, slot.id, slot.n_ctx, slot.task->params.sampling,
                                                             queue_results, slot.task->id, slot.task->index);
            slot.prompt_clear();
            slot.stats.update_prompt_start();
            slot.state      = SLOT_STATE_GENERATING;
            slot.audio_wait = true;
            send_partial_response(slot, {}, false, true);
        }

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated.append(token_str, result.tok);
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated.truncate(pos + stop_pos);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            res->tokens.assign(1, tkn.tok);
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            slot.update_sampling_stats();
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_tts_result(server_slot & slot, int32_t sample_rate, const char * data, size_t data_len, bool final) {
        auto res = std::make_unique<server_task_result_tts>();

        res->id          = slot.task->id;
        res->index       = slot.task->index;
        res->sample_rate = sample_rate;
        res->audio.assign(data, data_len);
        res->final       = final;

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        slot.update_sampling_stats();
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (decision.pooled) {
            const float * embd = llama_get_embeddings_seq(slot.ctx_tgt, slot.id);
            if (embd == nullptr) {
                send_error(slot, "failed to get the class scores", ERROR_TYPE_SERVER);
                return;
            }
            res->scores.assign(embd, embd + llama_model_n_embd_out(model_tgt));
        } else if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
            if (!decision.label_groups.empty()) {
                std::vector<float> scores;
                size_t i = 0;
                for (const int32_t n : decision.label_groups) {
                    GGML_ASSERT(n > 0 && i + n <= res->scores.size());
                    scores.push_back(*std::max_element(res->scores.begin() + i, res->scores.begin() + i + n));
                    i += n;
                }
                res->scores = std::move(scores);
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };

            // joint head (decision model): the scores are the first rows
            for (int32_t i = 0; i < decision.n_scores; i++) {
                const float * embd = i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
                if (embd == nullptr) {
                    send_error(slot, "failed to get embeddings", ERROR_TYPE_SERVER);
                    return;
                }
                res->scores.push_back(embd[0]);
            }

            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
            GGML_ASSERT(decision.column >= 0 && decision.column < n_embd_out);

            for (const int32_t marker : decision.markers) {
                const float * embd = get_embd(marker);
                if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                    send_error(slot, "failed to get embeddings, the question and its options must fit in one batch", ERROR_TYPE_SERVER);
                    return;
                }
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }
                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
        }

        SLT_DBG(slot, "%s", "sending decision result\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
        }

        std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot * a, const server_slot * b) {
            if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) {
                return a->prompt.n_tokens() == 0;
            }
            return a->t_last_used < b->t_last_used;
        });

        free_slots.resize(std::min(free_slots.size(), n_slots_needed));

        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // --prefix-cache: the longest saved prefix of the slot's new prompt, if the request may use one
    const server_prefix_store::entry * find_prefix(const server_slot & slot, int32_t n_decision_first) const {
        if (!prefix_store.enabled() || !slot.task->params.cache_prompt || slot.task->is_stateless() ||
            slot.task->type != SERVER_TASK_TYPE_COMPLETION || slot.task->n_tokens_shared > 0 || n_decision_first >= 0 ||
            slot.alora_invocation_start > 0 || slot.task->tokens.has_mtmd) {
            return nullptr;
        }
        return prefix_store.find(slot.task->tokens);
    }

    // replaces the slot's cells and tokens with a saved prefix; returns its token count, or 0 with the slot empty
    // (the entry may move or go: do not use it afterwards)
    int32_t prefix_restore(server_slot & slot, const server_prefix_store::entry & e) {
        const int64_t t_start = ggml_time_us();
        const std::string  path     = e.path;
        const llama_tokens expected = e.tokens;

        slot.mem.seq_rm(slot.id, -1, -1);
        slot.prompt.clear();

        llama_tokens tokens(expected.size());
        size_t n_tokens = 0;
        const size_t nread = llama_state_seq_load_file(ctx_tgt, path.c_str(), slot.id, tokens.data(), tokens.size(), &n_tokens);
        if (nread == 0 || n_tokens != expected.size() || tokens != expected) {
            SLT_WRN(slot, "prefix cache: failed to restore '%s' - deleted, the prompt is processed in full\n", path.c_str());
            slot.mem.seq_rm(slot.id, -1, -1);
            prefix_store.remove(path);
            slot.stats.cache_src = 0;
            return 0;
        }

        // no checkpoints: a cold slot has none before this point either, and the batch at the prefix's end takes one
        slot.prompt.tokens = server_tokens(tokens, false);

        const int64_t t_us = ggml_time_us() - t_start;
        SLT_INF(slot, "prefix cache: restored %zu tokens (%.1f MiB) from '%s' in %.1f ms\n",
                n_tokens, nread / 1024.0 / 1024.0, path.c_str(), t_us / 1000.0);

        prefix_store.touch(path);

        slot.stats.cache_src     = 3;
        slot.stats.t_restore_us += t_us;
        return (int32_t) n_tokens;
    }

    // --prefix-cache: called before the batch that starts at n_tokens_start is decoded, when the slot holds exactly the
    // cells of the tokens before it; saves them if they are a prompt's shared prefix processed from the first token
    void prefix_maybe_save(server_slot & slot, int32_t n_tokens_start, int32_t first_user_pos) {
        if (!prefix_store.enabled() || n_tokens_start != first_user_pos || n_tokens_start < prefix_store.n_min ||
            slot.stats.n_prompt_cached != 0 || slot.task->type != SERVER_TASK_TYPE_COMPLETION ||
            slot.task->is_stateless() || slot.task->n_tokens_shared > 0 || slot.task->tokens.has_mtmd ||
            slot.alora_invocation_start > 0) {
            return;
        }
        const llama_tokens & all = slot.task->tokens.get_tokens();
        llama_tokens prefix(all.begin(), all.begin() + n_tokens_start);
        if (prefix_store.has(prefix)) {
            return;
        }

        const int64_t t_start = ggml_time_us();
        const std::string path = prefix_store.path_of(prefix);
        const std::string tmp  = path + ".tmp";

        // with the whole sliding-window ring: a restore then occupies exactly the cells processing the prefix does (GPU
        // attention splits its keys by the cache's extent, so fewer cells round differently)
        const size_t nwrite = llama_state_seq_save_file_ext(ctx_tgt, tmp.c_str(), slot.id, prefix.data(), prefix.size(),
                LLAMA_STATE_SEQ_FLAGS_SWA_RING);
        std::error_code ec;
        if (nwrite == 0) {
            SLT_WRN(slot, "prefix cache: failed to write '%s'\n", tmp.c_str());
            std::filesystem::remove(std::filesystem::u8path(tmp), ec);
            return;
        }
        std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path), ec);
        if (ec) {
            SLT_WRN(slot, "prefix cache: failed to rename '%s': %s\n", tmp.c_str(), ec.message().c_str());
            std::filesystem::remove(std::filesystem::u8path(tmp), ec);
            return;
        }

        SLT_INF(slot, "prefix cache: saved %zu tokens (%.1f MiB) to '%s' in %.1f ms\n",
                prefix.size(), nwrite / 1024.0 / 1024.0, path.c_str(), (ggml_time_us() - t_start) / 1000.0);

        prefix_store.add({ path, std::move(prefix) });
    }

    // --checkpoint-gen: called before a generating slot's sampled token goes into the batch, when the slot holds the
    // cells of the tokens before it; a checkpoint there lets a client's re-rendered reply that differs from what was
    // generated after this point re-read only from here
    void checkpoint_gen_maybe(server_slot & slot) {
        if (ckpt_gen_tokens.empty() || !slot.spec_draft.empty() || !slot.pipe_win.empty() || !slot.inp_embd.empty() ||
            slot.task->type != SERVER_TASK_TYPE_COMPLETION ||
            std::find(ckpt_gen_tokens.begin(), ckpt_gen_tokens.end(), slot.sampled) == ckpt_gen_tokens.end()) {
            return;
        }
        const int64_t n_last = slot.prompt.checkpoints.empty() ? 0 : slot.prompt.checkpoints.back()->n_tokens;
        if (slot.prompt.n_tokens() < n_last + params_base.checkpoint_gen_min) {
            return;
        }
        const llama_pos pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
        if (pos_min < 0) {
            return;
        }
        create_checkpoint(slot, 0, pos_min, pos_max);
    }

    // the room a new checkpoint at n_tokens_new takes in the slot's list (create_checkpoint)
    void ckpt_make_room(server_slot & slot, const int id_task, const int64_t n_tokens_new) {
        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                params_base.checkpoint_spread == 0 &&
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            const auto & ck = **it;
            if (ck.id_task != id_task && last >= 0 && ck.n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        ck.pos_min, ck.pos_max, ck.n_tokens, (float) ck.size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = ck.n_tokens;
            ++it;
        }

        while (params_base.checkpoint_spread == 0 && slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = *slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if ((*it)->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", (*it)->n_tokens);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // --ctx-checkpoints-spread: a full list drops the checkpoint whose removal leaves the smallest gap
        while (params_base.checkpoint_spread > 0 && !slot.prompt.checkpoints.empty() &&
                slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            ckpt_evict_smallest_gap(slot, n_tokens_new);
        }
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    // a sliding-window model's checkpoints hold the whole ring (ckpt_trim): a restore then occupies the cells that
    // processing the prompt again does, so attention sums over the same cells, and one cut back to an earlier divergence
    // still has a complete window
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const bool ring = ckpt_trim && ckpt_ring;

        const int id_task = slot.task->id;

        slot.stats.n_ckpt_created++;

        const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;

        ckpt_make_room(slot, id_task, n_tokens_new);

        // filled in place: until this function returns, only the slot's list holds it
        auto ckpt = std::make_shared<common_prompt_checkpoint>();
        slot.prompt.checkpoints.push_back(ckpt);
        auto & cur = *ckpt;

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.ring = ring;

        const llama_state_seq_flags flags_tgt = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | (ring ? LLAMA_STATE_SEQ_FLAGS_SWA_RING : 0);

        const int64_t t_save_start = ggml_time_us();

        // --ctx-checkpoints-vram: into a device store of the slot's sequence, where one is free and can be allocated
        bool on_dev = false;
        if (ckpt_vram_n > 0) {
            const int32_t store = ckpt_store_free(slot);
            on_dev = store >= 0 && cur.update_tgt_store(ctx_tgt, slot.id, store, flags_tgt);
            if (!on_dev) {
                SLT_WRN(slot, "context checkpoints on the devices failed (store %d) - on the host from now on\n", store);
                ckpt_vram_n = 0;
            }
        }
        if (!on_dev) {
            cur.update_tgt(ctx_tgt, slot.id, flags_tgt);
        }
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        const int64_t t_save_us = ggml_time_us() - t_save_start;

        slot.stats.t_ckpt_save_us += t_save_us;
        slot.stats.n_ckpt_dev     += on_dev ? 1 : 0;

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB, on the devices %.3f MiB, store %d, %.1f ms)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024, (float) cur.size_dev / 1024 / 1024, cur.store, t_save_us / 1000.0);

        if (on_dev) {
            ckpt_vram_trim(slot);
        }
    }

    // --ctx-checkpoints-incall (C217): the checkpoint at batch row i_batch, where the call the batch joins there would
    // have started (n_tokens_cur tokens of it in the batch): its place in the list and its device store now, as
    // create_checkpoint takes them, its data once the decode call has filled the store (ckpt_incall_finish)
    void create_checkpoint_incall(server_slot & slot, const int64_t n_tokens_cur, int32_t i_batch) {
        slot.stats.n_ckpt_created++;

        const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;

        ckpt_make_room(slot, slot.task->id, n_tokens_new);

        const int32_t store = ckpt_store_free(slot);
        if (store < 0) {
            SLT_WRN(slot, "no free device store for the context checkpoint at n_tokens = %" PRId64 " inside the call - none taken\n", n_tokens_new);
            return;
        }

        auto ckpt = std::make_shared<common_prompt_checkpoint>();
        slot.prompt.checkpoints.push_back(ckpt);

        ckpt->id_task = slot.task->id;
        ckpt->update_pos(n_tokens_new, -1, -1);
        ckpt->store   = store;

        ckpt_incall.push_back({ ckpt, slot.id, i_batch });
    }

    server_slot * slot_by_id(int id) {
        for (auto & slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }
        return nullptr;
    }

    // whether the slot's list still holds an in-call checkpoint (a later one of the same batch may have evicted it)
    static bool ckpt_incall_listed(const server_slot & slot, const ckpt_incall_job & ic) {
        return std::find(slot.prompt.checkpoints.begin(), slot.prompt.checkpoints.end(), ic.ck) != slot.prompt.checkpoints.end();
    }

    // before the decode call of the batch rows [off, off + n): its fills inside the call and its micro-batch cuts, as rows
    // of the call (cuts). The draft context gets the same fills for its part of each checkpoint (under the checkpoint's
    // store id in that context), taken in its pass over the call
    void ckpt_incall_begin(int32_t off, int32_t n, std::vector<uint32_t> & cuts) {
        for (const int32_t b : incall_cuts) {
            if (b > off && b < off + n) {
                cuts.push_back((uint32_t) (b - off));
            }
        }
        for (auto & ic : ckpt_incall) {
            if (ic.done || ic.i_batch <= off || ic.i_batch >= off + n) {
                continue;
            }
            server_slot * slot = slot_by_id(ic.slot_id);
            if (slot == nullptr || !ckpt_incall_listed(*slot, ic)) {
                ic.done = true;
                continue;
            }
            llama_state_seq_store_at(ctx_tgt, slot->id, ic.ck->store, (uint32_t) (ic.i_batch - off), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (ctx_dft != nullptr) {
                llama_state_seq_store_at(ctx_dft, slot->id, ic.ck->store, (uint32_t) (ic.i_batch - off), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }
        llama_set_ubatch_cuts(ctx_tgt, cuts.data(), cuts.size());
    }

    // after that call (and the draft's pass over it): the checkpoint's data, as create_checkpoint saves it
    void ckpt_incall_finish(ckpt_incall_job & ic, int32_t off) {
        ic.done = true;

        server_slot * sp = slot_by_id(ic.slot_id);
        if (sp == nullptr || !ckpt_incall_listed(*sp, ic)) {
            return;
        }
        server_slot & slot = *sp;
        auto & cur = *ic.ck;

        const int64_t t_save_start = ggml_time_us();

        llama_pos pos_min = -1;
        llama_pos pos_max = -1;
        const size_t n = llama_state_seq_store_at_get(ctx_tgt, nullptr, 0, slot.id, cur.store, &pos_min, &pos_max);
        if (n == 0) {
            SLT_WRN(slot, "the context checkpoint at n_tokens = %" PRId64 " was not taken inside the call (store %d) - erased, and checkpoints cut the calls again from now on\n",
                    cur.n_tokens, cur.store);
            slot.prompt.checkpoints.remove(ic.ck);
            ckpt_incall_on = false;
            return;
        }

        // the draft's part as update_dft takes it: its fill in the draft's pass, in the host format
        if (ctx_dft != nullptr) {
            std::vector<uint8_t> blob(llama_state_seq_store_at_get(ctx_dft, nullptr, 0, slot.id, cur.store, nullptr, nullptr));
            const size_t n_host = blob.empty() ? 0 : llama_state_seq_store_get_size_host(ctx_dft, slot.id, cur.store);
            cur.data_dft.resize(n_host);
            if (n_host == 0 ||
                llama_state_seq_store_at_get(ctx_dft, blob.data(), blob.size(), slot.id, cur.store, nullptr, nullptr) != blob.size() ||
                llama_state_seq_store_get_data_host(ctx_dft, blob.data(), blob.size(), cur.data_dft.data(), n_host, slot.id, cur.store) != n_host) {
                SLT_WRN(slot, "the draft's part of the context checkpoint at n_tokens = %" PRId64 " was not taken inside the call - erased, and checkpoints cut the calls again from now on\n",
                        cur.n_tokens);
                slot.prompt.checkpoints.remove(ic.ck);
                ckpt_incall_on = false;
                return;
            }
        }

        cur.data_tgt.resize(n);
        llama_state_seq_store_at_get(ctx_tgt, cur.data_tgt.data(), n, slot.id, cur.store, nullptr, nullptr);
        cur.update_pos(cur.n_tokens, pos_min, pos_max);
        cur.size_dev = llama_state_seq_store_get_size_host(ctx_tgt, slot.id, cur.store) - n;
        // the draft's speculative state at the checkpoint, as the call that ended there would leave it
        common_speculative_get_state_at(spec.get(), slot.id, ic.i_batch - off, cur.data_spec);

        const int64_t t_save_us = ggml_time_us() - t_save_start;

        slot.stats.t_ckpt_save_us += t_save_us;
        slot.stats.n_ckpt_dev++;
        slot.stats.n_ckpt_incall++;

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB, on the devices %.3f MiB, store %d, %.1f ms) inside the call\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024, (float) cur.size_dev / 1024 / 1024, cur.store, t_save_us / 1000.0);

        ckpt_vram_trim(slot);
    }

    // the end of a batch: checkpoints of it that no call filled leave their lists (a retried or failed call)
    void ckpt_incall_drop(const char * why) {
        for (auto & ic : ckpt_incall) {
            if (ic.done) {
                continue;
            }
            ic.done = true;
            server_slot * slot = slot_by_id(ic.slot_id);
            if (slot != nullptr && ckpt_incall_listed(*slot, ic)) {
                SLT_WRN(*slot, "context checkpoint at n_tokens = %" PRId64 " erased: %s\n", ic.ck->n_tokens, why);
                slot->prompt.checkpoints.remove(ic.ck);
            }
        }
        ckpt_incall.clear();
        incall_cuts.clear();
        ckpt_incall_slot = -1;
    }

    // --ctx-checkpoints-spread (C28): erases the checkpoint whose removal leaves the smallest gap between its neighbours:
    // the prompt's start before the first, the new checkpoint at n_tokens_new after the last (so of two checkpoints at
    // a prompt's end, the earlier goes before one far back does); the older of equal gaps goes first
    void ckpt_evict_smallest_gap(server_slot & slot, int64_t n_tokens_new) {
        auto & cks = slot.prompt.checkpoints;

        auto    best     = cks.begin();
        int64_t best_gap = std::numeric_limits<int64_t>::max();
        int64_t prev     = 0;
        for (auto it = cks.begin(); it != cks.end(); ++it) {
            const int64_t next = std::next(it) != cks.end() ? (*std::next(it))->n_tokens : n_tokens_new;
            if (next - prev < best_gap) {
                best     = it;
                best_gap = next - prev;
            }
            prev = (*it)->n_tokens;
        }

        SLT_TRC(slot, "erasing context checkpoint by the smallest gap (n_tokens = %" PRId64 ", size = %.3f MiB, store %d)\n",
                (*best)->n_tokens, (float) ((*best)->size() + (*best)->size_dev) / 1024 / 1024, (*best)->store);

        cks.erase(best);
    }

    // --ctx-checkpoints-spread: a checkpoint inside the prompt goes at the first message start spread / 2 tokens or more
    // after ref (the last checkpoint, or the batch's start where it takes one), or spread tokens after ref without one;
    // none in the prompt's last batch, which the tail checkpoints cover
    bool ckpt_spread_at(const server_slot & slot, int64_t n, int64_t ref) const {
        const int64_t spread = params_base.checkpoint_spread;
        if (spread <= 0 || n + (int64_t) llama_n_ubatch(ctx_tgt) + 4 >= (int64_t) slot.task->n_tokens()) {
            return false;
        }
        if (n == ref + spread) {
            return true;
        }
        if (n < ref + spread/2) {
            return false;
        }
        for (const auto & span : slot.task->params.message_spans.spans) {
            if ((int64_t) span.pos == n) {
                return true;
            }
        }
        return false;
    }

    // --ctx-checkpoints-vram: a host copy that finished in the background replaces its checkpoint's store; wait = until
    // it is done
    void ckpt_spill_land(server_slot & slot, bool wait) {
        const auto it = ckpt_spills.find(slot.id);
        if (it == ckpt_spills.end()) {
            return;
        }
        if (!wait && it->second.data.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return;
        }

        const int32_t store = it->second.store;

        common_state_data data;
        try {
            data = it->second.data.get();
        } catch (const std::exception & e) {
            SLT_WRN(slot, "moving a context checkpoint to the host failed: %s\n", e.what());
        }
        ckpt_spills.erase(it);

        // the checkpoint may be gone by now (invalidated, or the slot holds another prompt): then so is the copy
        // a store's checkpoint is never shared (copies of a prompt take host copies or none): its host version replaces it
        for (auto ck = slot.prompt.checkpoints.begin(); ck != slot.prompt.checkpoints.end(); ++ck) {
            if ((*ck)->store != store) {
                continue;
            }
            if (data.empty()) {
                SLT_WRN(slot, "erasing context checkpoint (n_tokens = %" PRId64 "): its host copy failed\n", (*ck)->n_tokens);
                slot.prompt.checkpoints.erase(ck);
            } else {
                auto host = std::make_shared<common_prompt_checkpoint>(**ck);
                host->data_tgt = std::move(data);
                host->store    = -1;
                host->size_dev = 0;
                *ck = std::move(host);
                SLT_TRC(slot, "context checkpoint (n_tokens = %" PRId64 ") moved to the host, %.3f MiB\n",
                        (*ck)->n_tokens, (float) (*ck)->size() / 1024 / 1024);
            }
            break;
        }
    }

    // a free device store for a new checkpoint of the slot, -1 if none
    int32_t ckpt_store_free(server_slot & slot) {
        ckpt_spill_land(slot, false);

        for (int attempt = 0; attempt < 2; ++attempt) {
            std::vector<bool> used(ckpt_vram_stores, false);
            for (const auto & ck : slot.prompt.checkpoints) {
                if (ck->store >= 0 && ck->store < ckpt_vram_stores) {
                    used[ck->store] = true;
                }
            }

            const auto it = ckpt_spills.find(slot.id);
            if (it != ckpt_spills.end()) {
                used[it->second.store] = true;
            }

            for (int32_t k = 0; k < ckpt_vram_stores; ++k) {
                if (!used[k]) {
                    return k;
                }
            }

            if (it == ckpt_spills.end()) {
                break;
            }

            // all taken, one of them by a move to the host: wait for it
            ckpt_spill_land(slot, true);
        }

        return -1;
    }

    // keeps the slot's newest ckpt_vram_n checkpoints on the devices: the oldest beyond them moves to the host, in the
    // background unless --ctx-checkpoints-vram-sync (one at a time)
    void ckpt_vram_trim(server_slot & slot) {
        const auto spilling = [&]() {
            const auto it = ckpt_spills.find(slot.id);
            return it == ckpt_spills.end() ? -1 : it->second.store;
        };

        int32_t n_dev = 0;
        for (const auto & ck : slot.prompt.checkpoints) {
            n_dev += ck->store >= 0 && ck->store != spilling() ? 1 : 0;
        }
        if (n_dev <= ckpt_vram_n) {
            return;
        }

        ckpt_spill_land(slot, true);

        const auto ck = std::find_if(slot.prompt.checkpoints.begin(), slot.prompt.checkpoints.end(),
                [](const server_prompt_checkpoint_ptr & c) { return c->store >= 0; });
        if (ck == slot.prompt.checkpoints.end()) {
            return;
        }

        if (params_base.checkpoint_vram_sync) {
            const int64_t t_start = ggml_time_us();
            // a store's checkpoint is never shared: its host version replaces it
            auto host = std::make_shared<common_prompt_checkpoint>(**ck);
            if (!host->tgt_to_host(ctx_tgt, slot.id)) {
                SLT_WRN(slot, "erasing context checkpoint (n_tokens = %" PRId64 "): its host copy failed\n", (*ck)->n_tokens);
                slot.prompt.checkpoints.erase(ck);
                return;
            }
            *ck = std::move(host);
            SLT_TRC(slot, "context checkpoint (n_tokens = %" PRId64 ") moved to the host, %.3f MiB, %.1f ms\n",
                    (*ck)->n_tokens, (float) (*ck)->size() / 1024 / 1024, (ggml_time_us() - t_start) / 1000.0);
            return;
        }

        const size_t n = llama_state_seq_store_get_size_host(ctx_tgt, slot.id, (*ck)->store);

        auto & job = ckpt_spills[slot.id];
        job.store = (*ck)->store;
        job.data  = std::async(std::launch::async,
                [ctx = ctx_tgt, seq_id = slot.id, store = (*ck)->store, meta = (*ck)->data_tgt, n]() {
                    common_state_data data(n);
                    if (n == 0 || llama_state_seq_store_get_data_host(ctx, meta.data(), meta.size(), data.data(), data.size(), seq_id, store) != n) {
                        data.clear();
                    }
                    return data;
                });
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    template <typename Buf>
    static bool ckpt_read_buf(std::ifstream & ifs, Buf & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    template <typename Buf>
    static void ckpt_write_buf(std::ofstream & ofs, const Buf & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & ckpt : slot.prompt.checkpoints) {
            const auto & cur = *ckpt;
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            if (cur.store >= 0) {
                // a checkpoint on the devices (--ctx-checkpoints-vram) goes into the file as its host copy
                common_prompt_checkpoint host;
                host.data_tgt = cur.data_tgt;
                host.store    = cur.store;
                if (!host.tgt_to_host(ctx_tgt, slot.id)) {
                    SLT_WRN(slot, "failed to copy a context checkpoint from the devices for '%s'\n", filepath.c_str());
                    return false;
                }
                ckpt_write_buf(ofs, host.data_tgt, n_written);
            } else {
                ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            }
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        std::list<std::shared_ptr<common_prompt_checkpoint>> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            auto & cur = *ckpt;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(ckpt));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                checkpoints.pop_front();
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back()->data_dft.empty()) {
            const bool ok = checkpoints.back()->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur->clear_dft();
                }
            }
        }
        slot.prompt.checkpoints.assign(checkpoints.begin(), checkpoints.end());
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(spec.get(), slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(spec.get(), slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
            case SERVER_TASK_TYPE_REALTIME:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_DECISION:
                {
                    server_keepalive_kick();

                    if (task.need_sampling() && !decision.can_generate()) {
                        send_error(task, "This model is a decision model without text generation, use /v1/systemone", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %d on unknown completion id=%s, no live slot\n",
                                (int) task.params.control_action, task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == SERVER_CONTROL_ACTION_REASONING_END) {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else if (slot->rt) {
                        // a cancel acts now, everything else runs after what is already queued
                        if (task.params.control_action == SERVER_CONTROL_ACTION_RESPONSE_CANCEL) {
                            slot->rt->cancel();
                        } else {
                            slot->rt->push({
                                task.params.control_action,
                                std::move(task.params.control_text),
                                std::move(task.params.control_audio),
                                task.params.control_response,
                            });
                        }
                        slot->audio_wait = false;
                        res->success = true;
                    } else if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END) {
                        if (!slot->audio_stream) {
                            res->success = false;
                            res->message = "not a realtime transcription slot";
                        } else {
                            auto * stream = slot->audio_stream.get();
                            if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND) {
                                mtmd_audio_stream_push(stream, task.params.control_audio.data(), task.params.control_audio.size());
                            } else {
                                const int32_t ret = mtmd_audio_stream_flush(stream, task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END);
                                // a non-final commit completes its item once the rows of its audio are decoded
                                if (ret == 0 && task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT) {
                                    slot->audio_commit_rows.push_back(mtmd_audio_stream_get_n_rows_flushed(stream));
                                }
                            }
                            slot->audio_wait = false;
                            res->success = true;
                        }
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (int pass = 0; pass < 2; ++pass) {
            for (auto & slot : slots) {
                const bool is_child = slot.is_processing() && slot.task->is_child();
                if (is_child != (pass == 1)) {
                    continue;
                }
                try {
                    callback(slot);
                } catch (const std::exception & e) {
                    SLT_ERR(slot, "got exception: %s\n", e.what());
                    send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                    slot.release();
                }
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // fill slot.inp_embd with the next rows of the audio stream
    // returns false if the slot has nothing to decode now
    bool slot_audio_step(server_slot & slot) {
        if (!slot.inp_embd.empty()) {
            return true; // rows from a previous step, not decoded yet
        }

        const int32_t n = mtmd_audio_stream_step(slot.audio_stream.get(), slot.sampled);

        // complete the commits whose rows have their text sampled, before the new rows: their text goes to the next item
        const int64_t n_rows = mtmd_audio_stream_get_n_rows(slot.audio_stream.get());
        while (!slot.audio_commit_rows.empty() && slot.audio_commit_rows.front() <= n_rows) {
            slot.audio_commit_rows.pop_front();
            auto res = std::make_unique<server_task_result_cmpl_partial>();
            res->id                   = slot.task->id;
            res->index                = slot.task->index;
            res->n_decoded            = slot.stats.n_gen;
            res->n_prompt_tokens      = slot.task->n_tokens();
            res->res_type             = slot.task->params.res_type;
            res->oaicompat_model      = slot.task->params.oaicompat_model;
            res->oaicompat_cmpl_id    = slot.task->params.oaicompat_cmpl_id;
            res->audio_commit_drained = true;
            queue_results.send(std::move(res));
        }

        if (n > 0) {
            // new sequence (e.g. live ASR restart): drop the old one, the text goes on in a new line
            if (mtmd_audio_stream_is_seq_start(slot.audio_stream.get())) {
                slot.prompt_clear();
                if (slot.n_sent_text > 0) {
                    completion_token_output sep;
                    sep.tok          = LLAMA_TOKEN_NULL;
                    sep.text_to_send = "\n";
                    slot.generated.append(common_chat_input(sep.text_to_send));
                    slot.n_sent_text += sep.text_to_send.size();
                    send_partial_response(slot, sep, false);
                }
            }
            const float * rows = mtmd_audio_stream_get_output_embd(slot.audio_stream.get());
            slot.inp_embd.assign(rows, rows + (size_t) n * batch.n_embd);
            slot.audio_wait = false;
            return true;
        }
        if (n == 0) {
            slot.audio_wait = true;
            return false;
        }
        if (n == -2) {
            send_error(slot, "audio stream error", ERROR_TYPE_SERVER);
            slot.release();
            return false;
        }

        // end of stream
        slot.stop           = STOP_TYPE_EOS;
        slot.has_next_token = false;
        slot.print_timings();
        send_final_response(slot);
        slot.release();
        return false;
    }

    //
    // pipelined speculation (--pipe-depth, llama_pipe_*)
    //
    // A slot that generates alone keeps its newest sampled token and up to pipe_depth guesses of the tokens after it in
    // flight, each a one-token evaluation in a pipe slot, so the first GPUs work on a guess while the later ones still
    // work on the token before it. Tokens are sampled in order, on the device, with every draw keyed by the token's
    // position: a guess that equals the sampled token is already on its way, a wrong one is dropped with the guesses
    // behind it (its KV cells removed) and the sampled token goes in instead. The output is the one batched decoding
    // gives with the same position-keyed draws, whatever the depth, the guesses, or how often the slot switches.

    // whether the slot's next tokens may run pipelined (the caller checks that it generates alone): its sampling runs
    // entirely on the device and does not depend on the tokens before it, and its window fits in the context
    bool pipe_eligible(const server_slot & slot) const {
        if (params_base.pipe_depth <= 0 || slot.state != SLOT_STATE_GENERATING || !slot.task || !slot.smpl) {
            return false;
        }
        if (slot.task->type == SERVER_TASK_TYPE_TTS || !slot.task->need_sampling()) {
            return false;
        }
        if (slot.can_speculate() || slot.audio_stream || !slot.inp_embd.empty() || slot.mctx) {
            return false;
        }
        const auto & sp = slot.task->params.sampling;
        if (sp.n_probs > 0) {
            return false;
        }
        // a grammar may not reject a token in flight: pipelined only while it cannot (a lazy grammar before its
        // trigger, or inside the reasoning), with the device's token checked on the host afterwards (Q67)
        if (!sp.grammar.empty() && common_sampler_grammar_active(slot.smpl.get())) {
            return false;
        }
        // a reasoning budget may force a token, which only common_sampler_sample applies
        if (sp.reasoning_budget_tokens >= 0 || sp.reasoning_control) {
            return false;
        }
        // penalties read the tokens before the one sampled, which a guess's evaluation does not have yet
        if (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f) {
            return false;
        }
        for (const auto & lora : slot.lora) {
            if (lora.scale != 0.0f) {
                return false;
            }
        }
        if (!llama_sampler_chain_on_backend(common_sampler_get(slot.smpl.get()))) {
            return false;
        }
        return slot.prompt.n_tokens() + params_base.pipe_depth + 2 < slot.n_ctx;
    }

    // the slot goes back to batched decoding: everything in flight goes, its newest sampled token too, which the batch
    // decodes again (slot.sampled)
    void pipe_leave(server_slot & slot) {
        if (slot.pipe_win.empty()) {
            return;
        }
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, slot.pipe_win.front().pos, -1);
        llama_pipe_drain(ctx_tgt);
        slot.pipe_win.clear();
        slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - 1);
        slot.pipe_draft_pending = -1;
    }

    // whether the staircase drafter guesses for this slot
    bool pipe_drafts(const server_slot & slot) const {
        return pipe_draft_on && !pipe_draft_taps_only && pipe_draft_seq == slot.id;
    }

    // the guess of the copy lookup for the position after the slot's window, or -1
    llama_token pipe_copy_guess(server_slot & slot) {
        std::vector<llama_token> inflight;
        for (size_t i = 1; i < slot.pipe_win.size(); i++) {
            inflight.push_back(slot.pipe_win[i].tok);
        }
        return slot.pipe_guesser.guess(inflight.data(), inflight.size());
    }

    // starts the drafter on the position after the slot's window: the tokens in flight with their pipe slots
    void pipe_draft_launch(server_slot & slot) {
        std::vector<llama_pipe_draft_row> rows;
        for (const auto & e : slot.pipe_win) {
            rows.push_back({ e.tok, e.pos, e.pslot });
        }
        if (llama_pipe_draft_launch(ctx_tgt, rows.data(), (int32_t) rows.size()) == 0) {
            slot.pipe_draft_pending = slot.pipe_win.back().pos + 1;
        }
    }

    // after a submission: the drafter starts on the next position at once (unless the copy lookup has it), so that its
    // guess is ready when a pipe slot opens
    void pipe_draft_ahead(server_slot & slot) {
        if (!pipe_drafts(slot) || slot.pipe_win.empty() || slot.pipe_win.back().pos + 2 >= slot.n_ctx) {
            return;
        }
        // a look ahead: a store guess found here is counted when it is made
        const int32_t n_store = slot.pipe_guesser.n_store;
        const bool none = pipe_copy_guess(slot) < 0;
        slot.pipe_guesser.n_store = n_store;
        if (none) {
            pipe_draft_launch(slot);
        }
    }

    // submits tok at pos in a free pipe slot; false on error
    bool pipe_submit(server_slot & slot, llama_token tok, llama_pos pos) {
        const int32_t n_pipe = llama_pipe_n_slots(ctx_tgt);
        for (int32_t k = 0; k < n_pipe; k++) {
            const int32_t s = (pipe_next + k) % n_pipe;
            bool used = false;
            for (const auto & e : slot.pipe_win) {
                used = used || e.pslot == s;
            }
            if (used) {
                continue;
            }
            pipe_next = (s + 1) % n_pipe;
            llama_sampler_set_pos(common_sampler_get(slot.smpl.get()), pos);
            const int32_t ret = llama_pipe_submit(ctx_tgt, s, tok, pos, slot.id);
            if (ret != 0) {
                SLT_ERR(slot, "llama_pipe_submit failed (%d) at pos %d\n", ret, pos);
                return false;
            }
            slot.pipe_win.push_back({ s, tok, pos, false });
            return true;
        }
        SLT_ERR(slot, "%s", "no free pipe slot\n");
        return false;
    }

    // keeps up to pipe_depth guesses in flight behind the newest sampled token, within the context and the token limit
    void pipe_top_up(server_slot & slot) {
        while ((int32_t) slot.pipe_win.size() < params_base.pipe_depth + 1) {
            const llama_pos pos = slot.pipe_win.back().pos + 1;
            if (pos + 1 >= slot.n_ctx) {
                break;
            }
            // tokens beyond the limit would never be sampled
            if (slot.n_predict_max != -1 && (int32_t) slot.pipe_win.size() >= slot.n_remaining()) {
                break;
            }
            const size_t n_inflight = slot.pipe_win.size() - 1;
            llama_token g = pipe_copy_guess(slot);
            bool drafted = false;
            if (g < 0 && pipe_drafts(slot)) {
                // the drafter's guess, drawn with the uniform the device sampler takes at the position before
                if (slot.pipe_draft_pending != pos) {
                    pipe_draft_launch(slot);
                }
                slot.pipe_draft_pending = -1;
                g = llama_pipe_draft_get(ctx_tgt, llama_sampler_pos_uniform(common_sampler_get(slot.smpl.get()), pos - 1), nullptr);
                drafted = g >= 0;
            }
            if (g < 0 && (int32_t) n_inflight < params_base.pipe_floor) {
                // a filler: almost always wrong, but it keeps the GPUs busy, so their clocks stay up
                g = slot.pipe_win.back().tok;
            }
            if (g < 0 || !pipe_submit(slot, g, pos)) {
                break;
            }
            slot.pipe_win.back().drafted = drafted;
            slot.pipe_n_guess++;
            pipe_draft_ahead(slot);
        }
    }

    // one sampled token of a pipelined slot, as post_decode() handles it for a batched one
    void pipe_step(server_slot & slot) {
        llama_memory_t mem = llama_get_memory(ctx_tgt);

        if (slot.pipe_win.empty()) {
            // start: the token sampled last goes in first; the guesses read the prompt and what came out so far
            const llama_tokens text = slot.prompt.tokens.get_text_tokens();
            slot.pipe_guesser.reset(text.data(), text.size());
            if (!pipe_submit(slot, slot.sampled, slot.prompt.tokens.pos_next())) {
                abort_all_slots("pipelined decoding failed");
                return;
            }
            slot.prompt.tokens.push_back(slot.sampled);
            slot.pipe_guesser.push(slot.sampled);
            pipe_draft_ahead(slot);
        }

        pipe_top_up(slot);

        const server_slot::pipe_entry cur = slot.pipe_win.front();
        slot.pipe_win.pop_front();
        if (pipe_drafts(slot)) {
            // its output is about to be sampled: its row is a committed one from now on
            llama_pipe_draft_commit(ctx_tgt, cur.tok, cur.pos, cur.pslot);
        }

        llama_token id;
        {
            scoped_timer timer(t_sampl, n_sampl);
            id = llama_pipe_get_sampled(ctx_tgt, cur.pslot);
        }
        const bool guessed = !slot.pipe_win.empty();
        const bool hit     = guessed && slot.pipe_win.front().tok == id;
        if (guessed && slot.pipe_win.front().drafted) {
            slot.pipe_n_draft++;
            slot.pipe_n_draft_hit += hit;
        }

        common_sampler_accept(slot.smpl.get(), id, true);

        const int64_t t_now = ggml_time_us();

        slot.stats.n_gen += 1;

        if (slot.stats.n_gen == 1) {
            slot.stats.update_prompt_last();
            slot.t_print_last = t_now;
            slot.n_gen_last = 0;
        }

        slot.stats.update_gen_last();

        const bool special = params_base.special ||
            slot.task->params.sampling.preserved_tokens.find(id) != slot.task->params.sampling.preserved_tokens.end();

        completion_token_output result;
        result.tok          = id;
        result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, special);
        result.prob         = 1.0f;

        if (!process_token(result, slot)) {
            // nothing after the token just sampled belongs to the output
            llama_memory_seq_rm(mem, slot.id, cur.pos + 1, -1);
            llama_pipe_drain(ctx_tgt);
            slot.pipe_win.clear();

            slot.print_timings();
            send_final_response(slot);
            slot.release();
            return;
        }

        slot.print_timings_tg();

        // the token switched the grammar on (a tool call starts): batched decoding takes over from it, with every
        // token checked against the grammar; the token itself is decoded there (slot.sampled)
        if (!slot.task->params.sampling.grammar.empty() && common_sampler_grammar_active(slot.smpl.get())) {
            llama_memory_seq_rm(mem, slot.id, cur.pos + 1, -1);
            llama_pipe_drain(ctx_tgt);
            slot.pipe_win.clear();
            return;
        }

        if (hit) {
            // the guess in flight is this token
            slot.pipe_n_hit++;
        } else {
            if (guessed) {
                // a wrong guess: it goes with every guess behind it
                slot.pipe_n_miss++;
                llama_memory_seq_rm(mem, slot.id, slot.pipe_win.front().pos, -1);
                slot.pipe_win.clear();
                slot.pipe_draft_pending = -1;
            }
            if (!pipe_submit(slot, id, cur.pos + 1)) {
                abort_all_slots("pipelined decoding failed");
                return;
            }
        }
        slot.prompt.tokens.push_back(id);
        slot.pipe_guesser.push(id);

        // keep the GPUs busy while the server loop turns
        pipe_top_up(slot);
    }

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // checkpoint copies to the host that finished in the background take their checkpoints' place
        if (!ckpt_spills.empty()) {
            for (auto & slot : slots) {
                ckpt_spill_land(slot, false);
            }
        }

        // check if all slots are idle
        // a realtime slot waiting for audio has no work, the next audio control task wakes the loop
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing() && !slot.audio_wait) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        // pipelined speculation: a slot that generates alone runs through the pipe slots, one sampled token per turn
        if (params_base.pipe_depth > 0) {
            server_slot * busy   = nullptr;
            int           n_busy = 0;
            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    busy = &slot;
                    n_busy++;
                }
            }
            const bool pipe = n_busy == 1 && pipe_eligible(*busy);
            for (auto & slot : slots) {
                if (!slot.pipe_win.empty() && !(pipe && &slot == busy)) {
                    pipe_leave(slot);
                }
            }
            if (pipe) {
                pipe_step(*busy);
                return;
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        // note: TTS slots bypass the shared batch entirely
        try {
            process_tts_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_tts_slots() failed: %s\n", e.what());
            abort_all_slots("process_tts_slots() failed: " + std::string(e.what()));
        }

        // so do the realtime conversations
        try {
            process_realtime_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_realtime_slots() failed: %s\n", e.what());
            abort_all_slots("process_realtime_slots() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());

            // Q771: the micro-batch taper for this batch: the request's "ubatch_taper", else the server's
            const std::string & taper_want = slot_batched->task && !slot_batched->task->params.ubatch_taper.empty()
                ? slot_batched->task->params.ubatch_taper : taper_default;
            if (taper_want != taper_cur) {
                if (!llama_set_ubatch_taper(ctx_tgt, taper_want.c_str())) {
                    SRV_WRN("bad ubatch_taper '%s' in a request - the server's (%s) instead\n", taper_want.c_str(), taper_default.c_str());
                    llama_set_ubatch_taper(ctx_tgt, taper_default.c_str());
                }
                taper_cur = taper_want;
            }
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch.render(off, n_tokens);
                bool ok = decode(n_batch, off);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }

        ckpt_incall_drop("its decode call failed");
    }

    // realtime conversations (SERVER_TASK_TYPE_REALTIME) drive their own decode loop, see server_realtime_slot
    void process_realtime_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || !slot.rt) {
                return;
            }
            if (!slot.rt->update()) {
                send_error(slot, slot.rt->error, slot.rt->error_type);
                slot.release();
                return;
            }
            slot.audio_wait = slot.rt->is_idle();
        });
    }

    void process_tts_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || slot.task->type != SERVER_TASK_TYPE_TTS) {
                return;
            }

            llama_set_embeddings(ctx_tgt, true);

            if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                const int32_t ret = slot.tts.ctx.step_prompt(llama_n_batch(ctx_tgt));
                if (ret < 0) {
                    send_error(slot, "TTS prompt processing failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (ret == 0) {
                    // done prompt, do sample and save h_state_prompt
                    slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                    common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
                    const float * h_embd = llama_get_embeddings_ith(ctx_tgt, -1);
                    slot.tts.h_state_prompt.assign(h_embd, h_embd + llama_model_n_embd(model_tgt));
                    slot.tts.h_state = slot.tts.h_state_prompt.data();
                    slot.state = SLOT_STATE_GENERATING;
                }
                return;
            }

            const int32_t n_predict = slot.n_predict_max > 0 ? slot.n_predict_max : 512;
            if (slot.tts.stop || slot.tts.n_decoded >= n_predict) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                // generation truly ends here: force out any sub-window remainder still buffered
                if (slot.tts.ctx.flush() != 0 || slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "failed to finalize TTS output", ERROR_TYPE_SERVER);
                } else {
                    send_tts_result(slot, sample_rate, data, data_len, true);
                }
                slot.release();
                return;
            }

            const float * h_state_next = nullptr;
            bool          stop         = false;
            if (slot.tts.ctx.step_gen(slot.tts.sampled, slot.tts.h_state, &h_state_next, &stop) != 0) {
                send_error(slot, "TTS generation failed", ERROR_TYPE_SERVER);
                slot.release();
                return;
            }
            if (h_state_next == nullptr) {
                // end-of-speech without a new frame (e.g. pocket-tts eos head)
                slot.tts.stop = true;
                return;
            }
            slot.tts.h_state = h_state_next;
            slot.tts.n_decoded++;
            slot.tts.stop = stop;

            if (!stop) {
                slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
            }

            if (slot.task->params.stream) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                if (slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "TTS streaming output failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (data_len > 0) {
                    send_tts_result(slot, sample_rate, data, data_len, false);
                }
            }
        });
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.task && (slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME)) {
                // TTS and realtime conversation slots drive their own decode loop, never enter the shared batch
                return;
            }
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();
        ckpt_incall_drop("its batch was not decoded");

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME) {
                return;
            }

            if (slot.audio_stream && !slot_audio_step(slot)) {
                return; // waiting for audio, or the stream ended
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        const bool spec_reject = slot.use_spec_rejection();

                        const bool spec_dump = server_spec_dump::on();
                        if (spec_dump) {
                            slot.spec_info.clear();
                            slot.spec_info.k_max      = server_spec_dump::get().k;
                            slot.spec_info.full_block = server_spec_dump::get().full_block;
                            slot.spec_gram            = common_sampler_grammar_active(slot.smpl.get());
                        }

                        const bool spec_adapt = server_spec_adapt::on();
                        slot.spec_draft_p.clear();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                            /* .info     = */ spec_dump ? &slot.spec_info : nullptr,
                            /* .result_p = */ spec_adapt ? &slot.spec_draft_p : nullptr,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            // LLAMA_SPEC_ADAPT: verify only the prefix of the draft that pays (or record it, with the switch off)
            if (server_spec_adapt::on()) {
                spec_adapt_choose(slot);
            }

            slot.stats.n_draft_tokens += draft.size();

            // LLAMA_SPEC_DRAFT_KEEP / LLAMA_SPEC_FORCE (server_spec_verify): the draft cut, the grammar's forced tokens
            // appended; the width stays within what the target's outputs and recurrent snapshots were sized for
            if (server_spec_verify::get().any_draft()) {
                const int32_t n_width = std::min({ common_speculative_n_max(spec.get()),
                        common_speculative_n_max(&params_base.speculative), slot.get_n_draft_max() });
                slot.stats.n_draft_tokens += spec_force_extend(slot, n_width);
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            checkpoint_gen_maybe(slot);
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                // C217: a batch with checkpoints inside its call holds one slot's prompt
                if (ckpt_incall_slot >= 0) {
                    return;
                }

                if (!slot.is_processing() || slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    SLT_DBG(slot, "%s", "__TEST_TAG_WAIT_PARENT_SLOT__\n");
                    return;
                }

                // no prompt, the rows come from the audio stream
                if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        // a decision reads its logits from the one batch of its prompt
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt) && slot.task->type != SERVER_TASK_TYPE_DECISION) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // the outputs of a decision are read from one batch
                            const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;
                            if (n_decision_first >= 0 && slot.task->n_tokens() - n_decision_first > n_batch) {
                                send_error(slot,
                                           string_format("the question and its options (%d tokens) are too large to process. "
                                                         "increase the batch size (current batch size: %d)",
                                                         slot.task->n_tokens() - n_decision_first, n_batch),
                                           ERROR_TYPE_INVALID_REQUEST);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // the children start from the shared prefix, do not go past it
                                if (slot.task->n_tokens_shared > 0) {
                                    n_past = std::min(n_past, slot.task->n_tokens_shared);
                                }

                                // the outputs of a decision are read from the batch, so its first position must be evaluated
                                if (n_decision_first >= 0) {
                                    n_past = std::min(n_past, n_decision_first);
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            // a saved prefix longer than what the slot shares with the prompt replaces the slot's
                            // cells now; one within the shared part only if the slot cannot reuse that far (below)
                            const server_prefix_store::entry * pfx = find_prefix(slot, n_decision_first);
                            if (pfx && (int32_t) pfx->tokens.size() > n_past) {
                                n_past = prefix_restore(slot, *pfx);
                                pfx = nullptr;
                            } else if (n_past > 0) {
                                slot.stats.cache_src = 1;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // number of cached tokens shared with the new prompt, before any checkpoint is restored
                            const int n_prefix = n_past;

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                            // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
                            const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);

                            // a state or a checkpoint is useful when its pos_min is below this threshold
                            const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & ckpt) {
                                            const auto & cur = *ckpt;
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            // a ring checkpoint (every position from pos_min to pos_max) past the
                                            // divergence still serves it when its window reaches back far enough: the
                                            // cells from pos_next on are removed before the batch (seq_rm below), and
                                            // the window before it is the one the prompt had then. A default one holds
                                            // only the last n_swa positions, whatever pos_min says
                                            if (cur.pos_max > pos_next && !(ckpt_trim && ckpt_ring_past && cur.ring)) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    const int64_t t_ckpt_start = ggml_time_us();

                                    if (!do_reset) {
                                        const auto & cur = **it;

                                        // restore the context checkpoint
                                        if (!cur.load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                                            !cur.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                            if (cur.id_task != -1) {
                                                GGML_ABORT("failed to restore context checkpoint\n");
                                            }
                                            // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                            SLT_WRN(slot, "%s", "failed to load context checkpoint restored from a slot file\n");
                                            do_reset = true;
                                        }
                                    }

                                    if (!do_reset) {
                                        const auto & cur = **it;

                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, cur.data_spec);

                                        // a sliding-window checkpoint holds every position up to pos_max with its window:
                                        // the prompt resumes right after it, as the batch after the checkpoint did when
                                        // it was taken (so a GPU rounds that batch the same way); stock resumes at
                                        // pos_max, which a recurrent state needs
                                        pos_next = std::min(pos_next, std::max(cur.pos_min + 1, cur.pos_max + (ckpt_trim && ckpt_resume_next ? 1 : 0)));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) cur.n_tokens);
                                        SLT_INF(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB, on the devices %.3f MiB, %.1f ms)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_past, (float) cur.size() / 1024 / 1024, (float) cur.size_dev / 1024 / 1024, (ggml_time_us() - t_ckpt_start) / 1000.0);

                                        slot.stats.cache_src     = 2;
                                        slot.stats.n_ckpt_pos    = cur.n_tokens;
                                        slot.stats.t_restore_us += ggml_time_us() - t_ckpt_start;
                                    }

                                    if (do_reset) {
                                        SLT_INF(slot, "forcing full prompt re-processing due to lack of cache data (shared %d tokens, no checkpoint below %d)\n",
                                                n_past, pos_min_thold);
                                        pos_next = 0;
                                        n_past = 0;
                                        slot.stats.cache_src = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = **it;
                                    if (cur.pos_max > pos_next || cur.n_tokens > n_prefix) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }

                            // the slot fell back to fewer cells than a saved prefix holds
                            if (pfx && n_past < (int32_t) pfx->tokens.size()) {
                                n_past = prefix_restore(slot, *pfx);
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // the staircase drafter follows the slot that started last, from its prompt on
                        if (pipe_draft_on) {
                            llama_pipe_draft_reset(ctx_tgt, slot.id, n_past);
                            pipe_draft_seq = slot.id;
                        }

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // shared prompt prefix: once it is processed, the children continue from it with their own prompt
                    bool wait_shared = false;
                    if (slot.task->n_tokens_shared > 0) {
                        const bool is_shared_done = slot.prompt.n_tokens() == slot.task->n_tokens_shared;
                        for (auto & other : slots) {
                            if (other.state != SLOT_STATE_WAIT_OTHER || other.task->id_parent != slot.task->id) {
                                continue;
                            }
                            if (is_shared_done) {
                                SLT_TRC(slot, " - copying shared prompt (%d tokens) to child %d\n", slot.prompt.n_tokens(), other.id);
                                slot.copy_prompt_to(other, params_base.checkpoint_share);
                                other.state = SLOT_STATE_STARTED;
                            } else {
                                wait_shared = true;
                            }
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    // the end-of-prompt checkpoint and the separate micro-batch of the last tokens it needs, unless
                    // LLAMA_SERVER_TAIL_CKPT_MIN spaces them (server_tail_ckpt_min)
                    const bool do_tail_checkpoint = do_checkpoint && (server_tail_ckpt_min() == 0 ||
                            slot.prompt.checkpoints.empty() ||
                            slot.task->n_tokens() >= slot.prompt.checkpoints.back()->n_tokens + server_tail_ckpt_min());

                    // TODO: do the same for all models, then remove process_mtmd_chunk()
                    if (use_mixed_batch() && !slot.can_split() && input_tokens.has_mtmd) {
                        if (!add_prompt_mixed(slot)) {
                            return;
                        }
                    }

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // --prefix-cache: a prompt processed from its first token stops at its first user message, where
                    // its shared prefix is saved (a cold slot with checkpoints stops there anyway)
                    int32_t first_user_pos = -1;
                    for (const auto & span : spans.spans) {
                        if (span.role == COMMON_CHAT_ROLE_USER) {
                            first_user_pos = (int32_t) span.pos;
                            break;
                        }
                    }
                    const bool prefix_break = prefix_store.enabled() && slot.stats.n_prompt_cached == 0 &&
                                              first_user_pos >= prefix_store.n_min;

                    const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;

                    // C217 (--ctx-checkpoints-incall): the batch may go on past a break for a checkpoint, in parts: each
                    // part is the call that would have followed without the switch, joined only where it fits whole, and
                    // the decode call cuts its micro-batches where those calls end, so every number stays the same
                    const bool do_checkpoint_all = do_checkpoint;
                    const bool incall_slot = n_tokens_prev == 0 && do_checkpoint_all && !input_tokens.has_mtmd && !prefix_break &&
                            !wait_shared && alora_scale <= 0.0f && slot.task->decision.order.empty() && !use_mixed_batch();
                    for (int part = 0; ; ++part) {
                        do_checkpoint = do_checkpoint_all;

                        const int32_t n_part_prev     = part == 0 ? (int32_t) n_tokens_prev : batch.size();
                        const int64_t n_part_tok      = slot.prompt.n_tokens();
                        const int64_t spread_pos_prev = slot.ckpt_spread_pos;

                        // --ctx-checkpoints-spread: this batch starts where the last one stopped for a checkpoint inside the
                        // prompt; the next such checkpoint is spread tokens after the last one, or after this batch's start
                        // where it takes one
                        const int64_t n_batch_start = slot.prompt.n_tokens();
                        const bool    spread_start  = n_batch_start == slot.ckpt_spread_pos;
                        int64_t ckpt_spread_ref = slot.prompt.checkpoints.empty() ? 0 : slot.prompt.checkpoints.back()->n_tokens;
                        if (spread_start || spans.is_user_start(n_batch_start)) {
                            ckpt_spread_ref = std::max(ckpt_spread_ref, n_batch_start);
                        }
                        slot.ckpt_spread_pos = -1;

                        // add prompt tokens for processing in the current batch
                        bool part_break      = false; // the loop stopped at a break below
                        bool part_ckpt_break = false; // at one for a checkpoint at the next batch's start
                        while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                            // get next token to process
                            llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                            if (cur_tok == LLAMA_TOKEN_NULL) {
                                part_break = true;
                                break; // end of text chunk
                            }

                            // stop at the end of the shared prefix, the children are started from this state
                            if (wait_shared && slot.prompt.n_tokens() == slot.task->n_tokens_shared) {
                                part_break = true;
                                break;
                            }

                            // the outputs of a decision are read from one batch, do not split them
                            if (slot.prompt.n_tokens() == n_decision_first && batch.size() + slot.task->n_tokens() - n_decision_first > n_batch) {
                                part_break = true;
                                break;
                            }

                            // if this is an alora request with pre-invocation
                            // tokens that are not cached, we need to stop filling
                            // this batch at those pre-invocation tokens.
                            if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                                SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                                part_break = true;
                                break;
                            }

                            // embedding requires all tokens in the batch to be output;
                            // MTP also wants logits at every prompt position so the
                            // streaming hook can mirror t_h_nextn into ctx_dft.
                            add_ok &= batch.add(slot.id,
                                cur_tok,
                                /* pos       = */ slot.prompt.tokens.pos_next(),
                                /* output    = */ slot.need_embd(),
                                /* is_prompt = */ true);
                            if (!slot.task->decision.order.empty()) {
                                batch.set_decision_order(batch.size() - 1, slot.task->decision.order[slot.prompt.n_tokens()]);
                            }
                            slot.prompt.tokens.push_back(cur_tok);

                            // break at the last user message, or at user messages at least min step past the last checkpoint
                            if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                                const auto pos = slot.prompt.n_tokens();
                                const auto & checkpoints = slot.prompt.checkpoints;

                                if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back()->n_tokens + params_base.checkpoint_min_step) {
                                    part_break = part_ckpt_break = true;
                                    break;
                                }
                            }

                            if (do_checkpoint && ckpt_spread_at(slot, slot.prompt.n_tokens(), ckpt_spread_ref)) {
                                slot.ckpt_spread_pos = slot.prompt.n_tokens();
                                part_break = part_ckpt_break = true;
                                break;
                            }

                            if (prefix_break && (int32_t) slot.prompt.n_tokens() == first_user_pos) {
                                part_break = true;
                                break;
                            }

                            // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                            // create checkpoints that many tokens before the end of the prompt:
                            //  - 4 + n_ubatch
                            //  - 4
                            // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                            // a sliding-window model needs only the second, taken with its whole ring of n_swa + n_ubatch cells:
                            // it serves every restore the first would (ckpt_trim); --checkpoint-tail2 takes both, as stock
                            if (do_tail_checkpoint) {
                                const bool tail2 = !ckpt_trim || params_base.checkpoint_tail2;
                                const int checkpoint_offsets[] = {tail2 ? 4 + n_ubatch : 4, 4};

                                bool should_break = false;
                                for (int offset : checkpoint_offsets) {
                                    const int n_last = std::min(n_batch, offset);
                                    if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                        should_break = true;
                                        break;
                                    }
                                }
                                if (should_break) {
                                    part_break = part_ckpt_break = true;
                                    break;
                                }
                            }
                        }

                        // C217: a later part that the batch's room cut short would have gone on in its own call: it is
                        // taken back, and the next batch starts where it started, as without the switch
                        if (part > 0 && !part_break && slot.prompt.n_tokens() < slot.task->n_tokens()) {
                            batch.truncate(n_part_prev);
                            slot.prompt.tokens.keep_first(n_part_tok);
                            slot.ckpt_spread_pos = spread_pos_prev;
                            break;
                        }
                        if (part > 0) {
                            ckpt_incall_slot = slot.id;
                            incall_cuts.push_back(n_part_prev);
                        }

                        // the number of tokens added to the batch for the current slot
                        const auto n_tokens_cur = batch.size() - n_part_prev;

                        const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                        const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                        const bool is_user_start = spans.is_user_start(n_tokens_start);
                        const bool is_last_user_message = n_tokens_start == last_user_pos;

                        // one tail checkpoint with the whole sliding-window ring instead of two (--checkpoint-tail2): it serves
                        // the restores of both
                        const bool tail1 = ckpt_trim && !params_base.checkpoint_tail2;

                        // spaced end-of-prompt checkpoints (LLAMA_SERVER_TAIL_CKPT_MIN): none this time; user messages and
                        // --ctx-checkpoints-spread keep theirs
                        if (!do_tail_checkpoint && near_prompt_end && !is_user_start && !spread_start) {
                            do_checkpoint = false;
                        }

                        // entire prompt has been processed
                        if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                            slot.state = SLOT_STATE_DONE_PROMPT;

                            GGML_ASSERT(batch.size() > 0);

                            // extract the logits only for the last token
                            batch.set_output(batch.size() - 1, true);

                            slot.stats.n_gen = 0;
                            slot.i_batch     = batch.size() - 1;

                            slot.init_sampler();

                            if (slot.pipe_draws) {
                                llama_sampler_set_pos(common_sampler_get(slot.smpl.get()), slot.prompt.tokens.pos_next() - 1);
                            }
                        } else {
                            // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                            // message or we are near the end of the prompt
                            // (one tail checkpoint: only the last batch, 4 tokens before the end, takes it)
                            // (--ctx-checkpoints-spread: or the last batch stopped for one inside the prompt)
                            if (!is_user_start && !spread_start && (!near_prompt_end || tail1)) {
                                do_checkpoint = false;
                            }
                        }

                        // a batch that took no tokens leaves the checkpoint inside the prompt to the next one
                        if (spread_start && n_tokens_cur == 0) {
                            slot.ckpt_spread_pos = n_batch_start;
                        }

                        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                        // nothing to checkpoint yet
                        // TODO: is this check needed?
                        if (do_checkpoint && part == 0 && pos_min < 0) {
                            do_checkpoint = false;
                        }

                        // do not checkpoint after mtmd chunks
                        do_checkpoint = do_checkpoint && !has_mtmd;

                        // no need to create checkpoints that are too close together, unless it's the last user message
                        do_checkpoint = do_checkpoint && (
                                slot.prompt.checkpoints.empty() ||
                                is_last_user_message || near_prompt_end || spread_start ||
                                n_tokens_start > slot.prompt.checkpoints.back()->n_tokens + params_base.checkpoint_min_step);
                        SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                        // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                        //       yet processed and therefore it is not part of the checkpoint.
                        if (do_checkpoint) {
                            if (part == 0) {
                                create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                            } else {
                                create_checkpoint_incall(slot, n_tokens_cur, n_part_prev);
                            }
                        }

                        if (prefix_break && !has_mtmd) {
                            prefix_maybe_save(slot, n_tokens_start, first_user_pos);
                        }

                        // C217: on into the next part when this one stopped for a checkpoint at the next one's start and a
                        // device store is free for it
                        if (!(incall_slot && ckpt_incall_on && ckpt_vram_n > 0 && part_ckpt_break && batch.size() < n_batch &&
                              slot.prompt.n_tokens() < slot.task->n_tokens() && ckpt_store_free(slot) >= 0)) {
                            break;
                        }
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // see https://github.com/ggml-org/llama.cpp/pull/29969
    // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
    bool use_mixed_batch() const {
        return !llama_get_memory(ctx_tgt) || !llama_get_causal_attn(ctx_tgt);
    }

    // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
    // the caller makes sure that it fits in the batch
    // returns false on error, the slot is then released
    bool add_prompt_mixed(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const auto n_tokens_prev = batch.size();

        while (slot.prompt.n_tokens() < slot.task->n_tokens()) {
            const auto cur_token_idx = slot.prompt.n_tokens();
            const llama_token cur_tok = input_tokens[cur_token_idx];

            if (cur_tok != LLAMA_TOKEN_NULL) {
                const bool add_ok = batch.add(slot.id,
                    cur_tok,
                    /* pos       = */ slot.prompt.tokens.pos_next(),
                    /* output    = */ slot.need_embd(),
                    /* is_prompt = */ true);
                GGML_ASSERT(add_ok);
                if (!slot.task->decision.order.empty()) {
                    batch.set_decision_order(batch.size() - 1, slot.task->decision.order[cur_token_idx]);
                }
                slot.prompt.tokens.push_back(cur_tok);
                continue;
            }

            const auto & chunk = input_tokens.find_chunk(cur_token_idx);

            float * embd = slot.mbatch ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            if (!embd) {
                // encode on the worker thread, so we can still handle metrics tasks
                int32_t res = 0;
                queue_tasks.yield_to_queue([&]() {
                    res = encode_mtmd_chunk(slot, slot.mbatch, cur_token_idx);
                });
                embd = res == 0 ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            }

            if (!embd || !add_mtmd_chunk(slot, chunk.get(), embd, batch)) {
                SLT_ERR(slot, "%s", "failed to process mtmd chunk\n");
                // the batch must not keep the entries of a released slot
                batch.truncate(n_tokens_prev);
                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }

            slot.prompt.tokens.push_back_placeholder(chunk.get());
        }

        return true;
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd()) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        // screened output head: only if every slot with a row in this batch samples in a way it is exact for
        if (screen_head_m > 0) {
            bool screen = true;
            for (const auto & slot : slots) {
                if (slot.task && slot.i_batch >= off && slot.i_batch < off + (int) batch.view.size()) {
                    // LLAMA_SCREEN_HEAD_GRAMMAR=1: a grammar keeps the screen until it can reject a token (after a lazy
                    // grammar's trigger); off, any grammar turns it off
                    screen = screen && common_sampling_screen_head_ok(slot.task->params.sampling, screen_head_m, nullptr,
                            !common_screen_head_grammar() || common_sampler_grammar_active(slot.smpl.get()));
                }
            }
            llama_set_screen_head(ctx_tgt, screen);
            if ((int) screen != screen_head_state) {
                SRV_INF("screened output head %s\n", screen ? "on" : "off");
                screen_head_state = screen;
            }
        }

        // the shortlist head with a bound (LLAMA_HEAD_BOUND): off for batches whose sampling would always fall back
        // to the full head (exact either way; this saves the shortlist's products)
        if (head_bound_on) {
            common_head_bound_need need;
            bool use = true;
            for (const auto & slot : slots) {
                if (slot.task && slot.i_batch >= off && slot.i_batch < off + (int) batch.view.size()) {
                    use = use && !slot.task->params.sampling.backend_sampling &&
                        common_sampling_head_bound_need(slot.task->params.sampling, need);
                }
            }
            llama_set_head_bound(ctx_tgt, use);
        }

        // Q771: the server waits after this call when it reads an output, and when a prompt batch stopped short of n_batch
        // (for a checkpoint, a user message or the prompt's end); a full prompt batch is followed by the next one at once
        llama_set_ubatch_taper_next(ctx_tgt, -1, (has_output || (int32_t) batch.view.size() < (int32_t) llama_n_batch(ctx_tgt)) ? 1 : 0);

        // C217: the checkpoints taken inside this call, and its micro-batch cuts where the calls it joins would end
        std::vector<uint32_t> incall_view_cuts;
        const bool incall = !ckpt_incall.empty() || !incall_cuts.empty();
        if (incall) {
            ckpt_incall_begin(off, batch.view.size(), incall_view_cuts);
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (incall) {
            llama_set_ubatch_cuts(ctx_tgt, nullptr, 0);
        }

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            // C217: the smaller calls are cut elsewhere
            ckpt_incall_drop("its decode call was retried with a smaller batch");

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch.view.size(), has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            // C217: the draft's pass is cut as the target's call is
            if (incall && ctx_dft != nullptr) {
                llama_set_ubatch_cuts(ctx_dft, incall_view_cuts.data(), incall_view_cuts.size());
            }

            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });

            if (incall && ctx_dft != nullptr) {
                llama_set_ubatch_cuts(ctx_dft, nullptr, 0);
            }

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // C217: the checkpoints this call took inside it
        if (incall) {
            for (auto & ic : ckpt_incall) {
                if (!ic.done && ic.i_batch > off && ic.i_batch < off + (int32_t) batch.view.size()) {
                    ckpt_incall_finish(ic, off);
                }
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);
                    // children with their own prompt are started at the end of the shared prefix
                    GGML_ASSERT(slot.task->n_tokens_shared == 0);

                    slot.copy_state_to(*child, params_base.checkpoint_share);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside the current sub-batch
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_DECISION) {
                    send_decision(slot, batch.view, slot.i_batch - off);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // batch indices of the draft tokens, used to get the token probabilities below
            auto spec_i_batch = std::move(slot.spec_i_batch);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                } else if (use_rejection && server_spec_verify::get().block && !slot.task->params.sampling.backend_sampling) {
                    // LLAMA_SPEC_BLOCK_VERIFY (server_spec_verify): the drafted block verified jointly
                    accepted = common_sampler_sample_and_accept_n_block(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                }

                GGML_ASSERT(accepted.size() >= 1);

                if (server_spec_dump::on() && synth_probs.empty() && !slot.spec_is_replay) {
                    spec_dump_step(slot, slot.ctx_tgt, spec_i_batch, accepted, use_rejection);
                }

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                // the recurrent snapshots are only guaranteed for batches of up to n_rs_seq + 1 tokens, so a longer draft uses its checkpoint
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        if (server_spec_verify::get().block || server_spec_verify::get().keep_draws) {
                            // the draws this step used stay used, or the next step would draw them again (Q808)
                            common_sampler_copy_keep_rng(smpl_save.get(), slot.smpl.get());
                        } else {
                            common_sampler_copy(smpl_save.get(), slot.smpl.get());
                        }

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            if (slot.spec_adapt_step.pending) {
                spec_adapt_finish(slot, (int32_t) n_accepted);
            }

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            if (slot.spec_force_from >= 0) {
                slot.spec_force_acc += std::max<int32_t>(0, (int32_t) n_accepted - slot.spec_force_from);
                slot.spec_force_from = -1;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    std::vector<std::string> tts_voices;
    if (impl->mctx) {
        for (int32_t i = 0; i < mtmd_gen_audio_get_info(impl->mctx).n_voices; i++) {
            tts_voices.push_back(mtmd_gen_audio_voice_name(impl->mctx, i));
        }
    }

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(impl->model_tgt, impl->has_cap_chat(), impl->has_cap_tts()),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* has_cap_chat            */ impl->has_cap_chat(),
        /* has_cap_tts             */ impl->has_cap_tts(),
        /* has_audio_stream        */ impl->mtmd_cap.audio_stream,
        /* tts_voices              */ std::move(tts_voices),
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type        */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens    */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train       */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp        */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params          */ llama_model_n_params(impl->model_tgt),
        /* model_size              */ llama_model_size(impl->model_tgt),
        /* model_ftype             */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_chat()) {
        res->error(format_error_response("this server does not support chat/completions", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & usage   = arr[0]["usage"];
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }

                // OAI counts generated tokens across all choices
                int64_t n_completion_tokens = 0;
                for (auto & item : arr) {
                    n_completion_tokens += item["usage"]["completion_tokens"].get<int64_t>();
                }
                usage["completion_tokens"] = n_completion_tokens;
                usage["total_tokens"]      = usage["prompt_tokens"].get<int64_t>() + n_completion_tokens;

                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };

    if (!params.server_connect_code.empty()) {
        props["connect_code"] = params.server_connect_code;
    }
    if (!meta.tts_voices.empty()) {
        props["voices"] = meta.tts_voices;
    }
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

// body of /tts and /v1/audio/speech; multipart form fields arrive as strings, the numeric and boolean params get their types back
static json parse_tts_body(const server_http_req & req) {
    json body = json::parse(req.body);
    for (const char * k : { "top_k", "n_predict", "max_tokens", "seed" }) {
        if (body.contains(k) && body[k].is_string()) {
            body[k] = std::stoll(body[k].get<std::string>());
        }
    }
    for (const char * k : { "top_p", "temperature", "repeat_penalty", "speed" }) {
        if (body.contains(k) && body[k].is_string()) {
            body[k] = std::stof(body[k].get<std::string>());
        }
    }
    if (body.contains("stream") && body["stream"].is_string()) {
        body["stream"] = body["stream"] == "true";
    }
    return body;
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = SERVER_CONTROL_ACTION_REASONING_END;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session);
    };

    // realtime transcription over websocket, modeled after the OpenAI realtime API
    // audio is PCM16 mono at the model sample rate, sent as base64 in input_audio_buffer.append events
    // input_audio_buffer.commit with "final": true ends the input, closing the socket also ends it
    this->ws_realtime = [this](const server_http_req & req, server_http_context::ws_conn & conn) {
        auto send_event = [&conn](json ev) {
            ev["event_id"] = "event_" + random_string();
            conn.send(safe_json_to_str(ev));
        };
        auto send_error_event = [&send_event](const std::string & message, const std::string & type) {
            send_event(json {
                {"type", "error"},
                {"error", {
                    {"type",    type},
                    {"code",    nullptr},
                    {"message", message},
                    {"param",   nullptr},
                }},
            });
        };

        // models that answer after the audio hold a conversation, see server-realtime.h
        // live transcription: Voxtral Realtime (no chat), or an "asr" model in a session opened with ?intent=transcription
        const bool live_asr = !ctx_server.asr_prompt.empty() && req.get_param("intent") == "transcription";
        if (ctx_server.mtmd_cap.audio_stream && ctx_server.mtmd_cap.chat && !live_asr) {
            server_realtime_conversation({queue_tasks, queue_results, *meta, params.sampling, ctx_server.mctx, ctx_server.mtmd_cap, ctx_server.vocab}, conn);
            return;
        }

        if (!meta->has_audio_stream) {
            send_error_event("The current model does not support realtime transcription.", "invalid_request_error");
            conn.close();
            return;
        }

        const std::string session_id = "sess_" + random_string();
        const int model_rate = mtmd_get_audio_sample_rate(ctx_server.mctx);
        json session_config = json {
            {"id",     session_id},
            {"object", "realtime.transcription_session"},
            {"type",   "transcription"},
            {"audio",  {{"input", {
                {"format",          {{"type", "audio/pcm"}, {"rate", model_rate}}},
                {"transcription",   {{"model", meta->model_name}}},
                {"turn_detection",  nullptr},
                {"noise_reduction", nullptr},
            }}}},
        };

        auto res = create_response();
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_TRANSCRIBE_RT);
            task.id = rd.get_new_id();
            task.params.sampling          = params.sampling;
            task.params.stream            = true;
            task.params.oaicompat_cmpl_id = session_id;
            task.params.oaicompat_model   = meta->model_name;
            rd.post_task(std::move(task));
        }

        // the slot answers once its audio stream is live, the session starts there
        {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result || result->is_error()) {
                if (result) {
                    send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                }
                conn.close();
                return;
            }
        }
        send_event(json {{"type", "session.created"}, {"session", session_config}});

        // one item per committed audio buffer, deltas are attributed to the current item
        std::mutex  mutex_item;
        std::string item_id = "item_" + random_string();
        std::string item_prev; // the committed item before item_id, empty for the first one
        std::string item_text;

        // called with mutex_item held
        auto send_completed = [&]() {
            send_event(json {
                {"type",          "conversation.item.input_audio_transcription.completed"},
                {"item_id",       item_id},
                {"content_index", 0},
                {"transcript",    item_text},
            });
            item_text.clear();
        };

        auto post_control = [&](server_control_action action, std::vector<float> && audio) {
            server_response_reader ctl(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id                     = ctl.get_new_id();
            task.params.control_action  = action;
            task.params.control_cmpl_id = session_id;
            task.params.control_audio   = std::move(audio);
            ctl.post_task(std::move(task));
            auto result = ctl.next([&conn]() { return !conn.is_open(); });
            auto * control = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
            return control && control->success;
        };

        // client events are read on this thread, the transcript is sent from the thread below
        std::atomic<bool> input_ended{false};
        std::thread reader([&]() {
            int                        input_rate = model_rate;
            mtmd_helper::resampler_ptr resampler; // input_rate -> model_rate
            std::string msg;
            while (conn.read(msg) != 0) {
                json ev;
                try {
                    ev = json::parse(msg);
                } catch (const std::exception &) {
                    send_error_event("invalid JSON", "invalid_request_error");
                    continue;
                }
                const std::string type = json_value(ev, "type", std::string());
                if (type == "session.update" || type == "transcription_session.update") {
                    const json session = json_value(ev, "session", json::object());
                    const json audio   = session.is_object() ? session.value("audio", json::object()) : json::object();
                    const json in      = audio.is_object() ? audio.value("input", json::object()) : json::object();
                    if (in.is_object() && in.contains("format") && in["format"].is_object()) {
                        const int rate = json_value(in["format"], "rate", input_rate);
                        if (rate < 8000 || rate > 192000) {
                            send_error_event("session.update: the input audio rate must be between 8000 and 192000 Hz", "invalid_request_error");
                            continue;
                        }
                        if (rate != input_rate) {
                            input_rate = rate;
                            resampler.reset();
                            session_config["audio"]["input"]["format"]["rate"] = rate;
                        }
                    }
                    send_event(json {{"type", "session.updated"}, {"session", session_config}});
                } else if (type == "input_audio_buffer.append") {
                    std::string pcm;
                    try {
                        pcm = base64::decode(json_value(ev, "audio", std::string()));
                    } catch (const std::exception &) {
                        send_error_event("invalid base64 audio", "invalid_request_error");
                        continue;
                    }
                    std::vector<float> samples(pcm.size() / 2);
                    for (size_t i = 0; i < samples.size(); i++) {
                        int16_t v;
                        memcpy(&v, pcm.data() + 2 * i, sizeof(v));
                        samples[i] = v / 32768.0f;
                    }
                    if (!resampler) {
                        resampler.reset(mtmd_helper_resampler_init(input_rate, model_rate));
                        if (!resampler) {
                            send_error_event("failed to initialize the audio resampler", "server_error");
                            break;
                        }
                    }
                    const float * out = nullptr;
                    const int64_t n   = mtmd_helper_resampler_process(resampler.get(), samples.data(), samples.size(), &out);
                    if (n < 0) {
                        send_error_event("failed to resample the audio", "server_error");
                        break;
                    }
                    if (n == 0) {
                        continue;
                    }
                    if (out != samples.data()) {
                        samples.assign(out, out + n);
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_APPEND, std::move(samples))) {
                        break;
                    }
                } else if (type == "input_audio_buffer.commit") {
                    // "final": true ends the input, the last item is completed when the stream drains
                    if (json_value(ev, "final", false)) {
                        input_ended = true;
                        post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
                        continue;
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_COMMIT, {})) {
                        break;
                    }
                    // completed is sent once the rows of the committed audio are decoded (audio_commit_drained below)
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_event(json {
                        {"type",             "input_audio_buffer.committed"},
                        {"previous_item_id", item_prev.empty() ? json(nullptr) : json(item_prev)},
                        {"item_id",          item_id},
                    });
                } else if (type == "input_audio_buffer.clear") {
                    send_event(json {{"type", "input_audio_buffer.cleared"}});
                } else {
                    send_error_event("unknown event type: " + type, "invalid_request_error");
                }
            }
            if (!input_ended.exchange(true)) {
                post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
            }
            // the result loop waits on the socket, nobody reads it from here on
            conn.close();
        });

        while (true) {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result) {
                break; // connection closed
            }
            if (result->is_error()) {
                send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                break;
            }
            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                if (partial->audio_commit_drained) {
                    // complete the committed item and open the next one
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_completed();
                    item_prev = item_id;
                    item_id   = "item_" + random_string();
                } else if (!partial->content.text.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_item);
                    item_text += partial->content.text;
                    send_event(json {
                        {"type",          "conversation.item.input_audio_transcription.delta"},
                        {"item_id",       item_id},
                        {"content_index", 0},
                        {"delta",         partial->content.text},
                    });
                }
            } else if (result->is_stop()) {
                std::lock_guard<std::mutex> lock(mutex_item);
                send_completed();
                break;
            }
        }

        input_ended = true;
        conn.close();
        reader.join();
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        return handle_rerank_impl(req);
    };

    this->post_tts = [this](const server_http_req & req) {
        return handle_tts_impl(req, parse_tts_body(req));
    };

    this->post_speech_oai = [this](const server_http_req & req) {
        json body = convert_speech_to_tts(parse_tts_body(req));
        SRV_DBG("%s\n", "Request converted: OpenAI Speech -> TTS");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        return handle_tts_impl(req, std::move(body));
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.prepare_state(decision.parse_state(body, files));
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.mctx, ctx_server.init_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[i_result++].get());
                GGML_ASSERT(result != nullptr);
                scores.push_back(result->scores);
                n_tokens += result->n_tokens;
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   {
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    // dispatch to the appropriate tokenizer based on the shape of the input
    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_rerank_impl(const server_http_req & req) {
    auto res = create_response();
    if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const json body = json::parse(req.body);

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    bool is_tei_format = body.contains("texts");

    // expand one rerank input item: plain string or {"text": str|[str], "image": url|data-uri|raw-base64|[urls]}
    auto parse_item = [&](const json & item, std::string & out_text, std::vector<raw_buffer> & out_files) {
        if (item.is_string()) {
            out_text = item.get<std::string>();
            return;
        }
        if (!item.is_object()) {
            throw std::invalid_argument("rerank input must be a string or an object of the form {\"text\": ..., \"image\": ...}");
        }
        if (item.contains("video") || item.contains("fps") || item.contains("max_frames")) {
            throw std::invalid_argument("video/fps/max_frames are not supported for rerank input");
        }
        json text = item.value("text", json());
        if (text.is_string()) {
            out_text += text.get<std::string>();
        } else if (text.is_array()) {
            for (const auto & t : text) {
                if (!t.is_string()) {
                    throw std::invalid_argument("\"text\" array elements must be strings");
                }
                out_text += t.get<std::string>();
            }
        } else if (!text.is_null()) {
            throw std::invalid_argument("\"text\" must be a string or an array of strings");
        }
        json image = item.value("image", json());
        auto add_image = [&](const std::string & url) {
            // marker is appended by format_prompt_rerank, one per decoded file
            try {
                handle_media(out_files, url, params.media_path);
            } catch (const std::runtime_error & e) {
                throw std::invalid_argument(e.what());
            }
        };
        if (image.is_string()) {
            add_image(image.get<std::string>());
        } else if (image.is_array()) {
            for (const auto & u : image) {
                if (!u.is_string()) {
                    throw std::invalid_argument("\"image\" array elements must be strings");
                }
                add_image(u.get<std::string>());
            }
        } else if (!image.is_null()) {
            throw std::invalid_argument("\"image\" must be a string or an array of strings");
        }
    };

    // Parse query
    std::string query_str;
    std::vector<raw_buffer> query_files;
    if (body.count("query") == 1) {
        try {
            parse_item(body.at("query"), query_str, query_files);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Resolve Jina/TEI format ambiguity
    json documents_json = json_value(body, "documents",
        json_value(body, "texts", json::array()));
    if (!documents_json.is_array() || documents_json.empty()) {
        res->error(format_error_response("\"documents\" must be a non-empty array of strings or objects", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Parse documents
    std::vector<std::string>          documents(documents_json.size());
    std::vector<std::vector<raw_buffer>> document_files(documents_json.size());
    for (size_t i = 0; i < documents_json.size(); i++) {
        try {
            parse_item(documents_json[i], documents[i], document_files[i]);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int top_n = json_value(body, "top_n", (int)documents.size());
    if (top_n < 0) {
        res->error(format_error_response("\"top_n\" must be >= 0", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        tasks.reserve(documents.size());
        for (size_t i = 0; i < documents.size(); i++) {
            auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query_str, documents[i], query_files, document_files[i], ctx_server.init_opt);
            server_task task = server_task(SERVER_TASK_TYPE_RERANK);
            task.id     = rd.get_new_id();
            task.tokens = std::move(tmp);
            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = format_response_rerank(
        body,
        meta->model_name,
        responses,
        is_tei_format,
        documents,
        top_n);

    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_tts_impl(const server_http_req & req, json body) {
    auto res = create_response();
    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_tts()) {
        res->error(format_error_response("this server does not support audio generation", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const auto info = mtmd_gen_audio_get_info(ctx_server.mctx);

    std::string prompt = json_value(body, "input", json_value(body, "prompt", std::string()));
    if (prompt.empty()) {
        res->error(format_error_response("\"input\" must be a non-empty string", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const std::string response_format = json_value(body, "response_format", std::string("wav"));
    const bool pcm16 = response_format == "pcm16";

    server_task task(SERVER_TASK_TYPE_TTS);
    server_schema::eval_tts_schema(params, task, body);
    const bool stream = task.params.stream;
    // a model with a "tts" chat template (e.g. LFM3-Audio) speaks the prompt rendered from it
    const std::string lang       = json_value(body, "lang", std::string());
    const std::string voice      = json_value(body, "voice", std::string());
    const std::string tts_prompt = common_chat_get_tts_prompt(meta->chat_params.tmpls.get(), prompt, lang, voice);
    // a model with predefined voices only speaks with one of them
    mtmd::bitmap_ptr voice_ref;
    if (!voice.empty() && info.n_voices > 0) {
        voice_ref.reset(mtmd_gen_audio_get_voice(ctx_server.mctx, voice.c_str()));
        if (!voice_ref) {
            res->error(format_error_response("unknown voice \"" + voice + "\", see \"voices\" in GET /props", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }
    task.tts_inp.set_prompt(tts_prompt.empty() ? prompt : tts_prompt);
    task.tts_inp.set_lang(lang);
    task.tts_inp.data.stream    = stream;
    task.tts_inp.data.out_type  = response_format == "pcm" || pcm16
        ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
        : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    // note: -1 is clamped to 0 by the sampler, it will disable the penalty
    task.params.sampling.penalty_last_n = task.params.n_predict > 0 ? task.params.n_predict : 512;
    task.params.sampling.seed = task.tts_inp.data.seed; // same seed for the codec/vocoder RNG
    if (task.tts_inp.data.top_k > 0) {
        task.params.sampling.top_k = task.tts_inp.data.top_k;
    }
    if (task.tts_inp.data.top_p > 0) {
        task.params.sampling.top_p = task.tts_inp.data.top_p;
    }
    if (task.tts_inp.data.temp >= 0) {
        task.params.sampling.temp = task.tts_inp.data.temp;
    }

    // speaker reference: either an uploaded form file ("speaker_ref") or a base64 JSON field ("speaker_ref_b64")
    const unsigned char * speaker_ref_data = nullptr;
    size_t speaker_ref_len  = 0;
    std::string speaker_ref_b64_decoded;

    auto speaker_ref_file = req.files.find("speaker_ref");
    if (speaker_ref_file != req.files.end()) {
        speaker_ref_data = speaker_ref_file->second.data.data();
        speaker_ref_len  = speaker_ref_file->second.data.size();
    } else {
        std::string speaker_ref_b64 = json_value(body, "speaker_ref_b64", std::string());
        if (!speaker_ref_b64.empty()) {
            try {
                speaker_ref_b64_decoded = base64::decode(speaker_ref_b64);
            } catch (const std::exception &) {
                res->error(format_error_response("\"speaker_ref_b64\" is not valid base64", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            speaker_ref_data = (const unsigned char *) speaker_ref_b64_decoded.data();
            speaker_ref_len  = speaker_ref_b64_decoded.size();
        }
    }

    if (speaker_ref_len > 0) {
        if (voice_ref) {
            res->error(format_error_response("\"voice\" and \"speaker_ref\" cannot be used together", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!mtmd_support_audio(ctx_server.mctx)) {
            res->error(format_error_response("this model has no voice encoder, \"speaker_ref\" cannot be used", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, speaker_ref_data, speaker_ref_len, false, ctx_server.init_opt);
        if (!wrapper.bitmap) {
            res->error(format_error_response("failed to decode \"speaker_ref\"", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(wrapper.bitmap));
    } else if (voice_ref) {
        task.tts_inp.set_speaker_ref(std::move(voice_ref));
    } else if (info.n_voices > 0) {
        // without a voice prompt the model produces near silence: the first voice is the default
        const char * first = mtmd_gen_audio_voice_name(ctx_server.mctx, 0);
        SRV_WRN("no voice given, speaking with \"%s\"\n", first);
        task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(mtmd_gen_audio_get_voice(ctx_server.mctx, first)));
    } else {
        SRV_WRN("%s", "no speaker reference provided, the model may behave randomly\n");
    }

    auto & rd = res->rd;
    task.id = rd.get_new_id();
    rd.post_task(std::move(task));

    // raw LE mono samples; audio/L16 would imply 16-bit big-endian (RFC 2586)
    const std::string pcm_type = "audio/pcm;rate=" + std::to_string(info.sample_rate);
    const std::string content_type = pcm16 ? pcm_type + ";encoding=signed-int;bits=16"
        : response_format == "pcm" ? pcm_type + ";encoding=float;bits=32"
        : "audio/wav";

    // the pipeline outputs float32 samples
    auto take_audio = [pcm16](server_task_result_tts * tts_res) {
        const std::string & audio = tts_res->audio;
        return pcm16 ? pcm_f32_to_s16((const float *) audio.data(), audio.size() / sizeof(float)) : std::move(tts_res->audio);
    };

    if (!stream) {
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
        GGML_ASSERT(tts_res != nullptr);
        res->status       = 200;
        res->content_type = content_type;
        res->data         = take_audio(tts_res);
        return res;
    } else {
        auto first_result = rd.next(req.should_stop);
        if (!first_result) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }
        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }
        auto * first_tts_res = dynamic_cast<server_task_result_tts *>(first_result.get());
        GGML_ASSERT(first_tts_res != nullptr);

        res->status       = 200;
        res->content_type = content_type;
        res->data          = take_audio(first_tts_res);
        bool is_done       = first_tts_res->final;

        res->set_next([res_this = res.get(), is_done, take_audio](std::string & output) mutable -> bool {
            // flush buffered audio before is_done - the first result can already be final
            if (!res_this->data.empty()) {
                output = std::move(res_this->data);
                res_this->data.clear();
                return true;
            }
            if (is_done) {
                return false;
            }
            if (res_this->should_stop()) {
                return false;
            }

            server_response_reader & rd = res_this->rd;
            if (!rd.has_next()) {
                return false;
            }
            auto result = rd.next([&res_this]() { return res_this->should_stop(); });
            if (!result || result->is_error()) {
                return false;
            }
            auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
            GGML_ASSERT(tts_res != nullptr);
            output  = take_audio(tts_res);
            is_done = tts_res->final;
            return true;
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
