#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include "../src/llama-ext.h" // staging API: llama_head_bound_ith / llama_head_bound_full (Q619)

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    common_sampler_stats stats;

    // the last token was a backend sampler's draw that the grammar rejected, resampled on the host
    bool last_resampled_device = false;

    // tokens the grammar has checked (for LLAMA_GRAMMAR_TEST_REJECT)
    int32_t n_checked = 0;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    // full: the whole logits row, also for an output that a backend sampler sampled (fetched from the device)
    void set_logits(struct llama_context * ctx, int idx, bool full = false) {
        const float *       sampled_probs  = full ? nullptr : llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = full ? nullptr : llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = full ? nullptr : llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = full ? llama_get_logits_ith_fetch(ctx, idx) : llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
            samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    // a grammar and a reasoning budget keep backend sampling: common_sampler_sample checks the backend sampler's token
    // on the host and resamples from the full row only if the grammar rejects it, which is what the host path does
    // (LLAMA_GRAMMAR_HOST_SAMPLING=1 turns backend sampling off for such samplers instead)
    const char * env_host = getenv("LLAMA_GRAMMAR_HOST_SAMPLING");
    if (env_host && atoi(env_host) != 0) {
        if (grmr && params.backend_sampling) {
            LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

            params.backend_sampling = false;
        }

        if (rbudget && params.backend_sampling) {
            LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

            params.backend_sampling = false;
        }
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(const struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    auto * result = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng     = */ gsmpl->rng,
    };
    result->stats = gsmpl->stats;
    return result;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->rng        = src->rng;
    dst->stats      = src->stats;
    dst->t_total_us = src->t_total_us;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

// a single-token check against the grammar, as the grammar sampler's apply does it for the full row
static bool common_sampler_grammar_accepts(struct common_sampler * gsmpl, llama_token id) {
    llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
    llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

    llama_sampler_apply(gsmpl->grmr, &single_token_data_array);

    return single_token_data_array.data[0].logit != -INFINITY;
}

// every sampler of the chain on the host, also those a backend sampler runs (llama_sampler_apply on a chain that is
// bound to a context skips them)
static void common_sampler_apply_chain_host(struct common_sampler * gsmpl, llama_token_data_array * cur_p) {
    for (int32_t i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        llama_sampler * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        if (smpl->iface->apply != nullptr) {
            llama_sampler_apply(smpl, cur_p);
        }
    }
}

// with several outputs per sequence a backend sampler's random draw is taken from the dist sampler's stream only when
// its token is accepted; a rejected token used one, which is taken now, so that a resample gets a draw of its own
static void common_sampler_commit_draw(struct common_sampler * gsmpl) {
    for (int32_t i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        llama_sampler * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        if (strcmp(llama_sampler_name(smpl), "dist") == 0) {
            llama_sampler_accept(smpl, LLAMA_TOKEN_NULL);
        }
    }
}

// LLAMA_GRAMMAR_TEST_REJECT=N, for tests: every Nth token the grammar checks counts as rejected, on the host path and
// on the backend path alike, so that the resample runs at a known rate
static bool common_sampler_test_reject(struct common_sampler * gsmpl) {
    static const int every = [] {
        const char * e = getenv("LLAMA_GRAMMAR_TEST_REJECT");
        return e ? atoi(e) : 0;
    }();
    return every > 0 && ++gsmpl->n_checked % every == 0;
}

// a row of the shortlist head's logits (Q619): it stands only if its bound leaves out every token this sampling could
// pick from cur_p as it is now (after the reasoning budget and, when it comes first, the grammar, both of which only
// remove candidates); otherwise the full head's logits replace it. Returns whether they did: set cur_p again.
static bool common_sampler_head_bound(struct common_sampler * gsmpl, struct llama_context * ctx, int idx) {
    const float bound = llama_head_bound_ith(ctx, idx);
    if (bound == INFINITY) {
        return false; // the full head's row
    }

    // a forced token does not depend on the logits
    if (common_reasoning_budget_get_forced(gsmpl->rbudget) != LLAMA_TOKEN_NULL) {
        return false;
    }

    common_head_bound_need need;
    bool ok = !std::isnan(bound) && common_sampling_head_bound_need(gsmpl->params, need);
    if (ok) {
        // the model's suppressed tokens are a -INFINITY logit bias: they may push down as many of the highest
        int32_t n_suppress = 0;
        llama_vocab_get_suppress_tokens(llama_model_get_vocab(llama_get_model(ctx)), &n_suppress);

        const int32_t k   = need.k + n_suppress;
        const float   thr = bound/need.temp;

        int32_t n_above = 0;
        for (size_t i = 0; i < gsmpl->cur_p.size && n_above < k; ++i) {
            n_above += gsmpl->cur_p.data[i].logit/need.temp + need.log_p > thr;
        }
        ok = n_above >= k;
    }

    if (ok) {
        gsmpl->stats.n_head_short++;
        return false;
    }
    if (!llama_head_bound_full(ctx, idx)) {
        GGML_ABORT("the full head failed for a shortlist row");
    }
    gsmpl->stats.n_head_full++;
    return true;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    gsmpl->last_resampled_device = false;
    gsmpl->stats.n_sampled++;
    if (common_sampler_grammar_active(gsmpl)) {
        gsmpl->stats.n_constrained++;
    }

    // A backend sampler has already sampled a token with the same chain, from the same random draw. The host path
    // below samples from the chain and then checks the one token against the grammar, resampling from the masked full
    // row only if the grammar rejects it; here the backend sampler's token takes the place of that first sample.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'\n", __func__, id);

            gsmpl->stats.n_device++;

            // the reasoning budget forcing a token leaves it as the only candidate on the host path
            const llama_token forced = common_reasoning_budget_get_forced(rbudget);
            if (forced != LLAMA_TOKEN_NULL) {
                gsmpl->stats.n_forced++;
                id = forced;
                gsmpl->cur.assign(1, llama_token_data{ forced, 0.0f, 1.0f });
                cur_p = { gsmpl->cur.data(), 1, 0, false };
            } else {
                for (size_t i = 0; i < cur_p.size; ++i) {
                    if (cur_p.data[i].id == id) {
                        cur_p.selected = i;
                        break;
                    }
                }
            }

            if (!grammar_should_apply(gsmpl)) {
                return id;
            }

            if (!grammar_first) {
                if (common_sampler_grammar_accepts(gsmpl, id) && !common_sampler_test_reject(gsmpl)) {
                    return id;
                }

                // rejected: resample as the host path does, from the full row
                gsmpl->stats.n_rejected++;
                gsmpl->stats.n_rejected_device++;
                gsmpl->last_resampled_device = true;

                common_sampler_commit_draw(gsmpl);
            }

            // the grammar first (grammar_first), or the resample: the full row on the host
            gsmpl->set_logits(ctx, idx, /* full = */ true);

            llama_sampler_apply(rbudget, &cur_p);
            llama_sampler_apply(grmr,    &cur_p);
            common_sampler_apply_chain_host(gsmpl, &cur_p);

            GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

            return cur_p.data[cur_p.selected].id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    // a shortlist head's row the bound can't keep: the same steps on the full head's row
    if (common_sampler_head_bound(gsmpl, ctx, idx)) {
        gsmpl->set_logits(ctx, idx);

        llama_sampler_apply(rbudget, &cur_p);

        if (grammar_first && grammar_should_apply(gsmpl)) {
            llama_sampler_apply(grmr, &cur_p);
        }
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    if (common_sampler_grammar_accepts(gsmpl, id) && !common_sampler_test_reject(gsmpl)) {
        return id;
    }

    gsmpl->stats.n_rejected++;

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    // the grammar may have masked the shortlist's highest tokens
    if (common_sampler_head_bound(gsmpl, ctx, idx)) {
        gsmpl->set_logits(ctx, idx);

        llama_sampler_apply(rbudget,  &cur_p);

        if (grammar_should_apply(gsmpl)) {
            llama_sampler_apply(grmr,  &cur_p);
        }
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // do not accept draft tokens after an EOG - they are not output but would stay in the context
        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }

        // a backend sampler drew the later rows' tokens before this resample took a draw of its own: stop here, and
        // those positions are sampled again from fresh draws
        if (gsmpl->last_resampled_device) {
            break;
        }

        // the screened output head computed this batch while the grammar could not reject a token; once it can, the
        // later rows need the full head (common_sampling_screen_head_ok)
        if (common_screen_head_grammar() && llama_get_screen_head(ctx) && common_sampler_grammar_active(gsmpl)) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// LLAMA_SPEC_REJECT_STOP (Q1187). Without it the drafted tokens behind the end of a turn are verified: they are never
// output, and with a grammar the row after the end fails it and is drawn again from the full vocabulary on the host.
// Sample-and-match's third stop, after a device resample, is left out: here that resample's token is output only when
// the step ends at its row, so the later rows' device draws are independent of every token output
static std::atomic<int> spec_reject_stop_override{-1};

bool common_spec_reject_stop() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_SPEC_REJECT_STOP");
        return e && atoi(e) != 0;
    }();
    const int o = spec_reject_stop_override.load(std::memory_order_relaxed);
    return o < 0 ? on : o != 0;
}

void common_spec_reject_stop_set(bool on) {
    spec_reject_stop_override.store(on ? 1 : 0, std::memory_order_relaxed);
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const bool stop = common_spec_reject_stop();

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);

            // as sample-and-match: no drafted token after an EOG, and the full head once the grammar can reject
            if (stop && llama_vocab_is_eog(vocab, draft[i]) && i + 1 < draft.size()) {
                break;
            }
            if (stop && common_screen_head_grammar() && llama_get_screen_head(ctx) && common_sampler_grammar_active(gsmpl)) {
                break;
            }

            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

//
// block verification (Q797): Sun et al., "Block Verification Accelerates Speculative Decoding", arXiv:2403.10444
//

// the draft's candidates sorted by token, for lookups while walking the target's (often the whole vocabulary)
static std::vector<llama_token_data> spec_block_by_id(const std::vector<llama_token_data> & q) {
    std::vector<llama_token_data> s = q;
    std::sort(s.begin(), s.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.id < b.id; });
    return s;
}

static double spec_block_q(const std::vector<llama_token_data> & q_by_id, llama_token id) {
    const auto it = std::lower_bound(q_by_id.begin(), q_by_id.end(), id,
            [](const llama_token_data & e, llama_token v) { return e.id < v; });
    return it != q_by_id.end() && it->id == id ? (double) it->p : 0.0;
}

// S = sum_x max(0, p M_b(x) - M_s(x)); a token outside M_b's candidates contributes max(0, -M_s(x)) = 0
static double spec_block_mass(const std::vector<llama_token_data> & pb, const std::vector<llama_token_data> & q_by_id, double p) {
    double s = 0.0;
    for (const auto & e : pb) {
        const double r = p * (double) e.p - spec_block_q(q_by_id, e.id);
        if (r > 0.0) {
            s += r;
        }
    }
    return s;
}

common_spec_block_weights common_spec_block_compute(
        const std::vector<std::vector<llama_token_data>> & pb,
        const std::vector<std::vector<llama_token_data>> & q,
        const llama_tokens & draft) {
    const size_t g = draft.size();
    GGML_ASSERT(q.size() == g && "q must have one entry per draft token");
    GGML_ASSERT(pb.size() <= g && "pb has at most one entry per draft token");

    common_spec_block_weights w;
    w.p.assign(g + 1, 0.0);
    w.h.assign(g + 1, 0.0);
    w.p[0] = 1.0;

    for (size_t i = 0; i < pb.size(); ++i) {
        const auto q_by_id = spec_block_by_id(q[i]);

        // Equation 5, i < g: the mass the target still has over the draft at position i, weighed by p_i, against what
        // the earlier positions left (1 - p_i). With p_i = 1 and no mass left (target and draft agree here) the limit
        // is taken: every later weight stays 1, so tau never stops at i
        if (i >= 1) {
            const double s   = spec_block_mass(pb[i], q_by_id, w.p[i]);
            const double den = s + 1.0 - w.p[i];
            w.h[i] = den > 1e-12 ? s / den : (w.p[i] > 0.0 ? 1.0 : 0.0);
        }

        // Equation 9
        const double px = prob_of(pb[i].data(), pb[i].size(), draft[i]);
        const double qx = spec_block_q(q_by_id, draft[i]);
        w.p[i + 1] = qx > 0.0 ? std::min(1.0, w.p[i] * px / qx) : 0.0;
    }

    // Equation 5, i = g
    w.h[g] = w.p[g];

    return w;
}

std::vector<llama_token_data> common_spec_block_residual(
        const std::vector<llama_token_data> & pb, const std::vector<llama_token_data> & q, double p) {
    const auto q_by_id = spec_block_by_id(q);

    std::vector<llama_token_data> res;
    double sum = 0.0;
    for (const auto & e : pb) {
        const double r = p * (double) e.p - spec_block_q(q_by_id, e.id);
        if (r > 0.0) {
            res.push_back({ e.id, 0.0f, (float) r });
            sum += r;
        }
    }
    if (sum <= 0.0) {
        res.clear();
        return res;
    }
    for (auto & e : res) {
        e.p = (float) (e.p / sum);
    }
    return res;
}

// the samplers' states and the history, without the candidate scratch (rebuilt by the next sample): what the walk
// below changes and the restore puts back
static common_sampler * spec_block_clone_state(const common_sampler * gsmpl) {
    auto * result = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ {},
        /* .cur_p   = */ { nullptr, 0, -1, false },
        /* .rng     = */ gsmpl->rng,
    };
    result->stats = gsmpl->stats;
    return result;
}

static void spec_block_restore_state(const common_sampler * src, common_sampler * dst) {
    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);
    dst->prev = src->prev;
}

// the chain's random samplers whose state is their stream of draws only (no history: accept leaves them as they are
// on the host)
static bool spec_is_draw_sampler(const llama_sampler * smpl) {
    const char * name = llama_sampler_name(smpl);
    return name && (strcmp(name, "dist") == 0 || strcmp(name, "xtc") == 0);
}

// the draws of src's chain into dst's (chains of the same samplers). A restore must not hand back draws the walk has
// used: with a grammar, which candidates a position has depends on its draw (the drawn token checked first, the masked
// row resampled on a rejection), so the decision depends on it, and a draw used again would tie a later token to it
static void spec_block_copy_draws(common_sampler * src, common_sampler * dst) {
    const int32_t n = llama_sampler_chain_n(src->chain);
    if (n != llama_sampler_chain_n(dst->chain)) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        llama_sampler * s = llama_sampler_chain_get(src->chain, i);
        if (spec_is_draw_sampler(s)) {
            llama_sampler_copy(s, llama_sampler_chain_get(dst->chain, i));
        }
    }
}

// a draw from a normalized candidate list
static llama_token spec_block_draw(const std::vector<llama_token_data> & dist, double u) {
    for (const auto & e : dist) {
        u -= e.p;
        if (u <= 0.0) {
            return e.id;
        }
    }
    return dist.back().id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    const size_t g = draft.size();

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // The decision needs the target's distribution at every drafted position, each after the drafted tokens before
    // it (penalties and the grammar follow the accepted text): walk the block accepting the drafted tokens, then put
    // the sampler back and accept only those the decision keeps
    std::unique_ptr<common_sampler, common_sampler_deleter> save(spec_block_clone_state(gsmpl));

    std::vector<std::vector<llama_token_data>> pb; // M_b(. | X^i), as far as the walk goes
    pb.reserve(g);

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    double p_walk = 1.0; // p_i along the walk
    for (size_t i = 0; i < g; ++i) {
        // leaves the target distribution in the candidate array
        common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);

        // as in the token-by-token rejection: a candidate the grammar rejects carries no probability, and what is
        // left is rescaled
        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        std::vector<llama_token_data> dist;
        dist.reserve(cur_p->size);
        double sum = 0.0;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float pk = masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
            if (pk > 0.0f) {
                dist.push_back({ cur_p->data[k].id, 0.0f, pk });
                sum += pk;
            }
        }
        if (sum > 0.0) {
            for (auto & e : dist) {
                e.p = (float) (e.p / sum);
            }
        }
        pb.push_back(std::move(dist));

        const double px = prob_of(pb.back().data(), pb.back().size(), draft[i]);
        const double qx = prob_of(draft_q[i].data(), draft_q[i].size(), draft[i]);
        p_walk = qx > 0.0 ? std::min(1.0, p_walk * px / qx) : 0.0;

        // past a token the target cannot produce no position is accepted; the last position needs no walk
        if (p_walk <= 0.0 || i + 1 == g) {
            break;
        }

        common_sampler_accept(gsmpl, draft[i], true);
    }

    const common_spec_block_weights w = common_spec_block_compute(pb, draft_q, draft);

    // tau = the largest i with eta_i < h_i: the uniforms are independent, so draw them from the last position down
    // and stop at the first that passes
    size_t tau = 0;
    for (size_t i = g; i >= 1; --i) {
        const double h = w.h[i];
        if (h <= 0.0) {
            continue;
        }
        if (h >= 1.0 || uni(gsmpl->rng) < h) {
            tau = i;
            break;
        }
    }

    // tau < g: the next token from the residual at tau (the walk reached tau: tau = 0, or h_tau > 0, so p_tau > 0);
    // without residual mass, which only rounding leaves, from the target's distribution there, its limit
    llama_token y = LLAMA_TOKEN_NULL;
    if (tau < g) {
        GGML_ASSERT(tau < pb.size());
        const auto res = common_spec_block_residual(pb[tau], draft_q[tau], w.p[tau]);
        const auto & from = res.empty() ? pb[tau] : res;
        GGML_ASSERT(!from.empty());
        y = spec_block_draw(from, uni(gsmpl->rng));
    }

    // put the sampler back, except its draws: the uniforms above and the chain's draws of the walk stay used. The
    // counters count output tokens, as the token-by-token functions' do (the walk's rows past tau are not output); the
    // final sample below counts its own
    {
        auto stats = gsmpl->stats;
        stats.n_sampled = save->stats.n_sampled + (int32_t) tau + (tau < g ? 1 : 0);
        spec_block_copy_draws(gsmpl, save.get());
        spec_block_restore_state(save.get(), gsmpl);
        gsmpl->stats = stats;
    }

    std::vector<llama_token> result;
    result.reserve(tau + 1);

    for (size_t i = 0; i < tau; ++i) {
        common_sampler_accept(gsmpl, draft[i], true);
        result.push_back(draft[i]);
    }

    if (tau == g) {
        y = common_sampler_sample(gsmpl, ctx, idxs[g], grammar_first);
    }

    common_sampler_accept(gsmpl, y, true);
    result.push_back(y);

    return result;
}

void common_sampler_copy_keep_rng(const struct common_sampler * src, struct common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }
    // dst's draws, kept aside while the copy overwrites them
    std::unique_ptr<common_sampler, common_sampler_deleter> draws(spec_block_clone_state(dst));
    common_sampler_copy(src, dst);
    spec_block_copy_draws(draws.get(), dst);
    dst->rng = draws->rng;
}

//
// the grammar's forced text (Q802)
//

std::string common_sampler_grammar_forced_text(struct common_sampler * gsmpl, const llama_tokens & prefix, int32_t max_chars, size_t * n_valid) {
    if (n_valid) {
        *n_valid = 0;
    }
    if (!gsmpl || !gsmpl->grmr || max_chars <= 0) {
        return {};
    }

    // a copy takes the prefix, as common_sampler_accept would (the reasoning budget decides whether the grammar applies)
    std::unique_ptr<common_sampler, common_sampler_deleter> tmp(spec_block_clone_state(gsmpl));

    for (size_t i = 0; i < prefix.size(); ++i) {
        if (grammar_should_apply(tmp.get()) && !common_sampler_grammar_accepts(tmp.get(), prefix[i])) {
            return {};
        }
        common_sampler_accept(tmp.get(), prefix[i], true);
        if (n_valid) {
            *n_valid = i + 1;
        }
    }

    if (!grammar_should_apply(tmp.get())) {
        return {};
    }

    // a token element of the grammar adds its whole piece: room for some beyond four bytes a character
    std::string buf((size_t) max_chars * 4 + 256, '\0');
    const int32_t n = llama_sampler_grammar_forced_text(tmp->grmr, buf.data(), (int32_t) buf.size(), max_chars);
    buf.resize(n > 0 ? (size_t) n : 0);
    return buf;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

const common_sampler_stats & common_sampler_get_stats(const struct common_sampler * gsmpl) {
    static const common_sampler_stats none;
    return gsmpl ? gsmpl->stats : none;
}

bool common_sampler_grammar_active(const struct common_sampler * gsmpl) {
    return gsmpl && gsmpl->grmr && grammar_should_apply(gsmpl) &&
        !llama_sampler_grammar_awaiting_trigger(gsmpl->grmr);
}

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
