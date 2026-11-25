// llama-screen-check: checks the screened output head (llama_set_screen_head) against the full head on a real model.
//
//   llama-screen-check -m model-with-output_screen.gguf -f prompts.txt -n 128 [sampling flags] [other common flags]
//
// Every non-empty line of the prompt file is one prompt (escapes such as \n are processed). For each prompt:
//   1. greedy decoding with the full head, keeping every step's logits;
//   2. the same with the screened head, step by step against 1: exactly M finite logits, each bit-identical to the
//      full head's logit of that token, the full head's top 20 among them, and the same chosen token;
//   3. sampling with the given sampling flags and seed, full head and then screened head: the same tokens.
// Exits with 1 if any check fails.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

struct check_stats {
    int64_t steps          = 0;
    int64_t steps_m_finite = 0; // exactly M finite logits
    int64_t compared       = 0; // candidate logits compared with the full head
    int64_t mismatched     = 0; // ... not bit-identical
    int64_t top20_in       = 0; // the full head's top 20 among the candidates
    int64_t same_token     = 0;
    int64_t diverged       = 0; // prompts whose greedy or sampled tokens differ
};

// toks at positions pos0, pos0 + 1, ... of sequence 0 in one call (at most n_batch of them), logits for the last
static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
    for (size_t i = 0; i < toks.size(); ++i) {
        const llama_pos pos = pos0 + (llama_pos) i;
        const int32_t   idx = llama_batch_ext_add_token(batch.get(), 0, toks[i]);
        if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &pos) ||
                !llama_batch_ext_set_output_logits(batch.get(), idx, i + 1 == toks.size())) {
            LOG_ERR("%s: %zu tokens do not fit a batch of n_batch %u\n", __func__, toks.size(), llama_n_batch(ctx));
            return false;
        }
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static llama_token argmax(const float * v, int n) {
    return (llama_token) (std::max_element(v, v + n) - v);
}

// greedy or sampled generation; logits_out (if not null) gets every step's logits
static std::vector<llama_token> generate(llama_context * ctx, const std::vector<llama_token> & prompt, int n_predict,
        int n_vocab, common_sampler * smpl, std::vector<std::vector<float>> * logits_out) {
    std::vector<llama_token> out;
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!decode_tokens(ctx, prompt, 0)) {
        return out;
    }
    for (int t = 0; t < n_predict; ++t) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits_out) {
            logits_out->emplace_back(logits, logits + n_vocab);
        }
        llama_token tok;
        if (smpl) {
            tok = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, tok, true);
        } else {
            tok = argmax(logits, n_vocab);
        }
        out.push_back(tok);
        if (!decode_tokens(ctx, {tok}, (llama_pos) (prompt.size() + t))) {
            break;
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    common_log_flush_guard log_flush_guard;
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_predict = 128;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }
    const uint32_t m = llama_model_screen_head_m(model);
    if (m == 0) {
        LOG_ERR("the model has no output_screen tensor\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<std::string> prompts;
    {
        std::ifstream f(params.prompt_file);
        for (std::string line; std::getline(f, line);) {
            if (!line.empty()) {
                string_process_escapes(line); // \n etc. as in the CLI's prompts
                prompts.push_back(line);
            }
        }
        if (prompts.empty() && !params.prompt.empty()) {
            prompts.push_back(params.prompt);
        }
    }
    LOG_INF("%zu prompts, %d tokens each, %u candidates\n", prompts.size(), params.n_predict, m);

    check_stats st;
    for (size_t ip = 0; ip < prompts.size(); ++ip) {
        const std::vector<llama_token> prompt = common_tokenize(ctx, prompts[ip], true, true);

        // 1, 2: greedy, full head against screened head
        std::vector<std::vector<float>> full;
        llama_set_screen_head(ctx, false);
        const auto tok_full = generate(ctx, prompt, params.n_predict, n_vocab, nullptr, &full);

        std::vector<std::vector<float>> scr;
        llama_set_screen_head(ctx, true);
        const auto tok_scr = generate(ctx, prompt, params.n_predict, n_vocab, nullptr, &scr);

        const size_t n_steps = std::min(full.size(), scr.size());
        size_t first_diff = n_steps;
        for (size_t t = 0; t < n_steps; ++t) {
            const float * a = full[t].data();
            const float * b = scr[t].data();
            int64_t n_finite = 0;
            for (int j = 0; j < n_vocab; ++j) {
                if (std::isfinite(b[j])) {
                    ++n_finite;
                    ++st.compared;
                    st.mismatched += memcmp(&a[j], &b[j], sizeof(float)) != 0;
                }
            }
            std::vector<int> idx(n_vocab);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int x, int y) { return a[x] > a[y]; });
            bool in = true;
            for (int k = 0; k < 20; ++k) {
                in = in && std::isfinite(b[idx[k]]);
            }
            ++st.steps;
            st.steps_m_finite += n_finite == (int64_t) m;
            st.top20_in       += in;
            st.same_token     += tok_full[t] == tok_scr[t];
            if (tok_full[t] != tok_scr[t] && first_diff == n_steps) {
                first_diff = t;
            }
            if (tok_full[t] != tok_scr[t]) {
                break; // the contexts differ from here on
            }
        }
        st.diverged += first_diff != n_steps || tok_full.size() != tok_scr.size();

        // 3: the given sampling settings and seed, full head against screened head
        bool same_sampled = true;
        if (params.sampling.temp > 0.0f) {
            std::string why;
            if (!common_sampling_screen_head_ok(params.sampling, m, &why)) {
                LOG_WRN("sampling settings are not exact with the screened head (%s); sampled run skipped\n", why.c_str());
            } else {
                common_sampler * s_full = common_sampler_init(model, params.sampling);
                llama_set_screen_head(ctx, false);
                const auto smp_full = generate(ctx, prompt, params.n_predict, n_vocab, s_full, nullptr);
                common_sampler_free(s_full);

                common_sampler * s_scr = common_sampler_init(model, params.sampling);
                llama_set_screen_head(ctx, true);
                const auto smp_scr = generate(ctx, prompt, params.n_predict, n_vocab, s_scr, nullptr);
                common_sampler_free(s_scr);

                same_sampled = smp_full == smp_scr;
                st.diverged += !same_sampled;
            }
        }

        LOG_INF("prompt %zu: %zu prompt tokens, greedy %s (%zu steps), sampled %s\n", ip, prompt.size(),
                first_diff == n_steps ? "identical" : "DIFFERS", n_steps, same_sampled ? "identical" : "DIFFERS");
    }

    LOG_INF("\nsteps %lld: exactly %u finite logits %lld, full top-20 among them %lld, same token %lld\n",
            (long long) st.steps, m, (long long) st.steps_m_finite, (long long) st.top20_in, (long long) st.same_token);
    LOG_INF("candidate logits compared %lld, not bit-identical %lld; prompts that diverged %lld\n",
            (long long) st.compared, (long long) st.mismatched, (long long) st.diverged);
    const bool ok = st.mismatched == 0 && st.diverged == 0 && st.steps_m_finite == st.steps && st.top20_in == st.steps;
    LOG_INF("%s\n", ok ? "SCREEN CHECK PASSED" : "SCREEN CHECK FAILED");

    llama_backend_free();
    return ok ? 0 : 1;
}
