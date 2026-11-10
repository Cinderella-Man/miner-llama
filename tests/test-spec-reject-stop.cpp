#include "sampling.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// LLAMA_SPEC_REJECT_STOP (Q1187): with it on, common_sampler_sample_and_accept_n_rejection stops after an accepted
// end-of-generation token that is not the last drafted one, as common_sampler_sample_and_accept_n does; with it off it
// returns what it returned before. Every check here compares the two from the same sampler state:
//   fixed   drafts the target accepts for certain (q(x) = p(x)/2), with the EOG first, second, last and nowhere
//   random  drafts drawn from a q that mixes the target's p with noise, so that rejections, residual draws and
//           accepted EOGs all occur: the switch's result is the default's, cut after the first accepted EOG before
//           the last drafted token, and the default's otherwise, token for token
// The test model's vocabulary has no EOG token, so its EOS is set by a metadata override.

static const llama_token TEST_EOG = 7;

static int n_failed = 0;

static std::string str(const std::vector<llama_token> & v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? ", " : "") + std::to_string(v[i]);
    }
    return s + "}";
}

static void check(bool ok, const std::string & what) {
    if (!ok) {
        LOG_ERR("FAIL: %s\n", what.c_str());
        n_failed++;
    }
}

static std::vector<llama_token> verify(common_sampler * smpl, llama_context * ctx, const llama_tokens & draft,
        const std::vector<std::vector<llama_token_data>> & draft_q, bool on) {
    common_spec_reject_stop_set(on);
    const std::vector<int> idxs(draft.size() + 1, -1); // every row reads the prompt's last output
    return common_sampler_sample_and_accept_n_rejection(smpl, ctx, idxs, draft, draft_q, false);
}

// the default's result as the switch must cut it: after the first accepted EOG before the last drafted token
static std::vector<llama_token> cut(const std::vector<llama_token> & r, const llama_tokens & draft, const llama_vocab * vocab) {
    for (size_t k = 0; k + 1 < draft.size() && k < r.size() && r[k] == draft[k]; ++k) {
        if (llama_vocab_is_eog(vocab, draft[k])) {
            return std::vector<llama_token>(r.begin(), r.begin() + k + 1);
        }
    }
    return r;
}

static int test_spec_reject_stop(common_params & params, int n_trials) {
    params.sampling.backend_sampling = false;

    llama_model_kv_override eos = {};
    eos.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    strncpy(eos.key, "tokenizer.ggml.eos_token_id", sizeof(eos.key) - 1);
    eos.val_i64 = TEST_EOG;
    params.kv_overrides.clear();
    params.kv_overrides.push_back(eos);
    params.kv_overrides.push_back({}); // the end of the list

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to initialize the model and context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (TEST_EOG >= n_vocab || !llama_vocab_is_eog(vocab, TEST_EOG)) {
        LOG_ERR("token %d is not an EOG token of this vocabulary (%d tokens)\n", TEST_EOG, n_vocab);
        return 1;
    }

    // the switch's default: off unless the environment sets it
    {
        const char * e = getenv("LLAMA_SPEC_REJECT_STOP");
        const bool want = e && atoi(e) != 0;
        check(common_spec_reject_stop() == want, std::string("the default is ") + (want ? "on" : "off") + " with LLAMA_SPEC_REJECT_STOP=" + (e ? e : "(unset)"));
    }

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };
    common_batch batch = common_batch_get_one(ctx, prompt);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("failed to decode the test prompt\n");
        return 1;
    }

    // a logit bias lifts the EOG to the top of the target's candidates, so drafts accept it often
    float bias = 0.0f;
    {
        const float * logits = llama_get_logits_ith(ctx, -1);
        float max = -INFINITY;
        for (int i = 0; i < n_vocab; ++i) {
            max = std::max(max, logits[i]);
        }
        bias = max - logits[TEST_EOG] + 0.5f;
    }

    common_params_sampling sampling;
    sampling.seed       = 18;
    sampling.samplers   = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };
    sampling.top_k      = 10;
    sampling.temp       = 1.0f;
    sampling.logit_bias = { { TEST_EOG, bias } };

    common_sampler_ptr smpl(common_sampler_init(model, sampling));
    if (!smpl) {
        LOG_ERR("failed to initialize the sampler\n");
        return 1;
    }

    // the target's candidates: the same for every row, since every row reads the same logits
    common_sampler_sample(smpl.get(), ctx, -1);
    const llama_token_data_array * cur_p = common_sampler_get_candidates(smpl.get(), true);
    const std::vector<llama_token_data> p(cur_p->data, cur_p->data + cur_p->size);

    std::vector<llama_token> other;
    bool has_eog = false;
    for (const auto & e : p) {
        has_eog = has_eog || e.id == TEST_EOG;
        if (e.id != TEST_EOG && e.p > 0.0f) {
            other.push_back(e.id);
        }
    }
    if (!has_eog || other.size() < 3) {
        LOG_ERR("the target's %zu candidates need the EOG and three others\n", p.size());
        return 1;
    }

    auto p_of = [&](llama_token id) {
        for (const auto & e : p) {
            if (e.id == id) {
                return e.p;
            }
        }
        return 0.0f;
    };

    // fixed drafts: q(x) = p(x)/2, so p(x) >= q(x) and every drafted token is accepted without a draw
    {
        const llama_token eog = TEST_EOG, a = other[0], b = other[1], c = other[2];

        struct fixed_case {
            const char * name;
            llama_tokens draft;
            size_t       n_on; // the switch's result size; the default's is draft.size() + 1
        };
        const std::vector<fixed_case> cases = {
            { "EOG first",   { eog, a, b }, 1 },
            { "EOG second",  { a, eog, b }, 2 },
            { "EOG last",    { a, b, eog }, 4 }, // a trailing EOG still takes the next row, as sample-and-match
            { "EOG alone",   { eog },       2 },
            { "no EOG",      { a, b, c },   4 },
            { "EOG twice",   { a, eog, eog, b }, 2 },
        };

        for (const auto & fc : cases) {
            std::vector<std::vector<llama_token_data>> q;
            for (const llama_token t : fc.draft) {
                q.push_back({ llama_token_data{ t, 0.0f, 0.5f*p_of(t) } });
            }

            common_sampler_ptr s_off(common_sampler_clone(smpl.get()));
            common_sampler_ptr s_on (common_sampler_clone(smpl.get()));

            const auto r_off = verify(s_off.get(), ctx, fc.draft, q, false);
            const auto r_on  = verify(s_on.get(),  ctx, fc.draft, q, true);

            const std::vector<llama_token> want_off_prefix(fc.draft.begin(), fc.draft.end());
            check(r_off.size() == fc.draft.size() + 1 && std::equal(want_off_prefix.begin(), want_off_prefix.end(), r_off.begin()),
                    std::string(fc.name) + ": off gave " + str(r_off) + ", want the draft " + str(fc.draft) + " and one more");
            check(r_on.size() == fc.n_on && std::equal(r_on.begin(), r_on.end(), r_off.begin()),
                    std::string(fc.name) + ": on gave " + str(r_on) + ", want the first " + std::to_string(fc.n_on) + " of " + str(r_off));
            check(r_on == cut(r_off, fc.draft, vocab), std::string(fc.name) + ": on is not the default's cut");

            LOG("%-10s draft %-16s off %-20s on %s\n", fc.name, str(fc.draft).c_str(), str(r_off).c_str(), str(r_on).c_str());
        }
    }

    // random drafts: q mixes p with noise over the same candidates; drafted tokens are drawn from q
    {
        std::mt19937 rng(18);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        const size_t n_draft = 4;

        int n_cut = 0;        // trials the switch cut
        int n_same = 0;       // trials it left as they were
        int n_rejected = 0;   // trials whose default result ends in a residual draw
        int n_eog_out = 0;    // EOG tokens the default output after its first accepted one (never output in a reply)

        for (int trial = 0; trial < n_trials; ++trial) {
            std::vector<llama_token_data> q(p.size());
            float r_sum = 0.0f;
            std::vector<float> r(p.size());
            for (size_t k = 0; k < p.size(); ++k) {
                r[k] = uni(rng);
                r_sum += r[k];
            }
            for (size_t k = 0; k < p.size(); ++k) {
                q[k] = { p[k].id, 0.0f, 0.5f*p[k].p + 0.5f*r[k]/r_sum };
            }

            llama_tokens draft;
            for (size_t i = 0; i < n_draft; ++i) {
                float u = uni(rng);
                llama_token t = q.back().id;
                for (const auto & e : q) {
                    u -= e.p;
                    if (u <= 0.0f) {
                        t = e.id;
                        break;
                    }
                }
                draft.push_back(t);
            }
            const std::vector<std::vector<llama_token_data>> draft_q(n_draft, q);

            // the default advances smpl, so each trial starts where the last one left off
            common_sampler_ptr s_on(common_sampler_clone(smpl.get()));
            const auto r_off = verify(smpl.get(), ctx, draft, draft_q, false);
            const auto r_on  = verify(s_on.get(), ctx, draft, draft_q, true);

            const auto want = cut(r_off, draft, vocab);
            if (r_on != want) {
                check(false, "trial " + std::to_string(trial) + ": draft " + str(draft) + ", off " + str(r_off) + ", on " + str(r_on) + ", want " + str(want));
            }

            if (want.size() < r_off.size()) {
                n_cut++;
                for (size_t k = want.size(); k < r_off.size(); ++k) {
                    n_eog_out += llama_vocab_is_eog(vocab, r_off[k]);
                }
            } else {
                n_same++;
            }
            n_rejected += r_off.size() <= n_draft;
        }

        LOG("random: %d trials, %d cut after an accepted EOG, %d unchanged, %d ended in a residual draw, %d EOGs behind the cut\n",
                n_trials, n_cut, n_same, n_rejected, n_eog_out);

        // the trials must reach both branches, or they show nothing
        check(n_cut >= n_trials/20,  "too few trials accepted an EOG before the last drafted token: " + std::to_string(n_cut));
        check(n_same >= n_trials/20, "too few trials left unchanged: " + std::to_string(n_same));
        check(n_rejected >= n_trials/20, "too few trials ended in a residual draw: " + std::to_string(n_rejected));
    }

    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 256;

    common_init();

    int n_trials = 4000;
    std::vector<char *> args = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-trials") == 0 && i + 1 < argc) {
            n_trials = std::max(1, atoi(argv[++i]));
        } else {
            args.push_back(argv[i]);
        }
    }
    args.push_back(nullptr);

    if (!common_params_parse((int) args.size() - 1, args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    const int result = test_spec_reject_stop(params, n_trials);
    LOG("\n%s\n", result == 0 ? "All tests passed." : "Some tests failed.");

    llama_backend_free();
    return result;
}
