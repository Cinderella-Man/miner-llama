#include "sampling.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

// LLAMA_SPEC_REJECT_STOP (Q1187): with it on, common_sampler_sample_and_accept_n_rejection stops after an accepted
// end-of-generation token that is not the last drafted one, as common_sampler_sample_and_accept_n does; with it off it
// returns what it returned before. Every check here compares the two from the same sampler state:
//   fixed   drafts the target accepts for certain (q(x) = p(x)/2), with the EOG first, second, last and nowhere
//   random  drafts drawn from a q that mixes the target's p with noise, so that rejections, residual draws and
//           accepted EOGs all occur: the switch's result is the default's, cut after the first accepted EOG before
//           the last drafted token, and the default's otherwise, token for token
// Block verification (common_sampler_sample_and_accept_n_block, Q1263) with it on verifies the draft cut after its
// first EOG before the last drafted token, and takes no next token when it accepts that cut draft whole:
//   fixed   the same drafts and results as the rejection's
//   random  the switch's result is the default's on the cut draft, without the next token, and the default's on the
//           whole draft when there is nothing to cut, token for token; where both end in a residual draw the two
//           samplers are left in the same state (the walk stopped at the cut)
// grammar: a grammar that only an EOG completes, the tools' grammar after a call: the draft {EOG, a, b}. With the
// switch off both paths sample the row after the EOG under the grammar, which rejects the main sampler's draw there
// unless it is the EOG; on, neither does
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

static std::vector<llama_token> verify_block(common_sampler * smpl, llama_context * ctx, const llama_tokens & draft,
        const std::vector<std::vector<llama_token_data>> & draft_q, bool on) {
    common_spec_reject_stop_set(on);
    const std::vector<int> idxs(draft.size() + 1, -1);
    return common_sampler_sample_and_accept_n_block(smpl, ctx, idxs, draft, draft_q, false);
}

using verify_fn = std::vector<llama_token> (*)(common_sampler *, llama_context *, const llama_tokens &,
        const std::vector<std::vector<llama_token_data>> &, bool);

// the draft as block verification cuts it with the switch on: after its first EOG before the last drafted token
static size_t draft_cut(const llama_tokens & draft, const llama_vocab * vocab) {
    for (size_t k = 0; k + 1 < draft.size(); ++k) {
        if (llama_vocab_is_eog(vocab, draft[k])) {
            return k + 1;
        }
    }
    return draft.size();
}

// whether two samplers are in the same state: from copies, the same block verification and eight samples after it
static bool same_state(common_sampler * a, common_sampler * b, llama_context * ctx, const llama_tokens & draft,
        const std::vector<std::vector<llama_token_data>> & draft_q) {
    common_sampler_ptr ca(common_sampler_clone(a));
    common_sampler_ptr cb(common_sampler_clone(b));
    auto ra = verify_block(ca.get(), ctx, draft, draft_q, false);
    auto rb = verify_block(cb.get(), ctx, draft, draft_q, false);
    for (int k = 0; k < 8; ++k) {
        ra.push_back(common_sampler_sample(ca.get(), ctx, -1));
        common_sampler_accept(ca.get(), ra.back(), true);
        rb.push_back(common_sampler_sample(cb.get(), ctx, -1));
        common_sampler_accept(cb.get(), rb.back(), true);
    }
    return ra == rb;
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

        // block verification: q(x) = p(x)/2 gives every position the weight 1, so the block is accepted whole
        const std::vector<std::pair<const char *, verify_fn>> paths = { { "rejection", verify }, { "block", verify_block } };

        for (const auto & path : paths) {
            for (const auto & fc : cases) {
                std::vector<std::vector<llama_token_data>> q;
                for (const llama_token t : fc.draft) {
                    q.push_back({ llama_token_data{ t, 0.0f, 0.5f*p_of(t) } });
                }

                common_sampler_ptr s_off(common_sampler_clone(smpl.get()));
                common_sampler_ptr s_on (common_sampler_clone(smpl.get()));

                const auto r_off = path.second(s_off.get(), ctx, fc.draft, q, false);
                const auto r_on  = path.second(s_on.get(),  ctx, fc.draft, q, true);

                const std::string name = std::string(path.first) + ", " + fc.name;

                const std::vector<llama_token> want_off_prefix(fc.draft.begin(), fc.draft.end());
                check(r_off.size() == fc.draft.size() + 1 && std::equal(want_off_prefix.begin(), want_off_prefix.end(), r_off.begin()),
                        name + ": off gave " + str(r_off) + ", want the draft " + str(fc.draft) + " and one more");
                check(r_on.size() == fc.n_on && std::equal(r_on.begin(), r_on.end(), r_off.begin()),
                        name + ": on gave " + str(r_on) + ", want the first " + std::to_string(fc.n_on) + " of " + str(r_off));
                check(r_on == cut(r_off, fc.draft, vocab), name + ": on is not the default's cut");

                LOG("%-9s %-10s draft %-16s off %-20s on %s\n", path.first, fc.name, str(fc.draft).c_str(), str(r_off).c_str(), str(r_on).c_str());
            }
        }
    }

    const size_t n_draft = 4;

    // a random draft: q mixes p with noise over the same candidates; drafted tokens are drawn from q
    auto random_draft = [&](std::mt19937 & rng, llama_tokens & draft, std::vector<std::vector<llama_token_data>> & draft_q) {
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);

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

        draft.clear();
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
        draft_q.assign(n_draft, q);
    };

    // random drafts, token-by-token rejection
    {
        std::mt19937 rng(18);

        int n_cut = 0;        // trials the switch cut
        int n_same = 0;       // trials it left as they were
        int n_rejected = 0;   // trials whose default result ends in a residual draw
        int n_eog_out = 0;    // EOG tokens the default output after its first accepted one (never output in a reply)

        for (int trial = 0; trial < n_trials; ++trial) {
            llama_tokens draft;
            std::vector<std::vector<llama_token_data>> draft_q;
            random_draft(rng, draft, draft_q);

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

    // random drafts, block verification: the switch on the whole draft against the default on the cut draft
    {
        std::mt19937 rng(18);

        common_sampler_ptr sb(common_sampler_clone(smpl.get()));

        int n_cut = 0;        // trials whose draft the switch cut
        int n_cut_whole = 0;  // ... and accepted whole: the reply ends at the drafted EOG
        int n_cut_resid = 0;  // ... and ended in a residual draw before it (the samplers' states compared)
        int n_same = 0;       // trials with nothing to cut
        int n_behind = 0;     // tokens the default output behind its first EOG (never output in a reply)
        int n_vis_off = 0;    // tokens up to the first EOG, the default's
        int n_vis_on = 0;     // ... the switch's

        // the default's results in order, FNV-1a: the same from a build without the block's cut (sb carries the
        // sampler's state from trial to trial, so the digest covers that too)
        uint64_t digest = 1469598103934665603ull;

        auto visible = [&](const std::vector<llama_token> & r) {
            for (size_t k = 0; k < r.size(); ++k) {
                if (llama_vocab_is_eog(vocab, r[k])) {
                    return k + 1;
                }
            }
            return r.size();
        };

        for (int trial = 0; trial < n_trials; ++trial) {
            llama_tokens draft;
            std::vector<std::vector<llama_token_data>> draft_q;
            random_draft(rng, draft, draft_q);

            const size_t g = draft_cut(draft, vocab);

            // the default advances sb, so each trial starts where the last one left off
            common_sampler_ptr s_on (common_sampler_clone(sb.get()));
            common_sampler_ptr s_cut(common_sampler_clone(sb.get()));
            const auto r_off = verify_block(sb.get(),    ctx, draft, draft_q, false);
            const auto r_on  = verify_block(s_on.get(),  ctx, draft, draft_q, true);

            std::vector<llama_token> want = r_off;
            if (g < draft.size()) {
                const llama_tokens dc(draft.begin(), draft.begin() + g);
                const std::vector<std::vector<llama_token_data>> qc(draft_q.begin(), draft_q.begin() + g);
                want = verify_block(s_cut.get(), ctx, dc, qc, false);
                n_cut++;
                if (want.size() == g + 1) {
                    want.pop_back(); // the default's next token after the cut draft, which the switch does not take
                    n_cut_whole++;
                } else {
                    n_cut_resid++;
                    if (!same_state(s_on.get(), s_cut.get(), ctx, draft, draft_q)) {
                        check(false, "block trial " + std::to_string(trial) + ": draft " + str(draft) + ", on " + str(r_on) + ": the sampler's state differs from the cut draft's");
                    }
                }
            } else {
                n_same++;
            }

            if (r_on != want) {
                check(false, "block trial " + std::to_string(trial) + ": draft " + str(draft) + ", off " + str(r_off) + ", on " + str(r_on) + ", want " + str(want));
            }

            for (const llama_token t : r_off) {
                digest = (digest ^ (uint64_t) (uint32_t) t) * 1099511628211ull;
            }
            digest = (digest ^ 0xffffffffull) * 1099511628211ull;

            n_behind  += (int) (r_off.size() - visible(r_off));
            n_vis_off += (int) visible(r_off);
            n_vis_on  += (int) visible(r_on);
        }

        LOG("block: %d trials, %d cut (%d accepted whole, %d ending in a residual draw before the EOG), %d with nothing to cut, "
                "%d tokens behind an EOG in the default's results; tokens up to the first EOG: off %d, on %d\n",
                n_trials, n_cut, n_cut_whole, n_cut_resid, n_same, n_behind, n_vis_off, n_vis_on);
        LOG("block off digest %016llx over %d trials\n", (unsigned long long) digest, n_trials);

        check(n_cut  >= n_trials/20, "too few block trials drafted an EOG before the last drafted token: " + std::to_string(n_cut));
        check(n_same >= n_trials/20, "too few block trials with nothing to cut: " + std::to_string(n_same));
        check(n_cut_whole >= n_trials/50, "too few cut block trials accepted whole: " + std::to_string(n_cut_whole));
        check(n_cut_resid >= n_trials/50, "too few cut block trials ending in a residual draw: " + std::to_string(n_cut_resid));
    }

    // a grammar that only an EOG completes: the row after the EOG is sampled under it with the switch off, not with it on
    {
        common_params_sampling sg = sampling;
        sg.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= \"\"");

        common_sampler_ptr sgm(common_sampler_init(model, sg));
        if (!sgm) {
            LOG_ERR("failed to initialize the sampler with the grammar\n");
            return 1;
        }

        const llama_tokens draft = { TEST_EOG, other[0], other[1] };
        std::vector<std::vector<llama_token_data>> q;
        for (const llama_token t : draft) {
            q.push_back({ llama_token_data{ t, 0.0f, 0.5f*p_of(t) } });
        }

        const int n_g = 50;

        const std::vector<std::pair<const char *, verify_fn>> paths = { { "rejection", verify }, { "block", verify_block } };
        for (const auto & path : paths) {
            int rows_off = 0, rows_on = 0; // rows sampled under the grammar
            int rej_off  = 0, rej_on  = 0; // ... whose first draw it rejected: drawn again from the whole row

            for (int trial = 0; trial < n_g; ++trial) {
                common_sampler_ptr s_off(common_sampler_clone(sgm.get()));
                common_sampler_ptr s_on (common_sampler_clone(sgm.get()));
                const auto st = common_sampler_get_stats(sgm.get());

                const auto r_off = path.second(s_off.get(), ctx, draft, q, false);
                const auto r_on  = path.second(s_on.get(),  ctx, draft, q, true);

                const std::string name = std::string("grammar, ") + path.first + ", trial " + std::to_string(trial);
                check(r_off == std::vector<llama_token>{ TEST_EOG, TEST_EOG }, name + ": off gave " + str(r_off) + ", want the EOG twice");
                check(r_on  == std::vector<llama_token>{ TEST_EOG }, name + ": on gave " + str(r_on) + ", want the EOG alone");

                rows_off += common_sampler_get_stats(s_off.get()).n_constrained - st.n_constrained;
                rows_on  += common_sampler_get_stats(s_on.get()).n_constrained  - st.n_constrained;
                rej_off  += common_sampler_get_stats(s_off.get()).n_rejected - st.n_rejected;
                rej_on   += common_sampler_get_stats(s_on.get()).n_rejected  - st.n_rejected;

                // the next trial's draws: one row sampled, nothing accepted
                common_sampler_sample(sgm.get(), ctx, -1);
            }

            LOG("grammar %-9s %d trials: rows sampled under the grammar off %d, on %d; first draws it rejected off %d, on %d\n",
                    path.first, n_g, rows_off, rows_on, rej_off, rej_on);

            check(rows_off == 2*n_g, std::string("grammar, ") + path.first + ": off sampled " + std::to_string(rows_off) + " rows, want 2 a trial");
            check(rows_on  == n_g,   std::string("grammar, ") + path.first + ": on sampled " + std::to_string(rows_on) + " rows, want 1 a trial");
            check(rej_on < rej_off,  std::string("grammar, ") + path.first + ": on's rejections " + std::to_string(rej_on) + " not below off's " + std::to_string(rej_off));
        }
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
