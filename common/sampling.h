#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// as above, but verifies by rejection sampling; draft_q holds the draft's candidates per token
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first = false);

// LLAMA_SPEC_REJECT_STOP=1 (Q1187): the rejection above stops where sample-and-match stops for the rows behind a
// token: after an accepted end-of-generation token that is not the last drafted one, and where the screened head meets
// a grammar that has become active. Block verification below (Q1263) ends its block at the first drafted
// end-of-generation token that is not the last drafted one, and takes no next token when it accepts that block whole.
// Off by default (upstream's path). Read once; the setter overrides it (tests)
bool common_spec_reject_stop();
void common_spec_reject_stop_set(bool on);

// as above, but verifies the drafted block jointly: block verification (Sun et al., "Block Verification Accelerates
// Speculative Decoding", arXiv:2403.10444, Algorithm 2). Exact in distribution like token-by-token rejection, and its
// expected accepted length is never shorter. draft_q holds the draft's candidates per token, as for the rejection
// above; the uniforms come from the sampler's own stream
std::vector<llama_token> common_sampler_sample_and_accept_n_block(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first = false);

// Block verification's arithmetic (arXiv:2403.10444, Algorithm 2 with its Equations 4, 5 and 9) for a block of g
// drafted tokens X_1 .. X_g:
//   pb[i]  the target's distribution after the first i drafted tokens, M_b(. | X^i), for i = 0 .. n, n <= g - 1
//          (positions past n follow a token the target cannot produce, and are never accepted)
//   q[i]   the draft's distribution of X_(i+1), M_s(. | X^i), for i = 0 .. g - 1
// gives the weights p_0 .. p_g (p_0 = 1, p_i = min(1, p_(i-1) M_b(X_i | X^(i-1)) / M_s(X_i | X^(i-1)))) and the
// acceptance probabilities h_1 .. h_g (h_g = p_g; for i < g, h_i = S_i / (S_i + 1 - p_i) with
// S_i = sum_x max(0, p_i M_b(x | X^i) - M_s(x | X^i))). The accepted length tau is the largest i with eta_i < h_i
// (0 if none), eta_i independent uniforms; the next token is then drawn from M_b(. | X^g) when tau = g, else from
// common_spec_block_residual at tau
struct common_spec_block_weights {
    std::vector<double> p; // p_0 .. p_g
    std::vector<double> h; // h_0 (unused, 0) .. h_g
};

common_spec_block_weights common_spec_block_compute(
        const std::vector<std::vector<llama_token_data>> & pb,
        const std::vector<std::vector<llama_token_data>> & q,
        const llama_tokens & draft);

// p_res(. | X^i) = max(0, p_i M_b(. | X^i) - M_s(. | X^i)), normalized; empty when it has no mass
std::vector<llama_token_data> common_spec_block_residual(
        const std::vector<llama_token_data> & pb, const std::vector<llama_token_data> & q, double p);

// a copy as common_sampler_copy makes it, except that dst keeps its random draws (its stream of uniforms and the
// chain's dist and xtc samplers), so that after a restore the draws already used stay used
void common_sampler_copy_keep_rng(const struct common_sampler * src, struct common_sampler * dst);

// the text the grammar forces after `prefix`, tokens not yet accepted: the longest prefix of every continuation the
// grammar then allows, up to max_chars characters (llama_sampler_grammar_forced_text). Empty when there is no grammar,
// it does not apply (a lazy grammar before its trigger, inside the reasoning), or it rejects a token of prefix; n_valid
// (if set) receives how many tokens of prefix it accepts
std::string common_sampler_grammar_forced_text(struct common_sampler * gsmpl, const llama_tokens & prefix, int32_t max_chars, size_t * n_valid = nullptr);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// how a sampler's tokens were sampled (one request in llama-server)
struct common_sampler_stats {
    int32_t n_sampled         = 0; // tokens sampled
    int32_t n_device          = 0; // ... drawn by a backend sampler (llama_set_sampler)
    int32_t n_constrained     = 0; // ... while the grammar could reject them (after a lazy grammar's trigger, outside
                                   //     the reasoning)
    int32_t n_rejected        = 0; // ... whose first draw the grammar rejected: resampled from the masked full row
    int32_t n_rejected_device = 0; // ... of these, first drawn by a backend sampler (the row fetched from the device)
    int32_t n_forced          = 0; // ... forced by the reasoning budget
    int32_t n_head_short      = 0; // ... from a shortlist head's row that the bound let stand (llama_head_bound_ith)
    int32_t n_head_full       = 0; // ... from such a row replaced by the full head's (llama_head_bound_full)
};

const common_sampler_stats & common_sampler_get_stats(const struct common_sampler * gsmpl);

// whether the grammar can reject the next token: there is one, it applies (outside the reasoning) and, if it is lazy,
// it has seen its trigger
bool common_sampler_grammar_active(const struct common_sampler * gsmpl);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

// add the strings that are a single token in the vocab to the preserved tokens
void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens);

// add grammar triggers, a trigger word that is a single token becomes a token trigger and must be a preserved token
void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
