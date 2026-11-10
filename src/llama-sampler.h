#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

// a chain whose first backend sampler that adds to the graph is a top-k (C691): its k, and in *i_next the index of the
// chain's sampler after it; 0 for any other chain
int32_t llama_sampler_chain_backend_top_k(const llama_sampler * sampler, size_t * i_next);
// the chain's backend samplers from the i_first'th on, as its backend_apply runs them all
void llama_sampler_chain_backend_apply_from(
        llama_sampler * sampler, ggml_context * ctx, ggml_cgraph * gf, llama_sampler_data * data, size_t i_first);

// the graph input tensors a backend sampler (or a chain's backend samplers) set in backend_set_input, as left by the
// last backend_apply; and pointing it at those of another graph (the same samplers, in the same order)
std::vector<std::vector<ggml_tensor *>> llama_sampler_backend_get_inputs(llama_sampler * sampler);
void llama_sampler_backend_set_inputs(llama_sampler * sampler, const std::vector<std::vector<ggml_tensor *>> & inputs);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
