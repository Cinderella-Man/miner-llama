#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// A draft's output head over a shortlist of tokens (Q592). The logits of the ids below k_low come from a plain product
// with the first k_low rows of the head (in a BPE vocabulary the low ids are the early, frequent merges), those of up
// to m_cap further ids (the special tokens, a frequency list, the tokens of the context) from a mul_mat_id over their
// rows, and every other token gets -INFINITY, so the samplers see the whole vocabulary. Drafts never change the
// output: a token outside the list can only lower the acceptance.
struct llama_head_shortlist {
    uint32_t n_vocab = 0;
    uint32_t k_low   = 0; // ids below k_low are always in the list
    uint32_t m_cap   = 0; // room for further ids; the CUDA expert path takes at most 65535 of them

    std::vector<int32_t> extra;  // the further ids (>= k_low), in the order they were added
    std::vector<uint8_t> member; // [n_vocab]: 1 if the id is in the list

    // device copies, in one buffer of the head's buffer type (alloc())
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * ids     = nullptr; // I32 [m_cap]: extra
    ggml_tensor * neg_inf = nullptr; // F32 [n_vocab - k_low, 1]: -INFINITY
    size_t        n_dev   = 0;       // entries of extra already on the device

    // the host side; false for sizes it can't serve (k_low must leave at least one id out)
    bool init(uint32_t n_vocab, uint32_t k_low, uint32_t m_cap);

    // the device tensors, in a buffer of type buft (the head's)
    bool alloc(ggml_backend_buffer_type_t buft);

    bool contains(int32_t id) const {
        return id >= 0 && (uint32_t) id < n_vocab && member[id];
    }

    // adds the ids that are not in the list yet, while there is room; returns how many it added
    uint32_t add(const int32_t * ids, size_t n);

    // copies the ids added since the last sync to the device
    void sync();

    uint32_t n_extra() const { return (uint32_t) extra.size(); }
    uint32_t size()    const { return k_low + n_extra(); }
};

// The shortlist's logits [n_vocab, 1] for one hidden state h [n_embd, 1] and the head output [n_embd, n_vocab]: the same
// values as ggml_mul_mat(output, h) for the ids in the list, -INFINITY for the others. n_extra_plan > 0 builds the further
// ids' part for that many ids instead of the list's (at most m_cap): a graph that is only planned, never computed
// (LLAMA_SCHED_PER_SIZE_RESERVE, C803), so that its plan holds every later list
ggml_tensor * llama_head_shortlist_logits(ggml_context * ctx, const llama_head_shortlist & sl, ggml_tensor * output, ggml_tensor * h,
        uint32_t n_extra_plan = 0);
