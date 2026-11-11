#pragma once

// GGML_CUDA_GRAPHS_FORCE=1 (C702; the idea is lukolszewski/llama.cpp-multigpu c8832a5f0): a batched graph is captured
// whenever its properties change, without the warm-up of two unchanged calls. Under pipeline parallelism a prompt
// micro-batch's graph is rebuilt every micro-batch (its KV views grow, its input copy rotates), so the warm-up never
// completes and a split is launched kernel by kernel: the card's launch queue fills at the third queued micro-batch
// (R42). Captured, a split reaches the card as one launch per graph chunk.
//
// The parts that need no GPU live here: which graphs the switch takes, and the decision for one evaluation of such a
// graph. ggml_backend_cuda_graph_compute (ggml-cuda.cu) calls both; tests/test-cuda-graph-force.cpp checks them on a
// CPU, the decision through a model of the whole warm-up and capture logic with long random runs of evaluations.

#include "ggml.h"

#include <cstdint>

// Whether the switch takes a graph: one that works on more than max_rows tokens (the vector kernels' limit,
// MMVQ_MAX_BATCH_SIZE), i.e. a prompt micro-batch or a draft's catch-up pass. The tokens are read where they are
// unambiguous: MUL_MAT_ID's activations (their third dimension), ROPE's input (its third) and FLASH_ATTN_EXT's queries
// (their second); a plain MUL_MAT's rows can be heads (MLA's absorbed queries of one token) and never count. Decode and
// verify graphs keep the warm-up, which already ends in replays and makes their lazily made state (the persistent MoE
// kernel's mapped host words, the attention kernels' counters) outside a capture. Never a graph with an ARGSORT that
// CUB sorts on its segmented path (over 1,024 columns, several rows): while the stream captures, argsort.cu takes
// another algorithm there, so equal keys could come out in another order than in an eager launch.
static inline bool ggml_cuda_graph_force_eligible(ggml_tensor * const * nodes, const int n_nodes, const int64_t max_rows) {
    bool batched = false;
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * node = nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT_ID:
                batched = batched || node->src[1]->ne[2] > max_rows;
                break;
            case GGML_OP_ROPE:
                batched = batched || node->src[0]->ne[2] > max_rows;
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                batched = batched || node->src[0]->ne[1] > max_rows;
                break;
            case GGML_OP_ARGSORT:
                if (node->src[0]->ne[0] > 1024 && ggml_nrows(node->src[0]) > 1) {
                    return false;
                }
                break;
            default:
                break;
        }
    }
    return batched;
}

struct ggml_cuda_graph_force_plan {
    bool use_graph; // captured or replayed (else launched eagerly)
    bool capture;   // captured (else replayed), when use_graph
};

// The decision for one evaluation of a graph the switch takes, after ggml_cuda_graph_update_required has recorded its
// properties. warmup_complete and capture_gen are the key's (ggml_cuda_graph), shared with the warm-up's branches that
// take the key's other graphs; context_warm is the context's; stale(gen) says whether sample mode's phase set of this
// evaluation was captured before generation gen.
//   - The context's first such graph runs eagerly, so that lazily made state (the stream's cuBLAS handle and its
//     workspace) is made outside a capture.
//   - A replay only from instances captured from these very properties: warmup_complete says the key's last
//     evaluation was captured or replayed. An eager evaluation (the warm-up's, or the one above) recorded its
//     properties but left the older instances, so after it a graph is captured again even if nothing changed since.
//   - A change counts as a completed warm-up: a new capture generation, so sample mode's other sets are stale.
template <typename Stale>
static inline ggml_cuda_graph_force_plan ggml_cuda_graph_force_next(bool & context_warm, bool & warmup_complete,
        uint64_t & capture_gen, const bool properties_changed, const bool instances_empty, Stale && stale) {
    const bool props_current = warmup_complete && !properties_changed;
    if (!context_warm) {
        context_warm    = true;
        warmup_complete = false;
        return { false, false };
    }
    if (!props_current) {
        warmup_complete = true;
        capture_gen++;
    }
    return { true, !props_current || instances_empty || stale(capture_gen) };
}
