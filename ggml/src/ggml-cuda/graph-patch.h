#pragma once

// GGML_CUDA_GRAPHS_PATCH=1 (C779; with GGML_CUDA_GRAPHS_FORCE=1): a batched graph whose properties changed in a few
// nodes is not captured again. Under pipeline parallelism a prompt micro-batch's graph changes every micro-batch: the
// KV views, the mask and the k-pool tables grow with the cache, and the scheduler rotates each split's inputs through
// its copies. Every other node keeps its properties, so its kernels keep their parameters. The full capture records
// which CUDA nodes each dispatch group (a node, or a fusion of several) made; a later evaluation re-dispatches only
// the groups near a changed node into a scratch capture, sets those CUDA nodes' parameters in the instantiated
// executables (cudaGraphExecKernelNodeSetParams and its memset and memcpy forms) and replays them.
//
// The parts that need no GPU live here: which nodes changed in structure (C826: by the kinds of their sources, not the
// tensor objects), which groups are re-dispatched, how a capture's chain of CUDA nodes is cut into the groups that made
// it, whether an evaluation is patched or captured in full, and which CUDA nodes are set (C826: a memset or memcpy whose
// executable holds its new parameters is left as it is). ggml-cuda.cu calls them; tests/test-cuda-graph-patch.cpp
// checks them on a CPU through a model of the capture, the patch and their fallbacks.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

// a node's sources (GGML_MAX_SRC, which ggml-cuda.cu checks) and, after them, its view source
#define GGML_CUDA_GRAPH_PATCH_N_SRC 11

// C826: what kind of tensor a node's source is. The structure check compares a node's op, type and flags and, for each
// source and its view source, whether it is there and its kind; what a source holds (its data address, shape and
// strides) is a property of the node, whose change re-dispatches the node's group. So the scheduler's rotating input
// copies (separate tensors of one kind at other addresses) change a node's parameters, not its structure. C779 compared
// the sources as tensor objects, and on a P106 705 of 768 captures fell back as structure changes (R136 part 1)
struct ggml_cuda_graph_patch_src_kind {
    int32_t      type   = -1; // -1: no source
    int32_t      op     = 0;
    int32_t      flags  = 0;
    const void * buffer = nullptr;
    const void * extra  = nullptr;
};

struct ggml_cuda_graph_patch_node_kind {
    int32_t op    = 0;
    int32_t type  = 0;
    int32_t flags = 0;
    ggml_cuda_graph_patch_src_kind src[GGML_CUDA_GRAPH_PATCH_N_SRC];
};

// what differs between two kinds of a node: FIELD_NONE when they are the same structure; else the field, and in src
// the source it is in (-1: the node itself; GGML_CUDA_GRAPH_PATCH_N_SRC - 1: the view source)
enum ggml_cuda_graph_patch_kind_field {
    GGML_CUDA_GRAPH_PATCH_FIELD_NONE = 0,
    GGML_CUDA_GRAPH_PATCH_FIELD_OP,
    GGML_CUDA_GRAPH_PATCH_FIELD_TYPE,     // for a source: also one there and one not (type -1)
    GGML_CUDA_GRAPH_PATCH_FIELD_FLAGS,
    GGML_CUDA_GRAPH_PATCH_FIELD_BUFFER,
    GGML_CUDA_GRAPH_PATCH_FIELD_EXTRA,
};

struct ggml_cuda_graph_patch_kind_diff {
    int field = GGML_CUDA_GRAPH_PATCH_FIELD_NONE;
    int src   = -1;
};

static inline ggml_cuda_graph_patch_kind_diff ggml_cuda_graph_patch_kind_compare(const ggml_cuda_graph_patch_node_kind & a,
        const ggml_cuda_graph_patch_node_kind & b) {
    ggml_cuda_graph_patch_kind_diff d;
    if (a.op != b.op) {
        d.field = GGML_CUDA_GRAPH_PATCH_FIELD_OP;
    } else if (a.type != b.type) {
        d.field = GGML_CUDA_GRAPH_PATCH_FIELD_TYPE;
    } else if (a.flags != b.flags) {
        d.field = GGML_CUDA_GRAPH_PATCH_FIELD_FLAGS;
    }
    for (int j = 0; d.field == GGML_CUDA_GRAPH_PATCH_FIELD_NONE && j < GGML_CUDA_GRAPH_PATCH_N_SRC; j++) {
        const ggml_cuda_graph_patch_src_kind & x = a.src[j];
        const ggml_cuda_graph_patch_src_kind & y = b.src[j];
        d.src = j;
        if (x.type != y.type) {
            d.field = GGML_CUDA_GRAPH_PATCH_FIELD_TYPE;
        } else if (x.type < 0) {
            continue; // neither is there
        } else if (x.op != y.op) {
            d.field = GGML_CUDA_GRAPH_PATCH_FIELD_OP;
        } else if (x.flags != y.flags) {
            d.field = GGML_CUDA_GRAPH_PATCH_FIELD_FLAGS;
        } else if (x.buffer != y.buffer) {
            d.field = GGML_CUDA_GRAPH_PATCH_FIELD_BUFFER;
        } else if (x.extra != y.extra) {
            d.field = GGML_CUDA_GRAPH_PATCH_FIELD_EXTRA;
        }
    }
    if (d.field == GGML_CUDA_GRAPH_PATCH_FIELD_NONE) {
        d.src = -1;
    }
    return d;
}

static inline const char * ggml_cuda_graph_patch_field_name(const int f) {
    switch (f) {
        case GGML_CUDA_GRAPH_PATCH_FIELD_NONE:   return "none";
        case GGML_CUDA_GRAPH_PATCH_FIELD_OP:     return "op";
        case GGML_CUDA_GRAPH_PATCH_FIELD_TYPE:   return "type";
        case GGML_CUDA_GRAPH_PATCH_FIELD_FLAGS:  return "flags";
        case GGML_CUDA_GRAPH_PATCH_FIELD_BUFFER: return "buffer";
        case GGML_CUDA_GRAPH_PATCH_FIELD_EXTRA:  return "extra";
        default:                                 return "?";
    }
}

// C826: how a CUDA node of a re-dispatched group reaches the scratch capture's parameters. A kernel is set in the
// executable. A memset or memcpy is left as it is when its new parameters equal those the executable holds, which its
// template node mirrors (every set of one in the executable is also made in the template); else it is set in both.
// CUDA updates a memcpy in an executable only between 1-D operands, and a memset only in its address and value, so a
// group dispatched again because a node near it changed would otherwise have its unchanged 2-D copies refused
enum ggml_cuda_graph_patch_action {
    GGML_CUDA_GRAPH_PATCH_SET_EXEC = 0, // a kernel: set in the executable
    GGML_CUDA_GRAPH_PATCH_KEEP,         // a memset or memcpy whose parameters the executable holds already
    GGML_CUDA_GRAPH_PATCH_SET_BOTH,     // a memset or memcpy with new parameters: set in the executable, then the template
};

static inline int ggml_cuda_graph_patch_action_for(const bool is_kernel, const bool same_as_held) {
    if (is_kernel) {
        return GGML_CUDA_GRAPH_PATCH_SET_EXEC;
    }
    return same_as_held ? GGML_CUDA_GRAPH_PATCH_KEEP : GGML_CUDA_GRAPH_PATCH_SET_BOTH;
}

// one dispatch of the evaluation loop, as the full capture recorded it
struct ggml_cuda_graph_patch_group {
    int      start     = 0; // its first ggml node
    int      end       = 0; // its last ggml node (start + the nodes a fusion skipped)
    uint32_t chunk     = 0; // the chunk (CUDA graph) it was captured into
    uint32_t first     = 0; // its first CUDA node in that chunk's chain
    uint32_t count     = 0; // its CUDA nodes
    size_t   pool_used = 0; // the stream's pool use before it was dispatched
};

// The groups to re-dispatch, in order. A group is re-dispatched when a node whose properties changed is one of its
// nodes, or lies within `window` nodes after its first node: a fusion decision reads the nodes after its first one, so
// a change there could make the loop fuse differently. node_group[i] is the group holding node i (-1: a node the loop
// skips). Returns the ggml nodes of the chosen groups in n_nodes_out.
static inline std::vector<int> ggml_cuda_graph_patch_dirty(const std::vector<ggml_cuda_graph_patch_group> & groups,
        const std::vector<int32_t> & node_group, const std::vector<uint8_t> & changed, const int window, int & n_nodes_out) {
    std::vector<uint8_t> mark(groups.size(), 0);
    const int n = (int) std::min(node_group.size(), changed.size());
    for (int i = 0; i < n; i++) {
        if (!changed[i]) {
            continue;
        }
        if (node_group[i] >= 0) {
            mark[node_group[i]] = 1;
        }
        for (int s = std::max(0, i - std::max(window, 0)); s < i; s++) {
            const int32_t g = node_group[s];
            if (g >= 0 && groups[g].start == s) {
                mark[g] = 1;
            }
        }
    }
    std::vector<int> dirty;
    n_nodes_out = 0;
    for (size_t g = 0; g < groups.size(); g++) {
        if (mark[g]) {
            dirty.push_back((int) g);
            n_nodes_out += groups[g].end - groups[g].start + 1;
        }
    }
    return dirty;
}

// A chunk's chain of CUDA nodes cut into the groups that made it. tails[k] is the chain's last node after the k-th
// group of the chunk was dispatched (null: no node yet), as the capture reported it; the k-th group made the nodes
// after the (k-1)-th tail up to its own. Fills first and count of groups[g0 .. g0 + tails.size()); false when a tail is
// not in the chain, the tails go backwards or nodes follow the last tail (the chain is then not the groups' alone).
template <typename Node>
static inline bool ggml_cuda_graph_patch_ranges(const std::vector<Node> & chain, const std::vector<Node> & tails,
        std::vector<ggml_cuda_graph_patch_group> & groups, const size_t g0) {
    if (g0 + tails.size() > groups.size()) {
        return false;
    }
    std::unordered_map<Node, int64_t> pos;
    pos.reserve(chain.size());
    for (size_t i = 0; i < chain.size(); i++) {
        pos[chain[i]] = (int64_t) i;
    }
    int64_t prev = -1;
    for (size_t k = 0; k < tails.size(); k++) {
        int64_t p = -1;
        if (tails[k] != Node{}) {
            const auto it = pos.find(tails[k]);
            if (it == pos.end()) {
                return false;
            }
            p = it->second;
        }
        if (p < prev) {
            return false;
        }
        groups[g0 + k].first = (uint32_t) (prev + 1);
        groups[g0 + k].count = (uint32_t) (p - prev);
        prev = p;
    }
    return prev + 1 == (int64_t) chain.size();
}

// Why an evaluation the force plan captures is not patched (ggml_cuda_graph_patch_reason_name); patched when NONE
enum ggml_cuda_graph_patch_reason {
    GGML_CUDA_GRAPH_PATCH_NONE = 0,
    GGML_CUDA_GRAPH_PATCH_NOT_READY,   // no full capture of this graph recorded its groups since it last ran otherwise
    GGML_CUDA_GRAPH_PATCH_STRUCTURE,   // the node count, or a node's op, type, flags or its sources' kinds changed
    GGML_CUDA_GRAPH_PATCH_SHARE,       // the groups to re-dispatch hold more than the limit's share of the nodes
    GGML_CUDA_GRAPH_PATCH_SKIP,        // a node the loop skipped would now run, or the reverse
    GGML_CUDA_GRAPH_PATCH_POOL,        // the pool's use before a group differs from the capture's
    GGML_CUDA_GRAPH_PATCH_FUSE,        // a re-dispatched group fused another number of nodes
    GGML_CUDA_GRAPH_PATCH_NODES,       // a re-dispatched group made other CUDA nodes (count or kind)
    GGML_CUDA_GRAPH_PATCH_API,         // CUDA refused a parameter update
    GGML_CUDA_GRAPH_PATCH_REASON_COUNT,
};

static inline const char * ggml_cuda_graph_patch_reason_name(const int r) {
    switch (r) {
        case GGML_CUDA_GRAPH_PATCH_NONE:      return "patched";
        case GGML_CUDA_GRAPH_PATCH_NOT_READY: return "not_ready";
        case GGML_CUDA_GRAPH_PATCH_STRUCTURE: return "structure";
        case GGML_CUDA_GRAPH_PATCH_SHARE:     return "share";
        case GGML_CUDA_GRAPH_PATCH_SKIP:      return "skip";
        case GGML_CUDA_GRAPH_PATCH_POOL:      return "pool";
        case GGML_CUDA_GRAPH_PATCH_FUSE:      return "fuse";
        case GGML_CUDA_GRAPH_PATCH_NODES:     return "nodes";
        case GGML_CUDA_GRAPH_PATCH_API:       return "api";
        default:                              return "?";
    }
}

// Whether an evaluation the force plan captures may be patched, before any group is re-dispatched. ready: the last
// full capture of this graph recorded its groups and every evaluation since was captured, patched or replayed (an eager
// one leaves the executables behind the properties it records); n_instances and n_chunks: the executables and the
// chunks that recording made; n_dirty_nodes of n_nodes in the groups to re-dispatch; max_share_pct: the limit.
static inline int ggml_cuda_graph_patch_precheck(const bool ready, const bool structure_changed, const size_t n_instances,
        const size_t n_chunks, const int n_dirty_nodes, const int n_nodes, const int max_share_pct) {
    if (!ready || n_instances == 0 || n_instances != n_chunks) {
        return GGML_CUDA_GRAPH_PATCH_NOT_READY;
    }
    if (structure_changed) {
        return GGML_CUDA_GRAPH_PATCH_STRUCTURE;
    }
    if ((int64_t) n_dirty_nodes * 100 > (int64_t) max_share_pct * n_nodes) {
        return GGML_CUDA_GRAPH_PATCH_SHARE;
    }
    return GGML_CUDA_GRAPH_PATCH_NONE;
}
