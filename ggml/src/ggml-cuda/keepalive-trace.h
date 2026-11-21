#pragma once

// GGML_CUDA_KEEPALIVE_TRACE (C878): a debug line for each key GGML_CUDA_KEEPALIVE_EAGER finds new to a card. The key
// (ggml_cuda_keepalive_shapes in ggml-cuda.cu) is one number made of every field it mixes: per node its op and type,
// sizes, strides and op parameters, its sources' types, sizes and strides, and at level 3 what the cache's length picks
// in the attention and the top-k. With the switch on, those fields are also recorded per node, and when a key is new to
// a card its record is compared, node by node, with the earlier key on the card that differs in the fewest nodes. The
// line names that key, how many nodes differ, the first one (its index, name, op and sizes) and the field that differs
// there with its value before and now; then up to four more differing nodes the same way.
//
//   GGML_CUDA_KEEPALIVE_TRACE  unset, empty, 0 or not a number: off (the key is computed as before, nothing is recorded)
//                              1: on, keeping up to 16 MiB of recorded fields a card
//                              N > 1: on, keeping up to N MiB a card (at most 4096)
//
// Past that budget the oldest keys keep only one hash per node: they can still be the nearest key and name the first
// differing node, but no longer its field. At most 4096 keys a card are kept. The line comes only from a key the
// keep-alive computes, so only with GGML_CUDA_KEEPALIVE on and GGML_CUDA_KEEPALIVE_EAGER at 1, 2 or 3.
//
// Everything here runs on the host; tests/test-cuda-keepalive.cpp checks it on a CPU.

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// GGML_CUDA_KEEPALIVE_TRACE's value (nullptr: unset): the MiB of recorded fields kept a card, 0 off
static inline int ggml_cuda_keepalive_parse_trace(const char * v) {
    if (v == nullptr || *v == '\0') {
        return 0;
    }
    char * end = nullptr;
    const long n = strtol(v, &end, 10);
    if (end == v || *end != '\0' || n <= 0) {
        return 0;
    }
    return n == 1 ? 16 : (int) std::min<long>(n, 4096);
}

// the marks ggml_cuda_keepalive_shapes mixes at level 3 for a size along the cache's length and for a packed stride
static constexpr uint64_t GGML_CUDA_KA_MARK_DEPTH  = 0xd3b7d3b7d3b7d3b7ull;
static constexpr uint64_t GGML_CUDA_KA_MARK_PACKED = 0x9ac4ed9ac4ed9ac4ull;
// the prefixes it mixes the attention's and the top-k's picks with
static constexpr uint64_t GGML_CUDA_KA_MARK_FA     = 0xfa00000000000000ull;
static constexpr uint64_t GGML_CUDA_KA_MARK_TOPK   = 0x7c00000000000000ull;

// what a field is: its kind, its slot (0 the node, j + 1 its source j) and an index (a dimension or an op parameter word)
enum ggml_cuda_ka_field_kind {
    GGML_CUDA_KA_F_OP      = 0, // the node's op | type << 16
    GGML_CUDA_KA_F_NE      = 1, // a size
    GGML_CUDA_KA_F_NB      = 2, // a stride
    GGML_CUDA_KA_F_PARAM32 = 3, // an op parameter as a 32-bit word (the ops whose sizes along the cache's length level 3 marks)
    GGML_CUDA_KA_F_PARAM64 = 4, // an op parameter as a 64-bit word (every other op)
    GGML_CUDA_KA_F_SRC     = 5, // a source: (j + 1) | its type << 16
    GGML_CUDA_KA_F_FA      = 6, // what the cache's length picks in the attention (ggml_cuda_flash_attn_ext_length_choice)
    GGML_CUDA_KA_F_TOPK    = 7, // the top-k's path (ggml_cuda_top_k_path_of)
};

static constexpr uint16_t ggml_cuda_ka_field(const int kind, const int slot, const int idx) {
    return (uint16_t) ((kind << 10) | ((slot & 31) << 5) | (idx & 31));
}
static inline int ggml_cuda_ka_field_kind_of(const uint16_t code) { return code >> 10; }
static inline int ggml_cuda_ka_field_slot_of(const uint16_t code) { return (code >> 5) & 31; }
static inline int ggml_cuda_ka_field_idx_of (const uint16_t code) { return code & 31; }

// the fields of one evaluation's key, in the order they are mixed, grouped by the node they belong to
struct ggml_cuda_ka_trace_rec {
    int64_t               n_nodes = 0; // the graph's nodes, views and no-ops included (mixed first, as the seed)
    std::vector<int32_t>  node;        // per recorded node: its index in the graph
    std::vector<uint32_t> first;       // per recorded node: its first field; after end(), one more entry
    std::vector<uint16_t> code;
    std::vector<uint64_t> val;

    void begin_node(const int32_t i) {
        node.push_back(i);
        first.push_back((uint32_t) code.size());
    }
    void put(const uint16_t c, const uint64_t v) {
        code.push_back(c);
        val.push_back(v);
    }
    void end() {
        first.push_back((uint32_t) code.size());
    }
    size_t n() const {
        return node.size();
    }
};

// a key the card has seen: its record, or only a hash per node once the budget dropped its fields
struct ggml_cuda_ka_trace_entry {
    uint64_t              key     = 0;
    uint64_t              seq     = 0; // its number among the card's new keys, 1 the first
    int64_t               n_nodes = 0;
    std::vector<int32_t>  node;
    std::vector<uint64_t> hash;
    std::vector<uint32_t> first;       // empty once dropped
    std::vector<uint16_t> code;
    std::vector<uint64_t> val;

    bool has_fields() const {
        return !first.empty();
    }
    size_t field_bytes() const {
        return first.size()*sizeof(uint32_t) + code.size()*sizeof(uint16_t) + val.size()*sizeof(uint64_t);
    }
    void drop_fields() {
        std::vector<uint32_t>().swap(first);
        std::vector<uint16_t>().swap(code);
        std::vector<uint64_t>().swap(val);
    }
};

// one card's keys
struct ggml_cuda_ka_trace_card {
    std::unordered_set<uint64_t>         seen;
    std::deque<ggml_cuda_ka_trace_entry> entries;
    size_t                               field_bytes = 0;
    uint64_t                             n_new       = 0;
};

// one differing node: its position among the recorded nodes, its index in the new graph, and the first field that
// differs there (known when the earlier key's fields are kept and the node exists in both)
struct ggml_cuda_ka_trace_node_diff {
    size_t   pos         = 0;
    int32_t  index_now   = -1;
    bool     in_both     = true;  // false: past the end of one of the two records
    bool     field_known = false;
    bool     same_field  = true;  // false: the two field lists differ there in what the field is (other sources, op)
    uint16_t code_now    = 0;
    uint16_t code_before = 0;
    uint64_t val_now     = 0;
    uint64_t val_before  = 0;
    bool     have_now    = true;  // false: the node has fewer fields now than before, the first missing one is shown
    bool     have_before = true;
};

// what a new key is set against
struct ggml_cuda_ka_trace_diff {
    bool     fresh        = false; // the key is new to the card
    uint64_t key          = 0;
    uint64_t seq          = 0;
    int64_t  n_nodes      = 0;
    size_t   n_rec        = 0;     // recorded nodes now
    bool     has_near     = false;
    uint64_t near_key     = 0;
    uint64_t near_seq     = 0;
    int64_t  near_n_nodes = 0;
    size_t   near_n_rec   = 0;
    bool     near_fields  = false;
    size_t   n_diff       = 0;     // nodes that differ, position by position, plus the difference in their count
    std::vector<ggml_cuda_ka_trace_node_diff> nodes; // the first few, in order
};

static inline uint64_t ggml_cuda_ka_trace_mix(uint64_t h, const uint64_t v) {
    h ^= v;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 31;
    return h;
}

// the key as ggml_cuda_keepalive_shapes mixes it, from a record alone: equal to the key when the record holds every
// field the key mixed, in its order
static inline uint64_t ggml_cuda_ka_trace_key_of(const ggml_cuda_ka_trace_rec & rec) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t) rec.n_nodes;
    for (const uint64_t v : rec.val) {
        h = ggml_cuda_ka_trace_mix(h, v);
    }
    return h;
}

// the first field that differs at a node present in both records
static inline void ggml_cuda_ka_trace_first_field(const ggml_cuda_ka_trace_rec & now, const ggml_cuda_ka_trace_entry & before,
                                                  const size_t pos, ggml_cuda_ka_trace_node_diff & nd) {
    const uint32_t a0 = now.first[pos],    a1 = now.first[pos + 1];
    const uint32_t b0 = before.first[pos], b1 = before.first[pos + 1];
    const uint32_t na = a1 - a0, nb = b1 - b0;
    for (uint32_t k = 0; k < std::max(na, nb); ++k) {
        if (k >= na || k >= nb) {
            nd.field_known = true;
            nd.have_now    = k < na;
            nd.have_before = k < nb;
            nd.code_now    = k < na ? now.code[a0 + k] : 0;
            nd.val_now     = k < na ? now.val[a0 + k] : 0;
            nd.code_before = k < nb ? before.code[b0 + k] : 0;
            nd.val_before  = k < nb ? before.val[b0 + k] : 0;
            nd.same_field  = false;
            return;
        }
        const uint16_t ca = now.code[a0 + k], cb = before.code[b0 + k];
        const uint64_t va = now.val[a0 + k],  vb = before.val[b0 + k];
        if (ca != cb || va != vb) {
            nd.field_known = true;
            nd.code_now    = ca;
            nd.code_before = cb;
            nd.val_now     = va;
            nd.val_before  = vb;
            nd.same_field  = ca == cb;
            return;
        }
    }
}

// the key's record on a card: when the key is new to it, compared with the nearest earlier key and kept. budget: the
// bytes of fields the card keeps (older keys keep only their hashes past it); max_keys: the keys it keeps; max_nodes:
// how many differing nodes the result lists
static inline ggml_cuda_ka_trace_diff ggml_cuda_ka_trace_add(ggml_cuda_ka_trace_card & card, const uint64_t key,
        ggml_cuda_ka_trace_rec && rec, const size_t budget, const size_t max_keys = 4096, const size_t max_nodes = 5) {
    ggml_cuda_ka_trace_diff d;
    d.key = key;
    if (!card.seen.insert(key).second) {
        return d;
    }
    d.fresh   = true;
    d.seq     = ++card.n_new;
    d.n_nodes = rec.n_nodes;
    d.n_rec   = rec.n();
    if (rec.first.size() != rec.n() + 1) {
        rec.end();
    }

    std::vector<uint64_t> hash(rec.n());
    for (size_t p = 0; p < rec.n(); ++p) {
        uint64_t h = 0x243f6a8885a308d3ull;
        for (uint32_t k = rec.first[p]; k < rec.first[p + 1]; ++k) {
            h = ggml_cuda_ka_trace_mix(h, ((uint64_t) rec.code[k] << 48) ^ rec.val[k]);
            h = ggml_cuda_ka_trace_mix(h, rec.val[k]);
        }
        hash[p] = h;
    }

    // the nearest earlier key: the fewest differing nodes, then one whose fields are kept, then the latest
    const ggml_cuda_ka_trace_entry * nearest = nullptr;
    size_t best = SIZE_MAX;
    for (auto it = card.entries.rbegin(); it != card.entries.rend(); ++it) {
        const ggml_cuda_ka_trace_entry & e = *it;
        const size_t n_min = std::min(e.hash.size(), hash.size());
        size_t n = std::max(e.hash.size(), hash.size()) - n_min;
        for (size_t p = 0; p < n_min && n <= best; ++p) {
            n += e.hash[p] != hash[p];
        }
        if (n < best || (n == best && nearest != nullptr && !nearest->has_fields() && e.has_fields())) {
            best = n;
            nearest = &e;
        }
    }

    if (nearest != nullptr) {
        d.has_near     = true;
        d.near_key     = nearest->key;
        d.near_seq     = nearest->seq;
        d.near_n_nodes = nearest->n_nodes;
        d.near_n_rec   = nearest->hash.size();
        d.near_fields  = nearest->has_fields();
        d.n_diff       = best;
        const size_t n_max = std::max(nearest->hash.size(), hash.size());
        for (size_t p = 0; p < n_max && d.nodes.size() < max_nodes; ++p) {
            const bool in_both = p < nearest->hash.size() && p < hash.size();
            if (in_both && nearest->hash[p] == hash[p]) {
                continue;
            }
            ggml_cuda_ka_trace_node_diff nd;
            nd.pos       = p;
            nd.index_now = p < rec.n() ? rec.node[p] : -1;
            nd.in_both   = in_both;
            if (in_both && nearest->has_fields()) {
                ggml_cuda_ka_trace_first_field(rec, *nearest, p, nd);
            }
            d.nodes.push_back(nd);
        }
    }

    ggml_cuda_ka_trace_entry e;
    e.key     = key;
    e.seq     = d.seq;
    e.n_nodes = rec.n_nodes;
    e.node    = std::move(rec.node);
    e.hash    = std::move(hash);
    e.first   = std::move(rec.first);
    e.code    = std::move(rec.code);
    e.val     = std::move(rec.val);
    card.field_bytes += e.field_bytes();
    card.entries.push_back(std::move(e));

    while (card.entries.size() > std::max<size_t>(max_keys, 1)) {
        card.field_bytes -= card.entries.front().field_bytes();
        card.entries.pop_front();
    }
    for (size_t i = 0; card.field_bytes > budget && i + 1 < card.entries.size(); ++i) {
        ggml_cuda_ka_trace_entry & old = card.entries[i];
        if (old.has_fields()) {
            card.field_bytes -= old.field_bytes();
            old.drop_fields();
        }
    }
    return d;
}

// a field's name, such as "src[1].ne[1]" or "op_params[4]"
static inline std::string ggml_cuda_ka_trace_field_name(const uint16_t code) {
    const int kind = ggml_cuda_ka_field_kind_of(code);
    const int slot = ggml_cuda_ka_field_slot_of(code);
    const int idx  = ggml_cuda_ka_field_idx_of(code);
    char where[24] = "";
    if (slot > 0) {
        snprintf(where, sizeof(where), "src[%d].", slot - 1);
    }
    char buf[64];
    switch (kind) {
        case GGML_CUDA_KA_F_OP:      snprintf(buf, sizeof(buf), "op and type"); break;
        case GGML_CUDA_KA_F_NE:      snprintf(buf, sizeof(buf), "%sne[%d]", where, idx); break;
        case GGML_CUDA_KA_F_NB:      snprintf(buf, sizeof(buf), "%snb[%d]", where, idx); break;
        case GGML_CUDA_KA_F_PARAM32: snprintf(buf, sizeof(buf), "op_params[%d]", idx); break;
        case GGML_CUDA_KA_F_PARAM64: snprintf(buf, sizeof(buf), "op_params 64-bit word %d", idx); break;
        case GGML_CUDA_KA_F_SRC:     snprintf(buf, sizeof(buf), "src[%d] type", slot - 1); break;
        case GGML_CUDA_KA_F_FA:      snprintf(buf, sizeof(buf), "the attention's pick by the cache's length"); break;
        case GGML_CUDA_KA_F_TOPK:    snprintf(buf, sizeof(buf), "the top-k's path"); break;
        default:                     snprintf(buf, sizeof(buf), "field 0x%04x", code); break;
    }
    return buf;
}

// a field's value as the key mixed it, readable
static inline std::string ggml_cuda_ka_trace_field_value(const uint16_t code, const uint64_t v) {
    char buf[96];
    switch (ggml_cuda_ka_field_kind_of(code)) {
        case GGML_CUDA_KA_F_OP: {
            const uint64_t op = v & 0xffff, type = v >> 16;
            snprintf(buf, sizeof(buf), "%s/%s", op < (uint64_t) GGML_OP_COUNT ? ggml_op_name((ggml_op) op) : "?",
                     type < (uint64_t) GGML_TYPE_COUNT ? ggml_type_name((ggml_type) type) : "?");
            break;
        }
        case GGML_CUDA_KA_F_SRC: {
            const uint64_t type = v >> 16;
            snprintf(buf, sizeof(buf), "%s", type < (uint64_t) GGML_TYPE_COUNT ? ggml_type_name((ggml_type) type) : "?");
            break;
        }
        case GGML_CUDA_KA_F_NE:
            if (v == GGML_CUDA_KA_MARK_DEPTH) {
                snprintf(buf, sizeof(buf), "(along the cache's length, left out)");
            } else {
                snprintf(buf, sizeof(buf), "%" PRId64, (int64_t) v);
            }
            break;
        case GGML_CUDA_KA_F_NB:
            if (v == GGML_CUDA_KA_MARK_PACKED) {
                snprintf(buf, sizeof(buf), "(packed)");
            } else {
                snprintf(buf, sizeof(buf), "%" PRIu64, v);
            }
            break;
        case GGML_CUDA_KA_F_PARAM32:
            snprintf(buf, sizeof(buf), "%d (0x%08x)", (int32_t) (uint32_t) v, (uint32_t) v);
            break;
        case GGML_CUDA_KA_F_FA:
            snprintf(buf, sizeof(buf), "0x%" PRIx64, v ^ GGML_CUDA_KA_MARK_FA);
            break;
        case GGML_CUDA_KA_F_TOPK:
            snprintf(buf, sizeof(buf), "%" PRIu64, v ^ GGML_CUDA_KA_MARK_TOPK);
            break;
        default:
            snprintf(buf, sizeof(buf), "0x%016" PRIx64, v);
            break;
    }
    return buf;
}

// one differing node, described: describe(index) names the new graph's node at that index
static inline std::string ggml_cuda_ka_trace_node_text(const ggml_cuda_ka_trace_diff & d, const ggml_cuda_ka_trace_node_diff & nd,
        const std::function<std::string(int32_t)> & describe) {
    std::string s = nd.index_now >= 0 ? describe(nd.index_now) : "a node past the end of this graph's recorded nodes";
    if (!nd.in_both) {
        char buf[128];
        snprintf(buf, sizeof(buf), ": no counterpart (%zu recorded nodes before, %zu now)", d.near_n_rec, d.n_rec);
        return s + buf;
    }
    if (!nd.field_known) {
        return s + (d.near_fields ? ": differs (no field found)" :
                                    ": differs (the earlier key's fields are no longer kept; a larger GGML_CUDA_KEEPALIVE_TRACE keeps more)");
    }
    if (nd.same_field) {
        return s + ": " + ggml_cuda_ka_trace_field_name(nd.code_now) + " " +
               ggml_cuda_ka_trace_field_value(nd.code_before, nd.val_before) + " -> " +
               ggml_cuda_ka_trace_field_value(nd.code_now, nd.val_now);
    }
    const std::string before = nd.have_before ? ggml_cuda_ka_trace_field_name(nd.code_before) + " " +
                                                ggml_cuda_ka_trace_field_value(nd.code_before, nd.val_before) : "(none)";
    const std::string now    = nd.have_now    ? ggml_cuda_ka_trace_field_name(nd.code_now) + " " +
                                                ggml_cuda_ka_trace_field_value(nd.code_now, nd.val_now) : "(none)";
    return s + ": other fields, before " + before + ", now " + now;
}

// the debug line for a new key (without the function and card prefix)
static inline std::string ggml_cuda_ka_trace_line(const ggml_cuda_ka_trace_diff & d, const int level,
        const std::function<std::string(int32_t)> & describe) {
    char head[256];
    snprintf(head, sizeof(head), "key 0x%016" PRIx64 " new at EAGER=%d (the card's new key %" PRIu64 ", %" PRId64
             " nodes, %zu recorded)", d.key, level, d.seq, d.n_nodes, d.n_rec);
    std::string s = head;
    if (!d.has_near) {
        return s + ": the card's first key";
    }
    char nearest[256];
    snprintf(nearest, sizeof(nearest), ": nearest earlier key 0x%016" PRIx64 " (new key %" PRIu64 ", %" PRId64 " nodes, %zu recorded), "
             "%zu nodes differ", d.near_key, d.near_seq, d.near_n_nodes, d.near_n_rec, d.n_diff);
    s += nearest;
    if (d.nodes.empty()) {
        char buf[160];
        snprintf(buf, sizeof(buf), "; every recorded node is the same, the graph's node count (views and no-ops included) "
                 "%" PRId64 " -> %" PRId64, d.near_n_nodes, d.n_nodes);
        return s + buf;
    }
    for (size_t i = 0; i < d.nodes.size(); ++i) {
        s += i == 0 ? "; first " : "; then ";
        s += ggml_cuda_ka_trace_node_text(d, d.nodes[i], describe);
    }
    if (d.n_diff > d.nodes.size()) {
        char buf[64];
        snprintf(buf, sizeof(buf), "; %zu more", d.n_diff - d.nodes.size());
        s += buf;
    }
    return s;
}
