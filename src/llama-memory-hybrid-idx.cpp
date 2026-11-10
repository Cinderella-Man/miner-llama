#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>
#include <type_traits>

// LLAMA_QSA_KPOOL_DEV (C100): the k-pool tables kept on the devices
struct llama_memory_hybrid_idx::kpool_dev_state {
    struct table {
        ggml_tensor * cells = nullptr; // I32 [1, cap + n_dump]
        ggml_tensor * idxs  = nullptr; // I32 [kpool, cap + n_dump]
        uint64_t      gen   = 0;       // the commit this copy holds: it holds what the host holds while gen == host_gen
    };

    // the padded entries of an update write their own rows past the end, which no graph reads
    static constexpr uint32_t n_dump = 256;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<table>                   tables; // one per device
    std::vector<int32_t>                 table_of_il; // per model layer, -1 without an indexer cache

    uint32_t cap   = 0; // rows a graph reads at most: the padded count of every pool the cache can hold
    uint32_t kpool = 0;

    // what every copy in step holds in rows [0, n_valid)
    std::vector<int32_t> cells;
    std::vector<int32_t> idxs;
    uint32_t n_valid  = 0;
    uint64_t host_gen = 1;

    // the update the last applied ubatch built (rows into cells and idxs), staged by its graph and committed by next()
    std::vector<uint32_t> rows;
    std::vector<int32_t>  sent; // the copies that ubatch's graph updates
    uint32_t n_valid_next = 0;
    bool     pending      = false;
    bool     staged       = false;

    // LLAMA_QSA_KPOOL_DEV_CHECK: what the copies held before the update
    bool check = false;
    std::vector<int32_t> prev_cells;
    std::vector<int32_t> prev_idxs;
    uint32_t prev_n_valid = 0;
    bool     prev_in_step = false;

    uint64_t n_ubatch  = 0;
    uint64_t n_full    = 0; // updates that rewrote every row a graph reads
    uint64_t n_rows    = 0; // rows uploaded
    uint64_t n_read    = 0; // rows the graphs read (what the host used to upload)
    uint64_t n_checked = 0;
};

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        no_alloc,
        type_k, type_v, v_trans, kv_size, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, no_alloc, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    kpool_dev_init(model, no_alloc, offload);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool grouping
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
        // pooled keys, the destination rebuilds its layout and re-pools into the same rows
        mem_idx_stale_set(seq_id_dst, 0);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    if (kpool_dev) {
        for (const auto & buf : kpool_dev->bufs) {
            mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    if (mem_idx) {
        mem_idx->state_write(io, seq_id, flags);
    }
}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        // the indexer cache has the SWA type of the attention cache, so both read the same partial state
        get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if (mem_idx) {
            mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            // a restore that rewrites cells does it behind the pool layout's back
            if (!sinfos_attn.empty()) {
                mem_idx_stale_set(seq_id, 0);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> rep_gen; // per global cell, the generation that last marked a pool with that rep
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps

    // LLAMA_QSA_KPOOL_DEV: rows of the tables' update (kpool_dev_build) and its graph size, stable across decode steps
    uint32_t n_upd       = 0;
    uint32_t n_upd_g     = 1;
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

uint32_t kpool_dev_n_dump() {
    return llama_memory_hybrid_idx::kpool_dev_state::n_dump;
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() {
    if (kpool_dev && kpool_dev->n_ubatch > 0) {
        const auto & dv = *kpool_dev;
        LLAMA_LOG_INFO("%s: k-pool tables on %zu devices: %" PRIu64 " ubatches, %" PRIu64 " full, %" PRIu64 " of %" PRIu64
                       " rows uploaded (%.1f a ubatch), %" PRIu64 " checked\n", __func__, dv.tables.size(), dv.n_ubatch, dv.n_full,
                       dv.n_rows, dv.n_read, (double) dv.n_rows/dv.n_ubatch, dv.n_checked);
    }
}

void llama_memory_hybrid_idx::kpool_dev_init(const llama_model & model, bool no_alloc, bool offload) {
    // read per memory, so that a process can hold contexts with the switch and without it
    const char * env = getenv("LLAMA_QSA_KPOOL_DEV");
    const int mode = env ? atoi(env) : 0;
    if (mode <= 0 || no_alloc || mem_idx == nullptr || get_kpool() == 0 || model.arch != LLM_ARCH_QWEN4EXP) {
        return;
    }

    const std::vector<uint32_t> layer_ids = mem_idx->get_layer_ids();
    const bool draft = !layer_ids.empty() && std::all_of(layer_ids.begin(), layer_ids.end(),
            [&](uint32_t il) { return il >= model.hparams.n_layer(); });
    if (mode == 2 && !draft) {
        return;
    }
    // tensor split spreads each tensor over the devices by name, and a cache on the host has no device to keep them on
    if (!offload || model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        LLAMA_LOG_WARN("%s: LLAMA_QSA_KPOOL_DEV needs the indexer cache offloaded and no tensor split, keeping the uploads\n", __func__);
        return;
    }

    const uint32_t kpool   = get_kpool();
    const uint64_t n_cells = (uint64_t) mem_idx->get_size()*mem_idx->get_n_stream();
    const uint64_t n_pool  = uint64_t(mem_idx->get_size() / kpool) * mem_idx->get_n_seq_max();

    auto dv = std::make_unique<kpool_dev_state>();
    dv->kpool = kpool;
    dv->cap   = kpool_pad((uint32_t) n_pool);
    // the update carries rows and cells as F32, exact below 2^24
    GGML_ASSERT(n_cells < (1u << 24) && dv->cap + kpool_dev_state::n_dump < (1u << 24));

    const int64_t n_rows = (int64_t) dv->cap + kpool_dev_state::n_dump;

    dv->table_of_il.assign(model.hparams.n_layer_all, -1);
    std::map<ggml_backend_buffer_type_t, int32_t> table_of_buft;
    for (uint32_t il : layer_ids) {
        if (model.hparams.dsv4_compress_ratios[il] == 0) {
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(mem_idx->get_k_storage(il)->buffer);

        auto it = table_of_buft.find(buft);
        if (it == table_of_buft.end()) {
            ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            GGML_ASSERT(ctx != nullptr);
            dv->ctxs.emplace_back(ctx);

            kpool_dev_state::table t;
            t.cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1,     n_rows);
            t.idxs  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kpool, n_rows);
            ggml_format_name(t.cells, "kpool_dev_cells_%zu", dv->tables.size());
            ggml_format_name(t.idxs,  "kpool_dev_idxs_%zu",  dv->tables.size());

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error(format("failed to allocate the k-pool tables in %s", ggml_backend_buft_name(buft)));
            }
            // placed like weights: the update that writes them runs where they are
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(buf, 0);
            dv->bufs.emplace_back(buf);

            it = table_of_buft.emplace(buft, (int32_t) dv->tables.size()).first;
            dv->tables.push_back(t);
        }
        dv->table_of_il[il] = it->second;
    }
    if (dv->tables.empty()) {
        return;
    }

    dv->cells.assign(dv->cap, 0);
    dv->idxs.assign((size_t) dv->cap*kpool, 0);
    dv->rows.reserve(dv->cap);

    const char * env_check = getenv("LLAMA_QSA_KPOOL_DEV_CHECK");
    dv->check = env_check != nullptr && atoi(env_check) != 0;

    size_t size = 0;
    for (const auto & buf : dv->bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    LLAMA_LOG_INFO("%s: k-pool tables kept on %zu devices (%s): %u rows, %.2f MiB on the devices, %.2f MiB on the host%s\n",
            __func__, dv->tables.size(), draft ? "MTP draft" : "model", dv->cap, size/1024.0/1024.0,
            (dv->cells.size() + dv->idxs.size())*sizeof(int32_t)/1024.0/1024.0, dv->check ? ", checked" : "");

    kpool_dev    = std::move(dv);
    kpool_dev_on = true;
}

bool llama_memory_hybrid_idx::kpool_dev_check_on() const {
    return kpool_dev_on && kpool_dev->check;
}

void llama_memory_hybrid_idx::kpool_dev_disable() {
    if (kpool_dev_on) {
        LLAMA_LOG_INFO("%s: pipelined decoding: the k-pool tables are uploaded whole again\n", __func__);
    }
    kpool_dev_on = false;
}

void llama_memory_hybrid_idx::kpool_dev_commit() {
    if (!kpool_dev_on) {
        return;
    }
    auto & dv = *kpool_dev;
    // a ubatch whose graph never staged its update stays pending, so the next build starts over
    if (!dv.pending || !dv.staged) {
        return;
    }
    dv.n_valid = dv.n_valid_next;
    dv.host_gen++;
    for (int32_t t : dv.sent) {
        dv.tables[t].gen = dv.host_gen;
    }
    dv.pending = false;
    dv.staged  = false;
}

bool llama_memory_hybrid_idx::kpool_dev_stats(uint64_t * n_ubatch, uint64_t * n_full, uint64_t * n_rows, uint64_t * n_checked) const {
    if (!kpool_dev) {
        return false;
    }
    if (n_ubatch)  { *n_ubatch  = kpool_dev->n_ubatch;  }
    if (n_full)    { *n_full    = kpool_dev->n_full;    }
    if (n_rows)    { *n_rows    = kpool_dev->n_rows;    }
    if (n_checked) { *n_checked = kpool_dev->n_checked; }
    return true;
}

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.j_next  = 0;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
        }

        // Pools start at the first valid token
        size_t j = sq.j_next;
        if (hparams_idx.indexer_kpool_by_order) {
            // consecutive cells in sequence order, whatever their positions
            for (; j + kpool <= sq.cells.size(); j += kpool) {
                sq.pools.push_back((uint32_t) j);
            }
        } else {
            while (j + kpool <= sq.cells.size()) {
                const llama_pos p0 = sq.cells[j].first;
                if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                    ++j;
                    continue;
                }
                bool ok = true;
                for (uint32_t k = 1; k < kpool; ++k) {
                    if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    sq.pools.push_back((uint32_t) j);
                    j += kpool;
                } else {
                    ++j;
                }
            }
        }
        sq.j_next = j;

        lay.n_pool_real += (uint32_t) sq.pools.size();
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        // a full update of every row, or the padded bound of a short one
        st.n_upd   = kpool_pad(st.n_pool_real);
        st.n_upd_g = std::max(st.n_upd, kpool_dev_n_dump());
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    // so is the update of the k-pool tables on the devices
    if (mem != nullptr && i_kpool == i_cur) {
        mem->kpool_dev_commit();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
        if (kpool_dev_on()) {
            kpool_dev_build(get_ubatch());
        }
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(),  st.is_new.end(),  0);
        std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);
    st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
    // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const auto & sq  = lay.seqs[s];
        const size_t rep = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + kpool - 1].second;
        if (st.rep_gen[rep] != st.generation) {
            st.rep_gen[rep] = st.generation;
            st.is_new[pool_start[s] + k] = st.generation;
            ++st.n_new;
        }
    };

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_pad(kpool_cur().n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

bool llama_memory_hybrid_idx_context::kpool_dev_on() const {
    return mem != nullptr && mem->kpool_dev_get() != nullptr;
}

bool llama_memory_hybrid_idx_context::kpool_dev_check_on() const {
    return mem != nullptr && mem->kpool_dev_check_on();
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_upd() const {
    return kpool_cur().n_upd_g;
}

int32_t llama_memory_hybrid_idx_context::kpool_dev_table(int32_t il, ggml_tensor ** cells, ggml_tensor ** idxs) const {
    const auto * dv = mem != nullptr ? mem->kpool_dev_get() : nullptr;
    GGML_ASSERT(dv != nullptr);

    const int32_t t = il >= 0 && (size_t) il < dv->table_of_il.size() ? dv->table_of_il[il] : -1;
    if (t >= 0) {
        *cells = dv->tables[t].cells;
        *idxs  = dv->tables[t].idxs;
    }

    return t;
}

// The rows of pool_cells and pool_idxs that differ from what the copies on the devices hold, as set_input_kpool fills
// them without the switch: each pool's rep cell and member cells (n_kv for a sequence absent from a multi-stream ubatch),
// then the padded pools on a cell of the ubatch and n_kv. They go into the host's copy at once; next() commits them
// once the graph that writes them to the devices has run.
void llama_memory_hybrid_idx_context::kpool_dev_build(const llama_ubatch & ubatch) {
    auto & dv = *mem->kpool_dev_get();
    auto & st = *kpool_st;

    const auto & lay = mem->kpool_layout_get();

    const uint32_t kpool       = mem->get_kpool();
    const uint32_t n_kv        = get_idx()->get_n_kv();
    const uint32_t n_pool      = kpool_pad(st.n_pool_real);
    const uint32_t kv_size     = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    GGML_ASSERT(n_pool <= dv.cap);

    // an update no graph wrote, or a copy a graph left out: the copies may not hold the host's rows, so rewrite them all
    bool in_step = !dv.pending;
    for (const auto & t : dv.tables) {
        in_step = in_step && t.gen == dv.host_gen;
    }
    if (!in_step) {
        dv.n_valid = 0;
    }
    dv.pending = false;
    dv.staged  = false;

    if (dv.check) {
        dv.prev_cells   = dv.cells;
        dv.prev_idxs    = dv.idxs;
        dv.prev_n_valid = dv.n_valid;
        dv.prev_in_step = in_step;
    }

    const uint32_t n_valid = dv.n_valid;
    dv.rows.clear();

    auto put = [&](uint32_t ip, int32_t cell, const int32_t * idxs) {
        int32_t * c = dv.cells.data() + ip;
        int32_t * m = dv.idxs.data() + (size_t) ip*kpool;
        if (ip >= n_valid || *c != cell || memcmp(m, idxs, kpool*sizeof(int32_t)) != 0) {
            *c = cell;
            memcpy(m, idxs, kpool*sizeof(int32_t));
            dv.rows.push_back(ip);
        }
    };

    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            seq_in_ub[ubatch.seq_id[i][k]] = 1;
        }
    }

    const int32_t sentinel = (int32_t) n_kv;

    std::vector<int32_t> row(kpool);

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi, ++ip) {
            const uint32_t j = sq.pools[pi];
            GGML_ASSERT(ip + 1 < n_pool);

            for (uint32_t k = 0; k < kpool; ++k) {
                row[k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
            }
            put(ip, (int32_t) ((int64_t) sq.strm*kv_size + sq.cells[j + kpool - 1].second), row.data());
        }
    }
    GGML_ASSERT(ip == st.n_pool_real);

    // the padded pools: a cell of this ubatch, written before any read, and the sentinel
    int32_t dummy_cell = 0;
    {
        const auto & sq = lay.seqs[ubatch.seq_id[0][0]];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch.pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch.pos[0]);
        dummy_cell = (int32_t) ((int64_t) sq.strm*kv_size + it->second);
    }
    std::fill(row.begin(), row.end(), sentinel);
    for (; ip < n_pool; ++ip) {
        put(ip, dummy_cell, row.data());
    }

    dv.n_valid_next = std::max(n_valid, n_pool);
    dv.pending      = true;

    // a decode step updates the pads and a pool or two, rarely more after a sequence edit: pad to a stable bound, so its
    // graph is reused; a longer update is padded to the next multiple of 64
    const uint32_t n_upd = (uint32_t) dv.rows.size();
    const uint32_t bound = std::min<uint32_t>(GGML_PAD(64 + ubatch.n_tokens/kpool + 2*ubatch.n_seqs_unq + 8, 64u), kpool_dev_n_dump());

    st.n_upd   = n_upd;
    st.n_upd_g = n_upd <= bound ? bound : GGML_PAD(n_upd, 64u);
    GGML_ASSERT(st.n_upd_g - st.n_upd <= kpool_dev_n_dump());

    dv.n_ubatch++;
    dv.n_full += n_valid == 0;
    dv.n_rows += n_upd;
    dv.n_read += n_pool;
}

void llama_memory_hybrid_idx_context::set_input_kpool_upd(ggml_tensor * upd, const std::vector<int32_t> & tables) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(upd->buffer) && upd->type == GGML_TYPE_F32);

    auto & dv = *mem->kpool_dev_get();
    const auto & st = kpool_cur();

    const uint32_t kpool = dv.kpool;
    const uint32_t n_g   = st.n_upd_g;
    const uint32_t n_upd = st.n_upd;

    GGML_ASSERT(dv.pending && n_upd == dv.rows.size());
    GGML_ASSERT(ggml_nelements(upd) == (int64_t) (2 + kpool)*n_g);

    float * rows  = (float *) upd->data;
    float * cells = rows + n_g;
    float * idxs  = rows + 2*n_g;

    for (uint32_t i = 0; i < n_upd; ++i) {
        const uint32_t ip = dv.rows[i];
        rows [i] = (float) ip;
        cells[i] = (float) dv.cells[ip];
        for (uint32_t k = 0; k < kpool; ++k) {
            idxs[(size_t) i*kpool + k] = (float) dv.idxs[(size_t) ip*kpool + k];
        }
    }
    for (uint32_t i = n_upd; i < n_g; ++i) {
        rows [i] = (float) (dv.cap + (i - n_upd));
        cells[i] = 0.0f;
        std::fill(idxs + (size_t) i*kpool, idxs + (size_t) (i + 1)*kpool, 0.0f);
    }

    dv.sent   = tables;
    dv.staged = true;
}

void llama_memory_hybrid_idx_context::kpool_dev_check(const ggml_tensor * pool_mask, const ggml_tensor * tail_idxs,
        const ggml_tensor * new_pool_idxs, const ggml_tensor * new_pool_rep, const ggml_tensor * new_pool_pos,
        const ggml_tensor * upd, const std::vector<int32_t> & tables, const llama_ubatch * ubatch) const {
    auto & dv = *mem->kpool_dev_get();
    const auto & st = kpool_cur();

    const uint32_t kpool  = dv.kpool;
    const uint32_t n_pool = get_n_kpool();
    const uint32_t n_g    = st.n_upd_g;

    // the inputs as the host builds them without the switch, into tensors of their own
    const int n_t = 7;
    ggml_init_params ip = { n_t*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    auto dup = [&](const ggml_tensor * t) { return t ? ggml_dup_tensor(ctx.get(), t) : nullptr; };

    ggml_tensor * h_cells = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_pool);
    ggml_tensor * h_idxs  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, kpool, n_pool);
    ggml_tensor * h_mask  = dup(pool_mask);
    ggml_tensor * h_tail  = dup(tail_idxs);
    ggml_tensor * h_nidx  = dup(new_pool_idxs);
    ggml_tensor * h_nrep  = dup(new_pool_rep);
    ggml_tensor * h_npos  = dup(new_pool_pos);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(buf);

    set_input_kpool(h_cells, h_idxs, h_mask, h_tail, nullptr, h_nidx, h_nrep, ubatch, h_npos);

    auto same = [&](const ggml_tensor * a, const ggml_tensor * b, const char * name) {
        if (a == nullptr) {
            return;
        }
        if (ggml_nbytes(a) != ggml_nbytes(b) || memcmp(a->data, b->data, ggml_nbytes(a)) != 0) {
            GGML_ABORT("%s: %s differs from the host's without the switch (ubatch %" PRIu64 ")", __func__, name, dv.n_ubatch);
        }
    };
    same(pool_mask,     h_mask, "pool_mask");
    same(tail_idxs,     h_tail, "tail_idxs");
    same(new_pool_idxs, h_nidx, "new_pool_idxs");
    same(new_pool_rep,  h_nrep, "new_pool_rep");
    same(new_pool_pos,  h_npos, "new_pool_pos");

    // the host's copy, which the update leaves on the devices
    if (memcmp(dv.cells.data(), h_cells->data, ggml_nbytes(h_cells)) != 0 ||
        memcmp(dv.idxs.data(),  h_idxs->data,  ggml_nbytes(h_idxs))  != 0) {
        GGML_ABORT("%s: the host's copy of the tables differs from the tables without the switch (ubatch %" PRIu64 ")", __func__, dv.n_ubatch);
    }

    // each copy as the graphs before left it (the caller synchronized), then this update on it
    const float * u_rows  = (const float *) upd->data;
    const float * u_cells = u_rows + n_g;
    const float * u_idxs  = u_rows + 2*n_g;

    std::vector<int32_t> d_cells((size_t) dv.cap + kpool_dev_n_dump());
    std::vector<int32_t> d_idxs (d_cells.size()*kpool);
    for (int32_t t : tables) {
        const auto & tb = dv.tables[t];
        ggml_backend_tensor_get(tb.cells, d_cells.data(), 0, ggml_nbytes(tb.cells));
        ggml_backend_tensor_get(tb.idxs,  d_idxs.data(),  0, ggml_nbytes(tb.idxs));

        if (dv.prev_in_step) {
            if (memcmp(d_cells.data(), dv.prev_cells.data(), dv.prev_n_valid*sizeof(int32_t)) != 0 ||
                memcmp(d_idxs.data(),  dv.prev_idxs.data(),  (size_t) dv.prev_n_valid*kpool*sizeof(int32_t)) != 0) {
                GGML_ABORT("%s: copy %d (%s) does not hold what the host held (ubatch %" PRIu64 ")", __func__, t,
                        ggml_backend_buffer_name(tb.cells->buffer), dv.n_ubatch);
            }
        }

        for (uint32_t i = 0; i < n_g; ++i) {
            const int64_t r = (int64_t) u_rows[i];
            GGML_ASSERT((float) r == u_rows[i] && r >= 0 && r < (int64_t) d_cells.size());
            GGML_ASSERT(i < st.n_upd ? r < (int64_t) n_pool : r >= (int64_t) dv.cap);
            d_cells[r] = (int32_t) u_cells[i];
            for (uint32_t k = 0; k < kpool; ++k) {
                d_idxs[(size_t) r*kpool + k] = (int32_t) u_idxs[(size_t) i*kpool + k];
            }
        }

        if (memcmp(d_cells.data(), h_cells->data, ggml_nbytes(h_cells)) != 0 ||
            memcmp(d_idxs.data(),  h_idxs->data,  ggml_nbytes(h_idxs))  != 0) {
            GGML_ABORT("%s: copy %d (%s) with this update differs from the tables without the switch (ubatch %" PRIu64 ")",
                    __func__, t, ggml_backend_buffer_name(tb.cells->buffer), dv.n_ubatch);
        }
    }

    dv.n_checked++;
    if (dv.n_checked == 1 || dv.n_checked % 256 == 0) {
        LLAMA_LOG_INFO("%s: %" PRIu64 " ubatches checked: %zu copies identical to the tables without the switch (n_pool %u, "
                       "%u rows updated, %s)\n", __func__, dv.n_checked, tables.size(), n_pool, st.n_upd,
                       dv.prev_in_step ? "in step" : "rewritten");
    }
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    // without the tables (LLAMA_QSA_KPOOL_DEV) apply() built them into the host's copy, which the pads' search reads
    const bool tables = pool_cells != nullptr;
    GGML_ASSERT(tables == (pool_idxs != nullptr) && (tables || kpool_dev_on()));
    GGML_ASSERT(!tables || ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(!tables || ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = tables ? (uint32_t) pool_cells->ne[0] : kpool_pad(st.n_pool_real);
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(!tables || (pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool));
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(ggml_nelements(new_pool_idxs) == (int64_t) kpool*n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = tables ? (int32_t *) pool_cells->data : mem->kpool_dev_get()->cells.data();
    int32_t * pidx  = tables ? (int32_t *) pool_idxs->data  : nullptr;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            if (tables) {
                pcell[ip] = (int32_t) gcell(sq, rep);

                for (uint32_t k = 0; k < kpool; ++k) {
                    pidx[(size_t) ip*kpool + k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
                }
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                nrep[i_new] = gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    // They take the lowest cells that are no pool's rep: the first n_pad of those lie below n_pad + n_reps, so marking the
    // reps in that range finds them in linear time.
    // (LLAMA_QSA_PAD_LINEAR=0: the reps sorted and searched, as upstream; the same cells)
    static const bool pad_linear = [] {
        const char * env = getenv("LLAMA_QSA_PAD_LINEAR");
        return env == nullptr || atoi(env) != 0;
    }();
    if (n_new_g > n_new && !pad_linear) {
        std::vector<int64_t> reps(pcell, pcell + pool_end.size());
        std::sort(reps.begin(), reps.end());

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (std::binary_search(reps.begin(), reps.end(), pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    } else if (n_new_g > n_new) {
        const int64_t n_reps = (int64_t) pool_end.size();
        const int64_t bound  = std::min<int64_t>((int64_t) kv_size*n_stream_kv, (int64_t) (n_new_g - n_new) + n_reps);
        std::vector<uint8_t> is_rep(bound, 0);
        for (int64_t ip = 0; ip < n_reps; ++ip) {
            if (pcell[ip] < bound) {
                is_rep[pcell[ip]] = 1;
            }
        }

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (pad_cell < bound && is_rep[pad_cell]) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < bound);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool && tables; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            std::fill(row + p0, row + p0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}
