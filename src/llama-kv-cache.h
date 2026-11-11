#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
                         bool   no_alloc,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "",
        // lightning indexer keys are rotated with the indexer RoPE parameters
                       bool   is_indexer = false);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_seq_max() const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // The stream holding seq_id's cells.
    uint32_t get_stream(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
            slot_info_vec_t * sinfos_out,
      const slot_info_vec_t * sinfos_in);

    // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
    void state_clear(llama_seq_id seq_id);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // attention over the cells of each sequence of the ubatch instead of the whole cache (see ggml_flash_attn_ext_rows)
    // the ubatch splits into n_slices groups of tokens with one sequence each
    // if it cannot (f.ex. interleaved or coupled sequences), it is a single slice over the cells of all its sequences
    // TODO: allow slices with coupled sequences
    static uint32_t get_n_kv_slices(const llama_ubatch & ubatch);

    // padded length of the longest cell list of the slices; 0 if kv_rows are not used (multiple streams)
    uint32_t get_n_kv_rows(const llama_ubatch & ubatch) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    // C738: the cells' token ids (ext.tok) again from the ubatch that apply_ubatch() placed there, as apply_ubatch() sets them
    void apply_ubatch_tokens(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    std::vector<ggml_tensor *> get_k_rot() const;
    std::vector<ggml_tensor *> get_v_rot() const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_attn_mask (const llm_attn_mask & attn_mask, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // the mask built on the device (ggml_kq_mask) reads the cells' positions and sequences from a copy that each device
    // holding layers of this cache keeps: F32 [2, kv_size*n_stream + 1], per cell its position and, as bit s, whether
    // it holds sequence s (an empty cell holds none), plus a scratch row. A step that builds its mask on the device
    // scatters into every copy first the cells changed since the last step (seq_rm, a seq_cp within a stream: up to
    // kq_cells_n_changed_max of them, the rest of the list pointing at the scratch row), then the cells it writes. The
    // copies are stale after more changes than that, any other change to the cells, or a step whose cells did not reach
    // every copy (a host-mask step, say); the next device-mask step then uploads all cells.
    // Kept for offloaded caches with their own cells and at most 24 sequences (the bits and positions below 2^24 are
    // exact), except with tensor split.
    bool   get_kq_cells_resident() const;
    bool   get_kq_cells_stale()    const { return kq_cells_stale || kq_cells_n_applied > 1; }
    size_t get_kq_cells_n_copies() const { return kq_cells_devs.size(); }

    // the copy on layer il's device; il < 0: the first copy
    ggml_tensor * get_kq_cells(int32_t il) const;

    // a model layer whose cells are on the first copy's device
    int32_t get_kq_cells_il0() const { return kq_cells_il.at(0); }

    // the rows a step's update writes: its own cells, then the kq_cells_n_changed_max entries of the changed-cell list
    uint32_t get_kq_cells_n_upd(const llama_ubatch & ubatch) const { return ubatch.n_tokens + kq_cells_n_changed_max; }

    ggml_tensor * build_input_kq_cells_all(ggml_context * ctx) const; // F32 [2, kv_size*n_stream + 1]

    // a copy updated for this step: cells_all, if set, replaces it, then the values vals (F32 [2, n_upd]) go to the rows
    // that rows (I32 [n_upd]) names; a row named twice gets the same value twice
    ggml_tensor * update_kq_cells(ggml_context * ctx, ggml_tensor * cells, ggml_tensor * cells_all, ggml_tensor * vals, ggml_tensor * rows) const;

    // the cells a ubatch attends to, from an updated copy: F32 [2, n_kv, ns]
    ggml_tensor * view_kq_cells(ggml_context * ctx, ggml_tensor * cells, uint32_t n_kv, const slot_info & sinfo) const;

    // a step's update, as floats (all integers below 2^24): rows [n_upd] holds the step's cells in the order of
    // set_input_k_idxs, then the changed cells, then the scratch row; vals [2*n_upd] their current values
    void fill_kq_cells_upd(float * rows, float * vals, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_kq_cells_all(ggml_tensor * dst) const;

    // a step applied its cells (llama_kv_cache_context::apply); they reach the copies only through that step's graph
    void set_kq_cells_applied() { kq_cells_n_applied++; }

    // the step whose inputs were just set updates every copy: they hold the cells once it has run
    void set_kq_cells_synced() const { kq_cells_stale = false; kq_cells_n_applied = 0; kq_cells_changed.clear(); }

    // LLAMA_GPU_KQ_MASK_CHECK=1: read every device's copy (the caller synchronizes first), apply this step's update to it
    // (cells_all: the full upload or null; rows, vals: as fill_kq_cells_upd wrote them), expand the mask for the tokens'
    // pairs toks (F32 [2, n_tps, n_stream]) by ggml_kq_mask's rule and compare it with the host mask; abort on a difference
    void check_kq_cells(const float * cells_all, const float * rows, const float * vals, uint32_t n_upd, const float * toks, uint32_t n_tps,
            uint32_t n_kv, const slot_info & sinfo, const llama_ubatch * ubatch, bool causal_attn) const;

    static constexpr uint32_t kq_cells_n_changed_max = 32;

    // the mask set_input_attn_mask writes (rows == nullptr shape), as F32 into data [n_kv, n_tokens/n_stream, 1, n_stream]
    void fill_kq_mask_f32(float * data, int64_t n_kv, int64_t n_stream, const llama_ubatch * ubatch, bool causal_attn) const;

    uint32_t       get_n_swa()    const { return n_swa; }
    llama_swa_type get_swa_type() const { return swa_type; }


    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        ggml_tensor * k_rot = nullptr;
        ggml_tensor * v_rot = nullptr;

        // the cell copy of the layer's device (see get_kq_cells_resident), shared by the layers there
        ggml_tensor * kq_cells = nullptr;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // SWA
    const uint32_t n_swa = 0;

    const bool is_indexer = false;

    // env: LLAMA_ATTN_ROT_DISABLE
    uint32_t n_rot_k = 0;
    uint32_t n_rot_v = 0;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // keep constants separate from mutable KV buffers
    llama_memory_bufs attn_rot_bufs;

    void init_rotation(const char * name_tag);

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    llama_memory_bufs ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    // one cell copy per device (see get_kq_cells_resident); stale until a step uploads all cells
    std::vector<ggml_tensor *> kq_cells_devs;
    std::vector<int32_t>       kq_cells_il; // per copy, the first model layer stored on its device
    mutable bool     kq_cells_stale     = true;
    mutable uint32_t kq_cells_n_applied = 0; // steps applied since the copies were last updated

    // cells changed outside the steps' own writes since then (stream*kv_size + cell); kq_cells_track off: not recorded
    mutable std::vector<uint32_t> kq_cells_changed;
    bool kq_cells_track = true;

    void kq_cells_mark(uint32_t strm, uint32_t idx);

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * k,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // where a sequence processed from its first token puts these cells in an empty stream (empty if it cannot)
    slot_info canonical_slot(uint32_t strm, const llama_ubatch & ubatch, uint32_t n_cells) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);

    void state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    bool tokens_settable() const override;
    void set_tokens() override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;
    uint32_t get_n_kv_rows() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    std::vector<ggml_tensor *> get_k_rot() const;
    std::vector<ggml_tensor *> get_v_rot() const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_attn_mask (const llm_attn_mask & attn_mask, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void fill_kq_mask_f32    (float * data, int64_t n_kv, int64_t n_stream, const llama_ubatch * ubatch, bool causal_attn) const;

    // see llama_kv_cache::get_kq_cells_resident()
    bool          get_kq_cells_resident() const;
    bool          get_kq_cells_stale()    const;
    size_t        get_kq_cells_n_copies() const;
    ggml_tensor * get_kq_cells(int32_t il) const;
    int32_t       get_kq_cells_il0() const;
    uint32_t      get_kq_cells_n_upd(const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_kq_cells_all(ggml_context * ctx) const;
    ggml_tensor * update_kq_cells(ggml_context * ctx, ggml_tensor * cells, ggml_tensor * cells_all, ggml_tensor * vals, ggml_tensor * rows) const;
    ggml_tensor * view_kq_cells(ggml_context * ctx, ggml_tensor * cells) const;
    void          fill_kq_cells_upd(float * rows, float * vals, const llama_ubatch * ubatch) const;
    void          set_input_kq_cells_all(ggml_tensor * dst) const;
    void          set_kq_cells_synced() const;
    void          check_kq_cells(const float * cells_all, const float * rows, const float * vals, uint32_t n_upd, const float * toks, uint32_t n_tps,
                          const llama_ubatch * ubatch, bool causal_attn) const;

    uint32_t       get_n_swa()    const;
    llama_swa_type get_swa_type() const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;

    // see llama_kv_cache::get_n_kv_rows
    uint32_t n_kv_rows = 0;
};
