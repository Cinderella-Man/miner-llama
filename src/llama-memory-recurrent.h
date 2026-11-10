#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                         bool   no_alloc,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there; grp: the state group the ubatch writes
    bool find_slot(const llama_ubatch & ubatch, uint32_t grp = 0);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // State groups for pipelined decoding (llama_pipe_*, n_rs_seq 0 only): the tensors hold n_grp groups of size rows,
    // and each pipe slot's evaluations write their states to the group of the slot's index (llama_decode: group 0), at a
    // fixed offset in the slot's graph. Every evaluation reads its source state through s_copy, from the group that holds
    // it. Per cell (row index) ring_pos records which position's state each group holds, so removing a sequence's tail
    // (a wrong guess) only has to find the group that holds the state at the new end. A slot is reused only once the
    // executor has read its token, so the states of tokens still in flight stay; the one state before them that a
    // rollback may need is saved in a last, stash group when its slot is reused (pipe_stash).
    // set_n_groups keeps each cell's current state (moved to group 0) and forgets the older ones.
    uint32_t n_grp = 1;

    bool set_n_groups(uint32_t n);

    // the last group (n_grp - 1) is the stash: before a pipe slot's evaluation writes group g of seq_id's row, the
    // state at position keep (the one before the sequence's oldest token still to be read, which a rollback may need:
    // a wrong guess behind it, or an executor leaving pipelined decoding) must not be lost if g holds it. Returns the
    // row whose group g the caller copies into the stash (the ring already says the stash holds it), -1 if none
    int32_t pipe_stash(llama_seq_id seq_id, uint32_t g, llama_pos keep);

    // the group the next batch context writes (llama_context sets it around init_batch for a pipe slot)
    uint32_t grp_next = 0;

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // the cells find_slot moved to another row to make a batch's rows contiguous (llama_memory_recurrent_n_swaps)
    uint64_t n_swaps = 0;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        // largest rollback whose snapshot the last ubatch guarantees: a ubatch of n <= n_rs_seq + 1 tokens
        // writes snapshots 0..n-1, a longer one only guarantees the final state (ggml_gated_delta_net)
        uint32_t rs_max = 0;

        // state groups (see n_grp), properties of the row index rather than of the sequence in it:
        int32_t                grp      = 0;  // the group holding the state at pos
        int32_t                src_grp0 = 0;  // the group src0 is read from, staged with src0
        std::vector<llama_pos> ring_pos;      // per group, the position of the state it holds for this row (-1: none)

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the RS cache along with the allocated backend buffers:
    llama_memory_bufs ctxs_bufs;

    // what the tensors are made of, kept for set_n_groups: per layer its buffer type (null: no state), and the types
    std::vector<ggml_backend_buffer_type_t> layer_buft;
    ggml_type type_r = GGML_TYPE_F32;
    ggml_type type_s = GGML_TYPE_F32;

    // (re)creates r_l, s_l, p_l and ctxs_bufs with n_rows rows each, cleared
    void alloc_tensors(uint32_t n_rows);

    // a cell's ring after its row was written with the state at pos in group g: the states of positions before p_keep
    // in the other groups stay valid (0: none, when the row got another sequence's state), the others are forgotten
    void ring_set(mem_cell & cell, int32_t g, llama_pos pos, llama_pos p_keep) const;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    // the state group the ubatches' new states are written to (see llama_memory_recurrent::n_grp)
    uint32_t get_grp() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;

    const uint32_t grp = 0;
};
