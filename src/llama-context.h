#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-embd-cache.h"
#include "llama-head-bound.h"
#include "llama-head-shortlist.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-moe-cache.h"
#include "llama-ngram-prefetch.h"
#include "llama-pipe-drafter.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <mutex>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;
class llama_memory_recurrent;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

// a numbered device copy of one sequence state (llama_state_seq_get_data_store)
struct llama_state_store {
    // one context and buffer per buffer type, holding the copies
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> bufs;

    // the copies in the order the state wrote them, each with the size of the blob written before it: where its bytes go
    // in the host format
    std::vector<std::pair<ggml_tensor *, size_t>> pieces;

    size_t size_meta = 0; // the blob returned with the store
    size_t size_data = 0; // the bytes in the copies

    // a copy of that blob: a restore or a host copy checks that it is given the blob of this fill
    std::vector<uint8_t> meta;

    // C217: a fill inside a decode (llama_state_seq_store_at) queues its copies on the backends that compute the state,
    // with an event recorded on each after them; a host copy waits for these first. Empty after a fill that waited
    std::vector<ggml_backend_event_ptr> fences;
};

// LLAMA_TENSOR_HASH=<prefix>[,<prefix>...] (C762, checks): every graph node whose name starts with one of the prefixes is
// hashed (FNV-1a over its bytes) once it is computed, through the scheduler's eval callback, and logged with the buffer
// that holds it, so that two runs' intermediate tensors can be compared byte for byte; unused when the caller sets
// cb_eval. The callback makes the scheduler compute each split in pieces that end at those nodes
struct llama_tensor_hash {
    std::vector<std::string> prefixes;
    char     role = 'v'; // 'v' a target context, 'd' an MTP draft context
    uint64_t n    = 0;   // tensors hashed so far
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);
    float * get_logits_ith_fetch(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    ggml_tensor * get_layer_inp_dev() const { return layer_inp_dev; }
    ggml_tensor * get_h_nextn_dev()   const { return h_nextn_dev; }

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;
    bool get_screen_head() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_embeddings_layer_inp_dev(bool value);

    // whether decode accepts entries whose embd is a row of ctx_other's layer input device buffer
    bool supports_embd_row() const;

    // h_nextn device buffer, see llama_set_embeddings_nextn_dev()
    bool set_embeddings_nextn_dev(bool value);
    bool get_h_nextn_rows(int64_t row0, int64_t n_rows, float * out);
    bool set_h_nextn_rows(int64_t row0, int64_t n_rows, const float * data);

    // two halves of the h_nextn device buffer (C703), see llama_set_embeddings_nextn_dev_halves()
    bool    set_h_nextn_halves(int32_t n);
    bool    hold_h_nextn();
    int64_t get_h_nextn_row0() const { return h_nextn_dev ? h_nextn_last*h_nextn_half_rows() : 0; }
    int64_t h_nextn_half_rows() const { return h_nextn_dev ? h_nextn_dev->ne[1]/h_nextn_halves : 0; }
    int32_t h_nextn_half_of(int64_t row) const { return h_nextn_halves > 1 && row >= h_nextn_half_rows() ? 1 : 0; }
    // the halves the buffer holds now, 0 without a buffer (C813: one until the first hold with LLAMA_MTP_HROW_LAZY)
    int32_t get_h_nextn_halves() const { return h_nextn_dev ? h_nextn_halves : 0; }

    // state rows read from the h_nextn device buffer of src, see llama_set_embd_state_src()
    bool set_state_src(llama_context * src);
    bool supports_state_row() const;
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_screen_head(bool value);

    // Q771: the micro-batch taper (llama_set_ubatch_taper), and the caller's hints for the next decode call
    bool         set_ubatch_taper(const char * spec);
    const char * get_ubatch_taper() const;
    void         set_ubatch_taper_next(int32_t head, int32_t tail);

    void set_warmup(bool value);

    // a shortlist for this (draft) context's output head (llama_head_shortlist_*)
    bool     head_shortlist_init(uint32_t k_low, uint32_t m_cap);
    uint32_t head_shortlist_add(const llama_token * ids, size_t n);
    bool     head_shortlist_contains(llama_token id) const;
    uint32_t head_shortlist_size() const;
    void     set_head_shortlist(bool value);

    // the (draft) context's token embedding rows on its layer's card (llama_embd_cache_*)
    bool embd_cache_init(uint32_t n_rows, uint32_t max_tokens);
    bool embd_cache_stats(uint64_t * n_hit, uint64_t * n_miss, uint64_t * n_plain) const;

    // the (draft) context's hidden rows kept on its layer's card between passes (llama_nextn_carry_init)
    bool nextn_carry_init();
    void set_nextn_carry(bool in, bool out);

    // the next decode copies no h_nextn rows to the host (llama_skip_embeddings_nextn_next), and the counts
    void skip_h_nextn_next() { h_nextn_skip_next = true; }
    void h_nextn_host_stats(uint64_t * n_copied, uint64_t * n_skipped) const;

    // the prefix of this context's marks in a step trace (llama_step_trace_role)
    void set_step_trace_role(char role) { step_trace_role = role; }

    // the device of the MTP layer's weights and this context's backend for it
    bool mtp_layer_device(ggml_backend_dev_t & dev, ggml_backend_t & backend) const;

    // fewer routed experts on confident tokens (llama_set_moe_tail)
    void     set_moe_tail(bool on, float p, float w, bool renorm, bool hist);
    void     set_moe_tail_mode(bool graph, int skip, bool repeat);
    uint64_t moe_tail_counts(uint64_t * counts, int32_t n_counts) const;
    void     moe_tail_log() const;

    // route once per verify block (llama_set_moe_route_once)
    void     set_moe_route_once(uint32_t k, uint32_t max, bool hist, bool dry);
    uint64_t moe_route_once_counts(uint64_t * read, uint64_t * indep, int32_t n_counts, uint64_t * n_outside) const;
    void     moe_route_once_log(bool by_layer) const;

    // the main model's shortlist head with an exact bound (llama_head_bound_*)
    bool  head_bound_init();
    void  set_head_bound(bool value);
    float head_bound_ith(int32_t i);
    bool  head_bound_full(int32_t i);
    void  head_bound_stats(uint64_t * n_rows, uint64_t * n_full) const;

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    // the host's times of a ubatch for LLAMA_UBATCH_TRACE [TAG_UBATCH_TRACE]
    struct ubatch_trace_times {
        int64_t      t0    = 0;
        int64_t      pre   = 0;
        int64_t      build = 0;
        int64_t      alloc = 0;
        const char * why   = nullptr; // why the graph was rebuilt
    };

    // process_ubatch's first part: the memory applied, the graph chosen (reused, or built and allocated) and the
    // scheduler's caller_sync set for it (the caller clears it); nullptr with ret set on a failure
    llm_graph_result * process_ubatch_graph(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret,
                ubatch_trace_times & tt);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // C738 (LLAMA_VERIFY_PREPARE, llama_decode_prepare): the host part of a decode done before the ids of some of its
    // tokens are known: the batch split, the memory's slots applied, the graph chosen and the inputs that need no token
    // id set (with LLAMA_VERIFY_PREPARE=2 also the first device split's graph inputs uploaded). The next decode() submits
    // it: it must bring the same batch, token ids aside, or the process aborts. Returns 0 if prepared, 1 if not (then
    // nothing was done that the next decode would not do the same way first), < 0 on an error
    int  decode_prepare(const llama_batch_ext & batch_inp);
    bool decode_prepared() const { return vprep.active; }
    // a prepared decode that will not be submitted (error paths): the sequences it holds lose every cell (the memory
    // cannot give back exactly what apply() took before the graph ran); false if there were any
    bool decode_prepare_drop();

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    // numbered device stores (llama_state_seq_get_data_store)
    size_t state_seq_get_data_store(llama_seq_id seq_id, int32_t store,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data_store(llama_seq_id seq_id, int32_t store, const uint8_t * src, size_t size, llama_state_seq_flags flags);
    size_t state_seq_store_get_size_host(llama_seq_id seq_id, int32_t store);
    size_t state_seq_store_get_data_host(llama_seq_id seq_id, int32_t store, const uint8_t * src, size_t src_size, uint8_t * dst, size_t dst_size);
    void   state_seq_store_free(llama_seq_id seq_id, int32_t store);

    // C217: micro-batch cuts for the following decodes (llama_set_ubatch_cuts), and device-store fills inside the next
    // decode (llama_state_seq_store_at, llama_state_seq_store_at_get)
    void   set_ubatch_cuts(const uint32_t * cuts, size_t n);
    void   state_seq_store_at(llama_seq_id seq_id, int32_t store, uint32_t i_end, llama_state_seq_flags flags);
    size_t state_seq_store_at_get(llama_seq_id seq_id, int32_t store, uint8_t * dst, size_t size, llama_pos * pos_min, llama_pos * pos_max) const;

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count,
 llama_state_seq_flags   flags = 0);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    // (re)allocate the device buffer for cparams.embeddings_layer_inp_dev
    void layer_inp_dev_alloc();

    // (re)allocate the device buffer for cparams.embeddings_nextn_dev, false if it does not fit
    bool h_nextn_dev_alloc();

    // C813: make the second half of a one-half buffer, the rows of the first copied over; false if it does not fit
    bool h_nextn_grow();

    ggml_backend_t backend_from_dev(ggml_backend_dev_t dev) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    //
    // pipelined single-token decoding (experimental, see llama_pipe_* in llama.h)
    //

    int32_t pipe_init(int32_t n_slots, int32_t n_chunk_slots, int32_t n_chunk_tokens);
    int32_t pipe_n_slots() const { return (int32_t) pipe_slots.size(); }
    int32_t pipe_submit(int32_t i_slot, const llama_token * tokens, int32_t n_tokens, llama_pos pos, llama_seq_id seq_id);
    int32_t pipe_submit_fed(int32_t i_slot, int32_t i_slot_src, llama_pos pos, llama_seq_id seq_id);
    float * pipe_get_logits(int32_t i_slot);
    llama_token pipe_get_sampled(int32_t i_slot);
    bool    pipe_ready(int32_t i_slot);
    void    pipe_drain();

    // staircase drafter (experimental, see llama_pipe_draft_* in llama.h)
    int32_t pipe_draft_init(const llama_pipe_draft_params & params);
    void    pipe_draft_reset(llama_seq_id seq_id, llama_pos n_keep);
    llama_pipe_drafter * get_pipe_drafter() const { return pipe_drafter.get(); }

private:
    llm_graph_result * get_gf_res_prev(uint32_t n_tokens = 0);

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;
    llm_graph_cb graph_get_cb(ggml_backend_sched_t sched_graph) const;

    // create a new backend scheduler for a graph of the specified size and register the callbacks below
    void sched_new(size_t max_nodes, bool parallel);

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;
    llama_moe_cache_ptr moe_cache;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;

        // the attached chains when the scheduler was last reserved, by their samplers' names, sorted: a later set
        // within these needs no new reserve (LLAMA_SAMPLER_RESERVE=1: every attach and detach reserves, as stock)
        std::vector<std::string> reserved_chains;
        bool                     reserve_always = false;
    };

    // whether the reserve covers the attached samplers' graphs
    bool sampler_reserve_covered() const;

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    // logits of the last decode that a backend sampler sampled stay on the device: get_logits_ith fetches such a row
    // on demand, as long as no later ubatch of that decode, or a later decode, has reused the compute buffers
    ggml_tensor *        logits_dev_tensor  = nullptr; // the logits of the decode's last ubatch
    std::vector<int32_t> logits_dev_rows;              // per output row: its row in logits_dev_tensor, -1 on the host, -2 lost

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    // Small batches also get one arena per token count, so speculative verify batches of varying
    // size keep their own CUDA graphs instead of re-capturing one graph at every size change.
    std::vector<std::pair<uint32_t, llm_graph_result_ptr>> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // [TAG_SCHED_PER_SIZE] wip: with LLAMA_SCHED_PER_SIZE=1 the arenas of batches of up to 8 tokens get their own
    // scheduler, so alternating batch sizes (the MTP catch-up and draft steps) keep their graphs allocated and reuse
    // them, instead of rebuilding and re-splitting one graph in the shared scheduler at every size change
    struct sched_small_entry {
        uint32_t               key;
        ggml_backend_sched_ptr sched;
        llm_graph_result     * ready; // the graph allocated in this scheduler, nullptr when it has to be rebuilt
        bool                   in_flight; // with mtp_graph_keep: a graph of this scheduler submitted since the last wait
        bool                   planned;       // LLAMA_SCHED_PER_SIZE_RESERVE: planned over the whole memory (sched_small_plan)
        bool                   planned_extra; // and with the shortlist head's further ids in that plan
    };
    std::vector<sched_small_entry> sched_small;
    bool sched_per_size = false;

    // env: LLAMA_SCHED_PER_SIZE_RESERVE=1 (C803), with per-size schedulers (LLAMA_MTP_GRAPH_KEEP=1 or
    // LLAMA_SCHED_PER_SIZE=1): a per-size scheduler is planned once, when it is made, from its batch's graph over the
    // whole memory (and the head shortlist's whole room), as the shared scheduler is at the context's start, so that no
    // later graph of its shape outgrows the plan; without it the first graph's plan grows with the context, a re-plan
    // and a larger compute buffer each time (FN84, R60, R85: the MTP draft's 2 to 5 reallocations a request)
    bool sched_small_reserve = false;
    llm_graph_result_ptr gf_res_plan; // the graph sched_small_plan builds, never computed
    bool sched_small_plan(sched_small_entry & ss, const llama_ubatch & ubatch, llm_graph_type gtype);

    // env: LLAMA_MTP_GRAPH_KEEP=1 (C619), an MTP context only: schedulers per size, keyed also by the head's shortlist
    // and the hidden-row carry (gf_res_key), and a per-size scheduler waits only for its own graph in flight
    bool mtp_graph_keep = false;

    // env: LLAMA_GRAPH_PER_SIZE=0 keeps the two arenas of llama.cpp before #31156 (batches with and without outputs), so
    // verify batches of every width share one arena and one CUDA graph key, as on -ladder2; 1 (default, as #31156) gives
    // batches of up to 8 tokens one arena per token count. LLAMA_SCHED_PER_SIZE=1 needs it on.
    bool graph_per_size = true;

    // scheduler of the graph of the last process_ubatch(), nullptr for the shared one
    ggml_backend_sched_t sched_active = nullptr;

    // C738: the decode prepared by decode_prepare(), and the mode decode() and process_ubatch() run in: 0 a plain call,
    // 1 preparing (up to the inputs that need no token id), 2 submitting the prepared call (from the other inputs on)
    struct verify_prep {
        bool                                 active = false;
        std::vector<llama_batch_ext::token>  tokens; // the prepared batch, which the next decode must bring
        llama_memory_context_ptr             mctx;   // applied
        llm_graph_result                   * res   = nullptr;
        ggml_backend_sched_t                 sched = nullptr; // sched_active of the prepared ubatch
        int32_t                              h_nextn_w = 0;
        // the settings the graph was chosen with
        bool                                 embeddings  = false;
        bool                                 causal_attn = true;
        bool                                 screen_head = false;
        std::map<llama_seq_id, llama_sampler *> samplers;
        // the first device split's inputs uploaded ahead (LLAMA_VERIFY_PREPARE=2)
        int                                  n_preup     = 0;
        size_t                               preup_bytes = 0;
    } vprep;
    int vprep_mode  = 0;
    int vprep_level = 0; // LLAMA_VERIFY_PREPARE, read at construction (2: also the uploads)
    uint64_t vprep_n_prepared = 0;
    uint64_t vprep_n_refused  = 0;
    uint64_t vprep_n_dropped  = 0;
    uint64_t vprep_n_preup    = 0; // inputs uploaded ahead, in all

    // C738: the next decode() brings the prepared batch: compare it, and put its token ids where the prepared call keeps
    // them (the batch allocator, the ubatch, the memory's cells)
    void vprep_take_batch(const llama_batch_ext & batch_inp);

    ggml_backend_sched_t sched_res() const { return sched_active ? sched_active : sched.get(); }

    uint32_t gf_res_key(uint32_t n_tokens) const;

    // forget which graphs are allocated, after the graph results were reset
    void gf_res_invalidate();

    // one in-flight single-token evaluation of pipelined decoding: its own scheduler (inputs, inter-device copies and
    // compute buffers), its own graph (and therefore its own CUDA graphs), and a pinned buffer for its logits.
    // slots share only the weights and the memory module; every device runs the queued evaluations in submission order
    struct pipe_slot {
        ggml_backend_sched_ptr  sched;
        llm_graph_result_ptr    res;
        bool                    res_valid = false; // res holds a graph allocated on sched

        ggml_backend_event_t    ev_out  = nullptr; // recorded after the logits copy
        ggml_backend_dev_t      ev_dev  = nullptr;
        ggml_backend_buffer_ptr buf_out;
        float *                 logits  = nullptr;

        bool                    busy    = false;   // submitted and not yet waited for

        int32_t                 n_tokens_max = 1;  // tokens per evaluation the compute buffers are reserved for

        // the copy of the token input on the device that embeds the tokens, if no other device and not the CPU reads
        // the token ids (set when the graph is allocated); a fed submission writes the token there
        ggml_tensor *           tok_cpy     = nullptr;
        ggml_backend_t          tok_backend = nullptr;

        // the backends of the slot's scheduler: the context's, or for a chunk slot with LLAMA_PIPE_CHUNK_STREAM the
        // chunk backends in place of the GPU ones
        std::vector<ggml_backend_t> backends;

        // ggml_backend_event_query of the output device, if it has one
        bool (*ev_query)(ggml_backend_event_t) = nullptr;

        // its evaluation is still to be read by the executor (set at submit; cleared when its output is read, when
        // its sequence is submitted again at or before its position, or decoded with llama_decode): of a recurrent
        // model's sequence, the state before the oldest such token is one a rollback may still need
        bool                    live = false;
        llama_seq_id            seq  = -1;
        llama_pos               pos  = -1; // of its first token
    };

    // before slot i_slot writes seq_id's recurrent state for pos: saves the state a rollback may still need if the
    // slot's state group holds it (see llama_memory_recurrent::pipe_stash)
    void pipe_protect_state(int32_t i_slot, llama_seq_id seq_id, llama_pos pos);

    // the recurrent part of the memory for a hybrid or recurrent model (null otherwise): each pipe slot writes its
    // states to its own group of it (llama_memory_recurrent::n_grp), so a wrong guess can be dropped
    llama_memory_recurrent * pipe_mem_recr = nullptr;

    // LLAMA_PIPE_CHUNK_STREAM=1: a second backend per GPU for the chunk slots, so that prompt chunks run on streams of
    // their own; the context's GPU backends are created with the greatest stream priority, these with the default
    // (lowest), so a token's kernels are scheduled ahead of a chunk's on every card. Declared before pipe_slots, whose
    // schedulers use them, so that it is destroyed after them
    bool pipe_chunk_stream = false;
    std::vector<ggml_backend_ptr> pipe_chunk_backends;

    std::vector<pipe_slot> pipe_slots;

    // the staircase drafter fed by the pipe slots' taps and llama_decode's prompt captures (llama_pipe_draft_*)
    std::unique_ptr<llama_pipe_drafter> pipe_drafter;

    // the output head's shortlist (head_shortlist_init), referenced by cparams.head_short
    std::unique_ptr<llama_head_shortlist> head_short;

    // the hidden rows' carry on the layer's card (nextn_carry_init), referenced by cparams.nextn_carry
    ggml_context_ptr        nextn_carry_ctx;
    ggml_backend_buffer_ptr nextn_carry_buf;

    // the token embedding rows on the layer's card (embd_cache_init), referenced by cparams.embd_cache, and the backend
    // that runs the graphs reading them
    std::unique_ptr<llama_embd_cache> embd_cache;
    ggml_backend_t                    embd_cache_backend = nullptr;

    // LLAMA_TENSOR_HASH: the nodes hashed after they are computed, cparams.cb_eval's user data when set (checks)
    llama_tensor_hash tensor_hash;
    // LLAMA_NGRAM_PREFETCH (C761): the n-gram rows of a call's later micro-batches prefetched by a helper thread, or
    // only counted (LLAMA_UBATCH_TRACE, LLAMA_NGRAM_PREFETCH_STATS); nullptr when neither is set or the model has no
    // mapped n-gram table. ngram_pf_begin plans a decode call (the batch, the first micro-batch's size and the tokens
    // before the call from the attention cells)
    std::unique_ptr<llama_ngram_prefetch> ngram_pf;
    void ngram_pf_begin(uint32_t n_tokens_all, uint32_t n_first);

    // LLAMA_UBATCH_TRACE: the last micro-batch's input times (C761)
    std::vector<llm_graph_result::input_time> trace_inputs;

public:
    const llama_ngram_prefetch * get_ngram_pf() const { return ngram_pf.get(); }

private:

    // the exact bound and the full-head fallback for the main model's shortlist head (head_bound_init); requested with
    // set_head_bound, used by the decodes with one output
    std::unique_ptr<llama_head_bound> head_bound;
    bool head_bound_req = false;

    // the bound and the counts of the last shortlist row, once
    void head_bound_finish_row();

    // the routed experts kept per token and MoE layer (cparams.moe_tail_hist): moe_kept_hist[n] counts the (token,
    // layer) pairs that kept n experts, moe_kept_layer the experts kept per MoE layer of the graph, in graph order
    void moe_tail_collect(const llm_graph_result * res, uint32_t n_tokens);
    std::vector<uint64_t> moe_kept_hist;
    std::vector<double>   moe_kept_layer;
    std::vector<float>    moe_kept_buf;
    uint64_t              moe_kept_tokens   = 0;
    uint64_t              moe_kept_next_log = 0;

    // the distinct routed experts per block of route once and MoE layer (cparams.moe_route_once_hist), counted over
    // the full blocks (k tokens): read[d] / indep[d] the (block, layer) pairs whose tokens read d distinct experts /
    // would read them under independent routing (the router's own top-k), per MoE layer in graph order; own[c] the
    // later tokens' (token, layer) pairs whose own top-k has c experts in their block's first token's set
    void moe_route_once_collect(const llm_graph_result * res, const llama_ubatch & ubatch, uint32_t n_out);
    struct route_once_stats {
        std::vector<std::vector<uint64_t>> read;
        std::vector<std::vector<uint64_t>> indep;
        std::vector<uint64_t> own;
        std::vector<int32_t>  ids_read;
        std::vector<int32_t>  ids_indep;
        std::vector<int32_t>  lead_all;
        std::vector<int32_t>  lead_out;
        std::vector<uint64_t> seen_read;  // per expert: the stamp of the last block that counted it
        std::vector<uint64_t> seen_indep;
        std::vector<uint64_t> seen_lead;  // per expert: the stamp of the block whose first token's set holds it
        uint64_t stamp      = 0;
        uint64_t n_tokens   = 0;
        uint64_t n_short    = 0; // (block, layer) pairs of shorter blocks (a micro-batch's or a sequence's end)
        uint64_t n_outside  = 0; // later tokens' (token, layer) pairs that read an expert outside their block's set
        uint64_t next_log   = 0;
    } ro;

    // how the slots upload their host inputs (ggml_backend_sched_set_input_upload, LLAMA_PIPE_UPLOAD)
    int pipe_upload = 2;

    // how llama_decode uploads its host inputs when nothing is in flight (ggml_backend_sched_set_input_upload with
    // caller_sync, LLAMA_INPUT_UPLOAD): 2 where every device is of compute capability 6.1, 0 (stock) elsewhere
    int plain_upload = 0;

    // i_slot_src >= 0: the token is the sampled output of that slot, copied between the devices (pipe_submit_fed)
    int32_t pipe_submit_impl(int32_t i_slot, const llama_token * tokens, int32_t n_tokens, llama_pos pos,
                             llama_seq_id seq_id, int32_t i_slot_src);

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // device buffer with enabled layer inputs, [n_layer_inp*n_embd, n_batch], row = batch index
    ggml_context_ptr        ctx_layer_inp_dev;
    ggml_backend_buffer_ptr buf_layer_inp_dev;
    ggml_tensor *           layer_inp_dev = nullptr;
    ggml_backend_dev_t      layer_inp_dev_dev = nullptr;
    ggml_backend_t          layer_inp_backend = nullptr;

    // ordering with the reader context (no host sync):
    //   written - recorded by this context after the batch, waited on by the reader
    //   read    - recorded by the reader after it consumed the rows, waited on by this context
    ggml_backend_event_ptr ev_layer_inp_written;
    ggml_backend_event_ptr ev_layer_inp_read;
    bool ev_layer_inp_read_recorded = false;

    // device buffer with the h_nextn rows, F32 [n_embd_out, n_batch + n_seq_max]:
    // row r < n_batch is the token at batch index r of the last decode, row n_batch + s is set by the host for seq s
    // ordered with the reader context like the layer input buffer
    ggml_context_ptr        ctx_h_nextn_dev;
    ggml_backend_buffer_ptr buf_h_nextn_dev;
    ggml_tensor *           h_nextn_dev = nullptr;
    ggml_backend_dev_t      h_nextn_dev_dev = nullptr;
    ggml_backend_t          h_nextn_backend = nullptr;
    // C703: with two halves (h_nextn_halves = 2, LLAMA_MTP_CATCHUP_OVERLAP) the buffer holds the rows above twice, a
    // decode writes the half of the last one unless a reader holds those rows (hold_h_nextn()), then the other half;
    // each half has its own events. One half (the default) is the buffer above, events [0]
    int32_t                 h_nextn_halves    = 1;
    int32_t                 h_nextn_last      = 0;     // the half the last decode wrote
    bool                    h_nextn_held      = false; // the next decode writes the other half
    int64_t                 h_nextn_row0      = 0;     // the first row the decode's graphs write, see build_h_nextn_dev()
    ggml_backend_event_ptr  ev_h_nextn_written[2];
    ggml_backend_event_ptr  ev_h_nextn_read[2];
    bool ev_h_nextn_written_recorded[2] = { false, false };
    bool ev_h_nextn_read_recorded[2]    = { false, false };
    // LLAMA_MTP_HROW_LAZY (C813): with two halves asked for, the buffer keeps one half until the first hold, which makes the
    // second (1), the held rows copied over, so the second half's memory is used only once a call's rows are held while
    // the next call runs; 2 never makes it (a check of the no-room path). Without room the hold is refused for good and
    // the reader reads the rows before the next decode, as with one half
    int32_t                 h_nextn_lazy        = 0;
    int32_t                 h_nextn_halves_want = 1;     // the halves set_h_nextn_halves() asked for
    bool                    h_nextn_no_room     = false;
    // LLAMA_MTP_H_DEV (C693): 1 (default) the buffer above where the output device has events (upstream #31927), 0 the
    // rows extracted to host memory (the path before #31927), 2 as 1 and also on the CPU without events (checks)
    int32_t                 h_nextn_dev_mode  = 1;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    // numbered device copies, by (seq_id, store); the mutex guards the map, not the stores (a store's reader on another
    // thread and its writer are kept apart by the caller)
    std::map<std::pair<llama_seq_id, int32_t>, llama_state_store> state_stores;
    std::mutex state_stores_mtx;

    // C217: the caller's micro-batch cuts, and the device-store fills requested for the next decode (store_at_next)
    // and taken by the last one (store_at_last, with their blobs)
    struct store_at_req {
        llama_seq_id          seq_id;
        int32_t               store;
        uint32_t              i_end;
        llama_state_seq_flags flags;

        bool      done    = false;
        llama_pos pos_min = -1;
        llama_pos pos_max = -1;

        std::vector<uint8_t> blob;
    };
    std::vector<uint32_t>     ubatch_cuts;
    std::vector<store_at_req> store_at_next;
    std::vector<store_at_req> store_at_last;

    // fills r's store from the state as the host holds it now, the copies queued behind the work queued so far
    bool state_seq_fill_store_at(store_at_req & r);

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // env: LLAMA_GRAPH_REUSE_WAIT=1 reuses a graph under pipeline parallelism while a ubatch is in flight, after waiting
    // for all devices (the behavior before the in-flight rule)
    bool graph_reuse_wait = false;

    // Q771: the micro-batch taper (env: LLAMA_UBATCH_TAPER, llama_set_ubatch_taper), its spec, and the hints for the
    // next decode call (llama_set_ubatch_taper_next: -1 default, 0 no, 1 yes), which that call consumes
    llama_ubatch_taper taper;
    std::string        taper_spec      = "off";
    int32_t            taper_next_head = -1;
    int32_t            taper_next_tail = -1;

    // C742: the next decode copies no h_nextn rows to the host (llama_skip_embeddings_nextn_next), which that call
    // consumes; since the context was created, the micro-batches that copied their rows to the host and those that left
    // them out for the mark
    bool     h_nextn_skip_next = false;
    uint64_t n_h_nextn_host    = 0;
    uint64_t n_h_nextn_skipped = 0;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    // a graph was handed to the scheduler since the last synchronize(), so with pipeline parallelism it may still be
    // computing on the devices
    mutable bool gf_in_flight = false;

    // LLAMA_STEP_TRACE (C405, llama-step-trace.h): mark the stages of decode() and the waits, as "<role>.<stage>"
    bool step_trace      = false;
    char step_trace_role = 'v';

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
