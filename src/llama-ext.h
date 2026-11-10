#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// Write the enabled layer inputs to a device buffer [n_layer_inp*n_embd, n_batch] instead of host memory.
// Row r holds the token at batch index r, layers in ascending layer id order.
LLAMA_API void llama_set_embeddings_layer_inp_dev(struct llama_context * ctx, bool value);

// Whether decode accepts entries added with llama_batch_ext_add_embd_row()
// (needs a DFlash/DSpark decoder context whose ctx_other has a layer input device buffer).
LLAMA_API bool llama_supports_embd_row(const struct llama_context * ctx);

// Add an entry whose embedding is row `row` of the layer input device buffer of ctx_other.
// All entries of the batch must be of this kind. Positions are set with llama_batch_ext_set_pos(), n_pos_per_embd of them.
// Returns the batch index (>= 0), or the llama_batch_ext_add() error codes.
LLAMA_API int32_t llama_batch_ext_add_embd_row(struct llama_batch_ext * batch, llama_seq_id seq_id, int32_t row);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// A shortlist for a draft context's output head (Q592): while it is on, a decode with one output computes the logits
// of the token ids below k_low and of up to m_cap further ids only, and gives every other token -INFINITY. The
// further ids are added with llama_head_shortlist_add (ids already in the list and ids past the room are skipped).
// Supported by the qwen4exp MTP draft. Returns false when the head can't take one (scaled, repacked or split).
LLAMA_API bool     llama_head_shortlist_init    (struct llama_context * ctx, uint32_t k_low, uint32_t m_cap);
LLAMA_API uint32_t llama_head_shortlist_add     (struct llama_context * ctx, const llama_token * ids, size_t n_ids);
LLAMA_API bool     llama_head_shortlist_contains(const struct llama_context * ctx, llama_token id);
LLAMA_API uint32_t llama_head_shortlist_size    (const struct llama_context * ctx); // ids in the list, 0 without one
LLAMA_API void     llama_set_head_shortlist     (struct llama_context * ctx, bool value);

// A draft context's token embedding rows on the card that runs its layer (Q961): n_rows rows of the table, in its own
// type and bytes, which every batch of at most max_tokens tokens reads instead of the table (a row is copied from the
// table when a token first needs it, into the slot read longest ago); larger batches read the table. So a draft pass
// with the table on another card (-otd token_embd=CUDA<n>) runs on one card, with the same numbers. Supported by the
// qwen4exp MTP draft; false when the context has no MTP layer or the layer's card can't read the table's type.
LLAMA_API bool llama_embd_cache_init (struct llama_context * ctx, uint32_t n_rows, uint32_t max_tokens);
// tokens read from the cache, of them copied from the table first, and batches that read the table; false without one
LLAMA_API bool llama_embd_cache_stats(const struct llama_context * ctx, uint64_t * n_hit, uint64_t * n_miss, uint64_t * n_plain);

// A draft context's hidden rows kept on the card that runs its layer between passes (Q961): one row per sequence. While
// in is set, a decode's tokens take their h from their sequence's row instead of the batch's state embedding (which
// must still be set, and is not read); while out is set, a decode writes each output's h to its sequence's row and
// copies none to the host (llama_get_embeddings_nextn_ith then has nothing new). So a step's drafting passes send no
// hidden row over the bus. Supported by the qwen4exp MTP draft; false without an MTP layer.
LLAMA_API bool llama_nextn_carry_init(struct llama_context * ctx);
LLAMA_API void llama_set_nextn_carry (struct llama_context * ctx, bool in, bool out);

// Host timestamps along the server's steps (C405), with LLAMA_STEP_TRACE=<file> set: the marks of a step (a name and
// the steady clock's time) are kept until llama_step_trace_end writes them as one JSON line, if a context marked a stage
// since llama_step_trace_begin. The file's header gives the steady clock's time of its 0 ms, as GGML_CUDA_PROFILE_HOST's
// log line does for its own, so the step lines go beside that profile's per-card times. A context marks its decode's
// stages and its waits itself as "<role>.<stage>": 'v' for a main context, 'd' for an MTP context, or what
// llama_step_trace_role sets. With the variable unset every call returns at once and a context checks one flag.
// Names must outlive the step (string literals). One thread marks at a time.
LLAMA_API bool llama_step_trace_on   (void);
LLAMA_API void llama_step_trace_begin(const char * name);                // drops what is kept, then marks name
LLAMA_API void llama_step_trace_mark (const char * name);
LLAMA_API void llama_step_trace_count(const char * name, int64_t value); // added up per name on the step's line
LLAMA_API void llama_step_trace_end  (void);
LLAMA_API void llama_step_trace_role (struct llama_context * ctx, char role);

// The main model's head over the context's shortlist (llama_head_shortlist_init) with an exact bound on every other
// logit (Q619): while it is on, a decode with one output computes the shortlist's logits and gives every other token
// -INFINITY; llama_head_bound_ith(i) is then at least every left-out logit the full head would have computed (Cauchy-
// Schwarz on the head's input row and the head's row norms), and +INFINITY for a row with the full head's logits.
// A sampler that can't exclude every left-out token with it calls llama_head_bound_full(i), which computes the full
// head for that row with the same kernel and writes it into the logits. The context's tokens join the shortlist as
// they are decoded. Supported by the qwen4exp main graph; init reads the head's rows once (norms) and returns false
// for heads it can't bound (scaled, with a precision policy, or without a shortlist).
LLAMA_API bool  llama_head_bound_init (struct llama_context * ctx);
LLAMA_API void  llama_set_head_bound  (struct llama_context * ctx, bool value);
LLAMA_API float llama_head_bound_ith  (struct llama_context * ctx, int32_t i);
LLAMA_API bool  llama_head_bound_full (struct llama_context * ctx, int32_t i);
LLAMA_API void  llama_head_bound_stats(const struct llama_context * ctx, uint64_t * n_rows, uint64_t * n_full);

// Fewer routed experts on confident tokens (Q618): after the router's top-k, a slot's weight becomes 0 when the slots
// ranked above it already hold a share p of the token's routed weight, or its own share is below w (w < 1/k); renorm
// rescales the kept weights to the token's sum of its k weights. hist counts the experts kept per token and MoE layer
// (a synchronous read per decode), logged every 65,536 tokens and when the context is freed.
// LLAMA_MOE_TAIL=p=<p>,w=<w>,renorm=<0|1>,hist=<0|1>[,graph=<0|1>,skip=<n>,repeat=<0|1>] turns it on for every context. llama_moe_tail_counts fills counts[n] with the (token, layer) pairs that kept n experts,
// n < n_counts, and returns the pairs counted.
LLAMA_API void     llama_set_moe_tail   (struct llama_context * ctx, bool on, float p, float w, bool renorm, bool hist);
LLAMA_API uint64_t llama_moe_tail_counts(const struct llama_context * ctx, uint64_t * counts, int32_t n_counts);

// How the cut runs (Q660, Q661): by default GGML_OP_MOE_TAIL gives a dropped slot the id -1 in micro-batches of up to
// skip tokens (default 8), so that the expert kernels never read its rows, or with repeat its token's last kept expert
// (read again, at weight 0); larger micro-batches keep every slot's expert at weight 0. graph keeps the graph-level ops
// instead, which weight the dropped experts 0 and still compute them. LLAMA_MOE_TAIL's graph=<0|1>, skip=<n> and
// repeat=<0|1> set it. Call it before llama_set_moe_tail.
LLAMA_API void     llama_set_moe_tail_mode(struct llama_context * ctx, bool graph, int32_t skip, bool repeat);

// Route once per verify block (Q860): the tokens of a micro-batch form blocks of up to k consecutive positions of one
// sequence; a block's first token picks its routed experts as usual and every later token picks its own top-k only
// among those (its own router scores masked to that set, then the usual weight normalization), so a pass over the
// block reads one token's experts per layer. Lossy. k = 0: off; k = 1: the masked path with every token its own
// block's first (the same experts). max > 0: only micro-batches of at most max tokens (verify batches, not prompts).
// hist counts the distinct experts per full block and MoE layer, as read and under independent routing (a synchronous
// read per decode), logged every 65,536 tokens and when the context is freed; dry keeps the routing as it is and only
// counts. LLAMA_MOE_ROUTE_ONCE=<k>[,max=<n>][,hist=<0|1>][,dry=<0|1>] turns it on for every context.
// llama_moe_route_once_counts fills read[d] / indep[d] (d < n_counts) with the full blocks' (block, layer) pairs that
// read d distinct experts / would read them under independent routing, sets *n_outside to the later tokens'
// (token, layer) pairs that read an expert outside their block's first token's set, and returns the pairs counted.
LLAMA_API void     llama_set_moe_route_once   (struct llama_context * ctx, uint32_t k, uint32_t max, bool hist,
        bool dry);
LLAMA_API uint64_t llama_moe_route_once_counts(const struct llama_context * ctx, uint64_t * read, uint64_t * indep,
        int32_t n_counts, uint64_t * n_outside);

// The cells the recurrent memory's slot finder has moved to another row so far (Q1158): a moved sequence keeps its
// final state but not its rollback snapshots. 0 for a context without a recurrent memory. For tests and diagnostics.
LLAMA_API uint64_t llama_memory_recurrent_n_swaps(const struct llama_context * ctx);

// Write the unmasked h_nextn rows (see llama_set_embeddings_nextn) of each decode to a device buffer
// [n_embd_out, n_batch + n_seq_max] instead of host memory: row r < n_batch is the token at batch index r of the last decode,
// row n_batch + s is free for the caller, e.g. for seq s (see llama_set_embeddings_nextn_rows()).
// Returns false if the buffer cannot be used, the rows are then extracted to host memory.
LLAMA_API bool llama_set_embeddings_nextn_dev(struct llama_context * ctx, bool value);

// Copy rows of the h_nextn device buffer to or from host memory, ordered after the last decode and its readers
LLAMA_API bool llama_get_embeddings_nextn_rows(struct llama_context * ctx, int64_t row0, int64_t n_rows, float * out);
LLAMA_API bool llama_set_embeddings_nextn_rows(struct llama_context * ctx, int64_t row0, int64_t n_rows, const float * data);

// Read the state rows of batches (see llama_batch_ext_set_embd_state_row()) from the h_nextn device buffer of src.
// Returns whether decode accepts batches with state rows.
LLAMA_API bool llama_set_embd_state_src(struct llama_context * ctx, struct llama_context * src);

// The state of entry idx is row `row` of the h_nextn device buffer of the context set with llama_set_embd_state_src().
// All entries of a batch have a state row, or none.
LLAMA_API bool llama_batch_ext_set_embd_state_row(struct llama_batch_ext * batch, int32_t idx, int32_t row);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);
