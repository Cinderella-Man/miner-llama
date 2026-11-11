#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama-step-trace.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// LLAMA_STEP_TRACE (C405): a host timestamp of this context's stage; one flag when the variable is unset
#define LLAMA_STEP_MARK(name) do { if (step_trace) { llama_step_trace_put(step_trace_role, name); } } while (0)

//
// llama_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

// LLAMA_TENSOR_HASH (C762, checks): the scheduler's eval callback; asks for the nodes whose names start with a prefix,
// then hashes each one's bytes and logs it with its type, shape and the buffer that holds it
static bool llama_tensor_hash_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * th = (llama_tensor_hash *) user_data;
    bool want = false;
    for (const auto & p : th->prefixes) {
        if (strncmp(t->name, p.c_str(), p.size()) == 0) {
            want = true;
            break;
        }
    }
    if (ask || !want) {
        return ask ? want : true;
    }
    uint64_t h = 0xcbf29ce484222325ULL;
    const bool readable = t->buffer != nullptr && ggml_is_contiguous(t);
    if (readable) {
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        for (const uint8_t b : bytes) {
            h ^= b;
            h *= 0x100000001b3ULL;
        }
    }
    LLAMA_LOG_INFO("tensor-hash: %c %" PRIu64 " %s %s [%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] %s %016" PRIx64 "%s\n",
            th->role, th->n++, t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            t->buffer ? ggml_backend_buffer_name(t->buffer) : "none", readable ? h : 0, readable ? "" : " (not read)");
    return true;
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_flash_attn_kv_rows_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN_KV_ROWS,
    /*.name             =*/ "Flash Attention over K/V rows",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_top_k_mask_probe = {
    /*.op               =*/ LLM_FUSED_OP_TOP_K_MASK,
    /*.name             =*/ "fused top-k mask",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    step_trace      = llama_step_trace_active();
    step_trace_role = params.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? 'd' : 'v';

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && (!llm_arch_supports_rs_rollback(model.arch) || !hparams.causal_attn)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    cparams.embeddings_layer_inp_dev = false;
    cparams.embeddings_nextn_dev     = false;
    cparams.ctx_state_src            = nullptr;
    embd_layer_inp.resize(hparams.n_layer() + 1);
    cparams.pipe_tap_layer.resize(hparams.n_layer() + 1, false);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // LLAMA_TENSOR_HASH=<prefix>[,<prefix>...] (C762, checks): the named nodes hashed once computed, unless the caller
    // has an eval callback of its own
    if (const char * s = getenv("LLAMA_TENSOR_HASH"); s != nullptr && *s != '\0' && cparams.cb_eval == nullptr) {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                tensor_hash.prefixes.push_back(item);
            }
        }
        if (!tensor_hash.prefixes.empty()) {
            tensor_hash.role          = params.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? 'd' : 'v';
            cparams.cb_eval           = llama_tensor_hash_eval;
            cparams.cb_eval_user_data = &tensor_hash;
            LLAMA_LOG_INFO("%s: LLAMA_TENSOR_HASH: %zu name prefixes hashed after each evaluation\n", __func__,
                    tensor_hash.prefixes.size());
        }
    }

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        } else if (model.arch == LLM_ARCH_DFLASH) {
            // used to read the target's layer input device buffer
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    // the KV cache is shared with ctx_other, so follow its flash attention setting instead of resolving it here:
    // the probe disables it when the layers and the shared KV are on different devices, which breaks a quantized V cache
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        const bool flash_attn_other = cparams.ctx_other->get_cparams().flash_attn;
        if (!cparams.auto_fa && cparams.flash_attn != flash_attn_other) {
            LLAMA_LOG_WARN("%s: flash_attn = %d does not match the context the KV cache is shared with, using %d\n",
                    __func__, cparams.flash_attn, flash_attn_other);
        }
        cparams.flash_attn = flash_attn_other;
        cparams.auto_fa    = false;
    }

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // only the indexer models build this mask, skip the probe for the others
    cparams.fused_top_k_mask = true;
    cparams.auto_ftkm        = hparams.indexer_top_k > 0;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    if (const char * env = getenv("LLAMA_SAMPLER_RESERVE")) {
        sampling.reserve_always = atoi(env) != 0;
    }

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size      = params.moe_cache_size;
    cparams.moe_cache_cpu_share = params.moe_cache_cpu_share;

    // resolved after flash attention, see resolve_fused_ops()
    {
        cparams.fused_kv_rows = false;

        const char * LLAMA_KV_ROWS_DISABLE = getenv("LLAMA_KV_ROWS_DISABLE");
        if (!LLAMA_KV_ROWS_DISABLE) {
            cparams.auto_fkvr = cparams.kv_unified;
        } else {
            LLAMA_LOG_INFO("%s: LLAMA_KV_ROWS_DISABLE set, disabling kv_rows\n", __func__);
            cparams.auto_fkvr = false;
        }
    }

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    // the MoE down-projection rescale is dropped at one token by default where every device is of compute capability
    // 6.1, whose Pascal kernels compute the activation block sums in FP32 (llm_graph_moe_down_rescale)
    cparams.moe_down_rescale_tg = !model.devices.empty() &&
        llama_backend_dev_has_feature(model.devices[0].dev, "COMPUTE_6_1");

    // LLAMA_MTP_KV_ONLY=1: the draft's catch-up passes write only its layer's caches (Q1105); only the architectures
    // whose MTP graph calls llm_graph_context::kv_only_pass() use it (qwen4exp)
    {
        const char * LLAMA_MTP_KV_ONLY = getenv("LLAMA_MTP_KV_ONLY");
        cparams.mtp_kv_only = LLAMA_MTP_KV_ONLY ? atoi(LLAMA_MTP_KV_ONLY) != 0 : false;
        if (cparams.mtp_kv_only && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
            LLAMA_LOG_INFO("%s: LLAMA_MTP_KV_ONLY: a draft pass with no output row writes only its layer's caches\n", __func__);
        }
    }

    // screened output head: LLAMA_SCREEN_HEAD=1 enables it for models with an output_screen tensor,
    // LLAMA_SCREEN_HEAD_M=<M> overrides the model's number of candidates
    {
        const char * LLAMA_SCREEN_HEAD   = getenv("LLAMA_SCREEN_HEAD");
        const char * LLAMA_SCREEN_HEAD_M = getenv("LLAMA_SCREEN_HEAD_M");
        cparams.screen_head   = LLAMA_SCREEN_HEAD   ? atoi(LLAMA_SCREEN_HEAD) != 0 : false;
        cparams.screen_head_m = LLAMA_SCREEN_HEAD_M ? (uint32_t) atoi(LLAMA_SCREEN_HEAD_M) : 0;

        if (cparams.screen_head) {
            if (model.output_screen) {
                LLAMA_LOG_INFO("%s: screened output head on: %s screen, %u candidates\n", __func__,
                        ggml_type_name(model.output_screen->type),
                        cparams.screen_head_m ? cparams.screen_head_m : hparams.output_screen_top_m);
            } else {
                LLAMA_LOG_WARN("%s: LLAMA_SCREEN_HEAD is set but the model has no output_screen tensor\n", __func__);
            }
        }
    }

    // C691: LLAMA_SAMPLING_TOP_K_ROWS=1 takes the backend samplers' top-k of all output rows in one op (off by default;
    // read when the context is made)
    {
        const char * LLAMA_SAMPLING_TOP_K_ROWS = getenv("LLAMA_SAMPLING_TOP_K_ROWS");
        cparams.sampling_top_k_rows = LLAMA_SAMPLING_TOP_K_ROWS ? atoi(LLAMA_SAMPLING_TOP_K_ROWS) != 0 : false;
        if (cparams.sampling_top_k_rows) {
            LLAMA_LOG_INFO("%s: backend sampling: one top-k over the output rows (LLAMA_SAMPLING_TOP_K_ROWS=1)\n", __func__);
        }
    }

    // card-to-card hops: LLAMA_HOP_CAST=<type> sends the residual across every boundary between two GPUs of a layer
    // split in that type: cast on the card before the boundary, back to F32 on the card after it, so the scheduler
    // copies the narrow tensor (Flash-Next, 10,240 values a token: 40,960 B in F32, 20,480 in F16 or BF16, 10,880 in
    // Q8_0, 8,400 in Q6_K, 7,680 in Q5_1, 7,040 in Q5_0, 6,400 in Q4_1, 5,760 in Q4_0 or IQ4_NL; the types below 8 bits
    // are Q963's, each to be cleared by its KLD). The type's block must divide the residual's rows (Q6_K: 256), or that
    // hop stays F32. LLAMA_HOP_CAST_AT=<l1,l2,...> marks those layers instead (each the first layer after a boundary),
    // e.g. a rig's boundaries in a run on fewer cards or on the CPU. Unset, empty, 0, off or f32: off. Only the
    // architectures whose graph calls build_hop_cast use it (qwen4exp)
    {
        cparams.hop_type = GGML_TYPE_F32;
        cparams.hop_layer.assign(hparams.n_layer(), false);

        const char * LLAMA_HOP_CAST    = getenv("LLAMA_HOP_CAST");
        const char * LLAMA_HOP_CAST_AT = getenv("LLAMA_HOP_CAST_AT");
        if (LLAMA_HOP_CAST && *LLAMA_HOP_CAST && strcmp(LLAMA_HOP_CAST, "0") != 0 && strcmp(LLAMA_HOP_CAST, "off") != 0 &&
                strcmp(LLAMA_HOP_CAST, "f32") != 0) {
            // the type's name as ggml_type_name gives it, in any case (q6_k is q6_K)
            static const ggml_type hop_types[] = {
                GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0,
                GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL,
            };
            const auto same_name = [](const char * a, const char * b) {
                for (; *a && *b; ++a, ++b) {
                    if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) {
                        return false;
                    }
                }
                return *a == *b;
            };
            for (const ggml_type t : hop_types) {
                if (same_name(LLAMA_HOP_CAST, ggml_type_name(t))) {
                    cparams.hop_type = t;
                    break;
                }
            }
            if (cparams.hop_type == GGML_TYPE_F32) {
                LLAMA_LOG_WARN("%s: LLAMA_HOP_CAST=%s is not f16, bf16, q8_0, q6_k, q5_1, q5_0, q4_1, q4_0 or iq4_nl: hops "
                        "stay F32\n", __func__, LLAMA_HOP_CAST);
            }
        }

        if (cparams.hop_type != GGML_TYPE_F32) {
            const int n_layer_main = (int) hparams.n_layer();
            if (LLAMA_HOP_CAST_AT && *LLAMA_HOP_CAST_AT) {
                for (const char * p = LLAMA_HOP_CAST_AT; *p; ) {
                    char * end = nullptr;
                    const long il = strtol(p, &end, 10);
                    if (end == p) {
                        LLAMA_LOG_WARN("%s: LLAMA_HOP_CAST_AT: cannot read '%s'\n", __func__, p);
                        break;
                    }
                    if (il > 0 && il < n_layer_main) {
                        cparams.hop_layer[il] = true;
                    } else {
                        LLAMA_LOG_WARN("%s: LLAMA_HOP_CAST_AT: layer %ld is not in 1..%d\n", __func__, il, n_layer_main - 1);
                    }
                    p = *end == ',' ? end + 1 : end;
                }
            } else {
                for (int il = 1; il < n_layer_main; ++il) {
                    ggml_backend_dev_t dev_prev = model.dev_layer(il - 1);
                    ggml_backend_dev_t dev      = model.dev_layer(il);
                    if (dev != dev_prev && dev && dev_prev &&
                            ggml_backend_dev_type(dev)      == GGML_BACKEND_DEVICE_TYPE_GPU &&
                            ggml_backend_dev_type(dev_prev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        cparams.hop_layer[il] = true;
                    }
                }
            }

            std::string at;
            for (int il = 0; il < n_layer_main; ++il) {
                if (cparams.hop_layer[il]) {
                    at += (at.empty() ? "" : ",") + std::to_string(il);
                }
            }
            LLAMA_LOG_INFO("%s: card hops in %s before layers %s (%s)\n", __func__, ggml_type_name(cparams.hop_type),
                    at.empty() ? "none" : at.c_str(),
                    LLAMA_HOP_CAST_AT && *LLAMA_HOP_CAST_AT ? "LLAMA_HOP_CAST_AT" : "the layer split");
        }
    }

    // fewer routed experts on confident tokens (Q618): LLAMA_MOE_TAIL=p=<p>,w=<w>,renorm=<0|1>,hist=<0|1>,
    // graph=<0|1>,skip=<n>,repeat=<0|1>, any subset, defaults 1, 0, 0, 1, 0, 8, 0 (p = 1 and w = 0 drop nothing)
    cparams.moe_tail        = false;
    cparams.moe_tail_p      = 1.0f;
    cparams.moe_tail_w      = 0.0f;
    cparams.moe_tail_renorm = false;
    cparams.moe_tail_hist   = false;
    cparams.moe_tail_graph  = false;
    cparams.moe_tail_skip   = 8;
    cparams.moe_tail_repeat = false;
    cparams.head_bound_on   = false;
    if (const char * LLAMA_MOE_TAIL = getenv("LLAMA_MOE_TAIL")) {
        float p      = 1.0f;
        float w      = 0.0f;
        bool  renorm = false;
        bool  hist   = true;
        bool  graph  = false;
        int   skip   = 8;
        bool  repeat = false;
        std::stringstream ss(LLAMA_MOE_TAIL);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t eq = item.find('=');
            const std::string key = item.substr(0, eq);
            const std::string val = eq == std::string::npos ? "" : item.substr(eq + 1);
            if (key == "p") {
                p = std::stof(val);
            } else if (key == "w") {
                w = std::stof(val);
            } else if (key == "renorm") {
                renorm = std::stoi(val) != 0;
            } else if (key == "hist") {
                hist = std::stoi(val) != 0;
            } else if (key == "graph") {
                graph = std::stoi(val) != 0;
            } else if (key == "skip") {
                skip = std::stoi(val);
            } else if (key == "repeat") {
                repeat = std::stoi(val) != 0;
            } else if (!key.empty()) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_TAIL: unknown item '%s' (p, w, renorm, hist, graph, skip, repeat)\n",
                        __func__, item.c_str());
            }
        }
        set_moe_tail_mode(graph, skip, repeat);
        set_moe_tail(true, p, w, renorm, hist);
    }

    // route once per block (Q860): LLAMA_MOE_ROUTE_ONCE=<k>[,max=<n>][,hist=<0|1>][,dry=<0|1>]; k = 0 is off (the
    // default), 1 builds the masked path with every token its own block's first; max > 0 limits it to micro-batches of
    // at most n tokens (verify batches, not prompts); hist (default 0: it reads the graph back after every
    // micro-batch, so never for timing) counts the distinct experts per block and MoE layer; dry keeps the routing and
    // only counts
    cparams.moe_route_once      = 0;
    cparams.moe_route_once_max  = 0;
    cparams.moe_route_once_hist = false;
    cparams.moe_route_once_dry  = false;
    if (const char * LLAMA_MOE_ROUTE_ONCE = getenv("LLAMA_MOE_ROUTE_ONCE")) {
        long k    = 0;
        long max  = 0;
        bool hist = false;
        bool dry  = false;
        std::stringstream ss(LLAMA_MOE_ROUTE_ONCE);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t eq = item.find('=');
            const std::string key = eq == std::string::npos ? "k" : item.substr(0, eq);
            const std::string val = eq == std::string::npos ? item : item.substr(eq + 1);
            if (key == "k") {
                k = std::stol(val);
            } else if (key == "max") {
                max = std::stol(val);
            } else if (key == "hist") {
                hist = std::stoi(val) != 0;
            } else if (key == "dry") {
                dry = std::stoi(val) != 0;
            } else if (!key.empty()) {
                LLAMA_LOG_WARN("%s: LLAMA_MOE_ROUTE_ONCE: unknown item '%s' (k, max, hist, dry)\n", __func__,
                        item.c_str());
            }
        }
        if (k < 0 || max < 0) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_ROUTE_ONCE: k = %ld, max = %ld: negative, using 0\n", __func__, k, max);
            k   = std::max(k, 0L);
            max = std::max(max, 0L);
        }
        set_moe_route_once((uint32_t) std::min<long>(k, 1 << 20), (uint32_t) std::min<long>(max, 1 << 20), hist, dry);
    }

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }

        const char * LLAMA_GRAPH_REUSE_WAIT = getenv("LLAMA_GRAPH_REUSE_WAIT");
        graph_reuse_wait = LLAMA_GRAPH_REUSE_WAIT ? (atoi(LLAMA_GRAPH_REUSE_WAIT) != 0) : graph_reuse_wait;
    }

    {
        const char * env = getenv("LLAMA_SCHED_PER_SIZE"); // wip [TAG_SCHED_PER_SIZE]
        sched_per_size = env && atoi(env) != 0;

        const char * env_graph = getenv("LLAMA_GRAPH_PER_SIZE");
        graph_per_size = env_graph ? atoi(env_graph) != 0 : true;

        if (sched_per_size && !graph_per_size) {
            LLAMA_LOG_WARN("%s: LLAMA_SCHED_PER_SIZE=1 needs per-size graphs, ignored with LLAMA_GRAPH_PER_SIZE=0\n", __func__);
            sched_per_size = false;
        }
        LLAMA_LOG_INFO("%s: graphs per batch size up to 8 tokens: %s, schedulers per size: %s\n", __func__,
                graph_per_size ? "on" : "off (LLAMA_GRAPH_PER_SIZE=0)", sched_per_size ? "on (LLAMA_SCHED_PER_SIZE=1)" : "off");
    }

    // Q771: the micro-batch taper, off unless LLAMA_UBATCH_TAPER sets it (llama_set_ubatch_taper can change it later)
    if (const char * LLAMA_UBATCH_TAPER = getenv("LLAMA_UBATCH_TAPER")) {
        if (!set_ubatch_taper(LLAMA_UBATCH_TAPER)) {
            LLAMA_LOG_WARN("%s: LLAMA_UBATCH_TAPER ignored, the micro-batch taper stays off\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // LLAMA_PIPE_CHUNK_STREAM=1: the pipe's prompt chunks get streams of their own below these backends' priority
        // (pipe_init)
        {
            const char * env = getenv("LLAMA_PIPE_CHUNK_STREAM");
            pipe_chunk_stream = env != nullptr && atoi(env) != 0;
        }

        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, pipe_chunk_stream ? "priority=high" : nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // tensors of ctx_other (e.g. a shared KV cache) can be on devices that this model does not use
        if (cparams.ctx_other) {
            for (const auto & dev_other : llama_get_model(cparams.ctx_other)->devices) {
                const bool found = std::any_of(model.devices.begin(), model.devices.end(),
                        [&](const llama_device & dev) { return dev.dev == dev_other.dev; });
                if (found) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_dev_init(dev_other.dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev_other.dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
            /*.no_alloc  =*/ model.no_alloc(),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size, cparams.moe_cache_cpu_share);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    if (cparams.moe_tail && moe_kept_tokens > 0) {
        moe_tail_log();
    }
    if (cparams.moe_route_once > 0 && cparams.moe_route_once_hist && ro.n_tokens > 0) {
        moe_route_once_log(true);
    }
    if (head_bound) {
        head_bound_finish_row();
        head_bound->log(__func__);
    }

    pipe_drain();
    for (auto & slot : pipe_slots) {
        if (slot.ev_out) {
            ggml_backend_event_free(slot.ev_out);
        }
    }
    pipe_drafter.reset();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.no_alloc() && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    // the fallback attends to the whole unified cache with a dense mask
    if (cparams.auto_fkvr) {
        cparams.fused_kv_rows = cparams.flash_attn;
        resolve(llm_fused_op_flash_attn_kv_rows_probe, cparams.fused_kv_rows);
        cparams.auto_fkvr = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (cparams.auto_ftkm) {
        LLAMA_LOG_INFO("%s: resolving fused top-k mask support:\n", func);
        resolve(llm_fused_op_top_k_mask_probe, cparams.fused_top_k_mask);
        cparams.auto_ftkm = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_new(size_t max_nodes, bool parallel) {
    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, parallel, cparams.op_offload));

    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);

    // 2: an evaluation submitted with nothing in flight stages each split's host inputs and uploads them with one
    // asynchronous copy, instead of an event wait and a synchronous copy per input (process_ubatch); 0: as stock
    // llama.cpp. Not 1 (an asynchronous copy from the inputs themselves): the next micro-batch of a llama_decode call
    // may overwrite them before the copy has run. Off unless LLAMA_INPUT_UPLOAD is set: on nine P106-090 cards it gave
    // nothing measurable (release measurements 2026-10-25), and the stock path is the one every check covers.
    plain_upload = 0;
    if (const char * env = getenv("LLAMA_INPUT_UPLOAD")) {
        plain_upload = atoi(env) != 0 ? 2 : 0;
    }
    ggml_backend_sched_set_input_upload(sched.get(), plain_upload);
}

// a backend sampler chain by its samplers' names: chains with the same names build graphs of the same shape, up to
// sizes such as top-k's k, which the scheduler's allocator handles by growing its buffers if ever needed
static std::string sampler_chain_sig(llama_sampler * chain) {
    std::string sig;
    for (int32_t i = 0; i < llama_sampler_chain_n(chain); i++) {
        sig += llama_sampler_name(llama_sampler_chain_get(chain, i));
        sig += ',';
    }
    return sig;
}

static std::vector<std::string> sampler_chain_sigs(const std::map<llama_seq_id, llama_sampler *> & samplers) {
    std::vector<std::string> res;
    for (const auto & entry : samplers) {
        res.push_back(sampler_chain_sig(entry.second));
    }
    std::sort(res.begin(), res.end());
    return res;
}

bool llama_context::sampler_reserve_covered() const {
    const auto cur = sampler_chain_sigs(sampling.samplers);
    return std::includes(sampling.reserved_chains.begin(), sampling.reserved_chains.end(), cur.begin(), cur.end());
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    sampling.reserved_chains = sampler_chain_sigs(sampling.samplers);

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.clear();
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_invalidate();

    sched_new(max_nodes, cparams.pipeline_parallel);

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.no_alloc(), model.no_alloc() ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new(max_nodes, false);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.no_alloc());
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.no_alloc());
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.no_alloc());
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.no_alloc()) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    // nothing submitted since the last synchronize: every device is idle, so skip the sweep over all backends (a served
    // token synchronizes four times: after decode, in the sampler and twice in the getters); LLAMA_INPUT_UPLOAD=0 sweeps
    if (plain_upload > 0 && !gf_in_flight && n_queued_tokens == 0 &&
            std::none_of(pipe_slots.begin(), pipe_slots.end(), [](const pipe_slot & s) { return s.busy; })) {
        return;
    }

    LLAMA_STEP_MARK("sync");
    ggml_backend_sched_synchronize(sched.get());
    LLAMA_STEP_MARK("sync.end");
    gf_in_flight = false;

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & [key, res] : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_invalidate();
        sched_active = nullptr; // the memory module computes its graphs with the shared scheduler

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_logits_ith_fetch(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        const int64_t n_vocab = model.vocab.n_tokens();

        // an output sampled by a backend sampler: its logits are on the device, if still there
        if ((size_t) j < logits_dev_rows.size() && logits_dev_rows[j] != -1) {
            if (logits_dev_rows[j] == -2 || logits_dev_tensor == nullptr) {
                throw std::runtime_error("sampled on the device, and a later ubatch overwrote its logits");
            }
            synchronize();
            ggml_backend_tensor_get(logits_dev_tensor, logits.data + j*n_vocab, logits_dev_rows[j]*n_vocab*sizeof(float),
                    n_vocab*sizeof(float));
            logits_dev_rows[j] = -1;
        }

        return logits.data + j*n_vocab;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
        return nullptr;
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the toggle changes the graph shape (pooled: logits only; pooling NONE: the reserve graph outputs all tokens,
    // which crashed ggml_gallocr on the first /completion), so re-reserve on every toggle
    if (cparams.embeddings != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;

    // the device rows are batch rows of the unmasked extraction
    if (cparams.embeddings_nextn_dev && (!value || masked)) {
        cparams.embeddings_nextn_dev = false;
        h_nextn_dev_alloc();
    }
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;

    if (cparams.embeddings_layer_inp_dev) {
        layer_inp_dev_alloc();
    }
}

void llama_context::set_embeddings_layer_inp_dev(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.embeddings_layer_inp_dev != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings_layer_inp_dev = value;

    layer_inp_dev_alloc();
}

bool llama_context::supports_embd_row() const {
    // the DFlash/DSpark decoder graphs read the device buffer
    if (model.arch != LLM_ARCH_DFLASH) {
        return false;
    }

    const ggml_tensor * src = cparams.ctx_other ? cparams.ctx_other->get_layer_inp_dev() : nullptr;
    if (src == nullptr) {
        return false;
    }

    // one of our backends must be able to read the buffer, e.g. not a draft on another GPU than the target output
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src->buffer);
    for (const auto & backend : backends) {
        if (ggml_backend_supports_buft(backend.get(), buft)) {
            return true;
        }
    }

    return false;
}

ggml_backend_t llama_context::backend_from_dev(ggml_backend_dev_t dev) const {
    for (const auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == dev) {
            return backend.get();
        }
    }
    return nullptr;
}

void llama_context::layer_inp_dev_alloc() {
    synchronize();

    layer_inp_dev     = nullptr;
    layer_inp_dev_dev = nullptr;
    layer_inp_backend = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();

    if (!cparams.embeddings_layer_inp_dev) {
        return;
    }

    int64_t n_layer_inp = 0;
    for (bool enabled : cparams.embeddings_layer_inp) {
        n_layer_inp += enabled ? 1 : 0;
    }
    if (n_layer_inp == 0) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_layer_inp_dev.reset(ggml_init(params));

    layer_inp_dev = ggml_new_tensor_2d(ctx_layer_inp_dev.get(), GGML_TYPE_F32, n_layer_inp*model.hparams.n_embd, cparams.n_batch);
    ggml_set_name(layer_inp_dev, "layer_inp_dev");

    layer_inp_dev_dev = model.dev_output();
    auto * buft = layer_inp_dev_dev ? ggml_backend_dev_buffer_type(layer_inp_dev_dev) : ggml_backend_cpu_buffer_type();

    buf_layer_inp_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_layer_inp_dev.get(), buft));
    if (!buf_layer_inp_dev) {
        // e.g. a large n_batch that does not fit next to the model, keep the host copies
        LLAMA_LOG_WARN("%s: failed to allocate the layer input device buffer, using host copies\n", __func__);
        cparams.embeddings_layer_inp_dev = false;
        layer_inp_dev     = nullptr;
        layer_inp_dev_dev = nullptr;
        ctx_layer_inp_dev.reset();
        return;
    }

    LLAMA_LOG_INFO("%s: %10s layer input buffer size = %8.2f MiB (%d layers)\n", __func__,
            ggml_backend_buffer_name(buf_layer_inp_dev.get()), ggml_backend_buffer_get_size(buf_layer_inp_dev.get())/1024.0/1024.0, (int) n_layer_inp);

    layer_inp_backend = layer_inp_dev_dev ? backend_from_dev(layer_inp_dev_dev) : nullptr;
    if (layer_inp_backend && !ev_layer_inp_written) {
        ev_layer_inp_written.reset(ggml_backend_event_new(layer_inp_dev_dev));
        ev_layer_inp_read.reset(ggml_backend_event_new(layer_inp_dev_dev));
        ev_layer_inp_read_recorded = false;
    }
}

bool llama_context::h_nextn_dev_alloc() {
    synchronize();

    auto reset = [&]() {
        h_nextn_dev     = nullptr;
        h_nextn_dev_dev = nullptr;
        h_nextn_backend = nullptr;
        buf_h_nextn_dev.reset();
        ctx_h_nextn_dev.reset();
        for (int h = 0; h < 2; ++h) {
            ev_h_nextn_written_recorded[h] = false;
            ev_h_nextn_read_recorded[h]    = false;
        }
        h_nextn_last = 0;
        h_nextn_held = false;
        h_nextn_row0 = 0;
    };

    // __func__ inside the lambda would be "operator()"
    const char * func = __func__;
    auto fail = [&](const char * why) {
        LLAMA_LOG_WARN("%s: %s, extracting the h_nextn rows to host memory\n", func, why);
        cparams.embeddings_nextn_dev = false;
        reset();
        return false;
    };

    reset();

    if (!cparams.embeddings_nextn_dev) {
        return true;
    }

    h_nextn_dev_dev = model.dev_output();
    h_nextn_backend = h_nextn_dev_dev ? backend_from_dev(h_nextn_dev_dev) : nullptr;
    if (h_nextn_backend == nullptr) {
        return fail("no device backend for the model output");
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_h_nextn_dev.reset(ggml_init(params));

    h_nextn_dev = ggml_new_tensor_2d(ctx_h_nextn_dev.get(), GGML_TYPE_F32, model.hparams.n_embd_out(),
            (int64_t) h_nextn_halves*(cparams.n_batch + cparams.n_seq_max));
    ggml_set_name(h_nextn_dev, "h_nextn_dev");

    buf_h_nextn_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_h_nextn_dev.get(), ggml_backend_dev_buffer_type(h_nextn_dev_dev)));
    if (!buf_h_nextn_dev) {
        return fail("failed to allocate the h_nextn device buffer");
    }

    for (int h = 0; h < h_nextn_halves; ++h) {
        if (!ev_h_nextn_written[h]) {
            ev_h_nextn_written[h].reset(ggml_backend_event_new(h_nextn_dev_dev));
            ev_h_nextn_read[h].reset(ggml_backend_event_new(h_nextn_dev_dev));
        }
        // C703: the CPU computes a decode before it returns, so two halves there need no events (the CPU check)
        const bool cpu = ggml_backend_dev_type(h_nextn_dev_dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if ((!ev_h_nextn_written[h] || !ev_h_nextn_read[h]) && !(h_nextn_halves > 1 && cpu)) {
            return fail("the device has no events");
        }
    }

    LLAMA_LOG_INFO("%s: %10s h_nextn buffer size = %8.2f MiB%s\n", __func__,
            ggml_backend_buffer_name(buf_h_nextn_dev.get()), ggml_backend_buffer_get_size(buf_h_nextn_dev.get())/1024.0/1024.0,
            h_nextn_halves > 1 ? " (two halves)" : "");

    return true;
}

bool llama_context::set_embeddings_nextn_dev(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the device rows are batch rows, so the extraction must not be masked
    if (value && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
        return false;
    }

    if (cparams.embeddings_nextn_dev != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn_dev = value;

    return h_nextn_dev_alloc() && cparams.embeddings_nextn_dev == value;
}

bool llama_context::set_h_nextn_halves(int32_t n) {
    if (n < 1 || n > 2) {
        return false;
    }
    if (h_nextn_halves != n) {
        h_nextn_halves = n;
        // the next graphs see the new buffer
        sched_need_reserve = true;
        if (cparams.embeddings_nextn_dev && !h_nextn_dev_alloc()) {
            return false;
        }
    }
    return h_nextn_dev == nullptr || h_nextn_halves == n;
}

bool llama_context::hold_h_nextn() {
    if (h_nextn_dev == nullptr || h_nextn_halves < 2) {
        return false;
    }
    h_nextn_held = true;
    return true;
}

bool llama_context::get_h_nextn_rows(int64_t row0, int64_t n_rows, float * out) {
    if (h_nextn_dev == nullptr || row0 < 0 || n_rows < 0 || row0 + n_rows > h_nextn_dev->ne[1]) {
        return false;
    }
    const int32_t h = h_nextn_half_of(row0);
    if (ev_h_nextn_written_recorded[h]) {
        ggml_backend_event_synchronize(ev_h_nextn_written[h].get());
    } else {
        synchronize();
    }
    ggml_backend_tensor_get(h_nextn_dev, out, row0*h_nextn_dev->nb[1], n_rows*h_nextn_dev->nb[1]);
    return true;
}

bool llama_context::set_h_nextn_rows(int64_t row0, int64_t n_rows, const float * data) {
    if (h_nextn_dev == nullptr || row0 < 0 || n_rows < 0 || row0 + n_rows > h_nextn_dev->ne[1]) {
        return false;
    }
    // a reader may still read these rows
    const int32_t h = h_nextn_half_of(row0);
    if (ev_h_nextn_read_recorded[h]) {
        ggml_backend_event_synchronize(ev_h_nextn_read[h].get());
    }
    ggml_backend_tensor_set(h_nextn_dev, data, row0*h_nextn_dev->nb[1], n_rows*h_nextn_dev->nb[1]);
    return true;
}

bool llama_context::set_state_src(llama_context * src) {
    if (cparams.ctx_state_src != src) {
        sched_need_reserve = true;
    }
    cparams.ctx_state_src = src;
    return supports_state_row();
}

bool llama_context::supports_state_row() const {
    // the MTP graphs read the rows, see llm_graph_context::build_inp_embd_state()
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
        return false;
    }

    // the rows replace the host state rows, so they must have the width the graph takes for those
    const ggml_tensor * src = cparams.ctx_state_src ? cparams.ctx_state_src->get_h_nextn_dev() : nullptr;
    if (src == nullptr || src->ne[0] != (int64_t) llama_batch_ext_select_n_embd_state(cparams.ctx_type, model.hparams)) {
        return false;
    }

    // one of our backends must be able to read the buffer
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src->buffer);
    for (const auto & backend : backends) {
        if (ggml_backend_supports_buft(backend.get(), buft)) {
            return true;
        }
    }

    return false;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_screen_head(bool value) {
    if (cparams.screen_head == value) {
        return;
    }
    LLAMA_LOG_INFO("%s: screened output head %s\n", __func__, value ? "on" : "off");

    // like causal_attn: a flip rebuilds the graph, which allocates the few extra MB on the fly
    cparams.screen_head = value;
}

bool llama_context::get_screen_head() const {
    return cparams.screen_head;
}

bool llama_context::set_ubatch_taper(const char * spec) {
    llama_ubatch_taper t;
    std::string err;
    if (!t.parse(spec, err)) {
        LLAMA_LOG_ERROR("%s: bad micro-batch taper '%s': %s\n", __func__, spec ? spec : "", err.c_str());
        return false;
    }
    const std::string s = t.str();
    if (s != taper_spec) {
        LLAMA_LOG_INFO("%s: micro-batch taper %s (n_ubatch %u)\n", __func__, s.c_str(), cparams.n_ubatch);
    }
    // the plan is made per call, so a change takes effect at the next decode call
    taper      = t;
    taper_spec = s;
    return true;
}

const char * llama_context::get_ubatch_taper() const {
    return taper_spec.c_str();
}

void llama_context::set_ubatch_taper_next(int32_t head, int32_t tail) {
    taper_next_head = head < 0 ? -1 : head > 0;
    taper_next_tail = tail < 0 ? -1 : tail > 0;
}

bool llama_context::head_shortlist_init(uint32_t k_low, uint32_t m_cap) {
    const ggml_tensor * output = model.output;
    if (output == nullptr || output->buffer == nullptr) {
        LLAMA_LOG_ERROR("%s: the model has no output head in a buffer\n", __func__);
        return false;
    }
    if (model.output_s || model.output_in_s) {
        LLAMA_LOG_ERROR("%s: the output head has scales, which the shortlist does not apply\n", __func__);
        return false;
    }

    // the shortlist reads rows of the head where they are: plain host memory, or a device's own buffer type (not a
    // repacked or split one, whose rows are laid out differently)
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(output->buffer);
    if (ggml_backend_buffer_is_host(output->buffer)) {
        buft = ggml_backend_cpu_buffer_type();
    } else {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev == nullptr || buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_ERROR("%s: the output head is in a %s buffer, whose rows the shortlist can't address\n",
                    __func__, ggml_backend_buft_name(buft));
            return false;
        }
    }

    auto sl = std::make_unique<llama_head_shortlist>();
    if (!sl->init((uint32_t) output->ne[1], k_low, m_cap)) {
        LLAMA_LOG_ERROR("%s: a shortlist of the ids below %u and %u more does not fit a vocabulary of %" PRId64 " "
                "(1 <= ids < vocabulary, at most 65535 more)\n", __func__, k_low, m_cap, output->ne[1]);
        return false;
    }
    if (!sl->alloc(buft)) {
        LLAMA_LOG_ERROR("%s: failed to allocate the shortlist in %s\n", __func__, ggml_backend_buft_name(buft));
        return false;
    }

    head_short = std::move(sl);

    cparams.head_short    = head_short.get();
    cparams.head_short_on = false;
    cparams.head_short_n  = 0;

    LLAMA_LOG_INFO("%s: draft head shortlist: the ids below %u and up to %u more, in %s (%.2f MiB)\n", __func__,
            head_short->k_low, head_short->m_cap, ggml_backend_buft_name(buft),
            ggml_backend_buffer_get_size(head_short->buf.get()) / 1024.0 / 1024.0);

    return true;
}

uint32_t llama_context::head_shortlist_add(const llama_token * ids, size_t n) {
    if (!head_short) {
        return 0;
    }

    const uint32_t n_added = head_short->add(ids, n);
    if (n_added > 0) {
        // a graph in flight may still read the ids
        synchronize();
        head_short->sync();
        cparams.head_short_n = head_short->n_extra();
    }

    return n_added;
}

bool llama_context::head_shortlist_contains(llama_token id) const {
    return head_short && head_short->contains(id);
}

uint32_t llama_context::head_shortlist_size() const {
    return head_short ? head_short->size() : 0;
}

void llama_context::set_head_shortlist(bool value) {
    // a flip rebuilds the graph (llm_graph_params::allow_reuse)
    cparams.head_short_on = value && head_short != nullptr;
}

bool llama_context::mtp_layer_device(ggml_backend_dev_t & dev, ggml_backend_t & backend) const {
    // the card that runs the draft's layer: where its first weight is
    const ggml_tensor * w = nullptr;
    for (const auto & layer : model.layers) {
        if (layer.nextn.eh_proj != nullptr && layer.nextn.eh_proj->buffer != nullptr) {
            w = layer.nextn.eh_proj;
            break;
        }
    }
    if (w == nullptr) {
        LLAMA_LOG_ERROR("%s: the model has no MTP layer in a buffer\n", __func__);
        return false;
    }
    dev = ggml_backend_buffer_is_host(w->buffer) ?
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(w->buffer));
    backend = nullptr;
    for (const auto & b : backends) {
        if (dev != nullptr && ggml_backend_get_device(b.get()) == dev) {
            backend = b.get();
            break;
        }
    }
    if (backend == nullptr) {
        LLAMA_LOG_ERROR("%s: no backend of this context runs the MTP layer's device\n", __func__);
        return false;
    }
    return true;
}

bool llama_context::nextn_carry_init() {
    if (model.arch != LLM_ARCH_QWEN4EXP) {
        LLAMA_LOG_ERROR("%s: only the qwen4exp MTP draft keeps its hidden rows on the device\n", __func__);
        return false;
    }
    ggml_backend_dev_t dev     = nullptr;
    ggml_backend_t     backend = nullptr;
    if (!mtp_layer_device(dev, backend)) {
        return false;
    }

    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    nextn_carry_ctx.reset(ggml_init(ip));
    ggml_tensor * t = ggml_new_tensor_2d(nextn_carry_ctx.get(), GGML_TYPE_F32, model.hparams.n_embd_out(), cparams.n_seq_max);
    ggml_set_name(t, "nextn_carry");

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    nextn_carry_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(nextn_carry_ctx.get(), buft));
    if (!nextn_carry_buf) {
        LLAMA_LOG_ERROR("%s: failed to allocate the hidden rows' carry in %s\n", __func__, ggml_backend_buft_name(buft));
        nextn_carry_ctx.reset();
        return false;
    }
    // placed like weights: the ops that read and write it run where it is
    ggml_backend_buffer_set_usage(nextn_carry_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(nextn_carry_buf.get(), 0);

    cparams.nextn_carry     = t;
    cparams.nextn_carry_in  = false;
    cparams.nextn_carry_out = false;

    LLAMA_LOG_INFO("%s: draft hidden rows kept on the device: %u rows of %u F32 (%.2f MiB) in %s\n", __func__,
            cparams.n_seq_max, model.hparams.n_embd_out(),
            ggml_backend_buffer_get_size(nextn_carry_buf.get()) / 1024.0 / 1024.0, ggml_backend_buft_name(buft));

    return true;
}

void llama_context::set_nextn_carry(bool in, bool out) {
    // a change rebuilds the graph (llm_graph_params::allow_reuse)
    cparams.nextn_carry_in  = in  && cparams.nextn_carry != nullptr;
    cparams.nextn_carry_out = out && cparams.nextn_carry != nullptr;
}

bool llama_context::embd_cache_init(uint32_t n_rows, uint32_t max_tokens) {
    if (model.arch != LLM_ARCH_QWEN4EXP) {
        LLAMA_LOG_ERROR("%s: only the qwen4exp MTP draft reads its token rows from a row cache\n", __func__);
        return false;
    }
    const ggml_tensor * tok_embd = model.tok_embd;
    if (tok_embd == nullptr || tok_embd->buffer == nullptr) {
        LLAMA_LOG_ERROR("%s: the model has no token embeddings in a buffer\n", __func__);
        return false;
    }

    ggml_backend_dev_t dev     = nullptr;
    ggml_backend_t     backend = nullptr;
    if (!mtp_layer_device(dev, backend)) {
        return false;
    }

    auto ec = std::make_unique<llama_embd_cache>();
    if (!ec->init(tok_embd, n_rows, max_tokens)) {
        LLAMA_LOG_ERROR("%s: a cache of %u rows for batches of up to %u tokens does not fit a vocabulary of %" PRId64 " "
                "(1 <= tokens <= rows <= vocabulary)\n", __func__, n_rows, max_tokens, tok_embd->ne[1]);
        return false;
    }

    // the layer's card must read rows of the table's type (get_rows of a quantized type is not on every backend)
    {
        ggml_init_params ip = { 3*ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr c(ggml_init(ip));
        ggml_tensor * t   = ggml_new_tensor_2d(c.get(), tok_embd->type, tok_embd->ne[0], n_rows);
        ggml_tensor * ids = ggml_new_tensor_1d(c.get(), GGML_TYPE_I32, max_tokens);
        if (!ggml_backend_dev_supports_op(dev, ggml_get_rows(c.get(), t, ids))) {
            LLAMA_LOG_ERROR("%s: %s can't read %s rows\n", __func__, ggml_backend_dev_name(dev), ggml_type_name(tok_embd->type));
            return false;
        }
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!ec->alloc(buft)) {
        LLAMA_LOG_ERROR("%s: failed to allocate the row cache in %s\n", __func__, ggml_backend_buft_name(buft));
        return false;
    }

    embd_cache         = std::move(ec);
    embd_cache_backend = backend;

    cparams.embd_cache    = embd_cache.get();
    cparams.embd_cache_on = false;

    LLAMA_LOG_INFO("%s: draft embedding row cache: %u rows of %s (%.2f MiB) in %s, for batches of up to %u tokens; the "
            "table is in %s\n", __func__, n_rows, ggml_type_name(tok_embd->type),
            ggml_backend_buffer_get_size(embd_cache->buf.get()) / 1024.0 / 1024.0, ggml_backend_buft_name(buft), max_tokens,
            ggml_backend_buffer_name(tok_embd->buffer));

    return true;
}

bool llama_context::embd_cache_stats(uint64_t * n_hit, uint64_t * n_miss, uint64_t * n_plain) const {
    if (!embd_cache) {
        return false;
    }
    if (n_hit)   { *n_hit   = embd_cache->n_hit;   }
    if (n_miss)  { *n_miss  = embd_cache->n_miss;  }
    if (n_plain) { *n_plain = embd_cache->n_plain; }
    return true;
}

void llama_context::set_moe_tail(bool on, float p, float w, bool renorm, bool hist) {
    const auto & hparams = model.hparams;
    const uint32_t k = std::max<uint32_t>(hparams.n_expert_used_max(), 1);
    if (!(p > 0.0f) || p > 1.0f) {
        LLAMA_LOG_WARN("%s: p = %g is outside (0, 1], using 1\n", __func__, p);
        p = 1.0f;
    }
    // the first slot's share is at least 1/k, and it must stay
    if (!(w >= 0.0f) || w >= 1.0f/k) {
        LLAMA_LOG_WARN("%s: w = %g is outside [0, 1/%u), using 0\n", __func__, w, k);
        w = 0.0f;
    }

    cparams.moe_tail        = on && hparams.n_expert > 0;
    cparams.moe_tail_p      = p;
    cparams.moe_tail_w      = w;
    cparams.moe_tail_renorm = renorm;
    cparams.moe_tail_hist   = hist;

    if (on && hparams.n_expert == 0) {
        LLAMA_LOG_WARN("%s: the model has no routed experts\n", __func__);
    } else if (on) {
        const std::string how = cparams.moe_tail_graph ? std::string("graph-level ops (the dropped experts still read)") :
            std::string(cparams.moe_tail_repeat ? "GGML_OP_MOE_TAIL, dropped slots reading their token's last kept expert"
                                                : "GGML_OP_MOE_TAIL, dropped experts skipped") +
            " in micro-batches of up to " + std::to_string(cparams.moe_tail_skip) + " tokens";
        LLAMA_LOG_INFO("%s: moe tail on: a slot of the %u routed experts is dropped when the slots above it hold a share "
                "of %g, or its own share is below %g; kept weights %s; histogram %s; %s\n", __func__, k, p, w,
                renorm ? "renormalized" : "as they are", hist ? "on" : "off", how.c_str());
    }
}

void llama_context::set_moe_tail_mode(bool graph, int skip, bool repeat) {
    cparams.moe_tail_graph  = graph;
    cparams.moe_tail_skip   = (uint32_t) std::max(skip, 0);
    cparams.moe_tail_repeat = repeat;
}

uint64_t llama_context::moe_tail_counts(uint64_t * counts, int32_t n_counts) const {
    uint64_t n = 0;
    for (size_t c = 0; c < moe_kept_hist.size(); ++c) {
        n += moe_kept_hist[c];
        if (counts && (int32_t) c < n_counts) {
            counts[c] = moe_kept_hist[c];
        }
    }
    for (int32_t c = (int32_t) moe_kept_hist.size(); counts && c < n_counts; ++c) {
        counts[c] = 0;
    }
    return n;
}

void llama_context::moe_tail_collect(const llm_graph_result * res, uint32_t n_tokens) {
    const auto & ts = res->t_moe_kept;
    if (ts.empty()) {
        return;
    }

    size_t n_all = 0;
    for (const auto * t : ts) {
        n_all += ggml_nelements(t);
    }
    moe_kept_buf.resize(n_all);

    size_t off = 0;
    for (auto * t : ts) {
        if (ggml_nelements(t) == 0) {
            continue; // a layer cropped to the outputs of a ubatch without any
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched_res(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, moe_kept_buf.data() + off, 0, ggml_nbytes(t));
        off += ggml_nelements(t);
    }
    ggml_backend_sched_synchronize(sched_res());

    if (moe_kept_layer.size() < 2*ts.size()) {
        moe_kept_layer.resize(2*ts.size(), 0.0);
    }

    off = 0;
    for (size_t l = 0; l < ts.size(); ++l) {
        const int64_t n = ggml_nelements(ts[l]);
        for (int64_t i = 0; i < n; ++i) {
            const float v = moe_kept_buf[off + i];
            const size_t c = (size_t) std::max(0L, std::lrint(v));
            if (moe_kept_hist.size() <= c) {
                moe_kept_hist.resize(c + 1, 0);
            }
            moe_kept_hist[c]++;
            moe_kept_layer[2*l + 0] += v;
            moe_kept_layer[2*l + 1] += 1.0;
        }
        off += n;
    }

    moe_kept_tokens += n_tokens;
    if (moe_kept_tokens >= moe_kept_next_log) {
        if (moe_kept_next_log > 0) {
            moe_tail_log();
        }
        moe_kept_next_log = (moe_kept_tokens/65536 + 1)*65536;
    }
}

void llama_context::moe_tail_log() const {
    const uint32_t k = std::max<uint32_t>(model.hparams.n_expert_used_max(), 1);

    uint64_t n_pairs = 0;
    double   n_kept  = 0.0;
    for (size_t c = 0; c < moe_kept_hist.size(); ++c) {
        n_pairs += moe_kept_hist[c];
        n_kept  += (double) c*moe_kept_hist[c];
    }
    if (n_pairs == 0) {
        return;
    }

    std::string by_count;
    for (size_t c = 0; c < moe_kept_hist.size(); ++c) {
        if (moe_kept_hist[c] > 0) {
            by_count += format(" %zu: %.2f %%", c, 100.0*moe_kept_hist[c]/n_pairs);
        }
    }
    std::string by_layer;
    for (size_t l = 0; l + 1 < moe_kept_layer.size(); l += 2) {
        by_layer += format(" %.2f", moe_kept_layer[l + 1] > 0 ? moe_kept_layer[l]/moe_kept_layer[l + 1] : 0.0);
    }

    LLAMA_LOG_INFO("moe tail (p = %g, w = %g, renorm %d): %" PRIu64 " tokens, %" PRIu64 " token-layers; experts kept: "
            "mean %.3f of %u (%.1f %% of the routed experts' reads); by count:%s\n",
            cparams.moe_tail_p, cparams.moe_tail_w, (int) cparams.moe_tail_renorm, moe_kept_tokens, n_pairs,
            n_kept/n_pairs, k, 100.0*n_kept/n_pairs/k, by_count.c_str());
    LLAMA_LOG_INFO("moe tail: mean experts kept per MoE layer:%s\n", by_layer.c_str());
}

void llama_context::set_moe_route_once(uint32_t k, uint32_t max, bool hist, bool dry) {
    const auto & hparams = model.hparams;
    if (k > 0 && hparams.n_expert == 0) {
        LLAMA_LOG_WARN("%s: the model has no routed experts\n", __func__);
        k = 0;
    }
    if (k > 0 && hparams.n_expert_groups > 1) {
        LLAMA_LOG_WARN("%s: the model routes in expert groups, which route once does not handle: off\n", __func__);
        k = 0;
    }

    // a change rebuilds the graph (llm_graph_params::allow_reuse)
    cparams.moe_route_once      = k;
    cparams.moe_route_once_max  = k > 0 ? max : 0;
    cparams.moe_route_once_hist = k > 0 && hist;
    cparams.moe_route_once_dry  = k > 0 && dry;

    if (k > 0) {
        LLAMA_LOG_INFO("%s: moe route once on: blocks of %u consecutive tokens%s; %s; histogram %s\n", __func__, k,
                max > 0 ? format(" in micro-batches of at most %u tokens", max).c_str() : "",
                dry ? "a dry run (the routing as it is, only counted)" :
                      "a block's later tokens pick their routed experts among its first token's",
                hist ? "on" : "off");
    }
}

uint64_t llama_context::moe_route_once_counts(uint64_t * read, uint64_t * indep, int32_t n_counts,
        uint64_t * n_outside) const {
    for (int32_t d = 0; d < n_counts; ++d) {
        if (read) {
            read[d] = 0;
        }
        if (indep) {
            indep[d] = 0;
        }
    }
    uint64_t n = 0;
    for (size_t l = 0; l < ro.read.size(); ++l) {
        for (size_t d = 0; d < ro.read[l].size(); ++d) {
            n += ro.read[l][d];
            if (read && (int32_t) d < n_counts) {
                read[d] += ro.read[l][d];
            }
        }
        for (size_t d = 0; d < ro.indep[l].size(); ++d) {
            if (indep && (int32_t) d < n_counts) {
                indep[d] += ro.indep[l][d];
            }
        }
    }
    if (n_outside) {
        *n_outside = ro.n_outside;
    }
    return n;
}

void llama_context::moe_route_once_collect(const llm_graph_result * res, const llama_ubatch & ubatch, uint32_t n_out) {
    const auto & tr = res->t_moe_route_read;
    const auto & ti = res->t_moe_route_free;
    if (tr.empty()) {
        return; // a micro-batch of one token, or above moe_route_once_max
    }
    GGML_ASSERT(tr.size() == ti.size());

    const uint32_t k_blk    = cparams.moe_route_once;
    const int64_t  n_used   = tr[0]->ne[0];
    const size_t   n_expert = model.hparams.n_expert;
    llama_route_once_leaders(ubatch, k_blk, n_out, ro.lead_all, ro.lead_out);

    size_t n_all = 0;
    for (const auto * t : tr) {
        n_all += ggml_nelements(t);
    }
    ro.ids_read.resize(n_all);
    ro.ids_indep.resize(n_all);

    size_t off = 0;
    for (size_t l = 0; l < tr.size(); ++l) {
        GGML_ASSERT(tr[l]->type == GGML_TYPE_I32 && ti[l]->type == GGML_TYPE_I32 && ggml_are_same_shape(tr[l], ti[l]));
        if (ggml_nelements(tr[l]) == 0) {
            continue;
        }
        ggml_backend_t b_read  = ggml_backend_sched_get_tensor_backend(sched_res(), tr[l]);
        ggml_backend_t b_indep = ggml_backend_sched_get_tensor_backend(sched_res(), ti[l]);
        GGML_ASSERT(b_read != nullptr && b_indep != nullptr);
        ggml_backend_tensor_get_async(b_read,  tr[l], ro.ids_read.data()  + off, 0, ggml_nbytes(tr[l]));
        ggml_backend_tensor_get_async(b_indep, ti[l], ro.ids_indep.data() + off, 0, ggml_nbytes(ti[l]));
        off += ggml_nelements(tr[l]);
    }
    ggml_backend_sched_synchronize(sched_res());

    if (ro.seen_read.size() < n_expert) {
        ro.seen_read .assign(n_expert, 0);
        ro.seen_indep.assign(n_expert, 0);
        ro.seen_lead .assign(n_expert, 0);
    }
    if (ro.read.size() < tr.size()) {
        ro.read .resize(tr.size());
        ro.indep.resize(tr.size());
    }
    auto bump = [](std::vector<uint64_t> & h, size_t c) {
        if (h.size() <= c) {
            h.resize(c + 1, 0);
        }
        h[c]++;
    };

    off = 0;
    for (size_t l = 0; l < tr.size(); ++l) {
        const int64_t n_rows = tr[l]->ne[1];
        if (n_rows == 0) {
            continue;
        }
        GGML_ASSERT(tr[l]->ne[0] == n_used);
        const std::vector<int32_t> & lead = n_rows == (int64_t) ro.lead_all.size() ? ro.lead_all : ro.lead_out;
        GGML_ASSERT((int64_t) lead.size() == n_rows);

        const int32_t * r_ids = ro.ids_read.data()  + off;
        const int32_t * i_ids = ro.ids_indep.data() + off;
        for (int64_t i0 = 0; i0 < n_rows; ) {
            int64_t i1 = i0 + 1;
            while (i1 < n_rows && lead[i1] == lead[i0]) {
                i1++;
            }

            // the block [i0, i1): its first row's experts, then every row's against them
            const uint64_t s = ++ro.stamp;
            for (int64_t e = 0; e < n_used; ++e) {
                const int32_t x = r_ids[i0*n_used + e];
                GGML_ASSERT(x >= 0 && (size_t) x < n_expert);
                ro.seen_lead[x] = s;
            }
            uint32_t d_read  = 0;
            uint32_t d_indep = 0;
            for (int64_t i = i0; i < i1; ++i) {
                bool     outside = false;
                uint32_t own     = 0;
                for (int64_t e = 0; e < n_used; ++e) {
                    const int32_t x = r_ids[i*n_used + e];
                    const int32_t y = i_ids[i*n_used + e];
                    GGML_ASSERT(x >= 0 && (size_t) x < n_expert && y >= 0 && (size_t) y < n_expert);
                    if (ro.seen_read[x] != s) {
                        ro.seen_read[x] = s;
                        d_read++;
                    }
                    if (ro.seen_indep[y] != s) {
                        ro.seen_indep[y] = s;
                        d_indep++;
                    }
                    outside = outside || ro.seen_lead[x] != s;
                    own    += ro.seen_lead[y] == s;
                }
                if (i > i0) {
                    ro.n_outside += outside;
                    bump(ro.own, own);
                }
            }
            if (i1 - i0 == (int64_t) k_blk) {
                bump(ro.read[l],  d_read);
                bump(ro.indep[l], d_indep);
            } else {
                ro.n_short++;
            }
            i0 = i1;
        }
        off += n_rows*n_used;
    }

    ro.n_tokens += ubatch.n_tokens;
    if (ro.n_tokens >= ro.next_log) {
        if (ro.next_log > 0) {
            moe_route_once_log(false);
        }
        ro.next_log = (ro.n_tokens/65536 + 1)*65536;
    }
}

void llama_context::moe_route_once_log(bool by_layer) const {
    const uint32_t n_used = std::max<uint32_t>(model.hparams.n_expert_used_max(), 1);

    auto mean = [](const std::vector<uint64_t> & h, uint64_t * n_out) {
        uint64_t n = 0;
        double   s = 0.0;
        for (size_t c = 0; c < h.size(); ++c) {
            n += h[c];
            s += (double) c*h[c];
        }
        if (n_out) {
            *n_out = n;
        }
        return n > 0 ? s/n : 0.0;
    };
    auto by_count = [](const std::vector<uint64_t> & h) {
        uint64_t n = 0;
        for (const uint64_t c : h) {
            n += c;
        }
        std::string s;
        for (size_t c = 0; c < h.size(); ++c) {
            if (h[c] > 0) {
                s += format(" %zu: %.2f %%", c, 100.0*h[c]/n);
            }
        }
        return s;
    };

    auto add = [](std::vector<uint64_t> & dst, const std::vector<uint64_t> & src) {
        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t c = 0; c < src.size(); ++c) {
            dst[c] += src[c];
        }
    };

    std::vector<uint64_t> read;
    std::vector<uint64_t> indep;
    std::string per_layer;
    for (size_t l = 0; l < ro.read.size(); ++l) {
        add(read,  ro.read[l]);
        add(indep, ro.indep[l]);
        per_layer += format(" %.2f/%.2f", mean(ro.read[l], nullptr), mean(ro.indep[l], nullptr));
    }

    uint64_t n_pairs = 0;
    uint64_t n_later = 0;
    const double m_read  = mean(read, &n_pairs);
    const double m_indep = mean(indep, nullptr);
    const double m_own   = mean(ro.own, &n_later);
    if (n_pairs == 0 && ro.n_short == 0) {
        return;
    }

    LLAMA_LOG_INFO("moe route once (k = %u, max %u, dry %d): %" PRIu64 " tokens, %" PRIu64 " full blocks x MoE layers "
            "(%" PRIu64 " shorter); distinct routed experts per block and layer: read mean %.2f, independent routing "
            "mean %.2f (%.2f x a token's %u); the later tokens' own top-%u inside their block's first token's set: "
            "mean %.2f; later tokens reading outside it: %" PRIu64 " of %" PRIu64 " token-layers\n",
            cparams.moe_route_once, cparams.moe_route_once_max, (int) cparams.moe_route_once_dry, ro.n_tokens, n_pairs,
            ro.n_short, m_read, m_indep, m_indep/n_used, n_used, n_used, m_own, ro.n_outside, n_later);
    LLAMA_LOG_INFO("moe route once: distinct experts read per full block and layer, by count:%s\n",
            by_count(read).c_str());
    LLAMA_LOG_INFO("moe route once: distinct experts under independent routing, by count:%s\n",
            by_count(indep).c_str());
    LLAMA_LOG_INFO("moe route once: later tokens' own experts inside their block's set, by count:%s\n",
            by_count(ro.own).c_str());
    LLAMA_LOG_INFO("moe route once: mean distinct experts per MoE layer (graph order), read/independent:%s\n",
            per_layer.c_str());
    if (by_layer) {
        for (size_t l = 0; l < ro.indep.size(); ++l) {
            LLAMA_LOG_INFO("moe route once: MoE layer %zu, distinct experts under independent routing, by count:%s\n",
                    l, by_count(ro.indep[l]).c_str());
        }
    }
}

bool llama_context::head_bound_init() {
    const ggml_tensor * output = model.output;
    if (!head_short) {
        LLAMA_LOG_ERROR("%s: the head bound needs the context's shortlist (llama_head_shortlist_init)\n", __func__);
        return false;
    }
    if (model.output_s || model.output_in_s) {
        LLAMA_LOG_ERROR("%s: the output head has scales, which the bound does not apply\n", __func__);
        return false;
    }
    if (model.prec_policy.prec_acc.count(model.output) || model.prec_policy.prec_src1.count(model.output)) {
        LLAMA_LOG_ERROR("%s: the output head has a precision policy, which the shortlist's products do not get\n", __func__);
        return false;
    }

    // the backend the head's buffer belongs to (head_shortlist_init accepted the buffer type)
    ggml_backend_t backend = nullptr;
    if (ggml_backend_buffer_is_host(output->buffer)) {
        backend = backend_cpu;
    } else {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(output->buffer));
        for (auto & b : backends) {
            if (ggml_backend_get_device(b.get()) == dev) {
                backend = b.get();
                break;
            }
        }
    }
    if (backend == nullptr) {
        LLAMA_LOG_ERROR("%s: no backend of this context holds the output head\n", __func__);
        return false;
    }
    for (const auto & fn : set_n_threads_fns) {
        if (fn.first == backend) {
            fn.second(backend, cparams.n_threads);
        }
    }

    const int64_t t0 = ggml_time_us();

    auto hb = std::make_unique<llama_head_bound>();
    if (!hb->init(model.output, backend)) {
        return false;
    }
    head_bound = std::move(hb);

    LLAMA_LOG_INFO("%s: head bound: %u rows of %s read in %.1f s; largest |W_t| %.4f, |W_t - mu| %.4f, |mu| %.4f; the full "
            "head on %s when a sampler can't exclude every token left out\n", __func__, head_bound->n_vocab,
            ggml_type_name(output->type), (ggml_time_us() - t0)/1e6, head_bound->norm[head_bound->by_norm[0]],
            head_bound->cnorm[head_bound->by_cnorm[0]], head_bound->mu_norm, ggml_backend_name(backend));

    return true;
}

void llama_context::set_head_bound(bool value) {
    // used for the decodes with one output (decode())
    head_bound_req = value && head_bound != nullptr;
}

void llama_context::head_bound_finish_row() {
    auto & hb = *head_bound;
    if (hb.row < 0 || hb.have_bound) {
        return;
    }
    hb.bound      = hb.compute(*head_short);
    hb.have_bound = true;

    // the top logit of the row is the shortlist's: every other entry is -INFINITY
    const float * row = logits.data + (int64_t) hb.row*model.vocab.n_tokens();
    float top = -INFINITY;
    for (uint32_t t = 0; t < hb.n_vocab; ++t) {
        top = std::max(top, row[t]);
    }
    const double gap = (double) hb.bound - top;
    hb.n_greedy += top > hb.bound;
    hb.sum_gap  += gap;
    hb.min_gap   = std::min(hb.min_gap, gap);
    hb.sum_top  += top;
    hb.sum_bnd  += hb.bound;
}

float llama_context::head_bound_ith(int32_t i) {
    if (!head_bound || head_bound->row < 0) {
        return INFINITY;
    }
    synchronize();
    output_reorder();

    int64_t j = -1;
    try {
        j = output_resolve_row(i);
    } catch (const std::exception &) {
        return INFINITY;
    }
    if (j != head_bound->row) {
        return INFINITY;
    }
    head_bound_finish_row();

    return head_bound->bound;
}

bool llama_context::head_bound_full(int32_t i) {
    if (!head_bound || head_bound->row < 0) {
        return false;
    }
    synchronize();
    output_reorder();

    int64_t j = -1;
    try {
        j = output_resolve_row(i);
    } catch (const std::exception &) {
        return false;
    }
    if (j != head_bound->row) {
        return false;
    }
    head_bound_finish_row();

    if (!head_bound->full(logits.data + j*model.vocab.n_tokens(), *head_short, head_bound->bound)) {
        return false;
    }
    head_bound->row = -1;
    head_bound->n_full++;

    return true;
}

void llama_context::head_bound_stats(uint64_t * n_rows, uint64_t * n_full) const {
    if (n_rows) {
        *n_rows = head_bound ? head_bound->n_rows : 0;
    }
    if (n_full) {
        *n_full = head_bound ? head_bound->n_full : 0;
    }
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph

    cparams.causal_attn = value;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // no scheduler reserve needed because warmups are usually with small batches

    cparams.warmup = value;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    // a server attaches each request's chain and detaches it at the end: a full reserve (a new scheduler and three
    // worst-case graphs) per request is not needed when the reserve already covered the same chains, and a detach only
    // makes the graphs smaller
    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        if (sampling.reserve_always || !sampler_reserve_covered()) {
            sched_need_reserve = true;
        }

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0 && sampling.reserve_always) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    if (sampling.reserve_always) {
        sched_need_reserve = true;
    }

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    // LLAMA_UBATCH_TRACE: the host's parts of this micro-batch until its first split leaves for a card [TAG_UBATCH_TRACE]
    const bool    trace    = llama_ubatch_trace() > 0;
    const int64_t t_trace0 = trace ? ggml_time_us() : 0;
    int64_t       t_trace_build = 0;
    int64_t       t_trace_alloc = 0;
    const char *  trace_why     = nullptr; // why the graph was rebuilt

    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    LLAMA_STEP_MARK("ub.apply");

    auto * res = get_gf_res_prev(ubatch.n_tokens);
    auto * gf  = res->get_gf();

    // [TAG_SCHED_PER_SIZE] a batch of up to 8 tokens uses the scheduler of its arena
    sched_small_entry * ss = nullptr;
    if (sched_per_size && ubatch.n_tokens <= 8) {
        const uint32_t key = gf_res_key(ubatch.n_tokens);
        for (auto & e : sched_small) {
            if (e.key == key) {
                ss = &e;
                break;
            }
        }
        if (ss == nullptr) {
            sched_small.push_back({ key, ggml_backend_sched_ptr(ggml_backend_sched_new(
                    backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), graph_max_nodes(8), false, cparams.op_offload)), nullptr });
            ss = &sched_small.back();
            ggml_backend_sched_set_copy_callback(ss->sched.get(), sched_copy_experts, this);
            ggml_backend_sched_set_realloc_callback(ss->sched.get(), sched_graph_realloc, this);
            ggml_backend_sched_set_input_upload(ss->sched.get(), plain_upload);
            LLAMA_LOG_INFO("%s: scheduler for %u-token batches (%s outputs), %zu per-size schedulers\n", __func__,
                    ubatch.n_tokens, n_outputs > 0 ? "with" : "without", sched_small.size());
        }
    }
    sched_active = ss ? ss->sched.get() : nullptr;

    // the draft's token rows from its row cache (embd_cache_init, Q961): the batch's slots are set and the rows it does
    // not hold uploaded before the graph is chosen, since the graph reads either the cache or the table
    if (embd_cache) {
        cparams.embd_cache_on = gtype == LLM_GRAPH_TYPE_DECODER_MTP && ubatch.token != nullptr && ubatch.embd == nullptr &&
                embd_cache->prepare(ubatch.token, ubatch.n_tokens, embd_cache_backend);
    }

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // With pipeline parallelism the previous graph may still be computing on the devices. A reused graph keeps the
    // previous ubatch's copy of the inputs, so it would have to wait for every device before set_inputs; a rebuilt graph
    // takes the scheduler's next input copy and overlaps with the ones in flight (up to n_copies ubatches, as for
    // prompts whose KV length changes every ubatch). So reuse only when nothing is in flight: one ubatch per decode
    // call after the caller synchronized, as in token generation and speculative verify steps, which keep their graphs.
    // LLAMA_GRAPH_REUSE_WAIT=1 restores reuse with the full wait.
    const bool reuse_safe = graph_reuse_wait || !(cparams.pipeline_parallel && gf_in_flight);

    // plain_upload: with nothing in flight, the scheduler need not wait for an earlier evaluation before it overwrites
    // the input copies, and stages the host inputs (packs are laid out when a graph is allocated, so this is set before);
    // cleared once the graph is submitted, so that the scheduler's other users (K-shift, training) keep the waits
    struct caller_sync_scope {
        ggml_backend_sched_t s;
        ~caller_sync_scope() { ggml_backend_sched_set_caller_sync(s, false); }
    } caller_sync_reset { sched_res() };
    ggml_backend_sched_set_caller_sync(sched_res(), plain_upload > 0 && !gf_in_flight);

    // a per-size scheduler has one copy of its inputs (it is created without pipeline parallelism): when an earlier
    // ubatch of this decode call is still computing, it may be this scheduler's own graph, so wait before its inputs
    // are written again (only a decode call of several ubatches whose last one has at most 8 tokens gets here)
    if (ss && gf_in_flight) {
        ggml_backend_sched_synchronize(ss->sched.get());
    }

    const bool allocated = ss ? ss->ready == res : gf_res_prev_active == res;
    const int64_t t_trace_pre = trace ? ggml_time_us() : 0;

    if (!graph_reuse_disable && reuse_safe && allocated && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading (only with LLAMA_GRAPH_REUSE_WAIT=1).
        if (cparams.pipeline_parallel && (gf_in_flight || plain_upload == 0)) {
            ggml_backend_sched_synchronize(sched_res());
        }

        n_reused++;
        t_trace_build = t_trace_alloc = trace ? ggml_time_us() : 0;
        LLAMA_STEP_MARK("ub.reuse");
    } else {
        if (trace) {
            // the same order as the test above: can_reuse ran only if the first three passed
            trace_why = graph_reuse_disable ? "disabled" : !reuse_safe ? "in_flight" : !allocated ? "not_allocated" :
                    res->reuse_reject ? res->reuse_reject : "unknown";
            while (*trace_why >= '0' && *trace_why <= '9') {
                trace_why++; // a type name from typeid starts with its length
            }
        }

        if (ss) {
            ss->ready = nullptr;
        } else {
            gf_res_prev_active = nullptr;
        }
        res->reset();

        ggml_backend_sched_reset(sched_res());
        ggml_backend_sched_set_eval_callback(sched_res(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        t_trace_build = trace ? ggml_time_us() : 0;

        if (!ggml_backend_sched_alloc_graph(sched_res(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        t_trace_alloc = trace ? ggml_time_us() : 0;

        if (ss) {
            ss->ready = res;
        } else {
            gf_res_prev_active = res;
        }
        LLAMA_STEP_MARK("ub.build");
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // LLAMA_GPU_KQ_MASK_CHECK=1 reads the devices' cell copies while it sets the inputs: let the last step finish
        static const bool kq_mask_check = [] {
            const char * env = getenv("LLAMA_GPU_KQ_MASK_CHECK");
            return env != nullptr && atoi(env) != 0;
        }();
        // so does LLAMA_QSA_KPOOL_DEV_CHECK=1 with the k-pool tables' copies
        const auto * mem_idx = dynamic_cast<const llama_memory_hybrid_idx *>(memory.get());
        if (kq_mask_check || (mem_idx && mem_idx->kpool_dev_check_on())) {
            ggml_backend_sched_synchronize(sched.get());
        }

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }
    LLAMA_STEP_MARK("ub.in");

    const int64_t t_trace_inputs = trace ? ggml_time_us() : 0;
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (trace) {
        // pre: memory, scheduler choice and graph parameters; build: the reuse check, then the graph's construction
        // (reused: the wait for the previous evaluation); alloc: the scheduler's split and allocation with any re-plan;
        // inputs: the inputs set; submit: every split queued and the CPU's computed; card0: from the start until the
        // first split off the CPU was launched (-1: none)
        const int64_t t_dev0 = ggml_backend_sched_get_trace_t_dev0(sched_res());
        const int64_t t_end  = ggml_time_us();
        LLAMA_LOG_WARN("ubatch-trace: %s n_tokens = %u n_seqs = %u n_outputs = %u graph = %s%s%s pre = %lld build = %lld "
                "alloc = %lld inputs = %lld submit = %lld card0 = %lld total = %lld us\n",
                gtype == LLM_GRAPH_TYPE_DECODER_MTP ? "mtp" : gtype == LLM_GRAPH_TYPE_ENCODER ? "enc" : "dec",
                ubatch.n_tokens, ubatch.n_seqs, n_outputs, trace_why ? "rebuilt" : "reused", trace_why ? " why = " : "",
                trace_why ? trace_why : "", (long long) (t_trace_pre - t_trace0), (long long) (t_trace_build - t_trace_pre),
                (long long) (t_trace_alloc - t_trace_build), (long long) (t_trace_inputs - t_trace_alloc),
                (long long) (t_end - t_trace_inputs), (long long) (t_dev0 ? t_dev0 - t_trace0 : -1),
                (long long) (t_end - t_trace0));
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }
    LLAMA_STEP_MARK("ub.comp");

    // LLAMA_QSA_POOL_MASK_DEV_CHECK=1 reads back the masks the devices built, once the graph has run
    if (const auto * mem_idx = dynamic_cast<const llama_memory_hybrid_idx *>(memory.get());
            mem_idx && mem_idx->pool_mask_dev_check_on() && res->needs_check_outputs()) {
        ggml_backend_sched_synchronize(sched_res());
        res->check_outputs(&ubatch);
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

//
// pipelined single-token decoding
//

static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq);

// the backend sampler of one sequence, as the samplers of a pipelined token's graph: each slot's graph samples only for
// the sequence it serves
static std::map<llama_seq_id, llama_sampler *> pipe_samplers(const std::map<llama_seq_id, llama_sampler *> & all, llama_seq_id seq_id) {
    std::map<llama_seq_id, llama_sampler *> res;
    const auto it = all.find(seq_id);
    if (it != all.end()) {
        res[seq_id] = it->second;
    }
    return res;
}

int32_t llama_context::pipe_init(int32_t n_slots, int32_t n_chunk_slots, int32_t n_chunk_tokens) {
    if (!memory || cparams.embeddings || !cparams.causal_attn) {
        LLAMA_LOG_ERROR("%s: pipelined decoding needs a causal decoder context with a memory module\n", __func__);
        return -1;
    }
    if (n_slots < 1 || n_chunk_slots < 0 || (n_chunk_slots > 0 && (n_chunk_tokens < 1 || n_chunk_tokens > (int32_t) cparams.n_ubatch))) {
        LLAMA_LOG_ERROR("%s: invalid slots (%d, %d chunk slots of %d tokens, n_ubatch %u)\n", __func__, n_slots,
                n_chunk_slots, n_chunk_tokens, cparams.n_ubatch);
        return -1;
    }
    if (model.hparams.n_swa > 0 && cparams.n_ubatch < (uint32_t) n_slots) {
        // the sliding-window cache keeps n_swa + n_ubatch cells and reuses the oldest one; with fewer spare cells than
        // tokens in flight a guess could overwrite a cell that a token re-decoded after a wrong guess still attends to
        LLAMA_LOG_ERROR("%s: n_ubatch (%u) must be at least the number of slots (%d) for a sliding-window model\n",
                __func__, cparams.n_ubatch, n_slots);
        return -1;
    }

    // a recurrent state can't drop a tail: each slot writes the recurrent states of its tokens to its own state group,
    // and a token reads its source state from whichever group holds it, so a wrong guess leaves the state before it in
    // place. The delta-net models (Qwen3.5/3.6, Qwen3-Next, Qwen3.8-Flash-Next with its n-gram conv history) write
    // their states that way; others are refused
    llama_memory_recurrent * mem_recr = nullptr;
    if (llama_model_is_recurrent(&model) || llama_model_is_hybrid(&model)) {
        auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
        mem_recr = mem_hybrid ? mem_hybrid->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(memory.get());
        const bool arch_ok = model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE ||
                             model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN4EXP;
        if (mem_recr == nullptr || !arch_ok || cparams.n_rs_seq != 0) {
            LLAMA_LOG_ERROR("%s: pipelined decoding of a recurrent model supports the delta-net models with "
                    "n_rs_seq 0 only (architecture %s, n_rs_seq %u)\n", __func__, llm_arch_name(model.arch), cparams.n_rs_seq);
            return -1;
        }
    }

    synchronize();
    pipe_drain();

    // the slots' graphs are in flight together, so each uploads the k-pool tables whole (LLAMA_QSA_KPOOL_DEV)
    if (auto * mem_idx = dynamic_cast<llama_memory_hybrid_idx *>(memory.get())) {
        mem_idx->kpool_dev_disable();
    }

    // one state group per slot and the stash, before the reserve, which drops every graph that still points at the old
    // state tensors
    pipe_mem_recr = nullptr;
    if (mem_recr) {
        if (!mem_recr->set_n_groups(n_slots + n_chunk_slots + 1)) {
            LLAMA_LOG_ERROR("%s: failed to set up %d recurrent state groups\n", __func__, n_slots + n_chunk_slots + 1);
            return -1;
        }
        pipe_mem_recr = mem_recr;
    }

    sched_reserve();

    // 2: each split's host inputs in one asynchronous copy from pinned staging memory; 1: one asynchronous copy per
    // input; 0: one synchronous copy per input, as llama_decode does
    pipe_upload = 2;
    if (const char * env = getenv("LLAMA_PIPE_UPLOAD")) {
        pipe_upload = std::clamp(atoi(env), 0, 2);
    }

    for (auto & slot : pipe_slots) {
        if (slot.ev_out) {
            ggml_backend_event_free(slot.ev_out);
        }
    }
    pipe_slots.clear();
    pipe_slots.resize(n_slots + n_chunk_slots);

    // the chunk slots' backends: a second backend per GPU with LLAMA_PIPE_CHUNK_STREAM, else the context's
    pipe_chunk_backends.clear();
    std::vector<ggml_backend_t> chunk_backend_ptrs = backend_ptrs;
    if (n_chunk_slots > 0 && pipe_chunk_stream) {
        for (size_t i = 0; i < backend_ptrs.size(); i++) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend_ptrs[i]);
            if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize a chunk backend on %s\n", __func__, ggml_backend_dev_name(dev));
                pipe_slots.clear();
                pipe_chunk_backends.clear();
                return -1;
            }
            pipe_chunk_backends.emplace_back(backend);
            chunk_backend_ptrs[i] = backend;
        }
    }

    const int64_t n_vocab   = model.vocab.n_tokens();
    const size_t  max_nodes = gf_res_reserve->get_max_nodes();

    // pinned host memory of the output device, as for the regular output buffer
    ggml_backend_buffer_type_t buft_out = ggml_backend_cpu_buffer_type();
    {
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft_out = output_dev_host_buft;
        }
    }

    // reserve every slot's compute buffers with a one-token graph over the full memory (the largest n_kv)
    llama_memory_context_ptr mctx = memory->init_full();
    if (!mctx) {
        LLAMA_LOG_ERROR("%s: failed to initialize the memory module\n", __func__);
        return -1;
    }

    // with one KV stream per sequence a graph over the full memory covers every stream, so the reserve takes one
    // sequence per stream, as sched_reserve does: 1 token per stream for a one-token slot, chunk / streams for a chunk
    // slot (the attention work and the buffers are those of the real evaluation or larger)
    const uint32_t n_stream = cparams.kv_unified ? 1 : cparams.n_seq_max;

    const auto save_n_outputs = this->n_outputs;
    this->n_outputs = n_stream;

    size_t size_total = 0;
    std::vector<size_t> size_dev(backend_ptrs.size(), 0); // per backend index, over all slots

    // the samplers of one sequence, for the reserve
    const auto reserve_samplers = sampling.samplers.empty() ? sampling.samplers
                                                            : pipe_samplers(sampling.samplers, sampling.samplers.begin()->first);

    for (size_t i_slot = 0; i_slot < pipe_slots.size(); i_slot++) {
        auto & slot = pipe_slots[i_slot];
        slot.n_tokens_max = i_slot < (size_t) n_slots ? 1 : n_chunk_tokens;
        slot.backends     = i_slot < (size_t) n_slots ? backend_ptrs : chunk_backend_ptrs;

        slot.sched.reset(ggml_backend_sched_new(
                slot.backends.data(), backend_buft.data(), slot.backends.size(), max_nodes, false, cparams.op_offload));
        ggml_backend_sched_set_copy_callback(slot.sched.get(), sched_copy_experts, this);
        ggml_backend_sched_set_caller_sync(slot.sched.get(), true);
        ggml_backend_sched_set_input_upload(slot.sched.get(), pipe_upload);

        slot.res.reset(new llm_graph_result(max_nodes));

        llama_batch_allocr balloc_reserve(model.hparams.n_pos_per_embd());
        llama_ubatch ubatch = balloc_reserve.ubatch_reserve((slot.n_tokens_max + n_stream - 1)/n_stream, n_stream);

        ubatch_prepare_reserve(ubatch, n_stream, reserve_samplers, cparams.n_outputs_max_per_seq);

        auto gparams = graph_params(slot.res.get(), ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));
        gparams.sched    = slot.sched.get();
        gparams.cb       = graph_get_cb(slot.sched.get());
        gparams.samplers = reserve_samplers;

        slot.res->reset();

        auto * gf = model.build_graph(gparams);
        if (!gf || !ggml_backend_sched_reserve(slot.sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to reserve the compute buffers of a slot\n", __func__);
            this->n_outputs = save_n_outputs;
            pipe_slots.clear();
            return -1;
        }
        slot.res->reset();
        slot.res_valid = false;

        for (size_t i = 0; i < slot.backends.size(); i++) {
            const size_t size = ggml_backend_sched_get_buffer_size(slot.sched.get(), slot.backends[i]);
            size_total  += size;
            size_dev[i] += size;
        }

        slot.buf_out.reset(ggml_backend_buft_alloc_buffer(buft_out, n_vocab*sizeof(float)));
        if (!slot.buf_out) {
            LLAMA_LOG_ERROR("%s: failed to allocate the output buffer of a slot\n", __func__);
            this->n_outputs = save_n_outputs;
            pipe_slots.clear();
            return -1;
        }
        slot.logits = (float *) ggml_backend_buffer_get_base(slot.buf_out.get());
    }

    this->n_outputs = save_n_outputs;

    LLAMA_LOG_INFO("%s: %d slots + %d chunk slots of %d tokens, compute buffers %.2f MiB in total, %d splits per token, "
            "input upload %d, chunk streams %zu\n", __func__, n_slots, n_chunk_slots, n_chunk_tokens,
            size_total/1024.0/1024.0, ggml_backend_sched_get_n_splits(pipe_slots[0].sched.get()), pipe_upload,
            pipe_chunk_backends.size());

    // what pipelining costs each device (the memory budget on cards that hold a few layers each)
    for (size_t i = 0; i < backend_ptrs.size(); i++) {
        LLAMA_LOG_INFO("%s: %10s pipe slots' compute buffers = %8.2f MiB\n", __func__,
                ggml_backend_name(backend_ptrs[i]), size_dev[i]/1024.0/1024.0);
    }

    return 0;
}

int32_t llama_context::pipe_draft_init(const llama_pipe_draft_params & params) {
    if (params.path == nullptr) {
        LLAMA_LOG_ERROR("%s: no drafter file\n", __func__);
        return -1;
    }
    // the drafter's device: the params.device-th GPU, or the first layer's device
    ggml_backend_dev_t dev = nullptr;
    if (params.device >= 0) {
        int32_t k = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && k++ == params.device) {
                dev = d;
                break;
            }
        }
    } else {
        dev = model.dev_layer(0);
    }
    if (dev == nullptr) {
        LLAMA_LOG_ERROR("%s: no device %d for the drafter\n", __func__, params.device);
        return -1;
    }

    // the layer split as the devices run it: layers per device in order (the output head goes with the last)
    const int32_t n_layer = (int32_t) model.hparams.n_layer();
    std::vector<int32_t> split;
    bool dev_has_layers = false;
    for (int32_t il = 0; il < n_layer; il++) {
        ggml_backend_dev_t dl = model.dev_layer(il);
        if (il == 0 || dl != model.dev_layer(il - 1)) {
            split.push_back(0);
        }
        split.back()++;
        dev_has_layers = dev_has_layers || dl == dev;
    }

    synchronize();
    pipe_drain();

    auto d = llama_pipe_drafter::load(params, model.tok_embd, dev, split, n_layer, dev_has_layers);
    if (!d) {
        return -1;
    }
    for (int32_t l : d->tap_layers()) {
        if (l <= 0 || l >= (int32_t) cparams.pipe_tap_layer.size() - 1) {
            LLAMA_LOG_ERROR("%s: the drafter reads layer %d, the model has %u layers\n", __func__, l, model.hparams.n_layer());
            return -1;
        }
        cparams.pipe_tap_layer[l] = true;
    }
    pipe_drafter = std::move(d);
    pipe_drafter->reset(-1, 0);

    // graphs are built again, with the taps kept as outputs
    gf_res_invalidate();
    for (auto & slot : pipe_slots) {
        slot.res_valid = false;
    }
    return 0;
}

void llama_context::pipe_draft_reset(llama_seq_id seq_id, llama_pos n_keep) {
    if (!pipe_drafter) {
        return;
    }
    pipe_drafter->reset(seq_id, n_keep);
    // prompt micro-batches of an active drafter also compute every position's final hidden state
    const bool all = seq_id >= 0;
    if (cparams.pipe_embd_all != all) {
        cparams.pipe_embd_all = all;
        gf_res_invalidate();
    }
}

int32_t llama_context::pipe_submit(int32_t i_slot, const llama_token * tokens, int32_t n_tokens, llama_pos pos, llama_seq_id seq_id) {
    return pipe_submit_impl(i_slot, tokens, n_tokens, pos, seq_id, -1);
}

int32_t llama_context::pipe_submit_fed(int32_t i_slot, int32_t i_slot_src, llama_pos pos, llama_seq_id seq_id) {
    // the KV cells of M-RoPE and PLE models record their token (llama_kv_cache::has_cell_ext: saved with the state, and
    // read back by the PLE n-gram inputs of later tokens), but a fed token's id is still on the device when its cell is
    // written, so the cell would record the placeholder below
    if (model.hparams.n_pos_per_embd() > 1 || model.hparams.ple_n_heads > 0) {
        LLAMA_LOG_ERROR("%s: fed tokens need a model whose KV cells keep no token (not M-RoPE or PLE)\n", __func__);
        return -1;
    }
    if (i_slot_src < 0 || i_slot_src >= (int32_t) pipe_slots.size() || i_slot_src == i_slot) {
        LLAMA_LOG_ERROR("%s: invalid source slot %d\n", __func__, i_slot_src);
        return -1;
    }
    const auto & src = pipe_slots[i_slot_src];
    if (!src.res_valid || src.res->t_sampled.empty() || src.res->t_sampled[0] == nullptr) {
        LLAMA_LOG_ERROR("%s: slot %d holds no evaluation with a sampler on the device\n", __func__, i_slot_src);
        return -1;
    }
    // any valid id: the host's copy of the token input is not uploaded
    const llama_token placeholder = 0;
    return pipe_submit_impl(i_slot, &placeholder, 1, pos, seq_id, i_slot_src);
}

int32_t llama_context::pipe_submit_impl(int32_t i_slot, const llama_token * tokens, int32_t n_tokens, llama_pos pos,
                                        llama_seq_id seq_id, int32_t i_slot_src) {
    if (i_slot < 0 || i_slot >= (int32_t) pipe_slots.size()) {
        LLAMA_LOG_ERROR("%s: invalid slot %d\n", __func__, i_slot);
        return -1;
    }

    auto & slot = pipe_slots[i_slot];

    if (n_tokens < 1 || n_tokens > slot.n_tokens_max) {
        LLAMA_LOG_ERROR("%s: slot %d takes 1..%d tokens, got %d\n", __func__, i_slot, slot.n_tokens_max, n_tokens);
        return -1;
    }

    // the slot's scheduler runs with caller_sync: its previous evaluation must be complete before it is reused
    if (slot.busy) {
        ggml_backend_event_synchronize(slot.ev_out);
        slot.busy = false;
    }

    // n_tokens tokens of one sequence at pos, pos + 1, ...; only the last one is output
    std::vector<llama_token>    toks(tokens, tokens + n_tokens);
    std::vector<llama_pos>      ps(n_tokens);
    std::vector<int32_t>        n_sq(n_tokens, 1);
    llama_seq_id                sid = seq_id;
    std::vector<llama_seq_id *> sids(n_tokens, &sid);
    std::vector<int8_t>         outs(n_tokens, 0);
    for (int32_t i = 0; i < n_tokens; i++) {
        ps[i] = pos + i;
    }
    outs[n_tokens - 1] = 1;

    const llama_batch batch = { n_tokens, toks.data(), nullptr, ps.data(), n_sq.data(), sids.data(), outs.data() };

    llama_batch_compat compat(this, batch);

    if (!balloc->init(*compat.batch_ext, model.vocab, false)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    // handle any pending shifts/copies
    memory_update(false);

    // the slot's recurrent states go to its own state group, which may hold the state a rollback still needs
    if (pipe_mem_recr) {
        pipe_protect_state(i_slot, seq_id, pos);
        pipe_mem_recr->grp_next = (uint32_t) i_slot;
    }
    llama_memory_context_ptr mctx = memory->init_batch(*balloc, cparams.n_ubatch, false);
    if (pipe_mem_recr) {
        pipe_mem_recr->grp_next = 0;
    }
    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to find a memory slot for pos %d\n", __func__, pos);
        return 1;
    }

    const auto & ubatch = mctx->get_ubatch();

    if (!mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        return -2;
    }

    const auto samplers = pipe_samplers(sampling.samplers, seq_id);
    for (const auto & entry : samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    const auto save_n_outputs = this->n_outputs;
    this->n_outputs = 1;

    auto gparams = graph_params(slot.res.get(), ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));
    gparams.sched    = slot.sched.get();
    gparams.cb       = graph_get_cb(slot.sched.get());
    gparams.samplers = samplers;

    ggml_cgraph * gf = slot.res->get_gf();

    if (!(slot.res_valid && !graph_reuse_disable && slot.res->can_reuse(gparams))) {
        slot.res->reset();
        slot.res_valid = false;

        ggml_backend_sched_reset(slot.sched.get());

        gf = model.build_graph(gparams);
        if (!gf || !ggml_backend_sched_alloc_graph(slot.sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to build or allocate the graph\n", __func__);
            this->n_outputs = save_n_outputs;
            memory->seq_rm(seq_id, pos, -1);
            return -2;
        }

        slot.res_valid = true;

        // the token input's copy on the one device that reads the token ids; none if the CPU reads them (a node of
        // the split graph that still has the input itself as a source runs on the CPU) or several devices do
        slot.tok_cpy     = nullptr;
        slot.tok_backend = nullptr;
        ggml_tensor * t_tok = slot.res->get_inp_tokens();
        bool cpu_reads = t_tok == nullptr;
        for (int i = 0; t_tok && i < ggml_graph_n_nodes(gf) && !cpu_reads; i++) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                cpu_reads = cpu_reads || node->src[j] == t_tok;
            }
        }
        int n_dev_copies = 0;
        for (auto * backend : slot.backends) {
            ggml_tensor * cpy = t_tok ? ggml_backend_sched_get_input_copy(slot.sched.get(), t_tok, backend) : nullptr;
            if (cpy != nullptr && backend != backend_cpu) {
                slot.tok_cpy     = cpy;
                slot.tok_backend = backend;
                n_dev_copies++;
            }
        }
        if (cpu_reads || n_dev_copies != 1) {
            slot.tok_cpy     = nullptr;
            slot.tok_backend = nullptr;
        }
    }

    this->n_outputs = save_n_outputs;

    // LLAMA_GPU_KQ_MASK_CHECK=1 reads the devices' cell copies while it sets the inputs: the tokens still in flight
    // must have written theirs first (as in process_ubatch; only the check needs it, the devices run in order anyway)
    static const bool kq_mask_check = [] {
        const char * env = getenv("LLAMA_GPU_KQ_MASK_CHECK");
        return env != nullptr && atoi(env) != 0;
    }();
    if (kq_mask_check) {
        for (auto * backend : backend_ptrs) {
            ggml_backend_synchronize(backend);
        }
    }

    slot.res->set_inputs(&ubatch);

    // same thread settings as graph_compute() uses for a single token
    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, threadpool);
        }
    }
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, cparams.n_threads);
    }
    copy_experts.reset();

    // a fed token: the sampled id of slot i_slot_src goes from its device to the token input's copy on the device
    // that embeds the tokens, ordered after that evaluation's sampling and before this evaluation on both devices
    if (slot.res->get_inp_tokens() != nullptr) {
        ggml_backend_sched_set_input_skip(slot.sched.get(), slot.res->get_inp_tokens(), i_slot_src >= 0);
    }
    if (i_slot_src >= 0) {
        auto & src = pipe_slots[i_slot_src];
        ggml_tensor *  t_src  = src.res->t_sampled[0];
        ggml_backend_t be_src = ggml_backend_sched_get_tensor_backend(src.sched.get(), t_src);
        ggml_tensor *  t_dst  = slot.tok_cpy;
        if (t_dst == nullptr || be_src == nullptr || t_src->type != t_dst->type || ggml_nelements(t_src) != 1 ||
                ggml_nelements(t_dst) != 1 || ggml_nbytes(t_src) != ggml_nbytes(t_dst)) {
            LLAMA_LOG_ERROR("%s: a fed token needs the token ids read on one device only (LLAMA_INPUT_ON_GPU=1) and a "
                    "one-token sampler output\n", __func__);
            ggml_backend_sched_set_input_skip(slot.sched.get(), slot.res->get_inp_tokens(), false);
            memory->seq_rm(seq_id, pos, -1);
            return -1;
        }
        if (src.ev_out != nullptr) {
            // the device that embeds the tokens waits for the source evaluation alone: its event follows the copy of
            // the sampled id into pinned host memory, from which this device's stream uploads it. A copy queued on the
            // sampling device's stream would also wait for everything submitted there since (other conversations'
            // tokens in streams mode), so every fed token would drain the pipeline
            ggml_backend_event_wait(slot.tok_backend, src.ev_out);
            ggml_backend_tensor_set_async(slot.tok_backend, t_dst, src.logits, 0, sizeof(llama_token));
        } else {
            ggml_backend_tensor_copy_async(be_src, slot.tok_backend, t_src, t_dst);
        }
    }

    const auto status = ggml_backend_sched_graph_compute_async(slot.sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: graph compute failed with status %d\n", __func__, status);
        memory->seq_rm(seq_id, pos, -1);
        return -3;
    }

    // the staircase drafter's taps: each device copies out its anchors right after its work for this token
    if (pipe_drafter && n_tokens == 1 && i_slot_src < 0) {
        pipe_drafter->on_submit(i_slot, tokens[0], pos, seq_id, slot.res.get(), slot.sched.get());
    }

    // with a backend sampler only the sampled token comes back, otherwise the logits
    ggml_tensor * t_out = nullptr;
    size_t        n_out = 0;
    if (!samplers.empty()) {
        GGML_ASSERT(!slot.res->t_sampled.empty() && slot.res->t_sampled[0] != nullptr);
        t_out = slot.res->t_sampled[0];
        n_out = sizeof(llama_token);
    } else {
        t_out = slot.res->get_logits();
        n_out = model.vocab.n_tokens()*sizeof(float);
    }
    GGML_ASSERT(t_out != nullptr);

    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(slot.sched.get(), t_out);
    GGML_ASSERT(backend_res != nullptr);

    ggml_backend_tensor_get_async(backend_res, t_out, slot.logits, 0, n_out);

    ggml_backend_dev_t dev_res = ggml_backend_get_device(backend_res);
    if (slot.ev_out == nullptr || slot.ev_dev != dev_res) {
        if (slot.ev_out) {
            ggml_backend_event_free(slot.ev_out);
        }
        slot.ev_out   = ggml_backend_event_new(dev_res);
        slot.ev_dev   = dev_res;
        // a device without a registry (the meta device of a tensor split) has no event query
        ggml_backend_reg_t reg_res = ggml_backend_dev_backend_reg(dev_res);
        slot.ev_query = reg_res == nullptr ? nullptr : (bool (*)(ggml_backend_event_t)) ggml_backend_reg_get_proc_address(
                reg_res, "ggml_backend_event_query");
    }

    if (slot.ev_out) {
        ggml_backend_event_record(slot.ev_out, backend_res);
    } else {
        // the output device has no events: wait for the logits now
        ggml_backend_synchronize(backend_res);
    }

    slot.busy = slot.ev_out != nullptr;

    slot.live = true;
    slot.seq  = seq_id;
    slot.pos  = pos;

    return 0;
}

void llama_context::pipe_protect_state(int32_t i_slot, llama_seq_id seq_id, llama_pos pos) {
    // the sequence's tokens still to be read: an executor submits a position again only after dropping the tokens
    // from there on, and reads a slot's output before it reuses the slot
    llama_pos front = pos;
    for (size_t k = 0; k < pipe_slots.size(); k++) {
        auto & s = pipe_slots[k];
        if (!s.live || s.seq != seq_id) {
            continue;
        }
        if ((int32_t) k == i_slot || s.pos >= pos) {
            s.live = false;
            continue;
        }
        front = std::min(front, s.pos);
    }

    // the state before the oldest of them: a wrong guess or leaving pipelined decoding rolls back to it at most
    const int32_t row = pipe_mem_recr->pipe_stash(seq_id, (uint32_t) i_slot, front - 1);
    if (row < 0) {
        return;
    }

    // copy that row of every recurrent layer from the slot's group to the stash, on the layer's device and queued
    // before this slot's evaluation, which overwrites it
    const size_t   size  = pipe_mem_recr->size;
    const uint32_t stash = pipe_mem_recr->n_grp - 1;

    std::vector<ggml_tensor *> tensors;
    for (const auto * v : { &pipe_mem_recr->r_l, &pipe_mem_recr->s_l, &pipe_mem_recr->p_l }) {
        for (ggml_tensor * t : *v) {
            if (t != nullptr) {
                tensors.push_back(t);
            }
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 2*tensors.size()*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_views { ggml_init(params) };

    for (ggml_tensor * t : tensors) {
        ggml_tensor * src = ggml_view_1d(ctx_views.get(), t, t->ne[0], ((size_t) i_slot*size + row)*t->nb[1]);
        ggml_tensor * dst = ggml_view_1d(ctx_views.get(), t, t->ne[0], ((size_t) stash *size + row)*t->nb[1]);
        const ggml_status st_src = ggml_backend_view_init(src);
        const ggml_status st_dst = ggml_backend_view_init(dst);
        GGML_ASSERT(st_src == GGML_STATUS_SUCCESS && st_dst == GGML_STATUS_SUCCESS);

        // host memory (the CPU buffer type has no device): the CPU runs its part of every evaluation before the
        // submit returns, so a plain copy now is in order
        if (ggml_backend_buffer_is_host(t->buffer)) {
            ggml_backend_tensor_copy(src, dst);
            continue;
        }

        // the slot's own backend for the device (with LLAMA_PIPE_CHUNK_STREAM a chunk slot's runs on a stream of its
        // own), so the copy is ordered after the slot's last write of the group and before its next one
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
        ggml_backend_t backend = nullptr;
        for (auto * be : pipe_slots[i_slot].backends) {
            if (ggml_backend_get_device(be) == dev) {
                backend = be;
                break;
            }
        }
        GGML_ASSERT(backend != nullptr);

        ggml_backend_tensor_copy_async(backend, backend, src, dst);
    }

    LLAMA_LOG_DEBUG("%s: seq %d: the state at pos %d goes from group %d to the stash\n", __func__, seq_id, front - 1, i_slot);
}

float * llama_context::pipe_get_logits(int32_t i_slot) {
    if (i_slot < 0 || i_slot >= (int32_t) pipe_slots.size()) {
        return nullptr;
    }

    auto & slot = pipe_slots[i_slot];

    if (slot.busy) {
        ggml_backend_event_synchronize(slot.ev_out);
        slot.busy = false;
    }

    // read: no longer in flight for the executor
    slot.live = false;

    return slot.logits;
}

llama_token llama_context::pipe_get_sampled(int32_t i_slot) {
    if (i_slot < 0 || i_slot >= (int32_t) pipe_slots.size() || sampling.samplers.empty()) {
        return LLAMA_TOKEN_NULL;
    }
    // note: the slot's last token must belong to a sequence with a backend sampler

    const float * out = pipe_get_logits(i_slot);

    llama_token id;
    memcpy(&id, out, sizeof(id));

    return id;
}

bool llama_context::pipe_ready(int32_t i_slot) {
    if (i_slot < 0 || i_slot >= (int32_t) pipe_slots.size()) {
        return false;
    }

    auto & slot = pipe_slots[i_slot];

    if (slot.busy) {
        if (slot.ev_query == nullptr || !slot.ev_query(slot.ev_out)) {
            return false;
        }
        slot.busy = false;
    }

    return true;
}

void llama_context::pipe_drain() {
    for (auto & slot : pipe_slots) {
        if (slot.busy) {
            ggml_backend_event_synchronize(slot.ev_out);
            slot.busy = false;
        }
    }
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    sched_active = nullptr; // the reserve and memory updates below use the shared scheduler

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    if (!batch_inp.tokens.empty() && batch_inp.tokens[0].embd_row >= 0) {
        LLAMA_LOG_ERROR("%s: embd row batches are not supported by encode\n", __func__);
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we skip position continuity checks and output all tokens
    //       a decoder without memory (see decode()) outputs the marked tokens, unless it extracts embeddings
    if (!balloc->init(batch_inp, model.vocab, memory || cparams.embeddings)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    const uint32_t n_outputs_all = balloc->get_n_outputs();

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs_all);
        return -2;
    };

    const auto & out_ids = balloc->get_out_ids();
    for (uint32_t i = 0; i < n_outputs_all; ++i) {
        output_ids[out_ids[i]] = i;
    }

    n_outputs = n_outputs_all;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched_res(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_outputs*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched_res(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_outputs*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_outputs*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched_res(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_outputs*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_outputs*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    const int64_t t_trace0 = llama_ubatch_trace() > 0 ? ggml_time_us() : 0; // [TAG_UBATCH_TRACE]
    LLAMA_STEP_MARK("dec");

    sched_active = nullptr; // the reserve and memory updates below use the shared scheduler

    // Q771: the caller's taper hints are for this call only
    const int32_t taper_hint_head = taper_next_head;
    const int32_t taper_hint_tail = taper_next_tail;
    taper_next_head = -1;
    taper_next_tail = -1;

    // C217: the device-store fills requested for this call; their results stay until the next call
    store_at_last = std::move(store_at_next);
    store_at_next.clear();

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // a regular decode may reserve or update the memory module: let pipelined evaluations finish first
    pipe_drain();

    // and a sequence decoded here has left pipelined decoding: none of its pipelined tokens is still to be read (what
    // the recurrent state groups keep depends on that)
    if (pipe_mem_recr) {
        for (const auto & tok : batch_inp.tokens) {
            for (auto & ps : pipe_slots) {
                if (ps.live && (tok.seq_ids.empty() ? ps.seq == 0 : tok.seq_ids.count(ps.seq) > 0)) {
                    ps.live = false;
                }
            }
        }
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const bool has_embd_row = !batch_inp.tokens.empty() && batch_inp.tokens[0].embd_row >= 0;
    if (has_embd_row && !supports_embd_row()) {
        LLAMA_LOG_ERROR("%s: this context cannot read embd rows (needs a DFlash/DSpark decoder and a layer input device buffer in ctx_other)\n", __func__);
        return -1;
    }

    const bool has_state_row = !batch_inp.tokens.empty() && batch_inp.tokens[0].state_row >= 0;
    if (has_state_row && !supports_state_row()) {
        LLAMA_LOG_ERROR("%s: this context cannot read state rows (needs an MTP context and an h_nextn device buffer in ctx_state_src)\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }
    LLAMA_STEP_MARK("dec.bal");

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    // C217: the micro-batch cuts of this call, the caller's and the fills' (a batch of one sequence)
    if ((!ubatch_cuts.empty() || !store_at_last.empty()) && balloc->is_one_seq()) {
        const llama_seq_id seq_id = balloc->get_batch().seq_id[0][0];

        std::vector<uint32_t> cuts;
        for (const uint32_t c : ubatch_cuts) {
            if (c > 0 && c < n_tokens_all) {
                cuts.push_back(c);
            }
        }
        for (const auto & r : store_at_last) {
            if (r.seq_id == seq_id && r.i_end > 0 && r.i_end < n_tokens_all) {
                cuts.push_back(r.i_end);
            }
        }
        balloc->set_ubatch_cuts(std::move(cuts));
    }

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // the main model's shortlist head (Q619): the last row's counts, the batch's tokens join the shortlist (which waits
    // for the previous decode, before this one's tokens are queued), and a decode with one output computes the
    // shortlist's logits
    if (head_bound) {
        head_bound_finish_row();
        head_bound->row = -1;
        cparams.head_bound_on = head_bound_req && n_outputs_all == 1 && sampling.samplers.empty();
        if (cparams.head_bound_on && balloc->get_batch().token) {
            head_shortlist_add(balloc->get_batch().token, n_tokens_all);
        }
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    logits_dev_tensor = nullptr;
    logits_dev_rows.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    // Q771: the micro-batch taper's plan for this call, which the memory module's splits follow; without one (taper off,
    // other batches, or a plan equal to the plain split) they take n_ubatch tokens at a time as before. The head rule
    // applies when nothing is in flight now (after every wait above), unless the caller said otherwise.
    const bool taper_log = taper.enabled() && n_tokens_all > 8;
    bool taper_head = false;
    bool taper_tail = false;
    std::vector<uint32_t> taper_sizes;
    if (taper.enabled() && !embd_all && balloc->is_one_seq_tokens()) {
        taper_head = taper_hint_head < 0 ? !gf_in_flight : taper_hint_head > 0;
        taper_tail = taper_hint_tail != 0;
        balloc->set_ubatch_plan(llama_ubatch_taper_plan(taper, n_tokens_all, cparams.n_ubatch, taper_head, taper_tail));
    }

    llama_memory_context_ptr mctx;

    const int64_t t_trace_init = t_trace0 ? ggml_time_us() : 0;
    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }
    LLAMA_STEP_MARK("dec.mem");

    const int64_t t_trace_wait = t_trace0 ? ggml_time_us() : 0;

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    // ordering for the layer input device buffer
    llama_context * inject_src     = has_embd_row ? cparams.ctx_other : nullptr;
    ggml_backend_t  inject_backend = nullptr;
    if (inject_src) {
        inject_backend = inject_src->ev_layer_inp_written ? backend_from_dev(inject_src->layer_inp_dev_dev) : nullptr;
        if (inject_backend) {
            // wait on the host: a device-side wait would stall our queue for the whole target batch,
            // which trips the Metal command buffer timeout on long prompts
            ggml_backend_event_synchronize(inject_src->ev_layer_inp_written.get());
        } else {
            inject_src->synchronize();
        }
    }
    if (layer_inp_dev && ev_layer_inp_read && ev_layer_inp_read_recorded) {
        // do not overwrite rows the reader has not consumed yet
        ggml_backend_event_wait(layer_inp_backend, ev_layer_inp_read.get());
    }

    // ordering for the h_nextn device buffer, as for the layer input buffer
    // (C703: per half of the buffer; the batch's rows say which halves this decode reads)
    llama_context * state_src     = has_state_row ? cparams.ctx_state_src : nullptr;
    ggml_backend_t  state_backend = nullptr;
    bool            state_half[2] = { false, false };
    if (state_src) {
        state_backend = backend_from_dev(state_src->h_nextn_dev_dev);
        for (const auto & t : batch_inp.tokens) {
            state_half[state_src->h_nextn_half_of(t.state_row)] = true;
        }
        for (int h = 0; h < 2; ++h) {
            if (!state_half[h]) {
                continue;
            }
            if (state_src->ev_h_nextn_written_recorded[h]) {
                ggml_backend_event_synchronize(state_src->ev_h_nextn_written[h].get());
            } else {
                state_src->synchronize();
            }
        }
    }
    const bool write_h_nextn = cparams.embeddings_nextn_dev && h_nextn_dev != nullptr;
    int32_t    h_nextn_w     = 0;
    if (write_h_nextn) {
        // C703: a held half stays as it is, this decode writes the other one (the hold ends with a decode that completes)
        h_nextn_w    = h_nextn_held ? 1 - h_nextn_last : h_nextn_last;
        h_nextn_row0 = h_nextn_w*h_nextn_half_rows();
    }
    if (write_h_nextn && ev_h_nextn_read_recorded[h_nextn_w]) {
        ggml_backend_event_wait(h_nextn_backend, ev_h_nextn_read[h_nextn_w].get());
    }

    if (t_trace0) {
        // the call's host parts before its first micro-batch: prep (checks, waits for pipelined steps, the batch split
        // plan), init (the memory's slots for every micro-batch), wait (the output buffer, the other context's rows)
        const int64_t t = ggml_time_us();
        LLAMA_LOG_WARN("decode-trace: %s n_tokens = %u n_outputs = %u prep = %lld init = %lld wait = %lld us\n",
                cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "mtp" : "dec", n_tokens_all, n_outputs_all,
                (long long) (t_trace_init - t_trace0), (long long) (t_trace_wait - t_trace_init), (long long) (t - t_trace_wait));
    }
    LLAMA_STEP_MARK("dec.out");

    do {
        const auto & ubatch = mctx->get_ubatch();

        if (taper_log) {
            taper_sizes.push_back(ubatch.n_tokens);
        }

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // the staircase drafter commits this micro-batch's positions of its sequence
        if (pipe_drafter) {
            pipe_drafter->on_ubatch(ubatch, res, sched_res());
        }

        // C217: device-store fills where this micro-batch ends, the state as a call that ended here leaves it
        for (auto & r : store_at_last) {
            if (!r.done && r.i_end == n_tokens_prev + ubatch.n_tokens && ubatch.n_seqs_unq == 1 && ubatch.seq_id_unq[0] == r.seq_id) {
                if (!state_seq_fill_store_at(r)) {
                    LLAMA_LOG_WARN("%s: store %d of seq %d could not be filled inside the call\n", __func__, r.store, r.seq_id);
                }
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        // a draft pass that keeps its hidden rows on the device for the next pass (set_nextn_carry): no copy to the host
        if (cparams.nextn_carry && cparams.nextn_carry_out) {
            t_h_nextn = nullptr;
        }

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // this ubatch reused the compute buffers: rows that an earlier ubatch left on the device are gone
        for (auto & r : logits_dev_rows) {
            r = r >= 0 ? -2 : r;
        }
        if (logits_dev_rows.size() < (size_t) n_outputs_all) {
            logits_dev_rows.resize(n_outputs_all, -1);
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && !needs_raw_logits(ubatch, sampling.samplers)) {
            // every output of this ubatch is sampled by a backend sampler: its logits stay on the device
            logits_dev_tensor = t_logits;
            for (uint32_t k = 0; k < n_outputs; k++) {
                logits_dev_rows[n_outputs_prev + k] = (int32_t) k;
            }
        }
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched_res(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // the shortlist head's input row, for the bound and the fallback (head_bound_ith, head_bound_full)
        if (head_bound && res->t_head_h && n_outputs == 1 && logits.data && t_logits && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched_res(), res->t_head_h);
            GGML_ASSERT(backend_h != nullptr && ggml_nelements(res->t_head_h) == (int64_t) head_bound->n_embd);
            ggml_backend_tensor_get_async(backend_h, res->t_head_h, head_bound->h.data(), 0, ggml_nbytes(res->t_head_h));
            head_bound->row        = (int32_t) n_outputs_prev;
            head_bound->have_bound = false;
            head_bound->n_rows++;
            if (head_bound->n_rows % 4096 == 0) {
                head_bound->log(__func__);
            }
        }

        // the routed experts kept per token (a synchronous read)
        if (cparams.moe_tail && cparams.moe_tail_hist) {
            moe_tail_collect(res, ubatch.n_tokens);
        }

        // the distinct routed experts per block of route once (a synchronous read)
        if (cparams.moe_route_once > 0 && cparams.moe_route_once_hist) {
            moe_route_once_collect(res, ubatch, n_outputs);
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched_res(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            // the device buffer replaces the host extraction, see build_h_nextn_dev()
            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE && !write_h_nextn) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched_res(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched_res());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched_res(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched_res(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched_res(), &sampling.candidates_count);
        }
        LLAMA_STEP_MARK("ub.cp");

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    if (layer_inp_dev && ev_layer_inp_written) {
        ggml_backend_event_record(ev_layer_inp_written.get(), layer_inp_backend);
    }
    if (write_h_nextn) {
        if (ev_h_nextn_written[h_nextn_w]) {
            ggml_backend_event_record(ev_h_nextn_written[h_nextn_w].get(), h_nextn_backend);
            ev_h_nextn_written_recorded[h_nextn_w] = true;
        }
        h_nextn_last = h_nextn_w;
        h_nextn_held = false;
    }
    if (state_src) {
        if (state_backend) {
            for (int h = 0; h < 2; ++h) {
                if (state_half[h] && state_src->ev_h_nextn_read[h]) {
                    ggml_backend_event_record(state_src->ev_h_nextn_read[h].get(), state_backend);
                    state_src->ev_h_nextn_read_recorded[h] = true;
                }
            }
        } else {
            synchronize();
        }
    }
    if (inject_src) {
        if (inject_backend) {
            ggml_backend_event_record(inject_src->ev_layer_inp_read.get(), inject_backend);
            inject_src->ev_layer_inp_read_recorded = true;
        } else {
            synchronize();
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // Q771: the micro-batch sizes of this call, run-length coded (n x count)
    if (taper_log) {
        std::string sizes;
        for (size_t i = 0; i < taper_sizes.size(); ) {
            size_t j = i;
            while (j < taper_sizes.size() && taper_sizes[j] == taper_sizes[i]) {
                ++j;
            }
            sizes += (sizes.empty() ? "" : " ") + std::to_string(taper_sizes[i]) + (j - i > 1 ? "x" + std::to_string(j - i) : "");
            i = j;
        }
        LLAMA_LOG_INFO("%s: ubatch taper: %u tokens, head %d, tail %d, %zu ubatches: %s\n", __func__, n_tokens_all,
                (int) taper_head, (int) taper_tail, taper_sizes.size(), sizes.c_str());
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    LLAMA_STEP_MARK("dec.end");

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = !cparams.embeddings || graph_has_logits;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il] && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    if (cparams.embeddings_layer_inp_dev) {
        return extracted;
    }
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched_res(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (i0 < logits_dev_rows.size() && i1 < logits_dev_rows.size()) {
            std::swap(logits_dev_rows[i0], logits_dev_rows[i1]);
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_KIMI_K31 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_METIS16 ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_CLOVER_MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DEEPSEEK41 ||
        model.arch == LLM_ARCH_XING4_0 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

uint32_t llama_context::gf_res_key(uint32_t n_tokens) const {
    // batches of up to 8 tokens (decode, speculative verify) are keyed by their size too
    return (n_outputs > 0 ? 1u : 0u) | ((graph_per_size && n_tokens <= 8 ? n_tokens : 0u) << 1);
}

void llama_context::gf_res_invalidate() {
    gf_res_prev_active = nullptr;
    for (auto & e : sched_small) {
        e.ready = nullptr;
    }
}

llm_graph_result * llama_context::get_gf_res_prev(uint32_t n_tokens) {
    const uint32_t key = gf_res_key(n_tokens);
    for (auto & [k, res] : gf_res_prev) {
        if (k == key) {
            return res.get();
        }
    }
    gf_res_prev.emplace_back(key, llm_graph_result_ptr(new llm_graph_result(gf_res_reserve->get_max_nodes())));
    return gf_res_prev.back().second.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    sched_active = nullptr; // a reserve uses the shared scheduler

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & [key, res] : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_invalidate();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.model       =*/ &model,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched_res(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.layer_inp_dev =*/ layer_inp_dev,
        /*.layer_inp_src =*/ supports_embd_row() ? cparams.ctx_other->get_layer_inp_dev() : nullptr,
        /*.h_nextn_dev   =*/ cparams.embeddings_nextn_dev ? h_nextn_dev : nullptr,
        /*.h_nextn_src   =*/ supports_state_row() ? cparams.ctx_state_src->get_h_nextn_dev() : nullptr,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.h_nextn_row0  =*/ &h_nextn_row0,
    };
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    auto status = ggml_backend_sched_graph_compute_async(sched_res(), gf);
    gf_in_flight = true;
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[id] = true;
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    for (int64_t first = 0; first < n_expert; ) {
        if (!st.used[first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[last + 1]) {
            last++;
        }

        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size    = (last + 1 - next)*expert_size + padding;
        if (size > 0) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
        }

        first = last + 1;
    }

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return graph_get_cb(sched_res());
}

llm_graph_cb llama_context::graph_get_cb(ggml_backend_sched_t sched_graph) const {
    // sched_graph: the scheduler of the graph being built (a pipe slot has its own)
    return [this, sched_graph](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        // - a layer's attention mask built on the device (LLAMA_GPU_KQ_MASK) belongs on that layer's device, or the scheduler
        //   computes it once and copies it between devices, which is the transfer it exists to avoid; so does the staged
        //   input it is built from ("kq_stage"), or the scheduler uploads each part of the input separately
        //   (and so does the update of the k-pool tables kept on the devices, "kpool_stage", LLAMA_QSA_KPOOL_DEV, and the
        //   QSA pool mask built on the devices, "kpool_mask", LLAMA_QSA_POOL_MASK_DEV)
        const bool kq_dev = strcmp(name, "kq_mask") == 0 || strcmp(name, "kq_stage") == 0 || strcmp(name, "kpool_stage") == 0 ||
                            strcmp(name, "kpool_mask") == 0;
        // - a card hop's casts (LLAMA_HOP_CAST, build_hop_cast): the narrow cast on the card before the boundary (il is
        //   that card's last layer), the cast back on the card after it, so the scheduler copies the narrow tensor
        const bool hop = strcmp(name, "hop_narrow") == 0 || strcmp(name, "hop_wide") == 0;
        if ((ubatch.n_tokens < 32 || full_offload || kq_dev || hop) && il != -1) {
            if (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0 || kq_dev || hop) {
                // the scheduler's own backends: a chunk slot's are not the context's with LLAMA_PIPE_CHUNK_STREAM
                const auto & dev_layer = model.dev_layer(il);
                for (int i = 0; i < ggml_backend_sched_get_n_backends(sched_graph); i++) {
                    ggml_backend_t backend = ggml_backend_sched_get_backend(sched_graph, i);
                    if (ggml_backend_get_device(backend) == dev_layer) {
                        if (ggml_backend_supports_op(backend, cur)) {
                            ggml_backend_sched_set_tensor_backend(sched_graph, cur, backend);
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        if (winfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(winfos.size());
        offsets.reserve(winfos.size());
        sizes.reserve(winfos.size());
        data_offsets.reserve(winfos.size());

        for (const auto & winfo : winfos) {
            tensors.push_back(winfo.tensor);
            offsets.push_back(winfo.offset);
            sizes.push_back(winfo.size);
            data_offsets.push_back(winfo.data_offset);
        }

        std::vector<const ggml_tensor *> const_tensors(tensors.begin(), tensors.end());
        ggml_backend_tensor_get_n(const_tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), winfos.size());
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        if (rinfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(rinfos.size());
        offsets.reserve(rinfos.size());
        sizes.reserve(rinfos.size());
        data_offsets.reserve(rinfos.size());

        for (const auto & rinfo : rinfos) {
            tensors.push_back(rinfo.tensor);
            offsets.push_back(rinfo.offset);
            sizes.push_back(rinfo.size);
            data_offsets.push_back(rinfo.data_offset);
        }

        ggml_backend_tensor_set_n(tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), rinfos.size());
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    const uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            GGML_ASSERT(winfo.size % ggml_type_size(winfo.tensor->type) == 0);
            const int64_t n = winfo.size/ggml_type_size(winfo.tensor->type)*ggml_blck_size(winfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            std::vector<const ggml_tensor *> srcs(mbuf_cur.org.begin(), mbuf_cur.org.end());
            std::vector<ggml_tensor *> dsts(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
            ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.org.size());
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            GGML_ASSERT(rinfo.size % ggml_type_size(rinfo.tensor->type) == 0);
            const int64_t n = rinfo.size/ggml_type_size(rinfo.tensor->type)*ggml_blck_size(rinfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    std::vector<const ggml_tensor *> srcs(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
                    std::vector<ggml_tensor *> dsts(mbuf.org.begin(), mbuf.org.end());
                    ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.cpy.size());
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            std::vector<const ggml_tensor *> chunk_srcs;
            std::vector<ggml_tensor *> chunk_dsts;
            chunk_srcs.reserve(mbuf_cur.cpy.size() + mbuf.org.size());
            chunk_dsts.reserve(mbuf_cur.cpy.size() + mbuf.org.size());

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                GGML_ASSERT(n_copy % ggml_type_size(src_t->type) == 0);
                const int64_t n_el = n_copy / ggml_type_size(src_t->type) * ggml_blck_size(src_t->type);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                chunk_srcs.push_back(src_v);
                chunk_dsts.push_back(dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_backend_tensor_copy_n(chunk_srcs.data(), chunk_dsts.data(), chunk_srcs.size());

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

static ggml_backend_buffer_type_t llama_tensor_buft(const ggml_tensor * t) {
    return ggml_backend_buffer_get_type(t->view_src ? t->view_src->buffer : t->buffer);
}

// writes the rest of a state into the blob and copies its tensor data into a numbered store; commit() does the copy and
// reports whether the store's buffers could be allocated
class llama_io_write_store : public llama_io_write_i {
public:
    llama_io_write_store(uint8_t * p, size_t len, llama_state_store & store) : ptr(p), buf_size(len), store(store) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > 0) {
            winfos.push_back({tensor, offset, size, size_written});
        }
    }

    size_t n_bytes() override {
        return size_written;
    }

    // the copies into a store wait for the devices (llama_state_seq_get_data_store)
    bool commit() {
        ggml_context_ptr ctx_views;
        std::vector<const ggml_tensor *> srcs;
        std::vector<ggml_tensor *>       dsts;
        if (!prepare(ctx_views, srcs, dsts)) {
            return false;
        }

        ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), srcs.size());

        store.fences.clear();

        return true;
    }

    // allocates the store's buffers where needed, records the blob, and lists the copies: views of the state's rows
    // (in ctx_views) and the store's tensors; false if the buffers could not be allocated
    bool prepare(ggml_context_ptr & ctx_views, std::vector<const ggml_tensor *> & srcs, std::vector<ggml_tensor *> & dsts) {
        // a store keeps its buffers while the state's tensors keep their types, sizes and devices
        bool same = store.pieces.size() == winfos.size();
        for (size_t i = 0; same && i < winfos.size(); ++i) {
            const ggml_tensor * cpy = store.pieces[i].first;
            same = cpy->type == winfos[i].tensor->type && ggml_nbytes(cpy) == winfos[i].size &&
                   llama_tensor_buft(cpy) == llama_tensor_buft(winfos[i].tensor);
        }

        if (!same) {
            store = {};

            std::vector<ggml_backend_buffer_type_t> bufts; // of store.bufs
            for (const auto & w : winfos) {
                // the copies are 1d views of the rows: no block types
                if (ggml_blck_size(w.tensor->type) != 1 || w.size % ggml_element_size(w.tensor) != 0) {
                    LLAMA_LOG_WARN("%s: %s rows cannot be stored\n", __func__, ggml_type_name(w.tensor->type));
                    store = {};
                    return false;
                }

                auto * buft = llama_tensor_buft(w.tensor);

                const size_t i_buf = std::find(bufts.begin(), bufts.end(), buft) - bufts.begin();
                if (i_buf == bufts.size()) {
                    ggml_init_params params = {
                        /*.mem_size   =*/ winfos.size()*ggml_tensor_overhead(),
                        /*.mem_buffer =*/ NULL,
                        /*.no_alloc   =*/ true,
                    };
                    bufts.push_back(buft);
                    store.bufs.emplace_back(ggml_context_ptr(ggml_init(params)), nullptr);
                }

                ggml_tensor * cpy = ggml_new_tensor_1d(store.bufs[i_buf].first.get(), w.tensor->type, w.size/ggml_element_size(w.tensor));
                store.pieces.emplace_back(cpy, w.meta_off);
            }

            for (size_t i = 0; i < store.bufs.size(); ++i) {
                auto & [ctx, buf] = store.bufs[i];

                buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), bufts[i]));
                if (!buf) {
                    LLAMA_LOG_WARN("%s: failed to allocate a state store in '%s'\n", __func__, ggml_backend_buft_name(bufts[i]));
                    store = {};
                    return false;
                }
                LLAMA_LOG_INFO("%s: allocated a state store in '%s', %.3f MiB\n", __func__,
                        ggml_backend_buft_name(bufts[i]), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            }
        }

        ggml_init_params params = {
            /*.mem_size   =*/ (winfos.size() + 1)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx_views.reset(ggml_init(params));

        srcs.clear();
        dsts.clear();

        store.size_data = 0;
        for (size_t i = 0; i < winfos.size(); ++i) {
            const auto & w = winfos[i];

            ggml_tensor * src = ggml_view_1d(ctx_views.get(), w.tensor, w.size/ggml_element_size(w.tensor), w.offset);
            ggml_backend_view_init(src);

            srcs.push_back(src);
            dsts.push_back(store.pieces[i].first);

            store.pieces[i].second = w.meta_off;
            store.size_data += w.size;
        }

        store.size_meta = size_written;
        store.meta.assign(ptr - size_written, ptr);

        return true;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        size_t offset;
        size_t size;
        size_t meta_off;
    };
    std::vector<write_info> winfos;

    llama_state_store & store;
};

// reads the rest of a state from the blob, and its tensor data from a numbered store with commit()
class llama_io_read_store : public llama_io_read_i {
public:
    llama_io_read_store(const uint8_t * p, size_t len, const llama_state_store & store) : ptr(p), buf_size(len), store(store) {}

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > 0) {
            rinfos.push_back({tensor, offset, size});
        }
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

    // the restore enumerates the same bytes in the same order as the save, but may cut them into other row ranges: the
    // copies follow a byte cursor across both
    bool commit() {
        size_t total = 0;
        for (const auto & r : rinfos) {
            total += r.size;
        }
        if (total != store.size_data) {
            LLAMA_LOG_ERROR("%s: the state reads %zu bytes, the store holds %zu\n", __func__, total, store.size_data);
            return false;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ 2*(store.pieces.size() + rinfos.size() + 1)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_views(ggml_init(params));

        std::vector<const ggml_tensor *> srcs;
        std::vector<ggml_tensor *>       dsts;

        size_t i_src   = 0;
        size_t off_src = 0;
        for (const auto & r : rinfos) {
            for (size_t off_dst = 0; off_dst < r.size; ) {
                ggml_tensor * cpy = store.pieces[i_src].first;

                const size_t n = std::min(ggml_nbytes(cpy) - off_src, r.size - off_dst);
                if (cpy->type != r.tensor->type || n % ggml_element_size(cpy) != 0) {
                    LLAMA_LOG_ERROR("%s: the store's rows do not match the state's\n", __func__);
                    return false;
                }

                const int64_t ne = n/ggml_element_size(cpy);

                ggml_tensor * src = ggml_view_1d(ctx_views.get(), cpy,      ne, off_src);
                ggml_tensor * dst = ggml_view_1d(ctx_views.get(), r.tensor, ne, r.offset + off_dst);
                ggml_backend_view_init(src);
                ggml_backend_view_init(dst);

                srcs.push_back(src);
                dsts.push_back(dst);

                off_src += n;
                off_dst += n;
                if (off_src == ggml_nbytes(cpy)) {
                    ++i_src;
                    off_src = 0;
                }
            }
        }

        ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), srcs.size());

        return true;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        size_t offset;
        size_t size;
    };
    std::vector<read_info> rinfos;

    const llama_state_store & store;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

size_t llama_context::state_seq_get_data_store(llama_seq_id seq_id, int32_t store_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    flags |= LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

    llama_state_store * store = nullptr;
    {
        std::lock_guard<std::mutex> lock(state_stores_mtx);
        store = &state_stores[{seq_id, store_id}];
    }

    llama_io_write_store io(dst, size, *store);

    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        const size_t n = state_seq_write_data(io, seq_id, flags);

        if (!io.commit()) {
            return 0;
        }

        return n;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data_store(llama_seq_id seq_id, int32_t store_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    flags |= LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

    // the store belongs to the sequence that wrote src
    uint32_t     magic_read  = 0;
    llama_seq_id seq_id_read = -1;
    if (size >= sizeof(magic_read) + sizeof(seq_id_read)) {
        memcpy(&magic_read,  src,                      sizeof(magic_read));
        memcpy(&seq_id_read, src + sizeof(magic_read), sizeof(seq_id_read));
    }
    if (magic_read != io_magic) {
        LLAMA_LOG_ERROR("%s: wrong sequence state magic\n", __func__);
        return 0;
    }

    const llama_state_store * store = nullptr;
    {
        std::lock_guard<std::mutex> lock(state_stores_mtx);
        const auto it = state_stores.find({seq_id_read, store_id});
        if (it != state_stores.end() && it->second.size_meta == size && memcmp(it->second.meta.data(), src, size) == 0) {
            store = &it->second;
        }
    }
    if (store == nullptr) {
        LLAMA_LOG_ERROR("%s: store %d of seq %d does not hold this state\n", __func__, store_id, seq_id_read);
        return 0;
    }

    llama_io_read_store io(src, size, *store);

    try {
        io.read(&magic_read,  sizeof(magic_read));
        io.read(&seq_id_read, sizeof(seq_id_read));

        const size_t n = state_seq_read_data(io, seq_id, flags);

        if (!io.commit()) {
            // the cells are restored without their data: drop them
            if (memory) {
                memory->seq_rm(seq_id, -1, -1);
            }
            return 0;
        }

        return n;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

size_t llama_context::state_seq_store_get_size_host(llama_seq_id seq_id, int32_t store_id) {
    std::lock_guard<std::mutex> lock(state_stores_mtx);

    const auto it = state_stores.find({seq_id, store_id});
    if (it == state_stores.end() || it->second.size_meta == 0) {
        return 0;
    }

    return it->second.size_meta + it->second.size_data;
}

size_t llama_context::state_seq_store_get_data_host(llama_seq_id seq_id, int32_t store_id, const uint8_t * src, size_t src_size, uint8_t * dst, size_t dst_size) {
    std::vector<std::pair<ggml_tensor *, size_t>> pieces;
    std::vector<ggml_backend_event_t> fences;

    size_t size_meta = 0;
    size_t size_data = 0;
    {
        std::lock_guard<std::mutex> lock(state_stores_mtx);

        const auto it = state_stores.find({seq_id, store_id});
        if (it != state_stores.end() && it->second.size_meta == src_size && memcmp(it->second.meta.data(), src, src_size) == 0) {
            pieces    = it->second.pieces;
            size_meta = it->second.size_meta;
            size_data = it->second.size_data;
            for (const auto & ev : it->second.fences) {
                fences.push_back(ev.get());
            }
        }
    }

    if (size_meta == 0 || dst_size < size_meta + size_data) {
        LLAMA_LOG_ERROR("%s: store %d of seq %d does not hold this state, or dst is too small\n", __func__, store_id, seq_id);
        return 0;
    }

    // a fill inside a decode (C217): its copies first
    for (auto * ev : fences) {
        ggml_backend_event_synchronize(ev);
    }

    // the blob's bytes, with each copy's bytes put back where the state wrote them
    size_t    pos = 0;
    uint8_t * out = dst;
    for (const auto & [cpy, off] : pieces) {
        memcpy(out, src + pos, off - pos);
        out += off - pos;
        pos  = off;

        ggml_backend_tensor_get(cpy, out, 0, ggml_nbytes(cpy));
        out += ggml_nbytes(cpy);
    }
    memcpy(out, src + pos, size_meta - pos);
    out += size_meta - pos;

    return out - dst;
}

void llama_context::state_seq_store_free(llama_seq_id seq_id, int32_t store_id) {
    std::lock_guard<std::mutex> lock(state_stores_mtx);

    state_stores.erase({seq_id, store_id});
}

void llama_context::set_ubatch_cuts(const uint32_t * cuts, size_t n) {
    ubatch_cuts.assign(cuts, cuts + n);
}

void llama_context::state_seq_store_at(llama_seq_id seq_id, int32_t store, uint32_t i_end, llama_state_seq_flags flags) {
    store_at_req r;
    r.seq_id = seq_id;
    r.store  = store;
    r.i_end  = i_end;
    r.flags  = flags | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

    store_at_next.push_back(std::move(r));
}

size_t llama_context::state_seq_store_at_get(llama_seq_id seq_id, int32_t store, uint8_t * dst, size_t size, llama_pos * pos_min, llama_pos * pos_max) const {
    for (const auto & r : store_at_last) {
        if (!r.done || r.seq_id != seq_id || r.store != store) {
            continue;
        }
        if (dst != nullptr) {
            if (size < r.blob.size()) {
                return 0;
            }
            memcpy(dst, r.blob.data(), r.blob.size());
        }
        if (pos_min) {
            *pos_min = r.pos_min;
        }
        if (pos_max) {
            *pos_max = r.pos_max;
        }
        return r.blob.size();
    }

    return 0;
}

bool llama_context::state_seq_fill_store_at(store_at_req & r) {
    r.blob.resize(state_seq_get_size(r.seq_id, r.flags));
    if (r.blob.empty()) {
        return false;
    }

    llama_state_store * store = nullptr;
    {
        std::lock_guard<std::mutex> lock(state_stores_mtx);
        store = &state_stores[{r.seq_id, r.store}];
    }

    llama_io_write_store io(r.blob.data(), r.blob.size(), *store);

    ggml_context_ptr ctx_views;
    std::vector<const ggml_tensor *> srcs;
    std::vector<ggml_tensor *>       dsts;

    try {
        io.write(&io_magic,   sizeof(io_magic));
        io.write(&r.seq_id,   sizeof(r.seq_id));

        if (state_seq_write_data(io, r.seq_id, r.flags) != r.blob.size() || !io.prepare(ctx_views, srcs, dsts)) {
            r.blob.clear();
            return false;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        r.blob.clear();
        return false;
    }

    // each row copied on the stream of the backend whose device holds it: behind this micro-batch's writes to it, ahead
    // of the next one's
    std::vector<ggml_backend_t> used;
    for (size_t i = 0; i < srcs.size(); ++i) {
        ggml_backend_t backend = backend_from_dev(ggml_backend_buft_get_device(llama_tensor_buft(srcs[i])));
        if (backend == nullptr) {
            synchronize();
            ggml_backend_tensor_copy(srcs[i], dsts[i]);
            continue;
        }
        ggml_backend_tensor_copy_async(backend, backend, srcs[i], dsts[i]);
        if (std::find(used.begin(), used.end(), backend) == used.end()) {
            used.push_back(backend);
        }
    }

    store->fences.clear();
    for (ggml_backend_t backend : used) {
        ggml_backend_event_t ev = ggml_backend_event_new(ggml_backend_get_device(backend));
        if (ev != nullptr) {
            ggml_backend_event_record(ev, backend);
            store->fences.emplace_back(ev);
        }
    }

    r.pos_min = memory->seq_pos_min(r.seq_id);
    r.pos_max = memory->seq_pos_max(r.seq_id);
    r.done    = true;

    return true;
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count, llama_state_seq_flags flags) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, flags);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.no_alloc()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
            // the per-size schedulers (LLAMA_SCHED_PER_SIZE=1) allocate compute buffers of their own
            for (const auto & e : sched_small) {
                ret[buft].compute += ggml_backend_sched_get_buffer_size(e.sched.get(), backend);
            }
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    sched_active = nullptr; // the optimizer graphs use the shared scheduler
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_invalidate();
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.moe_cache_cpu_share         =*/ 0.0f,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4 || model->arch == LLM_ARCH_DEEPSEEK41) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

uint64_t llama_memory_recurrent_n_swaps(const llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
    auto * mem_recr   = mem_hybrid ? mem_hybrid->get_mem_recr() : dynamic_cast<llama_memory_recurrent *>(mem);
    return mem_recr ? mem_recr->n_swaps : 0;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_screen_head(llama_context * ctx, bool enable) {
    ctx->set_screen_head(enable);
}

bool llama_get_screen_head(const llama_context * ctx) {
    return ctx->get_screen_head();
}

bool llama_set_ubatch_taper(llama_context * ctx, const char * spec) {
    return ctx->set_ubatch_taper(spec);
}

const char * llama_get_ubatch_taper(const llama_context * ctx) {
    return ctx->get_ubatch_taper();
}

void llama_set_ubatch_taper_next(llama_context * ctx, int32_t head, int32_t tail) {
    ctx->set_ubatch_taper_next(head, tail);
}

bool llama_head_shortlist_init(llama_context * ctx, uint32_t k_low, uint32_t m_cap) {
    return ctx->head_shortlist_init(k_low, m_cap);
}

uint32_t llama_head_shortlist_add(llama_context * ctx, const llama_token * ids, size_t n_ids) {
    return ctx->head_shortlist_add(ids, n_ids);
}

bool llama_head_shortlist_contains(const llama_context * ctx, llama_token id) {
    return ctx->head_shortlist_contains(id);
}

uint32_t llama_head_shortlist_size(const llama_context * ctx) {
    return ctx->head_shortlist_size();
}

void llama_set_head_shortlist(llama_context * ctx, bool value) {
    ctx->set_head_shortlist(value);
}

bool llama_embd_cache_init(llama_context * ctx, uint32_t n_rows, uint32_t max_tokens) {
    return ctx->embd_cache_init(n_rows, max_tokens);
}

bool llama_embd_cache_stats(const llama_context * ctx, uint64_t * n_hit, uint64_t * n_miss, uint64_t * n_plain) {
    return ctx->embd_cache_stats(n_hit, n_miss, n_plain);
}

bool llama_nextn_carry_init(llama_context * ctx) {
    return ctx->nextn_carry_init();
}

bool llama_kpool_dev_stats(const llama_context * ctx, uint64_t * n_ubatch, uint64_t * n_full, uint64_t * n_rows, uint64_t * n_checked) {
    const auto * mem_idx = dynamic_cast<const llama_memory_hybrid_idx *>(llama_get_memory(ctx));
    return mem_idx != nullptr && mem_idx->kpool_dev_stats(n_ubatch, n_full, n_rows, n_checked);
}

bool llama_pool_mask_dev_stats(const llama_context * ctx, uint64_t * n_ubatch, uint64_t * n_checked, uint64_t * n_masks) {
    const auto * mem_idx = dynamic_cast<const llama_memory_hybrid_idx *>(llama_get_memory(ctx));
    if (mem_idx == nullptr || !mem_idx->pool_mask_dev_on()) {
        return false;
    }
    const auto & st = mem_idx->pool_mask_stats();
    if (n_ubatch)  { *n_ubatch  = st.n_ubatch;  }
    if (n_checked) { *n_checked = st.n_checked; }
    if (n_masks)   { *n_masks   = st.n_masks;   }
    return true;
}

void llama_set_nextn_carry(llama_context * ctx, bool in, bool out) {
    ctx->set_nextn_carry(in, out);
}

void llama_step_trace_role(llama_context * ctx, char role) {
    ctx->set_step_trace_role(role);
}

bool llama_head_bound_init(llama_context * ctx) {
    return ctx->head_bound_init();
}

void llama_set_head_bound(llama_context * ctx, bool value) {
    ctx->set_head_bound(value);
}

float llama_head_bound_ith(llama_context * ctx, int32_t i) {
    return ctx->head_bound_ith(i);
}

bool llama_head_bound_full(llama_context * ctx, int32_t i) {
    return ctx->head_bound_full(i);
}

void llama_head_bound_stats(const llama_context * ctx, uint64_t * n_rows, uint64_t * n_full) {
    ctx->head_bound_stats(n_rows, n_full);
}

void llama_set_moe_tail(llama_context * ctx, bool on, float p, float w, bool renorm, bool hist) {
    ctx->set_moe_tail(on, p, w, renorm, hist);
}

uint64_t llama_moe_tail_counts(const llama_context * ctx, uint64_t * counts, int32_t n_counts) {
    return ctx->moe_tail_counts(counts, n_counts);
}

void llama_set_moe_tail_mode(llama_context * ctx, bool graph, int32_t skip, bool repeat) {
    ctx->set_moe_tail_mode(graph, skip, repeat);
}

void llama_set_moe_route_once(llama_context * ctx, uint32_t k, uint32_t max, bool hist, bool dry) {
    ctx->set_moe_route_once(k, max, hist, dry);
}

uint64_t llama_moe_route_once_counts(const llama_context * ctx, uint64_t * read, uint64_t * indep, int32_t n_counts,
        uint64_t * n_outside) {
    return ctx->moe_route_once_counts(read, indep, n_counts, n_outside);
}

uint32_t llama_model_screen_head_m(const llama_model * model) {
    return model->output_screen ? model->hparams.output_screen_top_m : 0;
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits_ith_fetch(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_ith_fetch(i);
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

bool llama_set_embeddings_nextn_dev(llama_context * ctx, bool value) {
    return ctx->set_embeddings_nextn_dev(value);
}

bool llama_set_embeddings_nextn_dev_halves(llama_context * ctx, int32_t n) {
    return ctx->set_h_nextn_halves(n);
}

bool llama_hold_embeddings_nextn(llama_context * ctx) {
    return ctx->hold_h_nextn();
}

int64_t llama_get_embeddings_nextn_row0(const llama_context * ctx) {
    return ctx->get_h_nextn_row0();
}

bool llama_get_embeddings_nextn_rows(llama_context * ctx, int64_t row0, int64_t n_rows, float * out) {
    return ctx->get_h_nextn_rows(row0, n_rows, out);
}

bool llama_set_embeddings_nextn_rows(llama_context * ctx, int64_t row0, int64_t n_rows, const float * data) {
    return ctx->set_h_nextn_rows(row0, n_rows, data);
}

bool llama_set_embd_state_src(llama_context * ctx, llama_context * src) {
    return ctx->set_state_src(src);
}

void llama_set_embeddings_layer_inp_dev(llama_context * ctx, bool value) {
    ctx->set_embeddings_layer_inp_dev(value);
}

bool llama_supports_embd_row(const llama_context * ctx) {
    return ctx->supports_embd_row();
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_get_data_store(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, int32_t store, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data_store(seq_id, store, dst, size, flags);
}

size_t llama_state_seq_set_data_store(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id, int32_t store, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data_store(dest_seq_id, store, src, size, flags);
}

size_t llama_state_seq_store_get_size_host(llama_context * ctx, llama_seq_id seq_id, int32_t store) {
    return ctx->state_seq_store_get_size_host(seq_id, store);
}

// no synchronize: it may run on another thread, and the store's copies were complete when it was filled
size_t llama_state_seq_store_get_data_host(llama_context * ctx, const uint8_t * src, size_t src_size, uint8_t * dst, size_t dst_size, llama_seq_id seq_id, int32_t store) {
    return ctx->state_seq_store_get_data_host(seq_id, store, src, src_size, dst, dst_size);
}

void llama_state_seq_store_free(llama_context * ctx, llama_seq_id seq_id, int32_t store) {
    ctx->state_seq_store_free(seq_id, store);
}

void llama_set_ubatch_cuts(llama_context * ctx, const uint32_t * cuts, size_t n) {
    ctx->set_ubatch_cuts(cuts, n);
}

void llama_state_seq_store_at(llama_context * ctx, llama_seq_id seq_id, int32_t store, uint32_t i_end, llama_state_seq_flags flags) {
    ctx->state_seq_store_at(seq_id, store, i_end, flags);
}

size_t llama_state_seq_store_at_get(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, int32_t store, llama_pos * pos_min, llama_pos * pos_max) {
    return ctx->state_seq_store_at_get(seq_id, store, dst, size, pos_min, pos_max);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_save_file_ext(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count, llama_state_seq_flags flags) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_pipe_init(llama_context * ctx, int32_t n_slots) {
    return ctx->pipe_init(n_slots, 0, 0);
}

int32_t llama_pipe_init_ext(llama_context * ctx, int32_t n_slots, int32_t n_chunk_slots, int32_t n_chunk_tokens) {
    return ctx->pipe_init(n_slots, n_chunk_slots, n_chunk_tokens);
}

int32_t llama_pipe_n_slots(const llama_context * ctx) {
    return ctx->pipe_n_slots();
}

int32_t llama_pipe_submit(llama_context * ctx, int32_t slot, llama_token token, llama_pos pos, llama_seq_id seq_id) {
    return ctx->pipe_submit(slot, &token, 1, pos, seq_id);
}

int32_t llama_pipe_submit_tokens(llama_context * ctx, int32_t slot, const llama_token * tokens, int32_t n_tokens, llama_pos pos, llama_seq_id seq_id) {
    return ctx->pipe_submit(slot, tokens, n_tokens, pos, seq_id);
}

int32_t llama_pipe_submit_fed(llama_context * ctx, int32_t slot, int32_t slot_src, llama_pos pos, llama_seq_id seq_id) {
    return ctx->pipe_submit_fed(slot, slot_src, pos, seq_id);
}

float * llama_pipe_get_logits(llama_context * ctx, int32_t slot) {
    return ctx->pipe_get_logits(slot);
}

llama_token llama_pipe_get_sampled(llama_context * ctx, int32_t slot) {
    return ctx->pipe_get_sampled(slot);
}

bool llama_pipe_ready(llama_context * ctx, int32_t slot) {
    return ctx->pipe_ready(slot);
}

void llama_pipe_drain(llama_context * ctx) {
    ctx->pipe_drain();
}

llama_pipe_draft_params llama_pipe_draft_default_params(void) {
    llama_pipe_draft_params p = {
        /*.path   =*/ nullptr,
        /*.device =*/ -1,
        /*.n_ctx  =*/ 4096,
        /*.n_rows =*/ 8,
        /*.alpha  =*/ 0.0f,
        /*.top_k  =*/ 16,
        /*.stair  =*/ nullptr,
        /*.kind   =*/ nullptr,
    };
    return p;
}

int32_t llama_pipe_draft_init(llama_context * ctx, llama_pipe_draft_params params) {
    return ctx->pipe_draft_init(params);
}

bool llama_pipe_draft_ready(const llama_context * ctx) {
    return ctx->get_pipe_drafter() != nullptr;
}

void llama_pipe_draft_reset(llama_context * ctx, llama_seq_id seq_id, llama_pos n_keep) {
    ctx->pipe_draft_reset(seq_id, n_keep);
}

void llama_pipe_draft_commit(llama_context * ctx, llama_token token, llama_pos pos, int32_t slot) {
    if (auto * d = ctx->get_pipe_drafter()) {
        d->commit(token, pos, slot);
    }
}

int32_t llama_pipe_draft_launch(llama_context * ctx, const llama_pipe_draft_row * rows, int32_t n_rows) {
    auto * d = ctx->get_pipe_drafter();
    return d ? d->launch(rows, n_rows) : -1;
}

llama_token llama_pipe_draft_get(llama_context * ctx, float u, float * p_top) {
    auto * d = ctx->get_pipe_drafter();
    return d ? d->get(u, p_top) : -1;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
