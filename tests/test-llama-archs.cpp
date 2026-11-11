#include "common.h"
#include "speculative.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a pool of normal draws, generated once and reused for every model
static std::vector<float> g_normal_pool;
static const size_t NORMAL_POOL_SIZE = 16 * 1024 * 1024;

// fill the pool once; each tensor copies a slice of it
static void init_normal_pool(const size_t seed, const float stdev) {
    g_normal_pool.resize(NORMAL_POOL_SIZE);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);
    for (float & v : g_normal_pool) {
        v = dist(gen);
    }
}

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params &) {
    const size_t n = dst.size();
    const size_t pool_size = g_normal_pool.size();
    const size_t start = std::hash<std::string>{}(key) % pool_size;

    // copy a cyclic slice of the pool, so a tensor larger than the pool still fills
    size_t i = 0;
    size_t pos = start;
    while (i < n) {
        const size_t chunk = std::min(pool_size - pos, n - i);
        std::memcpy(dst.data() + i, g_normal_pool.data() + pos, chunk * sizeof(float));
        i += chunk;
        pos = 0;
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// RoPE factors divide the frequencies, a value close to 0 gives angles too large for F32 sin/cos to stay accurate
static void fill_rope_factors(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = 1.0f + fabsf(v);
    }
}

// FFN act scales divide the activation, a value close to 0 overflows the F16 staging of the Metal mul-mat kernel
static void fill_ffn_act(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = copysignf(1.0f + fabsf(v), v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "ssm_a",        fill_ssm_a        },
    { "_resid_",      fill_resid_gate   },
    { "exp_probs_b",  fill_exp_probs_b  },
    { "ffn.act",      fill_ffn_act      },
    { "rope_freqs",   fill_rope_factors },
    { "rope_factors", fill_rope_factors },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  --mtp                    With -o: save qwen4exp's MoE model with one MTP block as <dir>/qwen4exp-moe-mtp.gguf\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --kv-shift               Check K-shift against a decode at the shifted positions (CPU only)\n");
    LOG("  --mtp-kv-only            Check that LLAMA_MTP_KV_ONLY leaves the MTP draft's caches and drafts as they are (CPU only)\n");
    LOG("  --mtp-catchup-overlap    Check that LLAMA_MTP_CATCHUP_OVERLAP leaves the MTP draft's caches and drafts as they are (CPU only)\n");
    LOG("  --ngram-prefetch         Check that LLAMA_NGRAM_PREFETCH changes only when the n-gram rows are read (CPU only)\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

// writes arch-resolved GGUF keys into the fixture context, the test-side equivalent of llama_model_saver
struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("fatal error");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

// LLAMA_ARCHS_VOCAB=<gguf>: the tokenizer and chat template of a real model (its tokenizer.* keys, read from the file's
// header) instead of the dummy one, for tests that need real text: grammars, chat templates, tool calls
static gguf_context_ptr archs_vocab() {
    const char * path = getenv("LLAMA_ARCHS_VOCAB");
    if (path == nullptr || path[0] == 0) {
        return gguf_context_ptr();
    }
    gguf_init_params params = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context_ptr vctx(gguf_init_from_file(path, params));
    if (!vctx || gguf_find_key(vctx.get(), "tokenizer.ggml.tokens") < 0) {
        throw std::runtime_error(std::string("LLAMA_ARCHS_VOCAB: no tokenizer in ") + path);
    }
    return vctx;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const uint32_t n_layer_min = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(arch, ret.get());
    // LLAMA_ARCHS_N_CTX=<n>: a longer training context, for prompts such as a chat template's tool list
    const uint32_t n_ctx = getenv("LLAMA_ARCHS_N_CTX") ? (uint32_t) atoi(getenv("LLAMA_ARCHS_N_CTX")) : 256;
    const gguf_context_ptr vocab_ctx = archs_vocab();

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4) {
        // TODO: expand coverage to both GQA and MQA (see https://github.com/ggml-org/llama.cpp/pull/31401#pullrequestreview-5710165130)
        n_head = 4;
        n_layer = 10; // last 5 layers reuse KV, both SWA and full attention
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_LFM3) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, Engram, ratio-2 and ratio-1 sources, top-k reuse and a candidate-masked index layer
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_RICURSIVE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_LIMITE) {
        n_head  = 4;
        n_layer = 4; // dense connections on the last two layers
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_XING4_0) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_MISTRAL_LARGE4) {
            n_layer = 4; // cover NoPE, shared indexer and KV sharing layers
        }
    } else if (arch == LLM_ARCH_GLM5_NEXT) {
        // several heads so that -sm tensor can split them and the MLA flash attention has GQA > 1
        n_embd = 128;
        n_head = 16;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_METIS16) {
        n_layer = 4; // 2 passes over 2 physical layers, one attention and one Mamba-2
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }
    if (vocab_ctx) {
        n_vocab = (uint32_t) gguf_get_arr_n(vocab_ctx.get(), gguf_find_key(vocab_ctx.get(), "tokenizer.ggml.tokens"));
    }

    // LLAMA_TEST_ARCHS_N_VOCAB=<n>: a larger vocabulary, for the tests of shortlisted output heads
    if (const char * e = getenv("LLAMA_TEST_ARCHS_N_VOCAB")) {
        n_vocab = (uint32_t) atoi(e);
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE || arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_RICURSIVE) {
        n_head_kv = 2; // GQA coverage
    }
    if (arch == LLM_ARCH_BEAM) {
        n_head    = 4;
        n_head_kv = 2;
        n_layer   = 4;
    }
    n_layer = std::max(n_layer, n_layer_min);
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_LFM3 ||
            arch == LLM_ARCH_KIMI_LINEAR || arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_KIMI_K31 ||
            arch == LLM_ARCH_GLM5_NEXT
            ) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
            // the MLA cache holds one latent head
            for (uint32_t & n : n_head_per_layer) {
                n = std::min(n, uint32_t(1));
            }
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_METIS16) {
        // the layer types repeat in every pass
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il % 2 == 0 ? n_head_kv : 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_XING4_0) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    if (arch == LLM_ARCH_SENSENOVA_U1) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SPATIAL, 10000.0f);
    }

    if (arch == LLM_ARCH_XING4_0) {
        // Manifold-Constrained Hyper-Connections (MHC) multi-residual-stream blocks
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(20));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1e-6f);
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(5));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back((il + 1) % 5 != 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_BEAM || arch == LLM_ARCH_RICURSIVE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN, pattern);
        }
    } else if (arch == LLM_ARCH_LFM3) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>({0, 0, 1}));
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, false);
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    // LLAMA_ARCHS_INDEXER_TOP_K=<n>: a realistic top-k, for comparing two CPU runs whose selection must follow the
    // indexer's scores (LLAMA_ARCHS_LOGITS, e.g. LLAMA_QSA_INDEXER_ROWS=0 and 1)
    const char * env_top_k = getenv("LLAMA_ARCHS_INDEXER_TOP_K");
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        env_top_k ? (uint32_t) atoi(env_top_k) : uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_MISTRAL_LARGE4) {
        // layer 3 reads the KV cache of layer 2, both are NoPE
        ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN,     std::vector<uint32_t>({1, 1, 0, 0}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES,    std::vector<uint32_t>({1, 0, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_MAP,    std::vector<uint32_t>({0, 1, 2, 2}));
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,    30.0f);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 2, 2, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,            std::vector<uint32_t>({2, 4}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,       std::vector<uint32_t>({2, 4, 5}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({1, 3}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(3));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   uint32_t(64));
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({ 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                               67716810739261ull, 51510806800915ull, 30921347202721ull }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
        ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,             token_map);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    if (arch == LLM_ARCH_RICURSIVE) {
        std::vector<float> swiglu_clamp;
        swiglu_clamp.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            // the random weight scales keep the FFN activations small, a small limit still clamps so the roundtrip checks the per-layer values
            swiglu_clamp.push_back(il % 3 == 1 ? 0.0f : 1e-3f);
        }
        ms.add_kv(LLM_KV_EMBEDDING_SCALE, 16.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP,    swiglu_clamp);
    }

    if (arch == LLM_ARCH_METIS16) {
        ms.add_kv(LLM_KV_METIS_PASS_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,         uint32_t(4));
        ms.add_kv(LLM_KV_METIS_MEMORY_LENGTH,            uint32_t(64));
        ms.add_kv(LLM_KV_METIS_ROUTE_LENGTH,             uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_PASS_LORA_RANK,       uint32_t(8));
        ms.add_kv(LLM_KV_METIS_CONTINUATION_LOGIT_LIMIT, 4.0f);
        ms.add_kv(LLM_KV_SSM_NORM_EPS,                   1e-5f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,                n_embd/2);
    }

    if (arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_ATTENTION_SCALE,             0.1f);
        ms.add_kv(LLM_KV_ATTENTION_XSA_EPS,           1e-4f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SIGMOID_CAPPING, std::vector<float>({23.0f, 5.0f, 7.5f}));
        ms.add_kv(LLM_KV_MUDD_FEED_FORWARD_LENGTH,    uint32_t(8));
        ms.add_kv(LLM_KV_MUDD_TAP_COUNT,              uint32_t(2));
        ms.add_kv(LLM_KV_MUDD_TAP_INDICES,            std::vector<int32_t>({-1, -1, -1, -1, 0, 2, 1, 3}));
    }

    if (vocab_ctx) {
        // the real model's tokenizer.* keys: copy them all, then drop the others
        gguf_context_ptr tok(gguf_init_empty());
        gguf_set_kv(tok.get(), vocab_ctx.get());
        for (int64_t i = gguf_get_n_kv(tok.get()) - 1; i >= 0; --i) {
            const std::string key = gguf_get_key(tok.get(), i);
            if (key.rfind("tokenizer.", 0) != 0) {
                gguf_remove_key(tok.get(), key.c_str());
            }
        }
        gguf_set_kv(ret.get(), tok.get());
    } else
    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        uint32_t n_expert      = host_experts_test(arch) ? 64 : arch == LLM_ARCH_CLOVER_MOE ? 4 : 2;
        uint32_t n_expert_used = 2;
        // LLAMA_TEST_ARCHS_N_EXPERT=<n>,<used>: more routed experts, for the tests of per-token expert cuts
        if (const char * e = getenv("LLAMA_TEST_ARCHS_N_EXPERT")) {
            char * end = nullptr;
            n_expert = (uint32_t) strtoul(e, &end, 10);
            if (end && *end == ',') {
                n_expert_used = (uint32_t) strtoul(end + 1, nullptr, 10);
            }
        }
        ms.add_kv(LLM_KV_EXPERT_COUNT,               n_expert);
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          n_expert_used);
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            case LLM_ARCH_MELLUM:
                gating_func = 1;
                break;
            case LLM_ARCH_DEEPSEEK4:
            case LLM_ARCH_DEEPSEEK41:
                gating_func = 4;
                break;
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        }
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }

    if (arch == LLM_ARCH_BEAM) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_CLOVER_MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        std::vector<float> clamp_exp;
        std::vector<float> clamp_shexp;
        for (uint32_t il = 0; il < n_layer; il++) {
            clamp_exp  .push_back(il % 2 ? 4.0f : 0.0f);
            clamp_shexp.push_back(il % 2 ? 5.0f : 0.0f);
        }
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   clamp_exp);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, clamp_shexp);
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-llama-archs-" + std::to_string(++id) + ".gguf")).string();
}

static llama_model_ptr get_model(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const bool load_mtp = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;
    // llama_model_init_from_user cannot skip the nextn layers of a fixture that has them
    model_params.load_mtp = load_mtp;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    return model;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0) {
    llama_model_ptr model = get_model(gguf_ctx, file, seed, stdev, devs, split_mode, tensor_buft_overrides);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    // LLAMA_ARCHS_N_RS_SEQ=<n>: recurrent snapshots per sequence, so the rollback slots' writes run too
    if (const char * env = getenv("LLAMA_ARCHS_N_RS_SEQ")) {
        ctx_params.n_rs_seq = (uint32_t) atoi(env);
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

// the tokens decoded from an empty memory in steps of 1, 2, ..., 8 tokens: the sizes of decode steps and k-token verifies
static std::vector<float> get_logits_steps(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens = tokens.size();

    llama_memory_clear(llama_get_memory(lctx), true);

    common_batch batch(lctx);
    std::vector<float> ret;
    ret.reserve((size_t) n_tokens*n_vocab);
    uint32_t pos = 0;
    for (uint32_t step = 1; pos < n_tokens; step = step % 8 + 1) {
        const uint32_t n = std::min(step, n_tokens - pos);
        batch.clear();
        for (uint32_t i = 0; i < n; i++) {
            batch.add(tokens[pos + i], pos + i, 0, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            throw std::runtime_error("failed to decode a step");
        }
        for (uint32_t i = 0; i < n; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
        pos += n;
    }
    return ret;
}

// LLAMA_ARCHS_LOGITS=<prefix>: write an architecture's CPU logits (one batch, then the steps of get_logits_steps) to
// <prefix><name>.f32 (name: logits_dump_name), or compare with that file when it exists, so two builds or two settings
// (LLAMA_GDN_FUSE=0 and 1) can be compared; false when they differ by more than float rounding
static bool logits_dump_or_compare(const char * arch_name, const std::vector<float> & logits) {
    const char * prefix = getenv("LLAMA_ARCHS_LOGITS");
    if (prefix == nullptr) {
        return true;
    }
    const std::string path = std::string(prefix) + arch_name + ".f32";

    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        f = fopen(path.c_str(), "wb");
        if (f == nullptr || fwrite(logits.data(), sizeof(float), logits.size(), f) != logits.size()) {
            LOG_ERR("%s: cannot write %s\n", __func__, path.c_str());
            if (f != nullptr) {
                fclose(f);
            }
            return false;
        }
        fclose(f);
        LOG_INF("%s: wrote %zu logits to %s\n", __func__, logits.size(), path.c_str());
        return true;
    }

    std::vector<float> ref(logits.size());
    const bool same_size = fread(ref.data(), sizeof(float), ref.size(), f) == ref.size() && fgetc(f) == EOF;
    fclose(f);
    if (!same_size) {
        LOG_ERR("%s: %s holds a different number of logits\n", __func__, path.c_str());
        return false;
    }

    double max_abs = 0.0;
    size_t n_diff  = 0;
    for (size_t i = 0; i < logits.size(); i++) {
        max_abs = std::max(max_abs, (double) std::fabs(logits[i] - ref[i]));
        n_diff += logits[i] != ref[i];
    }
    const double e = nmse(ref, logits);
    LOG("%s: logits against %s: nmse %.3e, max abs diff %.3e, %zu of %zu differ\n",
        arch_name, path.c_str(), e, max_abs, n_diff, logits.size());
    return e <= 1e-8;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_FLINT:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_CLOVER_MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_XING4_0:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_KIMI_K31:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MISTRAL_LARGE4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_KOLIBRI1:
        case LLM_ARCH_METIS16:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

// the LLAMA_ARCHS_LOGITS file name of an architecture's model: an architecture with a Dense and a MoE model runs two
// different models, so the MoE one gets "-moe" (sharing one file, it compared against the Dense one's logits and
// failed); an architecture with one model keeps its name, so files written by earlier builds still compare
static std::string logits_dump_name(const llm_arch arch, const bool moe) {
    return std::string(llama_arch_name(arch)) + (moe && !moe_mandatory(arch) ? "-moe" : "");
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

// --mtp: the model fixture with one nextn (MTP) block after the trunk layers, so that llama-server's draft-mtp runs on
// a saved model (C405's CPU check of LLAMA_STEP_TRACE)
static gguf_context_ptr get_gguf_ctx_nextn(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret = get_gguf_ctx(arch, moe);
    const std::string key_block_count = llama_arch_kv_name(arch, LLM_KV_BLOCK_COUNT);
    const int64_t kid = gguf_find_key(ret.get(), key_block_count.c_str());
    GGML_ASSERT(kid >= 0);
    const uint32_t n_layer_all = gguf_get_val_u32(ret.get(), kid) + 1;
    gguf_set_val_u32(ret.get(), key_block_count.c_str(), n_layer_all);
    gguf_set_val_u32(ret.get(), llama_arch_kv_name(arch, LLM_KV_NEXTN_PREDICT_LAYERS).c_str(), 1);
    if (arch == LLM_ARCH_QWEN4EXP) {
        // the loader requires one compress ratio per layer, the MTP block's included
        const std::vector<uint32_t> ratios(n_layer_all, 4);
        gguf_set_arr_data(ret.get(), llama_arch_kv_name(arch, LLM_KV_ATTENTION_COMPRESS_RATIOS).c_str(), GGUF_TYPE_UINT32,
                ratios.data(), ratios.size());
    }
    return ret;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir,
        const bool mtp = false) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            if (mtp && (arch != LLM_ARCH_QWEN4EXP || !moe)) {
                continue;
            }
            gguf_context_ptr gguf_ctx = mtp ? get_gguf_ctx_nextn(arch, moe) : get_gguf_ctx(arch, moe);
            llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, nullptr, mtp);
            const std::string path = dir + "/" + llama_arch_name(arch) + (moe ? "-moe" : "-dense") + (mtp ? "-mtp.gguf" : ".gguf");
            LOG_INF("%s: '%s %s' -> %s ...", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model.get(), path.c_str());
            const std::uintmax_t size = std::filesystem::file_size(path);
            LOG_CNT(" (%.2f MB)\n", (double) size / (1024.0 * 1024.0));
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|\n");
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llama_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                        if (!encode && getenv("LLAMA_ARCHS_LOGITS") != nullptr) {
                            std::vector<float> logits_all = logits_cpu;
                            const std::vector<float> logits_steps =
                                get_logits_steps(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens);
                            logits_all.insert(logits_all.end(), logits_steps.begin(), logits_steps.end());
                            LOG("%s: one batch against steps of 1-8 tokens: nmse %.3e\n",
                                llama_arch_name(arch), nmse(logits_cpu, logits_steps));
                            if (!logits_dump_or_compare(logits_dump_name(arch, moe).c_str(), logits_all)) {
                                test_ok = false;
                            }
                        }
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llama_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llama_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    if (llama_arch_supports_save(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                        //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_dev.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_dev[i]) {
                                    test_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

struct kv_shift_move {
    llama_pos p0;
    llama_pos p1;
    llama_pos delta;
};

static void decode_range(llama_context * lctx, const llama_token * tokens, uint32_t n_tokens, llama_pos pos0, llama_seq_id seq_id, bool output) {
    common_batch batch(lctx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], pos0 + i, seq_id, output || i == n_tokens - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }
}

// the raw attention scores of each layer are linear in K, unlike the logits they do not depend on how sharp the attention is
struct kq_capture {
    bool active = false;
    std::vector<std::pair<std::string, std::vector<float>>> data;
};

// the scores are compared cell by cell, so the cells of the probe must match: prefix and probe fit in the SWA window,
// otherwise the probe can reuse a masked SWA cell in one run and not in the other
static bool kq_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    kq_capture * cap = (kq_capture *) user_data;
    const char * name = ggml_get_name(t);
    const bool want = cap->active && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        (strncmp(name, "kq-", 3) == 0 || strncmp(name, "indexer_kq-", 11) == 0 || strncmp(name, "indexer_score-", 14) == 0);
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto it = std::find_if(cap->data.begin(), cap->data.end(), [&](const auto & e) { return e.first == name; });
        if (it == cap->data.end()) {
            cap->data.emplace_back(name, std::move(buf));
        } else {
            it->second = std::move(buf);
        }
    }
    return true;
}

struct kq_diff {
    double      worst = 0.0;
    std::string first_bad; // first tensor above the threshold, later layers can inherit its error through Q
    std::string bad;       // all tensors above the threshold, with their nmse
};

static kq_diff compare_kq(const std::vector<std::pair<std::string, std::vector<float>>> & ref,
                          const std::vector<std::pair<std::string, std::vector<float>>> & cur, double thold) {
    if (ref.size() != cur.size()) {
        throw std::runtime_error("kq tensor count mismatch");
    }
    kq_diff res;
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i].first != cur[i].first || ref[i].second.size() != cur[i].second.size()) {
            throw std::runtime_error("kq tensor mismatch: " + ref[i].first);
        }
        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref[i].second.size(); j++) {
            const double a = ref[i].second[j];
            const double b = cur[i].second[j];
            // masked rows may hold -inf in both
            if (!std::isfinite(a) && a == b) {
                continue;
            }
            mse_a_b += (a - b)*(a - b);
            mse_a_0 += a*a;
        }
        const double val = mse_a_b == 0.0 ? 0.0 : mse_a_0 == 0.0 ? INFINITY : mse_a_b/mse_a_0;
        res.worst = std::max(res.worst, std::isnan(val) ? INFINITY : val);
        if (!(val <= thold)) {
            if (res.first_bad.empty()) {
                res.first_bad = ref[i].first;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), " %s=%.1e", ref[i].first.c_str(), val);
            res.bad += buf;
        }
    }
    return res;
}

// decodes the prefix at pos_prefix, applies each step of moves with its own K-shift and returns the logits of the probe at pos_probe
static std::vector<float> get_logits_kv_shift(
        llama_model * model, llama_context * lctx, kq_capture & cap, const std::vector<llama_token> & prefix, const std::vector<llama_token> & probe,
        llama_pos pos_prefix, const std::vector<std::vector<kv_shift_move>> & steps, llama_pos pos_probe) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    cap.data.clear();
    llama_memory_clear(mem, true);
    decode_range(lctx, prefix.data(), prefix.size(), pos_prefix, 0, false);

    for (size_t i = 0; i < steps.size(); i++) {
        for (const kv_shift_move & mv : steps[i]) {
            llama_memory_seq_add(mem, 0, mv.p0, mv.p1, mv.delta);
        }
        // there is no public API to apply a pending shift, a decode of a throwaway token in seq 1 does it
        if (i + 1 < steps.size()) {
            decode_range(lctx, probe.data(), 1, 0, 1, false);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }

    cap.active = true;
    decode_range(lctx, probe.data(), probe.size(), pos_probe, 0, true);
    cap.active = false;

    std::vector<float> ret;
    ret.reserve(probe.size()*n_vocab);
    for (uint32_t i = 0; i < probe.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
    }
    return ret;
}

// a K-shift is correct if decoding at [d, d+n) and shifting by -d gives the same probe logits as decoding at [0, n)
static int test_kv_shift(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // n_prefix + n_probe stays below the fixture SWA window (n_ctx/8 = 32)
    const uint32_t  n_prefix = 20;
    const uint32_t  n_probe  = 8;
    const llama_pos d        = 64;
    const llama_pos gap      = 4;

    const std::vector<llama_token> tokens = get_tokens(n_prefix + n_probe, 128, seed);
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);
    const std::vector<llama_token> probe (tokens.begin() + n_prefix, tokens.end());

    const std::vector<std::vector<kv_shift_move>> steps_shift  = { { { d, d + (llama_pos) n_prefix, -d } } };
    const std::vector<std::vector<kv_shift_move>> steps_shift2 = { { { d, d + (llama_pos) n_prefix, -d/2 } }, { { d/2, d/2 + (llama_pos) n_prefix, -d/2 } } };
    // control: the second half of the prefix ends up gap positions too far, which tells if the probe is sensitive to K positions at all
    const std::vector<std::vector<kv_shift_move>> steps_ctrl   = { {
        { d,                           d + (llama_pos) n_prefix/2, -d       },
        { d + (llama_pos) n_prefix/2,  d + (llama_pos) n_prefix,   -d + gap },
    } };

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }
    const std::string template_row = "|%" + std::to_string(max_arch_name_length) + "s|%6s|%10s|%10s|%10s|%10s|%s\n";

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_row.c_str(), "Model arch.", "Config", "Shift kq", "Shift2 kq", "Logits", "Control kq", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !arch_supported(arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        if (encode) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * config_name = moe ? "MoE" : "Dense";

            // the attention scale depends on the absolute position, so the hidden states of the prefix do too
            const double thold = arch == LLM_ARCH_MISTRAL_LARGE4 ? 1e-4 : 1e-6;

            // at least 4 layers, so that "every 4th layer" patterns (NoPE, full attention) are covered
            // exaone4 is SWA only with 64 layers, smallthinker always ropes with 4
            const uint32_t n_layer_min = arch == LLM_ARCH_EXAONE4 ? 64 : arch == LLM_ARCH_SMALLTHINKER ? 8 : 4;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, n_layer_min);
            // different bases, so that a shift using the wrong one is caught
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE).c_str(),     20000.0f);
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE_SWA).c_str(), 10000.0f);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }

            llama_model_ptr model;
            try {
                model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            } catch (const std::exception & err) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;31mERROR\033[0m (model load)");
                n_tests++;
                n_failed++;
                continue;
            }
            // absolute position embeddings are not translation invariant, the comparison does not hold
            if (llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NONE) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;33mSKIP\033[0m (no rope)");
                continue;
            }

            LOG("|%*s|%6s|", (int) max_arch_name_length, llama_arch_name(arch), config_name);
            fflush(stdout);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = 0;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.n_seq_max       = 2;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = GGML_TYPE_F32;
            ctx_params.type_v          = GGML_TYPE_F32;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // differs from the SWA default, so that a shift using the wrong scale is caught
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            ctx_params.rope_freq_scale   = 0.5f;
            kq_capture cap;
            ctx_params.cb_eval           = kq_capture_cb;
            ctx_params.cb_eval_user_data = &cap;

            ud.verbosity = LOG_LEVEL_OUTPUT;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            ud.verbosity = verbosity;
            if (!lctx) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (context)\n", "", "", "", "");
                continue;
            }
            llama_memory_t mem = llama_get_memory(lctx.get());
            if (mem == nullptr || !llama_memory_can_shift(mem)) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (no shift)\n", "", "", "", "");
                continue;
            }

            std::string status;
            char shift_str[16]  = "";
            char shift2_str[16] = "";
            char logits_str[16] = "";
            char ctrl_str[16]   = "";
            n_tests++;
            try {
                const std::vector<float> logits_ref = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, 0, {}, n_prefix);
                const auto kq_ref = cap.data;
                if (verbosity >= LOG_LEVEL_INFO) {
                    for (const auto & e : kq_ref) {
                        LOG("[%s: %zu] ", e.first.c_str(), e.second.size());
                    }
                }
                const std::vector<float> logits_shift = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift, n_prefix);
                const kq_diff diff_shift = compare_kq(kq_ref, cap.data, thold);
                get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift2, n_prefix);
                const kq_diff diff_shift2 = compare_kq(kq_ref, cap.data, thold);
                const std::vector<float> logits_ctrl = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_ctrl, n_prefix + gap);
                const kq_diff diff_ctrl = compare_kq(kq_ref, cap.data, thold);

                const double nmse_logits = nmse(logits_ref, logits_shift);
                snprintf(shift_str,  sizeof(shift_str),  "%.2e", diff_shift.worst);
                snprintf(shift2_str, sizeof(shift2_str), "%.2e", diff_shift2.worst);
                snprintf(logits_str, sizeof(logits_str), "%.2e", nmse_logits);
                snprintf(ctrl_str,   sizeof(ctrl_str),   "%.2e", diff_ctrl.worst);

                if (kq_ref.empty()) {
                    status = "\033[1;33mSKIP\033[0m (no kq captured)";
                } else if (!diff_shift.first_bad.empty() || !diff_shift2.first_bad.empty()) {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m" + (diff_shift.bad.empty() ? " x2:" + diff_shift2.bad : diff_shift.bad);
                } else if (!(diff_ctrl.worst > 10*thold)) {
                    status = "\033[1;33mOK\033[0m (low sensitivity)";
                } else {
                    status = "\033[1;32mOK\033[0m";
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%10s|%10s|%10s|%10s|%s\n", shift_str, shift2_str, logits_str, ctrl_str, status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

// the archs whose MTP graph writes only its caches on a pass without outputs while LLAMA_MTP_KV_ONLY is set (Q1105)
static bool mtp_kv_only_arch(const llm_arch arch) {
    return arch == LLM_ARCH_QWEN4EXP;
}

// the model fixture with one nextn (MTP) block after the trunk layers
static gguf_context_ptr get_gguf_ctx_mtp(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret = get_gguf_ctx(arch, moe);
    const std::string key_block_count = llama_arch_kv_name(arch, LLM_KV_BLOCK_COUNT);
    const int64_t kid = gguf_find_key(ret.get(), key_block_count.c_str());
    GGML_ASSERT(kid >= 0);
    const uint32_t n_layer_all = gguf_get_val_u32(ret.get(), kid) + 1;
    gguf_set_val_u32(ret.get(), key_block_count.c_str(), n_layer_all);
    gguf_set_val_u32(ret.get(), llama_arch_kv_name(arch, LLM_KV_NEXTN_PREDICT_LAYERS).c_str(), 1);
    if (arch == LLM_ARCH_QWEN4EXP) {
        // the MTP block is a QSA layer too, so that its pass writes indexer keys and pooled keys
        const std::vector<uint32_t> ratios(n_layer_all, 4);
        gguf_set_arr_data(ret.get(), llama_arch_kv_name(arch, LLM_KV_ATTENTION_COMPRESS_RATIOS).c_str(), GGUF_TYPE_UINT32,
                ratios.data(), ratios.size());
    }
    return ret;
}

struct mtp_kv_only_config {
    const char * name;
    ggml_type    type_kv;
    bool         fa;
    int          smpl; // 0: no backend sampler, 1: top-k (the draft-mtp driver's), 2: top-k and dist
};

struct mtp_kv_only_run {
    std::vector<uint8_t>     state_prompt; // the draft's sequence state after the prompt's catch-up
    std::vector<uint8_t>     state_end;    // and after every pass
    std::vector<float>       out;          // each draft pass's logits (or sampled logits) and h_nextn row
    std::vector<llama_token> cand;         // each draft pass's sampled candidates and token
    size_t                   n_nodes_catchup = 0;
    bool                     smpl_ok = true;

    // LLAMA_QSA_KPOOL_DEV (C100): the draft memory's copies of the k-pool tables (llama_kpool_dev_stats) and the k-pool
    // host inputs the catch-up ubatches' ops read, each one a copy its card receives (C748)
    bool     kpool_dev       = false;
    uint64_t kpool_full      = 0; // updates that rewrote every row
    uint64_t kpool_rows      = 0; // rows uploaded
    uint64_t kpool_checked   = 0; // ubatches LLAMA_QSA_KPOOL_DEV_CHECK compared
    size_t   catchup_ubatch  = 0;
    size_t   catchup_copies  = 0;
    size_t   catchup_bytes   = 0;
};

struct mtp_node_count {
    bool   on = false;
    size_t n  = 0;

    // with LLAMA_QSA_KPOOL_DEV: the k-pool host inputs ("kpool_*") read by an op, by the ubatch the memory counts
    const llama_context * ctx = nullptr;
    std::map<std::pair<uint64_t, const ggml_tensor *>, size_t> kpool_read;
    std::map<uint64_t, size_t> ubatches;
};

static bool mtp_view_op(const ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// LLAMA_MTP_KV_ONLY for the next context, or unset with nullptr
static void mtp_kv_only_setenv(const char * value) {
#ifdef _WIN32
    _putenv_s("LLAMA_MTP_KV_ONLY", value ? value : "");
#else
    if (value) {
        setenv("LLAMA_MTP_KV_ONLY", value, 1);
    } else {
        unsetenv("LLAMA_MTP_KV_ONLY");
    }
#endif
}

// what the draft-mtp driver does with an MTP context: the prompt's catch-up in three batches without outputs, a step of
// two draft passes (the second reads the first one's h), the draft region removed and the catch-up over the step's three
// accepted tokens, then the next step's first draft pass
static mtp_kv_only_run mtp_kv_only_sequence(
        llama_model * model, const std::vector<llama_token> & tokens, const std::vector<float> & h,
        const mtp_kv_only_config & cfg, const bool kv_only, const size_t seed) {
    // read once, when the context is made
    mtp_kv_only_setenv(kv_only ? "1" : nullptr);

    mtp_node_count count;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.n_ctx           = 0;
    ctx_params.n_batch         = 128;
    // micro-batches that end inside the k-pool blocks, so that blocks complete across them
    ctx_params.n_ubatch        = 18;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.type_k          = cfg.type_kv;
    ctx_params.type_v          = cfg.type_kv;
    ctx_params.flash_attn_type = cfg.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.n_outputs_max_per_seq = 4;
    ctx_params.cb_eval = [](struct ggml_tensor * t, bool ask, void * user_data) {
        mtp_node_count * c = (mtp_node_count *) user_data;
        if (ask && c->on) {
            c->n++;
            // the memory counts a ubatch in apply(), before its graph runs
            uint64_t ub = 0;
            if (c->ctx != nullptr && llama_kpool_dev_stats(c->ctx, &ub, nullptr, nullptr, nullptr)) {
                c->ubatches[ub]++;
                for (int i = 0; i < GGML_MAX_SRC && !mtp_view_op(t->op); ++i) {
                    const ggml_tensor * s = t->src[i];
                    if (s == nullptr) {
                        continue;
                    }
                    s = s->view_src != nullptr ? s->view_src : s;
                    if ((s->flags & GGML_TENSOR_FLAG_INPUT) && strncmp(s->name, "kpool_", 6) == 0) {
                        c->kpool_read[{ ub, s }] = ggml_nbytes(s);
                    }
                }
            }
        }
        return false;
    };
    ctx_params.cb_eval_user_data = &count;

    llama_context_ptr lctx(llama_init_from_model(model, ctx_params));
    mtp_kv_only_setenv(nullptr);
    if (!lctx) {
        throw std::runtime_error("failed to create MTP context");
    }
    llama_context * ctx = lctx.get();
    llama_set_embeddings_nextn(ctx, true, /*masked*/ true);
    count.ctx = ctx;

    mtp_kv_only_run run;

    llama_sampler * chain = nullptr;
    if (cfg.smpl > 0) {
        chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));
        if (cfg.smpl > 1) {
            llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t) seed));
        }
        if (!llama_set_sampler(ctx, 0, chain)) {
            llama_sampler_free(chain);
            run.smpl_ok = false;
            return run;
        }
    }

    const size_t n_embd   = llama_model_n_embd_out(model);
    const int    n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int    n_prompt = (int) tokens.size() - 4;

    auto catchup = [&](int p0, int p1) {
        common_batch batch(ctx);
        for (int pos = p0; pos < p1; pos++) {
            const int32_t idx = batch.add(tokens[pos], pos, 0, false);
            batch.set_embd_state(idx, { h.data() + pos*n_embd, 1, n_embd });
        }
        count.on = true;
        const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        count.on = false;
        if (rc != 0) {
            throw std::runtime_error("failed to decode an MTP catch-up batch");
        }
    };

    // one draft pass: returns its h_nextn row, records its outputs
    auto draft = [&](int pos, const float * h_row) {
        common_batch batch(ctx);
        const int32_t idx = batch.add(tokens[pos], pos, 0, true);
        batch.set_embd_state(idx, { h_row, 1, n_embd });
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("failed to decode an MTP draft pass");
        }
        if (cfg.smpl > 0) {
            const float * logits = llama_get_sampled_logits_ith(ctx, 0);
            const uint32_t n_logits = llama_get_sampled_logits_count_ith(ctx, 0);
            if (logits) {
                run.out.insert(run.out.end(), logits, logits + n_logits);
            }
            const llama_token * cand = llama_get_sampled_candidates_ith(ctx, 0);
            const uint32_t n_cand = llama_get_sampled_candidates_count_ith(ctx, 0);
            if (cand) {
                run.cand.insert(run.cand.end(), cand, cand + n_cand);
            }
            run.cand.push_back(llama_get_sampled_token_ith(ctx, 0));
        } else {
            const float * logits = llama_get_logits_ith(ctx, 0);
            GGML_ASSERT(logits != nullptr);
            run.out.insert(run.out.end(), logits, logits + n_vocab);
        }
        const float * h_out = llama_get_embeddings_nextn_ith(ctx, 0);
        GGML_ASSERT(h_out != nullptr);
        run.out.insert(run.out.end(), h_out, h_out + n_embd);
        return std::vector<float>(h_out, h_out + n_embd);
    };

    auto state = [&]() {
        std::vector<uint8_t> s(llama_state_seq_get_size(ctx, 0));
        llama_state_seq_get_data(ctx, s.data(), s.size(), 0);
        return s;
    };

    const int b0 = n_prompt*3/8;
    const int b1 = n_prompt*3/4;
    catchup(0,  b0);
    catchup(b0, b1);
    catchup(b1, n_prompt);
    run.state_prompt = state();

    const std::vector<float> h1 = draft(n_prompt, h.data() + n_prompt*n_embd);
    draft(n_prompt + 1, h1.data());

    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt, -1)) {
        throw std::runtime_error("failed to remove the draft region");
    }
    catchup(n_prompt, n_prompt + 3);
    draft(n_prompt + 3, h.data() + (n_prompt + 3)*n_embd);
    run.state_end = state();

    run.n_nodes_catchup = count.n;

    run.kpool_dev = llama_kpool_dev_stats(ctx, nullptr, &run.kpool_full, &run.kpool_rows, &run.kpool_checked);
    run.catchup_ubatch = count.ubatches.size();
    run.catchup_copies = count.kpool_read.size();
    for (const auto & r : count.kpool_read) {
        run.catchup_bytes += r.second;
    }

    if (chain) {
        llama_set_sampler(ctx, 0, nullptr);
        llama_sampler_free(chain);
    }
    return run;
}

static int test_mtp_kv_only(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const mtp_kv_only_config configs[] = {
        { "f16 fa",       GGML_TYPE_F16,  true,  0 },
        { "f16",          GGML_TYPE_F16,  false, 0 },
        { "q8_0 fa",      GGML_TYPE_Q8_0, true,  0 },
        { "f16 fa top-k", GGML_TYPE_F16,  true,  1 },
        { "f16 fa dist",  GGML_TYPE_F16,  true,  2 },
    };

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    // with LLAMA_QSA_KPOOL_DEV (C100) also the tables' full rewrites and rows uploaded, which a KV-only pass must leave
    // as the full pass leaves them (C748), and the k-pool copies and bytes of the catch-up ubatches
    LOG("|%10s|%13s|%12s|%12s|%12s|%12s|%11s|%11s|%9s|%11s|%13s|%s\n", "Model arch.", "Config", "KV, prompt", "Drafts",
            "KV, end", "Nodes off/on", "Full off/on", "Rows off/on", "Catch-up", "Copies", "Bytes", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !mtp_kv_only_arch(arch)) {
            continue;
        }
        const bool moe = moe_mandatory(arch) || moe_implemented(arch);

        gguf_context_ptr gguf_ctx = get_gguf_ctx_mtp(arch, moe);

        llama_model_ptr model;
        try {
            model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, nullptr, /*load_mtp*/ true);
        } catch (const std::exception & err) {
            LOG("|%10s|%13s|%12s|%12s|%12s|%12s|\033[1;31mERROR\033[0m (model load: %s)\n",
                    llama_arch_name(arch), "", "", "", "", "", err.what());
            n_tests++;
            n_failed++;
            continue;
        }

        // 100 prompt tokens, then the 4 positions of the draft step
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const std::vector<llama_token> tokens = get_tokens(104, n_vocab, seed);
        std::vector<float> h(llama_model_n_embd_out(model.get())*tokens.size());
        {
            std::mt19937 gen(seed + 1);
            std::normal_distribution<float> dis(0.0f, 1.0f);
            for (float & v : h) {
                v = dis(gen);
            }
        }

        for (const mtp_kv_only_config & cfg : configs) {
            n_tests++;
            LOG("|%10s|%13s|", llama_arch_name(arch), cfg.name);
            std::string status;
            try {
                const mtp_kv_only_run off = mtp_kv_only_sequence(model.get(), tokens, h, cfg, false, seed);
                const mtp_kv_only_run on  = mtp_kv_only_sequence(model.get(), tokens, h, cfg, true,  seed);
                if (!off.smpl_ok || !on.smpl_ok) {
                    LOG("%12s|%12s|%12s|%12s|%11s|%11s|%9s|%11s|%13s|\033[1;33mSKIP\033[0m (no backend sampler on this device)\n",
                            "", "", "", "", "", "", "", "", "");
                    n_tests--;
                    continue;
                }

                const bool same_prompt = off.state_prompt == on.state_prompt;
                const bool same_drafts = off.out.size() == on.out.size() && !off.out.empty() &&
                    memcmp(off.out.data(), on.out.data(), off.out.size()*sizeof(float)) == 0 && off.cand == on.cand;
                const bool same_end    = off.state_end == on.state_end;
                // the dist sampler keeps the full pass, so that its draws stay in step
                const bool nodes_ok    = cfg.smpl == 2 ? on.n_nodes_catchup == off.n_nodes_catchup
                                                       : on.n_nodes_catchup <  off.n_nodes_catchup;

                // the tables on the devices: a KV-only pass updates the copies as the full pass does, so the next pass
                // sends the changed rows only and never rewrites them all
                const bool tables_ok   = off.kpool_dev == on.kpool_dev &&
                    on.kpool_full == off.kpool_full && on.kpool_rows == off.kpool_rows;

                char nodes_str[32];
                snprintf(nodes_str, sizeof(nodes_str), "%zu/%zu", off.n_nodes_catchup, on.n_nodes_catchup);
                char full_str[32] = "-";
                char rows_str[32] = "-";
                char ub_str[32]   = "-";
                char copy_str[32] = "-";
                char byte_str[32] = "-";
                if (on.kpool_dev) {
                    snprintf(full_str, sizeof(full_str), "%" PRIu64 "/%" PRIu64, off.kpool_full, on.kpool_full);
                    snprintf(rows_str, sizeof(rows_str), "%" PRIu64 "/%" PRIu64, off.kpool_rows, on.kpool_rows);
                    snprintf(ub_str,   sizeof(ub_str),   "%zu/%zu", off.catchup_ubatch, on.catchup_ubatch);
                    snprintf(copy_str, sizeof(copy_str), "%zu/%zu", off.catchup_copies, on.catchup_copies);
                    snprintf(byte_str, sizeof(byte_str), "%zu/%zu", off.catchup_bytes, on.catchup_bytes);
                }
                LOG("%12s|%12s|%12s|%12s|%11s|%11s|%9s|%11s|%13s|", same_prompt ? "identical" : "DIFFERENT",
                        same_drafts ? "identical" : "DIFFERENT", same_end ? "identical" : "DIFFERENT", nodes_str,
                        full_str, rows_str, ub_str, copy_str, byte_str);

                if (same_prompt && same_drafts && same_end && nodes_ok && tables_ok) {
                    status = "\033[1;32mOK\033[0m";
                    if (off.kpool_checked + on.kpool_checked > 0) {
                        char chk_str[64];
                        snprintf(chk_str, sizeof(chk_str), " (ubatches checked off/on %" PRIu64 "/%" PRIu64 ")",
                                off.kpool_checked, on.kpool_checked);
                        status += chk_str;
                    }
                } else {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m";
                    if (!nodes_ok) {
                        status += cfg.smpl == 2 ? " (the pass was cut with a dist sampler)" : " (the pass was not cut)";
                    }
                    if (!tables_ok) {
                        status += " (the KV-only passes left the k-pool tables on the devices behind)";
                    }
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%s\n", status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

// LLAMA_NGRAM_PREFETCH (C761): the archs whose n-gram input issues its table rows through llama_ngram_prefetch
static bool ngram_prefetch_arch(const llm_arch arch) {
    return arch == LLM_ARCH_QWEN4EXP;
}

static void ngram_prefetch_setenv(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct ngram_prefetch_arm {
    const char * name;
    const char * on; // LLAMA_NGRAM_PREFETCH, nullptr: unset
    const char * mb; // LLAMA_NGRAM_PREFETCH_MB, nullptr: the default
};

struct ngram_prefetch_run {
    std::vector<float> logits;
    bool     stats         = false;
    bool     on            = false;
    uint64_t cap           = 0;
    uint64_t page          = 0;
    uint64_t planned       = 0;
    uint64_t rows_helper   = 0;
    uint64_t rows_inline   = 0;
    uint64_t rows_mismatch = 0;
    uint64_t window_max    = 0;
    std::vector<std::array<uint64_t, 11>> calls;
};

// the model file loaded with its n-gram table mapped (--lazy-mode on) and the decode calls of a server: a one-token
// warm-up (removed), a 120-token prompt and a 70-token continuation in micro-batches of 16, three one-token steps, a
// 40-token call, then a second sequence's 50-token prompt; every token's logits
static ngram_prefetch_run ngram_prefetch_sequence(const std::string & path, const ngram_prefetch_arm & arm,
        const std::vector<llama_token> & tokens) {
    // read once, when the context is made
    ngram_prefetch_setenv("LLAMA_NGRAM_PREFETCH",       arm.on);
    ngram_prefetch_setenv("LLAMA_NGRAM_PREFETCH_MB",    arm.mb);
    ngram_prefetch_setenv("LLAMA_NGRAM_PREFETCH_STATS", "1");

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t no_devs[1] = { nullptr };
    model_params.devices   = no_devs;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.lazy_mode = LLAMA_LAZY_MODE_ON;
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), model_params));
    if (!model) {
        throw std::runtime_error("failed to load " + path);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 512;
    ctx_params.n_batch         = 128;
    ctx_params.n_ubatch        = 16;
    ctx_params.n_seq_max       = 2;
    ctx_params.kv_unified      = true;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    ngram_prefetch_setenv("LLAMA_NGRAM_PREFETCH",       nullptr);
    ngram_prefetch_setenv("LLAMA_NGRAM_PREFETCH_MB",    nullptr);
    ngram_prefetch_setenv("LLAMA_NGRAM_PREFETCH_STATS", nullptr);
    if (!lctx) {
        throw std::runtime_error("failed to create the context");
    }
    llama_context * ctx = lctx.get();
    llama_memory_t  mem = llama_get_memory(ctx);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    ngram_prefetch_run run;
    common_batch batch(ctx);
    // tokens[t0, t0 + n) at positions p0, p0 + 1, ... of sequence s, every logit kept
    auto decode = [&](size_t t0, uint32_t n, llama_pos p0, llama_seq_id s, bool keep) {
        batch.clear();
        for (uint32_t i = 0; i < n; i++) {
            batch.add(tokens[t0 + i], p0 + (llama_pos) i, s, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            throw std::runtime_error("failed to decode");
        }
        if (keep) {
            for (uint32_t i = 0; i < n; i++) {
                const float * logits_ith = llama_get_logits_ith(ctx, i);
                run.logits.insert(run.logits.end(), logits_ith, logits_ith + n_vocab);
            }
        }
    };

    decode(0, 1, 0, 1, false);                 // the warm-up: the n-gram input is seen from here on
    llama_memory_seq_rm(mem, 1, -1, -1);
    decode(1,   120,   0, 0, true);            // positions 0..119, 8 micro-batches
    decode(121,  70, 120, 0, true);            // 120..189, 5 micro-batches
    decode(191,   1, 190, 0, true);            // three steps
    decode(192,   1, 191, 0, true);
    decode(193,   1, 192, 0, true);
    decode(194,  40, 193, 0, true);            // 193..232, 3 micro-batches
    decode(234,  50,   0, 1, true);            // a second sequence, 4 micro-batches

    run.stats = llama_ngram_prefetch_stats(ctx, &run.on, &run.cap, &run.page, &run.planned, &run.rows_helper,
            &run.rows_inline, &run.rows_mismatch, &run.window_max, &run.calls);
    return run;
}

static int test_ngram_prefetch(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // off: the inline path, counted; on: the default cap; small: 64 KiB (a few tokens' rows of the test table);
    // zero: no room at all, so set_input issues every row
    const ngram_prefetch_arm arms[] = {
        { "on",    "1", nullptr  },
        { "small", "1", "0.0625" },
        { "zero",  "1", "0"      },
    };
    const ngram_prefetch_arm arm_off = { "off", nullptr, nullptr };

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG("|%10s|%6s|%6s|%9s|%5s|%7s|%15s|%13s|%13s|%11s|%9s|%19s|%s\n", "Model arch.", "Type", "Arm", "Logits", "Calls",
            "Planned", "Rows off/on", "Helper rows", "Inline rows", "Pages", "Digests", "Window max/cap", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !ngram_prefetch_arch(arch)) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }

            // the model written to a file, so that its n-gram table can be mapped
            const std::string path = get_tmp_gguf_path();
            ngram_prefetch_run off;
            std::vector<llama_token> tokens;
            try {
                gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
                {
                    llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
                    llama_model_save_to_file(model.get(), path.c_str());
                    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
                    tokens = get_tokens(284, n_vocab, seed);
                }
                off = ngram_prefetch_sequence(path, arm_off, tokens);
            } catch (const std::exception & err) {
                LOG("|%10s|%6s|%6s|\033[1;31mERROR\033[0m (%s)\n", llama_arch_name(arch), moe ? "MoE" : "Dense", "off", err.what());
                n_tests++;
                n_failed++;
                std::filesystem::remove(path);
                continue;
            }
            for (size_t c = 0; c < off.calls.size(); ++c) {
                const auto & r = off.calls[c];
                LOG("ngram-call|%s|%s|off|%zu|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%" PRIu64
                        "|%" PRIu64 "|%" PRIu64 "|%016" PRIx64 "|%" PRIu64 "\n", llama_arch_name(arch), moe ? "MoE" : "Dense", c,
                        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]);
            }

            for (const ngram_prefetch_arm & arm : arms) {
                n_tests++;
                LOG("|%10s|%6s|%6s|", llama_arch_name(arch), moe ? "MoE" : "Dense", arm.name);
                std::string status;
                ngram_prefetch_run on;
                try {
                    on = ngram_prefetch_sequence(path, arm, tokens);

                    const bool same_logits = !off.logits.empty() && off.logits.size() == on.logits.size() &&
                        memcmp(off.logits.data(), on.logits.data(), off.logits.size()*sizeof(float)) == 0;

                    // per call: the same rows (count and digest) and pages as the inline path, no helper row that differed
                    // from set_input's, the helper's rows and set_input's adding up, and the window within the cap
                    bool same_calls = off.stats && on.stats && !off.on && on.on && off.calls.size() == on.calls.size() &&
                        !off.calls.empty();
                    bool cap_ok = on.window_max <= on.cap;
                    uint64_t rows_off = 0, rows_on = 0, pages_off = 0, pages_on = 0, n_dig = 0;
                    for (size_t c = 0; same_calls && c < off.calls.size(); ++c) {
                        const auto & a = off.calls[c];
                        const auto & b = on.calls[c];
                        same_calls = a[0] == b[0] && a[1] == b[1] && a[3] == b[3] && a[7] == b[7] && a[9] == b[9] &&
                            b[6] == 0 && b[3] == b[4] + b[5] && a[4] == 0;
                        cap_ok = cap_ok && b[10] <= on.cap;
                        rows_off += a[3]; rows_on += b[3]; pages_off += a[7]; pages_on += b[7]; n_dig += a[9] == b[9];
                    }
                    // the default cap takes some rows off set_input; no room at all takes none
                    const bool helper_ok = strcmp(arm.name, "zero") == 0 ? on.rows_helper == 0 : on.rows_helper > 0;
                    // the planned calls: the four calls of several micro-batches
                    const bool planned_ok = on.planned == 4;

                    char rows_str[32], help_str[32], inl_str[32], pages_str[32], dig_str[32], win_str[32];
                    snprintf(rows_str,  sizeof(rows_str),  "%" PRIu64 "/%" PRIu64, rows_off, rows_on);
                    snprintf(help_str,  sizeof(help_str),  "%" PRIu64, on.rows_helper);
                    snprintf(inl_str,   sizeof(inl_str),   "%" PRIu64, on.rows_inline);
                    snprintf(pages_str, sizeof(pages_str), "%" PRIu64 "/%" PRIu64, pages_off, pages_on);
                    snprintf(dig_str,   sizeof(dig_str),   "%" PRIu64 "/%zu", n_dig, off.calls.size());
                    snprintf(win_str,   sizeof(win_str),   "%" PRIu64 "/%" PRIu64, on.window_max, on.cap);
                    LOG("%9s|%5zu|%7" PRIu64 "|%15s|%13s|%13s|%11s|%9s|%19s|", same_logits ? "identical" : "DIFFERENT",
                            on.calls.size(), on.planned, rows_str, help_str, inl_str, pages_str, dig_str, win_str);

                    if (same_logits && same_calls && cap_ok && helper_ok && planned_ok && on.rows_mismatch == 0) {
                        status = "\033[1;32mOK\033[0m";
                    } else {
                        n_failed++;
                        status = "\033[1;31mFAIL\033[0m";
                        if (!same_calls) {
                            status += " (a call's rows, pages or digest differ from the inline path's)";
                        }
                        if (!cap_ok) {
                            status += " (the helper read past the cap)";
                        }
                        if (!helper_ok) {
                            status += " (the helper's share is not as the cap allows)";
                        }
                        if (!planned_ok) {
                            status += " (not the four planned calls)";
                        }
                    }
                } catch (const std::exception & err) {
                    n_failed++;
                    status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
                }
                LOG("%s\n", status.c_str());
                // each call's counts: index, n_tokens, micro-batches, planned, rows, helper, inline, mismatch, pages,
                // madvise bytes, digest, window max
                for (size_t c = 0; c < on.calls.size(); ++c) {
                    const auto & r = on.calls[c];
                    LOG("ngram-call|%s|%s|%s|%zu|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%" PRIu64
                            "|%" PRIu64 "|%" PRIu64 "|%" PRIu64 "|%016" PRIx64 "|%" PRIu64 "\n", llama_arch_name(arch),
                            moe ? "MoE" : "Dense", arm.name, c, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                            r[10]);
                }
            }
            std::filesystem::remove(path);
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

// LLAMA_MTP_CATCHUP_OVERLAP (C703): the draft-mtp driver with the target, in the server's order, with the switch off and on
struct mtp_overlap_config {
    const char * name;
    ggml_type    type_kv;
    bool         fa;
    bool         kv_only;          // LLAMA_MTP_KV_ONLY (C620) as well
    bool         backend_sampling; // the driver's top-k on the backend
};

struct mtp_overlap_run {
    std::vector<uint8_t>     state_ckpt;   // the draft's sequence state and the driver's, where the server takes a
    std::vector<uint8_t>     spec_ckpt;    // checkpoint (after the prompt's second call, through after_catchup)
    std::vector<uint8_t>     state_prompt; // both after the prompt
    std::vector<uint8_t>     spec_prompt;
    std::vector<uint8_t>     state_end;    // both after the last step
    std::vector<uint8_t>     spec_end;
    std::vector<llama_token> drafts;       // each step's draft, its length first
    std::vector<float>       probs;        // the drafter's probability of each drafted token
    int                      n_pending = 0; // prompt calls whose catch-up waited for the next call
};

// set or (with nullptr) unset an environment variable for the next context or driver
static void test_setenv(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// a prompt of 100 tokens in four calls (32, 32, 32 and 4, the last one with an output), a checkpoint's draft part and
// the cut of the memory past the batch's start as the server takes them between calls, three draft steps (each: the
// draft, its region removed from the draft's memory, the verify batch through the target and the driver, an accepted
// count of 0, 1, then all), a second turn of 45 tokens in two calls (32, 13) and a last draft step
static mtp_overlap_run mtp_overlap_sequence(
        llama_model * model, const std::vector<llama_token> & tokens, const mtp_overlap_config & cfg, const bool overlap) {
    test_setenv("LLAMA_MTP_KV_ONLY", cfg.kv_only ? "1" : nullptr);
    test_setenv("LLAMA_MTP_CATCHUP_OVERLAP", overlap ? "1" : nullptr);

    auto ctx_params_for = [&](const bool mtp) {
        llama_context_params p = llama_context_default_params();
        p.ctx_type        = mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
        p.n_ctx           = 0;
        p.n_batch         = 32;
        // the target's calls cross micro-batches; the draft's passes end inside the k-pool blocks
        p.n_ubatch        = mtp ? 18 : 8;
        p.n_threads       = 4;
        p.n_threads_batch = 4;
        p.type_k          = cfg.type_kv;
        p.type_v          = cfg.type_kv;
        p.flash_attn_type = cfg.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        p.n_outputs_max_per_seq = 4;
        return p;
    };

    llama_context_ptr lctx_tgt(llama_init_from_model(model, ctx_params_for(false)));
    llama_context_ptr lctx_dft(llama_init_from_model(model, ctx_params_for(true)));
    test_setenv("LLAMA_MTP_KV_ONLY", nullptr);
    if (!lctx_tgt || !lctx_dft) {
        test_setenv("LLAMA_MTP_CATCHUP_OVERLAP", nullptr);
        throw std::runtime_error("failed to create the target or the MTP context");
    }
    llama_context * ctx_tgt = lctx_tgt.get();
    llama_context * ctx_dft = lctx_dft.get();

    common_params_speculative sp;
    sp.types                  = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    sp.draft.ctx_tgt          = ctx_tgt;
    sp.draft.ctx_dft          = ctx_dft;
    sp.draft.n_max            = 3;
    sp.draft.n_min            = 0;
    sp.draft.p_min            = 0.0f;
    sp.draft.backend_sampling = cfg.backend_sampling;
    common_speculative_ptr spec(common_speculative_init(sp, 1));
    test_setenv("LLAMA_MTP_CATCHUP_OVERLAP", nullptr);
    if (!spec) {
        throw std::runtime_error("failed to create the draft-mtp driver");
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    mtp_overlap_run run;

    auto state_dft = [&]() {
        std::vector<uint8_t> s(llama_state_seq_get_size(ctx_dft, 0));
        llama_state_seq_get_data(ctx_dft, s.data(), s.size(), 0);
        return s;
    };

    auto argmax = [&](int32_t i) {
        const float * logits = llama_get_logits_ith(ctx_tgt, i);
        GGML_ASSERT(logits != nullptr);
        return (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
    };

    llama_tokens context;
    llama_pos    pos = 0;

    // a prompt's calls, the server's way: the target's call, then the driver; between calls the cut past the next
    // batch's start (common_memory::seq_rm) and, after the call ckpt_after, a checkpoint's draft part
    auto prompt = [&](int i0, const std::vector<int> & calls, int ckpt_after) {
        int i = i0;
        for (size_t c = 0; c < calls.size(); ++c) {
            const bool last = c + 1 == calls.size();

            common_memory mem_tgt;
            mem_tgt.init(ctx_tgt);
            mem_tgt.seq_rm(0, pos, -1);
            const llama_pos p0 = pos;
            common_speculative_after_catchup(spec.get(), [&, p0]() {
                common_memory mem_dft;
                mem_dft.init(ctx_dft);
                mem_dft.seq_rm(0, p0, -1);
            });

            common_batch batch(ctx_tgt);
            for (int k = 0; k < calls[c]; ++k, ++i) {
                batch.add(tokens[i], pos++, 0, last && k + 1 == calls[c]);
                context.push_back(tokens[i]);
            }
            if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                throw std::runtime_error("failed to decode a prompt call");
            }
            if (!common_speculative_process(spec.get(), batch)) {
                throw std::runtime_error("the driver failed on a prompt call");
            }
            run.n_pending += common_speculative_pending(spec.get()) ? 1 : 0;

            if ((int) c == ckpt_after) {
                common_speculative_after_catchup(spec.get(), [&]() {
                    run.state_ckpt = state_dft();
                    common_speculative_get_state(spec.get(), 0, run.spec_ckpt);
                });
            }
        }
        return i;
    };

    // a draft step from id_last at pos; the verify batch then accepts n_acc of the draft as far as the driver is told
    // (the target keeps every token, so that its memory needs no rollback)
    auto step = [&](llama_token id_last, int n_acc) {
        llama_tokens draft;
        std::vector<float> probs;
        common_speculative_get_draft_params(spec.get(), 0) = {
            /* .drafting = */ true,
            /* .n_max    = */ 3,
            /* .pos0     = */ pos,
            /* .id_last  = */ id_last,
            /* .prompt   = */ &context,
            /* .result   = */ &draft,
            /* .result_q = */ nullptr,
            /* .temp     = */ 1.0f,
            /* .seed     = */ LLAMA_DEFAULT_SEED,
            /* .info     = */ nullptr,
            /* .result_p = */ &probs,
        };
        common_speculative_draft(spec.get());
        run.drafts.push_back((llama_token) draft.size());
        run.drafts.insert(run.drafts.end(), draft.begin(), draft.end());
        run.probs.insert(run.probs.end(), probs.begin(), probs.end());

        if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos, -1)) {
            throw std::runtime_error("failed to remove the draft region");
        }

        common_batch batch(ctx_tgt);
        batch.add(id_last, pos++, 0, true);
        context.push_back(id_last);
        for (const llama_token t : draft) {
            batch.add(t, pos++, 0, true);
            context.push_back(t);
        }
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("failed to decode a verify batch");
        }
        if (!common_speculative_process(spec.get(), batch)) {
            throw std::runtime_error("the driver failed on a verify batch");
        }
        common_speculative_accept(spec.get(), 0, (uint16_t) std::min<int>(n_acc, (int) draft.size()));
        return argmax(batch.size() - 1);
    };

    int i = prompt(0, { 32, 32, 32, 4 }, 1);
    common_speculative_flush(spec.get());
    run.state_prompt = state_dft();
    common_speculative_get_state(spec.get(), 0, run.spec_prompt);

    common_speculative_begin(spec.get(), 0, context);
    llama_token id = argmax(-1);
    id = step(id, 0);
    id = step(id, 1);
    id = step(id, 3);

    // the next turn: the sampled token and 44 prompt tokens
    context.push_back(id);
    {
        common_batch batch(ctx_tgt);
        batch.add(id, pos++, 0, false);
        if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 ||
                !common_speculative_process(spec.get(), batch)) {
            throw std::runtime_error("failed to decode the turn's first token");
        }
    }
    i = prompt(i, { 31, 13 }, -1);
    common_speculative_begin(spec.get(), 0, context);
    step(argmax(-1), 2);

    common_speculative_flush(spec.get());
    run.state_end = state_dft();
    common_speculative_get_state(spec.get(), 0, run.spec_end);

    spec.reset();
    return run;
}

static int test_mtp_catchup_overlap(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const mtp_overlap_config configs[] = {
        { "f16 fa",         GGML_TYPE_F16,  true,  false, true  },
        { "f16",            GGML_TYPE_F16,  false, false, true  },
        { "q8_0 fa",        GGML_TYPE_Q8_0, true,  false, true  },
        { "f16 fa kv-only", GGML_TYPE_F16,  true,  true,  true  },
        { "f16 fa host",    GGML_TYPE_F16,  true,  false, false },
    };

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG("|%10s|%15s|%11s|%11s|%11s|%11s|%11s|%13s|%s\n", "Model arch.", "Config", "Checkpoint", "Prompt", "Drafts",
            "End", "Ckpt spec", "Waited off/on", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !mtp_kv_only_arch(arch)) {
            continue;
        }
        const bool moe = moe_mandatory(arch) || moe_implemented(arch);

        gguf_context_ptr gguf_ctx = get_gguf_ctx_mtp(arch, moe);

        llama_model_ptr model;
        try {
            model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, nullptr, /*load_mtp*/ true);
        } catch (const std::exception & err) {
            LOG("|%10s|%15s|%11s|%11s|%11s|%11s|%11s|%13s|\033[1;31mERROR\033[0m (model load: %s)\n",
                    llama_arch_name(arch), "", "", "", "", "", "", "", err.what());
            n_tests++;
            n_failed++;
            continue;
        }

        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const std::vector<llama_token> tokens = get_tokens(160, n_vocab, seed);

        for (const mtp_overlap_config & cfg : configs) {
            n_tests++;
            LOG("|%10s|%15s|", llama_arch_name(arch), cfg.name);
            std::string status;
            try {
                // the driver's own log lines stay out of the table
                struct log_pause {
                    log_pause()  { common_log_pause(common_log_main()); }
                    ~log_pause() { common_log_resume(common_log_main()); }
                };
                mtp_overlap_run off;
                mtp_overlap_run on;
                {
                    log_pause pause;
                    off = mtp_overlap_sequence(model.get(), tokens, cfg, false);
                    on  = mtp_overlap_sequence(model.get(), tokens, cfg, true);
                }

                auto same = [](const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
                    return !a.empty() && a == b;
                };
                const bool same_ckpt   = same(off.state_ckpt, on.state_ckpt);
                const bool same_prompt = same(off.state_prompt, on.state_prompt) && same(off.spec_prompt, on.spec_prompt);
                const bool same_drafts = !off.drafts.empty() && off.drafts == on.drafts && off.probs.size() == on.probs.size() &&
                    memcmp(off.probs.data(), on.probs.data(), off.probs.size()*sizeof(float)) == 0;
                const bool same_end    = same(off.state_end, on.state_end) && same(off.spec_end, on.spec_end);
                const bool same_cspec  = same(off.spec_ckpt, on.spec_ckpt);
                // off: no call waits; on: the prompts' calls without an output (3 + 1)
                const bool waited_ok   = off.n_pending == 0 && on.n_pending == 4;

                char waited_str[32];
                snprintf(waited_str, sizeof(waited_str), "%d/%d", off.n_pending, on.n_pending);
                LOG("%11s|%11s|%11s|%11s|%11s|%13s|", same_ckpt ? "identical" : "DIFFERENT", same_prompt ? "identical" : "DIFFERENT",
                        same_drafts ? "identical" : "DIFFERENT", same_end ? "identical" : "DIFFERENT",
                        same_cspec ? "identical" : "DIFFERENT", waited_str);

                if (same_ckpt && same_prompt && same_drafts && same_end && same_cspec && waited_ok) {
                    status = "\033[1;32mOK\033[0m";
                } else {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m";
                    if (!waited_ok) {
                        status += " (the catch-ups did not wait as they should)";
                    }
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%s\n", status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    const char * target_backend = nullptr;
    bool kv_shift = false;
    bool mtp_kv_only = false;
    bool mtp_catchup_overlap = false;
    bool ngram_prefetch = false;
    bool save_mtp = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llama_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--kv-shift") == 0) {
            kv_shift = true;
        } else if (strcmp(argv[i], "--mtp-kv-only") == 0) {
            mtp_kv_only = true;
        } else if (strcmp(argv[i], "--mtp-catchup-overlap") == 0) {
            mtp_catchup_overlap = true;
        } else if (strcmp(argv[i], "--ngram-prefetch") == 0) {
            ngram_prefetch = true;
        } else if (strcmp(argv[i], "--mtp") == 0) {
            save_mtp = true;
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    init_normal_pool(seed, stdev);

    try {
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out, save_mtp);
        }
        if (kv_shift) {
            return test_kv_shift(arch_filter, seed, stdev, verbosity);
        }
        if (mtp_kv_only) {
            return test_mtp_kv_only(arch_filter, seed, stdev, verbosity);
        }
        if (mtp_catchup_overlap) {
            return test_mtp_catchup_overlap(arch_filter, seed, stdev, verbosity);
        }
        if (ngram_prefetch) {
            return test_ngram_prefetch(arch_filter, seed, stdev, verbosity);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
