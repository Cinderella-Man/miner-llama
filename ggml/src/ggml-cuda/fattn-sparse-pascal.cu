#include "fattn-sparse-pascal.cuh"
#include "fattn-common.cuh"
#include "fattn-sparse-pascal-impl.cuh"

#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

struct fasp_env {
    fasp_rule rule;
    int       splits = 0;
    int       nsm    = 0;
    int       fault  = 0;
};

static int fasp_env_int(const char * name, const int def) {
    const char * env = getenv(name);
    return env ? atoi(env) : def;
}

static float fasp_env_float(const char * name, const float def) {
    const char * env = getenv(name);
    return env ? (float) atof(env) : def;
}

static const fasp_env & fasp_get_env() {
    static const fasp_env env = [] {
        fasp_env e;
        e.rule.mode             = fasp_env_int  ("GGML_CUDA_FATTN_SPARSE_PASCAL",                  0);
        e.rule.nq               = fasp_env_int  ("GGML_CUDA_FATTN_SPARSE_PASCAL_NQ",               8);
        e.rule.min_ratio        = fasp_env_float("GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO",        1.5f);
        e.rule.prompt           = fasp_env_int  ("GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT",           0);
        e.rule.prompt_nq        = fasp_env_int  ("GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT_NQ",        128);
        e.rule.prompt_min_ratio = fasp_env_float("GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT_MIN_RATIO", 1.0f);
        e.splits                = fasp_env_int  ("GGML_CUDA_FATTN_SPARSE_PASCAL_SPLITS",           0);
        e.nsm                   = fasp_env_int  ("GGML_CUDA_FATTN_SPARSE_PASCAL_NSM",              0);
        e.fault                 = fasp_env_int  ("GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT",            0);
        return e;
    }();
    return env;
}

static bool fasp_aligned(const void * p, const int64_t al) {
    return (uintptr_t) p % al == 0;
}

// K or V: rows of DPL elements per lane read as one 8- or 16-byte load (F16), or 2-byte loads (Q8_0)
static bool fasp_kv_ok(const ggml_tensor * t, const int D) {
    const int64_t al = t->type == GGML_TYPE_F16 ? (D == 256 ? 16 : 8) : 2;
    return t->nb[0] == ggml_type_size(t->type) && t->nb[1] % al == 0 && t->nb[2] % al == 0 && t->nb[3] % al == 0 &&
        fasp_aligned(t->data, al);
}

} // namespace

bool ggml_cuda_flash_attn_ext_sparse_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const fasp_env & env = fasp_get_env();
    if (env.rule.mode == 0) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || (env.rule.mode == 1 && turing_mma_available(cc))) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // K/V read through kv_rows (src[5]): only the MMA kernel gathers rows
    if (n_kv_max <= 0 || mask == nullptr || dst->src[5] != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    const int D = (int) Q->ne[0];
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || (D != 128 && D != 256) ||
            K->ne[0] != D || V->ne[0] != D || K->type != V->type ||
            (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t n_kv      = K->ne[1];
    const int64_t n_q       = Q->ne[1];
    const int64_t n_head    = Q->ne[2];
    const int64_t ne3       = Q->ne[3];
    const int64_t n_head_kv = K->ne[2];
    if (V->ne[1] != n_kv || V->ne[2] != n_head_kv || V->ne[3] != K->ne[3] || (K->ne[3] != 1 && K->ne[3] != ne3) ||
            n_head % n_head_kv != 0) {
        return false;
    }
    const int G = (int) (n_head/n_head_kv);
    if (G % FASP_HPW != 0 || G > 4*FASP_HPW || n_head_kv*n_q*ne3 > FASP_MAX_COLS) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[0] != n_kv || mask->ne[1] < n_q || mask->ne[2] != 1 ||
            (mask->ne[3] != 1 && mask->ne[3] != ne3) || mask->nb[0] != sizeof(half) || mask->nb[1] % sizeof(half) != 0 ||
            mask->nb[3] % sizeof(half) != 0) {
        return false;
    }
    // decode and verify by _MIN_RATIO per query token, prompt micro-batches (more than _NQ tokens) only with _PROMPT
    if (!fasp_rule_takes(env.rule, n_q, n_kv, n_kv_max)) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || ggml_nelements(sinks) != n_head)) {
        return false;
    }
    // the query as float4, the output as float4
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || Q->nb[3] % 16 != 0 ||
            !fasp_aligned(Q->data, 16) || !fasp_aligned(dst->data, 16) || !fasp_kv_ok(K, D) || !fasp_kv_ok(V, D)) {
        return false;
    }

    if (ctx.fattn_sparse_pascal.counters == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(&ctx.fattn_sparse_pascal.counters, FASP_MAX_COLS*sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(ctx.fattn_sparse_pascal.counters, 0, FASP_MAX_COLS*sizeof(unsigned)));
    }

    cudaStream_t stream = ctx.stream();

    // one list per query and mask sequence: its finite mask columns in cache order, then the count
    const int64_t n_lists = n_q*mask->ne[3];
    ggml_cuda_pool_alloc<int32_t> lists(ctx.pool(), (size_t) n_lists*n_kv_max + n_lists);
    ggml_cuda_flash_attn_ext_compact_mask(mask, lists.ptr, lists.ptr + (size_t) n_lists*n_kv_max, (int32_t) n_q, 1,
                                          n_kv_max, stream);

    const int n_cols = (int) (n_head_kv*n_q*ne3);
    const int nsm    = env.nsm > 0 ? env.nsm : ggml_cuda_info().devices[ctx.device].nsm;
    const int S      = env.splits > 0 ? env.splits : fasp_pick_splits(n_kv_max, n_cols, nsm);

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) S*n_cols*G*fasp_part_floats(D));

    fasp_args a;
    a.Q          = (const char *) Q->data;
    a.q_nb1      = Q->nb[1];
    a.q_nb2      = Q->nb[2];
    a.q_nb3      = Q->nb[3];
    a.K          = (const char *) K->data;
    a.k_nb1      = K->nb[1];
    a.k_nb2      = K->nb[2];
    a.k_nb3      = K->nb[3];
    a.V          = (const char *) V->data;
    a.v_nb1      = V->nb[1];
    a.v_nb2      = V->nb[2];
    a.v_nb3      = V->nb[3];
    a.kv_ne3     = (int) K->ne[3];
    a.mask       = (const half *) mask->data;
    a.m_nb1      = mask->nb[1]/sizeof(half);
    a.m_nb3      = mask->nb[3]/sizeof(half);
    a.m_ne3      = (int) mask->ne[3];
    a.idx        = lists.ptr;
    a.counts     = lists.ptr + (size_t) n_lists*n_kv_max;
    a.n_kv_max   = n_kv_max;
    a.sinks      = sinks ? (const float *) sinks->data : nullptr;
    a.dst        = (float *) dst->data;
    a.part       = part.get();
    a.counter    = ctx.fattn_sparse_pascal.counters;
    a.n_q        = (int) n_q;
    a.n_head     = (int) n_head;
    a.G          = G;
    a.scale_log2 = scale*FASP_LOG2_E;
    a.fault      = env.fault;

    const int n_z = (int) (n_q*ne3);
    if (D == 256) {
        if (K->type == GGML_TYPE_F16) {
            fasp_launch<256, FASP_F16>(a, S, (int) n_head_kv, n_z, stream);
        } else {
            fasp_launch<256, FASP_Q8_0>(a, S, (int) n_head_kv, n_z, stream);
        }
    } else {
        if (K->type == GGML_TYPE_F16) {
            fasp_launch<128, FASP_F16>(a, S, (int) n_head_kv, n_z, stream);
        } else {
            fasp_launch<128, FASP_Q8_0>(a, S, (int) n_head_kv, n_z, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_flash_attn_ext_sparse_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
