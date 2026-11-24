#include "moe-tail.cuh"

// one thread per token: the token's total, then the cut slot by slot (the fused router kernel does the same per lane)
template <bool out_ids>
static __global__ void k_moe_tail(const float * __restrict__ weights, const int32_t * __restrict__ ids,
                                  void * __restrict__ dst, const int k, const int n_tokens, const int64_t ids_stride,
                                  const int64_t dst_stride, const float p, const float w, const bool renorm,
                                  const bool skip, const bool repeat) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_tokens) {
        return;
    }
    const float * wt = weights + (int64_t) i*k;

    double sum = 0.0;
    for (int j = 0; j < k; ++j) {
        sum = __dadd_rn(sum, (double) wt[j]);
    }
    const float total = __double2float_rn(sum);

    if constexpr (out_ids) {
        const int32_t * id = ids + i*ids_stride;
        int32_t       * od = (int32_t *) dst + i*dst_stride;
        int32_t last = id[0];
        float   cum  = 0.0f;
        for (int j = 0; j < k; ++j) {
            if (ggml_cuda_moe_tail_keep(wt[j], total, cum, j, p, w)) {
                last = id[j];
            }
        }
        cum = 0.0f;
        for (int j = 0; j < k; ++j) {
            od[j] = ggml_cuda_moe_tail_keep(wt[j], total, cum, j, p, w) || !skip ? id[j] : repeat ? last : -1;
        }
    } else {
        float * ow = (float *) dst + (int64_t) i*k;
        double kept_sum = 0.0;
        float  cum      = 0.0f;
        for (int j = 0; j < k; ++j) {
            ow[j]    = ggml_cuda_moe_tail_keep(wt[j], total, cum, j, p, w) ? wt[j] : 0.0f;
            kept_sum = __dadd_rn(kept_sum, (double) ow[j]);
        }
        if (renorm) {
            const float f = __fdiv_rn(total, __double2float_rn(kept_sum));
            for (int j = 0; j < k; ++j) {
                ow[j] = __fmul_rn(ow[j], f);
            }
        }
    }
    GGML_UNUSED_VARS(ids, ids_stride, renorm, skip, repeat);
}

void ggml_cuda_op_moe_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * ids     = dst->src[1];

    GGML_ASSERT(weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights));
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int   k        = ids->ne[0];
    const int   n_tokens = ids->ne[1];
    const float p        = ggml_get_op_params_f32(dst, 0);
    const float w        = ggml_get_op_params_f32(dst, 1);
    const bool  renorm   = ggml_get_op_params_i32(dst, 2) != 0;
    const bool  skip     = n_tokens <= ggml_get_op_params_i32(dst, 3);
    const bool  repeat   = ggml_get_op_params_i32(dst, 4) != 0;

    if (n_tokens == 0) {
        return;
    }

    const int   block = 128;
    const dim3  grid((n_tokens + block - 1)/block);
    const auto  ids_stride = ids->nb[1]/sizeof(int32_t);
    const float * w_d = (const float *) weights->data;
    const int32_t * ids_d = (const int32_t *) ids->data;

    if (dst->type == GGML_TYPE_I32) {
        k_moe_tail<true><<<grid, block, 0, ctx.stream()>>>(w_d, ids_d, dst->data, k, n_tokens, ids_stride, k, p, w, renorm, skip, repeat);
    } else {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        k_moe_tail<false><<<grid, block, 0, ctx.stream()>>>(w_d, ids_d, dst->data, k, n_tokens, ids_stride, k, p, w, renorm, skip, repeat);
    }
    CUDA_CHECK(cudaGetLastError());
}

// GGML_CUDA_SKIP_IDS_ANY=0: only GGML_OP_MOE_TAIL's ids get the skip handling, every other ids tensor is read as it is, as
// upstream (default 1: ids of any producer but ARGSORT and TOP_K, which name valid experts only, may hold -1)
static bool ggml_cuda_skip_ids_any() {
    static const bool any = [] {
        const char * env = getenv("GGML_CUDA_SKIP_IDS_ANY");
        return env == nullptr || atoi(env) != 0;
    }();
    return any;
}

ggml_cuda_skip_mode ggml_cuda_ids_skip_mode(const ggml_tensor * ids) {
    const ggml_tensor * t = ids;
    while (t->view_src) {
        t = t->view_src;
    }
    switch (t->op) {
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return GGML_CUDA_SKIP_NONE;
        case GGML_OP_MOE_TAIL:
            if (t->type != GGML_TYPE_I32 || t->ne[1] > ggml_get_op_params_i32(t, 3)) {
                return GGML_CUDA_SKIP_NONE; // every slot keeps its own expert
            }
            return ggml_get_op_params_i32(t, 4) != 0 ? GGML_CUDA_SKIP_REPEAT : GGML_CUDA_SKIP_NEG;
        default:
            return ggml_cuda_skip_ids_any() ? GGML_CUDA_SKIP_NEG : GGML_CUDA_SKIP_NONE;
    }
}

// one thread per token: a slot of id -1 (and with dedup a repeat of an expert named before it) gets the next expert
// that no slot of the token names, so that every kernel reads a valid expert and no token names one twice (llama.cpp's
// MMQ reads past its buffers for a repeated expert, Q875)
static __global__ void k_skip_ids_fix(const char * __restrict__ ids, char * __restrict__ fixed, const int ne0,
                                      const int ne1, const int64_t nb1, const int n_expert, const bool dedup) {
    const int i1 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i1 >= ne1) {
        return;
    }
    const int32_t * row = (const int32_t *) (ids + i1*nb1);
    int32_t       * out = (int32_t *) (fixed + i1*nb1);
    int next = 0;
    for (int j = 0; j < ne0; ++j) {
        const int32_t e = row[j];
        bool bad = e < 0;
        for (int j2 = 0; dedup && !bad && j2 < j; ++j2) {
            bad = row[j2] == e;
        }
        if (!bad) {
            out[j] = e;
            continue;
        }
        for (;; ++next) {
            bool named = false;
            for (int j2 = 0; j2 < ne0; ++j2) {
                named = named || row[j2] == next;
            }
            if (!named) {
                break;
            }
        }
        out[j] = min(next++, n_expert - 1);
    }
}

// grid (slots, min(tokens, 65535)): the dst row of a slot whose id is -1 becomes zeros. A grid's y size is at most
// 65,535, so the y blocks stride over the tokens (one pass up to 65,535 tokens)
static __global__ void k_skip_ids_zero_rows(const char * __restrict__ ids, float * __restrict__ dst, const int64_t ids_nb1,
                                            const int64_t ne0, const int64_t s1, const int64_t s2, const int64_t n_tokens) {
    for (int64_t t = blockIdx.y; t < n_tokens; t += gridDim.y) {
        const int32_t id = *(const int32_t *) (ids + t*ids_nb1 + blockIdx.x*sizeof(int32_t));
        if (id >= 0) {
            continue;
        }
        float * row = dst + blockIdx.x*s1 + t*s2;
        for (int64_t i = threadIdx.x; i < ne0; i += blockDim.x) {
            row[i] = 0.0f;
        }
    }
}

ggml_cuda_skip_ids::ggml_cuda_skip_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids, const int64_t n_expert)
        : orig(ids) {
    const ggml_cuda_skip_mode mode = ids ? ggml_cuda_ids_skip_mode(ids) : GGML_CUDA_SKIP_NONE;
    if (mode == GGML_CUDA_SKIP_NONE || ids->ne[0] == 0 || ids->ne[1] == 0) {
        return;
    }
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(n_expert >= ids->ne[0]);
    fixed      = *ids;
    fixed.data = buf.alloc(ctx.pool(), ids->ne[1]*ids->nb[1]);
    const int ne1 = ids->ne[1];
    k_skip_ids_fix<<<(ne1 + 127)/128, 128, 0, ctx.stream()>>>((const char *) ids->data, (char *) fixed.data, ids->ne[0],
        ne1, ids->nb[1], (int) n_expert, mode == GGML_CUDA_SKIP_REPEAT);
    CUDA_CHECK(cudaGetLastError());
    active = true;
}

void ggml_cuda_ids_zero_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * orig, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[1] == orig->ne[0] && dst->ne[2] == orig->ne[1]);
    const dim3 grid(orig->ne[0], (unsigned int) std::min<int64_t>(orig->ne[1], 65535));
    k_skip_ids_zero_rows<<<grid, 128, 0, ctx.stream()>>>((const char *) orig->data, (float *) dst->data, orig->nb[1],
        dst->ne[0], dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), orig->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_skip_ids::zero_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) const {
    if (active) {
        ggml_cuda_ids_zero_rows(ctx, orig, dst);
    }
}
