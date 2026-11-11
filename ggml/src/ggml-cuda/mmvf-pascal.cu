#include "mmvf-pascal.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvf-pascal-impl.cuh"

// How the launch shape is chosen (G lanes per row group, R rows per group; 4 warps per block):
//   - G from the row width: the smallest of 8, 32 and 128 that leaves each lane at most 16 chunks of 8 weights
//     (K = 320: 8 lanes, 5 chunks each; K = 2,560: 32 lanes; K = 10,240: 128 lanes, 10 chunks each).
//   - R rows per group: 4 for 1 to 4 tokens, 8 for 5 to 8 (each activation load then serves 8 rows, which keeps the
//     activation loads at most 2 per weight load), lowered while the grid has fewer than 4 blocks per SM; if one row
//     per group still leaves too few blocks, G grows so that a few rows are spread over more lanes (K = 2,560 with
//     48 rows: 128 lanes). At 2 tokens and R = 4 a P106 runs the 320- and 10,240-wide projections as fast as at one
//     (R16); at 3 and 4 tokens the earlier rule's R = 2 cost 1.2-1.5x.
//   - 3 and 4 tokens (Q611, from R16's sweep on a P106): at most 32 lanes from the width, and a grid short of blocks
//     gets more lanes before fewer rows. At R = 2 the 10,240-wide rows ran 50.7 / 58.5 us at G = 32 against
//     59.4 / 70.0 at G = 128 (3 / 4 tokens; the 4 warps of a 32-lane block read the same activations), and the
//     2,560 x 48 alpha and beta 4.4 / 4.9 at G = 128, R = 2 against 5.6 / 7.3 at R = 1.
// On the P106-090 (5 SMs) this gives Flash-Next's 320 x 10,240 projections G = 8, R = 4 (160 blocks), 10,240 x 320
// G = 128, R = 4 (80 blocks; at 3-4 tokens G = 32, R = 4, 20 blocks), 10,240 x 4 G = 128, R = 1, 2,560 x 512 G = 32,
// R = 4 and 2,560 x 48 G = 128, R = 1 (at 3-4 tokens R = 2).

#define MMVF_PASCAL_MAX_COLS 8

namespace {

struct mmvfp_params {
    const uint4  * x;
    const float4 * y;
    const float  * bias;
    float        * dst;
    int            nrows;
    int            nchunks;
    int            stride_row;      // in 16-byte chunks
    int            stride_col_y4;   // in float4
    int            stride_col_dst;  // in floats
};

static int mmvfp_env_int(const char * name, const int def) {
    const char * env = getenv(name);
    return env ? atoi(env) : def;
}

static int mmvfp_mode() {
    static const int mode = mmvfp_env_int("GGML_CUDA_MMVF_PASCAL", 1);
    return mode;
}

// GGML_CUDA_MMVF_PASCAL_NSM=n: the SM count the launch shape is chosen for (0: the device's)
static int mmvfp_nsm_override() {
    static const int nsm = mmvfp_env_int("GGML_CUDA_MMVF_PASCAL_NSM", 0);
    return nsm;
}

// GGML_CUDA_MMVF_PASCAL_SHAPE=G,R: a forced launch shape, {0, 0} if unset
static const int * mmvfp_forced_shape() {
    static const int * shape = [] {
        static int gr[2] = { 0, 0 };
        const char * env = getenv("GGML_CUDA_MMVF_PASCAL_SHAPE");
        if (env && sscanf(env, "%d,%d", &gr[0], &gr[1]) == 2) {
            if ((gr[0] != 8 && gr[0] != 32 && gr[0] != 128) || (gr[1] != 1 && gr[1] != 2 && gr[1] != 4 && gr[1] != 8)) {
                GGML_LOG_WARN("%s: GGML_CUDA_MMVF_PASCAL_SHAPE=%s ignored: G must be 8, 32 or 128 and R 1, 2, 4 or 8\n",
                              __func__, env);
                gr[0] = gr[1] = 0;
            }
        }
        return gr;
    }();
    return shape;
}

// GGML_CUDA_MMVF_PASCAL_RULE: the launch-shape rule, 2 (default) as below, 1 without the 3-4 token rule (C84, Q611), 0 also
// with R <= 2 at 3-4 tokens (the first kernel's rule, C54)
static int mmvfp_rule() {
    static const int rule = mmvfp_env_int("GGML_CUDA_MMVF_PASCAL_RULE", 2);
    return rule;
}

// GGML_CUDA_MMVF_PASCAL_EXACT=1 (C750, Q1325): every product of 2 to 4 tokens adds each row in the order of one token.
// Where the 3-4 token rule gives 32 lanes and one token 128 (on a P106-090 the 10,240-wide hyper-connection products),
// mul_mat_vec_bf16_pascal_x4 takes it: the 128-lane order with the 32-lane shape's loads. Where the two shapes' lane
// counts differ otherwise (on cards of more SMs: 128 lanes at 3-4 tokens against 32 at one), the product takes its
// one-token shape. 0 (default): the rule alone, whose 3-4 token shapes may add in another order. A forced shape
// (GGML_CUDA_MMVF_PASCAL_SHAPE) is left as it is.
static bool mmvfp_exact() {
    static const bool on = mmvfp_env_int("GGML_CUDA_MMVF_PASCAL_EXACT", 0) != 0;
    return on;
}

// GGML_CUDA_MMVF_PASCAL_EXACT_FAULT=1 (checks only): mul_mat_vec_bf16_pascal_x4 leaves a quarter of each row's terms out
static bool mmvfp_exact_fault() {
    static const bool on = mmvfp_env_int("GGML_CUDA_MMVF_PASCAL_EXACT_FAULT", 0) != 0;
    return on;
}

// GGML_CUDA_MMVF_PASCAL_MAX_COLS=n: the kernel takes 1 to n tokens (default 8; 4 leaves 5-8 tokens to llama.cpp's
// BF16 path, as before C70)
static int mmvfp_max_cols() {
    static const int n = std::min(std::max(mmvfp_env_int("GGML_CUDA_MMVF_PASCAL_MAX_COLS", MMVF_PASCAL_MAX_COLS), 1),
                                  MMVF_PASCAL_MAX_COLS);
    return n;
}

static int mmvfp_blocks(const int nrows, const int G, const int R) {
    const int rows_per_block = (MMVF_PASCAL_NT / G) * R;
    return (nrows + rows_per_block - 1) / rows_per_block;
}

static void mmvfp_pick_shape(const int nchunks, const int nrows, const int ncols_dst, const int nsm, int & G, int & R) {
    const int rule = mmvfp_rule();
    const int rmax = rule == 0 ? (ncols_dst <= 2 ? 4 : 2) : (ncols_dst <= 4 ? 4 : 8);
    const int * forced = mmvfp_forced_shape();
    if (forced[0] != 0) {
        G = forced[0];
        R = forced[1];
        return;
    }
    const int target = 4 * nsm;
    // 3 and 4 tokens: at most 32 lanes from the width, and a short grid gets more lanes before fewer rows
    const bool mid = rule >= 2 && (ncols_dst == 3 || ncols_dst == 4);
    G = 8;
    while (G < (mid ? 32 : 128) && nchunks > 16 * G) {
        G *= 4;
    }
    R = rmax;
    if (mid) {
        while (G < 128 && mmvfp_blocks(nrows, G, R) < target && nchunks >= 8 * G) {
            G *= 4;
        }
    }
    while (R > 1 && mmvfp_blocks(nrows, G, R) < target) {
        R /= 2;
    }
    while (G < 128 && mmvfp_blocks(nrows, G, R) < target && nchunks >= 8 * G) {
        G *= 4;
    }
}

template <int G, int R, int T, bool has_bias>
static void mmvfp_launch(const mmvfp_params & p, cudaStream_t stream) {
    const dim3 grid(mmvfp_blocks(p.nrows, G, R), 1, 1);
    mul_mat_vec_bf16_pascal<G, R, T, has_bias><<<grid, MMVF_PASCAL_NT, 0, stream>>>(
        p.x, p.y, p.bias, p.dst, p.nrows, p.nchunks, p.stride_row, p.stride_col_y4, p.stride_col_dst);
}

template <int G, int R>
static void mmvfp_launch_cols(const mmvfp_params & p, const int ncols_dst, cudaStream_t stream) {
    if (p.bias) {
        GGML_ASSERT(ncols_dst == 1);
        mmvfp_launch<G, R, 1, true>(p, stream);
        return;
    }
    switch (ncols_dst) {
        case 1: mmvfp_launch<G, R, 1, false>(p, stream); break;
        case 2: mmvfp_launch<G, R, 2, false>(p, stream); break;
        case 3: mmvfp_launch<G, R, 3, false>(p, stream); break;
        case 4: mmvfp_launch<G, R, 4, false>(p, stream); break;
        case 5: mmvfp_launch<G, R, 5, false>(p, stream); break;
        case 6: mmvfp_launch<G, R, 6, false>(p, stream); break;
        case 7: mmvfp_launch<G, R, 7, false>(p, stream); break;
        case 8: mmvfp_launch<G, R, 8, false>(p, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

template <int R, int T>
static void mmvfp_launch_x4(const mmvfp_params & p, cudaStream_t stream) {
    const dim3 grid(mmvfp_blocks(p.nrows, 32, R), 1, 1);
    if (mmvfp_exact_fault()) {
        mul_mat_vec_bf16_pascal_x4<R, T, true><<<grid, MMVF_PASCAL_NT, 0, stream>>>(
            p.x, p.y, p.dst, p.nrows, p.nchunks, p.stride_row, p.stride_col_y4, p.stride_col_dst);
    } else {
        mul_mat_vec_bf16_pascal_x4<R, T, false><<<grid, MMVF_PASCAL_NT, 0, stream>>>(
            p.x, p.y, p.dst, p.nrows, p.nchunks, p.stride_row, p.stride_col_y4, p.stride_col_dst);
    }
}

template <int G>
static void mmvfp_launch_rows(const mmvfp_params & p, const int R, const int ncols_dst, cudaStream_t stream) {
    switch (R) {
        case 1: mmvfp_launch_cols<G, 1>(p, ncols_dst, stream); break;
        case 2: mmvfp_launch_cols<G, 2>(p, ncols_dst, stream); break;
        case 4: mmvfp_launch_cols<G, 4>(p, ncols_dst, stream); break;
        case 8: mmvfp_launch_cols<G, 8>(p, ncols_dst, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

static bool mmvfp_aligned(const void * ptr, const size_t al) {
    return (uintptr_t) ptr % al == 0;
}

} // namespace

bool ggml_cuda_mmvf_pascal_active(const int cc) {
    const int mode = mmvfp_mode();
    return GGML_CUDA_CC_IS_NVIDIA(cc) && (mode == 2 || (mode == 1 && cc == GGML_CUDA_CC_DP4A));
}

bool ggml_cuda_mmvf_pascal_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_cuda_mmvf_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    // 2-D weights with rows of whole 16-byte chunks, 1 to 8 tokens of one channel, float4-aligned activation columns
    return src0->ne[0] % 8 == 0 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src0->nb[0] == ggml_type_size(src0->type) && src0->nb[1] % 16 == 0 && mmvfp_aligned(src0->data, 16) &&
        src0->ne[1] <= INT_MAX && src0->nb[1] / 16 <= INT_MAX &&
        src1->ne[0] == src0->ne[0] && src1->ne[1] >= 1 && src1->ne[1] <= mmvfp_max_cols() &&
        src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && mmvfp_aligned(src1->data, 16) &&
        src1->nb[1] / 16 <= INT_MAX &&
        dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] && dst->nb[0] == sizeof(float) &&
        dst->nb[1] % sizeof(float) == 0 && dst->nb[1] / sizeof(float) <= INT_MAX;
}

bool ggml_cuda_mmvf_pascal(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_cuda_mm_fusion_args_host * fusion,
                           ggml_tensor * dst, cudaStream_t stream) {
    if (!ggml_cuda_mmvf_pascal_supports(src0, src1, dst)) {
        return false;
    }
    const float * bias = nullptr;
    if (fusion) {
        if (fusion->gate || fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up ||
                fusion->shared_gate || fusion->shared_dst) {
            return false;
        }
        if (fusion->x_bias) {
            const ggml_tensor * b = fusion->x_bias;
            if (b->type != GGML_TYPE_F32 || b->ne[0] != dst->ne[0] || dst->ne[1] != 1 || !ggml_is_contiguous(b)) {
                return false;
            }
            bias = (const float *) b->data;
        }
    }

    const int64_t ncols_dst = src1->ne[1];
    mmvfp_params p;
    p.x              = (const uint4  *) src0->data;
    p.y              = (const float4 *) src1->data;
    p.bias           = bias;
    p.dst            = (float *) dst->data;
    p.nrows          = (int) src0->ne[1];
    p.nchunks        = (int) (src0->ne[0] / 8);
    p.stride_row     = (int) (src0->nb[1] / 16);
    p.stride_col_y4  = (int) (src1->nb[1] / 16);
    p.stride_col_dst = (int) (dst->nb[1] / sizeof(float));

    const int nsm = mmvfp_nsm_override() > 0 ? mmvfp_nsm_override() : ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int G = 0;
    int R = 0;
    mmvfp_pick_shape(p.nchunks, p.nrows, (int) ncols_dst, nsm, G, R);
    if (mmvfp_exact() && ncols_dst >= 2 && ncols_dst <= 4 && mmvfp_forced_shape()[0] == 0) {
        int G1 = 0;
        int R1 = 0;
        mmvfp_pick_shape(p.nchunks, p.nrows, 1, nsm, G1, R1);
        // the rule gives this kernel R = 4 (the 32 lanes are kept only for a grid of enough 4-row blocks)
        if (G1 == 128 && G == 32 && R == 4 && (ncols_dst == 3 || ncols_dst == 4)) {
            if (ncols_dst == 3) {
                mmvfp_launch_x4<4, 3>(p, stream);
            } else {
                mmvfp_launch_x4<4, 4>(p, stream);
            }
            return true;
        }
        if (G1 != G) {
            G = G1;
            R = R1;
        }
    }
    switch (G) {
        case   8: mmvfp_launch_rows<  8>(p, R, (int) ncols_dst, stream); break;
        case  32: mmvfp_launch_rows< 32>(p, R, (int) ncols_dst, stream); break;
        case 128: mmvfp_launch_rows<128>(p, R, (int) ncols_dst, stream); break;
        default:  GGML_ABORT("fatal error");
    }
    return true;
}

#else

bool ggml_cuda_mmvf_pascal_active(const int cc) {
    GGML_UNUSED(cc);
    return false;
}

bool ggml_cuda_mmvf_pascal_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(src0, src1, dst);
    return false;
}

bool ggml_cuda_mmvf_pascal(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_cuda_mm_fusion_args_host * fusion,
                           ggml_tensor * dst, cudaStream_t stream) {
    GGML_UNUSED_VARS(src0, src1, fusion, dst, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
