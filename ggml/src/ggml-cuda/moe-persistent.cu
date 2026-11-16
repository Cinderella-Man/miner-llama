#include "moe-persistent.cuh"
#include "mmvq-pascal.cuh"
#include "mmvq-pascal-impl.cuh"
#include "unary.cuh"
#include "keepalive.cuh"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Layer shapes the kernel is built for (Flint).
#define MOEP_H   2048 // hidden size
#define MOEP_NE  256  // routed experts
#define MOEP_KU  6    // experts used per token
#define MOEP_F   512  // routed expert width
#define MOEP_FS  1024 // shared expert width

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

constexpr int H  = MOEP_H;
constexpr int NE = MOEP_NE;
constexpr int KU = MOEP_KU;
constexpr int F  = MOEP_F;
constexpr int FS = MOEP_FS;

constexpr int PK_THREADS = 256; // threads per block
constexpr int PK_NW      = PK_THREADS / WARP_SIZE;

// Work items: router rows (2 per item) | shared gate/up (4 rows) | routed gate/up (4 rows of one expert slot, slots in
// turn) | shared down (CR rows) | routed down (CR rows of one expert slot, by row group: g0 e0..e5, g1 e0..e5, ...).
// Every dependency points to an item with a smaller number. An output group has CR = 8 rows, or 16 where a down
// product's warp pass covers 16 rows (K-quants, one super-block per lane).
constexpr int RT_ROWS = 2;
constexpr int SG_ROWS = 4;
constexpr int EG_ROWS = 4;
constexpr int CR_MIN  = 8;
constexpr int N_RT    = NE / RT_ROWS;
constexpr int N_SG    = FS / SG_ROWS;
constexpr int N_EG1   = F / EG_ROWS;  // routed gate/up items per expert slot
constexpr int N_EG    = KU * N_EG1;
constexpr int I_RT    = 0;
constexpr int I_SG    = I_RT + N_RT;
constexpr int I_EG    = I_SG + N_SG;
constexpr int I_SD    = I_EG + N_EG;

template <int CR>
struct moep_items {
    static constexpr int N_GRP   = H / CR;
    static constexpr int N_SD    = N_GRP;
    static constexpr int N_ED    = KU * N_GRP;
    static constexpr int I_ED    = I_SD + N_SD;
    static constexpr int N_ITEMS = I_ED + N_ED;
};

// Counters and flags at the start of the context's buffer (32-bit words).
constexpr int S_NEXT    = 0;             // next work item
constexpr int S_RT_DONE = 1;             // router items done
constexpr int S_TOPK    = 2;             // top-k published
constexpr int S_SG_DONE = 3;             // shared gate/up items done
constexpr int S_SACT    = 4;             // shared activations quantized
constexpr int S_EG_DONE = 5;             // + e: routed gate/up items of expert slot e done
constexpr int S_EACT    = S_EG_DONE + KU; // + e: activations of expert slot e quantized
constexpr int S_EXIT    = S_EACT + KU;   // warps that have left
constexpr int S_ERR     = S_EXIT + 1;    // a wait timed out (never reset by the kernel)
constexpr int S_COMB    = 32;            // + g: contributions to output group g
constexpr int N_SYNC    = S_COMB + H / CR_MIN;

// The scratch of one layer after the counters, in bytes from the start of the buffer (16-byte aligned).
struct moep_scratch {
    static constexpr size_t logits  = (N_SYNC * sizeof(unsigned) + 15) / 16 * 16;  // float [NE]
    static constexpr size_t ids     = logits  + NE * sizeof(float);                // int32 [NE]
    static constexpr size_t weights = ids     + NE * sizeof(int32_t);              // float [KU], padded
    static constexpr size_t act     = weights + 16 * sizeof(float);                // float [KU][F]
    static constexpr size_t aq      = act     + KU * F * sizeof(float);            // block_q8_1 [KU][F/32]
    static constexpr size_t dexp    = aq      + KU * F / QK8_1 * sizeof(block_q8_1); // float [KU][H]
    static constexpr size_t sact    = dexp    + KU * H * sizeof(float);            // float [FS]
    static constexpr size_t sq      = sact    + FS * sizeof(float);                // block_q8_1 [FS/32]
    static constexpr size_t sd      = sq      + FS / QK8_1 * sizeof(block_q8_1);   // float [H]
    static constexpr size_t check   = sd      + H * sizeof(float);                 // float [H]: check mode output
    static constexpr size_t moe     = check   + H * sizeof(float);                 // float [H]: check mode routed sum
    static constexpr size_t size    = moe     + H * sizeof(float);
};

// F16_HACC: F16 weights accumulated in half2, as mul_mat_vec_f does where FP16 is fast (not on a compute 6.1 build);
// F16_PASCAL: in the Pascal F16 product's order, where the per-op path runs that kernel (ggml_cuda_mmvf_pascal_f16)
enum moep_router_type { MOEP_ROUTER_F32 = 0, MOEP_ROUTER_F16 = 1, MOEP_ROUTER_BF16 = 2, MOEP_ROUTER_F16_HACC = 3,
                        MOEP_ROUTER_F16_PASCAL = 4 };

struct moep_args {
    const float * x;          // layer input (residual stream)
    const float * norm_w;
    const void  * router;
    const char  * gate_exps;
    const char  * up_exps;
    const char  * down_exps;
    const char  * gate_sh;
    const char  * up_sh;
    const char  * down_sh;
    float       * out;        // ffn_out
    float       * moe;        // check mode: the routed experts' weighted sum, else nullptr
    char        * buf;        // counters, then scratch
    unsigned    * err;        // mapped host word: a wait timed out
    int           router_type;
    int           ncols;      // H, as the norm kernel receives it (its division by a runtime value under fast math)
    float         eps;
    float         clamp_val;
    float         scale_val;
    int           spin_limit; // polls of a flag before giving up
    int           fault;      // testing (GGML_CUDA_PERSISTENT_MOE_FAULT): 1 never publishes the top-k, so every wait on
                              // it times out; 2 scales the routing weights by 1.0001 (host side); 3 leaves the shared
                              // expert out of ffn_out
};

// ---- the per-op path's arithmetic, as device functions ------------------------------------------------------------
// norm.cu (add_rms_norm_mul_q8_1_f32 and rms_norm_f32 for one row of 2048: 512 threads with one float4 each),
// mmvf.cu (mul_mat_vec_f, 256-thread blocks for K = 2048), topk-moe.cu (topk_moe_cuda with sigmoid and norm, no bias,
// no groups), quantize.cu (quantize_q8_1), moe-weighted-reduction.cu, and the Pascal products of mmvq-pascal.cu
// (mmvq-pascal-impl.cuh). Only the code these paths execute is kept, unchanged in arithmetic and order.

// one thread's partial sum of squares over its float4
static __device__ __forceinline__ float moep_rms_partial(const float4 xs) {
    float tmp = 0.0f;
    tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    return tmp;
}

// one normed value: scale * x, then times the norm weight (two roundings, as the norm kernels do per component)
static __device__ __forceinline__ float moep_rms_apply(const float scale, const float x, const float m) {
    float r = scale*x;
    r *= m;
    return r;
}

// One thread's partial dot product of a router row (T = float, half or nv_bfloat16) with the normed input, which xn2(c)
// returns as a float2 for columns 2c and 2c + 1; columns c = tid, tid + 256, ... (mul_mat_vec_f: FP32 accumulation, or
// half2 accumulation of F16 weights with hacc).
template <typename T, bool hacc, typename XN2>
static __device__ __forceinline__ float moep_router_partial(const T * x, const XN2 & xn2, const int tid) {
    constexpr int ncols2 = H / 2;
    float sumf = 0.0f;
    if constexpr (hacc) {
#ifdef FP16_AVAILABLE
        const half2 * x2 = (const half2 *) x;
        half2 sumh2 = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int col2 = tid; col2 < ncols2; col2 += 256) {
            const half2  tmpx = __ldg(x2 + col2);
            const float2 tmpy = xn2(col2);
            sumh2 += tmpx * make_half2(tmpy.x, tmpy.y);
        }
        sumf = __low2float(sumh2) + __high2float(sumh2);
#else
        NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
    } else if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
#pragma unroll
        for (int col2 = tid; col2 < ncols2; col2 += 256) {
            const float2 tmpx = __ldg(x2 + col2);
            const float2 tmpy = xn2(col2);
            ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
#pragma unroll
        for (int col2 = tid; col2 < ncols2; col2 += 256) {
            const float2 tmpx = __half22float2(__ldg(x2 + col2));
            const float2 tmpy = xn2(col2);
            ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
        }
    } else {
        const unsigned int * x2 = (const unsigned int *) x;
#pragma unroll
        for (int col2 = tid; col2 < ncols2; col2 += 256) {
            const unsigned int w = __ldg(x2 + col2);
            const float tmpx0 = __uint_as_float(w << 16);          // nv_bfloat16 -> float
            const float tmpx1 = __uint_as_float(w & 0xFFFF0000u);
            const float2 tmpy = xn2(col2);
            ggml_cuda_mad(sumf, tmpx0, tmpy.x);
            ggml_cuda_mad(sumf, tmpx1, tmpy.y);
        }
    }
    return sumf;
}

// One router row computed by one warp standing in for mul_mat_vec_f's 256-thread block: each lane takes the columns of
// eight of the block's threads, then the same two butterfly reductions (per warp, then over the eight warp sums).
template <typename T, bool hacc, typename XN2>
static __device__ __forceinline__ float moep_router_row(const T * xr, const XN2 & xn2, const int lane) {
    float s[8];
#pragma unroll
    for (int vw = 0; vw < 8; ++vw) {
        s[vw] = moep_router_partial<T, hacc>(xr, xn2, vw * WARP_SIZE + lane);
    }
#pragma unroll
    for (int vw = 0; vw < 8; ++vw) {
        s[vw] = warp_reduce_sum(s[vw]);
    }
    float sumf = 0.0f; // buf_iw[lane]: the 8 warp sums, zero beyond
#pragma unroll
    for (int vw = 0; vw < 8; ++vw) {
        if (lane == vw) {
            sumf = s[vw];
        }
    }
    return warp_reduce_sum(sumf);
}

// One F16 router row as the Pascal F16 product (mul_mat_vec_f16_pascal, mmvq-pascal.cu) sums it when the per-op path
// runs it: lane l takes the 16-byte chunks l, l + 32, ... of the row (32 lanes per row at H = 2048), each with
// mmvqp_f16_dot8, then the xor-shuffle reduction.
template <typename XN2>
static __device__ __forceinline__ float moep_router_row_pascal_f16(const half * xr, const XN2 & xn2, const int lane) {
    static_assert(H % (8 * WARP_SIZE) == 0, "32 lanes per row");
    const uint4 * x16 = (const uint4 *) xr;
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < H / (8 * WARP_SIZE); ++it) {
        const int    c  = it * WARP_SIZE + lane; // columns 8c .. 8c + 7
        const uint4  w  = __ldg(x16 + c);
        const float2 p0 = xn2(4*c + 0);
        const float2 p1 = xn2(4*c + 1);
        const float2 p2 = xn2(4*c + 2);
        const float2 p3 = xn2(4*c + 3);
        acc = mmvqp_f16_dot8(w, make_float4(p0.x, p0.y, p1.x, p1.y), make_float4(p2.x, p2.y, p3.x, p3.y), acc);
    }
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off, WARP_SIZE);
    }
    return acc;
}

// topk_moe_cuda for one token with sigmoid gating and weight normalization: wt[i] holds the logit of expert
// lane + 32 i; writes ids[0..KU) and weights[0..KU).
static __device__ __forceinline__ void moep_topk(float (&wt)[NE / WARP_SIZE], float * weights, int32_t * ids,
                                                 const float clamp_val, const float scale_val, const int lane) {
    constexpr int experts_per_thread = NE / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = 1.f / (1.f + expf(-wt[i]));
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < KU; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if (wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            wt_sum += max_val;
        }
    }

    wt_sum              = warp_reduce_sum(wt_sum);
    wt_sum              = max(wt_sum, clamp_val);
    const float inv_sum = 1.0f / wt_sum;
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] *= inv_sum;
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < KU) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// quantize_q8_1: lane iqs of a warp holds value iqs of block ib (global or shared memory); the block sum is d times the
// sum of the int8 codes (#31247)
static __device__ __forceinline__ void moep_quantize_q8_1(const float xi, block_q8_1 * y, const int ib, const int iqs) {
    float amax = fabsf(xi);
    amax = warp_reduce_max(amax);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int  sum = warp_reduce_sum((int) q);
    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    y[ib].ds = make_half2(d, d * sum);
}

// The rows of one warp of mul_mat_vec_q_pascal, as the per-op kernel computes them, with staged weight loads. The lane
// group grp (lpr lanes) owns rows row0 .. row0 + R - 1 of matrix x0 (and of x1 when M == 2: up in x0, gate in x1), with
// row0 = ((warp_index << (5 - lpr_log2)) + grp) * R; y is this channel's Q8_1 activations, read as src says. On return
// every lane of a group holds its rows' sums in acc; returns row0. SEQ (M == 2): the up rows are staged and multiplied,
// then the gate rows through the same buffer; each row's sum is the same.
template <ggml_type type, int R, int C, int M, int src, int LPR, int NIT, int RB, bool SEQ = false>
static __device__ __forceinline__ int moep_rows(
        const char * x0, const char * x1, const int nrows, const char * y, const int warp_index, const int lane,
        char * smw_warp, float (&acc)[M][R]) {
    constexpr int BS = mmvqp_traits<type>::bs;
    constexpr int WB = C * BS;                     // weight bytes per lane and iteration
    constexpr int NA = C * mmvqp_traits<type>::na; // activation blocks per lane and iteration
    constexpr int YB = NA * sizeof(block_q8_1);    // activation bytes per lane and iteration
    constexpr int NU = (WB + 15) / 16;             // 16-byte units per lane and row, the last one predicated
    constexpr int lpr = 1 << LPR;
    static_assert((WB << LPR) % 16 == 0, "a staged segment must be a multiple of 16 bytes");

    const int sl   = lane & (lpr - 1);
    const int grp  = lane >> LPR;
    const int row0 = ((warp_index << (5 - LPR)) + grp) * R;

    const char * rw[M][R];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            rw[m][i] = (m == 0 ? x0 : x1) + (int64_t) min(row0 + i, nrows - 1) * RB;
        }
    }
    const char * yp = y + sl * YB;
    constexpr int wstep     = WB << LPR;
    constexpr int ystep     = YB << LPR;
    constexpr int seg_units = (WB << LPR) >> 4;
    constexpr int MS = SEQ ? 1 : M;                // matrices staged at once
    char * smw = smw_warp + (grp * R * MS * WB << LPR);

#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[m][i] = 0.0f;
        }
    }

    // stage the rows of matrices m0 .. m0 + MS - 1
    const auto stage = [&](const int m0) {
#pragma unroll
        for (int ms = 0; ms < MS; ++ms) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                uint4       * dsm  = (uint4 *) (smw + ((ms*R + i) * WB << LPR));
                const uint4 * srcw = (const uint4 *) rw[m0 + ms][i];
#pragma unroll
                for (int j = 0; j < NU; ++j) {
                    const int u = sl + (j << LPR);
                    if (j < NU - 1 || u < seg_units) {
                        dsm[u] = __ldg(srcw + u);
                    }
                }
            }
        }
        __syncwarp();
    };

    for (int it = 0; it < NIT; ++it) {
        stage(0);
        mmvqp_act<NA> a;
        mmvqp_load_act<type, NA, src>(a, yp);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            if (SEQ && m > 0) {
                __syncwarp();
                stage(m);
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const char * ws = smw + (((SEQ ? 0 : m)*R + i) * WB << LPR) + sl * WB;
                if constexpr (type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS) {
                    // a whole super-block per lane: its words are read from shared memory where they are used, instead
                    // of 44 or 34 registers holding them at once (the arithmetic is the same)
                    acc[m][i] = mmvqp_dot<type, C>((const uint32_t *) ws, a, acc[m][i]);
                } else {
                    uint32_t w[WB / 4];
                    mmvqp_lds_words<WB, mmvqp_gcd(WB, 16)>(w, ws);
                    acc[m][i] = mmvqp_dot<type, C>(w, a, acc[m][i]);
                }
                rw[m][i] += wstep;
            }
        }
        __syncwarp();
        yp += ystep;
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            if (off < lpr) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    acc[m][i] += __shfl_xor_sync(0xFFFFFFFF, acc[m][i], off, WARP_SIZE);
                }
            }
        }
    }
    return row0;
}

// The per-op kernel's epilogue for one lane: the lane with sl == i % lpr writes row row0 + i, plain or up * silu(gate).
template <int R, int M, int LPR>
static __device__ __forceinline__ void moep_store(const float (&acc)[M][R], const int row0, const int lane, const int nrows,
                                                  float * dst) {
    const int sl = lane & ((1 << LPR) - 1);
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int row = row0 + i;
        if ((i & ((1 << LPR) - 1)) != sl || row >= nrows) {
            continue;
        }
        float result = acc[0][i];
        if constexpr (M == 2) {
            const float gate_value = acc[M - 1][i];
            result *= ggml_cuda_op_silu_single(gate_value);
        }
        dst[row] = result;
    }
}

// Launch shape of one product inside the persistent kernel. C (blocks per lane) is mmvqp_choose's, so lanes per row and
// every row's summation order are those of the per-op kernel; R may differ (it does not change a row's sum).
constexpr __host__ __device__ int moep_lpr_log2(const int K, const int na, const int C) {
    const int chunks = K / (32 * na) / C;
    int l = 0;
    while (l < 5 && chunks % (2 << l) == 0) {
        l++;
    }
    return l;
}

constexpr __host__ __device__ int moep_max(const int a, const int b) {
    return a > b ? a : b;
}

constexpr __host__ __device__ int moep_min(const int a, const int b) {
    return a < b ? a : b;
}

// The per-op kernel's shape for a product of K-wide rows; RPP1: rows per warp pass with one row per lane group.
template <ggml_type type, bool gate, int K>
struct moep_shape {
    static constexpr mmvqp_shape S    = mmvqp_choose(type, gate);
    static constexpr int         LPR  = moep_lpr_log2(K, mmvqp_traits<type>::na, S.C);
    static constexpr int         RPP1 = 32 >> LPR;
};

// One product inside the persistent kernel: R rows per lane group, SEQ: a fused gate/up stages the up rows and then
// the gate rows through one buffer (where both at once would leave too little shared memory for two blocks per SM).
template <ggml_type type, bool gate, int K, int R_WANT, bool SEQ = false>
struct moep_prod {
    static constexpr int C   = moep_shape<type, gate, K>::S.C;
    static constexpr int NA  = mmvqp_traits<type>::na;
    static constexpr int LPR = moep_lpr_log2(K, NA, C);
    static constexpr int NIT = ((K / (32 * NA)) / C) >> LPR;
    static constexpr int RB  = K / (32 * NA) * mmvqp_traits<type>::bs; // row bytes
    static constexpr int R   = R_WANT;
    static constexpr int M   = gate ? 2 : 1;
    static constexpr int ROWS_PER_PASS = (32 >> LPR) * R;
    static constexpr int STAGE = 32 * R * (SEQ ? 1 : M) * C * mmvqp_traits<type>::bs; // staging bytes per warp
};

// The kernel's configuration for a (gate/up, down, shared) type combination: rows per output group (CR), the down
// products' R (the per-op kernel's, limited to one output group per pass), the routed gate/up rows per lane group
// (RG: 2 where the staging still fits two blocks per SM, which shares each activation load between two rows) and
// sequential gate/up staging where needed. C, and with it every row's summation order, is always the per-op kernel's.
template <ggml_type TGU, ggml_type TDN, ggml_type TSH>
struct moep_cfg {
    static constexpr int CR    = moep_max(CR_MIN, moep_max(moep_shape<TDN, false, F>::RPP1, moep_shape<TSH, false, FS>::RPP1));
    static constexpr int R_SD  = moep_min(moep_shape<TSH, false, FS>::S.R, CR / moep_shape<TSH, false, FS>::RPP1);
    static constexpr int R_ED  = moep_min(moep_shape<TDN, false, F>::S.R,  CR / moep_shape<TDN, false, F>::RPP1);
    static constexpr int SMEM  = 48 * 1024 - H / QK8_1 * (int) sizeof(block_q8_1) - 32 * (int) sizeof(float); // staging
    static constexpr int DOWN  = moep_max(moep_prod<TSH, false, FS, R_SD>::STAGE, moep_prod<TDN, false, F, R_ED>::STAGE);
    static constexpr bool SEQ_SG = PK_NW * moep_max(DOWN, moep_prod<TSH, true, H, 1>::STAGE) > SMEM;
    static constexpr int OTHER = moep_max(DOWN, moep_prod<TSH, true, H, 1, SEQ_SG>::STAGE);
    static constexpr bool RG2  = moep_shape<TGU, true, H>::LPR == 5 && PK_NW * moep_max(OTHER, moep_prod<TGU, true, H, 2>::STAGE) <= SMEM;
    static constexpr int RG    = RG2 ? 2 : 1;
    static constexpr bool SEQ_GU = !RG2 && PK_NW * moep_max(OTHER, moep_prod<TGU, true, H, 1>::STAGE) > SMEM;
};

static __device__ __forceinline__ unsigned moep_load_volatile(const unsigned * p) {
    return *(const volatile unsigned *) p;
}

// the next work item for the whole warp
static __device__ __forceinline__ unsigned moep_fetch(unsigned * sync, const int lane) {
    unsigned i = 0;
    if (lane == 0) {
        i = atomicAdd(&sync[S_NEXT], 1u);
    }
    return __shfl_sync(0xffffffff, i, 0);
}

// Counts this warp's finished item into *ctr (after its results are stored). True in all lanes for the warp that
// brings the count to n; that warp then sees every other contributor's results (read them with __ldcg).
static __device__ __forceinline__ bool moep_arrive(unsigned * ctr, const unsigned n, const int lane) {
    __threadfence();
    __syncwarp();
    unsigned old = 0;
    if (lane == 0) {
        old = atomicAdd(ctr, 1u);
    }
    old = __shfl_sync(0xffffffff, old, 0);
    __syncwarp();
    const bool last = old == n - 1;
    if (last) {
        __threadfence();
    }
    return last;
}

static __device__ __forceinline__ void moep_publish(unsigned * flag, const int lane) {
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomicExch(flag, 1u);
    }
}

// Waits until *flag is set. False (in all lanes) if the wait timed out or another warp reported a timeout.
static __device__ __forceinline__ bool moep_wait(const unsigned * flag, unsigned * sync, unsigned * err, const int spin_limit,
                                                 const int lane) {
    unsigned ok = 1;
    if (lane == 0) {
        int n = 0;
        while (moep_load_volatile(flag) == 0) {
            ++n;
            if (n >= spin_limit || ((n & 63) == 0 && moep_load_volatile(&sync[S_ERR]) != 0)) {
                atomicExch(&sync[S_ERR], 1u);
                *(volatile unsigned *) err = 1;
                ok = 0;
                break;
            }
        }
    }
    ok = __shfl_sync(0xffffffff, ok, 0);
    __syncwarp();
    __threadfence();
    return ok != 0;
}

// The persistent kernel for one layer. TGU, TDN, TSH: routed gate/up, routed down and shared expert types; RG: routed
// gate/up rows per warp pass (2 where the staging buffers still allow two blocks per SM).
template <ggml_type TGU, ggml_type TDN, ggml_type TSH, int MINB>
__launch_bounds__(PK_THREADS, MINB)
static __global__ void k_moe_layer(const moep_args a) {
    using CFG = moep_cfg<TGU, TDN, TSH>;
    constexpr int CR = CFG::CR;
    constexpr int RG = CFG::RG;
    using IT = moep_items<CR>;
    using GU = moep_prod<TGU, true,  H,  RG, CFG::SEQ_GU>;
    using SG = moep_prod<TSH, true,  H,  1,  CFG::SEQ_SG>;
    using SD = moep_prod<TSH, false, FS, CFG::R_SD>;
    using ED = moep_prod<TDN, false, F,  CFG::R_ED>;
    static_assert(EG_ROWS % GU::ROWS_PER_PASS == 0 && SG_ROWS % SG::ROWS_PER_PASS == 0 &&
                  CR % SD::ROWS_PER_PASS == 0 && CR % ED::ROWS_PER_PASS == 0 && CR <= WARP_SIZE,
                  "items must cover whole passes and output groups");
    constexpr int STAGE = moep_max(moep_max(GU::STAGE, SG::STAGE), moep_max(SD::STAGE, ED::STAGE));

    __shared__ __align__(16) block_q8_1 s_yq[H / QK8_1];
    __shared__ float                    s_red[32];
    __shared__ __align__(16) char       s_stage[PK_NW][STAGE];

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    unsigned   * sync    = (unsigned *) a.buf;
    float      * logits  = (float *) (a.buf + moep_scratch::logits);
    int32_t    * ids     = (int32_t *) (a.buf + moep_scratch::ids);
    float      * weights = (float *) (a.buf + moep_scratch::weights);
    float      * act     = (float *) (a.buf + moep_scratch::act);
    block_q8_1 * aq      = (block_q8_1 *) (a.buf + moep_scratch::aq);
    float      * dexp    = (float *) (a.buf + moep_scratch::dexp);
    float      * sact    = (float *) (a.buf + moep_scratch::sact);
    block_q8_1 * sq      = (block_q8_1 *) (a.buf + moep_scratch::sq);
    float      * sd      = (float *) (a.buf + moep_scratch::sd);

    // ---- RMS norm and Q8_1 copy of the input, in every block ----------------------------------------------------------
    // The norm kernel's 512 threads: thread tid stands for threads tid and tid + 256. The normed values are not kept:
    // the quantization and the router recompute each one with the same two multiplies.
    float scale;
    {
        constexpr int VPT = 512 / PK_THREADS;
        constexpr int NQ  = H / QK8_1 / PK_NW;  // Q8_1 blocks per warp
        float xq[NQ], mq[NQ];                   // this lane's value of each, loaded before the reduction
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            xq[k] = __ldg(a.x + (warp + k * PK_NW) * QK8_1 + lane);
            mq[k] = __ldg(a.norm_w + (warp + k * PK_NW) * QK8_1 + lane);
        }
        const float4 * x4 = (const float4 *) a.x;
#pragma unroll
        for (int v = 0; v < VPT; ++v) {
            float tmp = moep_rms_partial(__ldg(x4 + tid + v * PK_THREADS));
            tmp = warp_reduce_sum(tmp);
            if (lane == 0) {
                s_red[(tid + v * PK_THREADS) / WARP_SIZE] = tmp;
            }
        }
        __syncthreads();
        float tmp = 0.0f;
        if (lane < 512 / WARP_SIZE) {
            tmp = s_red[lane];
        }
        tmp = warp_reduce_sum(tmp);
        scale = rsqrtf(tmp / a.ncols + a.eps);
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            moep_quantize_q8_1(moep_rms_apply(scale, xq[k], mq[k]), s_yq, warp + k * PK_NW, lane);
        }
        __syncthreads();
    }

    // ---- work items ---------------------------------------------------------------------------------------------------
    char * smw = s_stage[warp];
    bool have_topk = false;
    int32_t eid_l = 0;   // lane e < KU: expert id of slot e
    float   ew_l  = 0.f; //              and its routing weight
    bool have_sact = false;
    unsigned eact_mask = 0;

    // the top-k, once per warp
    auto need_topk = [&]() -> bool {
        if (!have_topk) {
            if (!moep_wait(&sync[S_TOPK], sync, a.err, a.spin_limit, lane)) {
                return false;
            }
            if (lane < KU) {
                eid_l = __ldcg(&ids[lane]);
                ew_l  = __ldcg(&weights[lane]);
            }
            have_topk = true;
        }
        return true;
    };

    // output group g: the routed experts' weighted sum (moe_weighted_reduction), then the shared expert's down product
    // plus that sum (its fused add)
    auto combine = [&](const int g) {
        float w[KU];
#pragma unroll
        for (int e = 0; e < KU; ++e) {
            w[e] = __shfl_sync(0xffffffff, ew_l, e);
        }
        if (lane < CR) {
            const int r = g * CR + lane;
            // moe_weighted_reduction_f32 compiles to a rounded first product and one FMA per further expert (its
            // (expert * scale) * weight with scale 1, exact); written out, so that the compiler can't fuse the first
            // product into the second sum instead
            float sum = __fmul_rn(__ldcg(&dexp[r]), w[0]);
#pragma unroll
            for (int e = 1; e < KU; ++e) {
                sum = __fmaf_rn(__ldcg(&dexp[e * H + r]), w[e], sum);
            }
            if (a.moe != nullptr) {
                a.moe[r] = sum;
            }
            float result = a.fault == 3 ? 0.0f : __ldcg(&sd[r]);
            result += sum;
            a.out[r] = result;
        }
    };

    unsigned item = moep_fetch(sync, lane);
    while (item < (unsigned) IT::N_ITEMS) {
        const unsigned next = moep_fetch(sync, lane);
        if (item < (unsigned) I_SG) {
            // router rows
            const float2 * x2 = (const float2 *) a.x;
            const float2 * m2 = (const float2 *) a.norm_w;
            const auto xn2 = [&](const int c) {
                const float2 xi = __ldg(x2 + c);
                const float2 m  = __ldg(m2 + c);
                return make_float2(moep_rms_apply(scale, xi.x, m.x), moep_rms_apply(scale, xi.y, m.y));
            };
            for (int k = 0; k < RT_ROWS; ++k) {
                const int row = (item - I_RT) * RT_ROWS + k;
                float sumf;
                if (a.router_type == MOEP_ROUTER_F16_PASCAL) {
                    sumf = moep_router_row_pascal_f16((const half *) a.router + (int64_t) row * H, xn2, lane);
                } else if (a.router_type == MOEP_ROUTER_F16) {
                    sumf = moep_router_row<half, false>((const half *) a.router + (int64_t) row * H, xn2, lane);
                } else if (a.router_type == MOEP_ROUTER_F16_HACC) {
                    sumf = moep_router_row<half, true>((const half *) a.router + (int64_t) row * H, xn2, lane);
                } else if (a.router_type == MOEP_ROUTER_BF16) {
                    sumf = moep_router_row<nv_bfloat16, false>((const nv_bfloat16 *) a.router + (int64_t) row * H, xn2, lane);
                } else {
                    sumf = moep_router_row<float, false>((const float *) a.router + (int64_t) row * H, xn2, lane);
                }
                if (lane == 0) {
                    logits[row] = sumf;
                }
            }
            if (moep_arrive(&sync[S_RT_DONE], N_RT, lane)) {
                float wt[NE / WARP_SIZE];
#pragma unroll
                for (int i = 0; i < NE / WARP_SIZE; ++i) {
                    wt[i] = __ldcg(&logits[i * WARP_SIZE + lane]);
                }
                moep_topk(wt, weights, ids, a.clamp_val, a.scale_val, lane);
                if (a.fault != 1) {
                    moep_publish(&sync[S_TOPK], lane);
                }
            }
        } else if (item < (unsigned) I_EG) {
            // shared expert gate/up + SwiGLU
            constexpr int NP = SG_ROWS / SG::ROWS_PER_PASS;
            for (int p = 0; p < NP; ++p) {
                float acc[2][1];
                const int row0 = moep_rows<TSH, 1, SG::C, 2, MMVQP_SRC_SMEM, SG::LPR, SG::NIT, SG::RB, CFG::SEQ_SG>(
                    a.up_sh, a.gate_sh, FS, (const char *) s_yq, (item - I_SG) * NP + p, lane, smw, acc);
                moep_store<1, 2, SG::LPR>(acc, row0, lane, FS, sact);
            }
            if (moep_arrive(&sync[S_SG_DONE], N_SG, lane)) {
                float v[FS / QK8_1]; // all loads first, then the block quantizations
#pragma unroll
                for (int ib = 0; ib < FS / QK8_1; ++ib) {
                    v[ib] = __ldcg(&sact[ib * QK8_1 + lane]);
                }
#pragma unroll
                for (int ib = 0; ib < FS / QK8_1; ++ib) {
                    moep_quantize_q8_1(v[ib], sq, ib, lane);
                }
                moep_publish(&sync[S_SACT], lane);
            }
        } else if (item < (unsigned) I_SD) {
            // routed expert gate/up + SwiGLU, one expert slot per item
            if (!need_topk()) {
                break;
            }
            const int j = item - I_EG;
            const int e = j / N_EG1;
            const int64_t xoff = (int64_t) __shfl_sync(0xffffffff, eid_l, e) * F * GU::RB;
            constexpr int NP = EG_ROWS / GU::ROWS_PER_PASS;
            for (int p = 0; p < NP; ++p) {
                float acc[2][RG];
                const int row0 = moep_rows<TGU, RG, GU::C, 2, MMVQP_SRC_SMEM, GU::LPR, GU::NIT, GU::RB, CFG::SEQ_GU>(
                    a.up_exps + xoff, a.gate_exps + xoff, F, (const char *) s_yq, (j % N_EG1) * NP + p, lane, smw, acc);
                moep_store<RG, 2, GU::LPR>(acc, row0, lane, F, act + e * F);
            }
            if (moep_arrive(&sync[S_EG_DONE + e], N_EG1, lane)) {
                float v[F / QK8_1];
#pragma unroll
                for (int ib = 0; ib < F / QK8_1; ++ib) {
                    v[ib] = __ldcg(&act[e * F + ib * QK8_1 + lane]);
                }
#pragma unroll
                for (int ib = 0; ib < F / QK8_1; ++ib) {
                    moep_quantize_q8_1(v[ib], aq, e * (F / QK8_1) + ib, lane);
                }
                moep_publish(&sync[S_EACT + e], lane);
            }
        } else if (item < (unsigned) IT::I_ED) {
            // shared expert down projection, 8 rows
            if (!have_sact) {
                if (!moep_wait(&sync[S_SACT], sync, a.err, a.spin_limit, lane)) {
                    break;
                }
                have_sact = true;
            }
            const int g = item - I_SD;
            for (int p = 0; p < CR / SD::ROWS_PER_PASS; ++p) {
                float acc[1][SD::R];
                const int row0 = moep_rows<TSH, SD::R, SD::C, 1, MMVQP_SRC_LDCG, SD::LPR, SD::NIT, SD::RB>(
                    a.down_sh, nullptr, H, (const char *) sq, g * (CR / SD::ROWS_PER_PASS) + p, lane, smw, acc);
                moep_store<SD::R, 1, SD::LPR>(acc, row0, lane, H, sd);
            }
            if (moep_arrive(&sync[S_COMB + g], KU + 1, lane)) {
                if (!need_topk()) {
                    break;
                }
                combine(g);
            }
        } else {
            // routed expert down projection, 8 rows of one expert slot
            const int j = item - IT::I_ED;
            const int g = j / KU;
            const int e = j % KU;
            if (!need_topk()) {
                break;
            }
            if (!((eact_mask >> e) & 1)) {
                if (!moep_wait(&sync[S_EACT + e], sync, a.err, a.spin_limit, lane)) {
                    break;
                }
                eact_mask |= 1u << e;
            }
            const int64_t xoff = (int64_t) __shfl_sync(0xffffffff, eid_l, e) * H * ED::RB;
            for (int p = 0; p < CR / ED::ROWS_PER_PASS; ++p) {
                float acc[1][ED::R];
                const int row0 = moep_rows<TDN, ED::R, ED::C, 1, MMVQP_SRC_LDCG, ED::LPR, ED::NIT, ED::RB>(
                    a.down_exps + xoff, nullptr, H, (const char *) (aq + e * (F / QK8_1)),
                    g * (CR / ED::ROWS_PER_PASS) + p, lane, smw, acc);
                moep_store<ED::R, 1, ED::LPR>(acc, row0, lane, H, dexp + e * H);
            }
            if (moep_arrive(&sync[S_COMB + g], KU + 1, lane)) {
                combine(g);
            }
        }
        item = next;
    }

    // ---- leave; the last warp out resets the counters -------------------------------------------------------------------
    __syncwarp();
    unsigned old = 0;
    if (lane == 0) {
        old = atomicAdd(&sync[S_EXIT], 1u);
    }
    old = __shfl_sync(0xffffffff, old, 0);
    if (old == gridDim.x * PK_NW - 1) {
        __threadfence();
        for (int i = lane; i < N_SYNC; i += WARP_SIZE) {
            if (i != S_ERR) {
                sync[i] = 0;
            }
        }
        __threadfence();
    }
}

// ---- the kernels: one per (gate/up, down, shared) type combination ---------------------------------------------------

typedef void (*moep_kernel_t)(const moep_args);

struct moep_combo {
    ggml_type     gu;
    ggml_type     dn;
    ggml_type     sh;
    moep_kernel_t kernel;
};

// MINB: blocks per SM the kernel is built for (2: 128 registers per thread)
#define MOEP_COMBO(GU_, DN_, SH_, MINB_) { GU_, DN_, SH_, k_moe_layer<GU_, DN_, SH_, MINB_> }

// The combinations Flint files use: K2h-r16, G3 and S4-L0 (all of their MoE layers), and UD-Q4_K_XL (42 of its 47:
// the others have Q4_K or IQ3_S gate/up or a Q6_K down, which the Pascal products don't cover).
static const moep_combo moep_combos[] = {
    MOEP_COMBO(GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0,   GGML_TYPE_Q8_0,   2),
    MOEP_COMBO(GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0,   2),
    MOEP_COMBO(GGML_TYPE_Q8_0,   GGML_TYPE_Q5_0,   GGML_TYPE_Q8_0,   2),
    MOEP_COMBO(GGML_TYPE_Q5_0,   GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0,   2),
    MOEP_COMBO(GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K,   GGML_TYPE_Q8_0,   2),
    MOEP_COMBO(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0,   2),
    MOEP_COMBO(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K,   2),
    MOEP_COMBO(GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K,   2),
    MOEP_COMBO(GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K,   GGML_TYPE_Q8_0,   2),
    MOEP_COMBO(GGML_TYPE_Q5_K,   GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0,   2),
    MOEP_COMBO(GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K,   2),
    MOEP_COMBO(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, 2),
};
constexpr int N_COMBOS = sizeof(moep_combos) / sizeof(moep_combos[0]);

#undef MOEP_COMBO

static int moep_find_combo(const ggml_type gu, const ggml_type dn, const ggml_type sh) {
    for (int c = 0; c < N_COMBOS; ++c) {
        if (moep_combos[c].gu == gu && moep_combos[c].dn == dn && moep_combos[c].sh == sh) {
            return c;
        }
    }
    return -1;
}

static int moep_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_PERSISTENT_MOE");
        return env ? atoi(env) : 1;
    }();
    return mode;
}

static int moep_env_int(const char * name) {
    const char * env = getenv(name);
    return env ? atoi(env) : 0;
}

static bool moep_aligned(const ggml_tensor * t, const size_t al) {
    return (uintptr_t) t->data % al == 0 && t->nb[1] % al == 0 && t->nb[2] % al == 0 && t->nb[3] % al == 0;
}

// Blocks per launch: the resident maximum (two per SM where the kernel fits them), found once per kernel and device.
static int moep_grid(const int combo, const int device) {
    static int grid[N_COMBOS][GGML_CUDA_MAX_DEVICES] = {};
    static const int grid_override = moep_env_int("GGML_CUDA_PERSISTENT_MOE_GRID");
    if (grid_override > 0) {
        return grid_override;
    }
    if (grid[combo][device] == 0) {
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, moep_combos[combo].kernel, PK_THREADS, 0));
        grid[combo][device] = std::max(per_sm, 1) * ggml_cuda_info().devices[device].nsm;
    }
    return grid[combo][device];
}

// Byte-wise comparison of the persistent kernel's output with the per-op path's (check mode).
static __global__ void k_moep_compare(const uint32_t * a, const uint32_t * b, const int n, unsigned * n_differ) {
    __shared__ int any;
    if (threadIdx.x == 0) {
        any = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (a[i] != b[i]) {
            any = 1;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && any) {
        atomicAdd(n_differ, 1u);
    }
}

// check mode: the per-op path's nodes still to be compared with the kernel's scratch, per context (the layers of one
// context run one after another on its main stream)
struct moep_pending {
    const ggml_tensor * node;
    size_t              offset; // in the context's buffer
    int                 n;      // 32-bit words
    int                 stage;
};
static std::unordered_map<const ggml_backend_cuda_context *, std::vector<moep_pending>> moep_check_pending;

constexpr int MOEP_ERR_WORDS = 2 + GGML_CUDA_MOE_STAGE_COUNT; // [0] timeout, [1] unused, [2 + stage] layers differing

static void moep_alloc(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (st.buf != nullptr) {
        return;
    }
    // first use comes before any CUDA graph capture (ggml-cuda evaluates a graph uncaptured during its warmup), as for
    // the norm's Q8_1 buffer; the counters are zeroed in stream order before the first launch
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaMalloc(&st.buf, moep_scratch::size));
    CUDA_CHECK(cudaMemsetAsync(st.buf, 0, moep_scratch::size, ctx.stream()));
    ggml_cuda_keepalive_quiesce();
    CUDA_CHECK(cudaHostAlloc((void **) &st.err_host, MOEP_ERR_WORDS * sizeof(unsigned), cudaHostAllocMapped));
    memset(st.err_host, 0, MOEP_ERR_WORDS * sizeof(unsigned));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &st.err_dev, st.err_host, 0));
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    GGML_LOG_INFO("%s: %s (compute capability %d.%d): MoE layers at one token run as one persistent kernel "
                  "(GGML_CUDA_PERSISTENT_MOE=%d)\n", __func__, ctx.name.c_str(), cc / 100, (cc % 100) / 10, moep_mode());
}

} // namespace

void ggml_cuda_moe_persistent_rescale_notice(const int device) {
    static std::atomic<bool> warned[GGML_CUDA_MAX_DEVICES];
    if (!warned[device].exchange(true)) {
        GGML_LOG_WARN("%s: CUDA%d: MoE layers at one token keep the per-op path: the graph rescales the MoE down-projection "
                      "inputs (LLAMA_NO_MOE_DOWN_RESCALE is unset). Set LLAMA_NO_MOE_DOWN_RESCALE=tg to run each layer as "
                      "one persistent kernel: the Pascal kernels sum activations in FP32, so one-token steps need no "
                      "rescale\n", __func__, device);
    }
}

// 1 (default): compute capability 6.1 only; 2: any NVIDIA GPU (testing)
bool ggml_cuda_moe_persistent_enabled(const int cc) {
    const int mode = moep_mode();
    return GGML_CUDA_CC_IS_NVIDIA(cc) && (mode == 2 || (mode == 1 && cc == GGML_CUDA_CC_DP4A));
}

bool ggml_cuda_moe_persistent_check_mode() {
    static const bool check = moep_env_int("GGML_CUDA_PERSISTENT_MOE_CHECK") != 0;
    return check;
}

bool ggml_cuda_moe_persistent_supported(const ggml_cuda_moe_layer & L) {
    const ggml_tensor * x = L.x;
    if (x->type != GGML_TYPE_F32 || x->ne[0] != H || ggml_nrows(x) != 1 || !ggml_is_contiguous(x) ||
            L.dst->type != GGML_TYPE_F32 || ggml_nelements(L.dst) != H || !ggml_is_contiguous(L.dst) ||
            L.norm_w->type != GGML_TYPE_F32 || ggml_nelements(L.norm_w) != H || !ggml_is_contiguous(L.norm_w) ||
            (uintptr_t) x->data % 16 != 0 || (uintptr_t) L.norm_w->data % 16 != 0) {
        return false;
    }
    // the kernel reads x while it writes dst (x is read again by the residual add after the layer, so they never share
    // memory in a real graph)
    const char * x0 = (const char *) x->data;
    const char * d0 = (const char *) L.dst->data;
    if (x0 < d0 + ggml_nbytes(L.dst) && d0 < x0 + ggml_nbytes(x)) {
        return false;
    }
    const ggml_tensor * r = L.router;
    if ((r->type != GGML_TYPE_F16 && r->type != GGML_TYPE_BF16 && r->type != GGML_TYPE_F32) || r->ne[0] != H ||
            r->ne[1] != NE || r->ne[2] != 1 || r->ne[3] != 1 || !ggml_is_contiguous(r) || (uintptr_t) r->data % 16 != 0) {
        return false;
    }
    const auto expert_shape = [](const ggml_tensor * t, const int64_t k, const int64_t rows) {
        return t->ne[0] == k && t->ne[1] == rows && t->ne[2] == NE && t->ne[3] == 1 && ggml_is_contiguous(t);
    };
    const auto dense_shape = [](const ggml_tensor * t, const int64_t k, const int64_t rows) {
        return t->ne[0] == k && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
    };
    if (!expert_shape(L.gate_exps, H, F) || !expert_shape(L.up_exps, H, F) || !expert_shape(L.down_exps, F, H) ||
            !dense_shape(L.gate_sh, H, FS) || !dense_shape(L.up_sh, H, FS) || !dense_shape(L.down_sh, FS, H) ||
            L.gate_exps->type != L.up_exps->type || L.gate_sh->type != L.up_sh->type || L.gate_sh->type != L.down_sh->type) {
        return false;
    }
    // staged weight loads
    for (const ggml_tensor * t : { L.gate_exps, L.up_exps, L.down_exps, L.gate_sh, L.up_sh, L.down_sh }) {
        if (!moep_aligned(t, 16)) {
            return false;
        }
    }
    return moep_find_combo(L.gate_exps->type, L.down_exps->type, L.gate_sh->type) >= 0;
}

void ggml_cuda_op_moe_persistent(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_layer & L) {
    moep_alloc(ctx);
    auto & st = ctx.moe_persistent;
    const bool check = ggml_cuda_moe_persistent_check_mode();
    const int  combo = moep_find_combo(L.gate_exps->type, L.down_exps->type, L.gate_sh->type);
    GGML_ASSERT(combo >= 0);

    static const int fault = moep_env_int("GGML_CUDA_PERSISTENT_MOE_FAULT");

    moep_args a;
    a.x           = (const float *) L.x->data;
    a.norm_w      = (const float *) L.norm_w->data;
    a.router      = L.router->data;
    a.gate_exps   = (const char *) L.gate_exps->data;
    a.up_exps     = (const char *) L.up_exps->data;
    a.down_exps   = (const char *) L.down_exps->data;
    a.gate_sh     = (const char *) L.gate_sh->data;
    a.up_sh       = (const char *) L.up_sh->data;
    a.down_sh     = (const char *) L.down_sh->data;
    a.out         = check ? (float *) ((char *) st.buf + moep_scratch::check) : (float *) L.dst->data;
    a.moe         = check ? (float *) ((char *) st.buf + moep_scratch::moe) : nullptr;
    a.buf         = (char *) st.buf;
    a.err         = st.err_dev;
    if (L.router->type == GGML_TYPE_F16) {
        const bool hacc = fast_fp16_available(ggml_cuda_info().devices[ctx.device].cc) && L.router_prec == GGML_PREC_UNDEFINED;
        // the per-op path takes the Pascal F16 product wherever it applies (mmvf.cu), so the router follows it
        const bool pascal = ggml_cuda_mmvf_pascal_f16_applies(L.router->data, H, L.router->nb[1] / sizeof(half));
        a.router_type = pascal ? MOEP_ROUTER_F16_PASCAL : hacc ? MOEP_ROUTER_F16_HACC : MOEP_ROUTER_F16;
    } else {
        a.router_type = L.router->type == GGML_TYPE_BF16 ? MOEP_ROUTER_BF16 : MOEP_ROUTER_F32;
    }
    a.ncols       = H;
    a.eps         = L.eps;
    a.clamp_val   = L.clamp_val;
    a.scale_val   = fault == 2 ? L.scale_val * 1.0001f : L.scale_val; // testing: routing weights off by 1e-4
    a.spin_limit  = fault == 1 ? 1 << 16 : 1 << 24;
    a.fault       = fault;

    moep_combos[combo].kernel<<<moep_grid(combo, ctx.device), PK_THREADS, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());

    st.n_layers++;
    if (check) {
        st.check_dst = L.dst;
        auto & pending = moep_check_pending[&ctx];
        pending.clear();
        const size_t off[GGML_CUDA_MOE_STAGE_COUNT] = {
            moep_scratch::logits, moep_scratch::ids, moep_scratch::weights, moep_scratch::act, moep_scratch::dexp,
            moep_scratch::sact, moep_scratch::moe, moep_scratch::check };
        const int n[GGML_CUDA_MOE_STAGE_COUNT] = { NE, KU, KU, KU * F, KU * H, FS, H, H };
        for (int s = 0; s < GGML_CUDA_MOE_STAGE_COUNT; ++s) {
            pending.push_back({ L.stage[s], off[s], n[s], s });
        }
    }
}

void ggml_cuda_moe_persistent_check_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    auto & st = ctx.moe_persistent;
    auto & pending = moep_check_pending[&ctx];
    for (auto it = pending.begin(); it != pending.end(); ) {
        if (it->node != node) {
            ++it;
            continue;
        }
        k_moep_compare<<<1, 256, 0, ctx.stream()>>>((const uint32_t *) ((const char *) st.buf + it->offset),
            (const uint32_t *) node->data, it->n, st.err_dev + 2 + it->stage);
        CUDA_CHECK(cudaGetLastError());
        it = pending.erase(it);
    }
    if (node == st.check_dst) {
        st.check_dst = nullptr;
        st.n_checked++;
    }
}

void ggml_cuda_moe_persistent_begin_graph(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (st.err_host != nullptr && *(volatile unsigned *) &st.err_host[0] != 0) {
        GGML_ABORT("GGML_CUDA_PERSISTENT_MOE: a persistent MoE kernel timed out on %s (a bug); run with "
                   "GGML_CUDA_PERSISTENT_MOE=0 and report it", ctx.name.c_str());
    }
    st.n_layers  = 0;
    st.n_checked = 0;
    st.check_dst = nullptr;
}

void ggml_cuda_moe_persistent_end_graph(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (!ggml_cuda_moe_persistent_check_mode() || st.n_layers == 0) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    unsigned n_differ[GGML_CUDA_MOE_STAGE_COUNT];
    for (int s = 0; s < GGML_CUDA_MOE_STAGE_COUNT; ++s) {
        n_differ[s] = *(volatile unsigned *) &st.err_host[2 + s];
        st.err_host[2 + s] = 0;
    }
    GGML_LOG_INFO("%s: %s: persistent MoE kernel ran %d layers, %d compared with the per-op path, %u differed "
                  "(logits %u, ids %u, weights %u, routed act %u, routed down %u, shared act %u, routed sum %u)\n", __func__,
                  ctx.name.c_str(), st.n_layers, st.n_checked, n_differ[GGML_CUDA_MOE_STAGE_OUT],
                  n_differ[GGML_CUDA_MOE_STAGE_LOGITS], n_differ[GGML_CUDA_MOE_STAGE_IDS],
                  n_differ[GGML_CUDA_MOE_STAGE_WEIGHTS], n_differ[GGML_CUDA_MOE_STAGE_ACT],
                  n_differ[GGML_CUDA_MOE_STAGE_DOWN], n_differ[GGML_CUDA_MOE_STAGE_SACT], n_differ[GGML_CUDA_MOE_STAGE_MOE]);
}

void ggml_cuda_moe_persistent_free(ggml_backend_cuda_context & ctx) {
    auto & st = ctx.moe_persistent;
    if (st.buf != nullptr || st.err_host != nullptr) {
        ggml_cuda_keepalive_quiesce();
    }
    if (st.buf != nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaFree(st.buf));
        st.buf = nullptr;
    }
    if (st.err_host != nullptr) {
        CUDA_CHECK(cudaFreeHost(st.err_host));
        st.err_host = nullptr;
        st.err_dev  = nullptr;
    }
}

#else

bool ggml_cuda_moe_persistent_enabled(const int cc) {
    GGML_UNUSED(cc);
    return false;
}

bool ggml_cuda_moe_persistent_check_mode() {
    return false;
}

bool ggml_cuda_moe_persistent_supported(const ggml_cuda_moe_layer & layer) {
    GGML_UNUSED(layer);
    return false;
}

void ggml_cuda_moe_persistent_rescale_notice(int device) {
    GGML_UNUSED(device);
}

void ggml_cuda_op_moe_persistent(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_layer & layer) {
    GGML_UNUSED_VARS(ctx, layer);
    GGML_ABORT("fatal error");
}

void ggml_cuda_moe_persistent_check_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    GGML_UNUSED_VARS(ctx, node);
}

void ggml_cuda_moe_persistent_begin_graph(ggml_backend_cuda_context & ctx) {
    GGML_UNUSED(ctx);
}

void ggml_cuda_moe_persistent_end_graph(ggml_backend_cuda_context & ctx) {
    GGML_UNUSED(ctx);
}

void ggml_cuda_moe_persistent_free(ggml_backend_cuda_context & ctx) {
    GGML_UNUSED(ctx);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
