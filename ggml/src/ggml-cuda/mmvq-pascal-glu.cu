#include "mmvq-pascal-glu.cuh"
#include "mmvq-pascal.cuh"
#include "mmvq-pascal-dmt.cuh"
#include "mmvq-pascal-q5k.cuh"
#include "quantize.cuh"
#include "unary.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

// How the kernel works (C757, Q1333: Qwen3.8-Flash-Next's shared expert, gate and up 2,560 -> 640, at 2 to 8 tokens):
//   - One launch for gate, up and SWIGLU. Each block stages every token's activations once into shared memory (from the
//     Q8_1 blocks ggml_cuda_mul_mat_vec_q's quantizer writes; the int8 words as they come, per 32-value block and token
//     d8, and where a type needs them S = d8 * the sum of the int8 values (Q4_K, Q5_K) or Q3_K's d8 / f and offsets, as
//     the separate kernels compute them), so gate and up share one staging. Its warps work in pairs over row blocks of
//     a persistent grid: the pair's first warp computes the block's rows of the gate, the second the same rows of the
//     up, each with its own type's code; the two meet at a named barrier and write silu(gate) * up of those rows
//     (ggml_cuda_op_silu_single(gate) * up, the separate GLU kernel's expression on the same two FP32 values).
//   - Each row's sum is formed as the separate product's kernel forms it, so it is the same bit for bit:
//       Q2_0, Q3_K: the k-column kernel's order (kcol::unit_q2_0, kcol::unit_q3_K; the one-token kernel of both): L lanes
//       per row (the largest power of two, at most 32, dividing the row's units of 128 or 256 weights), lane sl taking
//       units sl, sl + L, ... in order, per 64 weights (Q2_0) acc = fmaf(d, fmaf(i1, d8_1, i0 * d8_0), acc), per 32
//       weights (Q3_K) acc = fmaf(fmaf(i1, d sc1, i0 * (d sc0)), d8 / f, acc);
//       IQ4_XS, Q4_K, Q5_K: the one-token Pascal kernel's order (mmvqp_dot_iq4_xs, mmvqp_dot_q4_K, mmvqp_dot_q5_K, as
//       C722's and C690's kernels repeat it): the same L rule over super-blocks, per 32 weights sumf = fmaf(d8, isum *
//       (ls - 32), sumf) or sumd = fmaf(d8, isum * sc, sumd) and summ = fmaf(m, S, summ), per super-block fmaf(d,
//       sumf, acc) or acc + d * sumd - dmin * summ;
//     and the lanes' sums added with the same xor shuffles. The integer sums are the same whatever order the int8
//     products are added in, so each type's decoded weights are put in the activations' order (Q2_0's and IQ4_XS's
//     lookups return them interleaved: two __byte_perm per pair of words, per row, not per token), and the
//     integer-to-float conversions are made exactly without I2F (the dp4a chains start at 0x4B400000, C722).
//   - Each warp stages its lane groups' row segments (L units of a row per iteration) into shared memory from the
//     16-byte boundary below them with 128-bit loads (any row alignment of the type), each lane group its own row, and
//     by default loads the next segments into registers before it computes the current ones (C722's prefetch).
//   - Rows per row block: the larger of the two types' rows per warp pass (32 / L); the other type's warp makes two or
//     more passes (at 2,560 weights: 16 rows, Q2_0's warp in two passes of 8).
//   - 5 to 8 tokens of a pair with an IQ4_XS, Q4_K or Q5_K product (C793, Q1388; GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 or 4):
//     mul_mat_vec_q_pascal_glu_ro8, the same steps with every type's warp code. GGML_CUDA_MMVQ_PASCAL_RO8 runs those
//     products of 5 to 8 tokens on C722's and C690's kernels at R = 1, whose sums are the one-token kernel's (C729), so
//     the order above holds there too; the pair is taken only where those kernels take the separate product. A kernel of
//     its own, so that mul_mat_vec_q_pascal_glu at 5 to 8 tokens (Q2_0 and Q3_K pairs) keeps its code and registers.

#define GLU_MAX_COLS 8
#define GLU_NP_MAX   4                    // warp pairs per block, at most
#define GLU_SMEM_MAX (48 * 1024)

static_assert(sizeof(block_q8_1) == 36, "the activation staging assumes 36-byte Q8_1 blocks");
static_assert(sizeof(block_q2_0) == 18 && sizeof(block_q3_K) == 110 && sizeof(block_q4_K) == 144 &&
              sizeof(block_q5_K) == 176 && sizeof(block_iq4_xs) == 136, "the units assume these blocks");

namespace {

enum glu_kind : int { GK_Q2_0 = 0, GK_Q3_K = 1, GK_Q4_K = 2, GK_Q5_K = 3, GK_IQ4_XS = 4, GK_COUNT = 5 };

// Per type: UB bytes and UW weights of a lane unit (two Q2_0 blocks, as the k-column kernel; one super-block of the
// others), AL the units' alignment.
template <int T> struct glu_traits;
template <> struct glu_traits<GK_Q2_0>   { static constexpr int UB = 36,  UW = 128, AL = 4;  };
template <> struct glu_traits<GK_Q3_K>   { static constexpr int UB = 110, UW = 256, AL = 2;  };
template <> struct glu_traits<GK_Q4_K>   { static constexpr int UB = 144, UW = 256, AL = 16; };
template <> struct glu_traits<GK_Q5_K>   { static constexpr int UB = 176, UW = 256, AL = 16; };
template <> struct glu_traits<GK_IQ4_XS> { static constexpr int UB = 136, UW = 256, AL = 8;  };

static int glu_ub(const int k) {
    static const int ub[GK_COUNT] = { 36, 110, 144, 176, 136 };
    return ub[k];
}
static int glu_uw(const int k) {
    return k == GK_Q2_0 ? 128 : 256;
}
static int glu_al(const int k) {
    static const int al[GK_COUNT] = { 4, 2, 16, 16, 8 };
    return al[k];
}

// 16-byte units a lane stages of one step's segment, at most: the segment of UB L bytes from the 16-byte boundary below
// it (which lies at most 16 - AL bytes before it; none for AL = 16), over L lanes, the most at L = 1 (glu_matrix_ok
// checks it for the product's L)
template <int T> constexpr __host__ __device__ int glu_maxu() {
    return (glu_traits<T>::UB + (glu_traits<T>::AL >= 16 ? 0 : 16 - glu_traits<T>::AL) + 15) / 16;
}
static int glu_maxu_host(const int k) {
    static const int mu[GK_COUNT] = { glu_maxu<GK_Q2_0>(), glu_maxu<GK_Q3_K>(), glu_maxu<GK_Q4_K>(), glu_maxu<GK_Q5_K>(),
                                      glu_maxu<GK_IQ4_XS>() };
    return mu[k];
}

// The staged activations, per group of 4 blocks of 32 values: the int8 words of each block and token (8 words), padded by
// 4 words so that the groups of consecutive units start in other banks; the per-block terms (one word or int2 per block
// and token), padded by 1.
template <int NT> constexpr __host__ __device__ int glu_qg() { return 32 * NT + 4; }
template <int NT> constexpr __host__ __device__ int glu_dg() { return 4 * NT + 1; }

struct glu_mat {
    const char * vx;
    int          row_bytes;
    int          kind;
    int          lpr_log2;  // lanes per row L = 1 << lpr_log2
    int          niter;     // units per lane and row
    int          npass;     // warp passes per row block (rows per row block / (32 >> lpr_log2))
    int          slot;      // bytes of a lane group's row slot in its warp's staging buffer
};

struct glu_args {
    glu_mat      m[2];      // the gate (the pairs' first warps), the up (their second)
    const char * vy;        // block_q8_1, column j from j * col_bytes_y
    int          col_bytes_y;
    float      * dst;       // column j from j * stride_col_dst
    int          stride_col_dst;
    int          nrows;
    int          nab;       // 32-value activation blocks per column
    int          rbr;       // rows per row block (a power of two)
    int          rbr_log2;
    int          nrb;       // row blocks
    int          np;        // warp pairs per block
    int          a3h_off;   // int2 units from the start of shared memory: Q3_K's offsets
    int          ad_off;    // words: d8
    int          as_off;    // words: S (Q4_K, Q5_K)
    int          a3d_off;   // words: d8 / f (Q3_K)
    int          need_s;
    int          need_q3;
    int          stage_off; // bytes: the warps' staging buffers
    int          pstage;    // bytes of a pair's staging buffers (the gate warp's, then the up warp's)
    int          wstage0;   // bytes of the gate warp's
    int          res_off;   // bytes: the pairs' result buffers
    int          fault;     // GGML_CUDA_MMVQ_PASCAL_GLU_MT_FAULT
};

// Every token's activation blocks into shared memory, by all warps of the block.
template <int NT>
static __device__ __forceinline__ void glu_stage_act(const glu_args & p, char * sm) {
    constexpr int QG = glu_qg<NT>();
    constexpr int DG = glu_dg<NT>();
    uint32_t * aq  = (uint32_t *) sm;
    float    * ad  = (float *) sm + p.ad_off;
    float    * as  = (float *) sm + p.as_off;
    float    * a3d = (float *) sm + p.a3d_off;
    int2     * a3h = (int2 *) sm + p.a3h_off;
    const int nth = blockDim.x * blockDim.y;
    for (int e = threadIdx.y * WARP_SIZE + threadIdx.x; e < NT * p.nab; e += nth) {
        const int j  = e / p.nab;
        const int gb = e - j * p.nab;
        const uint32_t * src = (const uint32_t *) (p.vy + j * p.col_bytes_y + gb * (int) sizeof(block_q8_1));
        uint32_t w[9];
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            w[k] = __ldg(src + k);
        }
        const float d8 = kcol::h2f_lo(w[0]);
        // the fault (GGML_CUDA_MMVQ_PASCAL_GLU_MT_FAULT=1): token j's activations staged as token j + 1's
        const int jd = p.fault ? (j + 1) % NT : j;
        const int qi = (gb >> 2) * QG + (gb & 3) * 8 * NT + jd * 8;
        const int di = (gb >> 2) * DG + (gb & 3) * NT + jd;
        uint4 * dq = (uint4 *) (aq + qi);
        dq[0] = make_uint4(w[1], w[2], w[3], w[4]);
        dq[1] = make_uint4(w[5], w[6], w[7], w[8]);
        ad[di] = d8;
        if (p.need_s) {
            // as mmvqp_load_act and C722's staging: S = d8 * the sum of the int8 values
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum = kcol::dp4a(0x01010101, (int) w[1 + k], sum);
            }
            as[di] = d8 * (float) sum;
        }
        if (p.need_q3) {
            // as kcol::quantize_records<KQ3_K> and C726's staging: block b of a super-block holds the codes at f = 1, 4,
            // 16, 16 times their value; d8 / f, and 0x4B400000 + -4 f times its two halves' sums
            int s0 = 0;
            int s1 = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s0 = kcol::dp4a(0x01010101, (int) w[1 + k], s0);
                s1 = kcol::dp4a(0x01010101, (int) w[5 + k], s1);
            }
            const int b = gb & 7;
            const int f = 1 << (2 * min(b & 3, 2));
            const int m = -4 * f;
            a3d[di] = d8 / (float) f;
            a3h[di] = make_int2(0x4B400000 + m * s0, 0x4B400000 + m * s1);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// One lane unit of one row times every token, per type: acc[j] gains the unit's products with token j exactly as the
// type's one-token kernel adds them. aq, ad, ...: the unit's first group of staged activations.

// Q2_0 (kcol::unit_q2_0): two blocks of 18 bytes, d at bytes 0 and 18, 4 code words of 16 weights at 2.. and 20..
template <int NT>
static __device__ __forceinline__ void glu_unit_q2_0(const kcol::unit_ptr & wu, const uint32_t * aq, const float * ad,
                                                     float (&acc)[NT]) {
    constexpr uint32_t t = 0x020100FF; // codes 0, 1, 2, 3 -> -1, 0, 1, 2
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        uint32_t w[5];
        uint32_t cw[4];
        float    d;
        if (c == 0) {
            kcol::unit_words<4, 5>(w, wu, 0);
            d = kcol::h2f_lo(w[0]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                cw[k] = kcol::prmt(w[k], w[k + 1], 0x5432);
            }
        } else {
            kcol::unit_words<4, 5>(w, wu, 4);
            d = kcol::h2f_hi(w[0]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                cw[k] = w[k + 1];
            }
        }
        // halves h = 0, 1 (activation blocks 2c, 2c + 1: code words 2h, 2h + 1), each word in natural order: the lookup
        // returns weights (0, 2, 4, 6), (1, 3, 5, 7), (8, 10, 12, 14), (9, 11, 13, 15) of a code word
        int q[2][8];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const uint32_t cc = cw[2 * h + v];
                const uint32_t e0 = kcol::prmt(t, t, cc);
                const uint32_t o0 = kcol::prmt(t, t, cc >> 2);
                const uint32_t e1 = kcol::prmt(t, t, cc >> 16);
                const uint32_t o1 = kcol::prmt(t, t, cc >> 18);
                q[h][4 * v + 0] = (int) kcol::prmt(e0, o0, 0x5140);
                q[h][4 * v + 1] = (int) kcol::prmt(e0, o0, 0x7362);
                q[h][4 * v + 2] = (int) kcol::prmt(e1, o1, 0x5140);
                q[h][4 * v + 3] = (int) kcol::prmt(e1, o1, 0x7362);
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const uint4 u0 = *(const uint4 *) (aq + (2 * c) * 8 * NT + j * 8);
            const uint4 u1 = *(const uint4 *) (aq + (2 * c) * 8 * NT + j * 8 + 4);
            const uint4 v0 = *(const uint4 *) (aq + (2 * c + 1) * 8 * NT + j * 8);
            const uint4 v1 = *(const uint4 *) (aq + (2 * c + 1) * 8 * NT + j * 8 + 4);
            const float d0 = ad[(2 * c) * NT + j];
            const float d1 = ad[(2 * c + 1) * NT + j];
            int i0 = 0x4B400000; // 12582912 as a float: the result reads as 12582912 + i0
            int i1 = 0x4B400000;
            i0 = kcol::dp4a(q[0][0], (int) u0.x, i0);
            i1 = kcol::dp4a(q[1][0], (int) v0.x, i1);
            i0 = kcol::dp4a(q[0][1], (int) u0.y, i0);
            i1 = kcol::dp4a(q[1][1], (int) v0.y, i1);
            i0 = kcol::dp4a(q[0][2], (int) u0.z, i0);
            i1 = kcol::dp4a(q[1][2], (int) v0.z, i1);
            i0 = kcol::dp4a(q[0][3], (int) u0.w, i0);
            i1 = kcol::dp4a(q[1][3], (int) v0.w, i1);
            i0 = kcol::dp4a(q[0][4], (int) u1.x, i0);
            i1 = kcol::dp4a(q[1][4], (int) v1.x, i1);
            i0 = kcol::dp4a(q[0][5], (int) u1.y, i0);
            i1 = kcol::dp4a(q[1][5], (int) v1.y, i1);
            i0 = kcol::dp4a(q[0][6], (int) u1.z, i0);
            i1 = kcol::dp4a(q[1][6], (int) v1.z, i1);
            i0 = kcol::dp4a(q[0][7], (int) u1.w, i0);
            i1 = kcol::dp4a(q[1][7], (int) v1.w, i1);
            const float f0 = __int_as_float(i0) - 12582912.0f; // (float) i0, exactly (|i0| <= 32 * 2 * 127)
            const float f1 = __int_as_float(i1) - 12582912.0f;
            acc[j] = fmaf(d, fmaf(f1, d1, f0 * d0), acc[j]);
        }
    }
}

// Q3_K (kcol::unit_q3_K, as C726's kernel): hmask words 0..7, codes 8..23, scales 24..26, d in word 27 (2-byte aligned).
template <int NT>
static __device__ __forceinline__ void glu_unit_q3_K(const kcol::unit_ptr & wu, const uint32_t * aq, const float * a3d,
                                                     const int2 * a3h, float (&acc)[NT]) {
    constexpr int QG = glu_qg<NT>();
    constexpr int DG = glu_dg<NT>();
    uint32_t hm[8];
    uint32_t sc[4];
    kcol::unit_words<2, 8>(hm, wu, 0);
    float d;
    {
        uint32_t w[4];
        kcol::unit_words<2, 4>(w, wu, 24);
        const uint32_t a0 = w[0], a1 = w[1], tmp = w[2];
        sc[0] = (a0 & 0x0f0f0f0f)        | (((tmp >> 0) & 0x03030303) << 4);
        sc[1] = (a1 & 0x0f0f0f0f)        | (((tmp >> 2) & 0x03030303) << 4);
        sc[2] = ((a0 >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
        sc[3] = ((a1 >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
        d = kcol::h2f_lo(w[3]);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        // blocks 2p and 2p + 1: n = p / 2, m = 2 (p & 1) and 2 (p & 1) + 1
        const int n = p >> 1;
        uint32_t qs[8];
        kcol::unit_words<2, 8>(qs, wu, 8 + 8 * n);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int b     = 2 * p + e;
            const int m     = b & 3;
            const int hbpos = m < 3 ? 2 + 2 * m : 6;  // the high bit's place in each byte
            const int sh    = hbpos - b;              // hmask bit b to bit hbpos
            int q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint32_t lo = m < 3 ? qs[k] & (0x03030303u << (2 * m)) : (qs[k] >> 2) & 0x30303030u;
                const uint32_t hb = (sh >= 0 ? hm[k] << sh : hm[k] >> -sh) & (0x01010101u << hbpos);
                q[k] = (int) (lo | hb);
            }
            // scales 2b and 2b + 1 less 32, exactly: the 6-bit scale as the low mantissa bits of 12582912 = 2^23 + 2^22
            const uint32_t sw = sc[b >> 1];
            const float s0 = __int_as_float((int) __byte_perm(sw, 0x4B400000u, 0x7650 | (2 * (b & 1))))     - 12582944.0f;
            const float s1 = __int_as_float((int) __byte_perm(sw, 0x4B400000u, 0x7650 | (2 * (b & 1) + 1))) - 12582944.0f;
            const float dsc0 = d * s0;
            const float dsc1 = d * s1;
            const int go = (b >> 2) * QG + (b & 3) * 8 * NT;
            const int gd = (b >> 2) * DG + (b & 3) * NT;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4 u0  = *(const uint4 *) (aq + go + j * 8);
                const uint4 u1  = *(const uint4 *) (aq + go + j * 8 + 4);
                const float a_d = a3d[gd + j];
                const int2  h   = a3h[gd + j];
                int i0 = h.x; // 0x4B400000 + the offset: the result reads as 12582912 + i0 as a float
                int i1 = h.y;
                i0 = kcol::dp4a(q[0], (int) u0.x, i0);
                i1 = kcol::dp4a(q[4], (int) u1.x, i1);
                i0 = kcol::dp4a(q[1], (int) u0.y, i0);
                i1 = kcol::dp4a(q[5], (int) u1.y, i1);
                i0 = kcol::dp4a(q[2], (int) u0.z, i0);
                i1 = kcol::dp4a(q[6], (int) u1.z, i1);
                i0 = kcol::dp4a(q[3], (int) u0.w, i0);
                i1 = kcol::dp4a(q[7], (int) u1.w, i1);
                const float f0 = __int_as_float(i0) - 12582912.0f; // (float) i0, exactly
                const float f1 = __int_as_float(i1) - 12582912.0f;
                acc[j] = fmaf(fmaf(f1, dsc1, f0 * dsc0), a_d, acc[j]);
            }
        }
    }
}

// Q4_K and Q5_K (mmvqp_dot_q4_K, mmvqp_dot_q5_K, as C722's and C690's kernels): dm, 3 words of scales, (Q5_K: 8 words
// qh,) 32 words qs; sub-blocks 2p and 2p + 1 take the low and high nibbles of qs words 8p .. 8p + 7.
template <int NT, bool Q5>
static __device__ __forceinline__ void glu_unit_q45_K(const kcol::unit_ptr & wu, const uint32_t * aq, const float * ad,
                                                      const float * as, float (&acc)[NT]) {
    constexpr int QG = glu_qg<NT>();
    constexpr int DG = glu_dg<NT>();
    uint32_t h[4];
    kcol::unit_words<16, 4>(h, wu, 0);
    [[maybe_unused]] uint32_t qh[8];
    if constexpr (Q5) {
        kcol::unit_words<16, 8>(qh, wu, 4);
    }
    const uint32_t sc_lo = h[1] & 0x3f3f3f3f;                                     // scales of sub-blocks 0..3
    const uint32_t m_lo  = h[2] & 0x3f3f3f3f;                                     // mins of sub-blocks 0..3
    const uint32_t sc_hi = (h[3] & 0x0f0f0f0f) | ((h[1] >> 2) & 0x30303030);     // scales of 4..7
    const uint32_t m_hi  = ((h[3] >> 4) & 0x0f0f0f0f) | ((h[2] >> 2) & 0x30303030);
    float sumd[NT];
    float summ[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        sumd[j] = 0.0f;
        summ[j] = 0.0f;
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t qs[8];
        kcol::unit_words<16, 8>(qs, wu, (Q5 ? 12 : 4) + 8 * p);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int s = 2 * p + hh;
            int q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t v = hh ? (qs[k] >> 4) & 0x0F0F0F0F : qs[k] & 0x0F0F0F0F;
                if constexpr (Q5) {
                    v |= s <= 4 ? (qh[k] << (4 - s)) & 0x10101010 : (qh[k] >> (s - 4)) & 0x10101010;
                }
                q[k] = (int) v;
            }
            // (float) sc and (float) m exactly: the 6-bit byte as the low mantissa bits of 12582912 = 2^23 + 2^22
            const uint32_t sel  = 0x7650 | (s & 3);
            const float    scf  = __int_as_float((int) __byte_perm(s < 4 ? sc_lo : sc_hi, 0x4B400000u, sel)) - 12582912.0f;
            const float    mf   = __int_as_float((int) __byte_perm(s < 4 ? m_lo  : m_hi,  0x4B400000u, sel)) - 12582912.0f;
            const float    cneg = -12582912.0f * scf;
            const int go = (s >> 2) * QG + (s & 3) * 8 * NT;
            const int gd = (s >> 2) * DG + (s & 3) * NT;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4 u0 = *(const uint4 *) (aq + go + j * 8);
                const uint4 u1 = *(const uint4 *) (aq + go + j * 8 + 4);
                int ia = 0x4B400000; // 12582912 as a float: the result reads as 12582912 + isum
                ia = kcol::dp4a(q[0], (int) u0.x, ia);
                ia = kcol::dp4a(q[1], (int) u0.y, ia);
                ia = kcol::dp4a(q[2], (int) u0.z, ia);
                ia = kcol::dp4a(q[3], (int) u0.w, ia);
                ia = kcol::dp4a(q[4], (int) u1.x, ia);
                ia = kcol::dp4a(q[5], (int) u1.y, ia);
                ia = kcol::dp4a(q[6], (int) u1.z, ia);
                ia = kcol::dp4a(q[7], (int) u1.w, ia);
                const float x = fmaf(__int_as_float(ia), scf, cneg); // (float) (isum * sc), exactly
                sumd[j] = fmaf(ad[gd + j], x, sumd[j]);
                summ[j] = fmaf(mf, as[gd + j], summ[j]);
            }
        }
    }
    const float d    = kcol::h2f_lo(h[0]);
    const float dmin = kcol::h2f_hi(h[0]);
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j] = acc[j] + d * sumd[j] - dmin * summ[j];
    }
}

// IQ4_XS (mmvqp_dot_iq4_xs, as C722's kernel): d | scales_h << 16, scales_l, 32 words qs (8-byte aligned).
template <int NT>
static __device__ __forceinline__ void glu_unit_iq4_xs(const kcol::unit_ptr & wu, const uint32_t * aq, const float * ad,
                                                       float (&acc)[NT]) {
    constexpr int QG = glu_qg<NT>();
    constexpr int DG = glu_dg<NT>();
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u; // kvalues_iq4nl
    uint32_t h[2];
    kcol::unit_words<8, 2>(h, wu, 0);
    float sumf[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        sumf[j] = 0.0f;
    }
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        uint32_t c[4];
        kcol::unit_words<8, 4>(c, wu, 2 + 4 * ib);
        // the lookup returns weights (4k, 4k + 16, 4k + 1, 4k + 17) and (4k + 2, 4k + 18, 4k + 3, 4k + 19) of code word k;
        // q[k] = weights 4k..4k+3, q[4 + k] = weights 4k+16..4k+19
        int q[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t sel = 0x32103210 | ((c[k] & 0x88888888) >> 1);
            const uint32_t r0  = __byte_perm(__byte_perm(t0, t1, c[k]),       __byte_perm(t2, t3, c[k]),       sel);
            const uint32_t r1  = __byte_perm(__byte_perm(t0, t1, c[k] >> 16), __byte_perm(t2, t3, c[k] >> 16), sel >> 16);
            q[k]     = (int) __byte_perm(r0, r1, 0x6420);
            q[4 + k] = (int) __byte_perm(r0, r1, 0x7531);
        }
        // (float) (ls - 32) exactly: the 6-bit ls as the low mantissa bits of 12582912 = 2^23 + 2^22
        const uint32_t ls   = ((h[1] >> (4 * ib)) & 0xF) | (((h[0] >> (16 + 2 * ib)) & 3) << 4);
        const float    lsf  = __int_as_float((int) (0x4B400000u | ls)) - 12582944.0f;
        const float    cneg = -12582912.0f * lsf;
        const int go = (ib >> 2) * QG + (ib & 3) * 8 * NT;
        const int gd = (ib >> 2) * DG + (ib & 3) * NT;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const uint4 u0 = *(const uint4 *) (aq + go + j * 8);
            const uint4 u1 = *(const uint4 *) (aq + go + j * 8 + 4);
            int ia = 0x4B400000; // 12582912 as a float: the result reads as 12582912 + isum
            ia = kcol::dp4a(q[0], (int) u0.x, ia);
            ia = kcol::dp4a(q[1], (int) u0.y, ia);
            ia = kcol::dp4a(q[2], (int) u0.z, ia);
            ia = kcol::dp4a(q[3], (int) u0.w, ia);
            ia = kcol::dp4a(q[4], (int) u1.x, ia);
            ia = kcol::dp4a(q[5], (int) u1.y, ia);
            ia = kcol::dp4a(q[6], (int) u1.z, ia);
            ia = kcol::dp4a(q[7], (int) u1.w, ia);
            const float x = fmaf(__int_as_float(ia), lsf, cneg); // (float) (isum * (ls - 32)), exactly
            sumf[j] = fmaf(ad[gd + j], x, sumf[j]);
        }
    }
    const float d = kcol::h2f_lo(h[0]);
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j] = fmaf(d, sumf[j], acc[j]);
    }
}

// The pair's two warps (64 threads) at named barrier 1 + pair (barrier 0 is __syncthreads).
static __device__ __forceinline__ void glu_pair_sync(const int pair) {
    asm volatile("bar.sync %0, %1;" :: "r"(1 + pair), "r"(2 * WARP_SIZE) : "memory");
}

// One warp's share: matrix m (role 0 the gate, 1 the up) over the pair's row blocks, then with the other warp of the pair
// the SWIGLU of each row block.
template <int T, int NT, bool PF>
static __device__ __forceinline__ void glu_warp(const glu_args & p, const glu_mat & m, const int role, const int pair,
                                                char * sm) {
    constexpr int UB   = glu_traits<T>::UB;
    constexpr int AL   = glu_traits<T>::AL;
    constexpr int GPU  = glu_traits<T>::UW / 128; // groups of 4 activation blocks per unit
    constexpr int MAXU = glu_maxu<T>();
    constexpr int QG   = glu_qg<NT>();
    constexpr int DG   = glu_dg<NT>();
    static_assert(MAXU <= 32, "the prefetch mask");

    const uint32_t * aq  = (const uint32_t *) sm;
    const float    * ad  = (const float *) sm + p.ad_off;
    const float    * as  = (const float *) sm + p.as_off;
    const float    * a3d = (const float *) sm + p.a3d_off;
    const int2     * a3h = (const int2 *) sm + p.a3h_off;

    const int lane     = threadIdx.x;
    const int lpr_log2 = m.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int P        = WARP_SIZE >> lpr_log2;    // rows per warp pass
    const int seg      = UB << lpr_log2;           // bytes of a row's segment per iteration
    char  * slotp   = sm + p.stage_off + pair * p.pstage + role * p.wstage0 + grp * m.slot;
    float * res     = (float *) (sm + p.res_off) + pair * 4 * NT * p.rbr; // [buffer][role][token][row]
    const int pstride = gridDim.x * p.np;

    int rb = blockIdx.x * p.np + pair;
    if (rb >= p.nrb) {
        return;
    }
    int pass = 0;
    int it   = 0;
    int buf  = 0;
    // the lane group's row of pass ps of row block b (clamped, so that the staging reads valid rows; its results are
    // not written)
    const auto row_ptr = [&](const int b, const int ps) {
        return m.vx + (int64_t) min((b << p.rbr_log2) + ps * P + grp, p.nrows - 1) * m.row_bytes;
    };
    const char * rp = row_ptr(rb, 0);

    // unit k = sl + q L of a step's segment g, from the 16-byte boundary below it, where the segment has it
    const auto unit_src = [&](const char * g, const int q, const uint4 * & src) {
        const int h = (int) ((uintptr_t) g & 15);
        const int k = sl + (q << lpr_log2);
        src = (const uint4 *) (g - h) + k;
        return 16 * k < h + seg;
    };

    [[maybe_unused]] uint4    v[MAXU];
    [[maybe_unused]] uint32_t vok = 0;
    if constexpr (PF) {
#pragma unroll
        for (int q = 0; q < MAXU; ++q) {
            const uint4 * src = nullptr;
            if (unit_src(rp, q, src)) {
                v[q] = __ldg(src);
                vok |= 1u << q;
            }
        }
    }

    float acc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j] = 0.0f;
    }

    while (true) {
        // the current segment into the lane group's row slot
        const char * g   = rp + it * seg;
        uint4      * dsm = (uint4 *) slotp;
#pragma unroll
        for (int q = 0; q < MAXU; ++q) {
            if constexpr (PF) {
                if (vok & (1u << q)) {
                    dsm[sl + (q << lpr_log2)] = v[q];
                }
            } else {
                const uint4 * src = nullptr;
                if (unit_src(g, q, src)) {
                    dsm[sl + (q << lpr_log2)] = __ldg(src);
                }
            }
        }
        __syncwarp();

        // the next step: the next iteration of this pass, the next pass of this row block, or the pair's next row block
        int          nit = it + 1;
        int          nps = pass;
        int          nrb = rb;
        const char * nrp = rp;
        if (nit == m.niter) {
            nit = 0;
            nps = pass + 1;
            if (nps == m.npass) {
                nps = 0;
                nrb = rb + pstride;
            }
            if (nrb < p.nrb) {
                nrp = row_ptr(nrb, nps);
            }
        }
        const bool more = nrb < p.nrb;
        if constexpr (PF) {
            if (more) {
                vok = 0;
#pragma unroll
                for (int q = 0; q < MAXU; ++q) {
                    const uint4 * src = nullptr;
                    if (unit_src(nrp + nit * seg, q, src)) {
                        v[q] = __ldg(src);
                        vok |= 1u << q;
                    }
                }
            }
        }

        // this lane's unit (it L + sl) of the row
        {
            const int h  = (int) ((uintptr_t) g & 15);
            const kcol::unit_ptr wu = kcol::make_unit<AL>(slotp + h + sl * UB);
            const int g0 = ((it << lpr_log2) + sl) * GPU;
            if constexpr (T == GK_Q2_0) {
                glu_unit_q2_0<NT>(wu, aq + g0 * QG, ad + g0 * DG, acc);
            } else if constexpr (T == GK_Q3_K) {
                glu_unit_q3_K<NT>(wu, aq + g0 * QG, a3d + g0 * DG, a3h + g0 * DG, acc);
            } else if constexpr (T == GK_Q4_K) {
                glu_unit_q45_K<NT, false>(wu, aq + g0 * QG, ad + g0 * DG, as + g0 * DG, acc);
            } else if constexpr (T == GK_Q5_K) {
                glu_unit_q45_K<NT, true>(wu, aq + g0 * QG, ad + g0 * DG, as + g0 * DG, acc);
            } else {
                glu_unit_iq4_xs<NT>(wu, aq + g0 * QG, ad + g0 * DG, acc);
            }
        }

        if (it == m.niter - 1) {
            // the row's sum over its L lanes, as the separate kernels add them
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                    if (off < lpr) {
                        acc[j] += __shfl_xor_sync(0xFFFFFFFF, acc[j], off, WARP_SIZE);
                    }
                }
            }
            if (sl == 0) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    res[((buf * 2 + role) * NT + j) * p.rbr + pass * P + grp] = acc[j];
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[j] = 0.0f;
            }
            if (pass == m.npass - 1) {
                // both warps' rows of this row block are in res: silu(gate) * up, as the separate GLU kernel
                glu_pair_sync(pair);
                const float * rg = res + (buf * 2 + 0) * NT * p.rbr;
                const float * ru = res + (buf * 2 + 1) * NT * p.rbr;
                for (int e = role * WARP_SIZE + lane; e < NT * p.rbr; e += 2 * WARP_SIZE) {
                    const int j   = e >> p.rbr_log2;
                    const int row = (rb << p.rbr_log2) + (e & (p.rbr - 1));
                    if (row < p.nrows) {
                        p.dst[j * p.stride_col_dst + row] = ggml_cuda_op_silu_single(rg[e]) * ru[e];
                    }
                }
                buf ^= 1;
            }
        }
        __syncwarp();
        if (!more) {
            break;
        }
        it   = nit;
        pass = nps;
        rb   = nrb;
        rp   = nrp;
    }
}

// Grid: (blocks, 1, 1), each looping over row blocks; block (32, 2 np): warp 2i the gate, 2i + 1 the up of pair i.
// Dynamic shared memory: the staged activations, the warps' staging buffers, the pairs' result buffers. The registers
// are bounded for one block of 4 pairs only: the shared memory allows at most 2 blocks of 2 pairs per SM at
// Flash-Next's shapes (est. 39 KB a block at 4 tokens), which 255 registers a thread still fit.
template <int NT, bool PF>
__launch_bounds__(2*GLU_NP_MAX*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_pascal_glu(const glu_args p) {
    extern __shared__ uint4 glu_sm[];
    char * sm = (char *) glu_sm;

    ggml_cuda_pdl_sync();

    glu_stage_act<NT>(p, sm);
    __syncthreads();

    const int role = threadIdx.y & 1;
    const int pair = threadIdx.y >> 1;
    const glu_mat m = role ? p.m[1] : p.m[0];
    if (m.kind == GK_Q2_0) {
        glu_warp<GK_Q2_0, NT, PF>(p, m, role, pair, sm);
    } else if (m.kind == GK_Q3_K) {
        glu_warp<GK_Q3_K, NT, PF>(p, m, role, pair, sm);
    } else {
        // IQ4_XS, Q4_K and Q5_K: 2 to 4 tokens (the separate kernels they repeat; 5 to 8: mul_mat_vec_q_pascal_glu_ro8)
        if constexpr (NT <= 4) {
            if (m.kind == GK_Q4_K) {
                glu_warp<GK_Q4_K, NT, PF>(p, m, role, pair, sm);
            } else if (m.kind == GK_Q5_K) {
                glu_warp<GK_Q5_K, NT, PF>(p, m, role, pair, sm);
            } else {
                glu_warp<GK_IQ4_XS, NT, PF>(p, m, role, pair, sm);
            }
        }
    }
}

// 5 to 8 tokens of a pair with an IQ4_XS, Q4_K or Q5_K product (C793): mul_mat_vec_q_pascal_glu's steps with every
// type's warp code; grid, block and shared memory as there
template <int NT, bool PF>
__launch_bounds__(2*GLU_NP_MAX*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_pascal_glu_ro8(const glu_args p) {
    static_assert(NT > 4 && NT <= GLU_MAX_COLS, "the 2-4-token pairs run mul_mat_vec_q_pascal_glu");
    extern __shared__ uint4 glu_sm[];
    char * sm = (char *) glu_sm;

    ggml_cuda_pdl_sync();

    glu_stage_act<NT>(p, sm);
    __syncthreads();

    const int role = threadIdx.y & 1;
    const int pair = threadIdx.y >> 1;
    const glu_mat m = role ? p.m[1] : p.m[0];
    if (m.kind == GK_Q2_0) {
        glu_warp<GK_Q2_0, NT, PF>(p, m, role, pair, sm);
    } else if (m.kind == GK_Q3_K) {
        glu_warp<GK_Q3_K, NT, PF>(p, m, role, pair, sm);
    } else if (m.kind == GK_Q4_K) {
        glu_warp<GK_Q4_K, NT, PF>(p, m, role, pair, sm);
    } else if (m.kind == GK_Q5_K) {
        glu_warp<GK_Q5_K, NT, PF>(p, m, role, pair, sm);
    } else {
        glu_warp<GK_IQ4_XS, NT, PF>(p, m, role, pair, sm);
    }
}

static int glu_env(const char * name, const int dflt) {
    const char * env = getenv(name);
    return env ? atoi(env) : dflt;
}

static int glu_mode() {
    static const int mode = glu_env("GGML_CUDA_MMVQ_PASCAL_GLU_MT", 0);
    return mode;
}

// GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 or 4 (C793): IQ4_XS, Q4_K and Q5_K products of 5 to 8 tokens taken too
static bool glu_ro8() {
    return glu_mode() == 3 || glu_mode() == 4;
}

// the register prefetch: off at 2 and 4
static bool glu_pf() {
    return glu_mode() != 2 && glu_mode() != 4;
}

static int glu_np_env() {
    static const int np = glu_env("GGML_CUDA_MMVQ_PASCAL_GLU_MT_NP", 0);
    return np;
}

static int glu_fault() {
    static const int fault = glu_env("GGML_CUDA_MMVQ_PASCAL_GLU_MT_FAULT", 0) != 0;
    return fault;
}

// Whether the k-column kernels run this type's separate products (GGML_CUDA_MMVQ_PASCAL_KCOL not 0, the type not in
// _KCOL_SKIP; read as mmvq-pascal-kcol.cu reads them) and the Pascal kernels don't skip it (GGML_CUDA_MMVQ_PASCAL_SKIP:
// at 2 tokens the dispatch table may send Q2_0 to them)
static bool glu_kcol_runs(const ggml_type type) {
    static const int mode = glu_env("GGML_CUDA_MMVQ_PASCAL_KCOL", 1);
    static const std::vector<ggml_type> skip = [] {
        std::vector<ggml_type> types;
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP");
        const std::string list = env ? env : "";
        size_t a = 0;
        while (a < list.size()) {
            size_t b = list.find(',', a);
            if (b == std::string::npos) {
                b = list.size();
            }
            const std::string name = list.substr(a, b - a);
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                const char * tn = ggml_type_name((ggml_type) t);
                if (tn && name == tn) {
                    types.push_back((ggml_type) t);
                }
            }
            a = b + 1;
        }
        return types;
    }();
    return mode != 0 && std::find(skip.begin(), skip.end(), type) == skip.end() && !ggml_cuda_mmvq_pascal_skips(type);
}

static int glu_kind_of(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:   return GK_Q2_0;
        case GGML_TYPE_Q3_K:   return GK_Q3_K;
        case GGML_TYPE_Q4_K:   return GK_Q4_K;
        case GGML_TYPE_Q5_K:   return GK_Q5_K;
        case GGML_TYPE_IQ4_XS: return GK_IQ4_XS;
        default:               return -1;
    }
}

// log2 of the lanes per row: the largest power of two, at most 32, that divides the row's units (kcol::pick_shape's and
// dmt_lpr_log2's rule)
static int glu_lpr_log2(const int nunits) {
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && nunits % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    return lpr_log2;
}

// Whether the kernel's sums for weight matrix w (2-D, contiguous rows) of ncols tokens are those of the separate product:
// the type's kernel of that product is the one this kernel repeats.
static bool glu_matrix_ok(const ggml_tensor * w, const int64_t ncols) {
    const int kind = glu_kind_of(w->type);
    if (kind < 0 || w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != ggml_type_size(w->type)) {
        return false;
    }
    const int64_t K = w->ne[0];
    const int64_t nrows = w->ne[1];
    const int64_t row_bytes = w->nb[1];
    const int al = glu_al(kind);
    if (K % glu_uw(kind) != 0 || K > INT_MAX || nrows > INT_MAX || row_bytes > INT_MAX ||
            row_bytes < K / glu_uw(kind) * glu_ub(kind) || row_bytes % al != 0 || (uintptr_t) w->data % al != 0) {
        return false;
    }
    // the units a lane stages per step fit its registers (glu_maxu)
    const int lpr_log2 = glu_lpr_log2((int) (K / glu_uw(kind)));
    const int hmax     = al >= 16 ? 0 : 16 - al;
    const int nunit16  = (hmax + (glu_ub(kind) << lpr_log2) + 15) / 16;
    if (((nunit16 + (1 << lpr_log2) - 1) >> lpr_log2) > glu_maxu_host(kind)) {
        return false;
    }
    switch (kind) {
        case GK_Q2_0:
        case GK_Q3_K:
            return glu_kcol_runs(w->type);
        // 5 to 8 tokens at GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 or 4 only (C793), where _RO8 runs the product read once
        case GK_Q4_K:
        case GK_IQ4_XS:
            return (ncols <= 4 || glu_ro8()) &&
                   ggml_cuda_mmvq_pascal_dmt_takes(w->type, K, nrows, ncols, row_bytes, w->data);
        default: // GK_Q5_K
            return (ncols <= 4 || glu_ro8()) && ggml_cuda_mmvq_pascal_q5k_mt_takes(K, nrows, ncols, row_bytes, w->data);
    }
}

// Bytes of a lane group's row slot for a type at L = 1 << lpr_log2 lanes per row: the segment from the 16-byte boundary
// below it plus the bytes a 2-byte-aligned unit reads past its end (kcol::unit_words, as the k-column kernel's row
// slots), in 16-byte units, and the padding (0 to 7 units) under which the lanes' reads of their units (at 16-byte-aligned
// rows) meet the fewest bank conflicts.
static int glu_slot(const int kind, const int lpr_log2) {
    static int cache[GK_COUNT][6] = {};
    int & c = cache[kind][lpr_log2];
    if (c > 0) {
        return c;
    }
    const int ub  = glu_ub(kind);
    const int al  = glu_al(kind);
    const int lpr = 1 << lpr_log2;
    const int seg = ub << lpr_log2;
    const int nu  = (15 + seg + 8 + 15) / 16;
    const int w   = al >= 16 ? 16 : al == 8 ? 8 : 4; // bytes per lane and read
    const int nl  = 128 / w;                         // lanes served together
    int best = nu * 16;
    int best_cost = INT_MAX;
    for (int pad = 0; pad < 8; ++pad) {
        const int slot = (nu + pad) * 16;
        int cost = 0;
        for (int l0 = 0; l0 < WARP_SIZE; l0 += nl) {
            int worst = 0;
            for (int bank = 0; bank < nl; ++bank) {
                int addrs[WARP_SIZE];
                int n = 0;
                for (int lane = l0; lane < l0 + nl; ++lane) {
                    const int a = ((lane >> lpr_log2) * slot + (lane & (lpr - 1)) * ub) / w * w;
                    if ((a / w) % nl != bank) {
                        continue;
                    }
                    bool seen = false;
                    for (int i = 0; i < n; ++i) {
                        seen = seen || addrs[i] == a;
                    }
                    if (!seen) {
                        addrs[n++] = a;
                    }
                }
                worst = std::max(worst, n);
            }
            cost += worst;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best = slot;
        }
    }
    c = best;
    return c;
}

struct glu_shape {
    int lpr_log2[2];
    int slot[2];
    int rbr_log2;
    int act;      // bytes of the staged activations
    int pstage;   // bytes of a pair's staging buffers
    int wstage0;
    int ngroups;
};

static glu_shape glu_shape_of(const int kg, const int ku, const int64_t K, const int nt, const bool need_s, const bool need_q3) {
    glu_shape s;
    const int kinds[2] = { kg, ku };
    int rbr_log2 = 0;
    int ws[2];
    for (int r = 0; r < 2; ++r) {
        s.lpr_log2[r] = glu_lpr_log2((int) (K / glu_uw(kinds[r])));
        s.slot[r]     = glu_slot(kinds[r], s.lpr_log2[r]);
        ws[r]         = (WARP_SIZE >> s.lpr_log2[r]) * s.slot[r];
        rbr_log2      = std::max(rbr_log2, 5 - s.lpr_log2[r]);
    }
    s.rbr_log2 = rbr_log2;
    s.wstage0  = ws[0];
    s.pstage   = ws[0] + ws[1];
    s.ngroups  = (int) (K / 128);
    const int qg = 32 * nt + 4;
    const int dg = 4 * nt + 1;
    const int aq  = s.ngroups * qg * 4;
    const int a3h = need_q3 ? s.ngroups * dg * 8 : 0;
    const int ad  = s.ngroups * dg * 4 * (1 + (need_s ? 1 : 0) + (need_q3 ? 1 : 0));
    s.act = (aq + a3h + ad + 15) / 16 * 16;
    return s;
}

static int glu_smem(const glu_shape & s, const int np, const int nt) {
    return s.act + np * s.pstage + np * 4 * nt * (1 << s.rbr_log2) * (int) sizeof(float);
}

// the kernel of NT tokens: W, a pair with an IQ4_XS, Q4_K or Q5_K product of 5 to 8 tokens (C793)
template <int NT, bool PF, bool W>
static void (*glu_kernel())(const glu_args) {
    if constexpr (W) {
        return mul_mat_vec_q_pascal_glu_ro8<NT, PF>;
    } else {
        return mul_mat_vec_q_pascal_glu<NT, PF>;
    }
}

template <int NT, bool PF, bool W>
static int glu_blocks_regs(const int threads) {
    // resident blocks per SM that the kernel's registers allow, per block size (np 1 to 4), queried once
    static int cache[GLU_NP_MAX + 1] = {};
    const int np = threads / (2 * WARP_SIZE);
    if (cache[np] == 0) {
        int n = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, glu_kernel<NT, PF, W>(), threads, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            n = 1;
        }
        cache[np] = std::max(n, 1);
    }
    return cache[np];
}

template <int NT, bool PF, bool W>
static void glu_launch(const glu_args & p, const int nsm, const int smem, cudaStream_t stream) {
    const int threads = 2 * p.np * WARP_SIZE;
    const int bps  = std::max(1, std::min(glu_blocks_regs<NT, PF, W>(threads), 96 * 1024 / smem));
    const int nblk = std::min((p.nrb + p.np - 1) / p.np, nsm * bps);
    glu_kernel<NT, PF, W>()<<<dim3(nblk, 1, 1), dim3(WARP_SIZE, 2 * p.np, 1), smem, stream>>>(p);
}

template <int NT, bool W = false>
static void glu_launch_pf(const glu_args & p, const int nsm, const int smem, const bool pf, cudaStream_t stream) {
    if (pf) {
        glu_launch<NT, true, W>(p, nsm, smem, stream);
    } else {
        glu_launch<NT, false, W>(p, nsm, smem, stream);
    }
}

// 5 to 8 tokens: W as above
template <int NT>
static void glu_launch_nt(const glu_args & p, const int nsm, const int smem, const bool pf, const bool w,
                          cudaStream_t stream) {
    if (w) {
        glu_launch_pf<NT, true>(p, nsm, smem, pf, stream);
    } else {
        glu_launch_pf<NT, false>(p, nsm, smem, pf, stream);
    }
}

} // namespace

bool ggml_cuda_mmvq_pascal_glu_mt_on() {
    return glu_mode() != 0;
}

bool ggml_cuda_mmvq_pascal_glu_mt(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
                                  ggml_tensor * glu) {
    if (glu_mode() == 0 || !ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
            glu->src[0] != gate || glu->src[1] != up) {
        return false;
    }
    const ggml_tensor * wg   = gate->src[0];
    const ggml_tensor * wu   = up->src[0];
    const ggml_tensor * src1 = gate->src[1];
    if (up->src[1] != src1 || src1->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 ||
            glu->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K     = wg->ne[0];
    const int64_t nrows = wg->ne[1];
    const int64_t ncols = src1->ne[1];
    if (ncols < 2 || ncols > GLU_MAX_COLS || wu->ne[0] != K || wu->ne[1] != nrows || src1->ne[0] != K ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 ||
            glu->ne[0] != nrows || glu->ne[1] != ncols || glu->ne[2] != 1 || glu->ne[3] != 1 ||
            glu->nb[0] != sizeof(float) || glu->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (!glu_matrix_ok(wg, ncols) || !glu_matrix_ok(wu, ncols)) {
        return false;
    }
    const int kg = glu_kind_of(wg->type);
    const int ku = glu_kind_of(wu->type);
    const bool need_s  = kg == GK_Q4_K || kg == GK_Q5_K || ku == GK_Q4_K || ku == GK_Q5_K;
    const bool need_q3 = kg == GK_Q3_K || ku == GK_Q3_K;
    const int  nt = (int) ncols;
    const glu_shape s = glu_shape_of(kg, ku, K, nt, need_s, need_q3);

    // warp pairs per block: GGML_CUDA_MMVQ_PASCAL_GLU_MT_NP, else the most (up to 4) that fit 48 KB
    int np = glu_np_env();
    if (np < 1 || np > GLU_NP_MAX) {
        np = GLU_NP_MAX;
        while (np > 1 && glu_smem(s, np, nt) > GLU_SMEM_MAX) {
            np--;
        }
    }
    const int smem = glu_smem(s, np, nt);
    if (smem > GLU_SMEM_MAX) {
        return false;
    }

    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int64_t col_bytes_y = ne10_padded / QK8_1 * (int64_t) sizeof(block_q8_1);
    const int64_t stride_col_dst = glu->nb[1] / sizeof(float);
    if ((ncols - 1) * col_bytes_y + K / QK8_1 * (int64_t) sizeof(block_q8_1) > INT_MAX ||
            (ncols - 1) * stride_col_dst + nrows > INT_MAX) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    // the activations as ggml_cuda_mul_mat_vec_q quantizes them
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ncols * ne10_padded * sizeof(block_q8_1) / QK8_1);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, src1_q8_1.get(), K, s11, s12, s13,
                               ne10_padded, ncols, 1, 1, stream);
    }

    glu_args p;
    const ggml_tensor * ws[2] = { wg, wu };
    const int kinds[2] = { kg, ku };
    for (int r = 0; r < 2; ++r) {
        glu_mat & m = p.m[r];
        m.vx        = (const char *) ws[r]->data;
        m.row_bytes = (int) ws[r]->nb[1];
        m.kind      = kinds[r];
        m.lpr_log2  = s.lpr_log2[r];
        m.niter     = (int) (K / glu_uw(kinds[r])) >> s.lpr_log2[r];
        m.npass     = 1 << (s.rbr_log2 - (5 - s.lpr_log2[r]));
        m.slot      = s.slot[r];
    }
    p.vy             = src1_q8_1.get();
    p.col_bytes_y    = (int) col_bytes_y;
    p.dst            = (float *) glu->data;
    p.stride_col_dst = (int) stride_col_dst;
    p.nrows          = (int) nrows;
    p.nab            = (int) (K / QK8_1);
    p.rbr_log2       = s.rbr_log2;
    p.rbr            = 1 << s.rbr_log2;
    p.nrb            = (int) ((nrows + p.rbr - 1) / p.rbr);
    p.np             = np;
    const int qg     = 32 * nt + 4;
    const int dg     = 4 * nt + 1;
    const int aq_b   = s.ngroups * qg * 4;
    const int a3h_b  = need_q3 ? s.ngroups * dg * 8 : 0;
    p.a3h_off        = aq_b / 8;
    p.ad_off         = (aq_b + a3h_b) / 4;
    p.as_off         = p.ad_off + s.ngroups * dg;
    p.a3d_off        = p.as_off + (need_s ? s.ngroups * dg : 0);
    p.need_s         = need_s;
    p.need_q3        = need_q3;
    p.stage_off      = s.act;
    p.pstage         = s.pstage;
    p.wstage0        = s.wstage0;
    p.res_off        = s.act + np * s.pstage;
    p.fault          = glu_fault();

    const int  nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const bool pf  = glu_pf();
    // 5 to 8 tokens with an IQ4_XS, Q4_K or Q5_K product (glu_matrix_ok took it under _GLU_MT=3 or 4): their kernel
    const bool w   = kg >= GK_Q4_K || ku >= GK_Q4_K;
    switch (nt) {
        case 2:  glu_launch_pf<2>(p, nsm, smem, pf, stream); break;
        case 3:  glu_launch_pf<3>(p, nsm, smem, pf, stream); break;
        case 4:  glu_launch_pf<4>(p, nsm, smem, pf, stream); break;
        case 5:  glu_launch_nt<5>(p, nsm, smem, pf, w, stream); break;
        case 6:  glu_launch_nt<6>(p, nsm, smem, pf, w, stream); break;
        case 7:  glu_launch_nt<7>(p, nsm, smem, pf, w, stream); break;
        default: glu_launch_nt<8>(p, nsm, smem, pf, w, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_pascal_glu_mt_on() {
    return false;
}

bool ggml_cuda_mmvq_pascal_glu_mt(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
                                  ggml_tensor * glu) {
    GGML_UNUSED_VARS(ctx, gate, up, glu);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
