#include "mmvq-pascal-dmt.cuh"
#include "mmvq-pascal.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-impl.cuh"

// How the kernels work (2 to 4 tokens of a plain IQ4_XS or Q4_K product; C690's Q5_K kernel, mmvq-pascal-q5k.cu, with
// each type's own super-block step):
//   - The lanes and summation order are mul_mat_vec_q_pascal's for the type (mmvqp_choose: C = 1, staged): L =
//     2^lpr_log2 lanes per row (the largest power of two, at most 32, dividing the row's super-blocks), lane sl of a
//     row's group taking super-blocks sl, sl + L, ... in that order, its segment staged by the lane group into the warp's
//     shared memory with 128-bit loads, and the lanes' sums added with the same xor shuffles. So every token's result is
//     the one-token kernel's, bit for bit. A lane group owns R = 1 (default) or 2 consecutive rows (dmt_pick_r); the rows
//     of a group share every load of the activations, and R changes no row's sum.
//   - Each 32-weight sub-block of a row is decoded once into eight words of int8 weights and multiplied with dp4a by
//     every token's activations: IQ4_XS's 4-bit indices looked up in kvalues_iq4nl with __byte_perm, Q4_K's nibbles
//     (the high ones shifted down, whose sum the one-token kernel takes 16 times too large and shifts back).
//   - The floating-point steps are the one-token kernel's, per sub-block s and token:
//       IQ4_XS (mmvqp_dot_iq4_xs): sumf = fmaf(d8, (float) (isum * (ls - 32)), sumf); per super-block fmaf(d, sumf, acc);
//       Q4_K (mmvqp_dot_q4_K): sumd = fmaf(d8, (float) (isum * sc), sumd) and summ = fmaf((float) m, S, summ), S = d8 *
//       (float) sum(int8); per super-block acc + d * sumd - dmin * summ.
//     The integer-to-float conversions are made without the quarter-rate I2F and the integer multiply, exactly: the dp4a
//     chain starts at 0x4B400000, so its result read as a float is 12582912 + isum (|isum| <= 32 * 127 * 127 < 2^22 for
//     IQ4_XS, 32 * 15 * 127 for Q4_K), and fmaf(12582912 + isum, f, -12582912 * f) is isum * f rounded once, which is
//     exact (|isum * (ls - 32)| <= 516128 * 32 < 2^24, |isum * sc| <= 60960 * 63); the 6-bit ls, sc and m come out of a
//     __byte_perm or an OR into the same float. These are the values the one-token kernel's conversions give, so every
//     FP32 operation sees the same operands.
//   - Each block stages all tokens' activations once into shared memory (the int8 words, IQ4_XS's in the one-token
//     kernel's lookup order, and per sub-block and token d8, Q4_K also S; padded so that the L lanes of a group read
//     distinct banks; every lane group of a warp reads the same L records, a broadcast) and loops over row blocks (a
//     persistent grid of about 2 blocks per SM), so the staging and its block sums serve many rows. At 4 tokens: K = 2560
//     is 11.5 KB (IQ4_XS) or 12.7 KB (Q4_K) beside 26-28.5 KB of weight staging at R = 2 (3 warps); K = 6144 is 27.5 KB
//     or 30.6 KB, beside 2 warps at R = 2 (IQ4_XS) or 3 at R = 1 (Q4_K). The k-column kernel stages each block's
//     activation records again for every 64 rows, in tiles.
//   - The lane groups' segments are padded so that the lanes' reads of their super-blocks (Q4_K: 128-bit, IQ4_XS's
//     136-byte super-blocks: 64-bit) hit distinct banks (dmt_gstride).
//   - Each warp loads its next segments into registers before it computes the current ones, but where dmt_pick_pf says.
//   - Q6_K (C727, GGML_CUDA_MMVQ_PASCAL_Q6K_MT): the one-token kernel's step (mmvqp_dot_q6_K) is, per sub-block s and
//     token, sumf = fmaf(d8, (float) (sc0 * (isum0 - t0) + sc1 * (isum1 - t1)), sumf), isum0 and isum1 the dp4a sums
//     of the 6-bit codes (0..63) and the activations of the sub-block's two halves, t0 and t1 32 times the halves' int8
//     sums; per super-block fmaf(d, sumf, acc). Here each half's dp4a chain starts at 0x4B400000 - t (staged per token
//     beside d8), so its result reads as the float 12582912 + (isum - t), |isum - t| <= 16 * 32 * 128 = 2^16;
//     fmaf(12582912 + (isum - t), sc, -12582912 * sc) is sc * (isum - t) rounded once, which is exact (|.| <= 2^23),
//     and the two halves' sum is the integer expression exactly (|.| <= 2^24). So no I2F and no integer multiply, and
//     every FP32 operation of the one-token kernel sees the same operands. The int8 scales come out exactly as the
//     float of 12582912 with (scale ^ 0x80) in its low mantissa bits, less 12583040.
//     Its 210-byte super-blocks are 2-byte aligned. The segments are staged in 16-byte units as the other types' (rows
//     of a multiple of 8 super-blocks, 16-byte aligned: Flash-Next's attn_output and ssm_out, 6144 -> 2560, 8 lanes per
//     row), and each lane reads its super-block from the staging with 32-bit loads of the aligned words around it,
//     shifted by 2 bytes where it starts at an odd half-word, as the one-token kernel reads its staging (mmvqp_lds_h);
//     the lane groups' segments are padded so that those reads hit distinct banks. At 4 tokens and K = 6144 the
//     activations take 33.6 KB, beside 2 warps at R = 1. A product of one warp a block is left to the k-column kernel,
//     as C726's kernel leaves it (R79: one warp a block lost to the k-column kernel on a P106), unless _DMT_NW=1.
//   - 5 to 8 tokens (GGML_CUDA_MMVQ_PASCAL_RO8, C729): the same kernel at NT = 5 to 8 and R = 1 only, so a token's sums
//     are still the one-token kernel's. Its activations take about NT / 4 times the 4-token staging: at K = 2560 and 8
//     tokens 20.6 KB of int8 words and 2.6 KB (IQ4_XS) or 5.2 KB (Q4_K) of terms, beside 5 warps' weight staging; K = 6144
//     fits only 5 tokens (IQ4_XS 6), at 1 to 3 warps, wider rows fewer (dmt_nw); one warp a block (C809) and the rest are
//     left to the k-column kernel. IQ4_XS and Q4_K only: Q6_K's 5 to 8 tokens stay there too (dmt_nt_mode; C795, Q1390).

#define DMT_NW_MAX   6                    // warps per block at R = 1, at most (3 at R = 2)
#define DMT_MAX_COLS 4                    // tokens with GGML_CUDA_MMVQ_PASCAL_DMT and _Q6K_MT
#define DMT_RO8_COLS 8                    // tokens with GGML_CUDA_MMVQ_PASCAL_RO8 (from 5)
#define DMT_NU_MAX   14                   // 16-byte units of a super-block, rounded up (Q6_K 13.1, Q4_K 9, IQ4_XS 8.5)
#define DMT_SMEM_MAX (48 * 1024)

static_assert(sizeof(block_q4_K) == 144 && sizeof(block_iq4_xs) == 136 && sizeof(block_q6_K) == 210,
              "the staging assumes these super-blocks");

namespace {

// Per type: SB bytes of a super-block, DSW words per sub-block and token of the staged activation terms (d8, and for
// Q4_K S, for Q6_K the two halves' dp4a starts).
template <ggml_type type> struct dmt_traits;
template <> struct dmt_traits<GGML_TYPE_Q4_K>   { static constexpr int SB = 144; static constexpr int DSW = 2; };
template <> struct dmt_traits<GGML_TYPE_IQ4_XS> { static constexpr int SB = 136; static constexpr int DSW = 1; };
template <> struct dmt_traits<GGML_TYPE_Q6_K>   { static constexpr int SB = 210; static constexpr int DSW = 3; };

static int dmt_sb(const ggml_type type) {
    return type == GGML_TYPE_Q4_K ? dmt_traits<GGML_TYPE_Q4_K>::SB :
           type == GGML_TYPE_Q6_K ? dmt_traits<GGML_TYPE_Q6_K>::SB : dmt_traits<GGML_TYPE_IQ4_XS>::SB;
}
static int dmt_dsw(const ggml_type type) {
    return type == GGML_TYPE_Q4_K ? dmt_traits<GGML_TYPE_Q4_K>::DSW :
           type == GGML_TYPE_Q6_K ? dmt_traits<GGML_TYPE_Q6_K>::DSW : dmt_traits<GGML_TYPE_IQ4_XS>::DSW;
}

// Bytes of shared memory past the warps' staging buffers: Q6_K's 32-bit reads of its 2-byte-aligned super-blocks reach
// up to 4 bytes past the last one (mmvqp_lds_h's padding).
static int dmt_tail(const ggml_type type) {
    return type == GGML_TYPE_Q6_K ? 16 : 0;
}

struct dmt_args {
    const char * vx;
    const char * vy;            // block_q8_1, column j from j * col_bytes_y
    float      * dst;           // column j from j * stride_col_dst
    int          nrows;
    int          row_bytes;
    int          nsb;           // super-blocks per row
    int          lpr_log2;      // lanes per row L = 1 << lpr_log2
    int          niter;         // super-blocks per lane and row (nsb / L)
    int          nwrb;          // warp row blocks: rows / ((32 >> lpr_log2) * R), rounded up
    int          seg_units;     // 16-byte units of a row's segment (SB L / 16)
    int          col_bytes_y;
    int          stride_col_dst;
    int          ds_off;        // words from the start of shared memory to the activation terms
    int          stage_off;     // bytes from the start of shared memory to the warps' staging buffers
    int          wstage;        // bytes of a warp's staging buffer
    int          gstride;       // bytes between two lane groups' segments in it (dmt_gstride)
};

// Words per super-block position of the staged activations: NT tokens x 8 sub-blocks of 8 int8 words, and of the terms
// (DSW words each), each padded so that consecutive positions start in other banks: an odd number of 4-word groups apart
// for the int8 words (128-bit reads), an odd number of 2-word (DSW = 2) or 1-word (DSW = 1) units for the terms. Q6_K
// (DSW = 3): the d8 of the 8 NT records, then their pairs of dp4a starts (64-bit reads), an odd number of 2-word units.
template <int NT> constexpr __host__ __device__ int dmt_q_words() { return 64 * NT + 4; }
template <ggml_type type, int NT> constexpr __host__ __device__ int dmt_ds_words() {
    return dmt_traits<type>::DSW == 3 ? 24 * NT + 2 : dmt_traits<type>::DSW == 2 ? 16 * NT + 2 : 8 * NT + 1;
}

static int dmt_act_bytes(const ggml_type type, const int nt, const int nsb) {
    const int q  = nsb * (64 * nt + 4) * 4;
    const int dw = dmt_dsw(type) == 3 ? 24 * nt + 2 : dmt_dsw(type) == 2 ? 16 * nt + 2 : 8 * nt + 1;
    const int ds = (nsb * dw * 4 + 15) / 16 * 16;
    return q + ds;
}

// Every token's activation blocks into shared memory, by all warps of the block: the int8 words (IQ4_XS: regrouped as
// mmvqp_load_act regroups them for its lookup) and per sub-block d8 (Q4_K also S = d8 * the sum of the int8 values, as
// mmvqp_load_act computes it; Q6_K also the starts of its two halves' dp4a chains, 0x4B400000 less mmvqp_load_act's
// t[0] and t[1], 32 times the halves' sums).
template <ggml_type type, int NT>
static __device__ __forceinline__ void dmt_stage_act(const dmt_args & p, uint32_t * aq, float * ads) {
    constexpr int QW = dmt_q_words<NT>();
    constexpr int DW = dmt_ds_words<type, NT>();
    const int nab = p.nsb * 8;
    const int nth = blockDim.x * blockDim.y;
    for (int e = threadIdx.y * WARP_SIZE + threadIdx.x; e < NT * nab; e += nth) {
        const int j = e / nab;
        const int b = e - j * nab;
        const uint32_t * src = (const uint32_t *) (p.vy + j * p.col_bytes_y + b * (int) sizeof(block_q8_1));
        uint32_t w[9];
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            w[k] = __ldg(src + k);
        }
        const float d8 = mmvqp_h2f_lo(w[0]);
        const int   sb = b >> 3;
        const int   s  = b & 7;
        uint32_t q[8];
        if constexpr (type == GGML_TYPE_IQ4_XS) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                q[2*k + 0] = __byte_perm(w[1 + k], w[5 + k], 0x5140);
                q[2*k + 1] = __byte_perm(w[1 + k], w[5 + k], 0x7362);
            }
            ads[sb * DW + s*NT + j] = d8;
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            int s0 = 0;
            int s1 = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                q[k]     = w[1 + k];
                q[k + 4] = w[5 + k];
                s0 = ggml_cuda_dp4a(0x01010101, (int) w[1 + k], s0);
                s1 = ggml_cuda_dp4a(0x01010101, (int) w[5 + k], s1);
            }
            ads[sb * DW + s*NT + j] = d8;
            *(int2 *) (ads + sb * DW + 8*NT + (s*NT + j) * 2) = make_int2(0x4B400000 - (s0 << 5), 0x4B400000 - (s1 << 5));
        } else {
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                q[k] = w[1 + k];
                sum  = ggml_cuda_dp4a(0x01010101, (int) w[1 + k], sum);
            }
            *(float2 *) (ads + sb * DW + (s*NT + j) * 2) = make_float2(d8, d8 * (float) sum);
        }
        uint4 * dq = (uint4 *) (aq + sb * QW + (s*NT + j) * 8);
        dq[0] = make_uint4(q[0], q[1], q[2], q[3]);
        dq[1] = make_uint4(q[4], q[5], q[6], q[7]);
    }
}

// IQ4_XS: the super-blocks of R rows at one position (sw[i]: row i's 136 bytes in shared memory, 8-byte aligned) times
// every token's activations at that position (aq: the int8 words, ad: d8), added to acc[i][j] exactly as
// mmvqp_dot_iq4_xs adds row i's super-block for token j alone.
template <int NT, int R>
static __device__ __forceinline__ void dmt_superblock_iq4_xs(const char * const (&sw)[R], const uint32_t * aq,
                                                             const float * ad, float (&acc)[R][NT]) {
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u; // kvalues_iq4nl
    uint32_t h[R][2]; // d | scales_h << 16, scales_l
#pragma unroll
    for (int i = 0; i < R; ++i) {
        mmvqp_lds_words<8, 8>(h[i], sw[i]);
    }
    float sumf[R][NT];
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            sumf[i][j] = 0.0f;
        }
    }
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        int   q[R][8];
        float lsf[R];
        float cneg[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uint32_t c[4];
            mmvqp_lds_words<16, 8>(c, sw[i] + 8 + 16*ib);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t sel = 0x32103210 | ((c[k] & 0x88888888) >> 1);
                q[i][2*k + 0] = (int) __byte_perm(__byte_perm(t0, t1, c[k]),       __byte_perm(t2, t3, c[k]),       sel);
                q[i][2*k + 1] = (int) __byte_perm(__byte_perm(t0, t1, c[k] >> 16), __byte_perm(t2, t3, c[k] >> 16), sel >> 16);
            }
            // (float) (ls - 32) exactly: the 6-bit ls as the low mantissa bits of 12582912 = 2^23 + 2^22
            const uint32_t ls = ((h[i][1] >> (4*ib)) & 0xF) | (((h[i][0] >> (16 + 2*ib)) & 3) << 4);
            lsf[i]  = __int_as_float((int) (0x4B400000u | ls)) - 12582944.0f;
            cneg[i] = -12582912.0f * lsf[i];
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const uint4 u0 = *(const uint4 *) (aq + (ib*NT + j)*8);
            const uint4 u1 = *(const uint4 *) (aq + (ib*NT + j)*8 + 4);
            const float d8 = ad[ib*NT + j];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                int ia = 0x4B400000; // 12582912 as a float: the result reads as 12582912 + isum
                ia = ggml_cuda_dp4a(q[i][0], (int) u0.x, ia);
                ia = ggml_cuda_dp4a(q[i][1], (int) u0.y, ia);
                ia = ggml_cuda_dp4a(q[i][2], (int) u0.z, ia);
                ia = ggml_cuda_dp4a(q[i][3], (int) u0.w, ia);
                ia = ggml_cuda_dp4a(q[i][4], (int) u1.x, ia);
                ia = ggml_cuda_dp4a(q[i][5], (int) u1.y, ia);
                ia = ggml_cuda_dp4a(q[i][6], (int) u1.z, ia);
                ia = ggml_cuda_dp4a(q[i][7], (int) u1.w, ia);
                const float x = fmaf(__int_as_float(ia), lsf[i], cneg[i]); // (float) (isum * (ls - 32)), exactly
                sumf[i][j] = fmaf(d8, x, sumf[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float d = mmvqp_h2f_lo(h[i][0]);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j] = fmaf(d, sumf[i][j], acc[i][j]);
        }
    }
}

// Q4_K: the super-blocks of R rows at one position (sw[i]: row i's 144 bytes, 16-byte aligned) times every token's
// activations at that position (aq: the int8 words, ads: (d8, S)), added to acc[i][j] exactly as mmvqp_dot_q4_K adds row
// i's super-block for token j alone.
template <int NT, int R>
static __device__ __forceinline__ void dmt_superblock_q4_K(const char * const (&sw)[R], const uint32_t * aq,
                                                           const float * ads, float (&acc)[R][NT]) {
    uint32_t h[R][4]; // dm, 3 words of scales
    uint32_t sc_lo[R], m_lo[R], sc_hi[R], m_hi[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        mmvqp_lds_words<16, 16>(h[i], sw[i]);
        sc_lo[i] = h[i][1] & 0x3f3f3f3f;                                         // scales of sub-blocks 0..3
        m_lo[i]  = h[i][2] & 0x3f3f3f3f;                                         // mins of sub-blocks 0..3
        sc_hi[i] = (h[i][3] & 0x0f0f0f0f) | ((h[i][1] >> 2) & 0x30303030);      // scales of 4..7
        m_hi[i]  = ((h[i][3] >> 4) & 0x0f0f0f0f) | ((h[i][2] >> 2) & 0x30303030);
    }
    float sumd[R][NT];
    float summ[R][NT];
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            sumd[i][j] = 0.0f;
            summ[i][j] = 0.0f;
        }
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t qs[R][8];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            mmvqp_lds_words<32, 16>(qs[i], sw[i] + 16 + 32*p);
        }
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int s = 2*p + hh;
            int   q[R][8];
            float scf[R];
            float mf[R];
            float cneg[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    q[i][k] = (int) (hh ? (qs[i][k] >> 4) & 0x0F0F0F0F : qs[i][k] & 0x0F0F0F0F);
                }
                // (float) sc and (float) m exactly: the 6-bit byte as the low mantissa bits of 12582912 = 2^23 + 2^22
                const uint32_t sel = 0x7650 | (s & 3);
                scf[i]  = __int_as_float((int) __byte_perm(s < 4 ? sc_lo[i] : sc_hi[i], 0x4B400000u, sel)) - 12582912.0f;
                mf[i]   = __int_as_float((int) __byte_perm(s < 4 ? m_lo[i]  : m_hi[i],  0x4B400000u, sel)) - 12582912.0f;
                cneg[i] = -12582912.0f * scf[i];
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4  u0 = *(const uint4  *) (aq + (s*NT + j)*8);
                const uint4  u1 = *(const uint4  *) (aq + (s*NT + j)*8 + 4);
                const float2 ds = *(const float2 *) (ads + (s*NT + j)*2);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    int ia = 0x4B400000; // 12582912 as a float: the result reads as 12582912 + isum
                    ia = ggml_cuda_dp4a(q[i][0], (int) u0.x, ia);
                    ia = ggml_cuda_dp4a(q[i][1], (int) u0.y, ia);
                    ia = ggml_cuda_dp4a(q[i][2], (int) u0.z, ia);
                    ia = ggml_cuda_dp4a(q[i][3], (int) u0.w, ia);
                    ia = ggml_cuda_dp4a(q[i][4], (int) u1.x, ia);
                    ia = ggml_cuda_dp4a(q[i][5], (int) u1.y, ia);
                    ia = ggml_cuda_dp4a(q[i][6], (int) u1.z, ia);
                    ia = ggml_cuda_dp4a(q[i][7], (int) u1.w, ia);
                    const float x = fmaf(__int_as_float(ia), scf[i], cneg[i]); // (float) (isum * sc), exactly
                    sumd[i][j] = fmaf(ds.x, x, sumd[i][j]);
                    summ[i][j] = fmaf(mf[i], ds.y, summ[i][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float d    = mmvqp_h2f_lo(h[i][0]);
        const float dmin = mmvqp_h2f_hi(h[i][0]);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j] = acc[i][j] + d * sumd[i][j] - dmin * summ[i][j];
        }
    }
}

// Words i0 .. i0 + N - 1 of a super-block at a 2-byte boundary p in shared memory, as mmvqp_lds_h gives them: the
// 4-byte-aligned words around them, shifted by 2 bytes where p is at an odd half-word (one read more than N).
template <int N>
static __device__ __forceinline__ void dmt_lds_h(uint32_t (&w)[N], const char * p, const int i0) {
    const int        mis = (int) ((uintptr_t) p & 2);
    const uint32_t * aw  = (const uint32_t *) (p - mis) + i0; // pointer arithmetic keeps the shared address space
    const uint32_t   sel = mis ? 0x5432 : 0x3210;
    uint32_t cur = aw[0];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const uint32_t nxt = aw[i + 1];
        w[i] = __byte_perm(cur, nxt, sel);
        cur = nxt;
    }
}

// Q6_K: the super-blocks of R rows at one position (sw[i]: row i's 210 bytes, 2-byte aligned) times every token's
// activations at that position (aq: the int8 words, ads: d8, then the dp4a starts), added to acc[i][j] exactly as
// mmvqp_dot_q6_K adds row i's super-block for token j alone. Its words (mmvqp_dot_q6_K's w): ql 0..31, qh 32..47,
// scales 48..51, d in the low half of 52; sub-block s = 4h + k takes the low (k < 2) or high nibbles of ql words
// 16h + 8(k & 1) + j, bits 2k, 2k + 1 of qh words 32 + 8h + j, and scales[2s], scales[2s + 1].
template <int NT, int R>
static __device__ __forceinline__ void dmt_superblock_q6_K(const char * const (&sw)[R], const uint32_t * aq,
                                                           const float * ads, float (&acc)[R][NT]) {
    float sumf[R][NT];
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            sumf[i][j] = 0.0f;
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t ql[R][16];
        uint32_t qh[R][8];
        uint32_t sc[R][2];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            dmt_lds_h<16>(ql[i], sw[i], 16*h);
            dmt_lds_h<8>(qh[i], sw[i], 32 + 8*h);
            dmt_lds_h<2>(sc[i], sw[i], 48 + 2*h);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int s = 4*h + k;
            int   q[R][8];
            float scf0[R], scf1[R];
            float cneg0[R], cneg1[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t l  = ql[i][8*(k & 1) + j];
                    const uint32_t hb = qh[i][j];
                    const uint32_t lo = k < 2 ? l & 0x0F0F0F0F : (l >> 4) & 0x0F0F0F0F;
                    const uint32_t hi = k == 0 ? (hb << 4) & 0x30303030 : k == 1 ? (hb << 2) & 0x30303030
                                      : k == 2 ? hb & 0x30303030 : (hb >> 2) & 0x30303030;
                    q[i][j] = (int) (lo | hi);
                }
                // (float) scales[2s] and (float) scales[2s + 1] exactly: (scale ^ 0x80) = scale + 128 as the low
                // mantissa bits of 12582912 = 2^23 + 2^22
                const uint32_t sx  = sc[i][k >> 1] ^ 0x80808080u;
                const uint32_t sel = 0x7650 | (2 * (k & 1));
                scf0[i]  = __int_as_float((int) __byte_perm(sx, 0x4B400000u, sel))     - 12583040.0f;
                scf1[i]  = __int_as_float((int) __byte_perm(sx, 0x4B400000u, sel + 1)) - 12583040.0f;
                cneg0[i] = -12582912.0f * scf0[i];
                cneg1[i] = -12582912.0f * scf1[i];
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4 u0 = *(const uint4 *) (aq + (s*NT + j)*8);
                const uint4 u1 = *(const uint4 *) (aq + (s*NT + j)*8 + 4);
                const float d8 = ads[s*NT + j];
                const int2  st = *(const int2 *) (ads + 8*NT + (s*NT + j)*2);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    // the results read as 12582912 + (isum0 - t0) and 12582912 + (isum1 - t1)
                    int ia0 = st.x;
                    ia0 = ggml_cuda_dp4a(q[i][0], (int) u0.x, ia0);
                    ia0 = ggml_cuda_dp4a(q[i][1], (int) u0.y, ia0);
                    ia0 = ggml_cuda_dp4a(q[i][2], (int) u0.z, ia0);
                    ia0 = ggml_cuda_dp4a(q[i][3], (int) u0.w, ia0);
                    int ia1 = st.y;
                    ia1 = ggml_cuda_dp4a(q[i][4], (int) u1.x, ia1);
                    ia1 = ggml_cuda_dp4a(q[i][5], (int) u1.y, ia1);
                    ia1 = ggml_cuda_dp4a(q[i][6], (int) u1.z, ia1);
                    ia1 = ggml_cuda_dp4a(q[i][7], (int) u1.w, ia1);
                    const float x0 = fmaf(__int_as_float(ia0), scf0[i], cneg0[i]); // sc0 * (isum0 - t0), exactly
                    const float x1 = fmaf(__int_as_float(ia1), scf1[i], cneg1[i]); // sc1 * (isum1 - t1), exactly
                    sumf[i][j] = fmaf(d8, x0 + x1, sumf[i][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        uint32_t dw[1];
        dmt_lds_h<1>(dw, sw[i], 52);
        const float d = mmvqp_h2f_lo(dw[0]);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j] = fmaf(d, sumf[i][j], acc[i][j]);
        }
    }
}

// Grid: (blocks, 1, 1), each looping over warp row blocks; block (32, NW). Dynamic shared memory: the staged activations
// (int8 words, then the terms), then the NW warps' staging buffers (R rows per lane group).
template <ggml_type type, int NT, int R, bool PF>
__launch_bounds__(DMT_NW_MAX*WARP_SIZE/R, 2)
static __global__ void mul_mat_vec_q_pascal_dmt(const dmt_args p) {
    constexpr int SB = dmt_traits<type>::SB;
    constexpr int NU = (SB + 15) / 16;              // 16-byte units per lane and row segment (the last may be partial)
    constexpr int QW = dmt_q_words<NT>();
    constexpr int DW = dmt_ds_words<type, NT>();
    extern __shared__ uint4 dmt_sm[];
    uint32_t * aq  = (uint32_t *) dmt_sm;
    float    * ads = (float *) dmt_sm + p.ds_off;

    const int lane = threadIdx.x;
    const int wid  = threadIdx.y;
    const int nw   = blockDim.y;

    ggml_cuda_pdl_sync();

    dmt_stage_act<type, NT>(p, aq, ads);
    __syncthreads();

    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int gpw      = WARP_SIZE >> lpr_log2;      // lane groups per warp
    const int seg      = SB << lpr_log2;             // bytes of a row's segment per iteration
    // the lane group's R row segments in the warp's staging buffer, and this lane's super-block in each
    char * smw = (char *) dmt_sm + p.stage_off + wid * p.wstage + grp * p.gstride;
    const char * sws[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        sws[i] = smw + i * seg + sl * SB;
    }
    const int wstride = gridDim.x * nw;

    int wrb = blockIdx.x * nw + wid;
    if (wrb >= p.nwrb) {
        return;
    }
    // the lane group's rows (clamped, so that the staging reads valid rows; their results are not written)
    const char * rw[R];
    const auto set_rows = [&](const char * (&r)[R], const int w) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            r[i] = p.vx + (int64_t) min((w * gpw + grp) * R + i, p.nrows - 1) * p.row_bytes;
        }
    };
    set_rows(rw, wrb);
    int it = 0;

    // this lane's units of a segment: sl + (k << lpr_log2), the last one only where the segment has it (IQ4_XS: 8.5
    // units per super-block)
    const auto has_unit = [&](const int k) {
        return k < NU - 1 || sl + (k << lpr_log2) < p.seg_units;
    };

    [[maybe_unused]] uint4 v[R][NU];
    if constexpr (PF) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int k = 0; k < NU; ++k) {
                if (has_unit(k)) {
                    v[i][k] = __ldg((const uint4 *) rw[i] + sl + (k << lpr_log2));
                }
            }
        }
    }

    float acc[R][NT];
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    while (true) {
        // the current segments into the warp's staging buffer
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uint4 * dsm = (uint4 *) (smw + i * seg);
            if constexpr (PF) {
#pragma unroll
                for (int k = 0; k < NU; ++k) {
                    if (has_unit(k)) {
                        dsm[sl + (k << lpr_log2)] = v[i][k];
                    }
                }
            } else {
                const uint4 * src = (const uint4 *) (rw[i] + it * seg);
#pragma unroll
                for (int k = 0; k < NU; ++k) {
                    if (has_unit(k)) {
                        dsm[sl + (k << lpr_log2)] = __ldg(src + sl + (k << lpr_log2));
                    }
                }
            }
        }
        __syncwarp();

        // the next step: the next iteration of this row block, or the warp's next row block
        int nit  = it + 1;
        int nwrb = wrb;
        const char * nrw[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            nrw[i] = rw[i];
        }
        if (nit == p.niter) {
            nit  = 0;
            nwrb = wrb + wstride;
            if (nwrb < p.nwrb) {
                set_rows(nrw, nwrb);
            }
        }
        const bool more = nwrb < p.nwrb;
        if constexpr (PF) {
            if (more) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const uint4 * src = (const uint4 *) (nrw[i] + nit * seg);
#pragma unroll
                    for (int k = 0; k < NU; ++k) {
                        if (has_unit(k)) {
                            v[i][k] = __ldg(src + sl + (k << lpr_log2));
                        }
                    }
                }
            }
        }

        const int sb = (it << lpr_log2) + sl;
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            dmt_superblock_iq4_xs<NT, R>(sws, aq + sb * QW, ads + sb * DW, acc);
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            dmt_superblock_q6_K<NT, R>(sws, aq + sb * QW, ads + sb * DW, acc);
        } else {
            dmt_superblock_q4_K<NT, R>(sws, aq + sb * QW, ads + sb * DW, acc);
        }

        if (it == p.niter - 1) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
#pragma unroll
                    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                        if (off < lpr) {
                            acc[i][j] += __shfl_xor_sync(0xFFFFFFFF, acc[i][j], off, WARP_SIZE);
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = (wrb * gpw + grp) * R + i;
                if ((i & (lpr - 1)) == sl && row < p.nrows) {
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        p.dst[j * p.stride_col_dst + row] = acc[i][j];
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[i][j] = 0.0f;
                }
            }
        }
        __syncwarp();
        if (!more) {
            break;
        }
        it  = nit;
        wrb = nwrb;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            rw[i] = nrw[i];
        }
    }
}

static int dmt_env(const char * name, const int dflt) {
    const char * env = getenv(name);
    return env ? atoi(env) : dflt;
}

static int dmt_mode() {
    static const int mode = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT", 0);
    return mode;
}

// GGML_CUDA_MMVQ_PASCAL_Q6K_MT: Q6_K's products in this kernel (C727), as _DMT sends IQ4_XS's and Q4_K's: 0 off, 1 on,
// 2 on without the register prefetch
static int dmt_q6k_mode() {
    static const int mode = dmt_env("GGML_CUDA_MMVQ_PASCAL_Q6K_MT", 0);
    return mode;
}

// the switch of type's products
static int dmt_type_mode(const ggml_type type) {
    return type == GGML_TYPE_Q6_K ? dmt_q6k_mode() : dmt_mode();
}

// GGML_CUDA_MMVQ_PASCAL_Q6K_MT_FAULT=1, for the checks: every token of a Q6_K product stages the first token's activations
static bool dmt_q6k_fault() {
    static const bool fault = dmt_env("GGML_CUDA_MMVQ_PASCAL_Q6K_MT_FAULT", 0) == 1;
    return fault;
}

// GGML_CUDA_MMVQ_PASCAL_RO8: 5 to 8 tokens (C729), 1 with the register prefetch, 2 without it
static int dmt_ro8() {
    static const int mode = dmt_env("GGML_CUDA_MMVQ_PASCAL_RO8", 0);
    return mode;
}

// GGML_CUDA_MMVQ_PASCAL_RO8_FAULT=1: every token of the 5-8-token products takes the first token's activations (a check
// that must fail)
static bool dmt_ro8_fault() {
    static const bool fault = dmt_env("GGML_CUDA_MMVQ_PASCAL_RO8_FAULT", 0) == 1;
    return fault;
}

// the switch of type's products of nt tokens: 2 to 4 tokens dmt_type_mode's (_DMT, _Q6K_MT); 5 to 8
// GGML_CUDA_MMVQ_PASCAL_RO8 for IQ4_XS and Q4_K, none for Q6_K (its kernels at 5 to 8 tokens are C795's)
static int dmt_nt_mode(const ggml_type type, const int nt) {
    if (nt <= DMT_MAX_COLS) {
        return dmt_type_mode(type);
    }
    return type == GGML_TYPE_Q6_K ? 0 : dmt_ro8();
}

// GGML_CUDA_MMVQ_PASCAL_DMT_R: rows per lane group, 1 (default since C758) or 2; 0 chooses per product as C722 did
// (dmt_pick_r); any other value 1
static int dmt_r_env() {
    static const int r = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT_R", 1);
    return r == 0 || r == 2 ? r : 1;
}

// GGML_CUDA_MMVQ_PASCAL_DMT_SKIP=iq4_xs,q4_K: weight types (ggml type names) left to the k-column kernel
static bool dmt_skip_type(const ggml_type type) {
    static const std::vector<ggml_type> skip = [] {
        std::vector<ggml_type> types;
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_DMT_SKIP");
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
    return std::find(skip.begin(), skip.end(), type) != skip.end();
}

// log2 of the lanes per row: the largest power of two, at most 32, that divides the row's super-blocks (as
// ggml_cuda_mmvq_pascal chooses it for C = 1)
static int dmt_lpr_log2(const int nsb) {
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && nsb % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    return lpr_log2;
}

// Bytes between two lane groups' row segments in a warp's staging buffer: R segments of SB L bytes and the smallest
// padding, a multiple of 16 bytes below 128, under which the lanes' reads of their super-blocks hit distinct banks: the
// 8 lanes of each quarter warp for 128-bit reads (16-byte-aligned super-blocks), the 16 lanes of each half warp for
// 64-bit reads (IQ4_XS's 136 bytes), the 32 lanes of the warp for 32-bit reads (Q6_K's 210 bytes: the aligned words
// around each super-block); without padding where none does that.
static int dmt_gstride(const int sb, const int r, const int lpr_log2) {
    const int L    = 1 << lpr_log2;
    const int base = r * (sb << lpr_log2);
    const int w    = sb % 16 == 0 ? 16 : sb % 8 == 0 ? 8 : 4; // bytes per lane and read
    const int nl   = 128 / w;                                 // lanes served together
    for (int pad = 0; pad < 128; pad += 16) {
        const int gs = base + pad;
        bool ok = true;
        for (int l0 = 0; l0 < WARP_SIZE && ok; l0 += nl) {
            int seen[32];
            for (int & s : seen) {
                s = -1;
            }
            for (int lane = l0; lane < l0 + nl && ok; ++lane) {
                const int a = ((lane >> lpr_log2) * gs + (lane & (L - 1)) * sb) / w * w;
                const int g = (a / w) % nl;
                ok   = seen[g] < 0 || seen[g] == a;
                seen[g] = a;
            }
        }
        if (ok) {
            return gs;
        }
    }
    return base;
}

static int dmt_wstage(const ggml_type type, const int r, const int lpr_log2) {
    return (WARP_SIZE >> lpr_log2) * dmt_gstride(dmt_sb(type), r, lpr_log2);
}

static int dmt_nw_env() {
    static const int nw_env = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT_NW", DMT_NW_MAX);
    return nw_env;
}

// The warps per block for nt tokens, nsb super-blocks per row and r rows per lane group: GGML_CUDA_MMVQ_PASCAL_DMT_NW
// (default and at most 6 at R = 1, 3 at R = 2), fewer where the shared memory would pass 48 KB; 0 if not even one warp
// fits.
static int dmt_nw(const ggml_type type, const int nt, const int nsb, const int r) {
    int       nw  = std::min(std::max(dmt_nw_env(), 1), DMT_NW_MAX / r);
    const int act = dmt_act_bytes(type, nt, nsb);
    const int ws  = dmt_wstage(type, r, dmt_lpr_log2(nsb));
    while (nw > 0 && act + nw * ws + dmt_tail(type) > DMT_SMEM_MAX) {
        nw--;
    }
    return nw;
}

// Rows per lane group: GGML_CUDA_MMVQ_PASCAL_DMT_R, 1 by default. With _R=0, C722's rule: 2 where its warps hold at
// least as many rows as R = 1's (it then halves the activation loads per row: 3 warps at K = 2560, 4 tokens), else 1
// (Q4_K at K = 6144: 3 warps of one row per lane group against 1 of two). On a P106 R = 1 is the faster wherever the rule
// takes 2 (R79: the four-token IQ4_XS attn_qkv, 10240 x 2560, 146.31 and 147.45 us at R = 1, 240.21 and 237.59 by the
// rule), so since C758 it is the default. 5 to 8 tokens run at R = 1 only.
static int dmt_pick_r(const ggml_type type, const int nt, const int nsb) {
    if (nt > DMT_MAX_COLS) {
        return 1;
    }
    const int r = dmt_r_env();
    if (r > 0) {
        return r;
    }
    const int nw2 = dmt_nw(type, nt, nsb, 2);
    const int nw1 = dmt_nw(type, nt, nsb, 1);
    return nw2 > 0 && 2 * nw2 >= nw1 ? 2 : 1;
}

// The register prefetch of each warp's next segments: never with GGML_CUDA_MMVQ_PASCAL_DMT=2; with =1 everywhere but at
// R = 1 for IQ4_XS's 2-token products at K = 2560, where on a P106 it costs time (R79, us with it and without: 6144 rows
// 94.00 and 95.42 against 75.74 and 75.90, 10240 rows 153.56 and 151.83 against 117.74 and 118.73, 12288 rows 180.54 and
// 180.89 against 135.00 and 136.04; at 3 and 4 tokens, at K = 6144 and at R = 2 it is faster, at Q4_K's 2 tokens at
// K = 2560 within the spread). So since C758; GGML_CUDA_MMVQ_PASCAL_DMT_PF=1 keeps it there too, as C722 did. Q6_K's
// products: with GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 at R = 1 (at R = 2 the two rows' prefetched segments pass the registers:
// 255 and 24 bytes of stack at 4 tokens), =2 never. At 5 to 8 tokens GGML_CUDA_MMVQ_PASCAL_RO8 says: 1 with it, 2
// without.
static bool dmt_pick_pf(const ggml_type type, const int nt, const int nsb, const int r) {
    static const bool pf_all = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT_PF", 0) == 1;
    if (dmt_nt_mode(type, nt) == 2) {
        return false;
    }
    if (type == GGML_TYPE_Q6_K) {
        return r == 1;
    }
    return pf_all || r != 1 || type != GGML_TYPE_IQ4_XS || nt != 2 || nsb != 10;
}

template <ggml_type type, int NT, int R>
static void dmt_launch(const dmt_args & p, const dim3 grid, const dim3 block, const int smem, const bool pf,
                       cudaStream_t stream) {
    if constexpr (type == GGML_TYPE_Q6_K && NT > DMT_MAX_COLS) {
        // Q6_K's products of 5 to 8 tokens never come here (dmt_nt_mode), so their kernels aren't compiled (C795)
        GGML_UNUSED_VARS(p, grid, block, smem, pf, stream);
        GGML_ABORT("Q6_K products of more than %d tokens", DMT_MAX_COLS);
    } else if constexpr (type == GGML_TYPE_Q6_K && R == 2) {
        GGML_UNUSED(pf); // dmt_pick_pf: never
        mul_mat_vec_q_pascal_dmt<type, NT, R, false><<<grid, block, smem, stream>>>(p);
    } else if (pf) {
        mul_mat_vec_q_pascal_dmt<type, NT, R, true><<<grid, block, smem, stream>>>(p);
    } else {
        mul_mat_vec_q_pascal_dmt<type, NT, R, false><<<grid, block, smem, stream>>>(p);
    }
}

template <ggml_type type>
static void dmt_launch_type(const int nt, const int r, const dmt_args & p, const dim3 grid, const dim3 block,
                            const int smem, const bool pf, cudaStream_t stream) {
    switch (nt * 4 + r) {
        case 2*4 + 1: dmt_launch<type, 2, 1>(p, grid, block, smem, pf, stream); break;
        case 3*4 + 1: dmt_launch<type, 3, 1>(p, grid, block, smem, pf, stream); break;
        case 4*4 + 1: dmt_launch<type, 4, 1>(p, grid, block, smem, pf, stream); break;
        case 2*4 + 2: dmt_launch<type, 2, 2>(p, grid, block, smem, pf, stream); break;
        case 3*4 + 2: dmt_launch<type, 3, 2>(p, grid, block, smem, pf, stream); break;
        case 5*4 + 1: dmt_launch<type, 5, 1>(p, grid, block, smem, pf, stream); break;
        case 6*4 + 1: dmt_launch<type, 6, 1>(p, grid, block, smem, pf, stream); break;
        case 7*4 + 1: dmt_launch<type, 7, 1>(p, grid, block, smem, pf, stream); break;
        case 8*4 + 1: dmt_launch<type, 8, 1>(p, grid, block, smem, pf, stream); break;
        default:      dmt_launch<type, 4, 2>(p, grid, block, smem, pf, stream); break;
    }
}

// K tiles (GGML_CUDA_MMVQ_PASCAL_DMT_KT, C728). At K = 6144 (L = 8 lanes per row, 3 iterations) four tokens' activations
// staged whole take 28128 B (IQ4_XS) or 31296 B (Q4_K) of a block's 48 KB, which leaves 4 or 3 warps a block at R = 1
// against 6 at K = 2560, and on a P106 four tokens then take 1.49 and 2.14 times one token's time against 1.14 to 1.24
// at K = 2560 (R79): at 12 warps a SM the kernel is about at its load-store count, below that short of warps. This
// kernel stages one iteration's activations at a time (a K tile: L super-block positions of every token), and the warps
// of a block step through the tiles together:
//   - A round: warp w of the block takes M warp row blocks, (c NW + w) M + m for the block's round c; for each tile
//     t = 0, ..., niter - 1 the block stages tile t, then each warp adds iteration t of its M row blocks; after the last
//     tile the warps add their lanes' sums and write their rows, and the block's next round starts at tile 0 again.
//   - Each lane adds its super-blocks t L + sl in the order t = 0, 1, ... with dmt_superblock_*, the activations in the
//     layout and with the values dmt_stage_act gives them, and the lanes' sums are added with the same xor shuffles: so
//     each token's sums are the one-token kernel's, bit for bit, as above. The M row blocks' sums stay in registers,
//     rotated after each step so that the step's code is there once.
//   - The tile comes in with coalesced 128-bit loads into a raw buffer (each token's block_q8_1 run), then is converted
//     into the tile, position fastest so that the 128-bit stores hit distinct banks; the next tile's loads are issued
//     into registers before the current tile's products, so that their latency hides behind them. Two barriers a tile.
//   - Shared memory at 4 tokens, K = 6144, 6 warps: 9376 B of tile, 9216 B raw and 26112 B of weight staging (IQ4_XS),
//     10432 + 9216 + 27648 B (Q4_K): 2 blocks of 6 warps a SM, as at K = 2560.
//   - The kernel takes only products the kernel above takes, at R = 1, so nothing outside this file changes.

#define DMT_KT_NW_MAX 6 // warps per block, at most
#define DMT_KT_EU     4 // 16-byte units of the raw tile a thread loads ahead, at most

template <ggml_type type, int NT, int M, bool PF>
__launch_bounds__(DMT_KT_NW_MAX*WARP_SIZE, 2)
static __global__ void mul_mat_vec_q_pascal_dmt_kt(const dmt_args p, const int ngrp, const int raw_off, const int fault) {
    constexpr int SB = dmt_traits<type>::SB;
    constexpr int NU = (SB + 15) / 16;
    constexpr int QW = dmt_q_words<NT>();
    constexpr int DW = dmt_ds_words<type, NT>();
    extern __shared__ uint4 dmt_sm[];
    uint32_t * aq  = (uint32_t *) dmt_sm;           // the tile: L positions of int8 words, then their terms
    float    * ads = (float *) dmt_sm + p.ds_off;
    char     * raw = (char *) dmt_sm + raw_off;     // the tile's block_q8_1 run of each token

    const int lane = threadIdx.x;
    const int wid  = threadIdx.y;
    const int nw   = blockDim.y;
    const int tid  = wid * WARP_SIZE + lane;
    const int nth  = nw * WARP_SIZE;

    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int gpw      = WARP_SIZE >> lpr_log2;
    const int seg      = SB << lpr_log2;
    const int tbytes   = 288 << lpr_log2;           // bytes of a token's raw tile: L x 8 blocks of 36 bytes
    const int tunits   = 18 << lpr_log2;            // its 16-byte units
    const int nru      = NT * tunits;
    const int nel      = NT << (lpr_log2 + 3);      // activation blocks of a tile
    char * smw = (char *) dmt_sm + p.stage_off + wid * p.wstage + grp * p.gstride;
    const char * const sws[1] = { smw + sl * SB };

    // the raw tile t of every token into registers (with GGML_CUDA_MMVQ_PASCAL_DMT_KT_FAULT=1 the next tile, a fault for
    // the checks), and from there into shared memory
    uint4 ru[DMT_KT_EU];
    const auto load_raw = [&](const int t) {
        const int ts = fault ? (t + 1 == p.niter ? 0 : t + 1) : t;
#pragma unroll
        for (int k = 0; k < DMT_KT_EU; ++k) {
            const int u = tid + k * nth;
            if (u < nru) {
                int j = 0; // the token: u / tunits
#pragma unroll
                for (int jj = 1; jj < NT; ++jj) {
                    j += u >= jj * tunits;
                }
                ru[k] = __ldg((const uint4 *) (p.vy + j * p.col_bytes_y + ts * tbytes) + (u - j * tunits));
            }
        }
    };
    const auto store_raw = [&]() {
#pragma unroll
        for (int k = 0; k < DMT_KT_EU; ++k) {
            const int u = tid + k * nth;
            if (u < nru) {
                ((uint4 *) raw)[u] = ru[k];
            }
        }
    };
    // the raw tile into the tile, as dmt_stage_act stages each block (position sbl, sub-block s, token j)
    const auto convert = [&]() {
        for (int e = tid; e < nel; e += nth) {
            const int sbl = e & (lpr - 1);
            const int s   = (e >> lpr_log2) & 7;
            const int j   = e >> (lpr_log2 + 3);
            const uint32_t * src = (const uint32_t *) (raw + j * tbytes + ((sbl << 3) + s) * (int) sizeof(block_q8_1));
            uint32_t w[9];
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                w[k] = src[k];
            }
            const float d8 = mmvqp_h2f_lo(w[0]);
            uint32_t q[8];
            if constexpr (type == GGML_TYPE_IQ4_XS) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    q[2*k + 0] = __byte_perm(w[1 + k], w[5 + k], 0x5140);
                    q[2*k + 1] = __byte_perm(w[1 + k], w[5 + k], 0x7362);
                }
                ads[sbl * DW + s*NT + j] = d8;
            } else {
                int sum = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    q[k] = w[1 + k];
                    sum  = ggml_cuda_dp4a(0x01010101, (int) w[1 + k], sum);
                }
                *(float2 *) (ads + sbl * DW + (s*NT + j) * 2) = make_float2(d8, d8 * (float) sum);
            }
            uint4 * dq = (uint4 *) (aq + sbl * QW + (s*NT + j) * 8);
            dq[0] = make_uint4(q[0], q[1], q[2], q[3]);
            dq[1] = make_uint4(q[4], q[5], q[6], q[7]);
        }
    };

    // the lane group's row in row block m of round c (clamped, so that the staging reads valid rows; their results are
    // not written), and whether this lane has unit k of a segment (as above)
    const auto row_ptr = [&](const int c, const int m) {
        return p.vx + (int64_t) min(((c * nw + wid) * M + m) * gpw + grp, p.nrows - 1) * p.row_bytes;
    };
    const auto has_unit = [&](const int k) {
        return k < NU - 1 || sl + (k << lpr_log2) < p.seg_units;
    };

    ggml_cuda_pdl_sync();

    int c = blockIdx.x;
    load_raw(0);
    [[maybe_unused]] uint4 v[NU];
    if constexpr (PF) {
        const uint4 * src = (const uint4 *) row_ptr(c, 0);
#pragma unroll
        for (int k = 0; k < NU; ++k) {
            if (has_unit(k)) {
                v[k] = __ldg(src + sl + (k << lpr_log2));
            }
        }
    }

    float acc[M][NT];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[m][j] = 0.0f;
        }
    }

    while (true) {
        for (int t = 0; t < p.niter; ++t) {
            // tile t: every warp is done with tile t - 1 when it stores its part of the raw tile
            store_raw();
            __syncthreads();
            convert();
            __syncthreads();
            if (t + 1 < p.niter) {
                load_raw(t + 1);
            } else if (c + (int) gridDim.x < ngrp) {
                load_raw(0);
            }

#pragma unroll 1
            for (int m = 0; m < M; ++m) {
                // the segment of row block m at iteration t into the warp's staging buffer
                uint4 * dsm = (uint4 *) smw;
                if constexpr (PF) {
#pragma unroll
                    for (int k = 0; k < NU; ++k) {
                        if (has_unit(k)) {
                            dsm[sl + (k << lpr_log2)] = v[k];
                        }
                    }
                } else {
                    const uint4 * src = (const uint4 *) (row_ptr(c, m) + t * seg);
#pragma unroll
                    for (int k = 0; k < NU; ++k) {
                        if (has_unit(k)) {
                            dsm[sl + (k << lpr_log2)] = __ldg(src + sl + (k << lpr_log2));
                        }
                    }
                }
                __syncwarp();

                // the next segment: row block m + 1, else iteration t + 1, else the block's next round
                if constexpr (PF) {
                    int mn = m + 1;
                    int tn = t;
                    int cn = c;
                    if (mn == M) {
                        mn = 0;
                        tn = t + 1;
                        if (tn == p.niter) {
                            tn = 0;
                            cn = c + gridDim.x;
                        }
                    }
                    if (cn < ngrp) {
                        const uint4 * src = (const uint4 *) (row_ptr(cn, mn) + tn * seg);
#pragma unroll
                        for (int k = 0; k < NU; ++k) {
                            if (has_unit(k)) {
                                v[k] = __ldg(src + sl + (k << lpr_log2));
                            }
                        }
                    }
                }

                // row block m's sums are acc[0]; after the step they go last, so that acc is in order after M steps
                float a[1][NT];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    a[0][j] = acc[0][j];
                }
                if constexpr (type == GGML_TYPE_IQ4_XS) {
                    dmt_superblock_iq4_xs<NT, 1>(sws, aq + sl * QW, ads + sl * DW, a);
                } else {
                    dmt_superblock_q4_K<NT, 1>(sws, aq + sl * QW, ads + sl * DW, a);
                }
#pragma unroll
                for (int i = 0; i < M - 1; ++i) {
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        acc[i][j] = acc[i + 1][j];
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[M - 1][j] = a[0][j];
                }
                __syncwarp();
            }
        }

        // the round's rows: the lanes' sums added as above, written by the group's first lane
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                    if (off < lpr) {
                        acc[m][j] += __shfl_xor_sync(0xFFFFFFFF, acc[m][j], off, WARP_SIZE);
                    }
                }
            }
            const int row = ((c * nw + wid) * M + m) * gpw + grp;
            if (sl == 0 && row < p.nrows) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    p.dst[j * p.stride_col_dst + row] = acc[m][j];
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[m][j] = 0.0f;
            }
        }
        c += gridDim.x;
        if (c >= ngrp) {
            break;
        }
    }
}

// GGML_CUDA_MMVQ_PASCAL_DMT_KT: 0 (default) off; 1 the K tiles where they are faster on a P106: Q4_K's products of 4
// tokens whose whole activations leave the kernel above at most DMT_KT_FULL_MAX warps a block (at Flash-Next's widths
// K = 6144: 3 warps, the tiles 6); 2 wherever the tiles fit, for the checks. Only with GGML_CUDA_MMVQ_PASCAL_DMT on and
// at R = 1.
static int dmt_kt_mode() {
    static const int mode = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT_KT", 0);
    return mode;
}

// GGML_CUDA_MMVQ_PASCAL_DMT_KT_M: row blocks per warp and round, 2 (default since C808: half the tile staging per row, in
// rounds twice as long) or 1. On a P106 2 is the faster in every tile case (R120b, us: IQ4_XS 2560 x 6144 at 4 tokens
// 132.96 / 133.04 at 1 against 128.48 / 128.06 at 2; Q4_K at 3 tokens 110.40 / 110.94 against 106.91 / 106.79, at 4
// 136.07 / 135.90 against 129.45 / 129.39).
static int dmt_kt_m() {
    static const int m = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT_KT_M", 2) == 1 ? 1 : 2;
    return m;
}

// Bytes of a tile: L positions of the int8 words, then of the terms, as dmt_act_bytes for L super-blocks
static int dmt_kt_tile_bytes(const ggml_type type, const int nt, const int lpr_log2) {
    return dmt_act_bytes(type, nt, 1 << lpr_log2);
}

// The most warps a block of the kernel above holds where GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 sends Q4_K's 4-token products to
// the tiles (C808). On a P106 (R120b, us per product at 2560 x 6144, C722's kernel against the tiles at M = 2) the tiles
// gain only from 3 warps up to 6: Q4_K at 4 tokens 142.29 / 141.75 against 129.45 / 129.39; from 4 warps (IQ4_XS at 4
// tokens, 112.60 / 113.33 against 128.48 / 128.06) and from 5 (Q4_K at 3 tokens, 98.07 / 99.10 against 106.91 / 106.79)
// they lose.
#define DMT_KT_FULL_MAX 3

// The warps per block of the K tiles for nt tokens and nsb super-blocks per row (R = 1) where they take the product, else
// 0: GGML_CUDA_MMVQ_PASCAL_DMT_KT_NW (default and at most 6), fewer where the block's shared memory would pass 48 KB;
// none for other types than IQ4_XS and Q4_K or other token counts than 2 to 4, where the threads can't load a tile
// ahead in DMT_KT_EU units each, or (with _KT=1) for other products than Q4_K's of 4 tokens, nor where the kernel above
// holds more than DMT_KT_FULL_MAX warps (nw_full) or as many as the tiles
static int dmt_kt_nw(const ggml_type type, const int nt, const int nsb, const int nw_full) {
    static const int nw_env = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT_KT_NW", DMT_KT_NW_MAX);
    const int mode = dmt_kt_mode();
    // the tiles have IQ4_XS's and Q4_K's steps and staging, at 2 to 4 tokens, only; any other product this file takes
    // keeps its own kernel
    if ((mode != 1 && mode != 2) || (type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_Q4_K) || nt < 2 || nt > 4) {
        return 0;
    }
    // _KT=1: only where they gain on a P106 (C808)
    if (mode == 1 && (type != GGML_TYPE_Q4_K || nt != 4 || nw_full > DMT_KT_FULL_MAX)) {
        return 0;
    }
    const int lpr_log2 = dmt_lpr_log2(nsb);
    const int fixed    = dmt_kt_tile_bytes(type, nt, lpr_log2) + nt * (288 << lpr_log2);
    const int ws       = dmt_wstage(type, 1, lpr_log2);
    int nw = std::min(std::max(nw_env, 1), DMT_KT_NW_MAX);
    while (nw > 0 && fixed + nw * ws > DMT_SMEM_MAX) {
        nw--;
    }
    if (nw == 0 || nt * (18 << lpr_log2) > DMT_KT_EU * nw * WARP_SIZE) {
        return 0;
    }
    return mode == 2 || nw > nw_full ? nw : 0;
}

template <ggml_type type, int NT, int M, bool PF>
static int dmt_kt_blocks_regs(const int nw) {
    // resident blocks per SM that the kernel's registers allow, per block size, queried once
    static int cache[DMT_KT_NW_MAX + 1] = {};
    if (cache[nw] == 0) {
        int n = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mul_mat_vec_q_pascal_dmt_kt<type, NT, M, PF>,
                                                          nw * WARP_SIZE, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            n = 1;
        }
        cache[nw] = std::max(n, 1);
    }
    return cache[nw];
}

template <ggml_type type, int NT, int M, bool PF>
static void dmt_kt_launch(const dmt_args & p, const int ngrp, const int raw_off, const int fault, const int nw,
                          const int smem, cudaStream_t stream) {
    // a persistent grid, as many blocks as the SMs hold at once by registers and shared memory, at most one per round
    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int bps  = std::max(1, std::min(dmt_kt_blocks_regs<type, NT, M, PF>(nw), 96 * 1024 / smem));
    const int nblk = std::min(ngrp, nsm * bps);
    mul_mat_vec_q_pascal_dmt_kt<type, NT, M, PF><<<dim3(nblk, 1, 1), dim3(WARP_SIZE, nw, 1), smem, stream>>>(
        p, ngrp, raw_off, fault);
}

template <ggml_type type, int M>
static void dmt_kt_launch_nt(const int nt, const bool pf, const dmt_args & p, const int ngrp, const int raw_off,
                             const int fault, const int nw, const int smem, cudaStream_t stream) {
    switch (nt * 2 + (pf ? 1 : 0)) {
        case 2*2 + 0: dmt_kt_launch<type, 2, M, false>(p, ngrp, raw_off, fault, nw, smem, stream); break;
        case 2*2 + 1: dmt_kt_launch<type, 2, M, true >(p, ngrp, raw_off, fault, nw, smem, stream); break;
        case 3*2 + 0: dmt_kt_launch<type, 3, M, false>(p, ngrp, raw_off, fault, nw, smem, stream); break;
        case 3*2 + 1: dmt_kt_launch<type, 3, M, true >(p, ngrp, raw_off, fault, nw, smem, stream); break;
        case 4*2 + 0: dmt_kt_launch<type, 4, M, false>(p, ngrp, raw_off, fault, nw, smem, stream); break;
        default:      dmt_kt_launch<type, 4, M, true >(p, ngrp, raw_off, fault, nw, smem, stream); break;
    }
}

// Launches the K tiles for a product the kernel above takes at R = 1 with nw_full warps a block, where
// GGML_CUDA_MMVQ_PASCAL_DMT_KT says so; false (nothing launched) otherwise
static bool dmt_kt(const ggml_type type, const void * vx, const void * vy, float * dst, const int nt, const int nsb,
                   const int64_t nrows, const int64_t row_bytes, const int64_t col_bytes_y, const int64_t stride_col_dst,
                   const int nw_full, cudaStream_t stream) {
    static const int fault = dmt_env("GGML_CUDA_MMVQ_PASCAL_DMT_KT_FAULT", 0) == 1 ? 1 : 0;
    const int nw = dmt_kt_nw(type, nt, nsb, nw_full);
    // the raw tiles are read with 16-byte loads
    if (nw == 0 || (uintptr_t) vy % 16 != 0 || col_bytes_y % 16 != 0) {
        return false;
    }
    const int m = dmt_kt_m();

    dmt_args p;
    p.vx             = (const char *) vx;
    p.vy             = (const char *) vy;
    p.dst            = dst;
    p.nrows          = (int) nrows;
    p.row_bytes      = (int) row_bytes;
    p.nsb            = nsb;
    p.lpr_log2       = dmt_lpr_log2(nsb);
    p.niter          = nsb >> p.lpr_log2;
    const int gpw    = WARP_SIZE >> p.lpr_log2; // rows per warp row block
    p.nwrb           = (int) ((nrows + gpw - 1) / gpw);
    p.seg_units      = (dmt_sb(type) << p.lpr_log2) / 16;
    p.col_bytes_y    = (int) col_bytes_y;
    p.stride_col_dst = (int) stride_col_dst;
    p.ds_off         = (1 << p.lpr_log2) * (64 * nt + 4);
    const int raw_off = dmt_kt_tile_bytes(type, nt, p.lpr_log2);
    p.stage_off      = raw_off + nt * (288 << p.lpr_log2);
    p.wstage         = dmt_wstage(type, 1, p.lpr_log2);
    p.gstride        = dmt_gstride(dmt_sb(type), 1, p.lpr_log2);
    const int smem   = p.stage_off + nw * p.wstage;
    const int ngrp   = (p.nwrb + nw * m - 1) / (nw * m);
    const bool pf    = dmt_pick_pf(type, nt, nsb, 1);

    if (type == GGML_TYPE_IQ4_XS) {
        if (m == 2) {
            dmt_kt_launch_nt<GGML_TYPE_IQ4_XS, 2>(nt, pf, p, ngrp, raw_off, fault, nw, smem, stream);
        } else {
            dmt_kt_launch_nt<GGML_TYPE_IQ4_XS, 1>(nt, pf, p, ngrp, raw_off, fault, nw, smem, stream);
        }
    } else {
        if (m == 2) {
            dmt_kt_launch_nt<GGML_TYPE_Q4_K, 2>(nt, pf, p, ngrp, raw_off, fault, nw, smem, stream);
        } else {
            dmt_kt_launch_nt<GGML_TYPE_Q4_K, 1>(nt, pf, p, ngrp, raw_off, fault, nw, smem, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

bool ggml_cuda_mmvq_pascal_dmt_takes(const ggml_type type, const int64_t K, const int64_t nrows, const int64_t ncols,
                                     const int64_t row_bytes, const void * x) {
    if (ncols < 2 || ncols > DMT_RO8_COLS || dmt_nt_mode(type, (int) ncols) == 0 ||
            (type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K)) {
        return false;
    }
    if (!ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) ||
            ggml_cuda_mmvq_pascal_skips(type) || dmt_skip_type(type)) {
        return false;
    }
    if (K < QK_K || K % QK_K != 0 || K > INT_MAX || nrows < 1 || nrows > INT_MAX || row_bytes > INT_MAX) {
        return false;
    }
    // the segments are staged in whole 16-byte units from 16-byte-aligned rows (IQ4_XS: an even number of lanes per row;
    // Q6_K: a multiple of 8)
    const int nsb = (int) (K / QK_K);
    const int sb  = dmt_sb(type);
    if (row_bytes < (int64_t) nsb * sb || row_bytes % 16 != 0 || (uintptr_t) x % 16 != 0 ||
            (sb << dmt_lpr_log2(nsb)) % 16 != 0) {
        return false;
    }
    const int nw = dmt_nw(type, (int) ncols, nsb, dmt_pick_r(type, (int) ncols, nsb));
    // A block of one warp only where _DMT_NW=1 asks for it, for Q6_K (one warp a block lost to the k-column kernel, R79)
    // and for the products of 5 to 8 tokens (C809, Q1415: on a P106 IQ4_XS 2560 x 6144 at 6 tokens took 595.98 and
    // 595.90 us at one warp a block, 277.57 and 279.01 where this kernel left it, R120b), until C794's K tiles take them
    const bool one_warp_declined = type == GGML_TYPE_Q6_K || ncols > DMT_MAX_COLS;
    return one_warp_declined && dmt_nw_env() != 1 ? nw >= 2 : nw > 0;
}

bool ggml_cuda_mmvq_pascal_dmt(const ggml_type type, const void * vx, const void * vy, float * dst, const int64_t K,
                               const int64_t nrows, const int64_t ncols, const int64_t row_bytes,
                               const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    if (!ggml_cuda_mmvq_pascal_dmt_takes(type, K, nrows, ncols, row_bytes, vx)) {
        return false;
    }
    const int64_t col_bytes_y = stride_col_y * (int64_t) sizeof(block_q8_1);
    if ((uintptr_t) vy % 4 != 0 || (ncols - 1) * col_bytes_y + (K / QK8_1) * (int64_t) sizeof(block_q8_1) > INT_MAX ||
            (ncols - 1) * stride_col_dst + nrows > INT_MAX) {
        return false;
    }

    const int nt  = (int) ncols;
    const int nsb = (int) (K / QK_K);
    const int r   = dmt_pick_r(type, nt, nsb);
    const int nw  = dmt_nw(type, nt, nsb, r);
    if (r == 1 && dmt_kt(type, vx, vy, dst, nt, nsb, nrows, row_bytes, col_bytes_y, stride_col_dst, nw, stream)) {
        return true;
    }

    dmt_args p;
    p.vx             = (const char *) vx;
    p.vy             = (const char *) vy;
    p.dst            = dst;
    p.nrows          = (int) nrows;
    p.row_bytes      = (int) row_bytes;
    p.nsb            = nsb;
    p.lpr_log2       = dmt_lpr_log2(nsb);
    p.niter          = nsb >> p.lpr_log2;
    const int rpw    = (WARP_SIZE >> p.lpr_log2) * r; // rows per warp row block
    p.nwrb           = (int) ((nrows + rpw - 1) / rpw);
    p.seg_units      = (dmt_sb(type) << p.lpr_log2) / 16;
    p.col_bytes_y    = type == GGML_TYPE_Q6_K && dmt_q6k_fault() ? 0 : (int) col_bytes_y; // the fault: token 0's, for all
    if (nt > DMT_MAX_COLS && dmt_ro8_fault()) {
        p.col_bytes_y = 0; // the fault: every token stages the first token's activations
    }
    p.stride_col_dst = (int) stride_col_dst;
    p.ds_off         = nsb * (64 * nt + 4);
    const int act    = dmt_act_bytes(type, nt, nsb);
    p.stage_off      = act;
    p.wstage         = dmt_wstage(type, r, p.lpr_log2);
    p.gstride        = dmt_gstride(dmt_sb(type), r, p.lpr_log2);
    const int smem   = act + nw * p.wstage + dmt_tail(type);

    // a persistent grid: as many blocks as the SMs hold at once by shared memory and threads (the launch bounds keep the
    // registers for 2 blocks of the most warps), at most one per nw warp row blocks
    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int bps  = std::max(1, std::min(96 * 1024 / smem, 2048 / (nw * WARP_SIZE)));
    const int nblk = (int) std::min<int64_t>((p.nwrb + nw - 1) / nw, (int64_t) nsm * bps);
    const dim3 grid(nblk, 1, 1);
    const dim3 block(WARP_SIZE, nw, 1);
    const bool pf = dmt_pick_pf(type, nt, nsb, r);

    if (type == GGML_TYPE_IQ4_XS) {
        dmt_launch_type<GGML_TYPE_IQ4_XS>(nt, r, p, grid, block, smem, pf, stream);
    } else if (type == GGML_TYPE_Q6_K) {
        dmt_launch_type<GGML_TYPE_Q6_K>(nt, r, p, grid, block, smem, pf, stream);
    } else {
        dmt_launch_type<GGML_TYPE_Q4_K>(nt, r, p, grid, block, smem, pf, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_pascal_dmt_takes(const ggml_type type, const int64_t K, const int64_t nrows, const int64_t ncols,
                                     const int64_t row_bytes, const void * x) {
    GGML_UNUSED_VARS(type, K, nrows, ncols, row_bytes, x);
    return false;
}

bool ggml_cuda_mmvq_pascal_dmt(const ggml_type type, const void * vx, const void * vy, float * dst, const int64_t K,
                               const int64_t nrows, const int64_t ncols, const int64_t row_bytes,
                               const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    GGML_UNUSED_VARS(type, vx, vy, dst, K, nrows, ncols, row_bytes, stride_col_y, stride_col_dst, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
