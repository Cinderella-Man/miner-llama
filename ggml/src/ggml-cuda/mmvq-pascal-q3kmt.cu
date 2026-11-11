#include "mmvq-pascal-q3kmt.cuh"
#include "mmvq-pascal.cuh"
#include "mmvq-pascal-kcol.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

// How the kernels work (2 to 8 tokens of a plain Q3_K product; C722's kernel, mmvq-pascal-dmt.cu, in the order of
// Q3_K's one-token kernel, which is the k-column kernel at one column, kcol::mul_mat_kcol<KQ3_K, 1, R>):
//   - The lanes and summation order are the k-column kernel's: L = 2^lpr_log2 lanes per row (the largest power of two, at
//     most 32, dividing the row's super-blocks), lane sl of a row's group taking super-blocks sl, sl + L, ... in that
//     order, each super-block's 32-weight blocks b = 0..7 in order, each block added to the lane's sum as
//       acc = fmaf(fmaf((float) i1, d * (sc1 - 32), (float) i0 * (d * (sc0 - 32))), d8 / f, acc)
//     (kcol::unit_q3_K: i0 and i1 the two 16-weight halves' integer sums, which start at -4 f times the halves'
//     activation sums; f = 1, 4, 16, 16 for b % 4 = 0..3, the codes read in place), and the lanes' sums added with the
//     same xor shuffles. That order depends on the row width only, so every token's result is the k-column kernel's at
//     one token, bit for bit. A lane group owns R = 1 (default) or 2 consecutive rows; the rows of a group share every
//     load of the activations, and R changes no row's sum.
//   - The activation terms come from the Q8_1 blocks ggml_cuda_mul_mat_vec_q quantizes, computed as the k-column
//     kernel's quantizer (kcol::quantize_records<KQ3_K>) computes them: quantize_q8_1 and it give the same int8 values
//     and the same FP16 scale d8 (the same FMUL by 1/127, MUFU.RCP, FMUL and rounding in their sm_61 code), then d8 / f
//     with the same expression, and the integer offsets -4 f * (sum of the 16 int8 values of a half).
//   - Each 32-weight block of a row is decoded once into eight words (the codes masked in place with the high bit moved
//     next to them, as kcol::unit_q3_K) and multiplied with dp4a by every token's activations.
//   - The integer-to-float conversions are made without the quarter-rate I2F, exactly: the dp4a chains start at
//     0x4B400000 plus the offset (staged so), so a half's result read as a float is 12582912 + i (|i| <= 16 * 4 * 16 *
//     127 < 2^22), and less 12582912 it is (float) i; the 6-bit scales sc come out of a __byte_perm into the same float,
//     less 12582944 they are (float) (sc - 32). Every FP32 operation then sees the k-column kernel's operands in its
//     order: d * (sc - 32) (FMUL), (float) i0 * dsc0 (FMUL), the two FFMA.
//   - Each block stages all tokens' activations once into shared memory (per super-block position, block and token: the
//     8 int8 words, d8 / f, and the two offsets with 0x4B400000; padded so that the L lanes of a group read distinct
//     banks; every lane group of a warp reads the same L records, a broadcast) and loops over row blocks (a persistent
//     grid of about 2 blocks per SM), so the staging serves many rows. At 4 tokens K = 2560 is 14.0 KB, K = 6144 33.7 KB.
//     The k-column kernel stages its activation records again for every block of 64 rows.
//   - The weights: Q3_K's 110-byte super-blocks are 2-byte aligned, so each warp stages each of its rows' segments (L
//     super-blocks) from the 16-byte boundary below it into a slot of 8 L 16-byte units with 128-bit loads, as the
//     k-column kernel's warp staging does; each lane reads its super-block's words from there (kcol::unit_words, a
//     __byte_perm per word where it lies 2 bytes off). The slots are padded (q3m_pad) so that the lanes' 32-bit reads
//     spread over the banks: without it every row's slot starts in the same bank.
//   - By default each warp loads its next segments into registers before it computes the current ones.

#define Q3M_NW_MAX   6                    // warps per block at R = 1, at most (3 at R = 2)
#define Q3M_MAX_COLS 8
#define Q3M_SB       110                  // bytes of a block_q3_K
#define Q3M_SMEM_MAX (48 * 1024)

static_assert(sizeof(block_q3_K) == Q3M_SB, "the staging assumes 110-byte super-blocks");
static_assert(sizeof(block_q8_1) == 36, "the activation staging assumes 36-byte Q8_1 blocks");

namespace {

struct q3m_args {
    const char * vx;
    const char * vy;            // block_q8_1, column j from j * col_bytes_y
    float      * dst;           // column j from j * stride_col_dst
    int          nrows;
    int          row_bytes;
    int          nsb;           // super-blocks per row
    int          lpr_log2;      // lanes per row L = 1 << lpr_log2 (L >= 2)
    int          niter;         // super-blocks per lane and row (nsb / L)
    int          nwrb;          // warp row blocks: rows / ((32 >> lpr_log2) * R), rounded up
    int          slot;          // bytes of a row's slot in a warp's staging buffer (8 L 16-byte units and the padding)
    int          col_bytes_y;
    int          stride_col_dst;
    int          d_off;         // words from the start of shared memory to the scales d8 / f
    int          h_off;         // 8-byte units from the start of shared memory to the offsets
    int          stage_off;     // bytes from the start of shared memory to the warps' staging buffers
    int          wstage;        // bytes of a warp's staging buffer
    int          fault;         // GGML_CUDA_MMVQ_PASCAL_Q3K_MT_FAULT
};

// Per super-block position of the staged activations: 16-byte units of the int8 words (NT tokens x 8 blocks x 2 units,
// plus one, so that consecutive positions start in other banks), and words of the scales (one per block and token, plus
// one) and 8-byte units of the offsets (the same count).
template <int NT> constexpr __host__ __device__ int q3m_q_units() { return 16 * NT + 1; }
template <int NT> constexpr __host__ __device__ int q3m_d_words() { return 8 * NT + 1; }

static int q3m_q_bytes(const int nt, const int nsb) {
    return nsb * (16 * nt + 1) * 16;
}
static int q3m_d_bytes(const int nt, const int nsb) {
    return (nsb * (8 * nt + 1) * 4 + 7) / 8 * 8;
}
static int q3m_act_bytes(const int nt, const int nsb) {
    return (q3m_q_bytes(nt, nsb) + q3m_d_bytes(nt, nsb) + nsb * (8 * nt + 1) * 8 + 15) / 16 * 16;
}

// Every token's activation blocks into shared memory, by all warps of the block: the int8 words as they come, and per
// 32-value block b of a super-block (f = 1, 4, 16, 16 for b % 4 = 0..3) d8 / f and 0x4B400000 + -4 f times the sums of
// its two halves' int8 values, as kcol::quantize_records<KQ3_K> computes its records' terms (d8 / f, -4 f * s16).
template <int NT>
static __device__ __forceinline__ void q3m_stage_act(const q3m_args & p, uint4 * sq, float * sd, int2 * sh) {
    constexpr int QU = q3m_q_units<NT>();
    constexpr int DW = q3m_d_words<NT>();
    const int nab = p.nsb * 8;
    const int nth = blockDim.x * blockDim.y;
    for (int e = threadIdx.y * WARP_SIZE + threadIdx.x; e < NT * nab; e += nth) {
        const int j  = e / nab;
        const int gb = e - j * nab;
        const uint32_t * src = (const uint32_t *) (p.vy + j * p.col_bytes_y + gb * (int) sizeof(block_q8_1));
        uint32_t w[9];
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            w[k] = __ldg(src + k);
        }
        const float d8 = kcol::h2f_lo(w[0]);
        const int   sb = gb >> 3;
        const int   b  = gb & 7;
        int s0 = 0;
        int s1 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            s0 = kcol::dp4a(0x01010101, (int) w[1 + k], s0);
            s1 = kcol::dp4a(0x01010101, (int) w[5 + k], s1);
        }
        const int f = 1 << (2 * min(b & 3, 2));
        const int m = -4 * f;
        // the fault (GGML_CUDA_MMVQ_PASCAL_Q3K_MT_FAULT=1): token j's activations staged as token j + 1's
        const int jd = p.fault ? (j + 1) % NT : j;
        uint4 * dq = sq + sb * QU + (b * NT + jd) * 2;
        dq[0] = make_uint4(w[1], w[2], w[3], w[4]);
        dq[1] = make_uint4(w[5], w[6], w[7], w[8]);
        sd[sb * DW + b * NT + jd] = d8 / (float) f;
        sh[sb * DW + b * NT + jd] = make_int2(0x4B400000 + m * s0, 0x4B400000 + m * s1);
    }
}

// One super-block position of R rows (wu[i]: row i's super-block in the staging buffer, 2-byte aligned) times every
// token's activations at that position (aq: the int8 words, ad: d8 / f, ah: the offsets), added to acc[i][j] exactly as
// kcol::unit_q3_K adds row i's super-block for token j alone.
template <int NT, int R>
static __device__ __forceinline__ void q3m_superblock(const kcol::unit_ptr (&wu)[R], const uint4 * aq, const float * ad,
                                                      const int2 * ah, float (&acc)[R][NT]) {
    uint32_t hm[R][8];
    uint32_t sc[R][4];
    float    d[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        kcol::unit_words<2, 8>(hm[i], wu[i], 0);
        uint32_t w[4];
        kcol::unit_words<2, 4>(w, wu[i], 24);
        const uint32_t a0 = w[0], a1 = w[1], tmp = w[2];
        sc[i][0] = (a0 & 0x0f0f0f0f)        | (((tmp >> 0) & 0x03030303) << 4);
        sc[i][1] = (a1 & 0x0f0f0f0f)        | (((tmp >> 2) & 0x03030303) << 4);
        sc[i][2] = ((a0 >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
        sc[i][3] = ((a1 >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
        d[i] = kcol::h2f_lo(w[3]);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        // blocks 2p and 2p + 1: n = p / 2, m = 2 (p & 1) and 2 (p & 1) + 1
        const int n = p >> 1;
        uint32_t qs[R][8];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            kcol::unit_words<2, 8>(qs[i], wu[i], 8 + 8 * n);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int b     = 2 * p + e;
            const int m     = b & 3;
            const int hbpos = m < 3 ? 2 + 2 * m : 6;  // the high bit's place in each byte
            const int sh    = hbpos - b;              // hmask bit b to bit hbpos
            int   q[R][8];
            float dsc0[R];
            float dsc1[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t lo = m < 3 ? qs[i][k] & (0x03030303u << (2 * m)) : (qs[i][k] >> 2) & 0x30303030u;
                    const uint32_t hb = (sh >= 0 ? hm[i][k] << sh : hm[i][k] >> -sh) & (0x01010101u << hbpos);
                    q[i][k] = (int) (lo | hb);
                }
                // scales 2b and 2b + 1 (bytes 2 (b & 1) and 2 (b & 1) + 1 of scale word b / 2), less 32, exactly: the
                // 6-bit scale as the low mantissa bits of 12582912 = 2^23 + 2^22
                const uint32_t sw = sc[i][b >> 1];
                const float s0 = __int_as_float((int) __byte_perm(sw, 0x4B400000u, 0x7650 | (2 * (b & 1))))     - 12582944.0f;
                const float s1 = __int_as_float((int) __byte_perm(sw, 0x4B400000u, 0x7650 | (2 * (b & 1) + 1))) - 12582944.0f;
                dsc0[i] = d[i] * s0;
                dsc1[i] = d[i] * s1;
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4 u0  = aq[(b * NT + j) * 2];
                const uint4 u1  = aq[(b * NT + j) * 2 + 1];
                const float a_d = ad[b * NT + j];
                const int2  h   = ah[b * NT + j];
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    int i0 = h.x; // 0x4B400000 + the offset: the result reads as 12582912 + i0 as a float
                    int i1 = h.y;
                    i0 = kcol::dp4a(q[i][0], (int) u0.x, i0);
                    i1 = kcol::dp4a(q[i][4], (int) u1.x, i1);
                    i0 = kcol::dp4a(q[i][1], (int) u0.y, i0);
                    i1 = kcol::dp4a(q[i][5], (int) u1.y, i1);
                    i0 = kcol::dp4a(q[i][2], (int) u0.z, i0);
                    i1 = kcol::dp4a(q[i][6], (int) u1.z, i1);
                    i0 = kcol::dp4a(q[i][3], (int) u0.w, i0);
                    i1 = kcol::dp4a(q[i][7], (int) u1.w, i1);
                    const float f0 = __int_as_float(i0) - 12582912.0f; // (float) i0, exactly
                    const float f1 = __int_as_float(i1) - 12582912.0f;
                    acc[i][j] = fmaf(fmaf(f1, dsc1[i], f0 * dsc0[i]), a_d, acc[i][j]);
                }
            }
        }
    }
}

// Grid: (blocks, 1, 1), each looping over warp row blocks; block (32, NW). Dynamic shared memory: the staged activations
// (int8 words, scales, offsets), then the NW warps' staging buffers ((32 >> lpr_log2) R row slots each).
template <int NT, int R, bool PF>
__launch_bounds__(Q3M_NW_MAX*WARP_SIZE/R, 2)
static __global__ void mul_mat_vec_q_pascal_q3kmt(const q3m_args p) {
    constexpr int QU = q3m_q_units<NT>();
    constexpr int DW = q3m_d_words<NT>();
    constexpr int NP = 8 * R;                       // 16-byte units of a warp's staging per lane (8 L per row slot)
    extern __shared__ uint4 q3m_sm[];
    uint4 * sq = q3m_sm;
    float * sd = (float *) q3m_sm + p.d_off;
    int2  * sh = (int2 *) q3m_sm + p.h_off;

    const int lane = threadIdx.x;
    const int wid  = threadIdx.y;
    const int nw   = blockDim.y;

    ggml_cuda_pdl_sync();

    q3m_stage_act<NT>(p, sq, sd, sh);
    __syncthreads();

    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int rw       = (WARP_SIZE >> lpr_log2) * R;  // rows per warp row block
    const int seg      = Q3M_SB << lpr_log2;           // bytes of a row's segment per iteration
    const int seg_log2 = lpr_log2 + 3;                 // a row slot holds 8 L 16-byte units
    const int nup      = 1 << seg_log2;
    const int lrow     = lane >> seg_log2;             // 1 for the upper half warp at L = 2 (2 row slots a pass), else 0
    const int pad      = p.slot - (16 << seg_log2);    // bytes of padding after each row slot
    char * smw = (char *) q3m_sm + p.stage_off + wid * p.wstage;
    const int wstride = gridDim.x * nw;

    // the low 4 bits of the offsets of the lane group's rows from the warp row block's first row
    int hoff[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        hoff[i] = ((grp * R + i) * p.row_bytes) & 15;
    }

    int wrb = blockIdx.x * nw + wid;
    if (wrb >= p.nwrb) {
        return;
    }
    int it = 0;

    // The 16-byte units of a step's segments that this lane stages (wb: the step's first row's segment, rem: the rows from
    // it to the matrix's end): unit u = q * 32 + lane of the warp's staging is unit k = u mod 8 L of row slot r = u / 8 L,
    // from the 16-byte boundary below the row's segment, where the segment has it (rows past the matrix and units past
    // the segment are left as they are: no lane reads them for a written row). In shared memory it sits at 16 u + r pad.
    const auto unit_src = [&](const char * wb, const int rem, const int q, const uint4 * & src) {
        const int    r = ((q * WARP_SIZE) >> seg_log2) + lrow;
        const int    k = (q * WARP_SIZE + lane) & (nup - 1);
        const char * g = wb + r * p.row_bytes;
        const int    h = (int) ((uintptr_t) g & 15);
        src = (const uint4 *) (g - h) + k;
        return r < rem && 16 * k < h + seg;
    };
    const auto unit_dst = [&](const int q) {
        const int r = ((q * WARP_SIZE) >> seg_log2) + lrow;
        return (uint4 *) (smw + (q * WARP_SIZE + lane) * 16 + r * pad);
    };
    const auto row_block = [&](const int w) {
        return p.vx + (int64_t) (w * rw) * p.row_bytes;
    };

    const char * cb  = row_block(wrb); // the current step's first row's segment
    int          rem = p.nrows - wrb * rw;

    [[maybe_unused]] uint4    v[NP];
    [[maybe_unused]] uint32_t vok = 0;
    if constexpr (PF) {
#pragma unroll
        for (int q = 0; q < NP; ++q) {
            const uint4 * src = nullptr;
            if (unit_src(cb, rem, q, src)) {
                v[q] = __ldg(src);
                vok |= 1u << q;
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
        for (int q = 0; q < NP; ++q) {
            if constexpr (PF) {
                if (vok & (1u << q)) {
                    *unit_dst(q) = v[q];
                }
            } else {
                const uint4 * src = nullptr;
                if (unit_src(cb, rem, q, src)) {
                    *unit_dst(q) = __ldg(src);
                }
            }
        }
        __syncwarp();

        // the next step: the next iteration of this row block, or the warp's next row block
        int          nit  = it + 1;
        int          nwrb = wrb;
        const char * nb   = cb + seg;
        int          nrem = rem;
        if (nit == p.niter) {
            nit  = 0;
            nwrb = wrb + wstride;
            nb   = row_block(nwrb);
            nrem = p.nrows - nwrb * rw;
        }
        const bool more = nwrb < p.nwrb;
        if constexpr (PF) {
            vok = 0;
            if (more) {
#pragma unroll
                for (int q = 0; q < NP; ++q) {
                    const uint4 * src = nullptr;
                    if (unit_src(nb, nrem, q, src)) {
                        v[q] = __ldg(src);
                        vok |= 1u << q;
                    }
                }
            }
        }

        // this lane's super-block of each of the lane group's rows (rows past the matrix: their results are not written)
        kcol::unit_ptr wu[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int s = grp * R + i;
            const int h = ((int) (uintptr_t) cb + hoff[i]) & 15;
            wu[i] = kcol::make_unit<2>(smw + s * p.slot + h + sl * Q3M_SB);
        }
        const int sb = (it << lpr_log2) + sl;
        q3m_superblock<NT, R>(wu, sq + sb * QU, sd + sb * DW, sh + sb * DW, acc);

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
                const int row = wrb * rw + grp * R + i;
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
        cb  = nb;
        rem = nrem;
    }
}

static int q3m_env(const char * name, const int dflt) {
    const char * env = getenv(name);
    return env ? atoi(env) : dflt;
}

static int q3m_mode() {
    static const int mode = q3m_env("GGML_CUDA_MMVQ_PASCAL_Q3K_MT", 0);
    return mode;
}

// GGML_CUDA_MMVQ_PASCAL_Q3K_MT_R: rows per lane group, 1 (default) or 2
static int q3m_r() {
    static const int r = q3m_env("GGML_CUDA_MMVQ_PASCAL_Q3K_MT_R", 1) == 2 ? 2 : 1;
    return r;
}

static int q3m_fault() {
    static const int fault = q3m_env("GGML_CUDA_MMVQ_PASCAL_Q3K_MT_FAULT", 0) != 0 ? 1 : 0;
    return fault;
}

// log2 of the lanes per row: the largest power of two, at most 32, that divides the row's super-blocks (as
// kcol::pick_shape chooses it)
static int q3m_lpr_log2(const int nsb) {
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && nsb % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    return lpr_log2;
}

// The most distinct words of one bank that a 32-bit read of the same word of every lane's super-block asks for, with
// rows' slots slot bytes apart (row slot s = group * r + i holds row s of the warp row block from the 16-byte boundary
// below its segment, h = s * row_bytes mod 16 bytes on; a shift of every row by the same multiple of 4 bytes, as the next
// iteration or row block makes for Flash-Next's rows, keeps it).
static int q3m_conflicts(const int lpr_log2, const int r, const int row_bytes, const int slot) {
    const int L = 1 << lpr_log2;
    int worst = 0;
    for (int i = 0; i < r; ++i) {
        int word[WARP_SIZE];
        for (int lane = 0; lane < WARP_SIZE; ++lane) {
            const int s = (lane >> lpr_log2) * r + i;
            const int h = (int) (((int64_t) s * row_bytes) & 15);
            word[lane] = (s * slot + ((h + Q3M_SB * (lane & (L - 1))) & ~3)) / 4;
        }
        for (int bank = 0; bank < 32; ++bank) {
            int seen[WARP_SIZE];
            int n = 0;
            for (int lane = 0; lane < WARP_SIZE; ++lane) {
                if (word[lane] % 32 != bank) {
                    continue;
                }
                bool dup = false;
                for (int k = 0; k < n; ++k) {
                    dup = dup || seen[k] == word[lane];
                }
                if (!dup) {
                    seen[n++] = word[lane];
                }
            }
            worst = std::max(worst, n);
        }
    }
    return worst;
}

// 16-byte units of padding after each row slot of 8 L units: GGML_CUDA_MMVQ_PASCAL_Q3K_MT_PAD (0 to 15), else the
// fewest of 0..7 with the fewest conflicts (q3m_conflicts), found once per lane count, R and row_bytes mod 16.
static int q3m_pad(const int lpr_log2, const int r, const int row_bytes) {
    static const int pad_env = q3m_env("GGML_CUDA_MMVQ_PASCAL_Q3K_MT_PAD", -1);
    if (pad_env >= 0) {
        return std::min(pad_env, 15);
    }
    static int cache[6][3][16];
    static bool init = false;
    if (!init) {
        for (auto & a : cache) for (auto & b : a) for (int & c : b) c = -1;
        init = true;
    }
    int & c = cache[lpr_log2][r][row_bytes & 15];
    if (c < 0) {
        int best = 0, best_n = INT_MAX;
        for (int pad = 0; pad < 8; ++pad) {
            const int n = q3m_conflicts(lpr_log2, r, row_bytes, (16 << (lpr_log2 + 3)) + 16 * pad);
            if (n < best_n) {
                best   = pad;
                best_n = n;
            }
        }
        c = best;
    }
    return c;
}

static int q3m_slot(const int lpr_log2, const int r, const int row_bytes) {
    return (16 << (lpr_log2 + 3)) + 16 * q3m_pad(lpr_log2, r, row_bytes);
}

// The warps per block for nt tokens, nsb super-blocks per row and r rows per lane group: GGML_CUDA_MMVQ_PASCAL_Q3K_MT_NW
// (default and at most 6 at R = 1, 3 at R = 2), fewer where the shared memory would pass 48 KB; 0 if not even one warp
// fits.
static int q3m_nw(const int nt, const int nsb, const int r, const int row_bytes) {
    static const int nw_env = q3m_env("GGML_CUDA_MMVQ_PASCAL_Q3K_MT_NW", Q3M_NW_MAX);
    const int lpr_log2 = q3m_lpr_log2(nsb);
    int       nw  = std::min(std::max(nw_env, 1), Q3M_NW_MAX / r);
    const int act = q3m_act_bytes(nt, nsb);
    const int ws  = (WARP_SIZE >> lpr_log2) * r * q3m_slot(lpr_log2, r, row_bytes);
    while (nw > 0 && act + nw * ws > Q3M_SMEM_MAX) {
        nw--;
    }
    return nw;
}

template <int NT, int R>
static void q3m_launch(const q3m_args & p, const dim3 grid, const dim3 block, const int smem, const bool pf,
                       cudaStream_t stream) {
    if (pf) {
        mul_mat_vec_q_pascal_q3kmt<NT, R, true><<<grid, block, smem, stream>>>(p);
    } else {
        mul_mat_vec_q_pascal_q3kmt<NT, R, false><<<grid, block, smem, stream>>>(p);
    }
}

template <int R>
static void q3m_launch_r(const int nt, const q3m_args & p, const dim3 grid, const dim3 block, const int smem,
                         const bool pf, cudaStream_t stream) {
    switch (nt) {
        case 2:  q3m_launch<2, R>(p, grid, block, smem, pf, stream); break;
        case 3:  q3m_launch<3, R>(p, grid, block, smem, pf, stream); break;
        case 4:  q3m_launch<4, R>(p, grid, block, smem, pf, stream); break;
        case 5:  q3m_launch<5, R>(p, grid, block, smem, pf, stream); break;
        case 6:  q3m_launch<6, R>(p, grid, block, smem, pf, stream); break;
        case 7:  q3m_launch<7, R>(p, grid, block, smem, pf, stream); break;
        default: q3m_launch<8, R>(p, grid, block, smem, pf, stream); break;
    }
}

} // namespace

bool ggml_cuda_mmvq_pascal_q3kmt_takes(const int64_t K, const int64_t nrows, const int64_t ncols, const int64_t row_bytes,
                                       const void * x) {
    if (q3m_mode() == 0 || ncols < 2 || ncols > Q3M_MAX_COLS) {
        return false;
    }
    if (!ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) ||
            !ggml_cuda_mmvq_pascal_kcol_on(GGML_TYPE_Q3_K)) {
        return false;
    }
    // (the kernel takes a row slot's offset, at most 63 rows, in 32 bits)
    if (K < 2 * QK_K || K % QK_K != 0 || K > INT_MAX || nrows < 1 || nrows > INT_MAX - 64 || row_bytes > INT_MAX / 64) {
        return false;
    }
    // at least 2 lanes per row (an even number of super-blocks): a row slot of 8 L 16-byte units then holds the segment
    // from the 16-byte boundary below it and the 4 bytes past a super-block's end that the reads of its last word take
    const int nsb = (int) (K / QK_K);
    if (nsb % 2 != 0 || row_bytes < (int64_t) nsb * Q3M_SB || row_bytes % 2 != 0 || (uintptr_t) x % 2 != 0) {
        return false;
    }
    return q3m_nw((int) ncols, nsb, q3m_r(), (int) row_bytes) > 0;
}

bool ggml_cuda_mmvq_pascal_q3kmt(const void * vx, const void * vy, float * dst, const int64_t K, const int64_t nrows,
                                 const int64_t ncols, const int64_t row_bytes, const int64_t stride_col_y,
                                 const int64_t stride_col_dst, cudaStream_t stream) {
    if (!ggml_cuda_mmvq_pascal_q3kmt_takes(K, nrows, ncols, row_bytes, vx)) {
        return false;
    }
    const int64_t col_bytes_y = stride_col_y * (int64_t) sizeof(block_q8_1);
    if ((uintptr_t) vy % 4 != 0 || (ncols - 1) * col_bytes_y + (K / QK8_1) * (int64_t) sizeof(block_q8_1) > INT_MAX ||
            (ncols - 1) * stride_col_dst + nrows > INT_MAX) {
        return false;
    }

    const int nt  = (int) ncols;
    const int nsb = (int) (K / QK_K);
    const int r   = q3m_r();
    const int nw  = q3m_nw(nt, nsb, r, (int) row_bytes);

    q3m_args p;
    p.vx             = (const char *) vx;
    p.vy             = (const char *) vy;
    p.dst            = dst;
    p.nrows          = (int) nrows;
    p.row_bytes      = (int) row_bytes;
    p.nsb            = nsb;
    p.lpr_log2       = q3m_lpr_log2(nsb);
    p.niter          = nsb >> p.lpr_log2;
    const int rpw    = (WARP_SIZE >> p.lpr_log2) * r; // rows per warp row block
    p.nwrb           = (int) ((nrows + rpw - 1) / rpw);
    p.slot           = q3m_slot(p.lpr_log2, r, (int) row_bytes);
    p.col_bytes_y    = (int) col_bytes_y;
    p.stride_col_dst = (int) stride_col_dst;
    p.d_off          = q3m_q_bytes(nt, nsb) / 4;
    p.h_off          = (q3m_q_bytes(nt, nsb) + q3m_d_bytes(nt, nsb)) / 8;
    const int act    = q3m_act_bytes(nt, nsb);
    p.stage_off      = act;
    p.wstage         = rpw * p.slot;
    p.fault          = q3m_fault();
    const int smem   = act + nw * p.wstage;

    // a persistent grid: as many blocks as the SMs hold at once by shared memory and threads (the launch bounds keep the
    // registers for 2 blocks of the most warps), at most one per nw warp row blocks
    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int bps  = std::max(1, std::min(96 * 1024 / smem, 2048 / (nw * WARP_SIZE)));
    const int nblk = (int) std::min<int64_t>((p.nwrb + nw - 1) / nw, (int64_t) nsm * bps);
    const dim3 grid(nblk, 1, 1);
    const dim3 block(WARP_SIZE, nw, 1);
    const bool pf = q3m_mode() != 2;

    if (r == 1) {
        q3m_launch_r<1>(nt, p, grid, block, smem, pf, stream);
    } else {
        q3m_launch_r<2>(nt, p, grid, block, smem, pf, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_pascal_q3kmt_takes(const int64_t K, const int64_t nrows, const int64_t ncols, const int64_t row_bytes,
                                       const void * x) {
    GGML_UNUSED_VARS(K, nrows, ncols, row_bytes, x);
    return false;
}

bool ggml_cuda_mmvq_pascal_q3kmt(const void * vx, const void * vy, float * dst, const int64_t K, const int64_t nrows,
                                 const int64_t ncols, const int64_t row_bytes, const int64_t stride_col_y,
                                 const int64_t stride_col_dst, cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, vy, dst, K, nrows, ncols, row_bytes, stride_col_y, stride_col_dst, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
