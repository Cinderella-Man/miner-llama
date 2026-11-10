#include "mmvq-pascal-q5k.cuh"
#include "mmvq-pascal.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-impl.cuh"

// How the kernel works (2 to 4 tokens of a plain Q5_K product):
//   - The lanes and summation order are mul_mat_vec_q_pascal's for Q5_K (mmvqp_choose: C = 1, staged): L = 2^lpr_log2
//     lanes per row (the largest power of two, at most 32, dividing the row's super-blocks), lane sl of a row's group
//     taking super-blocks sl, sl + L, ... in that order, its 176-byte segment staged by the lane group into the warp's
//     shared memory with 128-bit loads, and the lanes' sums added with the same xor shuffles. So every token's result is
//     the one-token kernel's, bit for bit. A lane group owns R = 1 (default) or 2 consecutive rows, which share every
//     load of the activations; R changes no row's sum.
//   - Each 32-weight sub-block of a row is decoded once into eight words of 5-bit codes and multiplied with dp4a by
//     every token's activations. The earlier multi-token kernel decodes it once per token (mul_mat_vec_q_pascal_mc).
//   - The floating-point steps are mmvqp_dot_q5_K's: per sub-block s and token, sumd = fmaf(d8, (float) (isum * sc), sumd)
//     and summ = fmaf((float) m, S, summ), with S = d8 * (float) sum(int8); per super-block acc + d * sumd - dmin * summ.
//     The integer-to-float conversions are made without the quarter-rate I2F and the integer multiply, exactly: the dp4a
//     chain starts at 0x4B400000, so its result read as a float is 12582912 + isum (|isum| <= 32 * 31 * 127 < 2^22), and
//     fmaf(12582912 + isum, sc, -12582912 * sc) is isum * sc rounded once, which is exact (|isum * sc| < 2^24); sc and m
//     (6 bits) come out of a __byte_perm into the same float. These are the values the one-token kernel's conversions
//     give, so every FP32 operation sees the same operands.
//   - Each block stages all tokens' activations once into shared memory (the int8 words and (d8, S) per sub-block and
//     token, padded so that the L lanes of a group read distinct banks; every lane group of a warp reads the same L
//     records, a broadcast) and loops over row blocks (a persistent grid of about 2 blocks per SM), so the staging and
//     its block sums serve many rows: K = 2560 at 4 tokens is 13 KB beside 33 KB of weight staging (6 warps at R = 1,
//     3 at R = 2). The k-column kernel stages each block's activation records again for every 64 rows, in tiles.
//   - By default each warp loads its next segments into registers before it computes the current ones (11 128-bit loads
//     per lane and row in flight during the 64 x 4 dp4a of a super-block).

#define Q5KMT_NW_MAX   6                    // warps per block at R = 1, at most (3 at R = 2)
#define Q5KMT_MAX_COLS 4
#define Q5KMT_SB       176                  // bytes of a super-block (block_q5_K)
#define Q5KMT_NU       (Q5KMT_SB / 16)      // its 16-byte units
#define Q5KMT_STAGE    (WARP_SIZE * Q5KMT_SB) // a warp's staging buffer per row of its lane groups
#define Q5KMT_SMEM_MAX (48 * 1024)

static_assert(sizeof(block_q5_K) == Q5KMT_SB, "block_q5_K is 176 bytes");

namespace {

struct q5kmt_args {
    const char * vx;
    const char * vy;            // block_q8_1, column j from j * col_bytes_y
    float      * dst;           // column j from j * stride_col_dst
    int          nrows;
    int          row_bytes;
    int          nsb;           // super-blocks per row
    int          lpr_log2;      // lanes per row L = 1 << lpr_log2
    int          niter;         // super-blocks per lane and row (nsb / L)
    int          nwrb;          // warp row blocks: rows / ((32 >> lpr_log2) * R), rounded up
    int          col_bytes_y;
    int          stride_col_dst;
    int          ds_off;        // words from the start of shared memory to the (d8, S) region
    int          stage_off;     // bytes from the start of shared memory to the warps' staging buffers
};

// Words per super-block position of the staged activations: NT tokens x 8 sub-blocks of 8 int8 words, and of (d8, S),
// each padded so that consecutive positions start an odd number of 4-word (2-word) bank groups apart.
template <int NT> constexpr __host__ __device__ int q5kmt_q_words()  { return 64 * NT + 4; }
template <int NT> constexpr __host__ __device__ int q5kmt_ds_words() { return 16 * NT + 2; }

static int q5kmt_act_bytes(const int nt, const int nsb) {
    const int q  = nsb * (64 * nt + 4) * 4;
    const int ds = (nsb * (16 * nt + 2) * 4 + 15) / 16 * 16;
    return q + ds;
}

// The super-blocks of R rows at one position (sw[i]: row i's 176 bytes in shared memory) times every token's
// activations at that position (aq: the int8 words, ads: (d8, S)), added to acc[i][j] exactly as mmvqp_dot_q5_K adds
// row i's super-block for token j alone.
template <int NT, int R>
static __device__ __forceinline__ void q5kmt_superblock(const char * const (&sw)[R], const uint32_t * aq, const float * ads,
                                                        float (&acc)[R][NT]) {
    uint32_t h[R][12]; // dm, 3 words of scales, 8 words qh
    uint32_t sc_lo[R], m_lo[R], sc_hi[R], m_hi[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        mmvqp_lds_words<48, 16>(h[i], sw[i]);
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
            mmvqp_lds_words<32, 16>(qs[i], sw[i] + 48 + 32*p);
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
                    const uint32_t nib = hh ? (qs[i][k] >> 4) & 0x0F0F0F0F : qs[i][k] & 0x0F0F0F0F;
                    const uint32_t hb  = s <= 4 ? (h[i][4 + k] << (4 - s)) & 0x10101010 : (h[i][4 + k] >> (s - 4)) & 0x10101010;
                    q[i][k] = (int) (nib | hb);
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

// Grid: (blocks, 1, 1), each looping over warp row blocks; block (32, NW). Dynamic shared memory: the staged activations
// (int8 words, then (d8, S)), then the NW warps' staging buffers (R rows per lane group).
template <int NT, int R, bool PF>
__launch_bounds__(Q5KMT_NW_MAX*WARP_SIZE/R, 2)
static __global__ void mul_mat_vec_q5_K_pascal_mt(const q5kmt_args p) {
    constexpr int QW = q5kmt_q_words<NT>();
    constexpr int DW = q5kmt_ds_words<NT>();
    extern __shared__ uint4 q5kmt_sm[];
    uint32_t * aq  = (uint32_t *) q5kmt_sm;
    float    * ads = (float *) q5kmt_sm + p.ds_off;

    const int lane = threadIdx.x;
    const int wid  = threadIdx.y;
    const int nw   = blockDim.y;

    ggml_cuda_pdl_sync();

    // every token's activation blocks: the int8 words and (d8, S), S = d8 * the sum of the int8 values, as
    // mmvqp_load_act computes them for Q5_K
    {
        const int nab = p.nsb * 8;
        for (int e = wid * WARP_SIZE + lane; e < NT * nab; e += nw * WARP_SIZE) {
            const int j = e / nab;
            const int b = e - j * nab;
            const uint32_t * src = (const uint32_t *) (p.vy + j * p.col_bytes_y + b * (int) sizeof(block_q8_1));
            uint32_t w[9];
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                w[k] = __ldg(src + k);
            }
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum = ggml_cuda_dp4a(0x01010101, (int) w[1 + k], sum);
            }
            const float d8 = mmvqp_h2f_lo(w[0]);
            const float S  = d8 * (float) sum;
            const int   sb = b >> 3;
            const int   s  = b & 7;
            uint4 * dq = (uint4 *) (aq + sb * QW + (s*NT + j) * 8);
            dq[0] = make_uint4(w[1], w[2], w[3], w[4]);
            dq[1] = make_uint4(w[5], w[6], w[7], w[8]);
            *(float2 *) (ads + sb * DW + (s*NT + j) * 2) = make_float2(d8, S);
        }
    }
    __syncthreads();

    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int gpw      = WARP_SIZE >> lpr_log2;      // lane groups per warp
    const int seg      = Q5KMT_SB << lpr_log2;       // bytes of a row's segment per iteration
    // the lane group's R row segments in the warp's staging buffer, and this lane's super-block in each
    char * smw = (char *) q5kmt_sm + p.stage_off + wid * R * Q5KMT_STAGE + grp * R * seg;
    const char * sws[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        sws[i] = smw + i * seg + sl * Q5KMT_SB;
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

    [[maybe_unused]] uint4 v[R][Q5KMT_NU];
    if constexpr (PF) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int k = 0; k < Q5KMT_NU; ++k) {
                v[i][k] = __ldg((const uint4 *) rw[i] + sl + (k << lpr_log2));
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
                for (int k = 0; k < Q5KMT_NU; ++k) {
                    dsm[sl + (k << lpr_log2)] = v[i][k];
                }
            } else {
                const uint4 * src = (const uint4 *) (rw[i] + it * seg);
#pragma unroll
                for (int k = 0; k < Q5KMT_NU; ++k) {
                    dsm[sl + (k << lpr_log2)] = __ldg(src + sl + (k << lpr_log2));
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
                    for (int k = 0; k < Q5KMT_NU; ++k) {
                        v[i][k] = __ldg(src + sl + (k << lpr_log2));
                    }
                }
            }
        }

        const int sb = (it << lpr_log2) + sl;
        q5kmt_superblock<NT, R>(sws, aq + sb * QW, ads + sb * DW, acc);

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

static int q5kmt_env(const char * name, const int dflt) {
    const char * env = getenv(name);
    return env ? atoi(env) : dflt;
}

static int q5kmt_mode() {
    static const int mode = q5kmt_env("GGML_CUDA_MMVQ_PASCAL_Q5K_MT", 0);
    return mode;
}

// GGML_CUDA_MMVQ_PASCAL_Q5K_MT_R: rows per lane group, 1 (default) or 2
static int q5kmt_r() {
    static const int r = q5kmt_env("GGML_CUDA_MMVQ_PASCAL_Q5K_MT_R", 1) == 2 ? 2 : 1;
    return r;
}

// The warps per block for nt tokens and nsb super-blocks per row: GGML_CUDA_MMVQ_PASCAL_Q5K_MT_NW (default and at most
// 6 at R = 1, 3 at R = 2), fewer where the shared memory would pass 48 KB; 0 if not even one warp fits.
static int q5kmt_nw(const int nt, const int nsb) {
    static const int nw_env = q5kmt_env("GGML_CUDA_MMVQ_PASCAL_Q5K_MT_NW", Q5KMT_NW_MAX);
    const int r   = q5kmt_r();
    int       nw  = std::min(std::max(nw_env, 1), Q5KMT_NW_MAX / r);
    const int act = q5kmt_act_bytes(nt, nsb);
    while (nw > 0 && act + nw * r * Q5KMT_STAGE > Q5KMT_SMEM_MAX) {
        nw--;
    }
    return nw;
}

static int q5kmt_lpr_log2(const int nsb) {
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && nsb % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    return lpr_log2;
}

template <int NT, int R>
static void q5kmt_launch(const q5kmt_args & p, const dim3 grid, const dim3 block, const int smem, const bool pf,
                         cudaStream_t stream) {
    if (pf) {
        mul_mat_vec_q5_K_pascal_mt<NT, R, true><<<grid, block, smem, stream>>>(p);
    } else {
        mul_mat_vec_q5_K_pascal_mt<NT, R, false><<<grid, block, smem, stream>>>(p);
    }
}

} // namespace

bool ggml_cuda_mmvq_pascal_q5k_mt_takes(const int64_t K, const int64_t nrows, const int64_t ncols, const int64_t row_bytes,
                                        const void * x) {
    if (q5kmt_mode() == 0 || ncols < 2 || ncols > Q5KMT_MAX_COLS) {
        return false;
    }
    if (!ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) ||
            ggml_cuda_mmvq_pascal_skips(GGML_TYPE_Q5_K)) {
        return false;
    }
    if (K < QK_K || K % QK_K != 0 || K > INT_MAX || nrows < 1 || nrows > INT_MAX || row_bytes > INT_MAX ||
            row_bytes < (K / QK_K) * Q5KMT_SB || row_bytes % 16 != 0 || (uintptr_t) x % 16 != 0) {
        return false;
    }
    return q5kmt_nw((int) ncols, (int) (K / QK_K)) > 0;
}

bool ggml_cuda_mmvq_pascal_q5k_mt(const void * vx, const void * vy, float * dst, const int64_t K, const int64_t nrows,
                                  const int64_t ncols, const int64_t row_bytes, const int64_t stride_col_y,
                                  const int64_t stride_col_dst, cudaStream_t stream) {
    if (!ggml_cuda_mmvq_pascal_q5k_mt_takes(K, nrows, ncols, row_bytes, vx)) {
        return false;
    }
    const int64_t col_bytes_y = stride_col_y * (int64_t) sizeof(block_q8_1);
    if ((uintptr_t) vy % 4 != 0 || (ncols - 1) * col_bytes_y + (K / QK8_1) * (int64_t) sizeof(block_q8_1) > INT_MAX ||
            (ncols - 1) * stride_col_dst + nrows > INT_MAX) {
        return false;
    }

    const int nt  = (int) ncols;
    const int nsb = (int) (K / QK_K);
    const int r   = q5kmt_r();
    const int nw  = q5kmt_nw(nt, nsb);

    q5kmt_args p;
    p.vx             = (const char *) vx;
    p.vy             = (const char *) vy;
    p.dst            = dst;
    p.nrows          = (int) nrows;
    p.row_bytes      = (int) row_bytes;
    p.nsb            = nsb;
    p.lpr_log2       = q5kmt_lpr_log2(nsb);
    p.niter          = nsb >> p.lpr_log2;
    const int rpw    = (WARP_SIZE >> p.lpr_log2) * r; // rows per warp row block
    p.nwrb           = (int) ((nrows + rpw - 1) / rpw);
    p.col_bytes_y    = (int) col_bytes_y;
    p.stride_col_dst = (int) stride_col_dst;
    p.ds_off         = nsb * (64 * nt + 4);
    const int act    = q5kmt_act_bytes(nt, nsb);
    p.stage_off      = act;
    const int smem   = act + nw * r * Q5KMT_STAGE;

    // a persistent grid: as many blocks as the SMs hold at once by shared memory and threads (the launch bounds keep the
    // registers for 2 blocks of the most warps), at most one per nw warp row blocks
    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int bps  = std::max(1, std::min(96 * 1024 / smem, 2048 / (nw * WARP_SIZE)));
    const int nblk = (int) std::min<int64_t>((p.nwrb + nw - 1) / nw, (int64_t) nsm * bps);
    const dim3 grid(nblk, 1, 1);
    const dim3 block(WARP_SIZE, nw, 1);
    const bool pf = q5kmt_mode() != 2;

    switch (nt * 4 + r) {
        case 2*4 + 1: q5kmt_launch<2, 1>(p, grid, block, smem, pf, stream); break;
        case 3*4 + 1: q5kmt_launch<3, 1>(p, grid, block, smem, pf, stream); break;
        case 4*4 + 1: q5kmt_launch<4, 1>(p, grid, block, smem, pf, stream); break;
        case 2*4 + 2: q5kmt_launch<2, 2>(p, grid, block, smem, pf, stream); break;
        case 3*4 + 2: q5kmt_launch<3, 2>(p, grid, block, smem, pf, stream); break;
        default:      q5kmt_launch<4, 2>(p, grid, block, smem, pf, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_pascal_q5k_mt_takes(const int64_t K, const int64_t nrows, const int64_t ncols, const int64_t row_bytes,
                                        const void * x) {
    GGML_UNUSED_VARS(K, nrows, ncols, row_bytes, x);
    return false;
}

bool ggml_cuda_mmvq_pascal_q5k_mt(const void * vx, const void * vy, float * dst, const int64_t K, const int64_t nrows,
                                  const int64_t ncols, const int64_t row_bytes, const int64_t stride_col_y,
                                  const int64_t stride_col_dst, cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, vy, dst, K, nrows, ncols, row_bytes, stride_col_y, stride_col_dst, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
