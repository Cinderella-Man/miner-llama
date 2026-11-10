#include "mmvq-pascal-q5k.cuh"
#include "mmvq-pascal.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-impl.cuh"

// How the kernel works (2 to 4 tokens of a plain Q5_K product):
//   - The lanes, rows and summation order are mul_mat_vec_q_pascal's for Q5_K (mmvqp_choose: R = 1, C = 1, staged):
//     L = 2^lpr_log2 lanes per row (the largest power of two, at most 32, dividing the row's super-blocks), lane sl of a
//     row's group taking super-blocks sl, sl + L, ... in that order, its 176-byte segment staged by the lane group into
//     the warp's shared memory with 128-bit loads, and the lanes' sums added with the same xor shuffles. So every
//     token's result is the one-token kernel's, bit for bit.
//   - Each 32-weight sub-block of a row is decoded once into eight words of 5-bit codes and multiplied with dp4a by
//     every token's activations. The one-token kernel decodes it once per token (mul_mat_vec_q_pascal_mc) and the
//     k-column kernel stages every row's activation records per row (mul_mat_kcol, 3 shared loads per sub-block, token
//     and row).
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
//     its block sums serve many rows: K = 2560 at 4 tokens is 13 KB, 6 warps' weight staging 33 KB.
//   - By default each warp loads its next segment into registers before it computes the current one (11 128-bit loads
//     per lane in flight during the 64 x 4 dp4a of a super-block).

#define Q5KMT_NW_MAX   6                    // warps per block, at most
#define Q5KMT_MAX_COLS 4
#define Q5KMT_SB       176                  // bytes of a super-block (block_q5_K)
#define Q5KMT_NU       (Q5KMT_SB / 16)      // its 16-byte units
#define Q5KMT_STAGE    (WARP_SIZE * Q5KMT_SB) // a warp's staging buffer: one super-block per lane
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
    int          nwrb;          // warp row blocks: rows / (32 >> lpr_log2), rounded up
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

// One super-block of a row (sw: its 176 bytes in shared memory) times every token's activations at its position (aq:
// the int8 words, ads: (d8, S)), added to acc[j] exactly as mmvqp_dot_q5_K adds it for token j alone.
template <int NT>
static __device__ __forceinline__ void q5kmt_superblock(const char * sw, const uint32_t * aq, const float * ads,
                                                        float (&acc)[NT]) {
    uint32_t h[12]; // dm, 3 words of scales, 8 words qh
    mmvqp_lds_words<48, 16>(h, sw);
    const uint32_t sc_lo = h[1] & 0x3f3f3f3f;                                   // scales of sub-blocks 0..3
    const uint32_t m_lo  = h[2] & 0x3f3f3f3f;                                   // mins of sub-blocks 0..3
    const uint32_t sc_hi = (h[3] & 0x0f0f0f0f) | ((h[1] >> 2) & 0x30303030);   // scales of 4..7
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
        mmvqp_lds_words<32, 16>(qs, sw + 48 + 32*p);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int s = 2*p + hh;
            int q[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint32_t nib = hh ? (qs[k] >> 4) & 0x0F0F0F0F : qs[k] & 0x0F0F0F0F;
                const uint32_t hb  = s <= 4 ? (h[4 + k] << (4 - s)) & 0x10101010 : (h[4 + k] >> (s - 4)) & 0x10101010;
                q[k] = (int) (nib | hb);
            }
            // (float) sc and (float) m exactly: the 6-bit byte as the low mantissa bits of 12582912 = 2^23 + 2^22
            const uint32_t sel  = 0x7650 | (s & 3);
            const float    scf  = __int_as_float((int) __byte_perm(s < 4 ? sc_lo : sc_hi, 0x4B400000u, sel)) - 12582912.0f;
            const float    mf   = __int_as_float((int) __byte_perm(s < 4 ? m_lo  : m_hi,  0x4B400000u, sel)) - 12582912.0f;
            const float    cneg = -12582912.0f * scf;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint4  u0 = *(const uint4  *) (aq + (s*NT + j)*8);
                const uint4  u1 = *(const uint4  *) (aq + (s*NT + j)*8 + 4);
                const float2 ds = *(const float2 *) (ads + (s*NT + j)*2);
                int ia = 0x4B400000; // 12582912 as a float: the result reads as 12582912 + isum
                ia = ggml_cuda_dp4a(q[0], (int) u0.x, ia);
                ia = ggml_cuda_dp4a(q[1], (int) u0.y, ia);
                ia = ggml_cuda_dp4a(q[2], (int) u0.z, ia);
                ia = ggml_cuda_dp4a(q[3], (int) u0.w, ia);
                ia = ggml_cuda_dp4a(q[4], (int) u1.x, ia);
                ia = ggml_cuda_dp4a(q[5], (int) u1.y, ia);
                ia = ggml_cuda_dp4a(q[6], (int) u1.z, ia);
                ia = ggml_cuda_dp4a(q[7], (int) u1.w, ia);
                const float x = fmaf(__int_as_float(ia), scf, cneg); // (float) (isum * sc), exactly
                sumd[j] = fmaf(ds.x, x, sumd[j]);
                summ[j] = fmaf(mf, ds.y, summ[j]);
            }
        }
    }
    const float d    = mmvqp_h2f_lo(h[0]);
    const float dmin = mmvqp_h2f_hi(h[0]);
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j] = acc[j] + d * sumd[j] - dmin * summ[j];
    }
}

// Grid: (blocks, 1, 1), each looping over warp row blocks; block (32, NW). Dynamic shared memory: the staged activations
// (int8 words, then (d8, S)), then the NW warps' staging buffers.
template <int NT, bool PF>
__launch_bounds__(Q5KMT_NW_MAX*WARP_SIZE, 2)
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
    const int gpw      = WARP_SIZE >> lpr_log2;      // rows per warp row block
    const int seg      = Q5KMT_SB << lpr_log2;       // bytes of a row's segment per iteration
    char *       smw = (char *) q5kmt_sm + p.stage_off + wid * Q5KMT_STAGE + grp * seg;
    const char * sws = smw + sl * Q5KMT_SB;          // this lane's super-block in the staging buffer
    const int    wstride = gridDim.x * nw;

    int wrb = blockIdx.x * nw + wid;
    if (wrb >= p.nwrb) {
        return;
    }
    // the lane group's row (clamped, so that the staging reads a valid row; its results are not written)
    const auto row_ptr = [&](const int w) {
        return p.vx + (int64_t) min(w * gpw + grp, p.nrows - 1) * p.row_bytes;
    };
    const char * rw = row_ptr(wrb);
    int it = 0;

    [[maybe_unused]] uint4 v[Q5KMT_NU];
    if constexpr (PF) {
#pragma unroll
        for (int k = 0; k < Q5KMT_NU; ++k) {
            v[k] = __ldg((const uint4 *) rw + sl + (k << lpr_log2));
        }
    }

    float acc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j] = 0.0f;
    }

    while (true) {
        // the current segment into the warp's staging buffer
        {
            uint4 * dsm = (uint4 *) smw;
            if constexpr (PF) {
#pragma unroll
                for (int k = 0; k < Q5KMT_NU; ++k) {
                    dsm[sl + (k << lpr_log2)] = v[k];
                }
            } else {
                const uint4 * src = (const uint4 *) (rw + it * seg);
#pragma unroll
                for (int k = 0; k < Q5KMT_NU; ++k) {
                    dsm[sl + (k << lpr_log2)] = __ldg(src + sl + (k << lpr_log2));
                }
            }
        }
        __syncwarp();

        // the next step: the next iteration of this row block, or the warp's next row block
        int          nit  = it + 1;
        int          nwrb = wrb;
        const char * nrw  = rw;
        if (nit == p.niter) {
            nit  = 0;
            nwrb = wrb + wstride;
            nrw  = nwrb < p.nwrb ? row_ptr(nwrb) : rw;
        }
        const bool more = nwrb < p.nwrb;
        if constexpr (PF) {
            if (more) {
                const uint4 * src = (const uint4 *) (nrw + nit * seg);
#pragma unroll
                for (int k = 0; k < Q5KMT_NU; ++k) {
                    v[k] = __ldg(src + sl + (k << lpr_log2));
                }
            }
        }

        const int sb = (it << lpr_log2) + sl;
        q5kmt_superblock<NT>(sws, aq + sb * QW, ads + sb * DW, acc);

        if (it == p.niter - 1) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                    if (off < lpr) {
                        acc[j] += __shfl_xor_sync(0xFFFFFFFF, acc[j], off, WARP_SIZE);
                    }
                }
            }
            const int row = wrb * gpw + grp;
            if (sl == 0 && row < p.nrows) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    p.dst[j * p.stride_col_dst + row] = acc[j];
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[j] = 0.0f;
            }
        }
        __syncwarp();
        if (!more) {
            break;
        }
        it  = nit;
        wrb = nwrb;
        rw  = nrw;
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

// The warps per block for nt tokens and nsb super-blocks per row: GGML_CUDA_MMVQ_PASCAL_Q5K_MT_NW (default 6), fewer
// where the shared memory would pass 48 KB; 0 if not even one warp fits.
static int q5kmt_nw(const int nt, const int nsb) {
    static const int nw_env = q5kmt_env("GGML_CUDA_MMVQ_PASCAL_Q5K_MT_NW", Q5KMT_NW_MAX);
    int nw = std::min(std::max(nw_env, 1), Q5KMT_NW_MAX);
    const int act = q5kmt_act_bytes(nt, nsb);
    while (nw > 0 && act + nw * Q5KMT_STAGE > Q5KMT_SMEM_MAX) {
        nw--;
    }
    return nw;
}

static int q5kmt_lpr_log2(const int64_t nsb) {
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && nsb % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    return lpr_log2;
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
    const int gpw    = WARP_SIZE >> p.lpr_log2;
    p.nwrb           = (int) ((nrows + gpw - 1) / gpw);
    p.col_bytes_y    = (int) col_bytes_y;
    p.stride_col_dst = (int) stride_col_dst;
    p.ds_off         = nsb * (64 * nt + 4);
    const int act    = q5kmt_act_bytes(nt, nsb);
    p.stage_off      = act;
    const int smem   = act + nw * Q5KMT_STAGE;

    // a persistent grid: as many blocks as the SMs hold at once (by shared memory and threads; the launch bounds keep
    // the registers for 2 blocks of 6 warps), at most one per nw warp row blocks
    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int bps  = std::max(1, std::min(96 * 1024 / smem, 2048 / (nw * WARP_SIZE)));
    const int nblk = (int) std::min<int64_t>((p.nwrb + nw - 1) / nw, (int64_t) nsm * bps);
    const dim3 grid(nblk, 1, 1);
    const dim3 block(WARP_SIZE, nw, 1);
    const bool pf = q5kmt_mode() != 2;

#define Q5KMT_LAUNCH(NT) \
    if (pf) { mul_mat_vec_q5_K_pascal_mt<NT, true ><<<grid, block, smem, stream>>>(p); } \
    else    { mul_mat_vec_q5_K_pascal_mt<NT, false><<<grid, block, smem, stream>>>(p); }
    switch (nt) {
        case 2:  Q5KMT_LAUNCH(2); break;
        case 3:  Q5KMT_LAUNCH(3); break;
        default: Q5KMT_LAUNCH(4); break;
    }
#undef Q5KMT_LAUNCH
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
