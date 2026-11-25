#include "mmvq-pascal-mtok.cuh"
#include "mmvq-pascal.cuh"
#include "mmvq-pascal-impl.cuh"
#include "mmvq-pascal-mtok-iq.cuh"
#include "mmvq-pascal-mtok-np.cuh"
#include "unary.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// How the kernel splits the work (Q2_0, MUL_MAT_ID, 2 to MMVQP_MTOK_MAX_COLS tokens):
//   - Groups: block y belongs to entry y of the step's (token, slot) entries (entry i = token i / nslots, slot
//     i % nslots). Among the entries that picked entry y's expert, in entry order, every gcap-th starts a group: the
//     block of such an entry serves it and the next gcap - 1 entries of that expert, and every other block exits at once.
//     With gcap = the token count (default) and no token picking an expert twice, every distinct expert of the step is
//     one group. Each warp finds its block's group itself with 8 ballots over the ids (one load per lane and 32 entries,
//     at most MMVQP_MTOK_MAX_ENTRIES entries), without shared memory or a barrier; mul_mat_vec_q_pascal_mc instead walks
//     the ids entry by entry in every thread.
//   - The block stages its group's activations in shared memory once, for all its row blocks: each Q8_1 block's int8
//     values regrouped into even and odd bytes as mmvqp_load_act does for Q2_0, in runs of 32 bytes with 16 bytes of
//     padding per lane chunk (so that the lanes of a row, which read different chunks, hit different banks), and the FP32
//     scales beside them. A token's 32 weights then cost two 128-bit shared loads, shared by the gate and up rows, and
//     no shuffling of bytes.
//   - Lane groups, blocks per lane (2) and weight staging are those of the one-token kernel (mmvq-pascal.cu): gate/up
//     (K = 2,560) 4 lanes per row, row segments staged through shared memory with 128-bit loads; down (K = 640, 180-byte
//     rows) one lane per row, each warp staging its 32 rows at once (Q825), or loading them directly where its tiles
//     can't be staged. One row per lane group: the accumulators of up to 8 tokens take the registers that the one-token
//     kernel gives a second row.
//   - Each Q2_0 code word is looked up once (4 __byte_perm, 3 shifts per 16 weights) into int8 words and multiplied by
//     every token of the group (4 dp4a per 16 weights per token); the one-token kernel repeats the lookups per token.
//   - A block loops over rb consecutive row blocks of its expert, so the group search and the activation staging serve
//     rb x 32 rows (gate/up) or rb x 128 rows (down).
//   - The routed + shared expert launch adds one block row (blockIdx.y = gridDim.y - 1) that serves all tokens with the
//     shared expert.
// Each token's sums are formed exactly as in mul_mat_vec_q_pascal: the same integer sums, the same FP32 operations in the
// same order and the same reduction over a row's lanes, so every result is bit-identical to the one-token kernel's.

#define MMVQP_MTOK_NWARPS      4
#define MMVQP_MTOK_MAX_COLS    8   // tokens, and entries per group
#define MMVQP_MTOK_MAX_ENTRIES 128 // (token, slot) entries per step: 4 per lane of each warp
#ifndef MMVQP_MTOK_MIN_BLOCKS
#define MMVQP_MTOK_MIN_BLOCKS  3   // resident blocks per SM the registers must allow (3: at most 170 per thread)
#endif

namespace {

constexpr int MTOK_C   = 2;                                 // Q2_0 blocks per lane and iteration
constexpr int MTOK_WB  = MTOK_C * (int) sizeof(block_q2_0); // weight bytes per lane and iteration
constexpr int MTOK_CNA = MTOK_C * (QK2_0 / QK8_1);          // Q8_1 activation blocks per lane chunk
constexpr int MTOK_US  = 8 * MTOK_CNA + 4;                  // staged words per lane chunk: the int8 values, 4 of padding

// how a lane gets its weight chunks: staged row segments, staged warp tiles of whole rows, or direct loads
enum mtok_staging { MTOK_SEG = 0, MTOK_TILE = 1, MTOK_DIRECT = 2 };

// Byte strides for x and the activations, float strides for dst.
struct mtok_args {
    const char    * vx;
    const char    * vy;
    const int32_t * ids;
    float         * dst;
    int             nrows;
    int             row_bytes;
    int             lpr_log2;   // lanes per row = 1 << lpr_log2
    int             niter;      // iterations per lane
    int             nrowblocks;
    int             rb;         // row blocks per block
    int             gcap;       // entries per group
    int             ncols;      // tokens
    int             nslots;     // ids: entries per token (experts used)
    int             ids_stride;
    int             nq;         // Q8_1 blocks per activation row
    int             act_u;      // bytes of a member's staged int8 values, followed by its nq FP32 scales
    int             act_tok;    // bytes per member
    int             tile_bytes; // MTOK_TILE: the warp tiles, at the start of the dynamic shared memory
    uint3           nchannels_y;
    uint3           sample_ratio;
    int64_t         channel_bytes_x;
    int64_t         sample_bytes_x;
    int64_t         channel_bytes_y;
    int64_t         sample_bytes_y;
    int64_t         col_bytes_y;
    int64_t         stride_channel_dst;
    int64_t         stride_sample_dst;
    int64_t         stride_col_dst;
};

// The group of block y (routed experts): the entries of entry y's expert from entry y on, at most gcap of them, if y
// starts one. Lane j < nt returns member j's activation offset (bytes from vy) and output offset (floats from dst), which
// the host checked to fit in an int.
static __device__ __forceinline__ bool mtok_group(const mtok_args & p, const uint32_t sample_dst, int & nt, int & yoff,
                                                  int & doff, uint32_t & expert) {
    const int lane = threadIdx.x;
    const int ne   = p.ncols * p.nslots;
    const int y    = blockIdx.y;
    const int ty   = y / p.nslots;
    const int ey   = p.ids[(y - ty*p.nslots) + ty*p.ids_stride];
    if (ey < 0) {
        // a skipped slot (id -1, moe-tail.cuh): zeros in its rows of every row block of the block, and no expert's rows
        // are read; no other entry matches it
        float * dst = p.dst + sample_dst*p.stride_sample_dst + (y - ty*p.nslots)*p.stride_channel_dst + ty*p.stride_col_dst;
        const int sl  = lane & ((1 << p.lpr_log2) - 1);
        const int grp = lane >> p.lpr_log2;
        for (int rbi = 0; rbi < p.rb; ++rbi) {
            const int rblk = (int) blockIdx.x * p.rb + rbi;
            if (rblk >= p.nrowblocks) {
                break;
            }
            const int row0 = ((rblk * MMVQP_MTOK_NWARPS + (int) threadIdx.y) << (5 - p.lpr_log2)) + grp;
            if (sl == 0 && row0 < p.nrows) {
                dst[row0] = 0.0f;
            }
        }
        return false;
    }
    int      rank = 0; // entries before y with the same expert
    uint32_t mk[4];    // entries from y on with the same expert, 32 per word
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const int  i   = lane + WARP_SIZE*h;
        const int  t   = i / p.nslots;
        const bool hit = i < ne && p.ids[(i - t*p.nslots) + t*p.ids_stride] == ey;
        rank += __popc(__ballot_sync(0xFFFFFFFF, hit && i < y));
        mk[h] = __ballot_sync(0xFFFFFFFF, hit && i >= y);
    }
    if (rank % p.gcap != 0) {
        return false;
    }
    // member j is the j-th set bit of mk
    int      total = 0;
    int      h     = -1;
    int      r     = 0;
    uint32_t m     = 0;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        const int c = __popc(mk[hh]);
        if (h < 0 && lane < total + c) {
            h = hh;
            r = lane - total;
            m = mk[hh];
        }
        total += c;
    }
    nt = min(p.gcap, total);
    if (lane < nt) {
        for (int q = 0; q < r; ++q) {
            m &= m - 1;
        }
        const int      i         = WARP_SIZE*h + __ffs(m) - 1;
        const int      t         = i / p.nslots;
        const int      k         = i - t*p.nslots;
        const uint32_t channel_y = fastmodulo((uint32_t) k, p.nchannels_y);
        yoff = (int) (sample_dst*p.sample_bytes_y + channel_y*p.channel_bytes_y + t*p.col_bytes_y);
        doff = (int) (sample_dst*p.stride_sample_dst + k*p.stride_channel_dst + t*p.stride_col_dst);
    }
    expert = (uint32_t) ey;
    return true;
}

// Q2_0 code word q (weights 0..15 in bit pairs) as the int8 words mmvqp_q2_0_isum multiplies: weights (0, 2, 4, 6),
// (1, 3, 5, 7), (8, 10, 12, 14), (9, 11, 13, 15), values -1..2.
static __device__ __forceinline__ void mtok_decode(const uint32_t q, int * out) {
    constexpr uint32_t t = 0x020100FF; // codes 0, 1, 2, 3 -> -1, 0, 1, 2
    out[0] = (int) __byte_perm(t, t, q);
    out[1] = (int) __byte_perm(t, t, q >> 2);
    out[2] = (int) __byte_perm(t, t, q >> 16);
    out[3] = (int) __byte_perm(t, t, q >> 18);
}

// MTOK_TILE: the warp's rows of row block rblk of matrix x (contiguous in memory) copied into its part of the tiles with
// coalesced 128-bit loads (mmvqp_stage_tile of mmvq-pascal.cu for one row per lane group). Returns the warp's copy.
static __device__ __forceinline__ const char * mtok_stage_tile(const char * x, const mtok_args & p, const int rblk,
                                                               char * tiles) {
    constexpr int UN = 4; // loads in flight per lane
    const int wrows  = WARP_SIZE >> p.lpr_log2;
    const int wrow0  = (rblk * MMVQP_MTOK_NWARPS + (int) threadIdx.y) * wrows;
    const int wbytes = wrows * p.row_bytes;
    uint4       * dsm = (uint4 *) (tiles + threadIdx.y * wbytes);
    const uint4 * src = (const uint4 *) (x + (int64_t) wrow0 * p.row_bytes);
    const int     nu  = (max(0, min(wrows, p.nrows - wrow0)) * p.row_bytes) >> 4;
    __syncwarp(); // every lane is done with the warp's previous tile
    for (int u0 = threadIdx.x; u0 < nu; u0 += UN * WARP_SIZE) {
        uint4 v[UN];
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            if (u0 + j * WARP_SIZE < nu) {
                v[j] = __ldg(src + u0 + j * WARP_SIZE);
            }
        }
#pragma unroll
        for (int j = 0; j < UN; ++j) {
            if (u0 + j * WARP_SIZE < nu) {
                dsm[u0 + j * WARP_SIZE] = v[j];
            }
        }
    }
    __syncwarp();
    return (const char *) dsm;
}

// The block's rows for its NT members: xu and xg the up (or only) and gate matrices of its expert, smw the lane group's
// segment staging (MTOK_SEG), tiles the warp tiles (MTOK_TILE), act the members' staged activations, doff (lane j)
// member j's output offset.
template <bool gate, int st, int NT>
static __device__ __forceinline__ void mtok_rows(const mtok_args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                                 const char * xu, const char * xg, char * smw, char * tiles,
                                                 const char * act, const int doff, float * dst, const int64_t bias_off) {
    constexpr int NW = MMVQP_MTOK_NWARPS;
    constexpr int WB = MTOK_WB;
    constexpr int M  = gate ? 2 : 1;      // matrices: up (or plain), gate
    constexpr int NU = (WB + 15) / 16;    // MTOK_SEG: 16-byte units per lane and row, the last one predicated
    constexpr int BS = (int) sizeof(block_q2_0);

    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int wstep    = WB << lpr_log2;
    [[maybe_unused]] const int seg_units = wstep >> 4;
    GGML_UNUSED_VARS(xg, smw, tiles);

    int dc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        dc[j] = __shfl_sync(0xFFFFFFFF, doff, j);
    }

    for (int rbi = 0; rbi < p.rb; ++rbi) {
        const int rblk = (int) blockIdx.x * p.rb + rbi;
        if (rblk >= p.nrowblocks) {
            break;
        }
        const int row0 = ((rblk * NW + (int) threadIdx.y) << (5 - lpr_log2)) + grp;

        const char * rw[M];
        rw[0] = xu + (int64_t) min(row0, p.nrows - 1) * p.row_bytes + (st == MTOK_DIRECT ? sl * WB : 0);
        if constexpr (gate) {
            rw[1] = xg + (int64_t) min(row0, p.nrows - 1) * p.row_bytes + (st == MTOK_DIRECT ? sl * WB : 0);
        }
        [[maybe_unused]] const char * smt = nullptr; // MTOK_TILE: this lane's chunk of its row
        if constexpr (st == MTOK_TILE) {
            smt = mtok_stage_tile(xu, p, rblk, tiles) + grp * p.row_bytes + sl * WB;
        }

        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[j][m] = 0.0f;
            }
        }

        for (int it = 0; it < p.niter; ++it) {
            uint32_t w[M][WB / 4];
            if constexpr (st == MTOK_SEG) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    uint4       * dsm = (uint4 *) (smw + (m * WB << lpr_log2));
                    const uint4 * src = (const uint4 *) rw[m];
#pragma unroll
                    for (int j = 0; j < NU; ++j) {
                        const int u = sl + (j << lpr_log2);
                        if (j < NU - 1 || u < seg_units) {
                            dsm[u] = __ldg(src + u);
                        }
                    }
                }
                __syncwarp();
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    mmvqp_lds_words<WB, 4>(w[m], smw + (m * WB << lpr_log2) + sl * WB);
                }
                __syncwarp(); // the chunks are in registers: the next segments may overwrite them
            } else if constexpr (st == MTOK_TILE) {
                mmvqp_lds_words<WB, 4>(w[0], smt);
                smt += wstep;
            } else {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    mmvqp_ldg_words<WB, 4>(w[m], rw[m]);
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                rw[m] += wstep;
            }

            const int    c  = (it << lpr_log2) + sl;               // the lane's chunk of the row
            const char * au = act + c * (MTOK_US * 4);             // its int8 activations, member 0
            const char * ad = act + p.act_u + c * (MTOK_CNA * 4);  // its activation scales, member 0
#pragma unroll
            for (int b = 0; b < MTOK_C; ++b) {
                const int o = b * BS;
                float d[M];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    d[m] = mmvqp_h2f_lo(mmvqp_u16_at(w[m], o));
                }
                int i0[NT][M]; // the integer sums of the block's first 32 weights
#pragma unroll
                for (int sb = 0; sb < 2; ++sb) {
                    int q[M][8];
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        mtok_decode(mmvqp_word_at(w[m], o + 2 + 8*sb), q[m]);
                        mtok_decode(mmvqp_word_at(w[m], o + 6 + 8*sb), q[m] + 4);
                    }
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint4 * pu = (const uint4 *) (au + j * p.act_tok + (2*b + sb) * 32);
                        const uint4   u0 = pu[0];
                        const uint4   u1 = pu[1];
                        float2 a2 = make_float2(0.0f, 0.0f);
                        if (sb == 1) {
                            a2 = *(const float2 *) (ad + j * p.act_tok + 8*b);
                        }
#pragma unroll
                        for (int m = 0; m < M; ++m) {
                            int isum = ggml_cuda_dp4a(q[m][0], (int) u0.x, 0);
                            isum = ggml_cuda_dp4a(q[m][1], (int) u0.y, isum);
                            isum = ggml_cuda_dp4a(q[m][2], (int) u0.z, isum);
                            isum = ggml_cuda_dp4a(q[m][3], (int) u0.w, isum);
                            isum = ggml_cuda_dp4a(q[m][4], (int) u1.x, isum);
                            isum = ggml_cuda_dp4a(q[m][5], (int) u1.y, isum);
                            isum = ggml_cuda_dp4a(q[m][6], (int) u1.z, isum);
                            isum = ggml_cuda_dp4a(q[m][7], (int) u1.w, isum);
                            if (sb == 0) {
                                i0[j][m] = isum;
                            } else {
                                // mmvqp_dot<GGML_TYPE_Q2_0>: d * (i1 * d8[1] + i0 * d8[0])
                                acc[j][m] = fmaf(d[m], fmaf((float) isum, a2.y, (float) i0[j][m] * a2.x), acc[j][m]);
                            }
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                if (off < lpr) {
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        acc[j][m] += __shfl_xor_sync(0xFFFFFFFF, acc[j][m], off, WARP_SIZE);
                    }
                }
            }
        }

        const int row = row0;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (sl != 0 || row >= p.nrows) {
                break;
            }
            float result = acc[j][0];
            if (fusion.x_bias) {
                result += ((const float *) fusion.x_bias)[bias_off + row];
            }
            if constexpr (gate) {
                float gate_value = acc[j][M - 1];
                if (fusion.gate_bias) {
                    gate_value += ((const float *) fusion.gate_bias)[bias_off + row];
                }
                switch (fusion.glu_op) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, fusion.glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
            dst[dc[j] + row] = result;
        }
    }
}

// The member count nt (1..NTM) as a constant, so that no token's loads and products wait behind a test of the count.
template <bool gate, int st, int NT, int NTM>
static __device__ __forceinline__ void mtok_rows_nt(const int nt, const mtok_args & p,
                                                    const ggml_cuda_mm_fusion_args_device & fusion, const char * xu,
                                                    const char * xg, char * smw, char * tiles, const char * act,
                                                    const int doff, float * dst, const int64_t bias_off) {
    if constexpr (NT == NTM) {
        mtok_rows<gate, st, NT>(p, fusion, xu, xg, smw, tiles, act, doff, dst, bias_off);
    } else {
        if (nt == NT) {
            mtok_rows<gate, st, NT>(p, fusion, xu, xg, smw, tiles, act, doff, dst, bias_off);
        } else {
            mtok_rows_nt<gate, st, NT + 1, NTM>(nt, p, fusion, xu, xg, smw, tiles, act, doff, dst, bias_off);
        }
    }
}

// Grid: (row blocks / rb, tokens * slots + shared expert, samples). Dynamic shared memory: the warp tiles (MTOK_TILE),
// then the staged activations of up to NTM members, act_tok bytes each.
template <bool gate, int st, int NTM>
__launch_bounds__(MMVQP_MTOK_NWARPS*WARP_SIZE, MMVQP_MTOK_MIN_BLOCKS)
static __global__ void mul_mat_vec_q2_0_pascal_mtok(const mtok_args p, const ggml_cuda_mm_fusion_args_device fusion) {
    constexpr int NW = MMVQP_MTOK_NWARPS;
    constexpr int M  = gate ? 2 : 1;
    static_assert(!gate || st != MTOK_TILE, "tile staging: one matrix");
    __shared__ uint4 sm[st == MTOK_SEG ? NW * WARP_SIZE * M * MTOK_WB / 16 : 1];
    extern __shared__ uint4 mtok_dyn[];

    const int      lane          = threadIdx.x;
    const bool     shared_expert = gate && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t sample_dst    = blockIdx.z;
    const uint32_t sample_x      = fastdiv(sample_dst, p.sample_ratio);

    int      nt   = 0;
    int      yoff = 0;
    int      doff = 0;
    uint32_t channel_x;
    ggml_cuda_pdl_sync();
    if (!shared_expert) {
        if (!mtok_group(p, sample_dst, nt, yoff, doff, channel_x)) {
            return;
        }
    } else {
        nt = p.ncols;
        if (lane < nt) {
            yoff = (int) (sample_dst*p.sample_bytes_y + lane*p.col_bytes_y);
            doff = (int) (sample_dst*p.stride_sample_dst + lane*fusion.shared_stride_col_dst);
        }
        channel_x = 0;
    }

    // the members' activations, staged by the whole block
    char * tiles = (char *) mtok_dyn;
    char * act   = tiles + p.tile_bytes;
    const int tid = threadIdx.y * WARP_SIZE + lane;
    for (int j = 0; j < nt; ++j) {
        const char * yj = p.vy + __shfl_sync(0xFFFFFFFF, yoff, j);
        char       * aj = act + j * p.act_tok;
        for (int q = tid; q < p.nq; q += NW * WARP_SIZE) {
            const unsigned int * ys = (const unsigned int *) (yj + q * (int) sizeof(block_q8_1));
            uint32_t v[9];
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                v[k] = __ldg(ys + k);
            }
            uint4 * du = (uint4 *) (aj + ((q / MTOK_CNA) * MTOK_US + (q % MTOK_CNA) * 8) * 4);
            du[0] = make_uint4(__byte_perm(v[1], v[2], 0x6420), __byte_perm(v[1], v[2], 0x7531),
                               __byte_perm(v[3], v[4], 0x6420), __byte_perm(v[3], v[4], 0x7531));
            du[1] = make_uint4(__byte_perm(v[5], v[6], 0x6420), __byte_perm(v[5], v[6], 0x7531),
                               __byte_perm(v[7], v[8], 0x6420), __byte_perm(v[7], v[8], 0x7531));
            ((float *) (aj + p.act_u))[q] = mmvqp_h2f_lo(v[0]);
        }
    }
    __syncthreads();

    const int64_t xoff     = sample_x*p.sample_bytes_x + channel_x*p.channel_bytes_x;
    const char *  xu       = (shared_expert ? (const char *) fusion.shared_up : p.vx) + xoff;
    const char *  xg       = gate ? (const char *) (shared_expert ? fusion.shared_gate : fusion.gate) + xoff : nullptr;
    float       * dst      = shared_expert ? fusion.shared_dst : p.dst;
    const int64_t bias_off = sample_dst*p.stride_sample_dst + channel_x*p.stride_channel_dst;
    char        * smw      = (char *) sm + (st == MTOK_SEG ?
        threadIdx.y * (WARP_SIZE * M * MTOK_WB) + ((lane >> p.lpr_log2) * M * MTOK_WB << p.lpr_log2) : 0);
    mtok_rows_nt<gate, st, 1, NTM>(nt, p, fusion, xu, xg, smw, tiles, act, doff, dst, bias_off);
}

template <bool gate, int st>
static void mtok_launch(const int ntm, const mtok_args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                        const dim3 grid, const int smem, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQP_MTOK_NWARPS, 1);
    if (ntm <= 2) {
        mul_mat_vec_q2_0_pascal_mtok<gate, st, 2><<<grid, block, smem, stream>>>(p, fusion);
    } else if (ntm <= 4) {
        mul_mat_vec_q2_0_pascal_mtok<gate, st, 4><<<grid, block, smem, stream>>>(p, fusion);
    } else {
        mul_mat_vec_q2_0_pascal_mtok<gate, st, 8><<<grid, block, smem, stream>>>(p, fusion);
    }
}

// GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB,G,T (see mmvq-pascal-mtok.cuh)
struct mtok_cfg {
    bool on          = true;
    int  cap         = 0;
    int  rb          = 0;
    bool gate_direct = false;
    bool tile        = true;
    bool tile_c269   = true; // T = 2: false, the decline before Q825
};

static const mtok_cfg & mtok_config() {
    static const mtok_cfg cfg = [] {
        mtok_cfg c;
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_MTOK");
        int v[5] = { 1, 0, 0, 0, 1 };
        const int n = env ? sscanf(env, "%d,%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3], &v[4]) : 0;
        c.on          = n < 1 || v[0] != 0;
        c.cap         = n >= 2 && v[1] >= 1 && v[1] <= MMVQP_MTOK_MAX_COLS ? v[1] : 0;
        c.rb          = n >= 3 && v[2] >= 1 ? v[2] : 0;
        c.gate_direct = n >= 4 && v[3] == 1;
        c.tile        = n < 5 || v[4] != 0;
        c.tile_c269   = n < 5 || v[4] != 2;
        return c;
    }();
    return cfg;
}

// GGML_CUDA_MMVQ_PASCAL_SKIP lists q2_0: Q2_0 goes to llama.cpp's kernels at every token count
static bool mtok_skip_q2_0() {
    static const bool skip = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_SKIP");
        const std::string list = env ? env : "";
        size_t a = 0;
        while (a < list.size()) {
            size_t b = list.find(',', a);
            if (b == std::string::npos) {
                b = list.size();
            }
            if (list.compare(a, b - a, ggml_type_name(GGML_TYPE_Q2_0)) == 0) {
                return true;
            }
            a = b + 1;
        }
        return false;
    }();
    return skip;
}

static bool mtok_aligned(const void * ptr, const int64_t al) {
    return ptr == nullptr || (uintptr_t) ptr % al == 0;
}

// The types of mmvq-pascal-mtok-iq.cuh (Q713).
static bool mtok_iq_type(const ggml_type type) {
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS ||
        type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ1_M;
}

// The types of mmvq-pascal-mtok-np.cuh (Q928).
static bool mtok_np_type(const ggml_type type) {
    return type == GGML_TYPE_N2P || type == GGML_TYPE_N3P || type == GGML_TYPE_N2P_128 || type == GGML_TYPE_N3P_128;
}

// Whether the comma-separated ggml type names of environment variable var list type.
static bool mtok_env_lists(const char * var, const ggml_type type) {
    const char * env = getenv(var);
    const std::string list = env ? env : "";
    size_t a = 0;
    while (a < list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) {
            b = list.size();
        }
        if (list.compare(a, b - a, ggml_type_name(type)) == 0) {
            return true;
        }
        a = b + 1;
    }
    return false;
}

// An IQ or N type is left to the other kernels when GGML_CUDA_MMVQ_PASCAL_SKIP lists it (all the Pascal kernels) or
// GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP does (these only: 2 to 4 tokens then run mul_mat_vec_q_pascal_mc, 5 to 8 llama.cpp's
// kernels, as before Q713). Read once per type.
static bool mtok_iq_skipped(const ggml_type type) {
    static int8_t skip[GGML_TYPE_COUNT];
    static bool   init = [] {
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            skip[t] = (mtok_iq_type((ggml_type) t) || mtok_np_type((ggml_type) t)) &&
                (mtok_env_lists("GGML_CUDA_MMVQ_PASCAL_SKIP", (ggml_type) t) ||
                mtok_env_lists("GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP", (ggml_type) t));
        }
        return true;
    }();
    GGML_UNUSED(init);
    return skip[type] != 0;
}

// GGML_CUDA_MTOK_IQ_FAST (C1138): 0 (default) the kernels of mmvq-pascal-mtok-iq.cuh for every type; 1 IQ2_XXS's fast
// kernel in CUDA C (mmvq-pascal-mtok-iq2_xxs-fast.cu) where its shapes allow (rows of 2 lanes: 256 n weights with
// n = 2 mod 4; 16-byte-aligned matrices whose rows are a multiple of 4 bytes); 2 the hand-written PTX of the same kernel
// where it covers the product, else as 1. Read once.
static int mtok_iq_fast_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_MTOK_IQ_FAST");
        const int v = env ? atoi(env) : 0;
        return v >= 0 && v <= 2 ? v : 0;
    }();
    return mode;
}

// The grid types and IQ1_M (mmvq-pascal-mtok-iq.cuh): the arguments as for Q2_0.
static bool mtok_iq_run(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream) {
    if (ids == nullptr || ncols_dst < 2 || ncols_dst > MMVQP_MTOK_MAX_COLS || !mtok_config().on || mtok_iq_skipped(type) ||
            !ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (fusion.x_scale || fusion.gate_scale || (fusion.shared_up && !fusion.gate)) {
        return false;
    }
    const int64_t ne = ncols_dst * nchannels_dst;
    if (ne > MMVQP_MTOK_MAX_ENTRIES || ncols_x % QK_K != 0) {
        return false;
    }
    const bool    gate   = fusion.gate != nullptr;
    const int64_t bs     = ggml_type_size(type);
    const int64_t blocks = ncols_x / QK_K; // one super-block per lane and iteration
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && blocks % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }

    // the one-token kernel's staging: the widest loads the rows and the segments allow
    const int64_t row_bytes = stride_row_x * bs;
    const auto x_aligned = [&](const int64_t al) {
        return mtok_aligned(vx, al) && mtok_aligned(fusion.gate, al) && mtok_aligned(fusion.shared_up, al) &&
            mtok_aligned(fusion.shared_gate, al) && row_bytes % al == 0 && (stride_channel_x * bs) % al == 0 &&
            (stride_sample_x * bs) % al == 0;
    };
    int su = 0;
    for (const int u : { 16, 8, 4 }) {
        if ((bs << lpr_log2) % u == 0 && x_aligned(u)) {
            su = u;
            break;
        }
    }
    if (su == 0 || row_bytes > INT_MAX || !mtok_aligned(vy, 4)) {
        return false;
    }

    // the kernel keeps the offsets of each member's activations and outputs in an int
    const int64_t y_extent = (nsamples_dst - 1) * stride_sample_y + (std::max(nchannels_y, nchannels_dst) - 1) *
        stride_channel_y + (ncols_dst - 1) * stride_col_y + blocks * (QK_K / QK8_1);
    const int64_t d_extent = (nsamples_dst - 1) * stride_sample_dst + (nchannels_dst - 1) * stride_channel_dst +
        (ncols_dst - 1) * std::max(stride_col_dst, (int64_t) fusion.shared_stride_col_dst) + nrows_x;
    if (y_extent * (int64_t) sizeof(block_q8_1) > INT_MAX || d_extent > INT_MAX) {
        return false;
    }

    const int act_tok = (int) (blocks * (MMVQP_MTOK_IQ_ACT_RUN + 8 * 4));
    const int gcap    = mtok_config().cap > 0 ? mtok_config().cap : (int) ncols_dst;
    const int ntm     = std::max(gcap, fusion.shared_up ? (int) ncols_dst : 1);
    const int smem    = ntm * act_tok;
    int smem_static = 0;
    switch (type) {
        case GGML_TYPE_IQ2_XXS: smem_static = mtok_iq::static_smem<GGML_TYPE_IQ2_XXS>(); break;
        case GGML_TYPE_IQ2_XS:  smem_static = mtok_iq::static_smem<GGML_TYPE_IQ2_XS>();  break;
        case GGML_TYPE_IQ2_S:   smem_static = mtok_iq::static_smem<GGML_TYPE_IQ2_S>();   break;
        case GGML_TYPE_IQ3_XXS: smem_static = mtok_iq::static_smem<GGML_TYPE_IQ3_XXS>(); break;
        case GGML_TYPE_IQ3_S:   smem_static = mtok_iq::static_smem<GGML_TYPE_IQ3_S>();   break;
        default:                smem_static = mtok_iq::static_smem<GGML_TYPE_IQ1_M>();   break;
    }
    if (smem + smem_static > 48 * 1024) {
        return false;
    }

    const int64_t rows_per_block = MMVQP_MTOK_IQ_NWARPS * (WARP_SIZE >> lpr_log2);
    const int     nrowblocks     = (int) ((nrows_x + rows_per_block - 1) / rows_per_block);
    const int     rb             = std::min(nrowblocks, mtok_config().rb > 0 ? mtok_config().rb : (nrowblocks >= 8 ? 4 : 1));

    mtok_iq::args p;
    p.vx                 = (const char *) vx;
    p.vy                 = (const char *) vy;
    p.ids                = ids;
    p.dst                = dst;
    p.nrows              = (int) nrows_x;
    p.row_bytes          = (int) row_bytes;
    p.lpr_log2           = lpr_log2;
    p.niter              = (int) (blocks >> lpr_log2);
    p.nrowblocks         = nrowblocks;
    p.rb                 = rb;
    p.gcap               = gcap;
    p.ncols              = (int) ncols_dst;
    p.nslots             = (int) nchannels_dst;
    p.ids_stride         = (int) ids_stride;
    p.nchunk             = (int) blocks;
    p.act_tok            = act_tok;
    p.nchannels_y        = init_fastdiv_values(nchannels_y);
    p.sample_ratio       = init_fastdiv_values(nsamples_dst / nsamples_x);
    p.channel_bytes_x    = stride_channel_x * bs;
    p.sample_bytes_x     = stride_sample_x * bs;
    p.channel_bytes_y    = stride_channel_y * (int64_t) sizeof(block_q8_1);
    p.sample_bytes_y     = stride_sample_y * (int64_t) sizeof(block_q8_1);
    p.col_bytes_y        = stride_col_y * (int64_t) sizeof(block_q8_1);
    p.stride_channel_dst = stride_channel_dst;
    p.stride_sample_dst  = stride_sample_dst;
    p.stride_col_dst     = stride_col_dst;

    const dim3 grid((unsigned) ((nrowblocks + rb - 1) / rb), (unsigned) (ne + (fusion.shared_up ? 1 : 0)),
                    (unsigned) nsamples_dst);

    // IQ2_XXS's fast kernel (C1138): rows of 2 lanes, every matrix 16-byte aligned and ending at a 16-byte boundary (its
    // lanes load the aligned 16-byte units around their super-blocks), rows a multiple of 4 bytes (a warp's rows, 4
    // apart, then start at the same word of a unit)
    if (type == GGML_TYPE_IQ2_XXS && mtok_iq_fast_mode() > 0 && lpr_log2 == 1 && row_bytes % 4 == 0 &&
            (nrows_x * row_bytes) % 16 == 0 && mtok_aligned(vx, 16) && mtok_aligned(fusion.gate, 16) &&
            mtok_aligned(fusion.shared_up, 16) && mtok_aligned(fusion.shared_gate, 16) &&
            (stride_channel_x * bs) % 16 == 0 && (stride_sample_x * bs) % 16 == 0 &&
            smem + ggml_cuda_mtok_iq2_xxs_fast_static_smem() <= 48 * 1024) {
        // GGML_CUDA_MTOK_IQ_FAST_FAULT=1, a test fault: the fast kernels take the up and gate matrices the wrong way
        // round, so that a check comparing them with the old kernel must see the gate/up products differ
        static const bool fault = [] {
            const char * env = getenv("GGML_CUDA_MTOK_IQ_FAST_FAULT");
            return env != nullptr && atoi(env) == 1;
        }();
        ggml_cuda_mm_fusion_args_device ff = fusion;
        mtok_iq::args pf = p;
        if (fault && gate) {
            pf.vx   = (const char *) fusion.gate;
            ff.gate = vx;
            std::swap(ff.shared_up, ff.shared_gate);
        }
        if (mtok_iq_fast_mode() == 2 && ggml_cuda_mtok_iq2_xxs_ptx_launch(pf, ff, grid, smem, gate, ntm, stream)) {
            return true;
        }
        ggml_cuda_mtok_iq_fast_launch_iq2_xxs(pf, ff, grid, smem, gate, ntm, stream);
        return true;
    }

    switch (type) {
        case GGML_TYPE_IQ2_XXS: ggml_cuda_mtok_iq_launch_iq2_xxs(p, fusion, grid, smem, gate, su, ntm, stream); break;
        case GGML_TYPE_IQ2_XS:  ggml_cuda_mtok_iq_launch_iq2_xs(p, fusion, grid, smem, gate, su, ntm, stream);  break;
        case GGML_TYPE_IQ2_S:   ggml_cuda_mtok_iq_launch_iq2_s(p, fusion, grid, smem, gate, su, ntm, stream);   break;
        case GGML_TYPE_IQ3_XXS: ggml_cuda_mtok_iq_launch_iq3_xxs(p, fusion, grid, smem, gate, su, ntm, stream); break;
        case GGML_TYPE_IQ3_S:   ggml_cuda_mtok_iq_launch_iq3_s(p, fusion, grid, smem, gate, su, ntm, stream);   break;
        default:                ggml_cuda_mtok_iq_launch_iq1_m(p, fusion, grid, smem, gate, su, ntm, stream);   break;
    }
    return true;
}

// N2P, N3P, N2P-128 and N3P-128 (mmvq-pascal-mtok-np.cuh): the arguments as for Q2_0.
static bool mtok_np_run(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream) {
    if (ids == nullptr || ncols_dst < 2 || ncols_dst > MMVQP_MTOK_MAX_COLS || !mtok_config().on || mtok_iq_skipped(type) ||
            !ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (fusion.x_scale || fusion.gate_scale || (fusion.shared_up && !fusion.gate)) {
        return false;
    }
    const int64_t ne = ncols_dst * nchannels_dst;
    const int64_t qk = ggml_blck_size(type);
    if (ne > MMVQP_MTOK_MAX_ENTRIES || ncols_x % qk != 0) {
        return false;
    }
    const bool    gate   = fusion.gate != nullptr;
    const int64_t bs     = ggml_type_size(type);
    const int64_t blocks = ncols_x / qk; // one block per lane and iteration
    const int     nb     = (int) (qk / QK8_1);
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && blocks % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }

    // the one-token kernel's staging: the widest loads the rows and the segments allow, else (no gate) warp tiles
    const int64_t row_bytes = stride_row_x * bs;
    const auto x_aligned = [&](const int64_t al) {
        return mtok_aligned(vx, al) && mtok_aligned(fusion.gate, al) && mtok_aligned(fusion.shared_up, al) &&
            mtok_aligned(fusion.shared_gate, al) && row_bytes % al == 0 && (stride_channel_x * bs) % al == 0 &&
            (stride_sample_x * bs) % al == 0;
    };
    int su = 0;
    for (const int u : { 16, 8, 4 }) {
        if ((bs << lpr_log2) % u == 0 && x_aligned(u)) {
            su = u;
            break;
        }
    }
    const int     act_tok = (int) (blocks * (mtok_np::act_run(nb) + nb * 4));
    const int     gcap    = mtok_config().cap > 0 ? mtok_config().cap : (int) ncols_dst;
    const int     ntm     = std::max(gcap, fusion.shared_up ? (int) ncols_dst : 1);
    const int64_t wbytes  = (int64_t) (WARP_SIZE >> lpr_log2) * row_bytes;
    const bool    tile_ok = !gate && mtok_config().tile && stride_row_x == blocks && wbytes % 16 == 0 &&
        (nrows_x * row_bytes) % 16 == 0 && mtok_aligned(vx, 16) && (stride_channel_x * bs) % 16 == 0 &&
        (stride_sample_x * bs) % 16 == 0 && MMVQP_MTOK_IQ_NWARPS * wbytes + ntm * act_tok <= 48 * 1024;
    // as for Q2_0: segments where they are whole 16-byte units, else (no gate) warp tiles, else narrower segment loads
    if (su != 16 && tile_ok) {
        su = 0;
    }
    if ((su == 0 && !tile_ok) || row_bytes > INT_MAX || !mtok_aligned(vy, 4)) {
        return false;
    }

    // the kernel keeps the offsets of each member's activations and outputs in an int
    const int64_t y_extent = (nsamples_dst - 1) * stride_sample_y + (std::max(nchannels_y, nchannels_dst) - 1) *
        stride_channel_y + (ncols_dst - 1) * stride_col_y + blocks * nb;
    const int64_t d_extent = (nsamples_dst - 1) * stride_sample_dst + (nchannels_dst - 1) * stride_channel_dst +
        (ncols_dst - 1) * std::max(stride_col_dst, (int64_t) fusion.shared_stride_col_dst) + nrows_x;
    if (y_extent * (int64_t) sizeof(block_q8_1) > INT_MAX || d_extent > INT_MAX) {
        return false;
    }

    const int tile_bytes = su == 0 ? (int) (MMVQP_MTOK_IQ_NWARPS * wbytes) : 0;
    const int smem       = tile_bytes + ntm * act_tok;
    int smem_static = 0;
    if (su != 0) {
        switch (type) {
            case GGML_TYPE_N2P:     smem_static = mtok_np::static_smem<GGML_TYPE_N2P>();     break;
            case GGML_TYPE_N3P:     smem_static = mtok_np::static_smem<GGML_TYPE_N3P>();     break;
            case GGML_TYPE_N2P_128: smem_static = mtok_np::static_smem<GGML_TYPE_N2P_128>(); break;
            default:                smem_static = mtok_np::static_smem<GGML_TYPE_N3P_128>(); break;
        }
    }
    if (smem + smem_static > 48 * 1024) {
        return false;
    }

    const int64_t rows_per_block = MMVQP_MTOK_IQ_NWARPS * (WARP_SIZE >> lpr_log2);
    const int     nrowblocks     = (int) ((nrows_x + rows_per_block - 1) / rows_per_block);
    const int     rb             = std::min(nrowblocks, mtok_config().rb > 0 ? mtok_config().rb : (nrowblocks >= 8 ? 4 : 1));

    mtok_np::args p;
    p.vx                 = (const char *) vx;
    p.vy                 = (const char *) vy;
    p.ids                = ids;
    p.dst                = dst;
    p.nrows              = (int) nrows_x;
    p.row_bytes          = (int) row_bytes;
    p.lpr_log2           = lpr_log2;
    p.niter              = (int) (blocks >> lpr_log2);
    p.nrowblocks         = nrowblocks;
    p.rb                 = rb;
    p.gcap               = gcap;
    p.ncols              = (int) ncols_dst;
    p.nslots             = (int) nchannels_dst;
    p.ids_stride         = (int) ids_stride;
    p.nchunk             = (int) blocks;
    p.act_tok            = act_tok;
    p.nchannels_y        = init_fastdiv_values(nchannels_y);
    p.sample_ratio       = init_fastdiv_values(nsamples_dst / nsamples_x);
    p.channel_bytes_x    = stride_channel_x * bs;
    p.sample_bytes_x     = stride_sample_x * bs;
    p.channel_bytes_y    = stride_channel_y * (int64_t) sizeof(block_q8_1);
    p.sample_bytes_y     = stride_sample_y * (int64_t) sizeof(block_q8_1);
    p.col_bytes_y        = stride_col_y * (int64_t) sizeof(block_q8_1);
    p.stride_channel_dst = stride_channel_dst;
    p.stride_sample_dst  = stride_sample_dst;
    p.stride_col_dst     = stride_col_dst;
    p.tile_bytes         = tile_bytes;

    const dim3 grid((unsigned) ((nrowblocks + rb - 1) / rb), (unsigned) (ne + (fusion.shared_up ? 1 : 0)),
                    (unsigned) nsamples_dst);
    switch (type) {
        case GGML_TYPE_N2P:     ggml_cuda_mtok_np_launch_n2p(p, fusion, grid, smem, gate, su, ntm, stream);     break;
        case GGML_TYPE_N3P:     ggml_cuda_mtok_np_launch_n3p(p, fusion, grid, smem, gate, su, ntm, stream);     break;
        case GGML_TYPE_N2P_128: ggml_cuda_mtok_np_launch_n2p_128(p, fusion, grid, smem, gate, su, ntm, stream); break;
        default:                ggml_cuda_mtok_np_launch_n3p_128(p, fusion, grid, smem, gate, su, ntm, stream); break;
    }
    return true;
}

} // namespace

bool ggml_cuda_mmvq_pascal_mtok(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream) {
    GGML_UNUSED(nchannels_x);
    if (mtok_iq_type(type)) {
        return mtok_iq_run(vx, type, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,
                           stride_col_dst, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                           nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
    }
    if (mtok_np_type(type)) {
        return mtok_np_run(vx, type, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,
                           stride_col_dst, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                           nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
    }
    if (type != GGML_TYPE_Q2_0 || ids == nullptr || ncols_dst < 2 || ncols_dst > MMVQP_MTOK_MAX_COLS ||
            !mtok_config().on || mtok_skip_q2_0() ||
            !ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    // the routed + shared expert launch exists only with a fused gate (as in mul_mat_vec_q_pascal_mc)
    if (fusion.x_scale || fusion.gate_scale || (fusion.shared_up && !fusion.gate)) {
        return false;
    }
    const int64_t ne = ncols_dst * nchannels_dst;
    if (ne > MMVQP_MTOK_MAX_ENTRIES) {
        return false;
    }

    const bool    gate   = fusion.gate != nullptr;
    const int64_t bs     = sizeof(block_q2_0);
    const int64_t blocks = ncols_x / QK2_0;
    if (ncols_x % QK2_0 != 0 || blocks % MTOK_C != 0) {
        return false;
    }
    const int64_t chunks = blocks / MTOK_C;
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && chunks % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }

    // the staging of the one-token kernel: segments where they are whole 16-byte units, else (no gate) warp tiles of
    // contiguous rows, else direct loads
    const int64_t row_bytes = stride_row_x * bs;
    const auto x_aligned = [&](const int64_t al) {
        return mtok_aligned(vx, al) && mtok_aligned(fusion.gate, al) && mtok_aligned(fusion.shared_up, al) &&
            mtok_aligned(fusion.shared_gate, al) && row_bytes % al == 0 && (stride_channel_x * bs) % al == 0 &&
            (stride_sample_x * bs) % al == 0;
    };
    if (row_bytes > INT_MAX || !x_aligned(4)) {
        return false;
    }
    const bool    seg_ok  = (MTOK_WB << lpr_log2) % 16 == 0 && x_aligned(16);
    // warp tiles (as mmvqp_stage_tile checks them in mmvq-pascal.cu): each warp's rows start at a 16-byte boundary and
    // every matrix ends at one, whatever the row width (Q825: K = 640's 180-byte rows, the routed down projection)
    const int64_t wbytes  = (int64_t) (WARP_SIZE >> lpr_log2) * row_bytes;
    const bool    tile_ok = !gate && mtok_config().tile && stride_row_x == blocks && wbytes % 16 == 0 &&
        (nrows_x * row_bytes) % 16 == 0 && mtok_aligned(vx, 16) && (stride_channel_x * bs) % 16 == 0 &&
        (stride_sample_x * bs) % 16 == 0 && MMVQP_MTOK_NWARPS * wbytes <= 48 * 1024 &&
        (mtok_config().tile_c269 || x_aligned(16));
    int st;
    if (gate) {
        st = seg_ok && !mtok_config().gate_direct ? MTOK_SEG : MTOK_DIRECT;
    } else if (seg_ok) {
        st = MTOK_SEG;
    } else if (tile_ok) {
        st = MTOK_TILE;
    } else {
        if (!mtok_config().tile_c269) {
            return false;
        }
        st = MTOK_DIRECT;
    }

    // the kernel keeps the offsets of each member's activations and outputs in an int
    const int64_t y_extent = (nsamples_dst - 1) * stride_sample_y + (std::max(nchannels_y, nchannels_dst) - 1) *
        stride_channel_y + (ncols_dst - 1) * stride_col_y + blocks * (QK2_0 / QK8_1);
    const int64_t d_extent = (nsamples_dst - 1) * stride_sample_dst + (nchannels_dst - 1) * stride_channel_dst +
        (ncols_dst - 1) * std::max(stride_col_dst, (int64_t) fusion.shared_stride_col_dst) + nrows_x;
    if (y_extent * (int64_t) sizeof(block_q8_1) > INT_MAX || d_extent > INT_MAX) {
        return false;
    }

    const int nq         = (int) (blocks * (QK2_0 / QK8_1));
    const int act_u      = (nq / MTOK_CNA) * MTOK_US * 4;
    const int act_tok    = act_u + nq * 4;
    const int gcap       = mtok_config().cap > 0 ? mtok_config().cap : (int) ncols_dst;
    const int ntm        = std::max(gcap, fusion.shared_up ? (int) ncols_dst : 1);
    const int tile_bytes = st == MTOK_TILE ? (int) (MMVQP_MTOK_NWARPS * wbytes) : 0;
    const int smem       = tile_bytes + ntm * act_tok;
    const int smem_seg   = st == MTOK_SEG ? MMVQP_MTOK_NWARPS * WARP_SIZE * (gate ? 2 : 1) * MTOK_WB : 0;
    if (smem + smem_seg > 48 * 1024) {
        return false;
    }

    const int64_t rows_per_block = MMVQP_MTOK_NWARPS * (WARP_SIZE >> lpr_log2);
    const int     nrowblocks     = (int) ((nrows_x + rows_per_block - 1) / rows_per_block);
    const int     rb             = std::min(nrowblocks, mtok_config().rb > 0 ? mtok_config().rb : (nrowblocks >= 8 ? 4 : 1));

    mtok_args p;
    p.vx                 = (const char *) vx;
    p.vy                 = (const char *) vy;
    p.ids                = ids;
    p.dst                = dst;
    p.nrows              = (int) nrows_x;
    p.row_bytes          = (int) row_bytes;
    p.lpr_log2           = lpr_log2;
    p.niter              = (int) (chunks >> lpr_log2);
    p.nrowblocks         = nrowblocks;
    p.rb                 = rb;
    p.gcap               = gcap;
    p.ncols              = (int) ncols_dst;
    p.nslots             = (int) nchannels_dst;
    p.ids_stride         = (int) ids_stride;
    p.nq                 = nq;
    p.act_u              = act_u;
    p.act_tok            = act_tok;
    p.tile_bytes         = tile_bytes;
    p.nchannels_y        = init_fastdiv_values(nchannels_y);
    p.sample_ratio       = init_fastdiv_values(nsamples_dst / nsamples_x);
    p.channel_bytes_x    = stride_channel_x * bs;
    p.sample_bytes_x     = stride_sample_x * bs;
    p.channel_bytes_y    = stride_channel_y * (int64_t) sizeof(block_q8_1);
    p.sample_bytes_y     = stride_sample_y * (int64_t) sizeof(block_q8_1);
    p.col_bytes_y        = stride_col_y * (int64_t) sizeof(block_q8_1);
    p.stride_channel_dst = stride_channel_dst;
    p.stride_sample_dst  = stride_sample_dst;
    p.stride_col_dst     = stride_col_dst;

    const dim3 grid((unsigned) ((nrowblocks + rb - 1) / rb), (unsigned) (ne + (fusion.shared_up ? 1 : 0)),
                    (unsigned) nsamples_dst);
    if (gate) {
        if (st == MTOK_SEG) {
            mtok_launch<true, MTOK_SEG>(ntm, p, fusion, grid, smem, stream);
        } else {
            mtok_launch<true, MTOK_DIRECT>(ntm, p, fusion, grid, smem, stream);
        }
    } else if (st == MTOK_SEG) {
        mtok_launch<false, MTOK_SEG>(ntm, p, fusion, grid, smem, stream);
    } else if (st == MTOK_TILE) {
        mtok_launch<false, MTOK_TILE>(ntm, p, fusion, grid, smem, stream);
    } else {
        mtok_launch<false, MTOK_DIRECT>(ntm, p, fusion, grid, smem, stream);
    }
    return true;
}

bool ggml_cuda_mmvq_pascal_mtok_takes(const ggml_type type, const int cc) {
    return (mtok_iq_type(type) || mtok_np_type(type)) && mtok_config().on && !mtok_iq_skipped(type) &&
        ggml_cuda_mmvq_pascal_active(cc);
}

#else

bool ggml_cuda_mmvq_pascal_mtok_takes(const ggml_type type, const int cc) {
    GGML_UNUSED_VARS(type, cc);
    return false;
}

bool ggml_cuda_mmvq_pascal_mtok(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, type, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                     nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
