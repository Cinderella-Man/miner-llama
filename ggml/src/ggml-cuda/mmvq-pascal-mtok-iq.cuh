#pragma once

// The multi-token expert kernels of mmvq-pascal-mtok.cu for the grid types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S)
// and IQ1_M (Q713): MUL_MAT_ID of 2 to 8 tokens, each distinct expert of the step read once for all the tokens that
// picked it, as the Q2_0 kernel there does (R7). The routed gate/up projections of Swift-1.5's IQ files.
//   - Groups, the block's members and the routed + shared expert launch are those of the Q2_0 kernel (mtok_group):
//     every gcap-th entry of an expert, in entry order, leads a group of up to gcap of its entries.
//   - The block stages its members' activations in shared memory once, for all its row blocks: per member and
//     super-block of the row (256 weights) the 256 int8 values (IQ1_M: in its lookup order) and 16 bytes of padding,
//     so that the lane groups' lanes, which read neighbouring super-blocks, hit different banks; the FP32 scales of its
//     8 Q8_1 blocks beside them. It also copies the type's grid into shared memory.
//   - Lane groups and staging are the one-token kernel's (mmvq-pascal.cu, half-word types): one super-block per lane
//     and iteration, lanes per row from the row width, a lane group's row segment staged through shared memory with
//     the widest loads the rows allow (16, 8 or 4 bytes), the gate and up matrices one after the other. One row per
//     lane group.
//   - Each 32-weight block is decoded once (mmvqp_iq::decode32: 8 words of signed int8 weights and the scales of its
//     two halves) and multiplied by every member, 8 dp4a per member.
// Each token's sum is formed as in the one-token kernel: the same integer sums per 32 weights, the same FP32 operations
// in the same order, the same reduction over a row's lanes, so a grid type's result is bit-identical to
// mul_mat_vec_q_pascal's (IQ1_M has no one-token Pascal kernel).

#include "common.cuh"
#include "mmvq-pascal-impl.cuh"
#include "unary.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#define MMVQP_MTOK_IQ_NWARPS 4
#define MMVQP_MTOK_IQ_ACT_RUN 272 // bytes of a member's staged int8 values per super-block: 256 and 16 of padding
#ifndef MMVQP_MTOK_IQ_MIN_BLOCKS
#define MMVQP_MTOK_IQ_MIN_BLOCKS 3 // resident blocks per SM the registers must allow (2 for groups of up to 8 members, whose
                                   // activations leave room for 2 blocks per SM: 255 registers, no spills)
#endif

namespace mtok_iq {

// Byte strides for x and the activations, float strides for dst.
struct args {
    const char    * vx;
    const char    * vy;
    const int32_t * ids;
    float         * dst;
    int             nrows;
    int             row_bytes;
    int             lpr_log2;   // lanes per row = 1 << lpr_log2
    int             niter;      // iterations (super-blocks) per lane
    int             nrowblocks;
    int             rb;         // row blocks per block
    int             gcap;       // entries per group
    int             ncols;      // tokens
    int             nslots;     // ids: entries per token (experts used)
    int             ids_stride;
    int             nchunk;     // super-blocks per row
    int             act_tok;    // bytes per member: nchunk runs of MMVQP_MTOK_IQ_ACT_RUN bytes, then 8 * nchunk scales
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

template <ggml_type type> constexpr __host__ __device__ int gtype() {
    return type == GGML_TYPE_IQ2_XXS ? mmvqp_iq::G_IQ2_XXS : type == GGML_TYPE_IQ2_XS ? mmvqp_iq::G_IQ2_XS :
        type == GGML_TYPE_IQ2_S ? mmvqp_iq::G_IQ2_S : type == GGML_TYPE_IQ3_XXS ? mmvqp_iq::G_IQ3_XXS :
        type == GGML_TYPE_IQ3_S ? mmvqp_iq::G_IQ3_S : mmvqp_iq::G_IQ1_M;
}

// ggml's device table of the type's grid (IQ1_M: iq1s_grid_gpu, its values plus one as nibbles)
template <ggml_type type>
static __device__ __forceinline__ const uint32_t * grid_table() {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return (const uint32_t *) iq2xxs_grid;
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        return (const uint32_t *) iq2xs_grid;
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return (const uint32_t *) iq2s_grid;
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return iq3xxs_grid;
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        return iq3s_grid;
    } else {
        return iq1s_grid_gpu;
    }
}

// One lane's super-block of NB bytes (NB % 4 == 2) at a 2-byte boundary in shared memory as NB / 4 + 1 words, shifted by
// 2 bytes where it starts at an odd half-word (mmvqp_lds_h of mmvq-pascal.cu).
template <int NB>
static __device__ __forceinline__ void lds_h(uint32_t * w, const char * p) {
    static_assert(NB % 4 == 2, "a block of an odd number of half-words");
    const int        mis = (int) ((uintptr_t) p & 2);
    const uint32_t * aw  = (const uint32_t *) (p - mis); // pointer arithmetic keeps the shared address space
    const uint32_t   sel = mis ? 0x5432 : 0x3210;
    uint32_t cur = aw[0];
#pragma unroll
    for (int i = 0; i < NB/4; ++i) {
        const uint32_t nxt = aw[i + 1];
        w[i] = __byte_perm(cur, nxt, sel);
        cur = nxt;
    }
    w[NB/4] = __byte_perm(cur, 0, sel);
}

template <int SU> struct stage_unit;
template <> struct stage_unit<16> { using t = uint4; };
template <> struct stage_unit<8>  { using t = uint2; };
template <> struct stage_unit<4>  { using t = unsigned int; };

// The group of block y (routed experts), as mtok_group of mmvq-pascal-mtok.cu: the entries of entry y's expert from
// entry y on, at most gcap of them, if y starts one. Lane j < nt returns member j's activation offset (bytes from vy)
// and output offset (floats from dst), which the host checked to fit in an int.
static __device__ __forceinline__ bool group(const args & p, const uint32_t sample_dst, int & nt, int & yoff, int & doff,
                                             uint32_t & expert) {
    const int lane = threadIdx.x;
    const int ne   = p.ncols * p.nslots;
    const int y    = blockIdx.y;
    const int ty   = y / p.nslots;
    const int ey   = p.ids[(y - ty*p.nslots) + ty*p.ids_stride];
    if (ey < 0) {
        // a skipped slot (id -1, moe-tail.cuh): zeros in its rows, as mtok_group writes them
        float * dst = p.dst + sample_dst*p.stride_sample_dst + (y - ty*p.nslots)*p.stride_channel_dst + ty*p.stride_col_dst;
        const int sl  = lane & ((1 << p.lpr_log2) - 1);
        const int grp = lane >> p.lpr_log2;
        for (int rbi = 0; rbi < p.rb; ++rbi) {
            const int rblk = (int) blockIdx.x * p.rb + rbi;
            if (rblk >= p.nrowblocks) {
                break;
            }
            const int row0 = ((rblk * MMVQP_MTOK_IQ_NWARPS + (int) threadIdx.y) << (5 - p.lpr_log2)) + grp;
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

// The block's rows for its NT members: xu and xg the up (or only) and gate matrices of its expert, smw the warp's
// staging buffer, tab the grid, act the members' staged activations, doff (lane j) member j's output offset.
template <ggml_type type, bool gate, int SU, int NT>
static __device__ __forceinline__ void rows(const args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                            const char * xu, const char * xg, char * smw, const uint32_t * tab,
                                            const char * act, const int doff, float * dst, const int64_t bias_off) {
    using unit_t = typename stage_unit<SU>::t;
    constexpr int G   = gtype<type>();
    constexpr int BS  = mmvqp_iq::gtraits<G>::BS;
    constexpr int NWD = mmvqp_iq::gtraits<G>::NW;
    constexpr int NW  = MMVQP_MTOK_IQ_NWARPS;
    constexpr int M   = gate ? 2 : 1;           // matrices: up (or plain), gate
    constexpr int NU  = (BS + SU - 1) / SU;      // staging units per lane and row, the last one predicated
    static_assert(BS % 4 == 2 || BS % 8 == 0, "2-byte- or 8-byte-aligned blocks");

    const int lane     = threadIdx.x;
    const int lpr_log2 = p.lpr_log2;
    const int lpr      = 1 << lpr_log2;
    const int sl       = lane & (lpr - 1);
    const int grp      = lane >> lpr_log2;
    const int wstep    = BS << lpr_log2;        // a row segment's bytes per iteration
    const int su_units = wstep / SU;
    GGML_UNUSED(xg);

    int dc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        dc[j] = __shfl_sync(0xFFFFFFFF, doff, j);
    }
    const float * ad0 = (const float *) (act + p.nchunk * MMVQP_MTOK_IQ_ACT_RUN); // member 0's scales
    const int     ast = p.act_tok / 4;                                           // floats between members

    for (int rbi = 0; rbi < p.rb; ++rbi) {
        const int rblk = (int) blockIdx.x * p.rb + rbi;
        if (rblk >= p.nrowblocks) {
            break;
        }
        const int row0 = ((rblk * NW + (int) threadIdx.y) << (5 - lpr_log2)) + grp;

        const char * rw[M];
        rw[0] = xu + (int64_t) min(row0, p.nrows - 1) * p.row_bytes;
        if constexpr (gate) {
            rw[1] = xg + (int64_t) min(row0, p.nrows - 1) * p.row_bytes;
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
            const int     c  = (it << lpr_log2) + sl;          // the lane's super-block of the row
            const char  * au = act + c * MMVQP_MTOK_IQ_ACT_RUN; // its int8 activations, member 0
            const float * ad = ad0 + 8 * c;                     // its activation scales, member 0
#pragma unroll
            for (int m = 0; m < M; ++m) {
                // the lane group's segment of matrix m, then the lane's super-block from it
                unit_t       * dsm = (unit_t *) (smw + grp * wstep);
                const unit_t * src = (const unit_t *) rw[m];
#pragma unroll
                for (int j = 0; j < NU; ++j) {
                    const int u = sl + (j << lpr_log2);
                    if (j < NU - 1 || u < su_units) {
                        dsm[u] = __ldg(src + u);
                    }
                }
                rw[m] += wstep;
                __syncwarp();
                uint32_t w[NWD];
                if constexpr (BS % 4 == 2) {
                    lds_h<BS>(w, smw + lane * BS);
                } else {
                    mmvqp_lds_words<BS, 8>(w, smw + lane * BS);
                }
                __syncwarp(); // the super-block is in registers: the next segment may overwrite it

                const float dd = mmvqp_iq::sblock_scale<G>(w);
                float sumf[NT];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    sumf[j] = 0.0f;
                }
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    int q[8];
                    int ls0, ls1;
                    mmvqp_iq::decode32<G>(w, ib, tab, q, ls0, ls1);
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint4 * pu = (const uint4 *) (au + j * p.act_tok + 32 * ib);
                        const uint4   u0 = pu[0];
                        const uint4   u1 = pu[1];
                        int i0 = ggml_cuda_dp4a(q[0], (int) u0.x, 0);
                        i0 = ggml_cuda_dp4a(q[1], (int) u0.y, i0);
                        i0 = ggml_cuda_dp4a(q[2], (int) u0.z, i0);
                        i0 = ggml_cuda_dp4a(q[3], (int) u0.w, i0);
                        int i1 = ggml_cuda_dp4a(q[4], (int) u1.x, 0);
                        i1 = ggml_cuda_dp4a(q[5], (int) u1.y, i1);
                        i1 = ggml_cuda_dp4a(q[6], (int) u1.z, i1);
                        i1 = ggml_cuda_dp4a(q[7], (int) u1.w, i1);
                        // the one-token kernel's integer: (2 s + 1) (p - 2 n) per scale
                        sumf[j] = fmaf(ad[j * ast + ib], (float) (ls0 * i0 + ls1 * i1), sumf[j]);
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[j][m] = fmaf(dd, sumf[j], acc[j][m]);
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

// The member count nt (1..NTM) as a constant.
template <ggml_type type, bool gate, int SU, int NT, int NTM>
static __device__ __forceinline__ void rows_nt(const int nt, const args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                               const char * xu, const char * xg, char * smw, const uint32_t * tab,
                                               const char * act, const int doff, float * dst, const int64_t bias_off) {
    if constexpr (NT == NTM) {
        rows<type, gate, SU, NT>(p, fusion, xu, xg, smw, tab, act, doff, dst, bias_off);
    } else {
        if (nt == NT) {
            rows<type, gate, SU, NT>(p, fusion, xu, xg, smw, tab, act, doff, dst, bias_off);
        } else {
            rows_nt<type, gate, SU, NT + 1, NTM>(nt, p, fusion, xu, xg, smw, tab, act, doff, dst, bias_off);
        }
    }
}

// Static shared memory of a kernel: the warps' staging buffers (one matrix's segments, 16 bytes of padding for
// mmvqp_lds_h's read past a block) and the grid.
template <ggml_type type> constexpr int static_smem() {
    constexpr int G = gtype<type>();
    return MMVQP_MTOK_IQ_NWARPS * (WARP_SIZE * mmvqp_iq::gtraits<G>::BS + 16) + 4 * mmvqp_iq::gtraits<G>::GW;
}

// Grid: (row blocks / rb, tokens * slots + shared expert, samples). Dynamic shared memory: the staged activations of up
// to NTM members, act_tok bytes each.
template <ggml_type type, bool gate, int SU, int NTM>
__launch_bounds__(MMVQP_MTOK_IQ_NWARPS*WARP_SIZE, NTM >= 8 ? 2 : MMVQP_MTOK_IQ_MIN_BLOCKS)
static __global__ void mul_mat_vec_iq_pascal_mtok(const args p, const ggml_cuda_mm_fusion_args_device fusion) {
    constexpr int G  = gtype<type>();
    constexpr int BS = mmvqp_iq::gtraits<G>::BS;
    constexpr int GW = mmvqp_iq::gtraits<G>::GW;
    constexpr int NW = MMVQP_MTOK_IQ_NWARPS;
    __shared__ uint4 sm[NW][(WARP_SIZE * BS + 16 + 15) / 16];
    __shared__ uint2 tab[GW / 2];
    extern __shared__ uint4 mtok_iq_dyn[];

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
        if (!group(p, sample_dst, nt, yoff, doff, channel_x)) {
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

    // the grid and the members' activations, staged by the whole block
    const int tid = threadIdx.y * WARP_SIZE + lane;
    {
        const uint32_t * g = grid_table<type>();
        for (int i = tid; i < GW; i += NW * WARP_SIZE) {
            ((uint32_t *) tab)[i] = g[i];
        }
    }
    char * act = (char *) mtok_iq_dyn;
    const int nq = 8 * p.nchunk;
    for (int j = 0; j < nt; ++j) {
        const char * yj = p.vy + __shfl_sync(0xFFFFFFFF, yoff, j);
        char       * aj = act + j * p.act_tok;
        for (int q = tid; q < nq; q += NW * WARP_SIZE) {
            const unsigned int * ys = (const unsigned int *) (yj + q * (int) sizeof(block_q8_1));
            uint32_t v[9];
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                v[k] = __ldg(ys + k);
            }
            uint4 * du = (uint4 *) (aj + (q / 8) * MMVQP_MTOK_IQ_ACT_RUN + (q % 8) * 32);
            if constexpr (G == mmvqp_iq::G_IQ1_M) {
                // each group of 8 values in the order (0, 4, 1, 5), (2, 6, 3, 7) of IQ1_M's lookup
                du[0] = make_uint4(__byte_perm(v[1], v[2], 0x5140), __byte_perm(v[1], v[2], 0x7362),
                                   __byte_perm(v[3], v[4], 0x5140), __byte_perm(v[3], v[4], 0x7362));
                du[1] = make_uint4(__byte_perm(v[5], v[6], 0x5140), __byte_perm(v[5], v[6], 0x7362),
                                   __byte_perm(v[7], v[8], 0x5140), __byte_perm(v[7], v[8], 0x7362));
            } else {
                du[0] = make_uint4(v[1], v[2], v[3], v[4]);
                du[1] = make_uint4(v[5], v[6], v[7], v[8]);
            }
            ((float *) (aj + p.nchunk * MMVQP_MTOK_IQ_ACT_RUN))[q] = mmvqp_h2f_lo(v[0]);
        }
    }
    __syncthreads();

    const int64_t xoff     = sample_x*p.sample_bytes_x + channel_x*p.channel_bytes_x;
    const char *  xu       = (shared_expert ? (const char *) fusion.shared_up : p.vx) + xoff;
    const char *  xg       = gate ? (const char *) (shared_expert ? fusion.shared_gate : fusion.gate) + xoff : nullptr;
    float       * dst      = shared_expert ? fusion.shared_dst : p.dst;
    const int64_t bias_off = sample_dst*p.stride_sample_dst + channel_x*p.stride_channel_dst;
    rows_nt<type, gate, SU, 1, NTM>(nt, p, fusion, xu, xg, (char *) sm[threadIdx.y], (const uint32_t *) tab, act, doff,
                                    dst, bias_off);
}

template <ggml_type type, bool gate, int SU>
static void launch_ntm(const int ntm, const args & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid,
                       const int smem, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQP_MTOK_IQ_NWARPS, 1);
    if (ntm <= 2) {
        mul_mat_vec_iq_pascal_mtok<type, gate, SU, 2><<<grid, block, smem, stream>>>(p, fusion);
    } else if (ntm <= 4) {
        mul_mat_vec_iq_pascal_mtok<type, gate, SU, 4><<<grid, block, smem, stream>>>(p, fusion);
    } else {
        mul_mat_vec_iq_pascal_mtok<type, gate, SU, 8><<<grid, block, smem, stream>>>(p, fusion);
    }
}

// su: bytes per staging load (16, 8 or 4); ntm: the most members a group can have
template <ggml_type type>
static void launch(const args & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid, const int smem,
                   const bool gate, const int su, const int ntm, cudaStream_t stream) {
    if (gate) {
        switch (su) {
            case 16: launch_ntm<type, true, 16>(ntm, p, fusion, grid, smem, stream); break;
            case 8:  launch_ntm<type, true, 8>(ntm, p, fusion, grid, smem, stream);  break;
            default: launch_ntm<type, true, 4>(ntm, p, fusion, grid, smem, stream);  break;
        }
    } else {
        switch (su) {
            case 16: launch_ntm<type, false, 16>(ntm, p, fusion, grid, smem, stream); break;
            case 8:  launch_ntm<type, false, 8>(ntm, p, fusion, grid, smem, stream);  break;
            default: launch_ntm<type, false, 4>(ntm, p, fusion, grid, smem, stream);  break;
        }
    }
}

} // namespace mtok_iq

// The launchers per type (mmvq-pascal-mtok-<type>.cu, one file per type so that they compile in parallel).
#define MMVQP_MTOK_IQ_DECL(name) \
    void ggml_cuda_mtok_iq_launch_##name(const mtok_iq::args & p, const ggml_cuda_mm_fusion_args_device & fusion, \
                                         const dim3 grid, const int smem, const bool gate, const int su, const int ntm, \
                                         cudaStream_t stream)
MMVQP_MTOK_IQ_DECL(iq2_xxs);
MMVQP_MTOK_IQ_DECL(iq2_xs);
MMVQP_MTOK_IQ_DECL(iq2_s);
MMVQP_MTOK_IQ_DECL(iq3_xxs);
MMVQP_MTOK_IQ_DECL(iq3_s);
MMVQP_MTOK_IQ_DECL(iq1_m);

// The fast versions (C1138, GGML_CUDA_MTOK_IQ_FAST; mmvq-pascal-mtok-<type>-fast.cu): the same arguments, the gate and
// up matrices in one pass; mtok_iq_run (mmvq-pascal-mtok.cu) checks the shapes and alignments each one needs.
#define MMVQP_MTOK_IQ_FAST_DECL(name) \
    void ggml_cuda_mtok_iq_fast_launch_##name(const mtok_iq::args & p, const ggml_cuda_mm_fusion_args_device & fusion, \
                                              const dim3 grid, const int smem, const bool gate, const int ntm, \
                                              cudaStream_t stream)
MMVQP_MTOK_IQ_FAST_DECL(iq2_xxs);
MMVQP_MTOK_IQ_FAST_DECL(iq2_xs);
MMVQP_MTOK_IQ_FAST_DECL(iq2_s);
MMVQP_MTOK_IQ_FAST_DECL(iq1_m);
int ggml_cuda_mtok_iq2_xxs_fast_static_smem();
int ggml_cuda_mtok_iq2_xs_fast_static_smem();
int ggml_cuda_mtok_iq2_s_fast_static_smem();
int ggml_cuda_mtok_iq1_m_fast_static_smem();
// The PTX versions of the fast kernels (GGML_CUDA_MTOK_IQ_FAST=2, mmvq-pascal-mtok-iq-ptx.cu; IQ2_XXS, IQ2_S and
// IQ2_XS): false where none takes the product (or it cannot load), which then runs the CUDA C version.
bool ggml_cuda_mtok_iq_ptx_launch(ggml_type type, const mtok_iq::args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                  const dim3 grid, const int smem, const bool gate, const int ntm, cudaStream_t stream);

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
