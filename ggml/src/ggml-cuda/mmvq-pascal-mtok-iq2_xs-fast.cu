// IQ2_XS's multi-token expert kernel, the fast version (C1138, code row 28r3; GGML_CUDA_MTOK_IQ_FAST=1): MUL_MAT_ID of 2
// to 8 tokens at rows of 2 lanes (row widths of 256 n weights with n = 2 mod 4, e.g. Qwen3.8-Flash-Next's 2,560),
// bit-identical to mul_mat_vec_iq_pascal_mtok<GGML_TYPE_IQ2_XS> (mmvq-pascal-mtok-iq.cuh): the same two integer sums per
// 32 weights, the same FP32 operations in the same order, the same reduction over a row's two lanes. The method is that
// of IQ2_XXS's and IQ2_S's fast kernels (mmvq-pascal-mtok-iq2_xxs-fast.cu, mmvq-pascal-mtok-iq2_s-fast.cu); what IQ2_XS
// changes:
//   - A 16-bit entry per 8 weights: a 9-bit grid index and a 7-bit sign pattern on top (the eighth sign their parity, as
//     IQ2_XXS's). The index is one BFE of the entry's word (512 grid entries: the code table, a word of eight nibble
//     codes per entry, is 2 KB, kept in 8 copies interleaved by word: 16 KB); the signs take IQ2_XXS's path (BFE, the
//     pairs into the selector's nibbles, POPC and BFI for the parity, one PRMT that spreads them to bit 2 of the codes).
//   - Two 4-bit scales per 32 weights, laid out as IQ2_S's (a byte per 32 weights, the low nibble for weights 0-15):
//     IQ2_S's two DP4A chains and exact FFMAs, (float) (ls0 * i0 + ls1 * i1) of the old kernel, and its scale floats.
//   - Row segments: 740-byte rows (4 mod 16), 74-byte super-blocks. Lane 1's super-block starts 74 bytes after lane 0's,
//     2 words further into a 16-byte unit than lane 0's (74 = 64 + 8 + 2), so with 16-byte loads the two lanes would
//     need different words and the warp-uniform choice of words by alignment would break. The lanes load 8-byte units
//     instead, 10 a lane and matrix from (segment & ~7) + 72 sl: both lanes' super-blocks then start at the same word of
//     their units (kappa = bit 2 of the segment's address, one word further each iteration of 148 bytes), lane 1's two
//     bytes later, and a warp-uniform branch of 2 cases picks the words.
// The routed + shared expert launch, skipped slots (id -1), biases and the GLU ops are the old kernel's.

#include "mmvq-pascal-mtok-iq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace mtok_iq2xs_fast {

using mtok_iq::args;

constexpr int      NW         = MMVQP_MTOK_IQ_NWARPS; // 4 warps, 16 rows each per row block
constexpr int      COPIES     = 8;                    // copies of the code table, interleaved by word
constexpr int      SB         = 74;                   // bytes per super-block
constexpr uint32_t LUT_POS    = 0x002b1908u;          // bytes 8, 25, 43, 0: the grid values by code
constexpr uint32_t LUT_NEG    = 0x00d5e7f8u;          // -8, -25, -43, 0
constexpr uint32_t LUT_SGN    = 0x44400400u;          // two sign bits (index b0 + 2 b1) -> bit 2 of two nibbles
constexpr uint32_t DP4A_BASE  = 0x4B400000u;          // the float 1.5 * 2^23
constexpr float    DP4A_BASEF = 12582912.0f;

static __device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// bits [pos, pos + len) of b replaced by the low len bits of a
static __device__ __forceinline__ uint32_t bfi(const uint32_t a, const uint32_t b, const uint32_t pos, const uint32_t len) {
    uint32_t d;
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(d) : "r"(a), "r"(b), "r"(pos), "r"(len));
    return d;
}

static __device__ __forceinline__ uint32_t bfe(const uint32_t a, const uint32_t pos, const uint32_t len) {
    uint32_t d;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(pos), "r"(len));
    return d;
}

// The 18 words of a lane's super-block after its d (w[k] = bytes 2 + 4k .. 5 + 4k: k = 2 ib, 2 ib + 1 the four 16-bit
// entries of block ib, 16 and 17 the scales of blocks 0-3 and 4-7), and the word whose low half (sel 0x5432) or high
// half (0x7654) is d. u: the lane's 10 8-byte units from the first 8-byte boundary at or below its super-block, which
// starts at byte 4 KAPPA (lane 0) or 4 KAPPA + 2 (lane 1).
template <int KAPPA>
static __device__ __forceinline__ void words(const uint32_t * u, const uint32_t sel, uint32_t * w, uint32_t & dw) {
#pragma unroll
    for (int k = 0; k < 18; ++k) {
        w[k] = prmt(u[KAPPA + k], u[KAPPA + k + 1], sel);
    }
    dw = u[KAPPA];
}

// One 32-weight block of a super-block: q[0..7] its int8 weights in the staged order (q[0..3] weights 0-15, q[4..7]
// weights 16-31). e01, e23: the words of its entries 0, 1 and 2, 3 (entry l of group l in the low half for l even, the
// high half for l odd: 9 index bits, then 7 sign bits).
static __device__ __forceinline__ void decode(const uint32_t e01, const uint32_t e23, const uint32_t * tab, int * q) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t e  = l < 2 ? e01 : e23;
        const uint32_t hb = 16 * (l & 1);                       // the entry's first bit in its word
        const uint32_t s7 = bfe(e, hb + 9, 7);                  // signs of weights 0..6 of group l
        uint32_t       x  = (s7 | (s7 << 6)) & 0x3333u;         // pairs (0,1) (4,5) (2,3) (6,-) in nibbles 0..3
        x = bfi(__popc(s7), x, 13, 1);                          // weight 7: the parity of the seven
        const uint32_t sn = prmt(LUT_SGN, 0, x);                // bit 2 of nibble n: the sign of weight PI[n]
        const uint32_t c  = tab[bfe(e, hb, 9) * COPIES];        // codes of the group's grid entry, nibble n weight PI[n]
        const uint32_t s  = c | sn;
        q[2*l + 0] = (int) prmt(LUT_POS, LUT_NEG, s);           // weights 0, 1, 4, 5 of group l
        q[2*l + 1] = (int) prmt(LUT_POS, LUT_NEG, s >> 16);     // weights 2, 3, 6, 7
    }
}

// The block's rows for its NT members (gate: up and gate of the expert, else the one matrix xu).
template <bool gate, int NT>
static __device__ __forceinline__ void rows(const args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                            const char * xu, const char * xg, const uint32_t * tab0, const char * act,
                                            const int doff, float * dst, const int64_t bias_off) {
    constexpr int M = gate ? 2 : 1; // matrices: up (or plain), gate
    const int      lane = threadIdx.x;
    const int      sl   = lane & 1;
    const int      grp  = lane >> 1;
    const uint32_t sel  = sl ? 0x7654u : 0x5432u;
    const uint32_t * tab = tab0 + (lane & (COPIES - 1));
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
        // rows 4 apart: warp w of the row block takes rows w, w + 4, ..., w + 60
        const int row0 = rblk * (NW * 16) + (int) threadIdx.y + 4 * grp;
        const int rowc = min(row0, p.nrows - 1);

        const char * rw[M];
        rw[0] = xu + (int64_t) rowc * p.row_bytes;
        if constexpr (gate) {
            rw[1] = xg + (int64_t) rowc * p.row_bytes;
        }
        // the word of its first 8-byte unit at which the warp's segments start (lane 0 = the warp's first row, which is
        // a row of the matrix whenever any of the warp's rows is); one word further each iteration (148 bytes)
        const int kappa0 = __shfl_sync(0xFFFFFFFF, (int) (((uintptr_t) rw[0] >> 2) & 1), 0);

        float acc[NT][M];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[j][m] = 0.0f;
            }
        }

        for (int it = 0; it < p.niter; ++it) {
            const int     c  = 2*it + sl;                       // the lane's super-block of the row
            const char  * au = act + c * MMVQP_MTOK_IQ_ACT_RUN; // its int8 activations, member 0
            const float * ad = ad0 + 8 * c;                     // its activation scales, member 0

            uint32_t w[M][18];
            uint32_t dw[M];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const char  * seg = rw[m] + it * (2 * SB);
                const uint2 * src = (const uint2 *) (((uintptr_t) seg & ~(uintptr_t) 7) + 72 * sl);
                uint32_t u[20];
#pragma unroll
                for (int k = 0; k < 10; ++k) {
                    const uint2 v = __ldg(src + k);
                    u[2*k + 0] = v.x;
                    u[2*k + 1] = v.y;
                }
                if (((kappa0 + it) & 1) == 0) {
                    words<0>(u, sel, w[m], dw[m]);
                } else {
                    words<1>(u, sel, w[m], dw[m]);
                }
            }

            float sumf[NT][M];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    sumf[j][m] = 0.0f;
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                // blocks 4 h .. 4 h + 3: their scales as bytes 2 s + 1
                uint32_t e0[M], e1[M];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const uint32_t sw = w[m][16 + h];
                    e0[m] = ((sw << 1) & 0x1E1E1E1Eu) | 0x01010101u; // byte b: 2 ls0 + 1 of block 4 h + b
                    e1[m] = ((sw >> 3) & 0x1E1E1E1Eu) | 0x01010101u; // 2 ls1 + 1
                }
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int ib = 4*h + b;
                    int   q[M][8];
                    float fl0[M], fl1[M], nc1[M];
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        decode(w[m][2*ib], w[m][2*ib + 1], tab, q[m]);
                        const uint32_t sb = 0x7440u | (uint32_t) b; // byte b of e under the exponent of 2^23
                        fl0[m] = __uint_as_float(prmt(e0[m], 0x4B000000u, sb)) - 8388608.0f;
                        fl1[m] = __uint_as_float(prmt(e1[m], 0x4B000000u, sb)) - 8388608.0f;
                        nc1[m] = fl1[m] * -DP4A_BASEF;
                    }
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint4 * pu = (const uint4 *) (au + j * p.act_tok + 32 * ib);
                        const uint4   u0 = pu[0];
                        const uint4   u1 = pu[1];
                        const float   a  = ad[j * ast + ib];
#pragma unroll
                        for (int m = 0; m < M; ++m) {
                            int i0 = ggml_cuda_dp4a(q[m][0], (int) u0.x, (int) DP4A_BASE);
                            i0 = ggml_cuda_dp4a(q[m][1], (int) u0.y, i0);
                            i0 = ggml_cuda_dp4a(q[m][2], (int) u0.z, i0);
                            i0 = ggml_cuda_dp4a(q[m][3], (int) u0.w, i0);
                            int i1 = ggml_cuda_dp4a(q[m][4], (int) u1.x, (int) DP4A_BASE);
                            i1 = ggml_cuda_dp4a(q[m][5], (int) u1.y, i1);
                            i1 = ggml_cuda_dp4a(q[m][6], (int) u1.z, i1);
                            i1 = ggml_cuda_dp4a(q[m][7], (int) u1.w, i1);
                            // (float) (ls0 * i0 + ls1 * i1) of the old kernel, exactly
                            const float p1 = fmaf(fl1[m], __int_as_float(i1), nc1[m]);
                            const float r  = fmaf(fl0[m], __int_as_float(i0) - DP4A_BASEF, p1);
                            sumf[j][m] = fmaf(a, r, sumf[j][m]);
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float dd = 0.125f * mmvqp_h2f_lo(sl ? dw[m] >> 16 : dw[m]);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[j][m] = fmaf(dd, sumf[j][m], acc[j][m]);
                }
            }
        }

#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[j][m] += __shfl_xor_sync(0xFFFFFFFF, acc[j][m], 1, WARP_SIZE);
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

template <bool gate, int NT, int NTM>
static __device__ __forceinline__ void rows_nt(const int nt, const args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                               const char * xu, const char * xg, const uint32_t * tab, const char * act,
                                               const int doff, float * dst, const int64_t bias_off) {
    if constexpr (NT == NTM) {
        rows<gate, NT>(p, fusion, xu, xg, tab, act, doff, dst, bias_off);
    } else {
        if (nt == NT) {
            rows<gate, NT>(p, fusion, xu, xg, tab, act, doff, dst, bias_off);
        } else {
            rows_nt<gate, NT + 1, NTM>(nt, p, fusion, xu, xg, tab, act, doff, dst, bias_off);
        }
    }
}

// Grid and dynamic shared memory as mul_mat_vec_iq_pascal_mtok's: (row blocks / rb, tokens * slots + shared expert,
// samples), the staged activations of up to NTM members (in the order above).
template <bool gate, int NTM>
__launch_bounds__(NW*WARP_SIZE, NTM >= 8 ? 2 : MMVQP_MTOK_IQ_MIN_BLOCKS)
static __global__ void mul_mat_vec_iq2_xs_pascal_mtok_fast(const args p, const ggml_cuda_mm_fusion_args_device fusion) {
    __shared__ uint4 tab[512 * COPIES / 4];
    extern __shared__ uint4 mtok_iq2xs_fast_dyn[];

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
        if (!mtok_iq::group(p, sample_dst, nt, yoff, doff, channel_x)) {
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

    const int tid = threadIdx.y * WARP_SIZE + lane;
    // the code table: entry e's codes (0, 1, 2 for 8, 25, 43: a grid byte >> 4) as nibbles, nibble n weight
    // (0, 1, 4, 5, 2, 3, 6, 7)[n], in 8 copies (two 16-byte stores)
    for (int e = tid; e < 512; e += NW * WARP_SIZE) {
        const uint64_t g  = iq2xs_grid[e];
        const uint32_t c0 = ((uint32_t) g >> 4) & 0x03030303u;         // weights 0..3
        const uint32_t c1 = ((uint32_t) (g >> 32) >> 4) & 0x03030303u; // weights 4..7
        const uint32_t c  = prmt(c0, c1, 0x6240u) | (prmt(c0, c1, 0x7351u) << 4);
        tab[2*e + 0] = make_uint4(c, c, c, c);
        tab[2*e + 1] = make_uint4(c, c, c, c);
    }
    // the members' activations: each group of 8 int8 values in the order (0, 1, 4, 5), (2, 3, 6, 7)
    char * act = (char *) mtok_iq2xs_fast_dyn;
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
            du[0] = make_uint4(prmt(v[1], v[2], 0x5410), prmt(v[1], v[2], 0x7632),
                               prmt(v[3], v[4], 0x5410), prmt(v[3], v[4], 0x7632));
            du[1] = make_uint4(prmt(v[5], v[6], 0x5410), prmt(v[5], v[6], 0x7632),
                               prmt(v[7], v[8], 0x5410), prmt(v[7], v[8], 0x7632));
            ((float *) (aj + p.nchunk * MMVQP_MTOK_IQ_ACT_RUN))[q] = mmvqp_h2f_lo(v[0]);
        }
    }
    __syncthreads();

    const int64_t xoff     = sample_x*p.sample_bytes_x + channel_x*p.channel_bytes_x;
    const char *  xu       = (shared_expert ? (const char *) fusion.shared_up : p.vx) + xoff;
    const char *  xg       = gate ? (const char *) (shared_expert ? fusion.shared_gate : fusion.gate) + xoff : nullptr;
    float       * dst      = shared_expert ? fusion.shared_dst : p.dst;
    const int64_t bias_off = sample_dst*p.stride_sample_dst + channel_x*p.stride_channel_dst;
    rows_nt<gate, 1, NTM>(nt, p, fusion, xu, xg, (const uint32_t *) tab, act, doff, dst, bias_off);
}

template <bool gate>
static void launch_ntm(const int ntm, const args & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid,
                       const int smem, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, NW, 1);
    if (ntm <= 2) {
        mul_mat_vec_iq2_xs_pascal_mtok_fast<gate, 2><<<grid, block, smem, stream>>>(p, fusion);
    } else if (ntm <= 4) {
        mul_mat_vec_iq2_xs_pascal_mtok_fast<gate, 4><<<grid, block, smem, stream>>>(p, fusion);
    } else {
        mul_mat_vec_iq2_xs_pascal_mtok_fast<gate, 8><<<grid, block, smem, stream>>>(p, fusion);
    }
}

} // namespace mtok_iq2xs_fast

// Static shared memory of the fast kernel: the code table.
int ggml_cuda_mtok_iq2_xs_fast_static_smem() {
    return 512 * mtok_iq2xs_fast::COPIES * (int) sizeof(uint32_t);
}

MMVQP_MTOK_IQ_FAST_DECL(iq2_xs) {
    if (gate) {
        mtok_iq2xs_fast::launch_ntm<true>(ntm, p, fusion, grid, smem, stream);
    } else {
        mtok_iq2xs_fast::launch_ntm<false>(ntm, p, fusion, grid, smem, stream);
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
