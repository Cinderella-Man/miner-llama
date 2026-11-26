// IQ1_M's multi-token expert kernel, the fast version (C1138, code row 28r4; GGML_CUDA_MTOK_IQ_FAST=1): MUL_MAT_ID of 2
// to 8 tokens at rows of 2 lanes (row widths of 256 n weights with n = 2 mod 4, e.g. Qwen3.8-Flash-Next's 2,560),
// bit-identical to mul_mat_vec_iq_pascal_mtok<GGML_TYPE_IQ1_M> (mmvq-pascal-mtok-iq.cuh): the same two integer sums per
// 32 weights, the same FP32 operations in the same order, the same reduction over a row's two lanes. The method is that
// of the IQ2 types' fast kernels (gate and up in one pass, each 4 weights' values in one PRMT from a nibble code, the
// integer sums' floats without conversions, the rows straight into registers); what IQ1_M changes:
//   - The code table is the grid itself: ggml's iq1s_grid_gpu holds grid + 1 (0, 1, 2) as nibbles, weights 0-3 in the
//     low nibbles of bytes 0-3 and 4-7 in the high ones, so a group's nibbles read (0, 4, 1, 5, 2, 6, 3, 7), the old
//     kernel's order (its activations are staged in it, act_perm_iq1). A nibble's code picks 8 (grid + delta) from the
//     register pair {-7, 1, 9, 0, -9, -1, 7, 0} (byte code + 4 for a negative delta).
//   - The delta's sign in the table: the table has 4,096 words, the 2,048 grid entries and then the same with bit 2 of
//     every nibble set (16 KB, one copy). A group's index is its qs byte and, above it, its qh nibble (3 high index bits
//     and the delta's sign on top): one PRMT puts the two side by side (odd groups from a copy of the qh word shifted
//     right by 4, so that their nibble is the low one of its byte) and its low 12 bits are the table index. The lookup
//     then gives the value PRMTs' selectors with the sign already in them: 6 instructions and one shared load per 8
//     weights, no sign arithmetic.
//   - Two 3-bit scales per 32 weights: IQ2_S's two DP4A chains (weights 0-15 and 16-31, each started at 0x4B400000) and
//     exact FFMAs, (float) (ls0 * i0 + ls1 * i1) of the old kernel. The scales of four blocks (a 32-bit word of two
//     16-bit scale words) become the bytes 2 s + 1 of two words with 7 logic and shift instructions, and each byte a
//     float by a PRMT under the exponent of 2^23 and an FADD.
//   - d: the four top nibbles of the scale words, once per super-block (the old kernel's sblock_scale).
//   - Row segments: 560-byte rows (0 mod 16) and 56-byte super-blocks with nothing in front, so lane 0's super-block
//     starts on a 16-byte boundary and lane 1's (56 bytes later) on an 8-byte one: each lane loads its own 7 aligned
//     8-byte units, whose words are the super-block's words (no choice of words by alignment, no word PRMTs).
// The routed + shared expert launch, skipped slots (id -1), biases and the GLU ops are the old kernel's.

#include "mmvq-pascal-mtok-iq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace mtok_iq1m_fast {

using mtok_iq::args;

constexpr int      NW         = MMVQP_MTOK_IQ_NWARPS; // 4 warps, 16 rows each per row block
constexpr int      SB         = 56;                   // bytes per super-block
constexpr int      NENT       = 2048;                 // grid entries
constexpr uint32_t NEG        = 0x44444444u;          // bit 2 of every nibble: a negative delta's codes
constexpr uint32_t LUT_POS    = 0x000901F9u;          // bytes -7, 1, 9, 0: 8 (grid + 1/8) by code grid + 1
constexpr uint32_t LUT_NEG    = 0x0007FFF7u;          // -9, -1, 7, 0: 8 (grid - 1/8)
constexpr uint32_t DP4A_BASE  = 0x4B400000u;          // the float 1.5 * 2^23
constexpr float    DP4A_BASEF = 12582912.0f;

static __device__ __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// One 32-weight block of a super-block: q[0..7] its int8 weights 8 (grid + delta) in the staged order (q[2l], q[2l + 1]
// group l's weights (0, 4, 1, 5), (2, 6, 3, 7); q[0..3] weights 0-15, q[4..7] weights 16-31). qs: the block's index
// word; qh0: the word holding its qh nibbles, byte b its groups 0 and 1, byte b + 1 its groups 2 and 3; qh4 = qh0 >> 4.
static __device__ __forceinline__ void decode(const uint32_t qs, const uint32_t qh0, const uint32_t qh4, const int b,
                                              const uint32_t * tab, int * q) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        // the group's qs byte, its qh nibble above it: the 11-bit grid index and, at bit 11, the delta's sign
        const uint32_t px = prmt(qs, (l & 1) ? qh4 : qh0, (uint32_t) (l | (4 + b + (l >> 1)) << 4));
        const uint32_t c  = tab[px & 0xFFFu]; // the entry's codes, bit 2 of each nibble set for a negative delta
        q[2*l + 0] = (int) prmt(LUT_POS, LUT_NEG, c);       // weights 0, 4, 1, 5 of group l
        q[2*l + 1] = (int) prmt(LUT_POS, LUT_NEG, c >> 16); // weights 2, 6, 3, 7
    }
}

// The block's rows for its NT members (gate: up and gate of the expert, else the one matrix xu).
template <bool gate, int NT>
static __device__ __forceinline__ void rows(const args & p, const ggml_cuda_mm_fusion_args_device & fusion,
                                            const char * xu, const char * xg, const uint32_t * tab, const char * act,
                                            const int doff, float * dst, const int64_t bias_off) {
    constexpr int M = gate ? 2 : 1; // matrices: up (or plain), gate
    const int lane = threadIdx.x;
    const int sl   = lane & 1;
    const int grp  = lane >> 1;
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
        // rows 4 apart, as the other types' fast kernels: warp w of the row block takes rows w, w + 4, ..., w + 60
        const int row0 = rblk * (NW * 16) + (int) threadIdx.y + 4 * grp;
        const int rowc = min(row0, p.nrows - 1);

        const char * rw[M];
        rw[0] = xu + (int64_t) rowc * p.row_bytes;
        if constexpr (gate) {
            rw[1] = xg + (int64_t) rowc * p.row_bytes;
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
            const int     c  = 2*it + sl;                       // the lane's super-block of the row
            const char  * au = act + c * MMVQP_MTOK_IQ_ACT_RUN; // its int8 activations, member 0
            const float * ad = ad0 + 8 * c;                     // its activation scales, member 0

            // the lane's super-block, word k = bytes 4k .. 4k + 3: k = ib block ib's grid index bytes (qs), 8 .. 11
            // the qh nibbles (two blocks a word), 12 and 13 the scale words of blocks 0-3 and 4-7 (with d on top)
            uint32_t w[M][14];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint2 * src = (const uint2 *) (rw[m] + it * (2 * SB) + SB * sl);
#pragma unroll
                for (int k = 0; k < 7; ++k) {
                    const uint2 v = __ldg(src + k);
                    w[m][2*k + 0] = v.x;
                    w[m][2*k + 1] = v.y;
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
                // blocks 4 h .. 4 h + 3: their scales as bytes 2 s + 1, byte b block 4 h + b (the scale word's 16-bit
                // halves hold blocks 4 h, 4 h + 1 and 4 h + 2, 4 h + 3: ls0 at bits 0-2 and 6-8, ls1 at 3-5 and 9-11)
                uint32_t e0[M], e1[M];
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const uint32_t sw = w[m][12 + h];
                    e0[m] = ((sw << 1) & 0x000E000Eu) | ((sw << 3) & 0x0E000E00u) | 0x01010101u; // 2 ls0 + 1
                    e1[m] = ((sw >> 2) & 0x000E000Eu) | (sw & 0x0E000E00u) | 0x01010101u;        // 2 ls1 + 1
                }
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int ib = 4*h + b;
                    int   q[M][8];
                    float fl0[M], fl1[M], nc1[M];
#pragma unroll
                    for (int m = 0; m < M; ++m) {
                        const uint32_t qh0 = w[m][8 + ib/2];
                        decode(w[m][ib], qh0, qh0 >> 4, 2 * (ib & 1), tab, q[m]);
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
                const float dd = mmvqp_iq::sblock_scale<mmvqp_iq::G_IQ1_M>(w[m]); // 0.125 d, d from the scale words
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
// samples), the staged activations of up to NTM members (in IQ1_M's order, as the old kernel stages them).
template <bool gate, int NTM>
__launch_bounds__(NW*WARP_SIZE, NTM >= 8 ? 2 : MMVQP_MTOK_IQ_MIN_BLOCKS)
static __global__ void mul_mat_vec_iq1_m_pascal_mtok_fast(const args p, const ggml_cuda_mm_fusion_args_device fusion) {
    __shared__ uint32_t tab[2 * NENT];
    extern __shared__ uint4 mtok_iq1m_fast_dyn[];

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
    // the code table: the grid entries (grid + 1 as nibbles), then the same with bit 2 of every nibble set (a negative
    // delta), indexed by the 11-bit grid index and the delta's sign above it
    for (int e = tid; e < NENT; e += NW * WARP_SIZE) {
        const uint32_t g = iq1s_grid_gpu[e];
        tab[e]        = g;
        tab[e + NENT] = g | NEG;
    }
    // the members' activations: each group of 8 int8 values in the order (0, 4, 1, 5), (2, 6, 3, 7), as the old kernel
    char * act = (char *) mtok_iq1m_fast_dyn;
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
            du[0] = make_uint4(prmt(v[1], v[2], 0x5140), prmt(v[1], v[2], 0x7362),
                               prmt(v[3], v[4], 0x5140), prmt(v[3], v[4], 0x7362));
            du[1] = make_uint4(prmt(v[5], v[6], 0x5140), prmt(v[5], v[6], 0x7362),
                               prmt(v[7], v[8], 0x5140), prmt(v[7], v[8], 0x7362));
            ((float *) (aj + p.nchunk * MMVQP_MTOK_IQ_ACT_RUN))[q] = mmvqp_h2f_lo(v[0]);
        }
    }
    __syncthreads();

    const int64_t xoff     = sample_x*p.sample_bytes_x + channel_x*p.channel_bytes_x;
    const char *  xu       = (shared_expert ? (const char *) fusion.shared_up : p.vx) + xoff;
    const char *  xg       = gate ? (const char *) (shared_expert ? fusion.shared_gate : fusion.gate) + xoff : nullptr;
    float       * dst      = shared_expert ? fusion.shared_dst : p.dst;
    const int64_t bias_off = sample_dst*p.stride_sample_dst + channel_x*p.stride_channel_dst;
    rows_nt<gate, 1, NTM>(nt, p, fusion, xu, xg, tab, act, doff, dst, bias_off);
}

template <bool gate>
static void launch_ntm(const int ntm, const args & p, const ggml_cuda_mm_fusion_args_device & fusion, const dim3 grid,
                       const int smem, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, NW, 1);
    if (ntm <= 2) {
        mul_mat_vec_iq1_m_pascal_mtok_fast<gate, 2><<<grid, block, smem, stream>>>(p, fusion);
    } else if (ntm <= 4) {
        mul_mat_vec_iq1_m_pascal_mtok_fast<gate, 4><<<grid, block, smem, stream>>>(p, fusion);
    } else {
        mul_mat_vec_iq1_m_pascal_mtok_fast<gate, 8><<<grid, block, smem, stream>>>(p, fusion);
    }
}

} // namespace mtok_iq1m_fast

// Static shared memory of the fast kernel: the code table.
int ggml_cuda_mtok_iq1_m_fast_static_smem() {
    return 2 * mtok_iq1m_fast::NENT * (int) sizeof(uint32_t);
}

MMVQP_MTOK_IQ_FAST_DECL(iq1_m) {
    if (gate) {
        mtok_iq1m_fast::launch_ntm<true>(ntm, p, fusion, grid, smem, stream);
    } else {
        mtok_iq1m_fast::launch_ntm<false>(ntm, p, fusion, grid, smem, stream);
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
