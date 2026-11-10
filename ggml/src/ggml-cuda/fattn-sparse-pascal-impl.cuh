#pragma once

// Sparse decode attention without tensor cores: the kernel, its shape and the per-lane helpers, free of ggml so that
// bench/fattn_sparse (public repo) builds the same code into its CPU check and its standalone GPU check.
// The ggml glue (conditions, scratch, the mask compaction, the switch) is fattn-sparse-pascal.cu.
//
// The call is FLASH_ATTN_EXT with the n_kv_max hint (ggml_flash_attn_ext_set_n_kv_max): at most n_kv_max finite mask
// entries per query row, as qwen4exp's QSA layers ask for (the indexer's top 2,048 cells plus the incomplete pool,
// 2,051 at a pool of 4). Upstream gathers those cells only in the Turing MMA kernel; below Turing every K and V row of
// the cache is read. Here:
//   - upstream's SIMT compaction (flash_attn_mask_to_sparse_indices, one list per query) lists each query's finite mask
//     columns in cache order, with their count;
//   - this kernel splits each list over gridDim.x blocks of equal shares; a block has one warp per 4 query heads of one
//     KV head (Flash-Next: 3 warps for 12 heads), all on the same cells, so a K or V row comes from DRAM once and from
//     L1/L2 for the other warps;
//   - lane l owns dimensions DPL*l .. DPL*l + DPL - 1 (DPL = D/32): the query (pre-scaled by scale*log2(e)) and the
//     V accumulator of its 4 heads live in registers, and each K/V row is converted from F16 or Q8_0 once per warp;
//   - per tile of 32 list entries (one per lane: its cell and its mask value), the scores of 8 keys x 4 heads are summed
//     over the lanes by one transpose-reduce, an online softmax in base 2 runs per head, and P*V adds each V row once
//     for the 4 heads;
//   - each warp writes its partial output, maximum and sum; the last block of a column (query, KV head) to finish (a
//     counter) merges every split, adds the sinks and writes the output.
// FP32 throughout. Results match the dense kernels to rounding (a different summation order), not bit for bit.
//
// Union lists (GGML_CUDA_FATTN_SPARSE_PASCAL_UNION, a second instance of the same kernel): a block takes tpb consecutive
// query tokens of a sequence (fasp_union_tokens: as many as FASP_UNION_WARPS warps hold) and one list, the union of
// their finite mask columns (upstream's compaction with ncols1 = tpb). Warp y takes the heads of warp y % (G/4) for
// token y / (G/4), every warp walks the same cells in the same order, so a K or V row comes from DRAM once per block
// instead of once per query token, and each warp's own mask row scores the cells its token does not keep -inf: a tile
// without any of its token's cells leaves its state exactly as it was (factor 1, P 0). Each token's output is the
// per-token kernel's to rounding (other tile and split boundaries); with the same list and splits it is the same bits.

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>

#if defined(__CUDA_ARCH__)
#define FASP_LDCG(p) __ldcg(p)
#define FASP_UNROLL  _Pragma("unroll")
#else
#define FASP_LDCG(p) (*(p))
#define FASP_UNROLL
#endif

constexpr int   FASP_WARP     = 32;
constexpr int   FASP_HPW      = 4;   // query heads per warp
constexpr int   FASP_TILE     = 32;  // list entries per tile, one per lane
constexpr int   FASP_KPR      = 8;   // keys per transpose-reduce: 8 keys x 4 heads = 32 values
constexpr int   FASP_MAX_COLS = 1024; // merge counters: columns (query x KV head) per launch
constexpr int   FASP_UNION_WARPS = 12; // most warps a block of the union kernel holds (384 threads: 168 registers each)
constexpr int   FASP_UNION_TOKENS = 8; // most query tokens per union list (upstream's compaction groups 1 to 8)
constexpr float FASP_LOG2_E   = 1.4426950408889634f;
constexpr int   FASP_Q8_0_BYTES = 34; // a Q8_0 block: FP16 d, 32 int8

enum fasp_type { FASP_F16 = 0, FASP_Q8_0 = 1 };

// floats per partial: D outputs, then the maximum and the sum, padded to 16 bytes
static __host__ __device__ constexpr int fasp_part_floats(const int D) {
    return D + 4;
}

// Byte strides. z = i3*n_q + iq numbers the query rows; a list belongs to (i3 % mask_ne3, iq).
struct fasp_args {
    const char    * Q;
    int64_t         q_nb1, q_nb2, q_nb3;  // query token, head, sequence
    const char    * K;
    int64_t         k_nb1, k_nb2, k_nb3;  // cell, KV head, sequence (k_ne3 1: shared)
    const char    * V;
    int64_t         v_nb1, v_nb2, v_nb3;
    int             kv_ne3;
    const half    * mask;                 // F16, one row per query
    int64_t         m_nb1, m_nb3;         // in halves
    int             m_ne3;
    const int32_t * idx;                  // lists of n_kv_max cells, the first counts[list] valid
    const int32_t * counts;
    int             n_kv_max;
    const float   * sinks;                // one per query head, or nullptr
    float         * dst;                  // [D, n_head, n_q, ne3], contiguous
    float         * part;                 // [split][column][head of the KV head][fasp_part_floats(D)]
    unsigned      * counter;              // one per column, zero between launches
    int             n_q;                  // queries per sequence
    int             n_head;               // query heads
    int             G;                    // query heads per KV head
    float           scale_log2;           // the softmax scale times log2(e)
    int             fault;                // GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT (testing the tests): 1 output x 1.01,
                                          // 2 mask values ignored, 3 every split's last list entry dropped, 4 (union
                                          // kernel) each token scored with the next token's mask row
};

// The entries [begin, end) of a list of count entries that split s of S reads: equal shares in whole tiles.
static __host__ __device__ __forceinline__ void fasp_split_range(const int count, const int s, const int S,
        int & begin, int & end) {
    const int share = ((count + S - 1)/S + FASP_TILE - 1)/FASP_TILE*FASP_TILE;
    begin = s*share < count ? s*share : count;
    end   = begin + share < count ? begin + share : count;
}

// Which calls the switch sends here (fattn-sparse-pascal.cu reads it from the environment; bench/fattn_sparse checks it
// on the host). A call of at most nq query tokens per sequence is decode or verify: each query token reads its own list
// and the dense kernels read the cache once for all of them, so mode 1 takes it from min_ratio x n_kv_max cells per
// query token. A call of more tokens is a prompt micro-batch, taken only with prompt on and up to prompt_nq tokens: the
// dense kernels compute every cell for each query token, so mode 1 takes it once the cache holds prompt_min_ratio x
// n_kv_max cells, whatever its size. Any other mode takes both at any cache size. With union lists on, a decode or verify
// call of tpb > 1 tokens a list reads ceil(n_q/tpb) lists a sequence, so mode 1 takes it from union_min_ratio x n_kv_max
// cells per list.
struct fasp_rule {
    int   mode             = 0;
    int   nq               = 8;
    float min_ratio        = 1.5f;
    int   prompt           = 0;
    int   prompt_nq        = 128;
    float prompt_min_ratio = 1.0f;
    int   uni              = 0;
    float union_min_ratio  = 1.5f;
};

// Query tokens per block and list of the union kernel: G/4 warps a token, at most FASP_UNION_WARPS warps a block.
static inline int fasp_union_tokens(const int G, const int64_t n_q) {
    const int t = FASP_UNION_WARPS/(G/FASP_HPW);
    const int tmax = t < FASP_UNION_TOKENS ? t : FASP_UNION_TOKENS;
    return n_q < tmax ? (int) n_q : tmax;
}

// tpb: query tokens per list of the union kernel (fasp_union_tokens), used only with r.uni on
static inline bool fasp_rule_takes(const fasp_rule & r, const int64_t n_q, const int64_t n_kv, const int32_t n_kv_max,
        const int tpb = 1) {
    if (n_q <= r.nq) {
        if (r.uni != 0 && tpb > 1) {
            return r.mode != 1 || !((double) n_kv < r.union_min_ratio*n_kv_max*((n_q + tpb - 1)/tpb));
        }
        return r.mode != 1 || !((double) n_kv < r.min_ratio*n_kv_max*n_q);
    }
    if (r.prompt == 0 || n_q > r.prompt_nq) {
        return false;
    }
    return r.mode != 1 || !((double) n_kv < r.prompt_min_ratio*n_kv_max);
}

// Splits per list: about 4 blocks per SM in all, at least 64 entries a split.
static __host__ __device__ __forceinline__ int fasp_pick_splits(const int n_kv_max, const int n_cols, const int nsm) {
    const int target = 4*nsm;
    int S = (target + n_cols - 1)/n_cols;
    const int smax = n_kv_max/64 > 1 ? n_kv_max/64 : 1;
    S = S < 1 ? 1 : S;
    return S < smax ? S : smax;
}

// Head of value i (of 32) of a transpose-reduce, and its key within the group of 8.
static __host__ __device__ __forceinline__ int fasp_tr_head(const int i) {
    return i % FASP_HPW;
}
static __host__ __device__ __forceinline__ int fasp_tr_key(const int i) {
    return i / FASP_HPW;
}

// The DPL values of dimensions DPL*lane .. DPL*lane + DPL - 1 of a K or V row, as floats.
template <int type, int DPL>
static __host__ __device__ __forceinline__ void fasp_load_row(const char * row, const int lane, float * out) {
    static_assert(DPL == 4 || DPL == 8, "4 or 8 dimensions per lane");
    if constexpr (type == FASP_F16) {
#if defined(__CUDA_ARCH__)
        if constexpr (DPL == 8) {
            const uint4 u = __ldg((const uint4 *) (row + 2*DPL*lane));
            const float2 a = __half22float2(*(const half2 *) &u.x);
            const float2 b = __half22float2(*(const half2 *) &u.y);
            const float2 c = __half22float2(*(const half2 *) &u.z);
            const float2 d = __half22float2(*(const half2 *) &u.w);
            out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
            out[4] = c.x; out[5] = c.y; out[6] = d.x; out[7] = d.y;
        } else {
            const uint2 u = __ldg((const uint2 *) (row + 2*DPL*lane));
            const float2 a = __half22float2(*(const half2 *) &u.x);
            const float2 b = __half22float2(*(const half2 *) &u.y);
            out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
        }
#else
        const half * h = (const half *) row + DPL*lane;
        for (int j = 0; j < DPL; ++j) {
            out[j] = __half2float(h[j]);
        }
#endif
    } else {
        // the lane's DPL values lie in one block: d, then int8 at byte offset 2 + i0 % 32 (even, so 2-byte loads)
        const int    i0  = DPL*lane;
        const char * blk = row + (i0/32)*FASP_Q8_0_BYTES;
#if defined(__CUDA_ARCH__)
        const float d = __half2float(__ldg((const half *) blk));
        const unsigned short * qs = (const unsigned short *) (blk + 2 + i0 % 32);
#pragma unroll
        for (int j = 0; j < DPL/2; ++j) {
            const unsigned short w = __ldg(qs + j);
            out[2*j + 0] = d*(float) (int8_t) (w & 0xFF);
            out[2*j + 1] = d*(float) (int8_t) (w >> 8);
        }
#else
        const float d = __half2float(*(const half *) blk);
        const int8_t * qs = (const int8_t *) (blk + 2 + i0 % 32);
        for (int j = 0; j < DPL; ++j) {
            out[j] = d*(float) qs[j];
        }
#endif
    }
}

// One lane's DPL dimensions of head h's output, merged over S splits; part points at split 0 of the head's partial.
// Returns the merged maximum and sum through m_out and l_out.
template <int DPL>
static __host__ __device__ __forceinline__ void fasp_merge_lane(const float * part, const int64_t split_stride,
        const int S, const int D, const int lane, float * o, float & m_out, float & l_out) {
    float m_all = -FLT_MAX/2.0f;
    for (int s = 0; s < S; ++s) {
        const float m = FASP_LDCG(part + s*split_stride + D);
        m_all = m > m_all ? m : m_all;
    }
    FASP_UNROLL
    for (int j = 0; j < DPL; ++j) {
        o[j] = 0.0f;
    }
    float l = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float * ps = part + s*split_stride;
        const float   f  = exp2f(FASP_LDCG(ps + D) - m_all);
        FASP_UNROLL
        for (int j = 0; j < DPL; ++j) {
            o[j] = fmaf(f, FASP_LDCG(ps + DPL*lane + j), o[j]);
        }
        l = fmaf(f, FASP_LDCG(ps + D + 1), l);
    }
    m_out = m_all;
    l_out = l;
}

// The sink of a head (base 2) joins the merged sum; returns the factor for the outputs.
static __host__ __device__ __forceinline__ float fasp_add_sink(const float sink_log2, const float m_all, float & l) {
    const float m2 = sink_log2 > m_all ? sink_log2 : m_all;
    const float f  = exp2f(m_all - m2);
    l = l*f + exp2f(sink_log2 - m2);
    return f;
}

#if defined(__CUDACC__)

// a where m is all ones, b where it is zero; bitwise, because `upper ? v[i] : v[i + n/2]` compiles to an indexed load
// from a local-memory copy of v (256 local loads and stores per call on sm_61)
static __device__ __forceinline__ float fasp_sel(const unsigned m, const float a, const float b) {
    return __uint_as_float((__float_as_uint(a) & m) | (__float_as_uint(b) & ~m));
}

// one step of the transpose-reduce: lanes with bit o set keep values o .. 2o - 1 and send 0 .. o - 1 (o a template
// parameter, so every index is a constant and v stays in registers)
template <int o>
static __device__ __forceinline__ void fasp_tr_step(float * v, const int lane) {
    const unsigned upper = lane & o ? 0xFFFFFFFFu : 0u;
#pragma unroll
    for (int i = 0; i < o; ++i) {
        const float send = fasp_sel(upper, v[i], v[i + o]);
        const float keep = fasp_sel(upper, v[i + o], v[i]);
        v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, o, FASP_WARP);
    }
}

// v[32] holds each lane's partial sums of 32 values; afterwards v[0] of lane l is the sum over all lanes of value l.
static __device__ __forceinline__ float fasp_transpose_reduce(float v[32]) {
    const int lane = threadIdx.x;
    fasp_tr_step<16>(v, lane);
    fasp_tr_step< 8>(v, lane);
    fasp_tr_step< 4>(v, lane);
    fasp_tr_step< 2>(v, lane);
    fasp_tr_step< 1>(v, lane);
    return v[0];
}

// grid (splits, KV heads, query rows z), block (32, G/4): warp y takes query heads 4y .. 4y + 3 of KV head blockIdx.y.
// uni (the union kernel): grid (splits, KV heads, lists: ceil(n_q/tpb) per sequence), block (32, G/4 x tpb): warp y
// takes query heads 4(y % (G/4)) .. + 3 of the list's token y / (G/4); warps past the call's last token only wait.
template <int D, int type, bool uni>
__launch_bounds__(FASP_WARP*(uni ? FASP_UNION_WARPS : 4), 1)
static __global__ void flash_attn_ext_sparse_pascal(const fasp_args a) {
    constexpr int DPL  = D/FASP_WARP;
    constexpr int PART = fasp_part_floats(D);
    constexpr int NWB  = uni ? FASP_UNION_WARPS : 4; // warps a block holds at most

    const int lane = threadIdx.x;
    const int wy   = threadIdx.y;                   // this warp's slot in the block
    const int w    = uni ? wy % (a.G/FASP_HPW) : wy; // its group of 4 heads
    const int S    = gridDim.x;
    const int s    = blockIdx.x;
    const int g    = blockIdx.y;
    int z, iq, i3, list, iq_mask;
    int n_lists = 0;
    bool valid = true;
    if constexpr (uni) {
        const int tpb = blockDim.y/(a.G/FASP_HPW); // query tokens per list
        n_lists = (a.n_q + tpb - 1)/tpb;           // lists per sequence
        const int j = blockIdx.z % n_lists;
        i3      = blockIdx.z / n_lists;
        iq      = j*tpb + wy / (a.G/FASP_HPW);
        valid   = iq < a.n_q;
        iq      = valid ? iq : j*tpb; // a warp past the last token reads the list's first one and computes nothing
        z       = i3*a.n_q + iq;
        list    = (i3 % a.m_ne3)*n_lists + j;
        iq_mask = a.fault == 4 && iq + 1 < a.n_q ? iq + 1 : iq;
    } else {
        z       = blockIdx.z;
        iq      = z % a.n_q;
        i3      = z / a.n_q;
        list    = (i3 % a.m_ne3)*a.n_q + iq;
        iq_mask = iq;
    }
    const int hq0  = g*a.G + FASP_HPW*w; // first query head of this warp
    const int hl   = lane % FASP_HPW;    // the head (of the warp's 4) whose scores this lane keeps

    __shared__                int   cell_sm[NWB][FASP_TILE];
    __shared__ __align__(16) float p_sm[NWB][FASP_TILE*FASP_HPW];
    __shared__                bool  last_block;

    float q[FASP_HPW][DPL];
#pragma unroll
    for (int h = 0; h < FASP_HPW; ++h) {
        const float * qh = (const float *) (a.Q + iq*a.q_nb1 + (int64_t) (hq0 + h)*a.q_nb2 + i3*a.q_nb3) + DPL*lane;
#pragma unroll
        for (int j = 0; j < DPL; j += 4) {
            const float4 t = *(const float4 *) (qh + j);
            q[h][j + 0] = t.x*a.scale_log2;
            q[h][j + 1] = t.y*a.scale_log2;
            q[h][j + 2] = t.z*a.scale_log2;
            q[h][j + 3] = t.w*a.scale_log2;
        }
    }

    float acc[FASP_HPW][DPL];
#pragma unroll
    for (int h = 0; h < FASP_HPW; ++h) {
#pragma unroll
        for (int j = 0; j < DPL; ++j) {
            acc[h][j] = 0.0f;
        }
    }
    float M = -FLT_MAX/2.0f; // running maximum (base 2) of head hl
    float L = 0.0f;          // running sum of head hl

    const int32_t * idx = a.idx + (int64_t) list*a.n_kv_max;
    const half  * mrow  = a.mask + (i3 % a.m_ne3)*a.m_nb3 + iq_mask*a.m_nb1;
    const int     count = a.counts[list];

    int e_begin, e_end;
    fasp_split_range(count, s, S, e_begin, e_end);
    if (a.fault == 3 && e_end > e_begin) {
        --e_end;
    }
    if (!valid) {
        e_end = e_begin;
    }

    const int  ikv = i3 % a.kv_ne3;
    const char * Kg = a.K + (int64_t) g*a.k_nb2 + ikv*a.k_nb3;
    const char * Vg = a.V + (int64_t) g*a.v_nb2 + ikv*a.v_nb3;

    for (int e0 = e_begin; e0 < e_end; e0 += FASP_TILE) {
        // one entry per lane: its cell and its mask value in base 2; entries past the split's end score -inf
        int   cell = -1;
        float mk   = -INFINITY;
        if (e0 + lane < e_end) {
            cell = idx[e0 + lane];
            mk   = a.fault == 2 ? 0.0f : __half2float(mrow[cell])*FASP_LOG2_E;
        }
        cell_sm[wy][lane] = cell;
        __syncwarp();

        // scores: s[j8] of lane l belongs to key 8*j8 + l/4 of the tile and head l%4
        float sc[FASP_TILE/FASP_KPR];
#pragma unroll
        for (int j8 = 0; j8 < FASP_TILE/FASP_KPR; ++j8) {
            float v[32];
#pragma unroll
            for (int j = 0; j < FASP_KPR; ++j) {
                const int c = cell_sm[wy][FASP_KPR*j8 + j];
                float kf[DPL];
                if (c >= 0) {
                    fasp_load_row<type, DPL>(Kg + (int64_t) c*a.k_nb1, lane, kf);
                } else {
#pragma unroll
                    for (int d = 0; d < DPL; ++d) {
                        kf[d] = 0.0f;
                    }
                }
#pragma unroll
                for (int h = 0; h < FASP_HPW; ++h) {
                    float t = q[h][0]*kf[0];
#pragma unroll
                    for (int d = 1; d < DPL; ++d) {
                        t = fmaf(q[h][d], kf[d], t);
                    }
                    v[j*FASP_HPW + h] = t;
                }
            }
            sc[j8] = fasp_transpose_reduce(v) +
                __shfl_sync(0xFFFFFFFF, mk, FASP_KPR*j8 + fasp_tr_key(lane), FASP_WARP);
        }

        // online softmax of head hl over the tile: the 8 lanes l with l%4 == hl share a head
        float m_tile = sc[0];
#pragma unroll
        for (int j8 = 1; j8 < FASP_TILE/FASP_KPR; ++j8) {
            m_tile = fmaxf(m_tile, sc[j8]);
        }
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFF, m_tile,  4, FASP_WARP));
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFF, m_tile,  8, FASP_WARP));
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFF, m_tile, 16, FASP_WARP));
        const float M_new     = fmaxf(M, m_tile);
        const float scale_old = exp2f(M - M_new);
        M = M_new;

        float l_tile = 0.0f;
#pragma unroll
        for (int j8 = 0; j8 < FASP_TILE/FASP_KPR; ++j8) {
            const float p = exp2f(sc[j8] - M);
            l_tile += p;
            p_sm[wy][(FASP_KPR*j8 + fasp_tr_key(lane))*FASP_HPW + hl] = p;
        }
        l_tile += __shfl_xor_sync(0xFFFFFFFF, l_tile,  4, FASP_WARP);
        l_tile += __shfl_xor_sync(0xFFFFFFFF, l_tile,  8, FASP_WARP);
        l_tile += __shfl_xor_sync(0xFFFFFFFF, l_tile, 16, FASP_WARP);
        L = L*scale_old + l_tile;

        // the accumulators of each head, rescaled by their head's factor (held by lane h)
#pragma unroll
        for (int h = 0; h < FASP_HPW; ++h) {
            const float f = __shfl_sync(0xFFFFFFFF, scale_old, h, FASP_WARP);
#pragma unroll
            for (int d = 0; d < DPL; ++d) {
                acc[h][d] *= f;
            }
        }
        __syncwarp();

        // P*V: each V row once for the 4 heads; an empty entry ends the tile (entries are filled from lane 0)
#pragma unroll 4
        for (int kk = 0; kk < FASP_TILE; ++kk) {
            const int c = cell_sm[wy][kk];
            if (c < 0) {
                break;
            }
            float vf[DPL];
            fasp_load_row<type, DPL>(Vg + (int64_t) c*a.v_nb1, lane, vf);
            const float4 p4 = *(const float4 *) &p_sm[wy][kk*FASP_HPW];
            const float p[FASP_HPW] = { p4.x, p4.y, p4.z, p4.w };
#pragma unroll
            for (int h = 0; h < FASP_HPW; ++h) {
#pragma unroll
                for (int d = 0; d < DPL; ++d) {
                    acc[h][d] = fmaf(p[h], vf[d], acc[h][d]);
                }
            }
        }
        __syncwarp();
    }

    // this warp's partial outputs, maxima and sums; the union kernel counts its blocks per list column (list, KV head)
    const int     n_cols       = uni ? gridDim.y*a.n_q*(gridDim.z/n_lists) : gridDim.y*gridDim.z;
    const int     col          = z*gridDim.y + g;
    const int     ctr          = uni ? blockIdx.z*gridDim.y + g : col;
    const int64_t split_stride = (int64_t) n_cols*a.G*PART;
    float * part = a.part + (int64_t) s*split_stride + ((int64_t) col*a.G + FASP_HPW*w)*PART;
    if (valid) {
#pragma unroll
        for (int h = 0; h < FASP_HPW; ++h) {
#pragma unroll
            for (int j = 0; j < DPL; j += 4) {
                *(float4 *) (part + h*PART + DPL*lane + j) = make_float4(acc[h][j], acc[h][j + 1], acc[h][j + 2], acc[h][j + 3]);
            }
        }
        if (lane < FASP_HPW) {
            part[lane*PART + D + 0] = M;
            part[lane*PART + D + 1] = L;
        }
    }

    // the last block of this column merges every split's partials
    __threadfence();
    __syncthreads();
    if (lane == 0 && wy == 0) {
        last_block = atomicAdd(&a.counter[ctr], 1u) == (unsigned) S - 1;
    }
    __syncthreads();
    if (!last_block) {
        return;
    }
    __threadfence();

    // a warp past the call's last token merges nothing but still meets the barrier below
    if (valid) {
#pragma unroll
        for (int h = 0; h < FASP_HPW; ++h) {
            const int hq = hq0 + h;
            float o[DPL];
            float m_all, l;
            fasp_merge_lane<DPL>(a.part + ((int64_t) col*a.G + FASP_HPW*w + h)*PART, split_stride, S, D, lane, o, m_all, l);
            float f = 1.0f;
            if (a.sinks) {
                f = fasp_add_sink(a.sinks[hq]*FASP_LOG2_E, m_all, l);
            }
            const float inv = l > 0.0f ? (a.fault == 1 ? 1.01f : 1.0f)*f/l : 0.0f;
            float * out = a.dst + ((int64_t) z*a.n_head + hq)*D + DPL*lane;
#pragma unroll
            for (int j = 0; j < DPL; j += 4) {
                *(float4 *) (out + j) = make_float4(o[j]*inv, o[j + 1]*inv, o[j + 2]*inv, o[j + 3]*inv);
            }
        }
    }

    __syncthreads();
    if (lane == 0 && wy == 0) {
        a.counter[ctr] = 0;
    }
}

// Launches the kernel for head size D and K/V type `type` (block (32, G/4), grid (S, n_head_kv, n_q*ne3)).
template <int D, int type>
static void fasp_launch(const fasp_args & a, const int S, const int n_head_kv, const int n_z, cudaStream_t stream) {
    const dim3 grid(S, n_head_kv, n_z);
    const dim3 block(FASP_WARP, a.G/FASP_HPW, 1);
    flash_attn_ext_sparse_pascal<D, type, false><<<grid, block, 0, stream>>>(a);
}

// Launches the union kernel for tpb query tokens a list (block (32, G/4 x tpb), grid (S, n_head_kv, ceil(n_q/tpb)*ne3));
// a.idx and a.counts hold the union lists of a.n_kv_max entries, ceil(n_q/tpb) per mask sequence.
template <int D, int type>
static void fasp_launch_union(const fasp_args & a, const int tpb, const int S, const int n_head_kv, const int ne3,
        cudaStream_t stream) {
    const dim3 grid(S, n_head_kv, (a.n_q + tpb - 1)/tpb*ne3);
    const dim3 block(FASP_WARP, a.G/FASP_HPW*tpb, 1);
    flash_attn_ext_sparse_pascal<D, type, true><<<grid, block, 0, stream>>>(a);
}

#endif // defined(__CUDACC__)
