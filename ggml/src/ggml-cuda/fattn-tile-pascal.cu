#include "fattn-tile-pascal.cuh"
#include "fattn-common.cuh"

#include <cmath>
#include <cstdlib>
#include <type_traits>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

constexpr int FTP_NCOLS2   = 8; // query heads per KV head in one block (the GQA group)
constexpr int FTP_NWARPS   = 4;
constexpr int FTP_NTHREADS = FTP_NWARPS*WARP_SIZE;
#ifndef FTP_MIN_BLOCKS
#define FTP_MIN_BLOCKS 3 // blocks per SM the compiler keeps registers for (__launch_bounds__)
#endif

// T: keys per softmax tile; DC: head dimensions per staged K chunk; TV: keys per staged V chunk.
template <int D> struct ftp_config;

template <> struct ftp_config<128> {
    static constexpr int ncols1 = 4;  // tokens per block: 32 columns
    static constexpr int T      = 64;
    static constexpr int DC     = 16;
    static constexpr int TV     = 8;
    static constexpr int PPAD   = 4;  // floats of padding per probability row
};

template <> struct ftp_config<256> {
    static constexpr int ncols1 = 2;  // tokens per block: 16 columns
    static constexpr int T      = 128;
    static constexpr int DC     = 8;
    static constexpr int TV     = 4;
    static constexpr int PPAD   = 0;
};

template <int D> struct ftp_shape {
    using cfg = ftp_config<D>;
    static constexpr int ncols1 = cfg::ncols1;
    static constexpr int T      = cfg::T;
    static constexpr int DC     = cfg::DC;
    static constexpr int TV     = cfg::TV;
    static constexpr int C      = ncols1*FTP_NCOLS2; // columns (token x head) per block
    static constexpr int CW     = C/FTP_NWARPS;      // columns per warp
    static constexpr int NCGW   = CW/4;              // groups of 4 columns per warp
    static constexpr int NKG    = WARP_SIZE/NCGW;    // lanes per column group: key groups (scores), dimension groups (output)
    static constexpr int KA     = T/NKG;             // keys per lane
    static constexpr int OQ     = D/(4*NKG);         // float4 output quads per lane and column
    static constexpr int KS     = DC + 4;            // row stride of a staged K chunk (floats)
    static constexpr int VS     = D + 4;             // row stride of a staged V chunk
    static constexpr int PS     = C + cfg::PPAD;     // row stride of the probabilities
    static constexpr int NKC    = D/DC;              // K chunks per tile
    static constexpr int NVC    = T/TV;              // V chunks per tile
    static constexpr int BUF    = T*KS > TV*VS ? T*KS : TV*VS;

    static_assert(CW % 4 == 0 && FTP_NCOLS2 % 4 == 0, "a lane's 4 columns must belong to one token");
    static_assert(T % NKG == 0 && D % (4*NKG) == 0, "bad lane split");
    static_assert(T*DC/8 == FTP_NTHREADS, "each thread stages 8 dimensions of one key per K chunk");
    static_assert(TV*D/8 == FTP_NTHREADS, "each thread stages 8 dimensions of one key per V chunk");
    static_assert(DC % 8 == 0 && T % TV == 0, "bad chunks");
    static_assert(4*(C*D + BUF + T*PS) <= 32*1024, "3 blocks per SM need at most 32 KiB of shared memory per block");
};

// 8 consecutive head dimensions of one key, as loaded from global memory, before conversion to FP32.
template <ggml_type type> struct ftp_piece;

template <> struct ftp_piece<GGML_TYPE_F16> {
    uint4 h; // 8 halves

    static __device__ __forceinline__ void load(ftp_piece & p, const char * row, const int dim0) {
        p.h = __ldg((const uint4 *) (row + 2*dim0));
    }
    static __device__ __forceinline__ void store(const ftp_piece & p, float * dst) {
        const float2 a = __half22float2(*(const half2 *) &p.h.x);
        const float2 b = __half22float2(*(const half2 *) &p.h.y);
        const float2 c = __half22float2(*(const half2 *) &p.h.z);
        const float2 d = __half22float2(*(const half2 *) &p.h.w);
        *(float4 *) (dst + 0) = make_float4(a.x, a.y, b.x, b.y);
        *(float4 *) (dst + 4) = make_float4(c.x, c.y, d.x, d.y);
    }
};

// Q8_0: blocks of 34 bytes (an F16 scale, 32 int8), so a row's fields are only 2-byte aligned; rows themselves are
// 4-byte aligned. The 8 quants are read as aligned 32-bit words and shifted into place.
template <> struct ftp_piece<GGML_TYPE_Q8_0> {
    int   q0, q1; // 8 int8
    float d;

    static __device__ __forceinline__ void load(ftp_piece & p, const char * row, const int dim0) {
        const int b = dim0/QK8_0;
        const char * pq = row + b*int(sizeof(block_q8_0)) + int(sizeof(ggml_half)) + dim0 % QK8_0;
        const int    oq = (int) ((uintptr_t) pq & 2);
        const int  * wq = (const int *) (pq - oq);
        const int w0 = __ldg(wq + 0);
        const int w1 = __ldg(wq + 1);
        if (oq) {
            const int w2 = __ldg(wq + 2);
            p.q0 = __funnelshift_r(w0, w1, 16);
            p.q1 = __funnelshift_r(w1, w2, 16);
        } else {
            p.q0 = w0;
            p.q1 = w1;
        }
        const char * pd = row + b*int(sizeof(block_q8_0));
        const int    od = (int) ((uintptr_t) pd & 2);
        const int    wd = __ldg((const int *) (pd - od));
        p.d = __half2float(__ushort_as_half((unsigned short) (wd >> (8*od))));
    }
    static __device__ __forceinline__ void store(const ftp_piece & p, float * dst) {
        *(float4 *) (dst + 0) = make_float4(
            float((int8_t) (p.q0 >>  0))*p.d, float((int8_t) (p.q0 >>  8))*p.d,
            float((int8_t) (p.q0 >> 16))*p.d, float((int8_t) (p.q0 >> 24))*p.d);
        *(float4 *) (dst + 4) = make_float4(
            float((int8_t) (p.q1 >>  0))*p.d, float((int8_t) (p.q1 >>  8))*p.d,
            float((int8_t) (p.q1 >> 16))*p.d, float((int8_t) (p.q1 >> 24))*p.d);
    }
};

// The tile kernel's pieces: load only issues the global loads, store decodes. The kernel loads a chunk ahead and stores
// it after the next barrier, so anything in load that consumes the loaded words waits out the whole memory latency on
// every chunk. ftp_piece<Q8_0>::load shifts the words into place right away (first use 5-11 instructions after the
// load in the SASS, against 150-400 for F16), which made Q8_0 prompts 32 % slower than F16 at 128 rows (rig
// 2026-11-20: 77.2 vs 58.5 ms at 72k keys; 58.0 with no K/V loads at all). (2026-11-20, #38 prompt path)
// Used with GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 (and always with KDP); F16 is the same either way.
template <ggml_type type> struct ftp_tpiece : ftp_piece<type> {};

template <> struct ftp_tpiece<GGML_TYPE_Q8_0> {
    int w0, w1, w2; // the aligned words holding the 8 quants
    int wd;         // the aligned word holding the scale
    int sh;         // bit 0: the quants start 2 bytes into w0; bit 1: the scale is the word's upper half

    static __device__ __forceinline__ void load(ftp_tpiece & p, const char * row, const int dim0) {
        const int b = dim0/QK8_0;
        const char * pq = row + b*int(sizeof(block_q8_0)) + int(sizeof(ggml_half)) + dim0 % QK8_0;
        const int    oq = (int) ((uintptr_t) pq & 2);
        const int  * wq = (const int *) (pq - oq);
        p.w0 = __ldg(wq + 0);
        p.w1 = __ldg(wq + 1);
        p.w2 = oq ? __ldg(wq + 2) : 0;
        const char * pd = row + b*int(sizeof(block_q8_0));
        const int    od = (int) ((uintptr_t) pd & 2);
        p.wd = __ldg((const int *) (pd - od));
        p.sh = (oq ? 1 : 0) | (od ? 2 : 0);
    }
    static __device__ __forceinline__ void store(const ftp_tpiece & p, float * dst) {
        ftp_piece<GGML_TYPE_Q8_0> q;
        q.q0 = (p.sh & 1) ? __funnelshift_r(p.w0, p.w1, 16) : p.w0;
        q.q1 = (p.sh & 1) ? __funnelshift_r(p.w1, p.w2, 16) : p.w1;
        q.d  = __half2float(__ushort_as_half((unsigned short) ((p.sh & 2) ? p.wd >> 16 : p.wd)));
        ftp_piece<GGML_TYPE_Q8_0>::store(q, dst);
    }
};

// kdp (GGML_CUDA_FATTN_TILE_PASCAL_KDP=1): Q8_0 K at head 256 kept as int8 in shared memory and scored with dp4a
// against a Q quantized to int8 per 32-dimension block (as the CPU reference and the stock vector kernel do). A chunk
// is 16 dimensions (half a Q8_0 block, one scale), so a tile takes 16 K chunks instead of 32, and each score step
// reads 16 bytes of K per key instead of 32 per 8 dimensions. On a GTX 1070 the FP32 scores of the F16 path cost
// ~15 ms of 58.5 at 72k keys and 128 rows; with kdp the Q8_0 scores cost ~2.5 ms of 47.
constexpr int FTP_KDC = 16; // dimensions per kdp K chunk

struct ftp_kq16 {
    int w0, w1, w2, w3, w4; // the aligned words holding the 16 quants
    int wd;                 // the aligned word holding the scale
    int sh;                 // bit 0: the quants start 2 bytes into w0; bit 1: the scale is the word's upper half

    // chunk g: dimensions 16g..16g+15 = block g/2, quants 16*(g%2)..+15
    static __device__ __forceinline__ void load(ftp_kq16 & p, const char * row, const int g) {
        const int b = g/2;
        const char * pq = row + b*int(sizeof(block_q8_0)) + int(sizeof(ggml_half)) + 16*(g % 2);
        const int    oq = (int) ((uintptr_t) pq & 2);
        const int  * wq = (const int *) (pq - oq);
        p.w0 = __ldg(wq + 0);
        p.w1 = __ldg(wq + 1);
        p.w2 = __ldg(wq + 2);
        p.w3 = __ldg(wq + 3);
        p.w4 = oq ? __ldg(wq + 4) : 0;
        const char * pd = row + b*int(sizeof(block_q8_0));
        const int    od = (int) ((uintptr_t) pd & 2);
        p.wd = __ldg((const int *) (pd - od));
        p.sh = (oq ? 1 : 0) | (od ? 2 : 0);
    }
    static __device__ __forceinline__ void store(const ftp_kq16 & p, int * dst_q, float * dst_d) {
        int4 q;
        if (p.sh & 1) {
            q = make_int4(__funnelshift_r(p.w0, p.w1, 16), __funnelshift_r(p.w1, p.w2, 16),
                          __funnelshift_r(p.w2, p.w3, 16), __funnelshift_r(p.w3, p.w4, 16));
        } else {
            q = make_int4(p.w0, p.w1, p.w2, p.w3);
        }
        *(int4 *) dst_q = q;
        *dst_d = __half2float(__ushort_as_half((unsigned short) ((p.sh & 2) ? p.wd >> 16 : p.wd)));
    }
};

#define FTP_FAULT_SCALE_DEFAULT 1.05f

__constant__ int   ftp_fault_dev       = 0;
__constant__ float ftp_fault_scale_dev = FTP_FAULT_SCALE_DEFAULT; // the output's factor under fault 1

// split (head 128): the scores are computed by lanes (16 key groups x the 2 halves of the head dimension), each with 4
// keys x the warp's 8 columns over its half (128 FMAs per 12 float4 loads instead of 64 per 8), and lane l and l ^ 16
// then swap the partial sums of each other's columns, which leaves the same layout as without the split.
template <int D, ggml_type type_K, ggml_type type_V, bool split, bool lds, bool kdp = false>
__launch_bounds__(FTP_NTHREADS, FTP_MIN_BLOCKS)
static __global__ void flash_attn_tile_pascal(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int64_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    // max_bias and logit_softcap are 0 and there are no kv rows (checked on the host); the other unused sizes follow
    // from Q, K and the mask
    GGML_UNUSED_VARS(KV_rows_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31, ne32, nb32);

    using S = ftp_shape<D>;
    using piece_K = std::conditional_t<lds, ftp_tpiece<type_K>, ftp_piece<type_K>>;
    using piece_V = std::conditional_t<lds, ftp_tpiece<type_V>, ftp_piece<type_V>>;
    constexpr int ncols1 = S::ncols1;
    constexpr int T      = S::T;
    constexpr int NKG    = S::NKG;
    constexpr int KA     = S::KA;
    constexpr int OQ     = S::OQ;
    static_assert(!split || (S::NKG == 16 && S::CW == 8 && S::DC == 16), "the split scores need 16 key groups, 8 columns per warp and 16-dimension chunks");

    const char  * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char  * GGML_CUDA_RESTRICT K        = K_ptr;
    const char  * GGML_CUDA_RESTRICT V        = V_ptr;
    const float * GGML_CUDA_RESTRICT sinks    = (const float *) sinks_ptr;
    const int   * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float       * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2      * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    const int fault = ftp_fault_dev;

    const int lane = threadIdx.x;
    const int tid  = threadIdx.y*WARP_SIZE + lane;
    const int kg   = lane % NKG;                         // key group (scores) and dimension group (output)
    const int col0 = threadIdx.y*S::CW + (lane/NKG)*4;  // the first of this lane's 4 columns
    const int jtok = col0 / FTP_NCOLS2;                  // the lane's token within the block
    const int hcol = col0 % FTP_NCOLS2;                  // the head of col0 within the GQA group

    const int col_Q_0   = blockIdx.x*ncols1;
    const int sequence  = blockIdx.z / (ne02/FTP_NCOLS2);
    const int head0     = blockIdx.z*FTP_NCOLS2 - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * K_h   = K + nb13*sequence + int64_t(nb12)*(head0/gqa_ratio);
    const char * V_h   = V + nb23*sequence + int64_t(nb22)*(head0/gqa_ratio);
    const half * maskh = (const half *) (mask_ptr + nb33*(sequence % ne33))
                       + int64_t(fastmodulo(col_Q_0 + jtok, ne01))*(nb31/sizeof(half));

    __shared__ __align__(16) float Q_s[S::C*D];
    __shared__ __align__(16) float KV_s[S::BUF];
    __shared__ __align__(16) float P_s[T*S::PS];

    static_assert(!kdp || (type_K == GGML_TYPE_Q8_0 && D == 256 && !split && T == FTP_NTHREADS &&
        S::C*D/32 == FTP_NTHREADS && S::CW == 4 && T*(FTP_KDC/4 + 1) <= S::BUF), "kdp: Q8_0 K, head 256, one thread per key");
    constexpr int NKCK = kdp ? D/FTP_KDC : S::NKC; // K chunks per tile
    [[maybe_unused]] int   * const Qi_s = (int *) Q_s;                    // kdp: the query as int8, [column][D/4]
    [[maybe_unused]] float * const dQ_s = Q_s + S::C*D/4;                 // and its scales, [column][D/32]
    [[maybe_unused]] int   * const Ki_s = (int *) KV_s;                   // kdp: a K chunk as int8, [key][FTP_KDC/4]
    [[maybe_unused]] float * const dK_s = KV_s + T*(FTP_KDC/4);           // and its scales, [key]

    ggml_cuda_pdl_sync();

    // the query of the block's columns, scaled as in the stock kernel
    if constexpr (kdp) {
        // one 32-dimension block of one column per thread, quantized as Q8_0 is
        const float * Q_f = (const float *) (Q + int64_t(nb03)*sequence + int64_t(nb02)*head0);
        const int jc = tid / (D/32);
        const int b  = tid % (D/32);
        const int j  = jc / FTP_NCOLS2;
        const int c  = jc % FTP_NCOLS2;
        const float * src = Q_f + c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float)) + 32*b;
        float x[32];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 t = *(const float4 *) (src + 4*i);
            x[4*i + 0] = t.x*scale;
            x[4*i + 1] = t.y*scale;
            x[4*i + 2] = t.z*scale;
            x[4*i + 3] = t.w*scale;
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, fabsf(x[i]));
        }
        const float d  = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q0 = __float2int_rn(x[4*i + 0]*id);
            const int q1 = __float2int_rn(x[4*i + 1]*id);
            const int q2 = __float2int_rn(x[4*i + 2]*id);
            const int q3 = __float2int_rn(x[4*i + 3]*id);
            Qi_s[jc*(D/4) + 8*b + i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | (int) ((unsigned) q3 << 24);
        }
        dQ_s[jc*(D/32) + b] = d;
    } else {
        const float * Q_f = (const float *) (Q + int64_t(nb03)*sequence + int64_t(nb02)*head0);
#pragma unroll
        for (int i0 = 0; i0 < S::C*D/4; i0 += FTP_NTHREADS) {
            const int i  = i0 + tid;
            const int jc = i / (D/4);
            const int d4 = i % (D/4);
            const int j  = jc / FTP_NCOLS2;
            const int c  = jc % FTP_NCOLS2;
            float4 t = *(const float4 *) (Q_f + c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float)) + 4*d4);
            t.x *= scale;
            t.y *= scale;
            t.z *= scale;
            t.w *= scale;
            *(float4 *) (Q_s + jc*D + 4*d4) = t;
        }
    }

    // chunk g of the tile at key k0: g < NKC is K dimensions g*DC..g*DC+DC-1 of all T keys,
    // g >= NKC is V rows (g - NKC)*TV..+TV-1. Each thread loads and stores 8 dimensions of one key.
    // split: chunk columns 0-7 are dimensions 8g..8g+7, columns 8-15 are D/2 + 8g..D/2 + 8g + 7
    [[maybe_unused]] piece_K pk; // not with kdp
    piece_V pv;
    const char * K_thr  = K_h + int64_t(tid / (S::DC/8))*nb11;  // the thread's K row at key 0 of a tile
    const char * V_thr  = V_h + int64_t(tid / (D/8))*nb21;      // and V row
    [[maybe_unused]] const int K_dim0 = split ? (tid % (S::DC/8))*(D/2) : (tid % (S::DC/8))*8;
    const int    V_dim0 = (tid % (D/8))*8;
    [[maybe_unused]] float * const K_st = KV_s + (tid / (S::DC/8))*S::KS + (tid % (S::DC/8))*8;
    float * const V_st  = KV_s + (tid / (D/8))*S::VS + V_dim0;
    [[maybe_unused]] ftp_kq16 kq; // kdp: the thread's key, 16 dimensions
    auto load_K = [&](const char * K_tile, const int g) { // K_tile: K_thr advanced to the tile
        if constexpr (kdp) {
            ftp_kq16::load(kq, K_tile, g);
        } else {
            piece_K::load(pk, K_tile, K_dim0 + g*(split ? S::DC/2 : S::DC));
        }
    };
    auto store_K = [&](const int g) {
        if constexpr (kdp) {
            ftp_kq16::store(kq, Ki_s + tid*(FTP_KDC/4), dK_s + tid);
        } else {
            piece_K::store(pk, K_st);
        }
    };
    auto load_V = [&](const char * V_tile, const int v) { // V_tile: V_thr advanced to the tile
        piece_V::load(pv, V_tile + v*(S::TV*nb21), V_dim0);
    };
    auto store_V = [&]() {
        piece_V::store(pv, V_st);
    };

    half mask_pf[KA];
    auto load_mask = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < KA; ++i) {
            mask_pf[i] = maskh[k0 + kg + NKG*i];
        }
    };

    float4 O[4][OQ];
    float  KQ_max[4];
    float  KQ_sum[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int q = 0; q < OQ; ++q) {
            O[j][q] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    const int k_step    = gridDim.y*T;

    int k0 = blockIdx.y*T;
    if (k0 < k_VKQ_max) {
        load_K(K_thr + int64_t(k0)*nb11, 0);
        load_mask(k0);
    }

    __syncthreads(); // Q_s

#pragma unroll 1
    while (k0 < k_VKQ_max) {
        float mask_f[KA];
        bool  any_key = false;
#pragma unroll
        for (int i = 0; i < KA; ++i) {
            mask_f[i] = __half2float(mask_pf[i]);
            any_key   = any_key || mask_f[i] != -INFINITY;
        }

        // skip a tile that no token of the block can see; the barrier also ends the previous tile's reads of KV_s
        const bool drop = fault == 3 && k0 + k_step >= k_VKQ_max;
        if (!__syncthreads_or(any_key) || drop) {
            k0 += k_step;
            if (k0 < k_VKQ_max) {
                load_K(K_thr + int64_t(k0)*nb11, 0);
                load_mask(k0);
            }
            continue;
        }
        const char * K_tile = K_thr + int64_t(k0)*nb11;
        const char * V_tile = V_thr + int64_t(k0)*nb21;

        // scores of KA keys (kg + NKG*i) x the lane's 4 columns, summed over the head dimension in order
        // (split: over each half in order, then the halves added)
        constexpr int CQ = split ? 8 : 4; // columns per lane while the scores are summed
        float KQ[KA][4];
        float KQ_part[KA][CQ];
#pragma unroll
        for (int i = 0; i < KA; ++i) {
#pragma unroll
            for (int c = 0; c < CQ; ++c) {
                KQ_part[i][c] = 0.0f;
            }
        }
        [[maybe_unused]] const int cq0 = split ? threadIdx.y*S::CW : col0; // not with kdp: the first of the lane's CQ columns
        [[maybe_unused]] const int dq0 = split ? (lane/16)*(D/2) : 0;      // the lane's half of the head dimension
        [[maybe_unused]] const int kq0 = split ? (lane/16)*(S::DC/2) : 0;  // and its columns in a K chunk

#pragma unroll 1
        for (int g = 0; g < NKCK; ++g) {
            if (g > 0) {
                __syncthreads();
            }
            store_K(g);
            __syncthreads();
            if (g + 1 < NKCK) {
                load_K(K_tile, g + 1);
            } else {
                load_V(V_tile, 0);
            }

            if constexpr (kdp) {
                int4  Q_i[4];
                float Q_d[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    Q_i[c] = *(const int4 *) (Qi_s + (col0 + c)*(D/4) + g*(FTP_KDC/4));
                    Q_d[c] = dQ_s[(col0 + c)*(D/32) + g/2];
                }
#pragma unroll
                for (int i = 0; i < KA; ++i) {
                    const int4  K_i = *(const int4 *) (Ki_s + (kg + NKG*i)*(FTP_KDC/4));
                    const float K_d = dK_s[kg + NKG*i];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        int s = ggml_cuda_dp4a(K_i.x, Q_i[c].x, 0);
                        s = ggml_cuda_dp4a(K_i.y, Q_i[c].y, s);
                        s = ggml_cuda_dp4a(K_i.z, Q_i[c].z, s);
                        s = ggml_cuda_dp4a(K_i.w, Q_i[c].w, s);
                        KQ_part[i][c] = fmaf(float(s), K_d*Q_d[c], KQ_part[i][c]);
                    }
                }
            } else {
                constexpr int DQ = split ? S::DC/2 : S::DC; // dimensions of the lane per chunk
#pragma unroll
                for (int d4 = 0; d4 < DQ/4; ++d4) {
                    float4 Q_k[CQ];
#pragma unroll
                    for (int c = 0; c < CQ; ++c) {
                        Q_k[c] = *(const float4 *) (Q_s + (cq0 + c)*D + dq0 + g*DQ + 4*d4);
                    }
#pragma unroll
                    for (int i = 0; i < KA; ++i) {
                        const float4 K_k = *(const float4 *) (KV_s + (kg + NKG*i)*S::KS + kq0 + 4*d4);
#pragma unroll
                        for (int c = 0; c < CQ; ++c) {
                            KQ_part[i][c] = fmaf(K_k.x, Q_k[c].x, KQ_part[i][c]);
                            KQ_part[i][c] = fmaf(K_k.y, Q_k[c].y, KQ_part[i][c]);
                            KQ_part[i][c] = fmaf(K_k.z, Q_k[c].z, KQ_part[i][c]);
                            KQ_part[i][c] = fmaf(K_k.w, Q_k[c].w, KQ_part[i][c]);
                        }
                    }
                }
            }
        }

        if constexpr (split) {
            // lane l keeps columns 4*(l/16) .. +3 and sends its half's sums of the other 4 to lane l ^ 16
            const bool upper = lane & 16;
#pragma unroll
            for (int i = 0; i < KA; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float keep = upper ? KQ_part[i][4 + j] : KQ_part[i][j];
                    const float send = upper ? KQ_part[i][j] : KQ_part[i][4 + j];
                    KQ[i][j] = keep + __shfl_xor_sync(0xFFFFFFFF, send, 16, WARP_SIZE);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < KA; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    KQ[i][j] = KQ_part[i][j];
                }
            }
        }

        // mask, running maximum over the NKG lanes of each column, probabilities, rescale of the output
        float KQ_max_new[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            KQ_max_new[j] = KQ_max[j];
#pragma unroll
            for (int i = 0; i < KA; ++i) {
                KQ[i][j] += mask_f[i];
                KQ_max_new[j] = fmaxf(KQ_max_new[j], KQ[i][j] + FATTN_KQ_MAX_OFFSET);
            }
#pragma unroll
            for (int offset = NKG/2; offset > 0; offset >>= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            float KQ_sum_add = 0.0f;
#pragma unroll
            for (int i = 0; i < KA; ++i) {
                KQ[i][j] = expf(KQ[i][j] - KQ_max[j]);
                KQ_sum_add += KQ[i][j];
            }
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_sum_add;

#pragma unroll
            for (int q = 0; q < OQ; ++q) {
                O[j][q].x *= KQ_max_scale;
                O[j][q].y *= KQ_max_scale;
                O[j][q].z *= KQ_max_scale;
                O[j][q].w *= KQ_max_scale;
            }
        }

        // the warp's probabilities stay in its own columns of P_s
#pragma unroll
        for (int i = 0; i < KA; ++i) {
            *(float4 *) (P_s + (kg + NKG*i)*S::PS + col0) = make_float4(KQ[i][0], KQ[i][1], KQ[i][2], KQ[i][3]);
        }
        __syncwarp();

        // output of the lane's 4 columns x dimensions 4*kg + 4*NKG*q .. +3, summed over the keys in order
#pragma unroll 1
        for (int v = 0; v < S::NVC; ++v) {
            __syncthreads();
            store_V();
            __syncthreads();
            if (v + 1 < S::NVC) {
                load_V(V_tile, v + 1);
            } else if (k0 + k_step < k_VKQ_max) {
                load_K(K_tile + int64_t(k_step)*nb11, 0);
                load_mask(k0 + k_step);
            }

#pragma unroll
            for (int kk = 0; kk < S::TV; ++kk) {
                const float4 P_k = *(const float4 *) (P_s + (v*S::TV + kk)*S::PS + col0);
#pragma unroll
                for (int q = 0; q < OQ; ++q) {
                    const float4 V_k = *(const float4 *) (KV_s + kk*S::VS + 4*kg + 4*NKG*q);
                    O[0][q].x = fmaf(V_k.x, P_k.x, O[0][q].x);
                    O[0][q].y = fmaf(V_k.y, P_k.x, O[0][q].y);
                    O[0][q].z = fmaf(V_k.z, P_k.x, O[0][q].z);
                    O[0][q].w = fmaf(V_k.w, P_k.x, O[0][q].w);
                    O[1][q].x = fmaf(V_k.x, P_k.y, O[1][q].x);
                    O[1][q].y = fmaf(V_k.y, P_k.y, O[1][q].y);
                    O[1][q].z = fmaf(V_k.z, P_k.y, O[1][q].z);
                    O[1][q].w = fmaf(V_k.w, P_k.y, O[1][q].w);
                    O[2][q].x = fmaf(V_k.x, P_k.z, O[2][q].x);
                    O[2][q].y = fmaf(V_k.y, P_k.z, O[2][q].y);
                    O[2][q].z = fmaf(V_k.z, P_k.z, O[2][q].z);
                    O[2][q].w = fmaf(V_k.w, P_k.z, O[2][q].w);
                    O[3][q].x = fmaf(V_k.x, P_k.w, O[3][q].x);
                    O[3][q].y = fmaf(V_k.y, P_k.w, O[3][q].y);
                    O[3][q].z = fmaf(V_k.z, P_k.w, O[3][q].z);
                    O[3][q].w = fmaf(V_k.w, P_k.w, O[3][q].w);
                }
            }
        }

        k0 += k_step;
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int offset = NKG/2; offset > 0; offset >>= 1) {
            KQ_sum[j] += __shfl_xor_sync(0xFFFFFFFF, KQ_sum[j], offset, WARP_SIZE);
        }
    }

    // attention sinks, in the first KV split only, as in the stock kernels
    if (sinks && blockIdx.y == 0 && fault != 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float sink = sinks[head0 + hcol + j];

            const float KQ_max_new_j = fmaxf(KQ_max[j], sink);
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new_j);
            KQ_max[j] = KQ_max_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + expf(sink - KQ_max[j]);
#pragma unroll
            for (int q = 0; q < OQ; ++q) {
                O[j][q].x *= KQ_max_scale;
                O[j][q].y *= KQ_max_scale;
                O[j][q].z *= KQ_max_scale;
                O[j][q].w *= KQ_max_scale;
            }
        }
    }

    if (col_Q_0 + jtok >= int(ne01.z)) {
        return;
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float out_scale = gridDim.y == 1 ? 1.0f/KQ_sum[j] : 1.0f;
        if (fault == 1) {
            out_scale *= ftp_fault_scale_dev;
        }

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + jtok)*ne02 + head0 + hcol + j)*gridDim.y + blockIdx.y;

#pragma unroll
        for (int q = 0; q < OQ; ++q) {
            const float4 o = make_float4(O[j][q].x*out_scale, O[j][q].y*out_scale, O[j][q].z*out_scale, O[j][q].w*out_scale);
            *(float4 *) (dst + int64_t(j_dst_unrolled)*D + 4*kg + 4*NKG*q) = o;
        }

        if (gridDim.y != 1 && kg == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[j], KQ_sum[j]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// The one-token decode kernel (GGML_CUDA_FATTN_DEC_PASCAL=N: up to N query rows), head 256, Q8_0 K and V, the 8-head GQA group per block.
// The tile kernel above spends ~1.3 ms per 104k-key layer on its own barriers and staging even with no K/V loads, so
// here every warp walks its own 32-key slices with no block barrier in the loop: each lane owns 8 head dimensions
// (lane*8..+7, inside Q8_0 block lane/4), Q is quantized to int8 per 32 dimensions as the stock vector kernel does for
// quantized K, the scores are dp4a sums reduced over the warp by a transposing butterfly that leaves lane l the score of
// head l/4, the probabilities go through a per-warp shared buffer, and V accumulates in registers. K and V rows are read
// straight from global memory, 8 keys per batch, the next batch loaded before the current one is used.
constexpr int FDP_NWARPS = 4;
constexpr int FDP_KW     = 32;               // keys per warp per tile
constexpr int FDP_T      = FDP_NWARPS*FDP_KW; // keys per block per tile
constexpr int FDP_KB     = 4;                // keys per load batch (8 spilled at 3 blocks per SM)

struct fdp_piece { // 8 Q8_0 quants of one key at the lane's dimensions, and their block's scale
    int q0, q1;
    float d;
};

static __device__ __forceinline__ void fdp_load(fdp_piece & p, const char * row, const int dim0) {
    ftp_piece<GGML_TYPE_Q8_0> t;
    ftp_piece<GGML_TYPE_Q8_0>::load(t, row, dim0);
    p.q0 = t.q0;
    p.q1 = t.q1;
    p.d  = t.d;
}

template <int D>
__launch_bounds__(FDP_NWARPS*WARP_SIZE, 3)
static __global__ void flash_attn_dec_pascal(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int64_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    // no sinks, max_bias, softcap or kv rows (checked on the host)
    GGML_UNUSED_VARS(sinks_ptr, KV_rows_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13,
        ne31, ne32, nb32);
    static_assert(D == 256, "head 256 only: 8 dimensions per lane");
    constexpr int NH = FTP_NCOLS2;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*WARP_SIZE + lane;
    const int hl   = lane/4;   // the head whose score this lane ends up with
    const int dim0 = lane*8;   // the lane's 8 dimensions

    const int fault = ftp_fault_dev;

    const int tok       = blockIdx.x;
    const int sequence  = blockIdx.z / (ne02/NH);
    const int head0     = blockIdx.z*NH - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * K_h   = K_ptr + nb13*sequence + int64_t(nb12)*(head0/gqa_ratio);
    const char * V_h   = V_ptr + nb23*sequence + int64_t(nb22)*(head0/gqa_ratio);
    const half * maskh = (const half *) (mask_ptr + nb33*(sequence % ne33))
                       + int64_t(fastmodulo(tok, ne01))*(nb31/sizeof(half));

    __shared__ __align__(16) float P_s[FDP_NWARPS][FDP_KW*NH]; // probabilities, [key][head]
    __shared__ __align__(16) float A_s[FDP_NWARPS][NH];        // rescale factors per head
    __shared__ __align__(16) float O_s[NH*D];                  // the warps' outputs, summed at the end
    __shared__ float M_s[FDP_NWARPS][NH];
    __shared__ float S_s[FDP_NWARPS][NH];

    ggml_cuda_pdl_sync();

    // the query of the lane's 8 dimensions for each head, scaled, as int8 with one scale per 32 dimensions
    int   Qi[NH][2];
    float Qd[NH];
    {
        const float * Q_f = (const float *) (Q_ptr + int64_t(nb03)*sequence + int64_t(nb02)*head0)
                          + fastmodulo(tok, ne01)*(nb01/sizeof(float)) + dim0;
#pragma unroll
        for (int h = 0; h < NH; ++h) {
            const float4 a = *(const float4 *) (Q_f + h*(nb02/sizeof(float)) + 0);
            const float4 b = *(const float4 *) (Q_f + h*(nb02/sizeof(float)) + 4);
            float x[8] = {a.x*scale, a.y*scale, a.z*scale, a.w*scale, b.x*scale, b.y*scale, b.z*scale, b.w*scale};
            float amax = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                amax = fmaxf(amax, fabsf(x[i]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, WARP_SIZE));
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, WARP_SIZE));
            const float d   = amax / 127.0f;
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            int q[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q[i] = __float2int_rn(x[i]*inv);
            }
            Qi[h][0] = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | (q[3] << 24);
            Qi[h][1] = (q[4] & 0xFF) | ((q[5] & 0xFF) << 8) | ((q[6] & 0xFF) << 16) | (q[7] << 24);
            Qd[h] = d;
        }
    }

    float O[NH][8];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            O[h][i] = 0.0f;
        }
    }
    float KQ_max = -FLT_MAX/2.0f; // of head hl
    float KQ_sum = 0.0f;

    const int k_VKQ_max = KV_max_ptr ? KV_max_ptr[sequence*gridDim.x + blockIdx.x] : ne11;
    const int k_step    = gridDim.y*FDP_T;

    float * const P_w = P_s[w];
    float * const A_w = A_s[w];

    // the batch loops stay rolled: fully unrolled the kernel was ~12k instructions, past the instruction cache
    // (2026-11-20 rig: 3.9 ms at 104k keys, slower than the tile kernel). The next batch is loaded into nx while the
    // current one (cu) is used, then copied over.
    fdp_piece cu[FDP_KB], nx[FDP_KB];
    // the mask of a batch's FDP_KB keys, read as one 8-byte vector: k0 is a multiple of 32 keys and a batch starts
    // FDP_KB keys further, so each read is 8-byte aligned and stays inside the warp's 32-key slice
    static_assert(FDP_KB*sizeof(half) == sizeof(uint2), "the mask of a batch is one uint2");
    uint2 mcu = make_uint2(0, 0), mnx = make_uint2(0, 0);
    const int q4 = lane % 4; // the lane's share of its head's keys in the softmax: keys q4, q4 + 4, ...

#pragma unroll 1
    for (int k0 = blockIdx.y*FDP_T + w*FDP_KW; k0 < k_VKQ_max; k0 += k_step) {
        const char * K_t = K_h + int64_t(k0)*nb11;
        const char * V_t = V_h + int64_t(k0)*nb21;

#pragma unroll
        for (int j = 0; j < FDP_KB; ++j) {
            fdp_load(cu[j], K_t + int64_t(j)*nb11, dim0);
        }
        mcu = *(const uint2 *) (maskh + k0);

        // scores of the 32 keys into P_w[k][head], masked
#pragma unroll 1
        for (int kb = 0; kb < FDP_KW/FDP_KB; ++kb) {
            if (kb + 1 < FDP_KW/FDP_KB) {
#pragma unroll
                for (int j = 0; j < FDP_KB; ++j) {
                    fdp_load(nx[j], K_t + int64_t((kb + 1)*FDP_KB + j)*nb11, dim0);
                }
                mnx = *(const uint2 *) (maskh + k0 + (kb + 1)*FDP_KB);
            } else {
#pragma unroll
                for (int j = 0; j < FDP_KB; ++j) {
                    fdp_load(nx[j], V_t + int64_t(j)*nb21, dim0);
                }
            }
            const unsigned mw[2] = {mcu.x, mcu.y}; // no address of mcu: that put it on the stack
#pragma unroll
            for (int j = 0; j < FDP_KB; ++j) {
                float v[NH];
#pragma unroll
                for (int h = 0; h < NH; ++h) {
                    const int s = ggml_cuda_dp4a(cu[j].q1, Qi[h][1], ggml_cuda_dp4a(cu[j].q0, Qi[h][0], 0));
                    v[h] = float(s)*(Qd[h]*cu[j].d);
                }
                // butterfly: after the xor-16/8/4 steps each lane holds one head's partial (head lane/4), then a sum
                {
                    const bool up = lane & 16;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float keep = up ? v[4 + i] : v[i];
                        const float send = up ? v[i] : v[4 + i];
                        v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, 16, WARP_SIZE);
                    }
                }
                {
                    const bool up = lane & 8;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const float keep = up ? v[2 + i] : v[i];
                        const float send = up ? v[i] : v[2 + i];
                        v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, 8, WARP_SIZE);
                    }
                }
                {
                    const bool up = lane & 4;
                    const float keep = up ? v[1] : v[0];
                    const float send = up ? v[0] : v[1];
                    v[0] = keep + __shfl_xor_sync(0xFFFFFFFF, send, 4, WARP_SIZE);
                }
                v[0] += __shfl_xor_sync(0xFFFFFFFF, v[0], 2, WARP_SIZE);
                v[0] += __shfl_xor_sync(0xFFFFFFFF, v[0], 1, WARP_SIZE);
                const float mf = __half2float(__ushort_as_half((unsigned short) (mw[j/2] >> (16*(j % 2)))));
                if (q4 == j % 4) {
                    P_w[(kb*FDP_KB + j)*NH + hl] = v[0] + mf;
                }
            }
#pragma unroll
            for (int j = 0; j < FDP_KB; ++j) {
                cu[j] = nx[j];
            }
            mcu = mnx;
        }
        __syncwarp();

        // online softmax of head hl: the 4 lanes of the head take 8 keys each and agree through 2 shuffles
        float sc[FDP_KW/4];
        float mx = KQ_max;
#pragma unroll
        for (int i = 0; i < FDP_KW/4; ++i) {
            sc[i] = P_w[(q4 + 4*i)*NH + hl];
            mx = fmaxf(mx, sc[i]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 1, WARP_SIZE));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 2, WARP_SIZE));
        const float alpha = expf(KQ_max - mx);
        KQ_max = mx;
        float sum_add = 0.0f;
#pragma unroll
        for (int i = 0; i < FDP_KW/4; ++i) {
            sc[i] = expf(sc[i] - KQ_max);
            sum_add += sc[i];
            P_w[(q4 + 4*i)*NH + hl] = sc[i];
        }
        sum_add += __shfl_xor_sync(0xFFFFFFFF, sum_add, 1, WARP_SIZE);
        sum_add += __shfl_xor_sync(0xFFFFFFFF, sum_add, 2, WARP_SIZE);
        KQ_sum = KQ_sum*alpha + sum_add;
        if (q4 == 0) {
            A_w[hl] = alpha;
        }
        __syncwarp();

        {
            const float4 a0 = *(const float4 *) (A_w + 0);
            const float4 a1 = *(const float4 *) (A_w + 4);
            const float a[NH] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
            for (int h = 0; h < NH; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    O[h][i] *= a[h];
                }
            }
        }

        // output: O[h][dims] += P[k][h] * V[k0 + k][dims]; V batch 0 is in cu
#pragma unroll 1
        for (int kb = 0; kb < FDP_KW/FDP_KB; ++kb) {
            if (kb + 1 < FDP_KW/FDP_KB) {
#pragma unroll
                for (int j = 0; j < FDP_KB; ++j) {
                    fdp_load(nx[j], V_t + int64_t((kb + 1)*FDP_KB + j)*nb21, dim0);
                }
            }
#pragma unroll
            for (int j = 0; j < FDP_KB; ++j) {
                const int k = kb*FDP_KB + j;
                const float4 p0 = *(const float4 *) (P_w + k*NH + 0);
                const float4 p1 = *(const float4 *) (P_w + k*NH + 4);
                const float p[NH] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
                float x[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    x[i]     = float((int8_t) (cu[j].q0 >> (8*i)));
                    x[4 + i] = float((int8_t) (cu[j].q1 >> (8*i)));
                }
#pragma unroll
                for (int h = 0; h < NH; ++h) {
                    const float ph = p[h]*cu[j].d;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        O[h][i] = fmaf(ph, x[i], O[h][i]);
                    }
                }
            }
            if (kb + 1 < FDP_KW/FDP_KB) {
#pragma unroll
                for (int j = 0; j < FDP_KB; ++j) {
                    cu[j] = nx[j];
                }
            }
        }
        __syncwarp(); // the next tile overwrites P_w and A_w
    }

    // combine the 4 warps: common maximum per head, rescale, sum the outputs through shared memory
    if (lane % 4 == 0) {
        M_s[w][hl] = KQ_max;
        S_s[w][hl] = KQ_sum;
    }
    __syncthreads();
    float f_w[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        float m = M_s[0][h];
#pragma unroll
        for (int u = 1; u < FDP_NWARPS; ++u) {
            m = fmaxf(m, M_s[u][h]);
        }
        f_w[h] = expf(M_s[w][h] - m);
    }
#pragma unroll
    for (int u = 0; u < FDP_NWARPS; ++u) {
        if (w == u) {
#pragma unroll
            for (int h = 0; h < NH; ++h) {
                float4 * o = (float4 *) (O_s + h*D + dim0);
                float4 r0 = make_float4(O[h][0]*f_w[h], O[h][1]*f_w[h], O[h][2]*f_w[h], O[h][3]*f_w[h]);
                float4 r1 = make_float4(O[h][4]*f_w[h], O[h][5]*f_w[h], O[h][6]*f_w[h], O[h][7]*f_w[h]);
                if (u > 0) {
                    const float4 s0 = o[0];
                    const float4 s1 = o[1];
                    r0.x += s0.x; r0.y += s0.y; r0.z += s0.z; r0.w += s0.w;
                    r1.x += s1.x; r1.y += s1.y; r1.z += s1.z; r1.w += s1.w;
                }
                o[0] = r0;
                o[1] = r1;
            }
        }
        __syncthreads();
    }

    if (tok >= int(ne01.z)) {
        return;
    }
    // thread tid writes head tid/16, dimensions (tid % 16)*16 .. +15
    {
        const int h = tid / (FDP_NWARPS*WARP_SIZE/NH);
        const int d0 = (tid % (FDP_NWARPS*WARP_SIZE/NH)) * (D/(FDP_NWARPS*WARP_SIZE/NH));
        float m = M_s[0][h];
#pragma unroll
        for (int u = 1; u < FDP_NWARPS; ++u) {
            m = fmaxf(m, M_s[u][h]);
        }
        float s = 0.0f;
#pragma unroll
        for (int u = 0; u < FDP_NWARPS; ++u) {
            s += S_s[u][h]*expf(M_s[u][h] - m);
        }
        float out_scale = gridDim.y == 1 ? 1.0f/s : 1.0f;
        if (fault == 1) {
            out_scale *= ftp_fault_scale_dev;
        }
        const int j_dst_unrolled = ((sequence*int(ne01.z) + tok)*ne02 + head0 + h)*gridDim.y + blockIdx.y;
#pragma unroll
        for (int i = 0; i < D/(FDP_NWARPS*WARP_SIZE/NH); i += 4) {
            const float4 o = *(const float4 *) (O_s + h*D + d0 + i);
            *(float4 *) (dst_ptr + int64_t(j_dst_unrolled)*D + d0 + i) =
                make_float4(o.x*out_scale, o.y*out_scale, o.z*out_scale, o.w*out_scale);
        }
        if (gridDim.y != 1 && d0 == 0) {
            dst_meta_ptr[j_dst_unrolled] = make_float2(m, s);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int D, ggml_type type_K, ggml_type type_V, bool split, bool lds = false>
static void ftp_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using S = ftp_shape<D>;
    const fattn_kernel_t kernel = flash_attn_tile_pascal<D, type_K, type_V, split, lds>;
    // K and V are read in their own types: no F16 copy
    launch_fattn<D, S::ncols1, FTP_NCOLS2>(ctx, dst, kernel, FTP_NWARPS, 0, S::T, false, false, false, false, WARP_SIZE);
}

static int ftp_env_int(const char * name, const int dflt) {
    const char * env = getenv(name);
    return env ? atoi(env) : dflt;
}

// head 128: 2 = the split scores (default), 1 = 4 keys x 4 columns per lane over the whole head
static bool ftp_split_128() {
    static const bool split = ftp_env_int("GGML_CUDA_FATTN_TILE_PASCAL_QK", 2) != 1;
    return split;
}

static int ftp_mode() {
    static const int mode = ftp_env_int("GGML_CUDA_FATTN_TILE_PASCAL", 1);
    return mode;
}

static int ftp_min_tokens(const bool q8_0) {
    static const int min_tokens = ftp_env_int("GGML_CUDA_FATTN_TILE_PASCAL_MIN", 0);
    return min_tokens > 0 ? min_tokens : (q8_0 ? 3 : 9);
}

// rows up to this many go to the decode kernel (head 256, Q8_0, no sinks); 0 = never (default)
static int fdp_max_rows() {
    static const int n = ftp_env_int("GGML_CUDA_FATTN_DEC_PASCAL", 0);
    return n;
}

// Q8_0 loads decoded at store in the tile kernel (ftp_tpiece); 0 = decoded at load (default)
static bool ftp_lds() {
    static const bool l = ftp_env_int("GGML_CUDA_FATTN_TILE_PASCAL_LDS", 0) != 0;
    return l;
}

// Q8_0 K at head 256 scored with dp4a in int8 (kdp); 0 = off (default)
static bool ftp_kdp() {
    static const bool k = ftp_env_int("GGML_CUDA_FATTN_TILE_PASCAL_KDP", 0) != 0;
    return k;
}

static int ftp_fault() {
    static const int fault = ftp_env_int("GGML_CUDA_FATTN_TILE_PASCAL_FAULT", 0);
    return fault;
}

// the factor fault 1 multiplies the output by (GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE, default 1.05); a value that is
// not a finite number gets a warning and the default
static float ftp_fault_scale() {
    static const float scale = [] {
        const char * env = getenv("GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE");
        if (env == nullptr) {
            return FTP_FAULT_SCALE_DEFAULT;
        }
        char * end = nullptr;
        const float v = strtof(env, &end);
        if (end == env || *end != '\0' || !std::isfinite(v)) {
            GGML_LOG_WARN("ftp_fault_scale: GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE=%s is not a number: %.2f is used\n",
                env, (double) FTP_FAULT_SCALE_DEFAULT);
            return FTP_FAULT_SCALE_DEFAULT;
        }
        return v;
    }();
    return scale;
}

static bool ftp_aligned(const void * p, const size_t n) {
    return (uintptr_t) p % n == 0;
}

static bool ftp_strides_aligned(const ggml_tensor * t, const size_t n) {
    return ftp_aligned(t->data, n) && t->nb[1] % n == 0 && t->nb[2] % n == 0 && t->nb[3] % n == 0;
}

} // namespace

bool ggml_cuda_fattn_tile_pascal_supported(const int device, const ggml_tensor * dst) {
    const int mode = ftp_mode();
    const int cc   = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !(mode == 2 || (mode == 1 && cc == GGML_CUDA_CC_DP4A))) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || K == nullptr || V == nullptr) {
        return false;
    }
    // K/V read through kv_rows (src[5]): only the MMA kernel gathers rows
    if (dst->src[5] != nullptr) {
        return false;
    }
    const int64_t D = Q->ne[0];
    if ((D != 128 && D != 256) || K->ne[0] != D || V->ne[0] != D) {
        return false;
    }
    const bool f16  = K->type == GGML_TYPE_F16  && V->type == GGML_TYPE_F16;
    const bool q8_0 = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
    if (!f16 && !q8_0) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0 || (Q->ne[2]/K->ne[2]) % FTP_NCOLS2 != 0 ||
            V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    // whole key tiles: the KV length is padded to FATTN_KQ_STRIDE
    if (K->ne[1] % FATTN_KQ_STRIDE != 0 || Q->ne[1] < ftp_min_tokens(q8_0)) {
        return false;
    }
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
            mask->ne[2] != 1 || mask->nb[1] % sizeof(half) != 0) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || ggml_nelements(sinks) < Q->ne[2])) {
        return false;
    }
    // the query as float4, F16 K/V rows as 16-byte loads, Q8_0 rows as 4-byte loads, the output as float4
    if (Q->nb[0] != sizeof(float) || !ftp_strides_aligned(Q, 16) || !ftp_strides_aligned(K, f16 ? 16 : 4) ||
            !ftp_strides_aligned(V, f16 ? 16 : 4) || !ggml_is_contiguous(dst) || !ftp_aligned(dst->data, 16)) {
        return false;
    }
    return true;
}

bool ggml_cuda_flash_attn_ext_tile_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_fattn_tile_pascal_supported(ctx.device, dst)) {
        return false;
    }

    // the fault switch and its factor live in each device's constant memory; set them once per device when used
    static bool fault_set[GGML_CUDA_MAX_DEVICES] = {false};
    if (ftp_fault() != 0 && !fault_set[ctx.device]) {
        const int   fault = ftp_fault();
        const float scale = ftp_fault_scale();
        CUDA_CHECK(cudaMemcpyToSymbol(ftp_fault_dev,       &fault, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(ftp_fault_scale_dev, &scale, sizeof(float)));
        static bool logged_fault = false;
        if (!logged_fault) {
            logged_fault = true;
            if (fault == 1) {
                GGML_LOG_INFO("%s: GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 (tests of the tests): output x %.9g\n",
                    __func__, (double) scale);
            } else {
                GGML_LOG_INFO("%s: GGML_CUDA_FATTN_TILE_PASCAL_FAULT=%d (tests of the tests)\n", __func__, fault);
            }
        }
        fault_set[ctx.device] = true;
    }

    const ggml_tensor * K = dst->src[1];
    const bool q8_0 = K->type == GGML_TYPE_Q8_0;

    // the one-token decode kernel (GGML_CUDA_FATTN_DEC_PASCAL); the mask is read as 8-byte vectors of 4 keys
    {
        const ggml_tensor * Q    = dst->src[0];
        const ggml_tensor * mask = dst->src[3];
        if (q8_0 && Q->ne[0] == 256 && Q->ne[1] <= fdp_max_rows() && dst->src[4] == nullptr &&
                ftp_aligned(mask->data, 16) && mask->nb[1] % 16 == 0 && mask->nb[3] % 16 == 0) {
            static bool logged_dec = false;
            if (!logged_dec) {
                logged_dec = true;
                GGML_LOG_INFO("%s: compute-6.1 decode attention kernel: head 256, q8_0 K/V, up to %d rows (GGML_CUDA_FATTN_DEC_PASCAL)\n",
                    __func__, fdp_max_rows());
            }
            launch_fattn<256, 1, FTP_NCOLS2>(ctx, dst, flash_attn_dec_pascal<256>, FDP_NWARPS, 0, FDP_T,
                false, false, false, false, WARP_SIZE);
            return true;
        }
    }

    // one log line per head size and cache type, so a run shows that the kernel took the prompts
    static bool logged[2][2] = {{false, false}, {false, false}};
    const int D = (int) dst->src[0]->ne[0];
    if (!logged[D == 256][q8_0]) {
        logged[D == 256][q8_0] = true;
        GGML_LOG_INFO("%s: compute-6.1 prompt attention kernel: head %d, %s K/V%s, %s scores (GGML_CUDA_FATTN_TILE_PASCAL=0 turns it off)\n",
            __func__, D, ggml_type_name(K->type), q8_0 && ftp_lds() ? " decoded at store" : "",
            D == 128 && ftp_split_128() ? "split" : "plain");
    }

    switch (dst->src[0]->ne[0]) {
        case 128:
            if (ftp_split_128()) {
                if (q8_0 && ftp_lds()) {
                    ftp_launch<128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true, true>(ctx, dst);
                } else if (q8_0) {
                    ftp_launch<128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true>(ctx, dst);
                } else {
                    ftp_launch<128, GGML_TYPE_F16, GGML_TYPE_F16, true>(ctx, dst);
                }
            } else {
                if (q8_0 && ftp_lds()) {
                    ftp_launch<128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, true>(ctx, dst);
                } else if (q8_0) {
                    ftp_launch<128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>(ctx, dst);
                } else {
                    ftp_launch<128, GGML_TYPE_F16, GGML_TYPE_F16, false>(ctx, dst);
                }
            }
            break;
        case 256:
            if (q8_0 && ftp_kdp()) {
                static bool logged_kd = false;
                if (!logged_kd) {
                    logged_kd = true;
                    GGML_LOG_INFO("%s: compute-6.1 prompt attention: q8_0 K scored in int8 with dp4a (GGML_CUDA_FATTN_TILE_PASCAL_KDP)\n", __func__);
                }
                using S = ftp_shape<256>;
                launch_fattn<256, S::ncols1, FTP_NCOLS2>(ctx, dst,
                    flash_attn_tile_pascal<256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, true, true>,
                    FTP_NWARPS, 0, S::T, false, false, false, false, WARP_SIZE);
            } else if (q8_0 && ftp_lds()) {
                ftp_launch<256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, true>(ctx, dst);
            } else if (q8_0) {
                ftp_launch<256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false>(ctx, dst);
            } else {
                ftp_launch<256, GGML_TYPE_F16, GGML_TYPE_F16, false>(ctx, dst);
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
    return true;
}

#else

bool ggml_cuda_fattn_tile_pascal_supported(const int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_flash_attn_ext_tile_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
