#pragma once

// The k-column Pascal matrix products of mmvq-pascal-kcol.cu without ggml's headers, so that a standalone benchmark can
// time the same code (bench/mmvq_pascal/src/kcol_mm.cuh in llms-on-crypto-mining-rigs is a copy). The weight layouts are
// ggml's (block_q8_0, block_q2_0, block_q3_K, block_q4_K, block_q5_K, block_q6_K, block_iq4_xs, block_q4_0, block_q5_0,
// block_iq4_nl, and since Q713 block_iq2_xxs, block_iq2_xs, block_iq2_s, block_iq3_xxs, block_iq3_s, block_iq1_m), read as
// bytes. The grid types and IQ1_M decode each 32-weight block once into signed int8 words with mmvqp_iq::decode32
// (mmvq-pascal-iq.cuh, also ggml-free; the bench copies it beside this file), each block copying its type's grid into
// shared memory once.
//
// What the kernels do (dst = x * y for k = 1 to 8 columns of y, the k tokens of a speculative-decoding verify):
//   - Each weight is read from memory once per launch and decoded once into signed or unsigned int8 words, which are
//     then multiplied with dp4a by the activations of every column. A one-token kernel that loops over the columns
//     decodes every weight k times and recomputes every activation term k times, so its cost grows almost per column.
//   - The activations are quantized from FP32 by the launch's own quantizer, exactly as quantize_q8_1 does it (one
//     scale per 32 values, the scale stored as FP16), but into records laid out for these kernels: per column and per
//     lane unit (the weights one lane takes per iteration), the int8 values (permuted where a type's decoding returns
//     its weights in another order), the FP32 scales and the type's activation terms (the block sums for the K-quants'
//     mins, the 16-value sums for Q3_K's and Q6_K's offsets, the 32-value sums for Q4_0's and Q5_0's), and padding so
//     that the lanes of a quarter warp read different banks. Each block copies the records of its K tile into shared
//     memory once with coalesced 128-bit loads, so the activation work per column is two 128-bit and one 64-bit shared
//     load per 32 weights, shared by the R rows of a lane group, and no activation term is recomputed per row or per
//     lane.
//   - A warp is cut into lane groups of L lanes (a power of two dividing the row's lane units, at most 32); a group owns
//     R rows. Each iteration the weights of the next L units of the group's rows are staged in shared memory with
//     coalesced 128-bit loads: by each lane group for its own rows where the segments are whole 16-byte units, else by
//     the whole warp, each row's segment from the 16-byte boundary below it (rows of 2-byte-aligned blocks, rows
//     narrower than a 16-byte unit per lane). Each lane then reads its unit's words from there.
//   - Per 32 weights, row and column: 8 dp4a and 3 or 4 FP32 operations (Q3_K, Q6_K: 5), with the weight scales turned
//     into FP32 once per row and shared by all columns.
//   - The sum of a row is formed per lane in unit order, then over the lanes of its group with shuffles in a fixed
//     order. It depends on L only, which depends on the row width only, so a column's result is the same for every
//     column count and launch shape.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>

#define KCOL_NT        128         // threads per block
#define KCOL_NW        4           // warps per block
#define KCOL_MAX_COLS  8
#ifndef KCOL_MIN_BLOCKS
#define KCOL_MIN_BLOCKS 2         // blocks per SM the compiler keeps registers for (__launch_bounds__)
#endif
#define KCOL_SMEM_MAX  (48 * 1024) // a block's shared memory without opting in to more

// KCOL_HD: how the decoding is compiled (device code here; bench/mmvq_pascal/verify/kcol_unit_check.cu also builds it
// for the host and checks each type's decoding against ggml's reference dequantization on a CPU).
#ifndef KCOL_HD
#define KCOL_HD __device__
#endif
#ifndef MMVQP_IQ_HD
#define MMVQP_IQ_HD KCOL_HD
#endif
#include "mmvq-pascal-iq.cuh"

namespace kcol {

enum ktype : int { KQ8_0 = 0, KQ2_0, KQ3_K, KQ4_K, KQ5_K, KQ6_K, KIQ4_XS, KQ4_0, KQ5_0, KIQ4_NL,
                   KIQ2_XXS, KIQ2_XS, KIQ2_S, KIQ3_XXS, KIQ3_S, KIQ1_M, KTYPE_COUNT };

// What an activation record holds besides the int8 values and the FP32 scale d of each 32-value block:
//   ACT_D:  nothing (the second float of each pair is 0);
//   ACT_DS: S = d * the sum of the block's int8 values (the K-quants' mins);
//   ACT_DH: two ints per block, the offset sums of its two halves (Q3_K: -4 f sum, Q6_K: -32 sum), the dp4a chains'
//           initial values;
//   ACT_DI: in place of the second float, one int: the offset times the block's sum (Q4_0: -8 sum, Q5_0: -16 sum), the
//           dp4a chain's initial value.
enum act_kind { ACT_D = 0, ACT_DS = 1, ACT_DH = 2, ACT_DI = 3 };
// The order of the int8 values in a block: as they come, IQ4_XS's lookup order, Q2_0's lookup order, Q5_0's order
// (word m holds values m, m + 8, m + 16, m + 24), IQ1_M's order (mmvqp_iq::act_perm_iq1).
enum act_perm { PERM_NONE = 0, PERM_IQ4 = 1, PERM_Q2 = 2, PERM_Q5 = 3, PERM_IQ1 = 4 };

// Per type: UB bytes and UW weights per lane unit (a lane's share of one iteration), AL the unit's alignment in the
// staging buffer, the activation records' contents and order.
template <int T> struct traits;
template <> struct traits<KQ8_0>   { static constexpr int UB = 68,  UW = 64,  AL = 4,  ACT = ACT_D,  PERM = PERM_NONE; };
template <> struct traits<KQ2_0>   { static constexpr int UB = 36,  UW = 128, AL = 4,  ACT = ACT_D,  PERM = PERM_Q2;   };
template <> struct traits<KQ3_K>   { static constexpr int UB = 110, UW = 256, AL = 2,  ACT = ACT_DH, PERM = PERM_NONE; };
template <> struct traits<KQ4_K>   { static constexpr int UB = 144, UW = 256, AL = 16, ACT = ACT_DS, PERM = PERM_NONE; };
template <> struct traits<KQ5_K>   { static constexpr int UB = 176, UW = 256, AL = 16, ACT = ACT_DS, PERM = PERM_NONE; };
template <> struct traits<KQ6_K>   { static constexpr int UB = 210, UW = 256, AL = 2,  ACT = ACT_DH, PERM = PERM_NONE; };
template <> struct traits<KIQ4_XS> { static constexpr int UB = 136, UW = 256, AL = 8,  ACT = ACT_D,  PERM = PERM_IQ4;  };
// two 32-weight blocks per unit, as Q2_0's two 64-weight blocks, so that units stay 4-byte aligned
template <> struct traits<KQ4_0>   { static constexpr int UB = 36,  UW = 64,  AL = 4,  ACT = ACT_DI, PERM = PERM_NONE; };
template <> struct traits<KQ5_0>   { static constexpr int UB = 44,  UW = 64,  AL = 4,  ACT = ACT_DI, PERM = PERM_Q5;   };
template <> struct traits<KIQ4_NL> { static constexpr int UB = 36,  UW = 64,  AL = 4,  ACT = ACT_D,  PERM = PERM_IQ4;  };
// one 256-weight super-block per unit (Q713): the grid types' blocks are 2-byte aligned, IQ1_M's 56 bytes 8-byte aligned
template <> struct traits<KIQ2_XXS> { static constexpr int UB = 66,  UW = 256, AL = 2, ACT = ACT_D, PERM = PERM_NONE; };
template <> struct traits<KIQ2_XS>  { static constexpr int UB = 74,  UW = 256, AL = 2, ACT = ACT_D, PERM = PERM_NONE; };
template <> struct traits<KIQ2_S>   { static constexpr int UB = 82,  UW = 256, AL = 2, ACT = ACT_D, PERM = PERM_NONE; };
template <> struct traits<KIQ3_XXS> { static constexpr int UB = 98,  UW = 256, AL = 2, ACT = ACT_D, PERM = PERM_NONE; };
template <> struct traits<KIQ3_S>   { static constexpr int UB = 110, UW = 256, AL = 2, ACT = ACT_D, PERM = PERM_NONE; };
template <> struct traits<KIQ1_M>   { static constexpr int UB = 56,  UW = 256, AL = 8, ACT = ACT_D, PERM = PERM_IQ1;  };

// The grid types and IQ1_M: their mmvqp_iq type (-1 for the other types) and the words of their grid in shared memory.
template <int T> constexpr __host__ __device__ int gtype_of() {
    return T == KIQ2_XXS ? mmvqp_iq::G_IQ2_XXS : T == KIQ2_XS ? mmvqp_iq::G_IQ2_XS : T == KIQ2_S ? mmvqp_iq::G_IQ2_S :
        T == KIQ3_XXS ? mmvqp_iq::G_IQ3_XXS : T == KIQ3_S ? mmvqp_iq::G_IQ3_S : T == KIQ1_M ? mmvqp_iq::G_IQ1_M : -1;
}
template <int T> constexpr __host__ __device__ int grid_words() {
    if constexpr (gtype_of<T>() >= 0) {
        return mmvqp_iq::gtraits<gtype_of<T>()>::GW;
    } else {
        return 0;
    }
}

// ACT_DI: the offset of the type's unsigned codes
template <int T> constexpr __host__ __device__ int act_off() { return T == KQ4_0 ? -8 : T == KQ5_0 ? -16 : 0; }

// Words of one activation record: the int8 values, (d, S) per block, the offset sums (ACT_DH), rounded up to a multiple
// of 4 words that is an odd multiple of 4 (so that records a lane apart start 4 banks apart).
constexpr __host__ __device__ int rec_words(const int UW, const int act) {
    const int nab = UW / 32;
    const int w   = (UW / 4 + 2 * nab + (act == ACT_DH ? 2 * nab : 0) + 3) / 4 * 4;
    return (w / 4) % 2 == 0 ? w + 4 : w;
}
template <int T> constexpr __host__ __device__ int rec_words() { return rec_words(traits<T>::UW, traits<T>::ACT); }

struct type_info { int UB, UW, AL, RS; };
inline type_info info(const int t) {
    switch (t) {
        case KQ8_0:   return { traits<KQ8_0>::UB,   traits<KQ8_0>::UW,   traits<KQ8_0>::AL,   rec_words<KQ8_0>()   };
        case KQ2_0:   return { traits<KQ2_0>::UB,   traits<KQ2_0>::UW,   traits<KQ2_0>::AL,   rec_words<KQ2_0>()   };
        case KQ3_K:   return { traits<KQ3_K>::UB,   traits<KQ3_K>::UW,   traits<KQ3_K>::AL,   rec_words<KQ3_K>()   };
        case KQ4_K:   return { traits<KQ4_K>::UB,   traits<KQ4_K>::UW,   traits<KQ4_K>::AL,   rec_words<KQ4_K>()   };
        case KQ5_K:   return { traits<KQ5_K>::UB,   traits<KQ5_K>::UW,   traits<KQ5_K>::AL,   rec_words<KQ5_K>()   };
        case KQ6_K:   return { traits<KQ6_K>::UB,   traits<KQ6_K>::UW,   traits<KQ6_K>::AL,   rec_words<KQ6_K>()   };
        case KIQ4_XS: return { traits<KIQ4_XS>::UB, traits<KIQ4_XS>::UW, traits<KIQ4_XS>::AL, rec_words<KIQ4_XS>() };
        case KQ4_0:   return { traits<KQ4_0>::UB,   traits<KQ4_0>::UW,   traits<KQ4_0>::AL,   rec_words<KQ4_0>()   };
        case KQ5_0:   return { traits<KQ5_0>::UB,   traits<KQ5_0>::UW,   traits<KQ5_0>::AL,   rec_words<KQ5_0>()   };
        case KIQ4_NL: return { traits<KIQ4_NL>::UB, traits<KIQ4_NL>::UW, traits<KIQ4_NL>::AL, rec_words<KIQ4_NL>() };
#define KCOL_IQ_INFO(T) return { traits<T>::UB, traits<T>::UW, traits<T>::AL, rec_words<T>() }
        case KIQ2_XXS: KCOL_IQ_INFO(KIQ2_XXS);
        case KIQ2_XS:  KCOL_IQ_INFO(KIQ2_XS);
        case KIQ2_S:   KCOL_IQ_INFO(KIQ2_S);
        case KIQ3_XXS: KCOL_IQ_INFO(KIQ3_XXS);
        case KIQ3_S:   KCOL_IQ_INFO(KIQ3_S);
        case KIQ1_M:   KCOL_IQ_INFO(KIQ1_M);
#undef KCOL_IQ_INFO
        default:      return { 0, 0, 0, 0 };
    }
}

// Bytes of the grid a type's kernels keep in shared memory (the grid types and IQ1_M; 0 for the others).
inline int grid_bytes(const int t) {
    switch (t) {
        case KIQ2_XXS: return 4 * grid_words<KIQ2_XXS>();
        case KIQ2_XS:  return 4 * grid_words<KIQ2_XS>();
        case KIQ2_S:   return 4 * grid_words<KIQ2_S>();
        case KIQ3_XXS: return 4 * grid_words<KIQ3_XXS>();
        case KIQ3_S:   return 4 * grid_words<KIQ3_S>();
        case KIQ1_M:   return 4 * grid_words<KIQ1_M>();
        default:       return 0;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Arithmetic helpers

static KCOL_HD __forceinline__ uint32_t prmt(const uint32_t a, const uint32_t b, const uint32_t sel) {
#ifdef __CUDA_ARCH__
    return __byte_perm(a, b, sel);
#else
    const uint64_t ab = ((uint64_t) b << 32) | a;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        r |= (uint32_t) ((ab >> (8 * ((sel >> (4 * i)) & 7))) & 0xff) << (8 * i);
    }
    return r;
#endif
}

static KCOL_HD __forceinline__ int dp4a(const int a, const int b, const int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

// a: 4 unsigned bytes, b: 4 signed bytes
static KCOL_HD __forceinline__ int dp4a_us(const uint32_t a, const int b, const int c) {
#if __CUDA_ARCH__ >= 610
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    const int8_t * b8 = (const int8_t *) &b;
    return c + (int) (a & 0xff) * b8[0] + (int) ((a >> 8) & 0xff) * b8[1] + (int) ((a >> 16) & 0xff) * b8[2] +
        (int) (a >> 24) * b8[3];
#endif
}

static KCOL_HD __forceinline__ float h2f(const unsigned short h) {
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    const float v = e == 0 ? m * (1.0f / 16777216.0f) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float) (1024 + m), e - 25);
    return h & 0x8000 ? -v : v;
#endif
}
static KCOL_HD __forceinline__ float h2f_lo(const uint32_t w) {
    return h2f((unsigned short) (w & 0xffff));
}
static KCOL_HD __forceinline__ float h2f_hi(const uint32_t w) {
    return h2f((unsigned short) (w >> 16));
}
static KCOL_HD __forceinline__ float u2f(const uint32_t u) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f;
    memcpy(&f, &u, 4);
    return f;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// Activation records

// The position of value i (0..31) of a block among the block's 32 int8 values in a record.
template <int PERM>
static __host__ __device__ __forceinline__ int act_pos(const int i) {
    if constexpr (PERM == PERM_IQ4) {
        // the IQ4_XS lookup returns weights (4k, 4k + 16, 4k + 1, 4k + 17) and (4k + 2, 4k + 18, 4k + 3, 4k + 19) as
        // words 2k and 2k + 1
        const int h = i >> 4;
        const int r = i & 3;
        return 8 * ((i & 15) >> 2) + 4 * (r >> 1) + 2 * (r & 1) + h;
    } else if constexpr (PERM == PERM_Q2) {
        // the Q2_0 lookup returns weights (8k, 8k + 2, 8k + 4, 8k + 6) and (8k + 1, 8k + 3, 8k + 5, 8k + 7) as words
        // 2k and 2k + 1
        const int r = i & 7;
        return 8 * (i >> 3) + 4 * (r & 1) + (r >> 1);
    } else if constexpr (PERM == PERM_Q5) {
        // Q5_0's decoding returns weights (m, m + 8, m + 16, m + 24) as word m
        return 4 * (i & 7) + (i >> 3);
    } else if constexpr (PERM == PERM_IQ1) {
        return mmvqp_iq::act_perm_iq1(i);
    } else {
        return i;
    }
}

// Quantizes x (ncols columns of K floats, stride_col floats apart) into records: one warp per 32-value block, as
// quantize_q8_1 (d = amax / 127, q = round(x / d), d kept as FP16). Grid: (ceil(K / 128), ncols), KCOL_NT threads.
template <int T>
__launch_bounds__(KCOL_NT)
static __global__ void quantize_records(const float * __restrict__ x, uint32_t * __restrict__ rec, const int K,
                                        const int stride_col, const int nunits) {
    constexpr int UW  = traits<T>::UW;
    constexpr int NAB = UW / 32;
    constexpr int RS  = rec_words<T>();
    const int lane = threadIdx.x % 32;
    const int blk  = blockIdx.x * (KCOL_NT / 32) + threadIdx.x / 32;
    const int col  = blockIdx.y;
    if (blk * 32 >= K) {
        return;
    }
    const float xi = x[(int64_t) col * stride_col + blk * 32 + lane];
    float amax = fabsf(xi);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off, 32));
    }
    const float d = amax / 127.0f;
    const int   q = amax == 0.0f ? 0 : (int) roundf(xi / d);
    int s16 = q;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        s16 += __shfl_xor_sync(0xFFFFFFFF, s16, off, 32);
    }
    const int s32 = s16 + __shfl_xor_sync(0xFFFFFFFF, s16, 16, 32);

    const int  u = blk / NAB;
    const int  b = blk - u * NAB;
    uint32_t * r = rec + ((int64_t) col * nunits + u) * RS;
    ((int8_t *) r)[32 * b + act_pos<traits<T>::PERM>(lane)] = (int8_t) q;

    const float d8 = __half2float(__float2half(d));
    if constexpr (traits<T>::ACT == ACT_DH) {
        // Q3_K: block b of a super-block holds the codes at bits 2 (b % 4) (b % 4 = 3: bits 4, 5 after a shift by 2) of
        // its bytes, read in place, f = 1, 4, 16, 16 times their value: the scale is d / f and the offset -4 f per value
        int f = 1;
        int m = -32;
        if constexpr (T == KQ3_K) {
            f = 1 << (2 * min(b & 3, 2));
            m = -4 * f;
        }
        if (lane == 0) {
            r[UW / 4 + 2 * b]     = __float_as_uint(d8 / (float) f);
            r[UW / 4 + 2 * b + 1] = 0;
            r[UW / 4 + 2 * NAB + 2 * b] = (uint32_t) (m * s16);
        } else if (lane == 16) {
            r[UW / 4 + 2 * NAB + 2 * b + 1] = (uint32_t) (m * s16);
        }
    } else if (lane == 0) {
        r[UW / 4 + 2 * b]     = __float_as_uint(d8);
        r[UW / 4 + 2 * b + 1] = traits<T>::ACT == ACT_DI ? (uint32_t) (act_off<T>() * s32) :
            __float_as_uint(traits<T>::ACT == ACT_DS ? d8 * (float) s32 : 0.0f);
    }
}

// One column's activations of block b of a lane unit: the 8 int8 words, d, and the second float or the two offsets.
struct act_blk {
    int   u[8];
    float d;
    float s;
    int   h0;
    int   h1;
};

template <int T>
static KCOL_HD __forceinline__ void load_act(act_blk & a, const uint32_t * rec, const int b) {
    constexpr int UW  = traits<T>::UW;
    constexpr int NAB = UW / 32;
    const uint4 q0 = ((const uint4 *) (rec + 8 * b))[0];
    const uint4 q1 = ((const uint4 *) (rec + 8 * b))[1];
    a.u[0] = (int) q0.x; a.u[1] = (int) q0.y; a.u[2] = (int) q0.z; a.u[3] = (int) q0.w;
    a.u[4] = (int) q1.x; a.u[5] = (int) q1.y; a.u[6] = (int) q1.z; a.u[7] = (int) q1.w;
    const uint2 ds = *(const uint2 *) (rec + UW / 4 + 2 * b);
    a.d = u2f(ds.x);
    a.s = u2f(ds.y);
    if constexpr (traits<T>::ACT == ACT_DH) {
        const int2 h = *(const int2 *) (rec + UW / 4 + 2 * NAB + 2 * b);
        a.h0 = h.x;
        a.h1 = h.y;
    } else if constexpr (traits<T>::ACT == ACT_DI) {
        a.h0 = (int) ds.y;
        a.h1 = 0;
    } else {
        a.h0 = 0;
        a.h1 = 0;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Weight units in the staging buffer

// A unit of the staging buffer: its address rounded down to 4 bytes and, for 2-byte-aligned units, the __byte_perm
// selector that shifts each pair of words by the 2 bytes it lies off.
struct unit_ptr {
    const uint32_t * a;
    uint32_t         sel;
};

template <int AL>
static KCOL_HD __forceinline__ unit_ptr make_unit(const char * p) {
    unit_ptr u;
    if constexpr (AL == 2) {
        const int mis = (int) ((uintptr_t) p & 2);
        u.a   = (const uint32_t *) (p - mis); // pointer arithmetic keeps the shared address space
        u.sel = mis ? 0x5432 : 0x3210;
    } else {
        u.a   = (const uint32_t *) p;
        u.sel = 0x3210;
    }
    return u;
}

// N words of a unit from word w0 on; w0 a compile-time constant after unrolling.
template <int AL, int N>
static KCOL_HD __forceinline__ void unit_words(uint32_t * w, const unit_ptr & u, const int w0) {
    if constexpr (AL == 2) {
        uint32_t cur = u.a[w0];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const uint32_t nxt = u.a[w0 + i + 1];
            w[i] = prmt(cur, nxt, u.sel);
            cur = nxt;
        }
    } else if constexpr (AL == 16) {
        // w0 and N are multiples of 4 at every use
        static_assert(N % 4 == 0, "128-bit unit loads");
#pragma unroll
        for (int i = 0; i < N; i += 4) {
            const uint4 v = *(const uint4 *) (u.a + w0 + i);
            w[i] = v.x; w[i + 1] = v.y; w[i + 2] = v.z; w[i + 3] = v.w;
        }
    } else if constexpr (AL == 8) {
        // w0 and N are even at every use
        static_assert(N % 2 == 0, "64-bit unit loads");
#pragma unroll
        for (int i = 0; i < N; i += 2) {
            const uint2 v = *(const uint2 *) (u.a + w0 + i);
            w[i] = v.x; w[i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            w[i] = u.a[w0 + i];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// One lane unit of R rows times NC columns, per type. acc[i][j] gains row i's products with column j. act: the unit's
// record of column 0 in shared memory; column j's is cstride words further.

template <int NC, int R>
static KCOL_HD __forceinline__ void unit_q8_0(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                 float (&acc)[R][NC]) {
    // two blocks of 34 bytes: d at bytes 0 and 34, the int8 values at 2..33 and 36..67
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        int   q[R][8];
        float d[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (c == 0) {
                uint32_t w[9];
                unit_words<4, 9>(w, wu[i], 0);
                d[i] = h2f_lo(w[0]);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    q[i][k] = (int) prmt(w[k], w[k + 1], 0x5432);
                }
            } else {
                uint32_t w[9];
                unit_words<4, 9>(w, wu[i], 8);
                d[i] = h2f_hi(w[0]);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    q[i][k] = (int) w[k + 1];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            act_blk a;
            load_act<KQ8_0>(a, act + j * cstride, c);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                int isum = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    isum = dp4a(q[i][k], a.u[k], isum);
                }
                acc[i][j] = fmaf((float) isum, d[i] * a.d, acc[i][j]);
            }
        }
    }
}

// sum((code - 1) * activation) over the 16 weights of one Q2_0 code word c (weight j in bits 2j, 2j + 1), the
// activations in act_pos<PERM_Q2>'s order: 4 lookups and 3 shifts per 16 weights.
static KCOL_HD __forceinline__ void q2_0_words(const uint32_t c, int * q) {
    constexpr uint32_t t = 0x020100FF; // codes 0, 1, 2, 3 -> -1, 0, 1, 2
    q[0] = (int) prmt(t, t, c);
    q[1] = (int) prmt(t, t, c >> 2);
    q[2] = (int) prmt(t, t, c >> 16);
    q[3] = (int) prmt(t, t, c >> 18);
}

template <int NC, int R>
static KCOL_HD __forceinline__ void unit_q2_0(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                 float (&acc)[R][NC]) {
    // two blocks of 18 bytes: d at bytes 0 and 18, 4 code words of 16 weights at 2, 6, 10, 14 and 20, 24, 28, 32;
    // activation blocks 2c and 2c + 1 take code words 0, 1 and 2, 3 of block c
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        int   q[R][2][8];
        float d[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uint32_t w[5];
            uint32_t cw[4];
            if (c == 0) {
                unit_words<4, 5>(w, wu[i], 0);
                d[i] = h2f_lo(w[0]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    cw[k] = prmt(w[k], w[k + 1], 0x5432);
                }
            } else {
                unit_words<4, 5>(w, wu[i], 4);
                d[i] = h2f_hi(w[0]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    cw[k] = w[k + 1];
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                q2_0_words(cw[2 * h + 0], q[i][h] + 0);
                q2_0_words(cw[2 * h + 1], q[i][h] + 4);
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            act_blk a0;
            act_blk a1;
            load_act<KQ2_0>(a0, act + j * cstride, 2 * c + 0);
            load_act<KQ2_0>(a1, act + j * cstride, 2 * c + 1);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                int i0 = 0;
                int i1 = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    i0 = dp4a(q[i][0][k], a0.u[k], i0);
                    i1 = dp4a(q[i][1][k], a1.u[k], i1);
                }
                acc[i][j] = fmaf(d[i], fmaf((float) i1, a1.d, (float) i0 * a0.d), acc[i][j]);
            }
        }
    }
}

// Q4_K and Q5_K: d, dmin and the 6-bit scales and mins of sub-blocks 0..7 from the first 4 words.
struct k_scales {
    float    d;
    float    dmin;
    uint32_t sc_lo, m_lo, sc_hi, m_hi;
};
static KCOL_HD __forceinline__ k_scales k_scales_of(const uint32_t * w) {
    k_scales s;
    s.d     = h2f_lo(w[0]);
    s.dmin  = h2f_hi(w[0]);
    s.sc_lo = w[1] & 0x3f3f3f3f;
    s.m_lo  = w[2] & 0x3f3f3f3f;
    s.sc_hi = (w[3] & 0x0f0f0f0f) | ((w[1] >> 2) & 0x30303030);
    s.m_hi  = ((w[3] >> 4) & 0x0f0f0f0f) | ((w[2] >> 2) & 0x30303030);
    return s;
}

// Q4_K (Q5 = false) and Q5_K (Q5 = true): sub-blocks 2p and 2p + 1 take the low and high nibbles of the same 8 code
// words (Q5_K: and bits 2p, 2p + 1 of the 8 qh words). Q4_K's high nibbles stay in place (unsigned dp4a on
// q & 0xF0F0F0F0), the factor 16 they carry moved into the row's FP32 scale.
template <int NC, int R, bool Q5>
static KCOL_HD __forceinline__ void unit_q45_K(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                  float (&acc)[R][NC]) {
    constexpr int T   = Q5 ? KQ5_K : KQ4_K;
    constexpr int QS0 = Q5 ? 12 : 4; // first word of the codes
    k_scales ks[R];
    uint32_t qh[R][Q5 ? 8 : 1];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        uint32_t w[4];
        unit_words<16, 4>(w, wu[i], 0);
        ks[i] = k_scales_of(w);
        if constexpr (Q5) {
            unit_words<16, 8>(qh[i], wu[i], 4);
        }
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t qs[R][8];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            unit_words<16, 8>(qs[i], wu[i], QS0 + 8 * p);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int s = 2 * p + h;
            int   q[R][8];
            float dsc[R];
            float mn[R];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int sc = (int) (((s < 4 ? ks[i].sc_lo : ks[i].sc_hi) >> (8 * (s & 3))) & 0xff);
                const int m  = (int) (((s < 4 ? ks[i].m_lo  : ks[i].m_hi ) >> (8 * (s & 3))) & 0xff);
                dsc[i] = ks[i].d * (float) sc;
                mn[i]  = -ks[i].dmin * (float) m;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    if constexpr (Q5) {
                        const uint32_t nib = h ? (qs[i][k] >> 4) & 0x0F0F0F0F : qs[i][k] & 0x0F0F0F0F;
                        const uint32_t hb  = s <= 4 ? (qh[i][k] << (4 - s)) & 0x10101010 : (qh[i][k] >> (s - 4)) & 0x10101010;
                        q[i][k] = (int) (nib | hb);
                    } else {
                        q[i][k] = (int) (h ? qs[i][k] & 0xF0F0F0F0 : qs[i][k] & 0x0F0F0F0F);
                    }
                }
                if constexpr (!Q5) {
                    if (h) {
                        dsc[i] *= 0.0625f;
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                act_blk a;
                load_act<T>(a, act + j * cstride, s);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    int isum = 0;
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        if (!Q5 && h) {
                            isum = dp4a_us((uint32_t) q[i][k], a.u[k], isum);
                        } else {
                            isum = dp4a(q[i][k], a.u[k], isum);
                        }
                    }
                    acc[i][j] = fmaf((float) isum * dsc[i], a.d, acc[i][j]);
                    acc[i][j] = fmaf(mn[i], a.s, acc[i][j]);
                }
            }
        }
    }
}

// The 8 indices of an IQ4_XS or IQ4_NL code word c (bytes 4k .. 4k + 3 of a 32-weight block's codes) looked up in
// kvalues_iq4nl with __byte_perm: weights (4k, 4k + 16, 4k + 1, 4k + 17) as q[0], (4k + 2, 4k + 18, 4k + 3, 4k + 19)
// as q[1] (act_pos<PERM_IQ4>'s order).
static KCOL_HD __forceinline__ void iq4nl_words(const uint32_t c, int * q) {
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u; // kvalues_iq4nl
    const uint32_t sel = 0x32103210 | ((c & 0x88888888) >> 1);
    q[0] = (int) prmt(prmt(t0, t1, c),       prmt(t2, t3, c),       sel);
    q[1] = (int) prmt(prmt(t0, t1, c >> 16), prmt(t2, t3, c >> 16), sel >> 16);
}

// IQ4_XS: block ib takes code words 2 + 4 ib .. 2 + 4 ib + 3; each word's 8 indices are looked up in kvalues_iq4nl
// with __byte_perm (two words of int8 values per code word, in act_pos<PERM_IQ4>'s order).
template <int NC, int R>
static KCOL_HD __forceinline__ void unit_iq4_xs(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                   float (&acc)[R][NC]) {
    float    d[R];
    uint32_t sh[R];
    uint32_t sl[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        uint32_t w[2];
        unit_words<8, 2>(w, wu[i], 0);
        d[i]  = h2f_lo(w[0]);
        sh[i] = w[0] >> 16;
        sl[i] = w[1];
    }
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        int   q[R][8];
        float dsc[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uint32_t c[4];
            unit_words<8, 4>(c, wu[i], 2 + 4 * ib);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                iq4nl_words(c[k], q[i] + 2 * k);
            }
            const int ls = (int) (((sl[i] >> (4 * ib)) & 0xF) | (((sh[i] >> (2 * ib)) & 3) << 4));
            dsc[i] = d[i] * (float) (ls - 32);
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            act_blk a;
            load_act<KIQ4_XS>(a, act + j * cstride, ib);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                int isum = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    isum = dp4a(q[i][k], a.u[k], isum);
                }
                acc[i][j] = fmaf((float) isum * dsc[i], a.d, acc[i][j]);
            }
        }
    }
}

// Q6_K: sub-block s = 4h + k (h = s / 4) takes the low (k < 2) or high nibbles of ql words 16h + 8 (k & 1) + j and bits
// 2k, 2k + 1 of qh words 32 + 8h + j; its two halves have the int8 scales of bytes 2s, 2s + 1 of the scale words
// 48..51. The offset -32 comes from the records' half sums, the dp4a chains' initial values.
template <int NC, int R>
static KCOL_HD __forceinline__ void unit_q6_K(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                 float (&acc)[R][NC]) {
    float d[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        uint32_t w[1];
        unit_words<2, 1>(w, wu[i], 52);
        d[i] = h2f_lo(w[0]);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        // sub-blocks 2p, 2p + 1: h = p / 2, k = 2 (p & 1) and 2 (p & 1) + 1
        const int hh = p >> 1;
        uint32_t ql[R][16];
        uint32_t qh[R][8];
        float    dsc[R][4];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            unit_words<2, 16>(ql[i], wu[i], 16 * hh);
            unit_words<2, 8>(qh[i], wu[i], 32 + 8 * hh);
            uint32_t sw[1];
            unit_words<2, 1>(sw, wu[i], 48 + p);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                dsc[i][e] = d[i] * (float) (int) (int8_t) (sw[0] >> (8 * e));
            }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int s = 2 * p + e;
            const int k = s & 3;
            int q[R][8];
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t l  = ql[i][8 * (k & 1) + j];
                    const uint32_t lo = k < 2 ? l & 0x0F0F0F0F : (l >> 4) & 0x0F0F0F0F;
                    const uint32_t hi = k == 0 ? (qh[i][j] << 4) & 0x30303030 : k == 1 ? (qh[i][j] << 2) & 0x30303030
                                      : k == 2 ? qh[i][j] & 0x30303030 : (qh[i][j] >> 2) & 0x30303030;
                    q[i][j] = (int) (lo | hi);
                }
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                act_blk a;
                load_act<KQ6_K>(a, act + j * cstride, s);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    int i0 = a.h0;
                    int i1 = a.h1;
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        i0 = dp4a(q[i][kk],     a.u[kk],     i0);
                        i1 = dp4a(q[i][kk + 4], a.u[kk + 4], i1);
                    }
                    acc[i][j] = fmaf(fmaf((float) i1, dsc[i][2 * e + 1], (float) i0 * dsc[i][2 * e]), a.d, acc[i][j]);
                }
            }
        }
    }
}

// Q3_K: hmask words 0..7, codes 8..23, scales 24..26, d in word 27. Block b = 4n + m of a super-block (n = b / 4,
// m = b % 4) takes bits 2m, 2m + 1 of code words 8 + 8n + k and bit b of hmask word k; its value is
// q + 4 hbit - 4. The codes are read in place (m < 3: masked, f = 4^m times their value; m = 3: shifted by 2, f = 16)
// with the high bit moved next to them, so that one mask, one shift and one LOP3 give each word; the records carry
// d / f and -4 f times the half sums, so the dp4a chains start at the offset. The 16 6-bit scales decode as in ggml.
template <int NC, int R>
static KCOL_HD __forceinline__ void unit_q3_K(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                 float (&acc)[R][NC]) {
    uint32_t hm[R][8];
    uint32_t sc[R][4];
    float    d[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        unit_words<2, 8>(hm[i], wu[i], 0);
        uint32_t w[4];
        unit_words<2, 4>(w, wu[i], 24);
        const uint32_t a0 = w[0], a1 = w[1], tmp = w[2];
        sc[i][0] = (a0 & 0x0f0f0f0f)        | (((tmp >> 0) & 0x03030303) << 4);
        sc[i][1] = (a1 & 0x0f0f0f0f)        | (((tmp >> 2) & 0x03030303) << 4);
        sc[i][2] = ((a0 >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
        sc[i][3] = ((a1 >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
        d[i] = h2f_lo(w[3]);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        // blocks 2p and 2p + 1: n = p / 2, m = 2 (p & 1) and 2 (p & 1) + 1
        const int n = p >> 1;
        uint32_t qs[R][8];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            unit_words<2, 8>(qs[i], wu[i], 8 + 8 * n);
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
                // scales 2b and 2b + 1: bytes 2 (b & 1) and 2 (b & 1) + 1 of scale word b / 2
                const uint32_t sw = sc[i][b >> 1] >> (16 * (b & 1));
                dsc0[i] = d[i] * (float) ((int) (sw & 0xff) - 32);
                dsc1[i] = d[i] * (float) ((int) ((sw >> 8) & 0xff) - 32);
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                act_blk a;
                load_act<KQ3_K>(a, act + j * cstride, b);
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    int i0 = a.h0;
                    int i1 = a.h1;
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        i0 = dp4a(q[i][kk],     a.u[kk],     i0);
                        i1 = dp4a(q[i][kk + 4], a.u[kk + 4], i1);
                    }
                    acc[i][j] = fmaf(fmaf((float) i1, dsc1[i], (float) i0 * dsc0[i]), a.d, acc[i][j]);
                }
            }
        }
    }
}

// Q4_0 (NL = false) and IQ4_NL (NL = true): two blocks of 18 bytes, as Q2_0's: d at bytes 0 and 18, 16 bytes of codes at
// 2..17 and 20..35 as 4 code words each; weight j < 16 of a block is the low nibble of code byte j, weight j + 16 its
// high nibble. Q4_0: the nibbles in place (weights 4k .. 4k + 3 and 4k + 16 .. 4k + 19 of code word k as words k and
// k + 4), the offset -8 from the records (the dp4a chain starts at -8 times the block's activation sum); IQ4_NL: the
// nibbles looked up in kvalues_iq4nl as IQ4_XS's.
template <int NC, int R, bool NL>
static KCOL_HD __forceinline__ void unit_q4_0(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                 float (&acc)[R][NC]) {
    constexpr int T = NL ? KIQ4_NL : KQ4_0;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        int   q[R][8];
        float d[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uint32_t w[5];
            uint32_t cw[4];
            if (c == 0) {
                unit_words<4, 5>(w, wu[i], 0);
                d[i] = h2f_lo(w[0]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    cw[k] = prmt(w[k], w[k + 1], 0x5432);
                }
            } else {
                unit_words<4, 5>(w, wu[i], 4);
                d[i] = h2f_hi(w[0]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    cw[k] = w[k + 1];
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if constexpr (NL) {
                    iq4nl_words(cw[k], q[i] + 2 * k);
                } else {
                    q[i][k]     = (int) (cw[k] & 0x0F0F0F0F);
                    q[i][k + 4] = (int) ((cw[k] >> 4) & 0x0F0F0F0F);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            act_blk a;
            load_act<T>(a, act + j * cstride, c);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                int isum = a.h0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    isum = dp4a(q[i][k], a.u[k], isum);
                }
                acc[i][j] = fmaf((float) isum * d[i], a.d, acc[i][j]);
            }
        }
    }
}

// Q5_0: two blocks of 22 bytes: d at bytes 0 and 22, the fifth bits qh at 2..5 and 24..27 (bit j: weight j), 16 bytes
// of codes at 6..21 and 28..43 (nibbles as Q4_0's). Word m of a block's int8 values holds weights m, m + 8, m + 16,
// m + 24 (act_pos<PERM_Q5>): their nibbles are the low and high nibbles of code bytes m and m + 8, which one
// __byte_perm gathers, and their fifth bits are bits m, m + 8, m + 16, m + 24 of qh, which one shift and mask move to
// bit 4 of each byte (as Q5_K's). The offset -16 comes from the records.
template <int NC, int R>
static KCOL_HD __forceinline__ void unit_q5_0(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                 float (&acc)[R][NC]) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        int   q[R][8];
        float d[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uint32_t w[6];
            uint32_t qh;
            uint32_t cw[4];
            if (c == 0) {
                unit_words<4, 6>(w, wu[i], 0);
                d[i] = h2f_lo(w[0]);
                qh   = prmt(w[0], w[1], 0x5432);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    cw[k] = prmt(w[k + 1], w[k + 2], 0x5432);
                }
            } else {
                unit_words<4, 6>(w, wu[i], 5);
                d[i] = h2f_hi(w[0]);
                qh   = w[1];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    cw[k] = w[k + 2];
                }
            }
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                // code bytes m, m + 8, m, m + 8: byte m % 4 of code words m / 4 and m / 4 + 2
                const uint32_t b   = (uint32_t) (m & 3);
                const uint32_t t   = prmt(cw[m >> 2], cw[(m >> 2) + 2], b | (b + 4) << 4 | b << 8 | (b + 4) << 12);
                const uint32_t nib = (t & 0x00000F0F) | ((t >> 4) & 0x0F0F0000);
                const uint32_t hb  = m <= 4 ? (qh << (4 - m)) & 0x10101010 : (qh >> (m - 4)) & 0x10101010;
                q[i][m] = (int) (nib | hb);
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            act_blk a;
            load_act<KQ5_0>(a, act + j * cstride, c);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                int isum = a.h0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    isum = dp4a(q[i][k], a.u[k], isum);
                }
                acc[i][j] = fmaf((float) isum * d[i], a.d, acc[i][j]);
            }
        }
    }
}

// The grid types and IQ1_M (Q713): one super-block per unit. Each row's super-block is read into registers once, each of
// its 32-weight blocks decoded once (mmvqp_iq::decode32: 8 words of signed int8 weights, the integer scales of its two
// halves) and dotted with every column: per 32 weights, row and column 8 dp4a and 4 FP32 operations, as Q6_K. tab: the
// type's grid in shared memory.
template <int NC, int R, int T>
static KCOL_HD __forceinline__ void unit_iq(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                            float (&acc)[R][NC], const uint32_t * tab) {
    constexpr int G  = gtype_of<T>();
    constexpr int NW = mmvqp_iq::gtraits<G>::NW;
    static_assert(G >= 0 && traits<T>::UB == mmvqp_iq::gtraits<G>::BS, "a grid type or IQ1_M");
    uint32_t w[R][NW];
    float    dd[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        unit_words<traits<T>::AL, NW>(w[i], wu[i], 0);
        dd[i] = mmvqp_iq::sblock_scale<G>(w[i]);
    }
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        int   q[R][8];
        float d0[R];
        float d1[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            int ls0, ls1;
            mmvqp_iq::decode32<G>(w[i], ib, tab, q[i], ls0, ls1);
            d0[i] = dd[i] * (float) ls0;
            d1[i] = dd[i] * (float) ls1;
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            act_blk a;
            load_act<T>(a, act + j * cstride, ib);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                int i0 = 0;
                int i1 = 0;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    i0 = dp4a(q[i][kk],     a.u[kk],     i0);
                    i1 = dp4a(q[i][kk + 4], a.u[kk + 4], i1);
                }
                acc[i][j] = fmaf(fmaf((float) i1, d1[i], (float) i0 * d0[i]), a.d, acc[i][j]);
            }
        }
    }
}

template <int T, int NC, int R>
static KCOL_HD __forceinline__ void unit_dot(const unit_ptr (&wu)[R], const uint32_t * act, const int cstride,
                                                float (&acc)[R][NC], const uint32_t * tab = nullptr) {
    (void) tab;
    if constexpr (gtype_of<T>() >= 0) {
        unit_iq<NC, R, T>(wu, act, cstride, acc, tab);
    } else if constexpr (T == KQ8_0) {
        unit_q8_0<NC, R>(wu, act, cstride, acc);
    } else if constexpr (T == KQ2_0) {
        unit_q2_0<NC, R>(wu, act, cstride, acc);
    } else if constexpr (T == KQ3_K) {
        unit_q3_K<NC, R>(wu, act, cstride, acc);
    } else if constexpr (T == KQ4_K) {
        unit_q45_K<NC, R, false>(wu, act, cstride, acc);
    } else if constexpr (T == KQ5_K) {
        unit_q45_K<NC, R, true>(wu, act, cstride, acc);
    } else if constexpr (T == KQ6_K) {
        unit_q6_K<NC, R>(wu, act, cstride, acc);
    } else if constexpr (T == KIQ4_XS) {
        unit_iq4_xs<NC, R>(wu, act, cstride, acc);
    } else if constexpr (T == KQ4_0) {
        unit_q4_0<NC, R, false>(wu, act, cstride, acc);
    } else if constexpr (T == KIQ4_NL) {
        unit_q4_0<NC, R, true>(wu, act, cstride, acc);
    } else {
        unit_q5_0<NC, R>(wu, act, cstride, acc);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The product kernel

struct args {
    const char     * vx;       // weights: nrows rows of row_bytes bytes
    const uint32_t * rec;      // activation records: column j, unit u at word (j * nunits + u) * RS
    float          * dst;      // column j at j * stride_col_dst
    int              nrows;
    int              row_bytes;
    int              nunits;   // lane units per row
    int              lpr_log2; // lanes per row group L = 1 << lpr_log2
    int              niter;    // iterations per row (nunits / L)
    int              itt;      // iterations per activation tile
    int              gen;      // 0: each lane group stages its own rows' 16-byte segments; 1: the warp stages all its rows
    int              seg_log2; // gen: 16-byte units per row slot = 1 << seg_log2
    int              stride_col_dst;
    const uint32_t * grid;     // grid types and IQ1_M: the grid in global memory (see grid_src)
};

// Where the kernels of a grid type (or IQ1_M) copy its grid from into shared memory: args::grid, unless the file that
// launches them specializes this for the type (the fork's mmvq-pascal-kcol-<type>.cu: ggml's own device tables).
template <int T> struct grid_src {
    static __device__ __forceinline__ const uint32_t * get(const args & p) { return p.grid; }
};

// Bytes of one warp's staging buffer.
inline __host__ __device__ int stage_bytes(const int UB, const int lpr_log2, const int R, const int gen, const int seg_log2) {
    const int rw = (32 >> lpr_log2) * R;
    return gen ? rw * (16 << seg_log2) : rw * (UB << lpr_log2);
}

// Grid: (row blocks, 1, 1), KCOL_NT threads; dynamic shared memory: the activation tile (NC columns of itt * L
// records), then the 4 warps' staging buffers, then (grid types and IQ1_M) the type's grid.
template <int T, int NC, int R>
__launch_bounds__(KCOL_NT, KCOL_MIN_BLOCKS)
static __global__ void mul_mat_kcol(const args p) {
    constexpr int UB = traits<T>::UB;
    constexpr int AL = traits<T>::AL;
    constexpr int RS = rec_words<T>();
    constexpr int GW = grid_words<T>();
    extern __shared__ uint4 kcol_sm[];

    const int lane     = threadIdx.x % 32;
    const int wid      = threadIdx.x / 32;
    const int lpr_log2 = p.lpr_log2;
    const int L        = 1 << lpr_log2;
    const int sl       = lane & (L - 1);
    const int grp      = lane >> lpr_log2;
    const int rw       = (32 >> lpr_log2) * R;                // rows per warp
    const int wrow0    = (blockIdx.x * KCOL_NW + wid) * rw;  // the warp's first row
    const int row0     = wrow0 + grp * R;                     // the lane group's first row
    const int seg      = UB << lpr_log2;                      // bytes of a row's segment per iteration
    const int nut      = p.itt << lpr_log2;                   // records per column and tile
    const int cstride  = nut * RS;                            // words between the columns of the tile

    uint32_t * sm_act = (uint32_t *) kcol_sm;
    char     * sm_w   = (char *) kcol_sm + NC * cstride * 4 +
        wid * stage_bytes(UB, lpr_log2, R, p.gen, p.seg_log2);
    const int slot = p.gen ? 16 << p.seg_log2 : seg;          // bytes per row in the staging buffer
    [[maybe_unused]] uint32_t * sm_tab = nullptr;              // the grid, copied before the first tile's barrier
    if constexpr (GW > 0) {
        sm_tab = (uint32_t *) ((char *) kcol_sm + NC * cstride * 4 + KCOL_NW * stage_bytes(UB, lpr_log2, R, p.gen, p.seg_log2));
        const uint32_t * src = grid_src<T>::get(p);
        for (int i = threadIdx.x; i < GW; i += KCOL_NT) {
            sm_tab[i] = __ldg(src + i);
        }
    }

    // the lane group's rows (clamped, so that segment staging reads valid rows; their results are not written)
    const char * rw_x[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        rw_x[i] = p.vx + (int64_t) min(row0 + i, p.nrows - 1) * p.row_bytes;
    }

    float acc[R][NC];
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int t0 = 0; t0 < p.niter; t0 += p.itt) {
        const int nit = min(p.itt, p.niter - t0);
        // the tile's activation records, column by column (contiguous runs of nit * L records)
        if (t0 > 0) {
            __syncthreads();
        }
        {
            const int n4 = (nit << lpr_log2) * RS / 4;
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const uint4 * src = (const uint4 *) (p.rec + ((int64_t) j * p.nunits + (t0 << lpr_log2)) * RS);
                uint4       * dsm = (uint4 *) (sm_act + j * cstride);
                for (int v = threadIdx.x; v < n4; v += KCOL_NT) {
                    dsm[v] = __ldg(src + v);
                }
            }
        }
        __syncthreads();

        for (int it = 0; it < nit; ++it) {
            const int seg_off = (t0 + it) * seg;
            unit_ptr  wu[R];
            if (p.gen) {
                // the warp stages every row's segment from the 16-byte boundary below it, rows (32 >> seg_log2) at a time
                const int nup  = 1 << p.seg_log2;
                const int rpp  = nup >= 32 ? 1 : 32 >> p.seg_log2;
                const int k0   = nup >= 32 ? lane : lane & (nup - 1);
                const int kstp = nup >= 32 ? 32 : nup;
                for (int r = nup >= 32 ? 0 : lane >> p.seg_log2; r < rw; r += rpp) {
                    if (wrow0 + r >= p.nrows) {
                        break;
                    }
                    const char *    g  = p.vx + (int64_t) (wrow0 + r) * p.row_bytes + seg_off;
                    const int       h  = (int) ((uintptr_t) g & 15);
                    const uint4 *   ga = (const uint4 *) (g - h);
                    const int       nu = (h + seg + 15) >> 4;
                    uint4 *         ds = (uint4 *) (sm_w + r * slot);
                    for (int k = k0; k < nu; k += kstp) {
                        ds[k] = __ldg(ga + k);
                    }
                }
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const int h = (int) ((uintptr_t) (rw_x[i] + seg_off) & 15);
                    wu[i] = make_unit<AL>(sm_w + (grp * R + i) * slot + h + sl * UB);
                }
            } else {
                // each lane group stages its own rows' segments, whole 16-byte units
                const int nu16 = seg >> 4;
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const uint4 * src = (const uint4 *) (rw_x[i] + seg_off);
                    uint4       * ds  = (uint4 *) (sm_w + (grp * R + i) * slot);
                    for (int k = sl; k < nu16; k += L) {
                        ds[k] = __ldg(src + k);
                    }
                }
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    wu[i] = make_unit<AL>(sm_w + (grp * R + i) * slot + sl * UB);
                }
            }
            __syncwarp();
            unit_dot<T, NC, R>(wu, sm_act + ((it << lpr_log2) + sl) * RS, cstride, acc, sm_tab);
            __syncwarp();
        }
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                if (off < L) {
                    acc[i][j] += __shfl_xor_sync(0xFFFFFFFF, acc[i][j], off, 32);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int row = row0 + i;
        if ((i & (L - 1)) != sl || row >= p.nrows) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            p.dst[(int64_t) j * p.stride_col_dst + row] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Launch shape and launch

struct shape {
    int lpr_log2;
    int R;
    int itt;
    int gen;
    int seg_log2;
    int smem;
    int nblocks;
    int fallback; // 1: pick_shape's defaults found no tile within a third of the SM and took the shape before Q823
};

// Resident blocks per SM that the registers and threads of the kernel of type t for ncols columns and R rows per lane
// group allow (no shared memory counted), or 0 where that is unknown: the caller's query of the compiled kernel
// (kernel_blocks below).
typedef int (*blocks_fn)(int t, int ncols, int R);

// The launch shape for ncols columns: L the largest power of two (at most 32) that divides the row's units, then, by
// default (Q823, from the P106's bench of every launch shape at Flash-Next's shapes and the in-model passes: the 32 KB
// budget alone made every k-token pass 7.5-21.8 ms faster, docs/data/2026-11-04_kshape):
//   - R = 2 rows per lane group only for 2 or more columns, where that leaves at least 8 blocks per SM, its registers
//     allow 3 blocks per SM and its shape fits a third of the SM's shared memory; else R = 1. The grid types and IQ1_M
//     always take R = 1 (at R = 2 their decoding takes most of the registers).
//   - The largest activation tile (in iterations) that fits a third of the SM's shared memory (smem_sm / 3: 32 KB on a
//     P106, so that a third block fits beside two), unless the kernel's registers allow only 2 blocks per SM (the grid
//     types from about 4 columns on) or the grid has at most 2 blocks per SM: then the largest that fits 48 KB, since
//     the smaller tile would buy no resident block.
//   - Where no tile fits that, the shape before Q823 (below), with s.fallback = 1 (mmvq-pascal-kcol.cu leaves such
//     products of 2 or more columns to the other kernels, Q824).
// r_force, itt_force or smem_budget (GGML_CUDA_MMVQ_PASCAL_KCOL_R, _ITT, _SMEM) select the shape as before Q823: R rows
// per lane group (r_force if given, else 2, or 1 where 2 leaves fewer than 3 blocks per SM or no activation tile fits, and
// for the 2-byte-aligned grid types); the largest activation tile (at most itt_force) that fits smem_budget (48 KB if not
// given) next to the staging buffers (and the grid of a grid type). _SMEM=49152 is the default before Q823.
// smem_sm: shared memory per SM (0: 96 KB, as on every Pascal card of compute capability 6.1); blocks: see blocks_fn
// (nullptr: registers taken to allow 3 blocks).
// Returns false where the row width or the alignment doesn't fit the kernels.
inline bool pick_shape(const int t, const int K, const int nrows, const int ncols, const int row_bytes, const uintptr_t x_addr,
                       const int nsm, const int r_force, const int itt_force, const int smem_budget, shape & s,
                       const int smem_sm = 0, const blocks_fn blocks = nullptr) {
    const type_info ti = info(t);
    if (ti.UW == 0 || K % ti.UW != 0 || ncols < 1 || ncols > KCOL_MAX_COLS || nrows < 1) {
        return false;
    }
    if (row_bytes % ti.AL != 0 || x_addr % ti.AL != 0) {
        return false;
    }
    const int nunits = K / ti.UW;
    int lpr_log2 = 0;
    while (lpr_log2 < 5 && nunits % (2 << lpr_log2) == 0) {
        lpr_log2++;
    }
    // segment staging needs whole 16-byte units; 2-byte-aligned units read 4 bytes past their end, which only the warp
    // staging's row slots leave room for
    const int seg = ti.UB << lpr_log2;
    const int gen = (ti.AL == 2 || seg % 16 != 0 || row_bytes % 16 != 0 || x_addr % 16 != 0) ? 1 : 0;
    if (gen && ti.AL == 16) {
        return false;
    }
    int seg_log2 = 0;
    if (gen) {
        // the segment from the 16-byte boundary below it, plus the words a 2-byte-aligned unit reads past its end
        const int nu = (15 + seg + 8 + 15) / 16;
        while ((1 << seg_log2) < nu) {
            seg_log2++;
        }
    }
    const int niter = nunits >> lpr_log2;
    const int act_per_iter = ncols * (1 << lpr_log2) * ti.RS * 4;
    const auto nblocks_of = [&](const int R) {
        const int rows_per_block = KCOL_NW * (32 >> lpr_log2) * R;
        return (nrows + rows_per_block - 1) / rows_per_block;
    };
    // R rows per lane group and the largest activation tile (at most itt_max iterations if given) that fits budget
    const auto fit = [&](const int R, const int budget, const int itt_max) {
        const int wbytes = KCOL_NW * stage_bytes(ti.UB, lpr_log2, R, gen, seg_log2) + grid_bytes(t);
        if (wbytes + act_per_iter > budget) {
            return false;
        }
        int itt = (budget - wbytes) / act_per_iter;
        itt = itt < niter ? itt : niter;
        if (itt_max > 0) {
            itt = itt_max < itt ? itt_max : itt;
        }
        s.lpr_log2 = lpr_log2;
        s.R        = R;
        s.itt      = itt;
        s.gen      = gen;
        s.seg_log2 = seg_log2;
        s.smem     = wbytes + ncols * itt * (1 << lpr_log2) * ti.RS * 4;
        s.nblocks  = nblocks_of(R);
        s.fallback = 0;
        return true;
    };
    // the shape before Q823: the grid types start at R = 1 (at R = 2 their decoding takes all 255 registers, some spill,
    // and their shared memory allows 2 blocks per SM either way)
    const auto pick_old = [&](const int budget) {
        const int r_max = grid_bytes(t) > 0 && ti.AL == 2 ? 1 : 2;
        for (int R = r_force > 0 ? r_force : r_max; R >= 1; --R) {
            if (r_force <= 0 && R > 1 && nblocks_of(R) < 3 * nsm) {
                continue;
            }
            if (fit(R, budget, itt_force)) {
                return true;
            }
            if (r_force > 0) {
                return false;
            }
        }
        return false;
    };
    if (r_force > 0 || itt_force > 0 || smem_budget > 0) {
        return pick_old(smem_budget > 0 && smem_budget <= KCOL_SMEM_MAX ? smem_budget : KCOL_SMEM_MAX);
    }

    // Q823: a third of the SM's shared memory per block where the registers allow a third block
    const int third = (smem_sm > 0 ? smem_sm : 96 * 1024) / 3;
    const int b3    = third < KCOL_SMEM_MAX ? third : KCOL_SMEM_MAX;
    const auto regs_allow_3 = [&](const int R) {
        const int b = blocks ? blocks(t, ncols, R) : 0;
        return b <= 0 || b >= 3;
    };
    if (grid_bytes(t) == 0 && ncols >= 2 && nblocks_of(2) >= 8 * nsm && regs_allow_3(2) && fit(2, b3, 0)) {
        return true;
    }
    if (fit(1, regs_allow_3(1) && nblocks_of(1) > 2 * nsm ? b3 : KCOL_SMEM_MAX, 0)) {
        return true;
    }
    if (!pick_old(KCOL_SMEM_MAX)) {
        return false;
    }
    s.fallback = 1;
    return true;
}

// C805 (GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT): a launch plan for a product of ncols >= 2 columns that pick_shape declined
// (one iteration of the columns' activation records and the staging buffers pass the budget). A column's sums depend on
// the row width only (above), not on the launch shape or the column count, so the product may run as several launches
// of cols columns each (the last one may have fewer), all with shape s, from the one quantizer launch: ncols split into
// k launches of ceil(ncols / k) columns, k the smallest that fits. mode 1: the budget of 48 KB (all ncols in one launch
// where they fit: the default rule's 48 KB step, pick_old above); mode 2: first launches whose shape fits smem_budget
// (0: the default rule), from 2 launches on, then as mode 1. r_force and itt_force as for pick_shape; where the forced R
// leaves nothing, R is chosen as without it. Returns false where not even one column fits (row width, alignment).
inline bool pick_exact(const int t, const int K, const int nrows, const int ncols, const int row_bytes, const uintptr_t x_addr,
                       const int nsm, const int r_force, const int itt_force, const int smem_budget, const int mode,
                       shape & s, int & cols, const int smem_sm = 0, const blocks_fn blocks = nullptr) {
    const auto try_cols = [&](const int c, const int budget) {
        if (pick_shape(t, K, nrows, c, row_bytes, x_addr, nsm, r_force, itt_force, budget, s, smem_sm, blocks) ||
                (r_force > 0 && pick_shape(t, K, nrows, c, row_bytes, x_addr, nsm, 0, itt_force, budget, s, smem_sm, blocks))) {
            cols = c;
            return true;
        }
        return false;
    };
    for (int pass = mode == 2 ? 0 : 1; pass < 2; ++pass) {
        int prev = 0;
        for (int k = pass == 0 ? 2 : 1; k <= ncols; ++k) {
            const int c = (ncols + k - 1) / k;
            if (c == prev) {
                continue;
            }
            prev = c;
            if (try_cols(c, pass == 0 ? smem_budget : KCOL_SMEM_MAX)) {
                return true;
            }
        }
    }
    return false;
}

#ifdef __CUDACC__
// Resident blocks per SM that the registers and threads of mul_mat_kcol<T, NC, R> allow on the current device
// (cudaOccupancyMaxActiveBlocksPerMultiprocessor without dynamic shared memory), queried once per kernel; 0 if the query
// fails. For pick_shape's blocks_fn, from the file that compiles the type's kernels.
template <int T, int NC, int R>
inline int kernel_blocks_nc() {
    static const int b = [] {
        int n = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mul_mat_kcol<T, NC, R>, KCOL_NT, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            n = 0;
        }
        return n;
    }();
    return b;
}

template <int T, int R>
inline int kernel_blocks_r(const int ncols) {
    switch (ncols) {
        case 1:  return kernel_blocks_nc<T, 1, R>();
        case 2:  return kernel_blocks_nc<T, 2, R>();
        case 3:  return kernel_blocks_nc<T, 3, R>();
        case 4:  return kernel_blocks_nc<T, 4, R>();
        case 5:  return kernel_blocks_nc<T, 5, R>();
        case 6:  return kernel_blocks_nc<T, 6, R>();
        case 7:  return kernel_blocks_nc<T, 7, R>();
        default: return kernel_blocks_nc<T, 8, R>();
    }
}

template <int T>
inline int kernel_blocks(const int ncols, const int R) {
    return R == 1 ? kernel_blocks_r<T, 1>(ncols) : kernel_blocks_r<T, 2>(ncols);
}
#endif // __CUDACC__

// Bytes of the activation records of ncols columns.
inline size_t records_bytes(const int t, const int K, const int ncols) {
    const type_info ti = info(t);
    return (size_t) ncols * (K / ti.UW) * ti.RS * 4;
}

struct params {
    const char * vx;
    const float * y;        // FP32 activations, ncols columns stride_col_y floats apart
    uint32_t    * rec;      // records_bytes(type, K, ncols) bytes
    float       * dst;
    int           K;
    int           nrows;
    int           row_bytes;
    int           ncols;
    int           stride_col_y;
    int           stride_col_dst;
    shape         s;
    const uint32_t * grid = nullptr; // grid types and IQ1_M: args::grid
    int           cols  = 0;       // C805: columns per launch (pick_exact; 0: all ncols in one launch)
    int           fault = 0;       // C805's check: every launch after the first reads the first launch's records
};

template <int T, int NC, int R>
inline void launch_kernel(const args & a, const shape & s, cudaStream_t stream) {
    mul_mat_kcol<T, NC, R><<<s.nblocks, KCOL_NT, s.smem, stream>>>(a);
}

template <int T, int R>
inline void launch_cols(const args & a, const shape & s, const int ncols, cudaStream_t stream) {
    switch (ncols) {
        case 1: launch_kernel<T, 1, R>(a, s, stream); break;
        case 2: launch_kernel<T, 2, R>(a, s, stream); break;
        case 3: launch_kernel<T, 3, R>(a, s, stream); break;
        case 4: launch_kernel<T, 4, R>(a, s, stream); break;
        case 5: launch_kernel<T, 5, R>(a, s, stream); break;
        case 6: launch_kernel<T, 6, R>(a, s, stream); break;
        case 7: launch_kernel<T, 7, R>(a, s, stream); break;
        default: launch_kernel<T, 8, R>(a, s, stream); break;
    }
}

// The quantizer, then the product.
template <int T>
inline void launch(const params & p, cudaStream_t stream) {
    const int nunits = p.K / traits<T>::UW;
    const dim3 qgrid((p.K / 32 + KCOL_NT / 32 - 1) / (KCOL_NT / 32), p.ncols, 1);
    quantize_records<T><<<qgrid, KCOL_NT, 0, stream>>>(p.y, p.rec, p.K, p.stride_col_y, nunits);

    args a;
    a.vx             = p.vx;
    a.rec            = p.rec;
    a.dst            = p.dst;
    a.nrows          = p.nrows;
    a.row_bytes      = p.row_bytes;
    a.nunits         = nunits;
    a.lpr_log2       = p.s.lpr_log2;
    a.niter          = nunits >> p.s.lpr_log2;
    a.itt            = p.s.itt;
    a.gen            = p.s.gen;
    a.seg_log2       = p.s.seg_log2;
    a.stride_col_dst = p.stride_col_dst;
    a.grid           = p.grid;
    // C805: p.cols columns per launch (one launch for all ncols where p.cols is 0 or ncols)
    const int step = p.cols > 0 && p.cols < p.ncols ? p.cols : p.ncols;
    for (int j0 = 0; j0 < p.ncols; j0 += step) {
        const int nc = p.ncols - j0 < step ? p.ncols - j0 : step;
        args aj = a;
        aj.rec = p.rec + (p.fault ? 0 : (size_t) j0 * nunits * rec_words<T>());
        aj.dst = p.dst + (int64_t) j0 * p.stride_col_dst;
        if (p.s.R == 1) {
            launch_cols<T, 1>(aj, p.s, nc, stream);
        } else {
            launch_cols<T, 2>(aj, p.s, nc, stream);
        }
    }
}

} // namespace kcol
