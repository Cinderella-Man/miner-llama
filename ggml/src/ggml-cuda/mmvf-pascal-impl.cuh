#pragma once

// The kernel of mmvf-pascal.cu without ggml's headers, so that a standalone benchmark can time the same code
// (bench/mmvq_pascal --bf16 in llms-on-crypto-mining-rigs carries a copy).

#include <cstdint>

#define MMVF_PASCAL_NT 128 // threads per block (4 warps)
// MMVF_PASCAL_MIN_BLOCKS: blocks per SM the compiler keeps registers for (__launch_bounds__); 0: no limit
#ifndef MMVF_PASCAL_MIN_BLOCKS
#define MMVF_PASCAL_MIN_BLOCKS 0
#endif
#if MMVF_PASCAL_MIN_BLOCKS > 0
#define MMVF_PASCAL_BOUNDS __launch_bounds__(MMVF_PASCAL_NT, MMVF_PASCAL_MIN_BLOCKS)
#else
#define MMVF_PASCAL_BOUNDS __launch_bounds__(MMVF_PASCAL_NT)
#endif
// MMVF_PASCAL_X4_MIN_BLOCKS: the same for mul_mat_vec_bf16_pascal_x4 (default 4: at most 128 registers a thread, so the
// 20 blocks of a 10,240 x 320 product run in one wave on a 5-SM P106-090); 0: no limit
#ifndef MMVF_PASCAL_X4_MIN_BLOCKS
#define MMVF_PASCAL_X4_MIN_BLOCKS 4
#endif
#if MMVF_PASCAL_X4_MIN_BLOCKS > 0
#define MMVF_PASCAL_X4_BOUNDS __launch_bounds__(MMVF_PASCAL_NT, MMVF_PASCAL_X4_MIN_BLOCKS)
#else
#define MMVF_PASCAL_X4_BOUNDS __launch_bounds__(MMVF_PASCAL_NT)
#endif

// 8 BF16 weights (one 16-byte load) times 8 FP32 activations, added to acc one weight at a time in column order. A
// BF16 value is the upper half of an FP32 one, so moving it into place converts it exactly.
static __device__ __forceinline__ float mmvfp_dot8(const uint4 w, const float4 a, const float4 b, float acc) {
    acc = fmaf(__uint_as_float(w.x << 16),         a.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xFFFF0000u), a.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16),         a.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xFFFF0000u), a.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16),         b.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xFFFF0000u), b.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16),         b.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xFFFF0000u), b.w, acc);
    return acc;
}

// G lanes per row group (8, 32, or 128, which spans the block's 4 warps), R rows per group, T tokens.
// x: nrows rows of nchunks 16-byte chunks (8 weights each), stride_row chunks apart. y: T columns of FP32 activations,
// stride_col_y4 float4 apart. dst, and bias if has_bias: T columns, stride_col_dst floats apart.
// Lane sl of a group takes chunks sl, sl + G, ... of each of its R rows, so every weight load of a warp is a coalesced
// 128-bit load; the activations of a chunk are loaded once for all R rows. Partial sums are reduced with shuffles in
// the warp, then (G = 128) over the warps through shared memory, always in the same order; after the shuffles every lane
// of a group holds all T * R sums, and lane lg writes those whose index t * R + i is lg modulo the group's lanes in the
// warp, so a lane writes several outputs where T * R exceeds them.
template <int G, int R, int T, bool has_bias>
MMVF_PASCAL_BOUNDS
static __global__ void mul_mat_vec_bf16_pascal(
        const uint4 * __restrict__ x, const float4 * __restrict__ y, const float * __restrict__ bias, float * __restrict__ dst,
        const int nrows, const int nchunks, const int stride_row, const int stride_col_y4, const int stride_col_dst) {
    constexpr int warp = 32;
    constexpr int NW   = MMVF_PASCAL_NT / warp;
    constexpr int GW   = G < warp ? G : warp;     // lanes of a group within one warp
    constexpr int WPG  = G > warp ? G / warp : 1; // warps per group
    static_assert(G == 8 || G == 32 || G == 128, "unsupported group size");

    const int tid  = threadIdx.x;
    const int sl   = tid % G;
    const int lg   = tid % GW;
    const int row0 = (blockIdx.x * (MMVF_PASCAL_NT / G) + tid / G) * R;

    const uint4 * xr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        xr[i] = x + (int64_t) min(row0 + i, nrows - 1) * stride_row + sl;
    }
    const float4 * yp = y + 2 * sl;

    float acc[T][R];
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[t][i] = 0.0f;
        }
    }

#pragma unroll 2
    for (int c = sl; c < nchunks; c += G) {
        uint4 w[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            w[i] = __ldg(xr[i]);
            xr[i] += G;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 a = __ldg(yp + t * stride_col_y4);
            const float4 b = __ldg(yp + t * stride_col_y4 + 1);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[t][i] = mmvfp_dot8(w[i], a, b, acc[t][i]);
            }
        }
        yp += 2 * G;
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int off = GW / 2; off > 0; off >>= 1) {
                acc[t][i] += __shfl_xor_sync(0xFFFFFFFF, acc[t][i], off, warp);
            }
        }
    }

    if constexpr (WPG > 1) {
        __shared__ float buf[NW][T * R];
        const int wid = tid / warp;
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if ((t * R + i) % GW == lg) {
                    buf[wid][t * R + i] = acc[t][i];
                }
            }
        }
        __syncthreads();
        if (wid % WPG != 0) {
            return;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if ((t * R + i) % GW == lg) {
#pragma unroll
                    for (int k = 1; k < WPG; ++k) {
                        acc[t][i] += buf[wid + k][t * R + i];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = row0 + i;
            if ((t * R + i) % GW != lg || row >= nrows) {
                continue;
            }
            float v = acc[t][i];
            if constexpr (has_bias) {
                v += bias[t * stride_col_dst + row];
            }
            dst[t * stride_col_dst + row] = v;
        }
    }
}

// The 128-lane kernel's sums with 32 lanes (C750): mul_mat_vec_bf16_pascal<128, R', T'>'s bits for every output, for any
// R' and T', with the loads of mul_mat_vec_bf16_pascal<32, R, T>. Lane j of a 32-lane group takes the 32-lane kernel's
// chunks j, j + 32, j + 64, ...; its n-th chunk, j + 32 n, is chunk n / 4 of the 128-lane kernel's lane j + 32 (n % 4),
// so it keeps four accumulators per output, acc[v] for lane j + 32 v, and adds each chunk to acc[n % 4] with the same
// FMAs. Each acc[v] is then reduced over the warp with the 128-lane kernel's butterfly within its warp v, and the four
// sums are added 0 + 1 + 2 + 3, as there warp 0 adds warps 1, 2 and 3. 4 T R accumulators a lane, 4 times the
// butterflies of the 32-lane kernel; no shared memory. fault (checks only, GGML_CUDA_MMVF_PASCAL_EXACT_FAULT=1) leaves
// acc[3] out of the sum, a quarter of the terms, which every check of the product must catch.
template <int R, int T, bool fault>
MMVF_PASCAL_X4_BOUNDS
static __global__ void mul_mat_vec_bf16_pascal_x4(
        const uint4 * __restrict__ x, const float4 * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int nchunks, const int stride_row, const int stride_col_y4, const int stride_col_dst) {
    constexpr int warp = 32;
    constexpr int V    = 4; // the 128-lane kernel's warps

    const int tid  = threadIdx.x;
    const int lg   = tid % warp;
    const int row0 = (blockIdx.x * (MMVF_PASCAL_NT / warp) + tid / warp) * R;

    const uint4 * xr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        xr[i] = x + (int64_t) min(row0 + i, nrows - 1) * stride_row + lg;
    }
    const float4 * yp = y + 2 * lg;

    float acc[V][T][R];
#pragma unroll
    for (int v = 0; v < V; ++v) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[v][t][i] = 0.0f;
            }
        }
    }

    // chunk c + 32 v into acc[v]
    auto step = [&](const int v) {
        uint4 w[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            w[i] = __ldg(xr[i] + v * warp);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 a = __ldg(yp + 2 * v * warp + t * stride_col_y4);
            const float4 b = __ldg(yp + 2 * v * warp + t * stride_col_y4 + 1);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[v][t][i] = mmvfp_dot8(w[i], a, b, acc[v][t][i]);
            }
        }
    };

    int c = lg;
#pragma unroll 1
    for (; c + (V - 1) * warp < nchunks; c += V * warp) {
#pragma unroll
        for (int v = 0; v < V; ++v) {
            step(v);
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            xr[i] += V * warp;
        }
        yp += 2 * V * warp;
    }
    // at most V - 1 chunks left, into acc[0], acc[1], ...
#pragma unroll
    for (int v = 0; v < V - 1; ++v) {
        if (c + v * warp < nchunks) {
            step(v);
        }
    }

#pragma unroll
    for (int v = 0; v < V; ++v) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int off = warp / 2; off > 0; off >>= 1) {
                    acc[v][t][i] += __shfl_xor_sync(0xFFFFFFFF, acc[v][t][i], off, warp);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = row0 + i;
            if ((t * R + i) % warp != lg || row >= nrows) {
                continue;
            }
            float s = acc[0][t][i];
            s += acc[1][t][i];
            s += acc[2][t][i];
            if constexpr (!fault) {
                s += acc[3][t][i];
            }
            dst[t * stride_col_dst + row] = s;
        }
    }
}
