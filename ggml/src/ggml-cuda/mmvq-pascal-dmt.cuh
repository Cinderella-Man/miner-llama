#pragma once

#include "common.cuh"

// Plain IQ4_XS and Q4_K products of 2 to 8 tokens (MUL_MAT, no fusion) on compute capability 6.1 that read and decode
// each weight once for all tokens and give every token, bit for bit, the sums of the one-token Pascal kernel
// (mul_mat_vec_q_pascal, mmvq-pascal.cu): Qwen3.8-Flash-Next's attention and recurrent projections in a speculative
// verify, which the k-column kernel runs at one row per lane group with sums other than the one-token kernel's (C722,
// Q1277; C690's Q5_K kernel, mmvq-pascal-q5k.cu, carried to the other types). The design is at the top of
// mmvq-pascal-dmt.cu. 5 to 8 tokens: GGML_CUDA_MMVQ_PASCAL_RO8 (C729; 1 on, 2 without the prefetch; R = 1 only).
//
// GGML_CUDA_MMVQ_PASCAL_DMT: 0 (default) off, the k-column and the earlier kernels as before; 1 on, at one row per lane
// group with the register prefetch of each warp's next weight segments, except IQ4_XS's 2 tokens at K = 2560, which run
// without it (since C758: on a P106 the fastest at Flash-Next's shapes or within the spread of it, R79); 2 on without
// the prefetch.
// GGML_CUDA_MMVQ_PASCAL_DMT_R: rows per lane group, 1 (default since C758) or 2, which share the activation loads; 0
// chooses per product as C722 did (2 where a block holds as many rows as at 1). GGML_CUDA_MMVQ_PASCAL_DMT_PF=1 (with 1):
// the prefetch for every product, as C722 did. GGML_CUDA_MMVQ_PASCAL_DMT_NW=n: warps per block at most (1 to 6 at R = 1,
// 1 to 3 at R = 2; fewer where the block's shared memory would pass 48 KB). GGML_CUDA_MMVQ_PASCAL_DMT_SKIP=q4_K: types
// left to the k-column kernel. The kernels follow GGML_CUDA_MMVQ_PASCAL (0 off, 2 any NVIDIA GPU) and
// GGML_CUDA_MMVQ_PASCAL_SKIP, as the one-token kernel does.

// Whether ggml_cuda_mmvq_pascal_dmt takes a product of ncols columns of a matrix of type with K weights per row, nrows
// rows row_bytes apart from x, on the current device (the k-column kernels leave such products to it).
bool ggml_cuda_mmvq_pascal_dmt_takes(ggml_type type, int64_t K, int64_t nrows, int64_t ncols, int64_t row_bytes,
                                     const void * x);

// dst = x * y for ncols Q8_1 columns vy (stride_col_y blocks apart, as ggml_cuda_mul_mat_vec_q quantizes them) into
// ncols FP32 columns of dst (stride_col_dst floats apart). Returns false (nothing launched) where it doesn't take the
// product.
bool ggml_cuda_mmvq_pascal_dmt(ggml_type type, const void * vx, const void * vy, float * dst, int64_t K, int64_t nrows,
                               int64_t ncols, int64_t row_bytes, int64_t stride_col_y, int64_t stride_col_dst,
                               cudaStream_t stream);
