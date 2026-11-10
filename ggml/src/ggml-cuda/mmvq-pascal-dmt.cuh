#pragma once

#include "common.cuh"

// Plain IQ4_XS and Q4_K products of 2 to 4 tokens (MUL_MAT, no fusion) on compute capability 6.1 that read and decode
// each weight once for all tokens and give every token, bit for bit, the sums of the one-token Pascal kernel
// (mul_mat_vec_q_pascal, mmvq-pascal.cu): Qwen3.8-Flash-Next's attention and recurrent projections in a speculative
// verify, which the k-column kernel runs at one row per lane group with sums other than the one-token kernel's (C722,
// Q1277; C690's Q5_K kernel, mmvq-pascal-q5k.cu, carried to the other types). The design is at the top of
// mmvq-pascal-dmt.cu.
//
// GGML_CUDA_MMVQ_PASCAL_DMT: 0 (default) off, the k-column and the earlier kernels as before; 1 on; 2 on without the
// register prefetch of each warp's next weight segment (for measuring it). GGML_CUDA_MMVQ_PASCAL_DMT_R: rows per lane
// group, 0 (default) chosen per product, 1 or 2. GGML_CUDA_MMVQ_PASCAL_DMT_NW=n: warps per block at most (1 to 6 at
// R = 1, 1 to 3 at R = 2; fewer where the block's shared memory would pass 48 KB). GGML_CUDA_MMVQ_PASCAL_DMT_SKIP=q4_K:
// types left to the k-column kernel. The kernels follow GGML_CUDA_MMVQ_PASCAL (0 off, 2 any NVIDIA GPU) and
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
