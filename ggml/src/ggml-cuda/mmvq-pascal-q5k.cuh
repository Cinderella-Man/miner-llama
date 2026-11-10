#pragma once

#include "common.cuh"

// Plain Q5_K products of 2 to 4 tokens (MUL_MAT, no fusion) on compute capability 6.1 that read and decode each weight
// once for all tokens and give every token, bit for bit, the sums of the one-token Pascal kernel (mul_mat_vec_q_pascal,
// mmvq-pascal.cu), for the output layer of a speculative-decoding verify: Qwen3.8-Flash-Next's 2560 x 248,320 head on
// the last card, whose four-token product took 2.82 times its one-token time on the k-column kernel (Q1222, C690). The
// design is at the top of mmvq-pascal-q5k.cu.
//
// GGML_CUDA_MMVQ_PASCAL_Q5K_MT: 0 (default) off, the k-column and the earlier multi-token kernels as before; 1 on; 2 on
// without the register prefetch of each warp's next weight segment (for measuring it). GGML_CUDA_MMVQ_PASCAL_Q5K_MT_NW=n:
// warps per block (1 to 6, default 6; fewer where the block's shared memory would pass 48 KB). The kernels follow
// GGML_CUDA_MMVQ_PASCAL (0 off, 2 any NVIDIA GPU) and GGML_CUDA_MMVQ_PASCAL_SKIP=q5_K, as the one-token kernel does.

// Whether ggml_cuda_mmvq_pascal_q5k_mt takes a product of ncols columns of a Q5_K matrix with K weights per row,
// nrows rows row_bytes apart from x, on the current device (the k-column kernels leave such products to it).
bool ggml_cuda_mmvq_pascal_q5k_mt_takes(int64_t K, int64_t nrows, int64_t ncols, int64_t row_bytes, const void * x);

// dst = x * y for ncols Q8_1 columns vy (stride_col_y blocks apart, as ggml_cuda_mul_mat_vec_q quantizes them) into
// ncols FP32 columns of dst (stride_col_dst floats apart). Returns false (nothing launched) where it doesn't take the
// product.
bool ggml_cuda_mmvq_pascal_q5k_mt(const void * vx, const void * vy, float * dst, int64_t K, int64_t nrows, int64_t ncols,
                                  int64_t row_bytes, int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream);
