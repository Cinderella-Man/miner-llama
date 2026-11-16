#pragma once

#include "common.cuh"

// Plain Q5_K products of 2 to 8 tokens (MUL_MAT, no fusion) on compute capability 6.1 that read and decode each weight
// once for all tokens and give every token, bit for bit, the sums of the one-token Pascal kernel (mul_mat_vec_q_pascal,
// mmvq-pascal.cu), for the output layer of a speculative-decoding verify: Qwen3.8-Flash-Next's 2560 x 248,320 head on
// the last card, whose four-token product took 2.82 times its one-token time on the k-column kernel (Q1222, C690). The
// design is at the top of mmvq-pascal-q5k.cu. Optionally one-token products too (C767), where at the head's shape two
// tokens on this kernel took 0.85 of one token on the one-token kernel (R75). 5 to 8 tokens: GGML_CUDA_MMVQ_PASCAL_RO8.
//
// GGML_CUDA_MMVQ_PASCAL_Q5K_MT: 0 (default) off, the k-column and the earlier multi-token kernels as before; 1 on,
// at one row per lane group without the register prefetch (since C766: on a P106 the fastest at 2 to 4 tokens, R75);
// 2 the same, with the prefetch kept off even under _PF=1 (before C766, 1 meant two rows with the prefetch and 2 the
// same without it). GGML_CUDA_MMVQ_PASCAL_Q5K_MT_R=2: two rows per lane group, which share the activation loads
// (default 1, as the one-token kernel). GGML_CUDA_MMVQ_PASCAL_Q5K_MT_PF=1 (with 1): the register prefetch of each
// warp's next weight segments. GGML_CUDA_MMVQ_PASCAL_Q5K_MT_NW=n: warps per block (1 to 6 at R = 1, 1 to 3 at R = 2,
// default the most; fewer where the block's shared memory would pass 48 KB). The kernels follow GGML_CUDA_MMVQ_PASCAL
// (0 off, 2 any NVIDIA GPU) and GGML_CUDA_MMVQ_PASCAL_SKIP=q5_K, as the one-token kernel does.
//
// GGML_CUDA_MMVQ_PASCAL_Q5K_1R (C767, Q1348): 0 (default) off, one-token Q5_K products go to the one-token kernel as
// before; 1 the same kernel takes the plain one-token products too (the output layer of a plain decode step and of
// the draft's full-head passes), each result the one-token kernel's bit for bit, with the settings above (_MT_R,
// _MT_PF, _MT_NW) and whatever _MT is. GGML_CUDA_MMVQ_PASCAL_Q5K_1R_ROWS=n: only products of at least n rows
// (default 0, every one; 100000 leaves the output layer alone). GGML_CUDA_MMVQ_PASCAL_Q5K_1R_FAULT=1: a check that must
// fail, each lane group leaving its row's last iteration out of the one-token products this switch takes.

// Whether ggml_cuda_mmvq_pascal_q5k_mt takes a product of ncols columns of a Q5_K matrix with K weights per row,
// nrows rows row_bytes apart from x, on the current device (the k-column kernels leave such products to it).
bool ggml_cuda_mmvq_pascal_q5k_mt_takes(int64_t K, int64_t nrows, int64_t ncols, int64_t row_bytes, const void * x);

// dst = x * y for ncols Q8_1 columns vy (stride_col_y blocks apart, as ggml_cuda_mul_mat_vec_q quantizes them) into
// ncols FP32 columns of dst (stride_col_dst floats apart). Returns false (nothing launched) where it doesn't take the
// product.
bool ggml_cuda_mmvq_pascal_q5k_mt(const void * vx, const void * vy, float * dst, int64_t K, int64_t nrows, int64_t ncols,
                                  int64_t row_bytes, int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream);
