#pragma once

#include "common.cuh"

// Plain Q3_K products of 2 to 8 tokens (MUL_MAT, no fusion) on compute capability 6.1 that read and decode each weight
// once for all tokens, stage the tokens' activations once per block, and give every token, bit for bit, the sums of the
// one-token kernel, which for Q3_K is the k-column kernel itself (mmvq-pascal-kcol.cu at one column): Qwen3.8-Flash-Next's
// Q3_K attention and recurrent projections (attn_qkv, attn_gate, attn_q, ssm_out) and the shared expert's Q3_K products
// in a speculative verify (C726, Q1285; C722's method, mmvq-pascal-dmt.cu, in the k-column kernel's order). The design is
// at the top of mmvq-pascal-q3kmt.cu.
//
// GGML_CUDA_MMVQ_PASCAL_Q3K_MT: 0 (default) off, the k-column kernel as before; 1 on; 2 on without the register prefetch
// of each warp's next weight segments. GGML_CUDA_MMVQ_PASCAL_Q3K_MT_R: rows per lane group, 1 (default) or 2.
// GGML_CUDA_MMVQ_PASCAL_Q3K_MT_NW=n: warps per block at most (1 to 6 at R = 1, 1 to 3 at R = 2; fewer where the block's
// shared memory would pass 48 KB). GGML_CUDA_MMVQ_PASCAL_Q3K_MT_PAD=n: n 16-byte units of padding after each row's slot
// in the weight staging (default: the fewest that spread the lanes' reads over the banks; 0 for none).
// GGML_CUDA_MMVQ_PASCAL_Q3K_MT_FAULT=1: a fault for the checks (each token takes the next token's activations), which
// test-backend-ops and multitoken_bitid must fail. The kernel follows GGML_CUDA_MMVQ_PASCAL (0 off, 2 any NVIDIA GPU),
// GGML_CUDA_MMVQ_PASCAL_KCOL and GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP: it takes nothing where the k-column kernel does not
// take Q3_K, whose one-token sums it keeps.

// Whether ggml_cuda_mmvq_pascal_q3kmt takes a Q3_K product of ncols columns of a matrix with K weights per row, nrows
// rows row_bytes apart from x, on the current device (the k-column kernels leave such products to it).
bool ggml_cuda_mmvq_pascal_q3kmt_takes(int64_t K, int64_t nrows, int64_t ncols, int64_t row_bytes, const void * x);

// dst = x * y for ncols Q8_1 columns vy (stride_col_y blocks apart, as ggml_cuda_mul_mat_vec_q quantizes them) of a
// Q3_K matrix x into ncols FP32 columns of dst (stride_col_dst floats apart). Returns false (nothing launched) where it
// doesn't take the product.
bool ggml_cuda_mmvq_pascal_q3kmt(const void * vx, const void * vy, float * dst, int64_t K, int64_t nrows, int64_t ncols,
                                 int64_t row_bytes, int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream);
