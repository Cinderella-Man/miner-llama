#pragma once

#include "common.cuh"

// A gate product, an up product and their SWIGLU of 2 to 8 tokens as one launch on compute capability 6.1 (Pascal):
// Qwen3.8-Flash-Next's shared expert in a speculative verify, where 41 of 47 layers otherwise run the gate and the up as
// two small launches of 2,560 -> 640 and the SWIGLU as a third (C757, Q1333). Gate and up may be of different types (33
// of those 41 layers mix them). Each block stages every token's activations once; its warps work in pairs, one on rows
// of the gate and one on the same rows of the up, each in the summation order of its type's one-token kernel, and the
// pair writes silu(gate) * up of those rows. Each gate and up sum is, bit for bit, what the separate product gives
// (the one-token sums), and the SWIGLU is the separate op's expression on the same values. The design is at the top of
// mmvq-pascal-glu.cu.
//
// Types and the separate products it reproduces (a pair is taken only where both of its products are taken):
//   - Q2_0, Q3_K: the k-column kernel's order (their one-token kernel's), 2 to 8 tokens; taken where the k-column
//     kernels run (GGML_CUDA_MMVQ_PASCAL_KCOL not 0, the type not in _KCOL_SKIP or GGML_CUDA_MMVQ_PASCAL_SKIP);
//   - IQ4_XS, Q4_K: the one-token Pascal kernel's order, 2 to 4 tokens, only where GGML_CUDA_MMVQ_PASCAL_DMT's kernel
//     takes the separate product (C722); with the switch at 3 or 4 also 5 to 8 tokens, only where the same kernel takes
//     the separate product under GGML_CUDA_MMVQ_PASCAL_RO8 (C729, C793);
//   - Q5_K: the same, only where GGML_CUDA_MMVQ_PASCAL_Q5K_MT's kernel takes it (C690; 5 to 8 tokens under _RO8 at 3
//     or 4).
// Other types (Q6_K among them) and GLU ops other than plain SWIGLU are left to the separate launches.
//
// GGML_CUDA_MMVQ_PASCAL_GLU_MT: 0 (default) off, every path as before; 1 on; 2 on without the register prefetch of each
// warp's next weight segment (for measuring it); 3 and 4 as 1 and 2, and pairs with an IQ4_XS, Q4_K or Q5_K product of
// 5 to 8 tokens taken too where _RO8 runs that product read once (C793, Q1388: a kernel of its own, so the 2-8-token
// kernels of Q2_0 and Q3_K pairs keep their code). GGML_CUDA_MMVQ_PASCAL_GLU_MT_NP=n: warp pairs per block, 1 to 4
// (default: the most that fit 48 KB of shared memory). GGML_CUDA_MMVQ_PASCAL_GLU_MT_FAULT=1 (checks only): each token
// takes the next token's activations, so the results must differ. The kernels follow GGML_CUDA_MMVQ_PASCAL (0 off, 2 any
// NVIDIA GPU), as the other Pascal kernels do.

// Whether the switch is on (cheap; read once).
bool ggml_cuda_mmvq_pascal_glu_mt_on();

// glu = SWIGLU(gate, up) for the MUL_MAT nodes gate and up (same activations src[1], 2 to 8 columns) as one launch, if
// these kernels take it; returns false (nothing launched) otherwise. The caller has checked that the three nodes may be
// fused (the gate's and up's results are used by glu only, the memory ranges).
bool ggml_cuda_mmvq_pascal_glu_mt(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
                                  ggml_tensor * glu);
