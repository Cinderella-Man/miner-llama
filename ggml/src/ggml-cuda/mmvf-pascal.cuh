#pragma once

#include "common.cuh"

// BF16 matrix-vector products of 1 to 8 tokens with FP32 activations, tuned for compute capability 6.1 (Pascal, which
// has no BF16 arithmetic). mul_mat_vec_f gives every row a block whose threads read 4 bytes each, which on a 5-SM
// P106-090 leaves short rows latency-bound: the 320-wide hyper-connection projections of Qwen3.8-Flash-Next
// (10,240 rows) ran at ~20 GB/s. These kernels give each row group a lane group sized to the row width (8, 32 or 128
// lanes), let 1, 2 or 4 rows share every load of the activations and read 8 weights per lane with one coalesced
// 128-bit load. Each weight is converted to FP32 exactly (a BF16 value is the upper half of an FP32 one) and
// accumulated in FP32 with one FMA, so the products differ from an FP32 reference only by the summation order.
//
// GGML_CUDA_MMVF_PASCAL: 0 off, 1 compute capability 6.1 only (default), 2 any NVIDIA GPU (for testing).
// GGML_CUDA_MMVF_PASCAL_NSM=n: choose the launch shape as for a card of n SMs (e.g. 5, to run the P106-090's shapes on
// another GPU). GGML_CUDA_MMVF_PASCAL_SHAPE=G,R: force G lanes per row group (8, 32, 128) and R rows per group
// (1, 2, 4, 8) for testing and tuning. GGML_CUDA_MMVF_PASCAL_EXACT=1 (or 2, another kernel): products of 2 to 4 tokens
// add each row in the order of one token (mmvf-pascal.cu).

// Whether the kernels run on a device of compute capability cc under GGML_CUDA_MMVF_PASCAL.
bool ggml_cuda_mmvf_pascal_active(int cc);

// Whether ggml_cuda_mmvf_pascal takes the product dst = src0 * src1 (without fusion) on the current device.
bool ggml_cuda_mmvf_pascal_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Runs the product if it is supported, with fusion's bias added (one token) if it has one; fusion with a gate is left to
// mul_mat_vec_f. Returns false (nothing launched) otherwise.
bool ggml_cuda_mmvf_pascal(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_cuda_mm_fusion_args_host * fusion,
                           ggml_tensor * dst, cudaStream_t stream);
