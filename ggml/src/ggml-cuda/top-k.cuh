#pragma once

#include "common.cuh"

// the kernels ggml_cuda_op_top_k runs for a call (top-k.cu, top_k_pick)
enum ggml_cuda_top_k_path {
    GGML_CUDA_TOP_K_PATH_BITONIC     = 1, // a bitonic argsort in shared memory
    GGML_CUDA_TOP_K_PATH_RADIX       = 2, // the radix select
    GGML_CUDA_TOP_K_PATH_CUB_TOP_K   = 3, // CUB's DeviceTopK, row by row (CCCL 3.4.3 and later)
    GGML_CUDA_TOP_K_PATH_CUB_ARGSORT = 4, // CUB's argsort
};

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_CUDA_KEEPALIVE_EAGER=3 (C866): the path ggml_cuda_op_top_k takes for this node on the device; runs nothing
ggml_cuda_top_k_path ggml_cuda_top_k_path_of(int device, const ggml_tensor * dst);
