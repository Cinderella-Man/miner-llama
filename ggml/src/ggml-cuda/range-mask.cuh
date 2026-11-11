#include "common.cuh"

#define CUDA_RANGE_MASK_BLOCK_SIZE 256

void ggml_cuda_op_range_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
