#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// GGML_CUDA_KEEPALIVE_EAGER=3 (C866): what the cache's length picks for this FLASH_ATTN_EXT call (fattn.cu), as one
// number; launches nothing
uint64_t ggml_cuda_flash_attn_ext_length_choice(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
