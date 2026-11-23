#pragma once

#include "common.cuh"

// Prompt-size attention for compute capability 6.1 (Pascal with dp4a, the 5-SM P106-090): FLASH_ATTN_EXT with an F32
// query of head size 128 or 256, 8 query heads per KV head (or a multiple of 8), F16 or Q8_0 K and V, an F16 mask and
// optional sinks, at 9 or more query tokens (3 or more with Q8_0), as Flint (head 128) and the Qwen3.6 class (head 256)
// read prompts. It replaces the stock TILE kernel for these calls; anything else stays with the stock kernels.
//
// Without fast FP16 the stock TILE kernel stores K and V as FP32 in shared memory and multiplies them with 2-key x
// 4-column register tiles, so every FMA needs about half a float from shared memory, which on sm_61 delivers 32 floats
// per clock per SM against 128 FMAs: it runs at roughly a third of the FP32 peak. This kernel keeps the same FP32
// arithmetic and raises the reuse per loaded value:
//   - a block of 4 warps works on 4 tokens x 8 heads (head 128) or 2 tokens x 8 heads (head 256) of one KV head;
//   - each lane computes the scores of 4 keys x 4 columns (float4 loads along the head dimension) and owns the output of
//     the same 4 columns x 8 dimensions, so the softmax rescale stays in registers and the probabilities of a warp go
//     through its own slice of shared memory without a block barrier;
//   - K and V are staged in FP32 chunks, loaded into registers one chunk ahead and converted from F16 or dequantized
//     from Q8_0 on the way in, so a Q8_0 cache needs no F16 copy;
//   - a key tile whose mask is -inf for every token of the block is skipped (the sliding-window ring, causal padding).
// Scores sum the head dimension in order and P*V sums the keys in order, as the stock FP32 kernel does; the softmax
// tiles are 64 keys at head 128 (the stock tile) and 128 keys at head 256 (stock: 32).
//
// GGML_CUDA_FATTN_TILE_PASCAL: 0 off, 1 compute capability 6.1 only (default), 2 any NVIDIA GPU (for testing).
// GGML_CUDA_FATTN_TILE_PASCAL_MIN: fewest query tokens it takes (default 9 with F16 K/V, 3 with Q8_0 K/V).
// GGML_CUDA_FATTN_TILE_PASCAL_FAULT (tests of the tests): 1 output x 1.05, 2 sinks dropped, 3 each block's last key
// tile dropped.
// GGML_CUDA_FATTN_DEC_PASCAL=N (off by default): Q8_0 K/V at head 256 with up to N query tokens (1-4: decode, the pipe,
// the MTP verify) go to a decode kernel without block barriers in its key loop: warps on their own 32-key slices, Q as
// int8, dp4a scores, V in registers. It is only reached from GGML_CUDA_FATTN_TILE_PASCAL_MIN tokens, so set _MIN=1.
// GGML_CUDA_FATTN_TILE_PASCAL_KDP=1 (off by default): Q8_0 K at head 256 is staged as int8 and scored with dp4a against
// a Q quantized per 32 dimensions, instead of dequantized to FP32.
// GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 (off by default): the tile kernel issues Q8_0 K/V loads raw and decodes them when
// it stores them to shared memory, a chunk later, instead of right after the load. Same arithmetic, same results.

// Whether the kernel takes this FLASH_ATTN_EXT node on this device (the same answer at allocation and compute time).
bool ggml_cuda_fattn_tile_pascal_supported(int device, const ggml_tensor * dst);

// Runs the node if ggml_cuda_fattn_tile_pascal_supported; returns false (and does nothing) otherwise.
bool ggml_cuda_flash_attn_ext_tile_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
