#pragma once

#include "common.cuh"

// Sparse decode attention for GPUs without Turing tensor cores (the P106-090 and other Pascal cards): FLASH_ATTN_EXT
// with the n_kv_max hint, as qwen4exp's 12 QSA layers call it, reads only the cells the mask keeps (the indexer's
// ~2,051) instead of the whole cache. Upstream's gather of those cells exists only in the Turing MMA kernel, so below
// Turing every layer reads every cell: 64 MiB per layer at 32k (F16), against 4 MiB here. The kernel and its design are
// in fattn-sparse-pascal-impl.cuh; this file holds the conditions, the scratch and the switch.
//
// It takes: an F32 query of head size 128 or 256 (K and V the same), 4, 8, 12 or 16 query heads per KV head, F16 or
// Q8_0 K and V (both the same type), an F16 mask, n_kv_max > 0, at most GGML_CUDA_FATTN_SPARSE_PASCAL_NQ query tokens
// (default 8: decode and verify batches; prompt micro-batches keep the dense kernels unless _PROMPT), at most 1,024
// query tokens x KV heads, optional sinks, no ALiBi or softcap, no kv_rows. Anything else stays with the stock kernels,
// as does any GPU with Turing tensor cores in mode 1. The rule is fasp_rule_takes (fattn-sparse-pascal-impl.cuh).
//
// GGML_CUDA_FATTN_SPARSE_PASCAL: 0 off (default), 1 on GPUs without Turing tensor cores (compute capability below 7.5, or
// a build without Turing code) once the cache holds at least _MIN_RATIO x n_kv_max cells per query token (each token
// reads its own list, the dense kernels read the cache once for all of them), 2 any NVIDIA GPU at any cache size (for
// testing).
// GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO: cells per n_kv_max and query token from which mode 1 takes a call (default
// 1.5).
// GGML_CUDA_FATTN_SPARSE_PASCAL_NQ: most query tokens per sequence of a decode or verify call (default 8).
// GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT: 0 off (default), 1 calls of more than _NQ and at most _PROMPT_NQ query tokens
// per sequence (prompt micro-batches) go to the kernel too: in mode 1 once the cache holds _PROMPT_MIN_RATIO x n_kv_max
// cells, whatever the call's size (the dense kernels compute every cell for each query token of a prompt), in mode 2
// at any cache size.
// GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT_MIN_RATIO: cells per n_kv_max from which mode 1 takes a prompt call (default
// 1.0: below n_kv_max cells the mask keeps them all and the kernel reads no fewer than the dense kernels).
// GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT_NQ: most query tokens per sequence of a prompt call (default 128).
// GGML_CUDA_FATTN_SPARSE_PASCAL_UNION: 0 off (default), 1 a call of two or more query tokens goes to the union kernel: a
// block takes up to 12 warps' worth of consecutive query tokens (Flash-Next's 12 heads per KV head: 4 tokens) and reads
// one list, the union of theirs, so each kept cell is read once per block instead of once per query token. Each token's
// output equals the per-token kernel's to rounding. Mode 1 then takes a decode or verify call from _UNION_MIN_RATIO x
// n_kv_max cells per union list (a verify of up to 4 tokens: per call) instead of per query token; prompt calls keep
// their rule.
// GGML_CUDA_FATTN_SPARSE_PASCAL_UNION_MIN_RATIO: cells per n_kv_max and union list from which mode 1 takes a decode or
// verify call with _UNION (default 1.5).
// GGML_CUDA_FATTN_SPARSE_PASCAL_SPLITS: splits per list (default: about 4 blocks per SM, at least 64 cells a split).
// GGML_CUDA_FATTN_SPARSE_PASCAL_NSM: SM count the splits assume (default: the device's; 5 emulates a P106-090).
// GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT (testing the tests): 1 output x 1.01, 2 mask values ignored, 3 every split's last
// list entry dropped, 4 (union kernel) each query token scored with the next token's mask row.

// Runs the node and returns true if the path takes it; returns false (and does nothing) otherwise.
bool ggml_cuda_flash_attn_ext_sparse_pascal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
