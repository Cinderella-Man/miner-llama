#pragma once

// GGML_CUDA_MMVF_F32_MT_ROWS (C807, Q1413; C822, Q1436): the row limit of GGML_CUDA_MMVF_F32_MT, which sends an F32
// weight matrix times 4 to 8 tokens to mul_mat_vec_f (mmvf.cu). mul_mat_vec_f runs a block per row and each block reads
// every token's activations again, so it beats cuBLAS only up to some row count. On a P106-090 that count follows the
// token count, because cuBLAS is about three times faster at 4 tokens than at 5 to 8: R132 timed 12.11 / 12.21 us at
// 32 rows of 2,560 and 4 tokens against 34.53 / 34.49 at 5, and mul_mat_vec_f gained at 33 rows at 4-8 tokens and at
// 64 rows at 5-8, but lost at 64 rows at 4 (13.05 / 13.17 -> 13.48 / 13.58 us). The value:
//   unset    32 rows at every token count (C807's default)
//   N        N rows at every token count; 0 or below: any row count (C754's rule, for timing)
//   auto     a limit per token count: 48 rows at 4 tokens, 64 at 5 to 8 (est. from R132's perf on a P106-090 at rows
//            of 2,560; a rig run places it)
//   a,b,...  one limit per token count from 4 tokens on (a at 4, b at 5, ...); the last entry holds for the token
//            counts after it, and entries past 8 tokens are ignored
// Each number, alone or in a list, is read as atoi reads it (an empty or non-numeric entry is 0: any row count), so a
// single number means exactly what it meant in C807. A build before C822 reads the whole value with atoi: "auto" is
// 0 there (any row count, the router of Qwen3.8-Flash-Next on mul_mat_vec_f too) and a list its first entry.
//
// The parts that need no GPU live here: ggml_cuda_mul_mat_vec_f_f32_mt (mmvf.cu) reads the value once and asks for each
// product, and tests/test-cuda-mmvf-f32-rows.cpp checks both on a CPU.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#define GGML_CUDA_MMVF_F32_ROWS_MIN_TOKENS 4
#define GGML_CUDA_MMVF_F32_ROWS_MAX_TOKENS 8 // MMVF_MAX_BATCH_SIZE (mmvf.cuh)
#define GGML_CUDA_MMVF_F32_ROWS_DEFAULT    32
#define GGML_CUDA_MMVF_F32_ROWS_AUTO       "48,64"

struct ggml_cuda_mmvf_f32_rows {
    // the limit at 4, 5, ..., 8 tokens; 0 or below: any row count
    int max_rows[GGML_CUDA_MMVF_F32_ROWS_MAX_TOKENS - GGML_CUDA_MMVF_F32_ROWS_MIN_TOKENS + 1];
};

// The limits a value of GGML_CUDA_MMVF_F32_MT_ROWS sets (nullptr: unset)
static inline ggml_cuda_mmvf_f32_rows ggml_cuda_mmvf_f32_rows_parse(const char * value) {
    ggml_cuda_mmvf_f32_rows r;
    const int n = (int) (sizeof(r.max_rows) / sizeof(r.max_rows[0]));
    if (value == nullptr) {
        for (int i = 0; i < n; i++) {
            r.max_rows[i] = GGML_CUDA_MMVF_F32_ROWS_DEFAULT;
        }
        return r;
    }
    if (strcmp(value, "auto") == 0) {
        value = GGML_CUDA_MMVF_F32_ROWS_AUTO;
    }
    const char * p    = value;
    int          last = 0;
    for (int i = 0; i < n; i++) {
        if (p != nullptr) {
            last = atoi(p);
            p    = strchr(p, ',');
            if (p != nullptr) {
                p++;
            }
        }
        r.max_rows[i] = last;
    }
    return r;
}

// Whether the limit lets the switch take a matrix of ne01 rows times ne11 tokens (4 to 8; any other token count: no)
static inline bool ggml_cuda_mmvf_f32_rows_take(const ggml_cuda_mmvf_f32_rows & r, const int64_t ne01, const int64_t ne11) {
    if (ne11 < GGML_CUDA_MMVF_F32_ROWS_MIN_TOKENS || ne11 > GGML_CUDA_MMVF_F32_ROWS_MAX_TOKENS) {
        return false;
    }
    const int max_rows = r.max_rows[ne11 - GGML_CUDA_MMVF_F32_ROWS_MIN_TOKENS];
    return max_rows <= 0 || ne01 <= max_rows;
}
