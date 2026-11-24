#pragma once

// The Pascal matrix-vector kernels' dispatch tables, the parts that need no GPU: mmvq-pascal-kcol.cu and mmvq-pascal.cu
// ask them for each product, and tests/test-cuda-mmvq-pascal-pick.cpp checks them on a CPU.
//
// 1. Q824: the products of 2 to 8 tokens (columns) that the k-column kernels (mmvq-pascal-kcol.cu) leave to the other
//    kernels in their mode 1 (GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE, on by default).
// 2. C899 (Q1625): GGML_CUDA_MMVQ_PASCAL_MT, which plain products of 2 to 4 tokens of the read-once types run
//    mul_mat_vec_q_pascal_mt (mmvq-pascal.cu), the kernel that decodes each weight once for all tokens with each token's
//    sums those of the one-token kernel (Q211). The value:
//      unset, 0       off: mul_mat_vec_q_pascal_mc for what reaches mmvq-pascal.cu, the k-column kernels where table 1
//                     keeps the product (as before C899)
//      2              the measured rule below: only the plain Q5_0 and Q8_0 products that a P106 ran faster on
//                     mul_mat_vec_q_pascal_mt go to it, and the k-column kernels leave those to it; every other product
//                     runs as with the switch unset
//      other numbers  (1) every product of 2 to 4 tokens of these types that reaches mmvq-pascal.cu, experts included (as
//                     before C899; the k-column kernels decide as with the switch unset)
//    Read as atoi reads it, so it is on or off for every value exactly as before C899; only 2 changes from "every product"
//    to the rule. A build before C899 reads 2 as 1.
//
// The rule (table 2) is R197's per-product timings on a P106-090 (Ornith-1.5-122B Pascal N2P GGUF on nine cards, the v5
// preset against the same with _MT=1 and _KCOL_SKIP=q5_0,q8_0: llms-on-crypto-mining-rigs
// docs/data/2026-11-19_r197_ornith_passes/compare.tsv.gz, rows "label"; docs/data/2026-11-21_c899_kcol_ornith/ derives
// the table). A product goes to mul_mat_vec_q_pascal_mt where both of R197's loads timed it faster there than on the
// kernel it ran on with the switch unset, each by more than both arms' spread between their two loads:
//   - Q5_0, more than 4,096 values in (measured: 8,192 in, 3,072 rows; ssm_out, attn_output): 2, 3 and 4 tokens, against
//     the k-column kernels (ssm_out 9.27 -> 6.71 ms a pass at 2 tokens, 12.49 -> 11.97 at 4)
//   - Q5_0, 2,049 to 4,096 values in (measured: 3,072 in, against mul_mat_vec_q_pascal_mc, where table 1 sends them):
//     3 tokens at every row count (64 to 248,320 rows measured); 2 tokens only up to 4,096 rows (64 and 1,024 rows
//     faster, 8,192 to 248,320 slower: attn_qkv 9.41 -> 9.55); 4 tokens never (attn_qkv 14.16 -> 15.57)
//   - Q5_0, up to 2,048 values in (measured: 1,024 in, 3,072 rows; the shared expert's down): 3 tokens only (2 slower,
//     4 level)
//   - Q8_0, up to 4,096 values in (measured: 3,072 in, 512 rows; attn_k, attn_v): 2, 3 and 4 tokens, against the
//     k-column kernels
// The ranges between and around the measured shapes are est.: the 4,096 bounds follow table 1's, the 2,048 one and the
// 4,096-row one sit between measured shapes. No other type, and no expert product (MUL_MAT_ID), is in the rule: they run
// as with the switch unset. R197 timed Ornith's file only; another file's shapes are not measured.

#include "ggml.h"

#include <cstdint>
#include <cstdlib>

// Q824: the products of 2 to 8 columns that the other kernels (mul_mat_vec_q_pascal_mc, llama.cpp's mul_mat_vec_q) run
// faster on a P106 than the k-column kernels with Q823's launch shapes: per type and row width (kmin <= K <= kmax
// weights), bit k - 1 of mask set for k columns. From test-backend-ops perf inside ggml at Flash-Next's shapes, the
// k-column side scaled by the bench's gain from Q823's shapes (llms-on-crypto-mining-rigs
// docs/data/2026-11-04_kshape/ggml_est.tsv, from docs/data/2026-11-04_kcurve_p106/perf_dense.tsv, perf_cov.tsv and
// perf_iq.md). Rows wider than 4096 weights are the 6144-wide ssm_out and attn_output and the retype's 10240-wide
// hc_down, all 2560 or fewer rows; the narrower ones the 2560-wide attention, the output layer (Q5_K, Q6_K) and the
// shared expert's 640-wide down projection (Q2_0, Q4_0, Q5_0). A product whose launch shape is pick_shape's fallback (no
// tile within a third of the SM: the K-quants and IQ4_XS at K = 6144 from 4-6 columns on, Q6_K's 2560-wide rows) also
// goes to the other kernels: wherever the perf runs timed such a shape, they were faster (level for Q4_K at 8 columns).
struct ggml_cuda_kcol_rule {
    ggml_type type;
    int64_t   kmin;
    int64_t   kmax;
    uint32_t  mask;
};
#define GGML_CUDA_PICK_COLS(k) (1u << ((k) - 1))
#define GGML_CUDA_PICK_COLS_2_8 0xFEu

// Whether table 1 leaves a product of ncols (2 to 8) columns of type with rows of K weights to the other kernels.
static inline bool ggml_cuda_kcol_table_declines(const ggml_type type, const int64_t K, const int64_t ncols) {
    static const ggml_cuda_kcol_rule table[] = {
        { GGML_TYPE_IQ4_XS,  4097, INT64_MAX, GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(5) | GGML_CUDA_PICK_COLS(6) | GGML_CUDA_PICK_COLS(7) | GGML_CUDA_PICK_COLS(8) },
        { GGML_TYPE_Q4_K,    4097, INT64_MAX, GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) | GGML_CUDA_PICK_COLS(4) | GGML_CUDA_PICK_COLS(6) | GGML_CUDA_PICK_COLS(7) },
        { GGML_TYPE_Q8_0,    4097, INT64_MAX, GGML_CUDA_PICK_COLS_2_8 },
        { GGML_TYPE_IQ2_XXS, 4097, INT64_MAX, GGML_CUDA_PICK_COLS_2_8 },
        { GGML_TYPE_IQ2_XS,  4097, INT64_MAX, GGML_CUDA_PICK_COLS_2_8 },
        { GGML_TYPE_IQ2_S,   4097, INT64_MAX, GGML_CUDA_PICK_COLS_2_8 },
        { GGML_TYPE_IQ3_XXS, 4097, INT64_MAX, GGML_CUDA_PICK_COLS_2_8 },
        { GGML_TYPE_IQ3_S,   4097, INT64_MAX, GGML_CUDA_PICK_COLS_2_8 },
        { GGML_TYPE_Q5_K,    0,    4096,      GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) },
        { GGML_TYPE_Q6_K,    0,    4096,      GGML_CUDA_PICK_COLS_2_8 },
        { GGML_TYPE_Q4_0,    0,    4096,      GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) | GGML_CUDA_PICK_COLS(4) },
        { GGML_TYPE_Q5_0,    0,    4096,      GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) | GGML_CUDA_PICK_COLS(4) },
        { GGML_TYPE_Q2_0,    0,    4096,      GGML_CUDA_PICK_COLS(2) },
        { GGML_TYPE_IQ2_XXS, 0,    4096,      GGML_CUDA_PICK_COLS(2) },
        { GGML_TYPE_IQ3_S,   0,    4096,      GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) },
        { GGML_TYPE_IQ2_S,   0,    4096,      GGML_CUDA_PICK_COLS(3) },
        { GGML_TYPE_IQ3_XXS, 0,    4096,      GGML_CUDA_PICK_COLS(3) },
    };
    for (const ggml_cuda_kcol_rule & r : table) {
        if (r.type == type && K >= r.kmin && K <= r.kmax) {
            return (r.mask >> (ncols - 1)) & 1u;
        }
    }
    return false;
}

#define GGML_CUDA_MMVQ_PASCAL_MT_OFF  0
#define GGML_CUDA_MMVQ_PASCAL_MT_ALL  1
#define GGML_CUDA_MMVQ_PASCAL_MT_RULE 2

// The mode a value of GGML_CUDA_MMVQ_PASCAL_MT sets (nullptr: unset)
static inline int ggml_cuda_mmvq_pascal_mt_parse(const char * value) {
    if (value == nullptr) {
        return GGML_CUDA_MMVQ_PASCAL_MT_OFF;
    }
    const int v = atoi(value);
    return v == 0 ? GGML_CUDA_MMVQ_PASCAL_MT_OFF : v == 2 ? GGML_CUDA_MMVQ_PASCAL_MT_RULE : GGML_CUDA_MMVQ_PASCAL_MT_ALL;
}

// C899: the plain products that mode 2 sends to mul_mat_vec_q_pascal_mt, per type, row width (kmin <= K <= kmax values
// in) and row count (rmin <= nrows <= rmax), bit k - 1 of mask set for k tokens (2 to 4). The first row that matches the
// type, width and row count decides.
struct ggml_cuda_mmvq_pascal_mt_rule {
    ggml_type type;
    int64_t   kmin;
    int64_t   kmax;
    int64_t   rmin;
    int64_t   rmax;
    uint32_t  mask;
};

// Whether table 2 sends a plain product of type, rows of K values, nrows rows and ncols tokens to
// mul_mat_vec_q_pascal_mt
static inline bool ggml_cuda_mmvq_pascal_mt_rule_takes(const ggml_type type, const int64_t K, const int64_t nrows,
                                                       const int64_t ncols) {
    if (ncols < 2 || ncols > 4) {
        return false;
    }
    static const ggml_cuda_mmvq_pascal_mt_rule rules[] = {
        { GGML_TYPE_Q5_0, 4097, INT64_MAX, 1,    INT64_MAX, GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) | GGML_CUDA_PICK_COLS(4) },
        { GGML_TYPE_Q5_0, 2049, 4096,      1,    4096,      GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) },
        { GGML_TYPE_Q5_0, 2049, 4096,      4097, INT64_MAX, GGML_CUDA_PICK_COLS(3) },
        { GGML_TYPE_Q5_0, 1,    2048,      1,    INT64_MAX, GGML_CUDA_PICK_COLS(3) },
        { GGML_TYPE_Q8_0, 1,    4096,      1,    INT64_MAX, GGML_CUDA_PICK_COLS(2) | GGML_CUDA_PICK_COLS(3) | GGML_CUDA_PICK_COLS(4) },
    };
    for (const ggml_cuda_mmvq_pascal_mt_rule & r : rules) {
        if (r.type == type && K >= r.kmin && K <= r.kmax && nrows >= r.rmin && nrows <= r.rmax) {
            return (r.mask >> (ncols - 1)) & 1u;
        }
    }
    return false;
}

// The types mul_mat_vec_q_pascal_mt runs (mmvqp_mt_type in mmvq-pascal.cu)
static inline bool ggml_cuda_mmvq_pascal_mt_type(const ggml_type type) {
    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_N2P || type == GGML_TYPE_N3P ||
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_0;
}

// mmvq-pascal.cu: whether a product that reached ggml_cuda_mmvq_pascal runs mul_mat_vec_q_pascal_mt (where its staged
// activations fit; mul_mat_vec_q_pascal_mc otherwise). has_ids: an expert product (MUL_MAT_ID); max_entries: the most
// (token, slot) entries of an expert product the kernel's warps scan (MMVQP_MT_MAX_ENTRIES).
static inline bool ggml_cuda_mmvq_pascal_mt_on(const int mode, const ggml_type type, const bool has_ids, const int64_t K,
                                               const int64_t nrows, const int64_t ncols, const int64_t nchannels_dst,
                                               const int64_t nsamples_dst, const int64_t max_entries) {
    if (ncols <= 1 || !ggml_cuda_mmvq_pascal_mt_type(type)) {
        return false;
    }
    switch (mode) {
        case GGML_CUDA_MMVQ_PASCAL_MT_ALL:
            return !has_ids || nchannels_dst * ncols <= max_entries;
        case GGML_CUDA_MMVQ_PASCAL_MT_RULE:
            return !has_ids && nchannels_dst == 1 && nsamples_dst == 1 && ggml_cuda_mmvq_pascal_mt_rule_takes(type, K, nrows, ncols);
        default:
            return false;
    }
}

// mmvq-pascal-kcol.cu: whether the k-column kernels leave a plain product to mul_mat_vec_q_pascal_mt (mode 2 only; the
// caller asks only in its mode 1, and not for a product GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT keeps)
static inline bool ggml_cuda_mmvq_pascal_mt_kcol_declines(const int mode, const ggml_type type, const int64_t K,
                                                          const int64_t nrows, const int64_t ncols) {
    return mode == GGML_CUDA_MMVQ_PASCAL_MT_RULE && ggml_cuda_mmvq_pascal_mt_rule_takes(type, K, nrows, ncols);
}

#undef GGML_CUDA_PICK_COLS
#undef GGML_CUDA_PICK_COLS_2_8
