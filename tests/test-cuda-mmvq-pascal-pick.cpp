// The Pascal matrix-vector kernels' dispatch tables, the parts that run without a GPU (ggml/src/ggml-cuda/
// mmvq-pascal-pick.h; Q824, C899):
//   - GGML_CUDA_MMVQ_PASCAL_MT is on or off for every value exactly as before C899 (atoi of the value not 0), and only
//     "2" (as atoi reads it) selects the measured rule.
//   - Unset (0) and 1, both decisions that read the switch are those of the code before C899 (pascal-p106-v5 at
//     3dfad876f) over every type, row width, row count, token count, expert or plain product, channel and sample count
//     below: mmvq-pascal.cu's choice of mul_mat_vec_q_pascal_mt (its p.mt expression, copied here) and the k-column
//     kernels' new decline (never).
//   - The k-column kernels' table (Q824) moved into the header unchanged: it decides every type, width and token count
//     as 3dfad876f's copy below.
//   - Mode 2 changes nothing outside plain Q5_0 and Q8_0 products of 2 to 4 tokens.
//   - Mode 2 on Ornith-1.5-122B's 13 dense Q5_0 and Q8_0 products (R197, llms-on-crypto-mining-rigs
//     docs/data/2026-11-19_r197_ornith_passes): in a model of the dispatch order (mmvq.cu: the k-column kernels, then
//     mmvq-pascal.cu), each product at each token count runs the read-once kernel where R197 timed it faster there, the
//     kernel it runs with the switch unset where R197 timed that faster, and either where the two were level. The
//     kernels with the switch unset are those R197 read from the source. The model leaves out what these shapes never
//     meet: the k-column kernels' launch-shape fallback, alignment and size limits.
// Prints the model's kernel per product and token count for the switch unset, =2 and R197's read-once arm.
// Usage: test-cuda-mmvq-pascal-pick   (exit status 0 when every check passes)

#include "../ggml/src/ggml-cuda/mmvq-pascal-pick.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

// 3dfad876f (mmvq-pascal.cu): GGML_CUDA_MMVQ_PASCAL_MT on
static bool old_mt_enabled(const char * env) {
    return env != nullptr && atoi(env) != 0;
}

// 3dfad876f (mmvq-pascal.cu, ggml_cuda_mmvq_pascal): p.mt
static bool old_p_mt(const bool enabled, const ggml_type type, const bool ids, const int64_t ncols_dst,
                     const int64_t nchannels_dst) {
    return ncols_dst > 1 && enabled &&
        (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_N2P || type == GGML_TYPE_N3P ||
         type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_0) &&
        (!ids || nchannels_dst * ncols_dst <= 64);
}

// 3dfad876f (mmvq-pascal-kcol.cu): kcol_table and kcol_table_declines
struct old_rule {
    ggml_type type;
    int64_t   kmin;
    int64_t   kmax;
    uint32_t  mask;
};
#define KCOL_COLS(k) (1u << ((k) - 1))
#define KCOL_COLS_2_8 0xFEu
static const old_rule old_kcol_table[] = {
    { GGML_TYPE_IQ4_XS,  4097, INT64_MAX, KCOL_COLS(2) | KCOL_COLS(5) | KCOL_COLS(6) | KCOL_COLS(7) | KCOL_COLS(8) },
    { GGML_TYPE_Q4_K,    4097, INT64_MAX, KCOL_COLS(2) | KCOL_COLS(3) | KCOL_COLS(4) | KCOL_COLS(6) | KCOL_COLS(7) },
    { GGML_TYPE_Q8_0,    4097, INT64_MAX, KCOL_COLS_2_8 },
    { GGML_TYPE_IQ2_XXS, 4097, INT64_MAX, KCOL_COLS_2_8 },
    { GGML_TYPE_IQ2_XS,  4097, INT64_MAX, KCOL_COLS_2_8 },
    { GGML_TYPE_IQ2_S,   4097, INT64_MAX, KCOL_COLS_2_8 },
    { GGML_TYPE_IQ3_XXS, 4097, INT64_MAX, KCOL_COLS_2_8 },
    { GGML_TYPE_IQ3_S,   4097, INT64_MAX, KCOL_COLS_2_8 },
    { GGML_TYPE_Q5_K,    0,    4096,      KCOL_COLS(2) | KCOL_COLS(3) },
    { GGML_TYPE_Q6_K,    0,    4096,      KCOL_COLS_2_8 },
    { GGML_TYPE_Q4_0,    0,    4096,      KCOL_COLS(2) | KCOL_COLS(3) | KCOL_COLS(4) },
    { GGML_TYPE_Q5_0,    0,    4096,      KCOL_COLS(2) | KCOL_COLS(3) | KCOL_COLS(4) },
    { GGML_TYPE_Q2_0,    0,    4096,      KCOL_COLS(2) },
    { GGML_TYPE_IQ2_XXS, 0,    4096,      KCOL_COLS(2) },
    { GGML_TYPE_IQ3_S,   0,    4096,      KCOL_COLS(2) | KCOL_COLS(3) },
    { GGML_TYPE_IQ2_S,   0,    4096,      KCOL_COLS(3) },
    { GGML_TYPE_IQ3_XXS, 0,    4096,      KCOL_COLS(3) },
};
#undef KCOL_COLS
#undef KCOL_COLS_2_8

static bool old_kcol_table_declines(const ggml_type type, const int64_t K, const int64_t ncols) {
    for (const old_rule & r : old_kcol_table) {
        if (r.type == type && K >= r.kmin && K <= r.kmax) {
            return (r.mask >> (ncols - 1)) & 1u;
        }
    }
    return false;
}

static const std::vector<int64_t> KS    = { 32, 64, 256, 512, 640, 1024, 1536, 2048, 2049, 2560, 2880, 3072, 4096, 4097,
                                            5120, 6144, 8192, 10240, 16384, 65536 };
static const std::vector<int64_t> ROWS  = { 1, 64, 320, 512, 1024, 2560, 3072, 4096, 4097, 8192, 12288, 16384, 248320 };
static const std::vector<int64_t> CHANS = { 1, 2, 8, 16, 17, 32, 64 };

static void check_parse() {
    const char * values[] = { nullptr, "", "0", "00", "abc", "1", "2", "3", "-1", "-2", " 2", "2x", "02", "+2", "12", "1.5",
                              "2.0", "true", "on" };
    for (const char * v : values) {
        const int m = ggml_cuda_mmvq_pascal_mt_parse(v);
        CHECK((m != GGML_CUDA_MMVQ_PASCAL_MT_OFF) == old_mt_enabled(v));
        CHECK(m == GGML_CUDA_MMVQ_PASCAL_MT_OFF || m == GGML_CUDA_MMVQ_PASCAL_MT_ALL || m == GGML_CUDA_MMVQ_PASCAL_MT_RULE);
        CHECK((m == GGML_CUDA_MMVQ_PASCAL_MT_RULE) == (v != nullptr && atoi(v) == 2));
    }
    CHECK(ggml_cuda_mmvq_pascal_mt_parse(nullptr) == GGML_CUDA_MMVQ_PASCAL_MT_OFF);
    CHECK(ggml_cuda_mmvq_pascal_mt_parse("1") == GGML_CUDA_MMVQ_PASCAL_MT_ALL);
    CHECK(ggml_cuda_mmvq_pascal_mt_parse("2") == GGML_CUDA_MMVQ_PASCAL_MT_RULE);
}

// modes 0 and 1: the decisions of 3dfad876f; mode 2: nothing outside plain Q5_0 and Q8_0 products of 2 to 4 tokens
static void check_unchanged() {
    for (int ti = 0; ti < GGML_TYPE_COUNT; ti++) {
        const ggml_type type = (ggml_type) ti;
        for (const int64_t K : KS) {
            for (const int64_t nrows : ROWS) {
                for (int64_t ncols = 0; ncols <= 9; ncols++) {
                    // Q824's table, moved: the same for every width and token count of 2 to 8
                    if (ncols >= 2 && ncols <= 8 && nrows == 1) {
                        CHECK(ggml_cuda_kcol_table_declines(type, K, ncols) == old_kcol_table_declines(type, K, ncols));
                    }
                    for (const int mode : { 0, 1, 2 }) {
                        const bool kd = ggml_cuda_mmvq_pascal_mt_kcol_declines(mode, type, K, nrows, ncols);
                        if (mode != 2) {
                            CHECK(!kd);
                        } else if (kd) {
                            CHECK((type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q8_0) && ncols >= 2 && ncols <= 4);
                        }
                        for (const bool ids : { false, true }) {
                            for (const int64_t nch : CHANS) {
                                for (const int64_t nsm : { 1, 2 }) {
                                    const bool on = ggml_cuda_mmvq_pascal_mt_on(mode, type, ids, K, nrows, ncols, nch, nsm, 64);
                                    if (mode != 2) {
                                        CHECK(on == old_p_mt(mode == 1, type, ids, ncols, nch));
                                    } else {
                                        CHECK(on == (!ids && nch == 1 && nsm == 1 &&
                                                     ggml_cuda_mmvq_pascal_mt_rule_takes(type, K, nrows, ncols)));
                                        if (on) {
                                            CHECK((type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q8_0) && ncols >= 2 && ncols <= 4);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// The kernel a plain 2-D Q5_0 or Q8_0 product runs, in the order of mmvq.cu (k-column kernels in their mode 1 with the
// table on and _KCOL_EXACT off, then mmvq-pascal.cu, then llama.cpp's mul_mat_vec_q), leaving out the k-column kernels'
// launch-shape fallback and every alignment and size limit. skip: GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0.
static std::string kernel(const int mode, const bool skip, const ggml_type type, const int64_t K, const int64_t nrows,
                          const int64_t ncols) {
    bool kcol = true;
    if (skip && ncols <= 4) {
        kcol = false;                                           // C828
    } else if (ncols == 1) {
        kcol = type == GGML_TYPE_Q8_0 && K % 64 == 0 && (K / 64) % 4 != 0;
    } else if (ggml_cuda_kcol_table_declines(type, K, ncols)) {
        kcol = false;                                           // Q824
    } else if (ggml_cuda_mmvq_pascal_mt_kcol_declines(mode, type, K, nrows, ncols)) {
        kcol = false;                                           // C899
    }
    if (kcol) {
        return "kcol";
    }
    if (ncols > 4) {
        return "mmvq";
    }
    if (ncols == 1) {
        return "one";
    }
    return ggml_cuda_mmvq_pascal_mt_on(mode, type, false, K, nrows, ncols, 1, 1, 64) ? "mt" : "mc";
}

struct ornith_product {
    const char * name;
    ggml_type    type;
    int64_t      K;
    int64_t      nrows;
    const char * r197; // at 2, 3, 4 tokens: m the read-once kernel faster, o the kernel unset faster, t level (R197)
};

// R197's classes (compare.tsv.gz rows "label", depth 0, the repeated request): m or o where both loads' paired
// differences have one sign and each is larger than both arms' spreads between their loads, t otherwise
static const ornith_product ORNITH[] = {
    { "ssm_out",        GGML_TYPE_Q5_0, 8192, 3072,   "mmm" },
    { "attn_output",    GGML_TYPE_Q5_0, 8192, 3072,   "mmm" },
    { "attn_qkv",       GGML_TYPE_Q5_0, 3072, 12288,  "omo" },
    { "attn_gate",      GGML_TYPE_Q5_0, 3072, 8192,   "omo" },
    { "attn_q",         GGML_TYPE_Q5_0, 3072, 16384,  "omo" },
    { "output",         GGML_TYPE_Q5_0, 3072, 248320, "omo" },
    { "ffn_gate_shexp", GGML_TYPE_Q5_0, 3072, 1024,   "mmo" },
    { "ffn_up_shexp",   GGML_TYPE_Q5_0, 3072, 1024,   "mmo" },
    { "ssm_alpha",      GGML_TYPE_Q5_0, 3072, 64,     "tmo" },
    { "ssm_beta",       GGML_TYPE_Q5_0, 3072, 64,     "mmt" },
    { "ffn_down_shexp", GGML_TYPE_Q5_0, 1024, 3072,   "omt" },
    { "attn_k",         GGML_TYPE_Q8_0, 3072, 512,    "mmm" },
    { "attn_v",         GGML_TYPE_Q8_0, 3072, 512,    "mmm" },
};

static void check_ornith() {
    printf("%-16s %-6s %6s %7s | %-34s | %-34s | %s\n", "product", "type", "in", "rows", "unset: 1..8 tokens",
           "MT=2: 1..8 tokens", "R197 ro (MT=1, KCOL_SKIP=q5_0,q8_0)");
    for (const ornith_product & p : ORNITH) {
        std::string s0, s2, sr;
        for (int64_t k = 1; k <= 8; k++) {
            const std::string k0 = kernel(0, false, p.type, p.K, p.nrows, k);
            const std::string k2 = kernel(2, false, p.type, p.K, p.nrows, k);
            const std::string kr = kernel(1, true,  p.type, p.K, p.nrows, k);
            s0 += (k > 1 ? " " : "") + k0;
            s2 += (k > 1 ? " " : "") + k2;
            sr += (k > 1 ? " " : "") + kr;
            // unset: the kernels R197 read from the source
            if (k == 1) {
                CHECK(k0 == "one");
            } else if (k >= 5) {
                CHECK(k0 == "kcol");
            } else {
                CHECK(k0 == (p.type == GGML_TYPE_Q8_0 || p.K > 4096 ? "kcol" : "mc"));
            }
            // R197's read-once arm: the read-once kernel at 2 to 4 tokens
            CHECK(kr == (k == 1 ? "one" : k <= 4 ? "mt" : "kcol"));
            // mode 2: as unset outside 2 to 4 tokens; inside, the faster kernel of R197
            if (k == 1 || k >= 5) {
                CHECK(k2 == k0);
            } else {
                const char c = p.r197[k - 2];
                if (c == 'm') {
                    CHECK(k2 == "mt");
                } else if (c == 'o') {
                    CHECK(k2 == k0);
                } else {
                    CHECK(k2 == "mt" || k2 == k0);
                }
            }
        }
        printf("%-16s %-6s %6lld %7lld | %-34s | %-34s | %s\n", p.name, ggml_type_name(p.type), (long long) p.K,
               (long long) p.nrows, s0.c_str(), s2.c_str(), sr.c_str());
    }
}

int main() {
    check_parse();
    check_unchanged();
    check_ornith();
    printf("%d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
