// GGML_CUDA_MMVF_F32_MT_ROWS's parts that run without a GPU (ggml/src/ggml-cuda/mmvf-f32-rows.h; C807, C822): the
// value's limits per token count, and the decision whether GGML_CUDA_MMVF_F32_MT takes a matrix of so many rows times
// so many tokens.
//   - A single number, and the value unset, decide every product as C807's rule did (atoi of the value, default 32;
//     0 or below: any row count), over every row count from 1 to 1,100 and every token count from 0 to 12.
//   - auto is 48 rows at 4 tokens and 64 at 5-8; a list sets one limit per token count from 4 on, its last entry
//     holding after it; nothing is taken outside 4-8 tokens.
//   - Qwen3.8-Flash-Next's F32 products (the shared expert's gate, 1 row, and the router, 512 rows, at 4-8 tokens) are
//     decided the same unset, at 32 and at auto: the gate taken, the router left to cuBLAS.
// Usage: test-cuda-mmvf-f32-rows   (exit status 0 when every check passes)

#include "../ggml/src/ggml-cuda/mmvf-f32-rows.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static const int TMIN = GGML_CUDA_MMVF_F32_ROWS_MIN_TOKENS;
static const int TMAX = GGML_CUDA_MMVF_F32_ROWS_MAX_TOKENS;

// C807's rule (pascal-p106-v4-b12246-c807, mmvf.cu): max_rows = atoi(value) (32 unset); a product of more rows than a
// positive max_rows is left to cuBLAS, and the switch's other tests take only 4 to 8 tokens
static bool c807_take(const char * value, const int64_t ne01, const int64_t ne11) {
    const int max_rows = value ? atoi(value) : 32;
    if (max_rows > 0 && ne01 > max_rows) {
        return false;
    }
    return ne11 >= 4 && ne11 <= 8;
}

static void check_limits(const char * value, const std::vector<int> & want) {
    const ggml_cuda_mmvf_f32_rows r = ggml_cuda_mmvf_f32_rows_parse(value);
    for (int t = TMIN; t <= TMAX; t++) {
        const int got = r.max_rows[t - TMIN];
        const int exp = want[t - TMIN];
        n_checks++;
        if (got != exp) {
            n_fail++;
            fprintf(stderr, "value '%s' at %d tokens: limit %d, expected %d\n", value ? value : "(unset)", t, got, exp);
        }
    }
}

// every row count from 1 to 1,100 and every token count from 0 to 12 against a rule
template <typename F>
static void check_rule(const char * value, F want) {
    const ggml_cuda_mmvf_f32_rows r = ggml_cuda_mmvf_f32_rows_parse(value);
    int bad = 0;
    for (int64_t ne11 = 0; ne11 <= 12; ne11++) {
        for (int64_t ne01 = 1; ne01 <= 1100; ne01++) {
            const bool got = ggml_cuda_mmvf_f32_rows_take(r, ne01, ne11);
            const bool exp = want(ne01, ne11);
            n_checks++;
            if (got != exp) {
                n_fail++;
                if (bad++ < 5) {
                    fprintf(stderr, "value '%s', %lld rows, %lld tokens: %s, expected %s\n", value ? value : "(unset)",
                            (long long) ne01, (long long) ne11, got ? "taken" : "left", exp ? "taken" : "left");
                }
            }
        }
    }
}

int main() {
    // a single number, and the value unset: C807's rule exactly
    std::vector<std::string> singles = { "", "0", "-1", "-64", "1", "7", "31", "32", "33", "40", "47", "48", "49", "63",
                                         "64", "65", "100", "128", "511", "512", "513", "1024", "abc", "32abc", " 48",
                                         "+48", "048", "x,48", "AUTO", "Auto", "auto ", " auto" };
    for (int v = -3; v <= 600; v++) {
        singles.push_back(std::to_string(v));
    }
    for (const std::string & s : singles) {
        if (s.find(',') != std::string::npos) {
            continue; // a list: below
        }
        check_rule(s.c_str(), [&](int64_t ne01, int64_t ne11) { return c807_take(s.c_str(), ne01, ne11); });
    }
    check_rule(nullptr, [](int64_t ne01, int64_t ne11) { return c807_take(nullptr, ne01, ne11); });
    check_limits(nullptr, { 32, 32, 32, 32, 32 });
    check_limits("0", { 0, 0, 0, 0, 0 });
    check_limits("AUTO", { 0, 0, 0, 0, 0 }); // only "auto" itself: anything else non-numeric reads as atoi does

    // auto: 48 rows at 4 tokens, 64 at 5-8
    check_limits("auto", { 48, 64, 64, 64, 64 });
    check_limits(GGML_CUDA_MMVF_F32_ROWS_AUTO, { 48, 64, 64, 64, 64 });
    check_rule("auto", [](int64_t ne01, int64_t ne11) {
        return (ne11 == 4 && ne01 <= 48) || (ne11 >= 5 && ne11 <= 8 && ne01 <= 64);
    });

    // lists: one limit per token count from 4 on, the last one holding after it, entries past 8 tokens ignored
    check_limits("48,64", { 48, 64, 64, 64, 64 });
    check_limits("64,48", { 64, 48, 48, 48, 48 });
    check_limits("33,49,65,31,7", { 33, 49, 65, 31, 7 });
    check_limits("33,49,65,31,7,99,5", { 33, 49, 65, 31, 7 });
    check_limits("40,0", { 40, 0, 0, 0, 0 });
    check_limits("0,40", { 0, 40, 40, 40, 40 });
    check_limits("16,", { 16, 0, 0, 0, 0 });
    check_limits(",16", { 0, 16, 16, 16, 16 });
    check_limits("-5,20", { -5, 20, 20, 20, 20 });
    check_limits("48, 64", { 48, 64, 64, 64, 64 }); // atoi skips the leading space
    check_rule("64,48", [](int64_t ne01, int64_t ne11) {
        return (ne11 == 4 && ne01 <= 64) || (ne11 >= 5 && ne11 <= 8 && ne01 <= 48);
    });
    check_rule("33,49,65,31,7", [](int64_t ne01, int64_t ne11) {
        static const int lim[5] = { 33, 49, 65, 31, 7 };
        return ne11 >= 4 && ne11 <= 8 && ne01 <= lim[ne11 - 4];
    });
    check_rule("40,0", [](int64_t ne01, int64_t ne11) {
        return (ne11 == 4 && ne01 <= 40) || (ne11 >= 5 && ne11 <= 8);
    });

    // Qwen3.8-Flash-Next: the gate (1 row) taken and the router (512 rows) left at 4-8 tokens, unset, at 32 and at auto,
    // and nothing taken at 1-3 tokens (mul_mat_vec_f takes those without the switch)
    for (const char * v : { (const char *) nullptr, "32", "auto", "48,64" }) {
        const ggml_cuda_mmvf_f32_rows r = ggml_cuda_mmvf_f32_rows_parse(v);
        for (int64_t ne11 = 1; ne11 <= 8; ne11++) {
            CHECK(ggml_cuda_mmvf_f32_rows_take(r, 1, ne11) == (ne11 >= 4));
            CHECK(!ggml_cuda_mmvf_f32_rows_take(r, 512, ne11));
        }
    }

    // the replay set f32x's shapes at 4-8 tokens under auto and under 64,48 (scripts/record_checks.sh: F32X4AUTO,
    // F32X4LIST and the runs above them select these)
    {
        const ggml_cuda_mmvf_f32_rows a = ggml_cuda_mmvf_f32_rows_parse("auto");
        const ggml_cuda_mmvf_f32_rows l = ggml_cuda_mmvf_f32_rows_parse("64,48");
        for (int64_t ne11 = 4; ne11 <= 8; ne11++) {
            for (int64_t m : { 1, 7, 17, 31, 32, 33, 48 }) {
                CHECK(ggml_cuda_mmvf_f32_rows_take(a, m, ne11));
                CHECK(ggml_cuda_mmvf_f32_rows_take(l, m, ne11));
            }
            for (int64_t m : { 49, 64 }) {
                CHECK(ggml_cuda_mmvf_f32_rows_take(a, m, ne11) == (ne11 >= 5));
                CHECK(ggml_cuda_mmvf_f32_rows_take(l, m, ne11) == (ne11 == 4));
            }
            for (int64_t m : { 65, 129, 256, 512 }) {
                CHECK(!ggml_cuda_mmvf_f32_rows_take(a, m, ne11));
                CHECK(!ggml_cuda_mmvf_f32_rows_take(l, m, ne11));
            }
        }
    }

    printf("test-cuda-mmvf-f32-rows: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
