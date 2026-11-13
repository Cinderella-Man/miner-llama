#include "mmvq-pascal-kcol.cuh"
#include "mmvq-pascal.cuh"
#include "mmvq-pascal-q5k.cuh"
#include "mmvq-pascal-dmt.cuh"
#include "mmvq-pascal-q3kmt.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "mmvq-pascal-kcol-impl.cuh"

namespace {

static int kcol_env_int(const char * name, const int def) {
    const char * env = getenv(name);
    return env ? atoi(env) : def;
}

static int kcol_mode() {
    static const int mode = kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL", 1);
    return mode;
}

struct kcol_overrides {
    int r;
    int itt;
    int smem;
    int nsm;
    int smsm;
};

static const kcol_overrides & kcol_shape_env() {
    static const kcol_overrides o = {
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_R", 0),
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_ITT", 0),
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM", 0),
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_NSM", 0),
        kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM", 0),
    };
    return o;
}

// GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_K,q6_K: weight types (ggml type names) left to the other kernels
static bool kcol_skip_type(const ggml_type type) {
    static const std::vector<ggml_type> skip = [] {
        std::vector<ggml_type> types;
        const char * env = getenv("GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP");
        const std::string list = env ? env : "";
        size_t a = 0;
        while (a < list.size()) {
            size_t b = list.find(',', a);
            if (b == std::string::npos) {
                b = list.size();
            }
            const std::string name = list.substr(a, b - a);
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                const char * tn = ggml_type_name((ggml_type) t);
                if (tn && name == tn) {
                    types.push_back((ggml_type) t);
                }
            }
            a = b + 1;
        }
        return types;
    }();
    return std::find(skip.begin(), skip.end(), type) != skip.end();
}

// C828: the skip hands a type's products to the Pascal kernels (mmvq-pascal.cu), which take 1 to 4 columns
// (MMVQP_MAX_COLS); products of 5 to 8 columns stay with these kernels, as without the switch (on a P106 llama.cpp's
// mul_mat_vec_q, where they went before, is slower there: R150)
static constexpr int64_t kcol_skip_max_cols = 4;

static int kcol_type(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return kcol::KQ8_0;
        case GGML_TYPE_Q2_0:   return kcol::KQ2_0;
        case GGML_TYPE_Q3_K:   return kcol::KQ3_K;
        case GGML_TYPE_Q4_K:   return kcol::KQ4_K;
        case GGML_TYPE_Q5_K:   return kcol::KQ5_K;
        case GGML_TYPE_Q6_K:   return kcol::KQ6_K;
        case GGML_TYPE_IQ4_XS: return kcol::KIQ4_XS;
        case GGML_TYPE_Q4_0:   return kcol::KQ4_0;
        case GGML_TYPE_Q5_0:   return kcol::KQ5_0;
        case GGML_TYPE_IQ4_NL: return kcol::KIQ4_NL;
        case GGML_TYPE_IQ2_XXS: return kcol::KIQ2_XXS;
        case GGML_TYPE_IQ2_XS:  return kcol::KIQ2_XS;
        case GGML_TYPE_IQ2_S:   return kcol::KIQ2_S;
        case GGML_TYPE_IQ3_XXS: return kcol::KIQ3_XXS;
        case GGML_TYPE_IQ3_S:   return kcol::KIQ3_S;
        case GGML_TYPE_IQ1_M:   return kcol::KIQ1_M;
        default:               return -1;
    }
}

// GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE: 1 (default) the dispatch table below, 0 the k-column kernels wherever they take the
// product (as before Q824)
static bool kcol_table_on() {
    static const bool on = kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE", 1) != 0;
    return on;
}

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
struct kcol_rule {
    ggml_type type;
    int64_t   kmin;
    int64_t   kmax;
    uint32_t  mask;
};
#define KCOL_COLS(k) (1u << ((k) - 1))
#define KCOL_COLS_2_8 0xFEu
static const kcol_rule kcol_table[] = {
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

// Whether the table leaves a product of ncols (2 to 8) columns of type with rows of K weights to the other kernels.
static bool kcol_table_declines(const ggml_type type, const int64_t K, const int64_t ncols) {
    for (const kcol_rule & r : kcol_table) {
        if (r.type == type && K >= r.kmin && K <= r.kmax) {
            return (r.mask >> (ncols - 1)) & 1u;
        }
    }
    return false;
}

// kcol::blocks_fn: the resident blocks per SM that the registers of the type's kernel allow
static int kcol_blocks(const int t, const int ncols, const int R) {
    switch (t) {
        case kcol::KQ8_0:    return ggml_cuda_kcol_blocks_q8_0(ncols, R);
        case kcol::KQ2_0:    return ggml_cuda_kcol_blocks_q2_0(ncols, R);
        case kcol::KQ3_K:    return ggml_cuda_kcol_blocks_q3_K(ncols, R);
        case kcol::KQ4_K:    return ggml_cuda_kcol_blocks_q4_K(ncols, R);
        case kcol::KQ5_K:    return ggml_cuda_kcol_blocks_q5_K(ncols, R);
        case kcol::KQ6_K:    return ggml_cuda_kcol_blocks_q6_K(ncols, R);
        case kcol::KIQ4_XS:  return ggml_cuda_kcol_blocks_iq4_xs(ncols, R);
        case kcol::KQ4_0:    return ggml_cuda_kcol_blocks_q4_0(ncols, R);
        case kcol::KQ5_0:    return ggml_cuda_kcol_blocks_q5_0(ncols, R);
        case kcol::KIQ4_NL:  return ggml_cuda_kcol_blocks_iq4_nl(ncols, R);
        case kcol::KIQ2_XXS: return ggml_cuda_kcol_blocks_iq2_xxs(ncols, R);
        case kcol::KIQ2_XS:  return ggml_cuda_kcol_blocks_iq2_xs(ncols, R);
        case kcol::KIQ2_S:   return ggml_cuda_kcol_blocks_iq2_s(ncols, R);
        case kcol::KIQ3_XXS: return ggml_cuda_kcol_blocks_iq3_xxs(ncols, R);
        case kcol::KIQ3_S:   return ggml_cuda_kcol_blocks_iq3_s(ncols, R);
        case kcol::KIQ1_M:   return ggml_cuda_kcol_blocks_iq1_m(ncols, R);
        default:             return 0;
    }
}

// GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT (C805): 0 (default) as before; 1 or 2: a product of 2 to 8 columns of a type whose
// one-token sums are this kernel's (kcol_one_token_sums) is never left to another kernel for the dispatch table (Q824)
// or for shared memory: where its activations do not fit the budget (GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM, or the default
// rule's 48 KB), it runs as kcol::pick_exact plans it (1: within 48 KB, in one launch where all columns fit; 2: launches
// within the budget first). Elsewhere such a product runs on llama.cpp's mul_mat_vec_q (5 to 8 columns, and Q3_K and
// IQ4_NL at every count), whose sums are not the one-token sums, or on mmvq-pascal.cu's kernel of 2 to 4 tokens (Q2_0,
// Q8_0); here each token's sums stay the one-token ones at every token count. The one-column products this kernel runs
// (Q3_K, IQ4_NL, those Q8_0 rows; in mode 2 every type) stay here too, so that a budget under one column's needs
// (29 KB for Q3_K's 32-lane rows) cannot move the one-token sums themselves to another kernel.
// GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT_FAULT=1 (checks only): every launch after a split product's first reads the first
// launch's activations, so a split product must differ.
static int kcol_exact() {
    static const int v = kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT", 0);
    return v == 1 || v == 2 ? v : 0;
}

static bool kcol_exact_fault() {
    static const bool f = kcol_env_int("GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT_FAULT", 0) != 0;
    return f;
}

// C805: the types whose one-token products have this kernel's sums: Q3_K and IQ4_NL, whose one-token products it runs;
// Q8_0 in rows whose 64-weight chunks aren't a multiple of 4, the same; and Q2_0, whose one-token Pascal kernel
// (mmvq-pascal.cu) sums the same 128-weight chunks (two blocks per lane) over the same lanes in the same order with the
// same integer and FP32 operations.
static bool kcol_one_token_sums(const int t, const int64_t K) {
    return t == kcol::KQ3_K || t == kcol::KIQ4_NL || t == kcol::KQ2_0 ||
        (t == kcol::KQ8_0 && K % 64 == 0 && (K / 64) % 4 != 0);
}

// shared memory per SM of device dev, read once
static int kcol_smem_sm(const int dev) {
    static int smem[GGML_CUDA_MAX_DEVICES] = {};
    if (smem[dev] == 0) {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess || v <= 0) {
            (void) cudaGetLastError();
            v = 96 * 1024;
        }
        smem[dev] = v;
    }
    return smem[dev];
}

} // namespace

bool ggml_cuda_mmvq_pascal_kcol_on(const ggml_type type) {
    return kcol_mode() != 0 && kcol_type(type) >= 0 && !kcol_skip_type(type);
}

bool ggml_cuda_mmvq_pascal_kcol_skips(const ggml_type type, const int64_t ncols) {
    return ncols >= 1 && ncols <= kcol_skip_max_cols && kcol_skip_type(type);
}

bool ggml_cuda_mmvq_pascal_kcol(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst) {
    const int mode = kcol_mode();
    if (mode == 0 || !ggml_cuda_mmvq_pascal_active(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    const int t = kcol_type(src0->type);
    if (t < 0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // plain 2-D products: rows of src0 nb[1] bytes apart, ncols FP32 columns of src1 and dst
    const int64_t K     = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int64_t ncols = src1->ne[1];
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            src1->ne[0] != K || dst->ne[0] != nrows || dst->ne[1] != ncols || ncols < 1 || ncols > KCOL_MAX_COLS) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src1->nb[1] % sizeof(float) != 0 || dst->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (src0->nb[1] > INT_MAX || nrows > INT_MAX || K > INT_MAX || src1->nb[1] / sizeof(float) > INT_MAX ||
            dst->nb[1] / sizeof(float) > INT_MAX) {
        return false;
    }
    // GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP's types at 1 to 4 columns (C828)
    if (ggml_cuda_mmvq_pascal_kcol_skips(src0->type, ncols)) {
        return false;
    }
    // GGML_CUDA_MMVQ_PASCAL_Q5K_MT (_Q5K_1R, _RO8): Q5_K's products of 2 to 4 columns (1, 5 to 8) go to
    // mmvq-pascal-q5k.cu (each token's sums those of the one-token kernel)
    if (src0->type == GGML_TYPE_Q5_K && ggml_cuda_mmvq_pascal_q5k_mt_takes(K, nrows, ncols, src0->nb[1], src0->data)) {
        return false;
    }
    // GGML_CUDA_MMVQ_PASCAL_DMT (_RO8): IQ4_XS's and Q4_K's products of 2 to 4 (5 to 8) columns go to mmvq-pascal-dmt.cu
    // (the same), and with GGML_CUDA_MMVQ_PASCAL_Q6K_MT Q6_K's of 2 to 4 (C727)
    if (ggml_cuda_mmvq_pascal_dmt_takes(src0->type, K, nrows, ncols, src0->nb[1], src0->data)) {
        return false;
    }
    // GGML_CUDA_MMVQ_PASCAL_Q3K_MT: Q3_K's products of 2 to 8 columns go to mmvq-pascal-q3kmt.cu (each token's sums those
    // of this kernel at one column)
    if (src0->type == GGML_TYPE_Q3_K && ggml_cuda_mmvq_pascal_q3kmt_takes(K, nrows, ncols, src0->nb[1], src0->data)) {
        return false;
    }
    if (ncols == 1 && mode == 1) {
        // one column: only where the one-token kernels are slow or missing
        const bool q8_short = t == kcol::KQ8_0 && K % 64 == 0 && (K / 64) % 4 != 0;
        if (t != kcol::KQ3_K && t != kcol::KIQ4_NL && !q8_short) {
            return false;
        }
    }

    // C805: products whose one-token sums are this kernel's stay here (GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT)
    const bool exact = kcol_exact() != 0 && kcol_one_token_sums(t, K);

    // Q824: where the other kernels are faster (mode 1, 2 or more columns)
    const bool table = mode == 1 && ncols >= 2 && kcol_table_on() && !exact;
    if (table && kcol_table_declines(src0->type, K, ncols)) {
        return false;
    }

    const kcol_overrides & o = kcol_shape_env();
    const int nsm = o.nsm > 0 ? o.nsm : ggml_cuda_info().devices[ctx.device].nsm;
    const int smsm = o.smsm > 0 ? o.smsm : kcol_smem_sm(ctx.device);
    kcol::shape s;
    int cols = 0; // C805: columns per launch (0: all)
    if (!kcol::pick_shape(t, (int) K, (int) nrows, (int) ncols, (int) src0->nb[1], (uintptr_t) src0->data, nsm, o.r, o.itt,
                          o.smem, s, smsm, kcol_blocks)) {
        if (!exact || !kcol::pick_exact(t, (int) K, (int) nrows, (int) ncols, (int) src0->nb[1], (uintptr_t) src0->data, nsm,
                                        o.r, o.itt, o.smem, kcol_exact(), s, cols, smsm, kcol_blocks)) {
            return false;
        }
    }
    if (table && s.fallback) {
        return false;
    }

    ggml_cuda_pool_alloc<char> rec(ctx.pool(), kcol::records_bytes(t, (int) K, (int) ncols));

    kcol::params p;
    p.vx             = (const char *) src0->data;
    p.y              = (const float *) src1->data;
    p.rec            = (uint32_t *) rec.get();
    p.dst            = (float *) dst->data;
    p.K              = (int) K;
    p.nrows          = (int) nrows;
    p.row_bytes      = (int) src0->nb[1];
    p.ncols          = (int) ncols;
    p.stride_col_y   = (int) (src1->nb[1] / sizeof(float));
    p.stride_col_dst = (int) (dst->nb[1] / sizeof(float));
    p.s              = s;
    p.cols           = cols;
    p.fault          = cols > 0 && kcol_exact_fault() ? 1 : 0;

    cudaStream_t stream = ctx.stream();
    switch (t) {
        case kcol::KQ8_0:   ggml_cuda_kcol_launch_q8_0(p, stream);   break;
        case kcol::KQ2_0:   ggml_cuda_kcol_launch_q2_0(p, stream);   break;
        case kcol::KQ3_K:   ggml_cuda_kcol_launch_q3_K(p, stream);   break;
        case kcol::KQ4_K:   ggml_cuda_kcol_launch_q4_K(p, stream);   break;
        case kcol::KQ5_K:   ggml_cuda_kcol_launch_q5_K(p, stream);   break;
        case kcol::KQ6_K:   ggml_cuda_kcol_launch_q6_K(p, stream);   break;
        case kcol::KIQ4_XS: ggml_cuda_kcol_launch_iq4_xs(p, stream); break;
        case kcol::KQ4_0:   ggml_cuda_kcol_launch_q4_0(p, stream);   break;
        case kcol::KQ5_0:   ggml_cuda_kcol_launch_q5_0(p, stream);   break;
        case kcol::KIQ4_NL: ggml_cuda_kcol_launch_iq4_nl(p, stream); break;
        case kcol::KIQ2_XXS: ggml_cuda_kcol_launch_iq2_xxs(p, stream); break;
        case kcol::KIQ2_XS:  ggml_cuda_kcol_launch_iq2_xs(p, stream);  break;
        case kcol::KIQ2_S:   ggml_cuda_kcol_launch_iq2_s(p, stream);   break;
        case kcol::KIQ3_XXS: ggml_cuda_kcol_launch_iq3_xxs(p, stream); break;
        case kcol::KIQ3_S:   ggml_cuda_kcol_launch_iq3_s(p, stream);   break;
        case kcol::KIQ1_M:   ggml_cuda_kcol_launch_iq1_m(p, stream);   break;
        default:            GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_pascal_kcol_on(const ggml_type type) {
    GGML_UNUSED(type);
    return false;
}

bool ggml_cuda_mmvq_pascal_kcol_skips(const ggml_type type, const int64_t ncols) {
    GGML_UNUSED_VARS(type, ncols);
    return false;
}

bool ggml_cuda_mmvq_pascal_kcol(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
