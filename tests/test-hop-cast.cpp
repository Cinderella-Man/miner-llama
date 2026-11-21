// test-hop-cast: checks for the card-hop switches (LLAMA_HOP_CAST, LLAMA_HOP_CAST_AT, GGML_CUDA_HOP_*)
//
//  1. Round trips through ggml_cast on every backend there is (the CPU, and each GPU): F32 -> F16 / BF16 / Q8_0 /
//     Q6_K / Q5_1 / Q5_0 / Q4_1 / Q4_0 / IQ4_NL -> F32 at the shapes of Flash-Next's hops, [2560, 4, T] for T = 1, 4 and
//     512. Data the narrow type represents comes back bit for bit; random data (normal, with outlier channels) comes
//     back within the type's rounding bound (hop_bound: half a step of the type's grid, a whole step where the grid
//     clips the side opposite the block's largest value, plus the rounding of the scales to F16). Each GPU's round trip
//     is also compared with the CPU's (printed: the share of values equal bit for bit, the largest difference).
//  2. With -m <model>: the model's logits for one token sequence, evaluated as a prompt in micro-batches, one 4-token
//     pass (a verify pass) and single-token steps, every position's logits kept, once per arm. An arm is a name and
//     environment variables set before its context is made (every variable any arm names is unset first), e.g.
//         --arm off --arm "q8_0 LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1" --arm "staged exact GGML_CUDA_HOP_STAGED=2"
//     The first arm is the reference and runs twice (its logits must repeat bit for bit). An arm marked "exact" must
//     give the reference's logits bit for bit (the staged hop copies bytes exactly). For every arm, an eval-callback pass
//     over the same sequence checks each hop_wide tensor (the residual after its round trip) against its hop_narrow's
//     source (the residual before the cast) within the bound of part 1, counts the hops, and checks that layer 0's
//     output (l_last-0, before any boundary) equals the reference's bit for bit. Without hops nothing may differ.
//     Printed per arm: max |logit difference|, mean and max KL divergence from the reference, top-1 agreement; per
//     boundary of a cast arm: the source's largest |x| and rms, the round trip's largest and rms error.
// Usage: test-hop-cast [--unit-only] [-m model.gguf] [-ngl N] [-ts a,b,...] [-t threads] [--arm "name [exact] VAR=value ..."]...
//                      [--prompt N] [--ubatch N] [--tsv file]
// Exit status 0 when every check passes.

#include "llama.h"
#include "llama-cpp.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

static void check(bool ok, const char * fmt, ...) {
    n_checks++;
    if (!ok) {
        n_fail++;
        va_list ap;
        va_start(ap, fmt);
        fprintf(stderr, "FAIL: ");
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        va_end(ap);
    }
}

// ---- bounds -----------------------------------------------------------------------------------------------------------

// the largest |x|, the smallest and the largest x of the block of n values around element i (rows of ne0 elements)
struct block_range { double amax = 0.0, vmin = INFINITY, vmax = -INFINITY; };

static block_range range_of(const float * x, size_t i, size_t ne0, size_t n) {
    const size_t row = i / ne0, col = i % ne0;
    const size_t b0  = row * ne0 + col / n * n;
    block_range r;
    for (size_t j = b0; j < b0 + n; ++j) {
        r.amax = std::max(r.amax, fabs((double) x[j]));
        r.vmin = std::min(r.vmin, (double) x[j]);
        r.vmax = std::max(r.vmax, (double) x[j]);
    }
    return r;
}

// the largest error the round trip through type may leave at element i of x (rows of ne0 elements)
static double hop_bound(ggml_type type, const float * x, size_t i, size_t ne0) {
    const double v   = fabs((double) x[i]);
    const double f16 = std::ldexp(1.0, -11); // the relative rounding of a scale to F16
    switch (type) {
        case GGML_TYPE_F16:  return std::max(v * f16, std::ldexp(1.0, -25));
        case GGML_TYPE_BF16: return v * std::ldexp(1.0, -8);
        // d = the block's max |x| / 127: half a step, plus the rounding of d to F16 at |q| <= 127
        case GGML_TYPE_Q8_0: return 0.57 * range_of(x, i, ne0, 32).amax / 127.0 + 4e-6;
        // d = max |x| / 2^(bits-1) with q in [-2^(bits-1), 2^(bits-1) - 1]: the side opposite the largest value clips at a
        // whole step, plus the rounding of d at |q| <= 2^(bits-1)
        case GGML_TYPE_Q5_0: return (1.0 + 16 * f16) * range_of(x, i, ne0, 32).amax / 16.0 + 4e-6;
        case GGML_TYPE_Q4_0: return (1.0 +  8 * f16) * range_of(x, i, ne0, 32).amax /  8.0 + 4e-6;
        // d = (max - min) / (2^bits - 1), m = min: half a step, the rounding of d at q <= 2^bits - 1 and of m
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_1: {
            const block_range r = range_of(x, i, ne0, 32);
            const double qmax = type == GGML_TYPE_Q5_1 ? 31.0 : 15.0;
            const double d = (r.vmax - r.vmin) / qmax;
            return (0.5 + qmax * f16) * d * 1.01 + fabs(r.vmin) * f16 * 1.01 + 4e-6;
        }
        // the 16-value grid of IQ4_NL ({-127 .. 113} d, gaps up to 24 d) and its least-squares refit of d: a quarter of the
        // block's max |x| (a layout or index error is the whole of it)
        case GGML_TYPE_IQ4_NL: return 0.25 * range_of(x, i, ne0, 32).amax + 4e-6;
        // Q6_K: a 16-value sub-block's scale s ~ its max |x| / 32 (clipped at 31 steps on one side, searched over
        // 31.1 .. 32.9), stored as an int8 multiple of the super-block's d = max |s| / 128 (an error of d / 2 a step):
        // within 0.0606 of the sub-block's max |x| and 15.5 d (0.004 of the super-block's max |x|), bounded by 0.08 and
        // 0.005
        case GGML_TYPE_Q6_K: return 0.08 * range_of(x, i, ne0, 16).amax + 0.005 * range_of(x, i, ne0, 256).amax + 4e-6;
        default: return 0.0;
    }
}

// count of elements of b outside the bound around a, and the largest error relative to the bound
static size_t out_of_bound(ggml_type type, const float * a, const float * b, size_t n, size_t ne0, double & worst) {
    size_t bad = 0;
    worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double bound = hop_bound(type, a, i, ne0);
        const double err   = fabs((double) a[i] - (double) b[i]);
        if (!(err <= bound)) {
            bad++;
        }
        worst = std::max(worst, bound > 0 ? err / bound : (err > 0 ? INFINITY : 0.0));
    }
    return bad;
}

// ---- part 1: round trips on every backend -------------------------------------------------------------------------

static float to_f16(float v) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
}

// data the narrow type holds exactly: values rounded to F16 or BF16, or blocks of q * d (+ m) with F16 scales whose
// largest value is the one the quantizer derives its scale from (Q8_0: ±127 d; Q5_0 / Q4_0: -16 d / -8 d; Q5_1 / Q4_1:
// m = ±2^e, q from 0 to 31 / 15; IQ4_NL: d times the grid, -127 d among them; Q6_K: per 16-value sub-block d sc q with
// one q = -32, one sub-block of each super-block at sc = -128)
static std::vector<float> representable(ggml_type type, size_t n, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n);
    // an F16 scale of either sign, exponents spread over 9 octaves
    auto scale = [&](size_t b) {
        const float d = to_f16(std::ldexp(1.0f + fabsf(nd(rng)), -6 + (int) (b % 9)));
        return b & 2 ? -d : d;
    };
    if (type == GGML_TYPE_Q8_0) {
        std::uniform_int_distribution<int> qd(-127, 127);
        for (size_t b = 0; b < n / 32; ++b) {
            const float d = to_f16(std::ldexp(1.0f + fabsf(nd(rng)), -6 + (int) (b % 9)));
            for (size_t j = 0; j < 32; ++j) {
                x[b*32 + j] = (float) qd(rng) * d;
            }
            x[b*32 + b % 32] = (b & 1 ? -127.0f : 127.0f) * d;
        }
    } else if (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q4_0) {
        const int h = type == GGML_TYPE_Q5_0 ? 16 : 8;
        std::uniform_int_distribution<int> qd(-h, h - 1);
        for (size_t b = 0; b < n / 32; ++b) {
            const float d = scale(b);
            for (size_t j = 0; j < 32; ++j) {
                x[b*32 + j] = (float) qd(rng) * d;
            }
            x[b*32 + b % 32] = (float) -h * d;
        }
    } else if (type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q4_1) {
        const int qmax = type == GGML_TYPE_Q5_1 ? 31 : 15;
        std::uniform_int_distribution<int> qd(0, qmax);
        for (size_t b = 0; b < n / 32; ++b) {
            const float d = fabsf(scale(b));
            const float m = b % 5 == 0 ? 0.0f : std::ldexp(b & 1 ? -1.0f : 1.0f, -6 + (int) (b % 7));
            for (size_t j = 0; j < 32; ++j) {
                x[b*32 + j] = m + (float) qd(rng) * d;
            }
            x[b*32 + b % 32]        = m;
            x[b*32 + (b + 16) % 32] = m + (float) qmax * d;
        }
    } else if (type == GGML_TYPE_IQ4_NL) {
        static const int8_t grid[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
        std::uniform_int_distribution<int> qd(0, 15);
        for (size_t b = 0; b < n / 32; ++b) {
            const float d = scale(b);
            for (size_t j = 0; j < 32; ++j) {
                x[b*32 + j] = d * (float) grid[qd(rng)];
            }
            x[b*32 + b % 32] = d * (float) grid[0];
        }
    } else if (type == GGML_TYPE_Q6_K) {
        std::uniform_int_distribution<int> qd(-32, 31), sd(1, 127);
        for (size_t b = 0; b < n / 256; ++b) {
            const float d = scale(b);
            for (size_t s = 0; s < 16; ++s) {
                const int   sc = s == b % 16 ? -128 : (sd(rng) * (s & 1 ? -1 : 1));
                const float ds = d * (float) sc;
                for (size_t j = 0; j < 16; ++j) {
                    x[b*256 + s*16 + j] = ds * (float) qd(rng);
                }
                x[b*256 + s*16 + (b + s) % 16] = ds * -32.0f;
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            const float v = nd(rng) * std::ldexp(1.0f, (int) (i % 13) - 6);
            x[i] = type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)) : ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
        }
    }
    return x;
}

// random residual-like data: normal values with a few outlier channels per row
static std::vector<float> residual_like(size_t n, size_t ne0, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t col = i % ne0;
        x[i] = nd(rng) * (col % 97 == 5 ? 60.0f : (col % 31 == 7 ? 8.0f : 1.0f));
    }
    return x;
}

static std::vector<float> round_trip(ggml_backend_t backend, ggml_type type, const std::vector<float> & x, int64_t ne0,
        int64_t ne1, int64_t ne2) {
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * in     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_tensor * narrow = ggml_cast(ctx, in, type);
    ggml_tensor * wide   = ggml_cast(ctx, narrow, GGML_TYPE_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, wide);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(in, x.data(), 0, ggml_nbytes(in));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> y(x.size());
    ggml_backend_tensor_get(wide, y.data(), 0, ggml_nbytes(wide));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return y;
}

// the narrow types LLAMA_HOP_CAST takes (llama-context.cpp), by the name ggml_type_name gives, in any case
static const ggml_type hop_types[] = {
    GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0,
    GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL,
};

static ggml_type hop_type_of(const std::string & v) {
    for (ggml_type t : hop_types) {
        const std::string name = ggml_type_name(t);
        if (name.size() == v.size() && std::equal(name.begin(), name.end(), v.begin(),
                [](char a, char b) { return tolower((unsigned char) a) == tolower((unsigned char) b); })) {
            return t;
        }
    }
    return GGML_TYPE_F32;
}

static void test_round_trips() {
    const int64_t ne0 = 2560, ne1 = 4;
    printf("part 1: round trips F32 -> type -> F32 at [2560, 4, T]\n");
    printf("  %-14s %-6s %4s  %-22s %-56s %s\n", "backend", "type", "T", "representable data", "residual-like data",
            "against the CPU's round trip");
    // the CPU first: each GPU's residual-like round trip is compared with the CPU's
    std::vector<ggml_backend_dev_t> devs;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        const auto dt = ggml_backend_dev_type(dev);
        if (dt == GGML_BACKEND_DEVICE_TYPE_CPU) {
            devs.insert(devs.begin(), dev);
        } else if (dt == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(dev);
        }
    }
    std::map<std::pair<int, int64_t>, std::vector<float>> cpu_y;
    for (ggml_backend_dev_t dev : devs) {
        const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        for (ggml_type type : hop_types) {
            for (int64_t ne2 : { 1, 4, 512 }) {
                const size_t n = (size_t) (ne0 * ne1 * ne2);
                std::mt19937 rng(1234 + (unsigned) type * 7 + (unsigned) ne2);

                const std::vector<float> xr = representable(type, n, rng);
                const std::vector<float> yr = round_trip(backend, type, xr, ne0, ne1, ne2);
                const bool exact = memcmp(xr.data(), yr.data(), n * sizeof(float)) == 0;
                check(exact, "%s %s T=%lld: representable data did not come back bit for bit",
                        ggml_backend_name(backend), ggml_type_name(type), (long long) ne2);

                const std::vector<float> x = residual_like(n, ne0, rng);
                const std::vector<float> y = round_trip(backend, type, x, ne0, ne1, ne2);
                double worst = 0.0;
                const size_t bad = out_of_bound(type, x.data(), y.data(), n, ne0, worst);
                double max_err = 0.0, rms = 0.0, rms_x = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    max_err = std::max(max_err, fabs((double) x[i] - y[i]));
                    rms   += ((double) x[i] - y[i]) * ((double) x[i] - y[i]);
                    rms_x += (double) x[i] * x[i];
                }
                rms   = sqrt(rms / n);
                rms_x = sqrt(rms_x / n);
                check(bad == 0, "%s %s T=%lld: %zu of %zu elements outside the bound (worst %.3f of it)",
                        ggml_backend_name(backend), ggml_type_name(type), (long long) ne2, bad, n, worst);

                // a GPU's round trip against the CPU's (the reference implementation of the type)
                char vs_cpu[96] = "(the reference)";
                const auto key = std::make_pair((int) type, ne2);
                if (is_cpu) {
                    cpu_y[key] = y;
                } else if (cpu_y.count(key)) {
                    const std::vector<float> & yc = cpu_y[key];
                    size_t same = 0;
                    double max_diff = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        same    += memcmp(&y[i], &yc[i], sizeof(float)) == 0;
                        max_diff = std::max(max_diff, fabs((double) y[i] - yc[i]));
                    }
                    snprintf(vs_cpu, sizeof(vs_cpu), "%.4f %% bit for bit, max |diff| %.3g", 100.0 * same / n, max_diff);
                } else {
                    snprintf(vs_cpu, sizeof(vs_cpu), "(no CPU run)");
                }
                printf("  %-14s %-6s %4lld  %-22s max err %.3g, rms %.3g (%.2e of the data's), worst/bound %.3f%s  %s\n",
                        ggml_backend_name(backend), ggml_type_name(type), (long long) ne2, exact ? "exact" : "DIFFERS",
                        max_err, rms, rms / rms_x, worst, bad ? "  OUT OF BOUND" : "", vs_cpu);
            }
        }
        ggml_backend_free(backend);
    }
}

// ---- part 2: the model ----------------------------------------------------------------------------------------------

struct arm {
    std::string name;
    bool exact = false;
    std::vector<std::pair<std::string, std::string>> env;
};

static arm parse_arm(const std::string & spec) {
    arm a;
    std::istringstream is(spec);
    std::string tok;
    while (is >> tok) {
        const size_t eq = tok.find('=');
        if (a.name.empty() && eq == std::string::npos) {
            a.name = tok;
        } else if (tok == "exact") {
            a.exact = true;
        } else if (eq != std::string::npos) {
            a.env.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
        }
    }
    return a;
}

static ggml_type arm_type(const arm & a) {
    for (const auto & [k, v] : a.env) {
        if (k == "LLAMA_HOP_CAST") {
            return hop_type_of(v);
        }
    }
    return GGML_TYPE_F32;
}

// what the eval callback collects
struct probe {
    ggml_type type = GGML_TYPE_F32;
    std::vector<float> src;                 // the last hop_narrow's source
    std::vector<float> layer0;              // l_last-0 of every evaluation, in order
    int    n_narrow = 0, n_wide = 0, n_bad_hops = 0;
    size_t n_hop_values = 0;
    double worst = 0.0, max_err = 0.0;
    // per boundary (the hop_wide tensor's name): hops, values, the source's largest |x| (F16's range is 65,504) and
    // sum of squares, the error's largest |e| and sum of squares
    struct per_hop { int n = 0; size_t values = 0; double max_src = 0.0, ss_src = 0.0, max_err = 0.0, ss_err = 0.0; };
    std::map<std::string, per_hop> hops;
};

static std::vector<float> tensor_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    } else {
        v.clear();
    }
    return v;
}

static bool probe_cb(ggml_tensor * t, bool ask, void * user_data) {
    probe * p = (probe *) user_data;
    // the casts themselves, not the views of them ("hop_wide-1 (reshaped)")
    const bool plain  = strchr(t->name, ' ') == nullptr;
    const bool narrow = plain && strncmp(t->name, "hop_narrow-", 11) == 0;
    const bool wide   = plain && strncmp(t->name, "hop_wide-", 9) == 0;
    const bool layer0 = strcmp(t->name, "l_last-0") == 0;
    if (ask) {
        return narrow || wide || layer0;
    }
    if (layer0) {
        const std::vector<float> v = tensor_f32(t);
        p->layer0.insert(p->layer0.end(), v.begin(), v.end());
    }
    if (narrow) {
        p->n_narrow++;
        p->src = tensor_f32(t->src[0]); // the residual before the cast; its buffer is intact until later nodes run
    }
    if (wide) {
        p->n_wide++;
        const std::vector<float> v = tensor_f32(t);
        if (v.size() != p->src.size() || v.empty()) {
            p->n_bad_hops++;
        } else {
            double worst = 0.0;
            const size_t bad = out_of_bound(p->type, p->src.data(), v.data(), v.size(), (size_t) t->ne[0], worst);
            p->n_bad_hops += bad != 0;
            p->worst = std::max(p->worst, worst);
            probe::per_hop & h = p->hops[t->name];
            h.n++;
            h.values += v.size();
            for (size_t i = 0; i < v.size(); ++i) {
                const double e = fabs((double) v[i] - p->src[i]);
                p->max_err = std::max(p->max_err, e);
                h.max_err  = std::max(h.max_err, e);
                h.ss_err  += e*e;
                h.max_src  = std::max(h.max_src, fabs((double) p->src[i]));
                h.ss_src  += (double) p->src[i] * p->src[i];
            }
            p->n_hop_values += v.size();
        }
    }
    return true;
}

struct run_opts {
    int n_prompt = 48;
    int n_ubatch = 16;
    int n_verify = 4;
    int n_steps  = 6;
    int n_gpu_layers = 0;
    int n_threads = 4;
    std::vector<float> tensor_split; // -ts: the layer split's shares per device
};

// every position's logits for the sequence: the prompt (in micro-batches), one verify pass, single steps
static bool run_sequence(llama_model * model, const run_opts & o, probe * p, std::vector<float> & logits) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_total = o.n_prompt + o.n_verify + o.n_steps;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = std::max(o.n_prompt, 64);
    cp.n_ubatch        = o.n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = o.n_threads;
    cp.n_threads_batch = o.n_threads;
    cp.no_perf         = true;
    if (p) {
        cp.cb_eval           = probe_cb;
        cp.cb_eval_user_data = p;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot make a context\n");
        return false;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> tok(1, n_vocab - 1);
    std::vector<llama_token> tokens(n_total);
    for (auto & t : tokens) {
        t = tok(rng);
    }

    logits.clear();
    llama_batch_ext_ptr batch(llama_batch_ext_init(ctx)); // holds up to n_batch entries
    int pos = 0;
    auto eval = [&](int n) -> bool {
        llama_batch_ext_clear(batch.get());
        for (int i = 0; i < n; ++i) {
            const llama_pos p   = pos + i;
            const int32_t   idx = llama_batch_ext_add_token(batch.get(), 0, tokens[pos + i]);
            if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &p) ||
                    !llama_batch_ext_set_output_logits(batch.get(), idx, true)) {
                fprintf(stderr, "cannot add position %d to the batch (n_batch %u)\n", pos + i, llama_n_batch(ctx));
                return false;
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "llama_process failed at position %d\n", pos);
            return false;
        }
        for (int i = 0; i < n; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            logits.insert(logits.end(), l, l + n_vocab);
        }
        pos += n;
        return true;
    };
    bool ok = eval(o.n_prompt) && eval(o.n_verify);
    for (int s = 0; ok && s < o.n_steps; ++s) {
        ok = eval(1);
    }
    batch.reset();
    llama_free(ctx);
    return ok;
}

struct cmp_stats {
    double max_abs = 0.0, kld_mean = 0.0, kld_max = 0.0, top1 = 0.0;
    bool   identical = false;
};

static cmp_stats compare(const std::vector<float> & ref, const std::vector<float> & x, int n_vocab) {
    cmp_stats s;
    s.identical = ref.size() == x.size() && memcmp(ref.data(), x.data(), ref.size() * sizeof(float)) == 0;
    const size_t n_pos = ref.size() / n_vocab;
    int same = 0;
    for (size_t p = 0; p < n_pos; ++p) {
        const float * a = ref.data() + p*n_vocab;
        const float * b = x.data()   + p*n_vocab;
        double ma = -INFINITY, mb = -INFINITY;
        int ia = 0, ib = 0;
        for (int i = 0; i < n_vocab; ++i) {
            s.max_abs = std::max(s.max_abs, fabs((double) a[i] - b[i]));
            if (a[i] > ma) { ma = a[i]; ia = i; }
            if (b[i] > mb) { mb = b[i]; ib = i; }
        }
        same += ia == ib;
        double za = 0.0, zb = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            za += exp(a[i] - ma);
            zb += exp(b[i] - mb);
        }
        double kld = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            const double la = a[i] - ma - log(za);
            const double lb = b[i] - mb - log(zb);
            kld += exp(la) * (la - lb);
        }
        s.kld_mean += kld;
        s.kld_max = std::max(s.kld_max, kld);
    }
    s.kld_mean /= std::max<size_t>(n_pos, 1);
    s.top1 = n_pos ? (double) same / n_pos : 0.0;
    return s;
}

static void set_arm_env(const arm & a, const std::set<std::string> & all_vars) {
    for (const auto & v : all_vars) {
        unsetenv(v.c_str());
    }
    for (const auto & [k, v] : a.env) {
        setenv(k.c_str(), v.c_str(), 1);
    }
}

static void test_model(const char * path, const std::vector<arm> & arms, const run_opts & o, const char * tsv) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.n_gpu_layers;
    std::vector<float> ts(llama_max_devices(), 0.0f);
    if (!o.tensor_split.empty()) {
        std::copy_n(o.tensor_split.begin(), std::min(o.tensor_split.size(), ts.size()), ts.begin());
        mp.tensor_split = ts.data();
        mp.split_mode   = LLAMA_SPLIT_MODE_LAYER;
    }
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        check(false, "cannot load %s", path);
        return;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::set<std::string> all_vars;
    for (const auto & a : arms) {
        for (const auto & kv : a.env) {
            all_vars.insert(kv.first);
        }
    }

    FILE * ft = tsv ? fopen(tsv, "w") : nullptr;
    if (ft) {
        fprintf(ft, "arm\tenv\texact_expected\tidentical\tmax_abs_logit_diff\tkld_mean\tkld_max\ttop1_agree\thops\t"
                "hop_values\thops_out_of_bound\thop_max_err\thop_worst_of_bound\tlayer0_identical\n");
    }

    printf("\npart 2: %s, %d positions per arm (prompt %d in micro-batches of %d, a %d-token pass, %d single steps)\n",
            path, o.n_prompt + o.n_verify + o.n_steps, o.n_prompt, o.n_ubatch, o.n_verify, o.n_steps);
    printf("  %-12s %-9s %-12s %-12s %-12s %-8s %-6s %-12s %s\n", "arm", "identical", "max |dlogit|", "mean KLD",
            "max KLD", "top-1", "hops", "hop max err", "hop checks");

    std::vector<float> ref, ref_layer0;
    for (size_t ai = 0; ai < arms.size(); ++ai) {
        const arm & a = arms[ai];
        set_arm_env(a, all_vars);

        std::vector<float> logits;
        if (!run_sequence(model, o, nullptr, logits)) {
            check(false, "arm %s: the sequence did not run", a.name.c_str());
            continue;
        }
        if (ai == 0) {
            ref = logits;
            std::vector<float> again;
            const bool ok = run_sequence(model, o, nullptr, again);
            check(ok && again == ref, "arm %s (the reference) did not repeat bit for bit", a.name.c_str());
        }

        probe p;
        p.type = arm_type(a);
        std::vector<float> logits_cb;
        const bool ok_cb = run_sequence(model, o, &p, logits_cb);
        check(ok_cb, "arm %s: the callback pass did not run", a.name.c_str());
        if (ai == 0) {
            ref_layer0 = p.layer0;
        }
        const bool layer0_same = p.layer0 == ref_layer0 && !p.layer0.empty();

        const cmp_stats s = compare(ref, logits, n_vocab);
        const bool cast = p.type != GGML_TYPE_F32;
        if (a.exact || (!cast && ai > 0)) {
            check(s.identical, "arm %s: logits differ from the reference (max %.3g)", a.name.c_str(), s.max_abs);
        }
        check(p.n_narrow == p.n_wide, "arm %s: %d narrow casts against %d wide ones", a.name.c_str(), p.n_narrow, p.n_wide);
        if (cast) {
            // a cast arm marked exact has no boundary to cast at (LLAMA_HOP_CAST without a second GPU)
            if (a.exact) {
                check(p.n_wide == 0, "arm %s: %d hops in an arm that must not change anything", a.name.c_str(), p.n_wide);
            } else {
                check(p.n_wide > 0, "arm %s: no hop in the graph", a.name.c_str());
            }
            check(p.n_bad_hops == 0, "arm %s: %d hops outside the round-trip bound", a.name.c_str(), p.n_bad_hops);
        } else {
            check(p.n_wide == 0, "arm %s: %d hops in the graph without LLAMA_HOP_CAST", a.name.c_str(), p.n_wide);
        }
        check(layer0_same, "arm %s: layer 0's output (before any boundary) differs from the reference", a.name.c_str());

        std::string env;
        for (const auto & [k, v] : a.env) {
            env += (env.empty() ? "" : " ") + k + "=" + v;
        }
        printf("  %-12s %-9s %-12.4g %-12.4g %-12.4g %-8.4f %-6d %-12.4g %s\n", a.name.c_str(), s.identical ? "yes" : "no",
                s.max_abs, s.kld_mean, s.kld_max, s.top1, p.n_wide, p.max_err,
                !cast ? (p.n_wide == 0 ? "none, as expected" : "UNEXPECTED HOPS")
                      : (p.n_bad_hops == 0 ? "all within bound" : "OUT OF BOUND"));
        // per boundary: the source's range and the round trip's error relative to it
        for (const auto & [name, h] : p.hops) {
            const double rms_src = sqrt(h.ss_src / std::max<size_t>(h.values, 1));
            const double rms_err = sqrt(h.ss_err / std::max<size_t>(h.values, 1));
            printf("    %-16s %4d hops, source max |x| %.4g, rms %.4g; error max %.4g, rms %.4g (%.3g of the source's rms)\n",
                    name.c_str(), h.n, h.max_src, rms_src, h.max_err, rms_err, rms_src > 0 ? rms_err / rms_src : 0.0);
        }
        if (ft) {
            fprintf(ft, "%s\t%s\t%d\t%d\t%.6g\t%.6g\t%.6g\t%.6f\t%d\t%zu\t%d\t%.6g\t%.4f\t%d\n", a.name.c_str(), env.c_str(),
                    a.exact, s.identical, s.max_abs, s.kld_mean, s.kld_max, s.top1, p.n_wide, p.n_hop_values,
                    p.n_bad_hops, p.max_err, p.worst, layer0_same);
        }
    }
    if (ft) {
        fclose(ft);
    }
    set_arm_env(arm{}, all_vars);
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * tsv = nullptr;
    bool unit_only = false;
    run_opts o;
    std::vector<arm> arms;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "-m") {
            model_path = next();
        } else if (a == "-ngl") {
            o.n_gpu_layers = atoi(next());
        } else if (a == "-t") {
            o.n_threads = atoi(next());
        } else if (a == "-ts") {
            for (const char * p = next(); *p; ) {
                char * end = nullptr;
                o.tensor_split.push_back(strtof(p, &end));
                if (end == p) {
                    break;
                }
                p = *end == ',' ? end + 1 : end;
            }
        } else if (a == "--arm") {
            arms.push_back(parse_arm(next()));
        } else if (a == "--prompt") {
            o.n_prompt = atoi(next());
        } else if (a == "--ubatch") {
            o.n_ubatch = atoi(next());
        } else if (a == "--tsv") {
            tsv = next();
        } else if (a == "--unit-only") {
            unit_only = true;
        } else {
            fprintf(stderr, "usage: %s [--unit-only] [-m model.gguf] [-ngl N] [-ts a,b,...] [-t threads] [--arm \"name [exact] VAR=value ...\"]... "
                    "[--prompt N] [--ubatch N] [--tsv file]\n", argv[0]);
            return 2;
        }
    }

    llama_backend_init();
    ggml_backend_load_all();

    test_round_trips();

    if (model_path && !unit_only) {
        if (arms.empty()) {
            arms.push_back(parse_arm("off"));
            for (ggml_type t : hop_types) {
                arms.push_back(parse_arm(std::string(ggml_type_name(t)) + " LLAMA_HOP_CAST=" + ggml_type_name(t)));
            }
        }
        test_model(model_path, arms, o, tsv);
    }

    llama_backend_free();
    printf("\n%d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
