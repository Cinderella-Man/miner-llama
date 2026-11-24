// The order of ggml_top_k's columns on each backend device, and whether it changes from call to call (C985, Q1718).
//
// ggml_top_k leaves the order of a row's k columns unspecified. On CUDA, a row wider than the bitonic and argsort
// paths take (more than 4,096 columns for one row, or several rows) runs the radix select, whose gather writes the
// columns at the slots an atomicAdd hands out, so the order changes from call to call. A backend sampling chain
// top-k -> temperature -> dist sums the candidates' probabilities in that order, so one seed can pick other tokens.
// GGML_CUDA_TOP_K_SORT=1 sorts each row: descending value, equal values by the lower column.
//
// For every case and device this runs one ggml_top_k graph many times on a fixed input and reports, per case:
//   calls, the distinct outputs (orders) seen, the calls whose output is the reference order (descending value, equal
//   values by the lower column), the calls with a row whose values are not the reference's k largest (an error; where
//   the k-th value is shared, a backend may keep other columns of it), and, for a case with a draw, the tokens the dist
//   step picks with the candidates in each call's order.
//
// Cases:
//   - every input file given, in the format of C962's probe (docs/data/2026-11-23_c962_sampled_reply_gpu/run/probe/):
//     a line "n_vocab k seed pos temp", then k lines "token_id probability". The row: every other column a fixed
//     pseudo-random value in [-4, 4), each candidate 30 + temp * ln(p), so the top-k picks those k columns and the
//     dist step's probabilities are p; its draw u is the fork's position-keyed draw (src/llama-sampler.cpp,
//     llama_sampler_pos_seed and the first uniform_real_distribution<double> draw of std::mt19937). Each file runs as
//     one row and as four copies of the row (the several output rows of LLAMA_SAMPLING_TOP_K_ROWS);
//   - "ties": a row of 150,144 columns whose top 8 hold three pairs of equal values, and "ties-edge": a row whose 8th
//     value is shared by five columns (the select keeps the lowest columns), each as one row and four rows.
//
// Usage: test-top-k-sort [-n calls] [-d device] [-r] [--expect-sorted] [input files...]
//   -n calls         calls per case and device (default 2000)
//   -d device        only devices whose name starts with this (e.g. CUDA0); default every device
//   -r               write the input again before every call
//   --expect-sorted  exit 1 if a call on a non-CPU device does not give the reference order
// Exit code: 1 if any set is wrong (or, with --expect-sorted, any order), 0 otherwise.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

struct top_k_case {
    std::string        name;
    int64_t            ncols = 0;
    int64_t            nrows = 1;
    int                k     = 0;
    std::vector<float> row;           // one row, repeated nrows times
    bool               has_draw = false;
    float              temp     = 1.0f;
    float              u        = 0.0f;
};

// src/llama-sampler.cpp: llama_sampler_pos_seed and the first draw after llama_sampler_set_pos
static uint32_t pos_seed(uint32_t seed, int32_t pos) {
    uint64_t x = ((uint64_t) seed << 32) ^ (uint64_t) (uint32_t) pos;
    x += 0x9e3779b97f4a7c15ULL;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^=  x >> 31;
    return (uint32_t) (x ^ (x >> 32));
}

static float pos_uniform(uint32_t seed, int32_t pos) {
    std::mt19937 rng(pos_seed(seed, pos));
    std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    return (float) dist(rng);
}

// the backend chain's dist step on the candidates in the given order (src/llama-sampler.cpp,
// llama_sampler_dist_backend_apply): scale by 1/temp, softmax, running sum, index = n - count(sum > u), the count
// clamped to [1, n]
static int dist_pick(const std::vector<float> & logits, float temp, float u) {
    const int n = (int) logits.size();
    std::vector<float> x(n);
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) {
        x[i] = logits[i] * (1.0f / temp);
        mx   = std::max(mx, x[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    float cum   = 0.0f;
    int   above = 0;
    for (int i = 0; i < n; i++) {
        cum   += x[i] / sum;
        above += (cum - u) > 0.0f;
    }
    above = std::min(std::max(above, 1), n);
    return n - above;
}

// the fixed background of every row: pseudo-random values in [-4, 4)
static std::vector<float> background(int64_t ncols) {
    std::vector<float> row(ncols);
    uint32_t s = 12345u;
    for (int64_t i = 0; i < ncols; i++) {
        s      = s * 1664525u + 1013904223u;
        row[i] = -4.0f + 8.0f * ((s >> 8) * (1.0f / 16777216.0f));
    }
    return row;
}

static bool read_case(const char * path, top_k_case & c) {
    FILE * f = fopen(path, "r");
    if (!f) {
        perror(path);
        return false;
    }
    int n_vocab = 0, k = 0, pos = 0;
    unsigned seed = 0;
    float temp = 1.0f;
    if (fscanf(f, "%d %d %u %d %f", &n_vocab, &k, &seed, &pos, &temp) != 5 || n_vocab <= 0 || k <= 0 || k > n_vocab) {
        fprintf(stderr, "%s: bad header\n", path);
        fclose(f);
        return false;
    }
    c.ncols = n_vocab;
    c.k     = k;
    c.row   = background(n_vocab);
    for (int i = 0; i < k; i++) {
        int id = 0;
        double p = 0.0;
        if (fscanf(f, "%d %lf", &id, &p) != 2 || id < 0 || id >= n_vocab || p <= 0.0) {
            fprintf(stderr, "%s: bad candidate line %d\n", path, i);
            fclose(f);
            return false;
        }
        c.row[id] = 30.0f + temp * (float) log(p);
    }
    fclose(f);
    std::string name = path;
    const size_t slash = name.find_last_of('/');
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    if (name.size() > 3 && name.compare(name.size() - 3, 3, ".in") == 0) {
        name.resize(name.size() - 3);
    }
    c.name     = name;
    c.has_draw = true;
    c.temp     = temp;
    c.u        = pos_uniform(seed, pos);
    return true;
}

// the reference: the k columns of the largest values, in descending order of value, equal values by the lower column
static std::vector<int32_t> reference(const std::vector<float> & row, int k) {
    std::vector<int32_t> idx(row.size());
    for (size_t i = 0; i < row.size(); i++) {
        idx[i] = (int32_t) i;
    }
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
        return row[a] > row[b] || (row[a] == row[b] && a < b);
    });
    idx.resize(k);
    return idx;
}

struct case_result {
    int64_t calls        = 0;
    int64_t sorted_calls = 0;   // calls whose every row is the reference order
    int64_t set_errors   = 0;   // calls with a row whose values are not the reference's
    std::map<std::vector<int32_t>, int64_t> orders;   // the whole output (every row) -> calls
    std::map<int32_t, int64_t> picks;                 // token picked in row 0 -> calls
};

static bool run_case(ggml_backend_dev_t dev, const top_k_case & c, int n_calls, bool rewrite, case_result & res) {
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ncols, c.nrows);
    ggml_tensor * y = ggml_top_k(ctx, x, c.k);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "%s: could not start the backend\n", ggml_backend_dev_name(dev));
        ggml_free(ctx);
        return false;
    }
    if (!ggml_backend_supports_op(backend, y)) {
        fprintf(stderr, "%s: TOP_K of %lld x %lld not supported, skipped\n", ggml_backend_dev_name(dev), (long long) c.ncols,
                (long long) c.nrows);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> input((size_t) c.ncols * c.nrows);
    for (int64_t r = 0; r < c.nrows; r++) {
        std::copy(c.row.begin(), c.row.end(), input.begin() + r * c.ncols);
    }
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));

    const std::vector<int32_t> ref = reference(c.row, c.k);
    std::vector<float> ref_vals;
    for (const int32_t col : ref) {
        ref_vals.push_back(c.row[col]);
    }

    std::vector<int32_t> out((size_t) c.k * c.nrows);
    for (int it = 0; it < n_calls; it++) {
        if (rewrite) {
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", ggml_backend_dev_name(dev));
            break;
        }
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        res.calls++;
        bool sorted = true;
        bool set_ok = true;
        for (int64_t r = 0; r < c.nrows; r++) {
            const std::vector<int32_t> got(out.begin() + r * c.k, out.begin() + (r + 1) * c.k);
            sorted = sorted && got == ref;
            // the values of the row's columns, distinct columns in range, against the reference's k largest values
            std::vector<int32_t> cols(got);
            std::sort(cols.begin(), cols.end());
            const bool cols_ok = std::adjacent_find(cols.begin(), cols.end()) == cols.end() && cols.front() >= 0 &&
                                 cols.back() < c.ncols;
            if (!cols_ok) {
                set_ok = false;
                continue;
            }
            std::vector<float> vals;
            for (const int32_t col : got) {
                vals.push_back(c.row[col]);
            }
            std::sort(vals.begin(), vals.end(), std::greater<float>());
            set_ok = set_ok && vals == ref_vals;
        }
        res.sorted_calls += sorted;
        res.set_errors   += !set_ok;
        res.orders[out]++;
        if (c.has_draw && set_ok) {
            std::vector<float> lg(c.k);
            for (int i = 0; i < c.k; i++) {
                lg[i] = c.row[out[i]];
            }
            res.picks[out[dist_pick(lg, c.temp, c.u)]]++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static std::string order_str(const std::vector<int32_t> & v, int k) {
    std::string s;
    for (int i = 0; i < k && i < (int) v.size(); i++) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s;
}

int main(int argc, char ** argv) {
    int n_calls = 2000;
    std::string dev_prefix;
    bool rewrite = false;
    bool expect_sorted = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) {
            n_calls = atoi(argv[++i]);
        } else if (a == "-d" && i + 1 < argc) {
            dev_prefix = argv[++i];
        } else if (a == "-r") {
            rewrite = true;
        } else if (a == "--expect-sorted") {
            expect_sorted = true;
        } else if (a[0] == '-') {
            fprintf(stderr, "usage: %s [-n calls] [-d device] [-r] [--expect-sorted] [input files...]\n", argv[0]);
            return a == "-h" || a == "--help" ? 0 : 2;
        } else {
            files.push_back(a);
        }
    }

    ggml_backend_load_all();

    std::vector<top_k_case> cases;
    for (const auto & f : files) {
        top_k_case c;
        if (!read_case(f.c_str(), c)) {
            return 2;
        }
        cases.push_back(c);
    }
    {
        // three pairs of equal values in the top 8, the pairs' columns far apart (so in different blocks of the select)
        top_k_case c;
        c.name  = "ties";
        c.ncols = 150144;
        c.k     = 8;
        c.row   = background(c.ncols);
        const int   cols[8] = { 140001, 7, 90210, 65618, 14, 120748, 11844, 24160 };
        const float vals[8] = { 31.0f, 31.0f, 30.0f, 30.0f, 29.5f, 29.0f, 29.0f, 28.0f };
        for (int i = 0; i < 8; i++) {
            c.row[cols[i]] = vals[i];
        }
        cases.push_back(c);
        // the 8th value shared by five columns: the select keeps the lowest two of them
        top_k_case e = c;
        e.name = "ties-edge";
        e.row  = background(e.ncols);
        const int   ecols[11] = { 100, 50000, 149000, 3, 77777, 120000, 9, 60000, 130000, 1000, 80000 };
        const float evals[11] = { 33.0f, 32.0f, 31.0f, 30.0f, 29.0f, 28.0f, 27.0f, 27.0f, 27.0f, 27.0f, 27.0f };
        for (int i = 0; i < 11; i++) {
            e.row[ecols[i]] = evals[i];
        }
        cases.push_back(e);
    }
    // each case as one row and as four rows
    std::vector<top_k_case> all;
    for (const auto & c : cases) {
        all.push_back(c);
        top_k_case m = c;
        m.nrows = 4;
        m.name += "-x4";
        all.push_back(m);
    }

    const char * sw = getenv("GGML_CUDA_TOP_K_SORT");
    printf("GGML_CUDA_TOP_K_SORT=%s, %d calls per case and device%s\n", sw ? sw : "(unset)", n_calls,
           rewrite ? ", the input written before every call" : "");
    printf("case\tdevice\tcols\trows\tk\tcalls\torders\treference_order_calls\tset_errors\tu\tpicks\n");

    int rc = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); d++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        const std::string dev_name = ggml_backend_dev_name(dev);
        if (!dev_prefix.empty() && dev_name.compare(0, dev_prefix.size(), dev_prefix) != 0) {
            continue;
        }
        const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        for (const auto & c : all) {
            case_result res;
            if (!run_case(dev, c, n_calls, rewrite, res)) {
                continue;
            }
            std::string picks;
            for (const auto & [tok, n] : res.picks) {
                picks += (picks.empty() ? "" : " ") + std::to_string(tok) + ":" + std::to_string(n);
            }
            char u[32];
            snprintf(u, sizeof(u), "%.4f", c.u);
            printf("%s\t%s\t%lld\t%lld\t%d\t%lld\t%zu\t%lld\t%lld\t%s\t%s\n", c.name.c_str(), dev_name.c_str(),
                   (long long) c.ncols, (long long) c.nrows, c.k, (long long) res.calls, res.orders.size(),
                   (long long) res.sorted_calls, (long long) res.set_errors, c.has_draw ? u : "-",
                   picks.empty() ? "-" : picks.c_str());
            // the most common row-0 orders, for the log
            std::vector<std::pair<int64_t, std::string>> by_count;
            for (const auto & [o, n] : res.orders) {
                by_count.push_back({ n, order_str(o, c.k) });
            }
            std::sort(by_count.rbegin(), by_count.rend());
            for (size_t i = 0; i < by_count.size() && i < 4; i++) {
                printf("#\t%s\t%s\torder\t%lld\t%s\n", c.name.c_str(), dev_name.c_str(), (long long) by_count[i].first,
                       by_count[i].second.c_str());
            }
            printf("#\t%s\t%s\treference\t%s\n", c.name.c_str(), dev_name.c_str(), order_str(reference(c.row, c.k), c.k).c_str());
            if (res.set_errors > 0) {
                rc = 1;
            }
            if (expect_sorted && !is_cpu && res.sorted_calls != res.calls) {
                rc = 1;
            }
        }
    }
    printf("result: %s\n", rc == 0 ? "pass" : "FAIL");
    return rc;
}
