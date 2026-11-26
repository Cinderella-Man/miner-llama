// The CUDA backend's graph cache must tell a node's sources of different types apart (C1157). ggml-cuda replays the
// CUDA graph it captured for a ggml graph while every node, and each node's sources' data addresses, shapes and strides,
// are as they were in the last evaluation (ggml_cuda_graph_update_required). Two types with the same element or block
// size (F16 and BF16; Q4_0 and IQ4_NL) give a matrix the same strides, so a matrix of one type at the address of a
// matrix of the other matched, and the graph captured for the first type ran its kernels on the second type's bytes
// (C1111: test-backend-ops' BF16 cases that landed at their F16 siblings' addresses failed with an error near 0.5).
//
// The test evaluates matrix times vector on the first CUDA card until the graph is captured and replayed, then changes
// only the matrix's type (the same tensor: the same address, shape and strides; its data written anew in the new type),
// evaluates it three times, then changes it back twice more, and checks every result against the host's product of
// the values the type holds. A build without the fix fails each evaluation after a change of type that it replays;
// with GGML_CUDA_DISABLE_GRAPHS=1 every build passes.
// Usage: test-cuda-graph-src-types   (exit status 0 when every check passes, 1 when one fails, 2 without a CUDA card)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr int64_t K = 512; // a whole number of every type's blocks, and no row padding on the card
static constexpr int64_t M = 128;
static constexpr double  MAX_NMSE = 5e-4; // test-backend-ops' limit for MUL_MAT

// draws K*M values in [-1, 1], writes them to `a` in `type`, and returns the values the type holds
static std::vector<float> fill(ggml_tensor * a, ggml_type type, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(K * M);
    for (float & x : v) {
        x = u(rng);
    }
    std::vector<uint8_t> raw(ggml_row_size(type, K) * M);
    ggml_quantize_chunk(type, v.data(), raw.data(), 0, M, K, nullptr);
    ggml_backend_tensor_set(a, raw.data(), 0, raw.size());
    std::vector<float> held(K * M);
    ggml_get_type_traits(type)->to_float(raw.data(), held.data(), K * M);
    return held;
}

// one pair of types with the same strides: x three times, y three times, x twice; true when every result is right
static bool run_pair(ggml_backend_t backend, ggml_type x, ggml_type y, int & n_eval, int & n_bad) {
    if (ggml_type_size(x) != ggml_type_size(y) || ggml_blck_size(x) != ggml_blck_size(y)) {
        fprintf(stderr, "%s and %s have different strides\n", ggml_type_name(x), ggml_type_name(y));
        return false;
    }
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, x, K, M);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "no buffer\n");
        ggml_free(ctx);
        return false;
    }

    std::mt19937 rng(1157);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> vb(K);
    for (float & v : vb) {
        v = u(rng);
    }
    ggml_backend_tensor_set(b, vb.data(), 0, K * sizeof(float));

    const ggml_type order[] = { x, x, x, y, y, y, x, x };
    std::vector<float> held;
    std::vector<float> res(M);
    bool ok = true;
    for (size_t s = 0; s < sizeof(order) / sizeof(order[0]); ++s) {
        if (s == 0 || order[s] != order[s - 1]) {
            a->type = order[s]; // the same tensor, address, shape and strides; only the type changes
            held = fill(a, order[s], rng);
        }
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(out, res.data(), 0, M * sizeof(float));
        double err = 0.0, ref2 = 0.0;
        for (int64_t m = 0; m < M; ++m) {
            double r = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                r += (double) held[m * K + k] * (double) vb[k];
            }
            err  += (res[m] - r) * (res[m] - r);
            ref2 += r * r;
        }
        const double nmse = err / ref2;
        const bool good = std::isfinite(nmse) && nmse <= MAX_NMSE;
        printf("  %s, evaluation %zu (%s%s): NMSE %.3g %s\n", ggml_type_name(x), s + 1, ggml_type_name(order[s]),
               s > 0 && order[s] != order[s - 1] ? ", type changed" : "", nmse, good ? "OK" : "FAIL");
        n_eval++;
        if (!good) {
            n_bad++;
            ok = false;
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr || strstr(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") == nullptr) {
        fprintf(stderr, "no CUDA device\n");
        return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "the device did not start\n");
        return 2;
    }
    const char * no_graphs = getenv("GGML_CUDA_DISABLE_GRAPHS");
    printf("test-cuda-graph-src-types on %s, CUDA graphs %s\n", ggml_backend_dev_description(dev),
           no_graphs ? "off (GGML_CUDA_DISABLE_GRAPHS)" : "on");

    const ggml_type pairs[][2] = {
        { GGML_TYPE_F16,  GGML_TYPE_BF16   },
        { GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL },
    };
    int n_eval = 0, n_bad = 0;
    for (const auto & p : pairs) {
        run_pair(backend, p[0], p[1], n_eval, n_bad);
        run_pair(backend, p[1], p[0], n_eval, n_bad);
    }
    ggml_backend_free(backend);
    printf("%d of %d evaluations right: %s\n", n_eval - n_bad, n_eval, n_bad == 0 ? "PASS" : "FAIL");
    return n_bad == 0 ? 0 : 1;
}
