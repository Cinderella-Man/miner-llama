// A graph cached on an RPC server, reused by its uid after a buffer on that server was freed.
//
// The server drops every graph it caches when a buffer is freed (rpc_server::free_buffer), but keeps their entries. The
// client must then send such a graph whole again: a GRAPH_RECOMPUTE for it is rejected, the server closes the
// connection and the client aborts at its next command. The client must also keep counting the dropped entries, as the
// server does, so that both sides still clear their caches at the same point (GRAPH_CACHE_MAX in ggml-rpc.cpp).
//
// Usage: test-rpc-graph-cache <endpoint> [<unused endpoint>] (tests/test-rpc-multi-server.sh starts the servers)

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

// GRAPH_CACHE_MAX in ggml-rpc.cpp: the graphs cached per device before both sides clear their caches
static const uint64_t GRAPH_CACHE_MAX = 1024;

static const int N = 16;

// out = 2*a; a[i] = v + i, computed with the graph's uid set to uid
static void compute(ggml_backend_t backend, ggml_cgraph * graph, ggml_tensor * a, uint64_t uid, float v) {
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = v + i;
    }
    ggml_backend_tensor_set(a, data.data(), 0, ggml_nbytes(a));
    graph->uid = uid;
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
}

static int check(ggml_backend_t backend, ggml_cgraph * graph, ggml_tensor * a, ggml_tensor * out, uint64_t uid, float v,
        const char * what) {
    compute(backend, graph, a, uid, v);
    std::vector<float> got(N);
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
    int bad = 0;
    for (int i = 0; i < N; i++) {
        bad += got[i] != 2.0f*(v + i) ? 1 : 0;
    }
    printf("%-60s uid %4llu: %s\n", what, (unsigned long long) uid, bad == 0 ? "OK" : "WRONG");
    return bad == 0 ? 0 : 1;
}

// allocates a buffer on the server and frees it again: the server drops every graph it caches
static void free_a_buffer(ggml_backend_t backend) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc >= 2);
    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_rpc_init(argv[1], 0);
    GGML_ASSERT(backend != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 4*ggml_tensor_overhead() + ggml_graph_overhead_custom(4, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_tensor * out = ggml_scale(ctx, a, 2.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    int failed = 0;
    failed += check(backend, graph, a, out, 1, 1.0f, "first compute (the server caches it)");
    failed += check(backend, graph, a, out, 1, 2.0f, "reused (recomputed by uid)");

    free_a_buffer(backend);
    failed += check(backend, graph, a, out, 1, 3.0f, "reused after another buffer was freed");
    failed += check(backend, graph, a, out, 1, 4.0f, "reused again");

    // both caches at 1,000 entries, then all of them dropped by a free; the next graphs fill the server's cache to
    // GRAPH_CACHE_MAX and it clears, so a graph sent before that clear must be sent whole again
    for (uint64_t uid = 2; uid <= 1000; uid++) {
        compute(backend, graph, a, uid, 0.0f);
    }
    free_a_buffer(backend);
    const uint64_t first = 2001;
    const uint64_t n_new = GRAPH_CACHE_MAX - 1000 + 6; // the server clears its cache at the 25th of them
    for (uint64_t uid = first; uid < first + n_new; uid++) {
        compute(backend, graph, a, uid, 0.0f);
    }
    failed += check(backend, graph, a, out, first + n_new - 7, 5.0f, "sent before the server cleared its cache");
    failed += check(backend, graph, a, out, first + n_new - 1, 6.0f, "sent after the server cleared its cache");
    failed += check(backend, graph, a, out, 1, 7.0f, "the first graph, dropped and cleared since");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("%s\n", failed == 0 ? "all checks OK" : "FAILED");
    return failed == 0 ? 0 : 1;
}
