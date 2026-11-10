// GGML_CUDA_GRAPHS_FORCE's parts that run without a GPU (ggml/src/ggml-cuda/graph-force.h): which graphs the switch
// takes, and the decision for one evaluation of such a graph, the latter through a model of the whole of
// ggml_backend_cuda_graph_compute's warm-up and capture logic, its stock branches included, over long random runs of
// evaluations: graphs the switch takes and graphs it leaves sharing keys and a context, graphs CUDA cannot capture,
// properties that repeat, and sample mode's phase sets. An executable is an int, the properties it was captured from;
// a replay must only ever launch executables of the current properties. A control runs the port's first decision
// (692bbbc9a) in the same model and must be caught replaying stale executables.
// Usage: test-cuda-graph-force   (exit status 0 when every check passes)

#include "ggml.h"
#include "../ggml/src/ggml-cuda/graph-force.h"
#include "../ggml/src/ggml-cuda/profile-light.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static constexpr int64_t MAX_ROWS = 8; // MMVQ_MAX_BATCH_SIZE

// ---- which graphs the switch takes ---------------------------------------------------------------------------------

struct graph_builder {
    ggml_context * ctx;

    graph_builder() {
        ggml_init_params params = { 64*1024*1024, nullptr, true };
        ctx = ggml_init(params);
    }
    ~graph_builder() { ggml_free(ctx); }

    ggml_tensor * t(int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1, ggml_type type = GGML_TYPE_F32) {
        return ggml_new_tensor_4d(ctx, type, a, b, c, d);
    }

    // a layer of n tokens: a weight product, RoPE'd queries, flash attention and a MoE product
    std::vector<ggml_tensor *> layer(int64_t n, bool rope = true, bool fa = true, bool moe = true, bool mm = true) {
        std::vector<ggml_tensor *> nodes;
        const int64_t n_embd = 64, d = 16, n_head = 4, n_kv = 256, n_expert = 8, n_used = 2;
        ggml_tensor * x = t(n_embd, n);
        if (mm) {
            nodes.push_back(ggml_mul_mat(ctx, t(n_embd, n_embd), x));
        }
        if (rope) {
            ggml_tensor * q = t(d, n_head, n);
            nodes.push_back(ggml_rope(ctx, q, t(n, 1, 1, 1, GGML_TYPE_I32), d, 0));
        }
        if (fa) {
            ggml_tensor * q = t(d, n, n_head);
            ggml_tensor * k = t(d, n_kv, n_head, 1, GGML_TYPE_F16);
            ggml_tensor * v = t(d, n_kv, n_head, 1, GGML_TYPE_F16);
            nodes.push_back(ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f));
        }
        if (moe) {
            ggml_tensor * ids = t(n_used, n, 1, 1, GGML_TYPE_I32);
            nodes.push_back(ggml_mul_mat_id(ctx, t(n_embd, n_embd, n_expert), t(n_embd, 1, n), ids));
        }
        return nodes;
    }
};

static bool eligible(const std::vector<ggml_tensor *> & nodes) {
    return ggml_cuda_graph_force_eligible(nodes.data(), (int) nodes.size(), MAX_ROWS);
}

static void test_eligible() {
    graph_builder g;

    // by tokens: decode (1), a verify block (4), the vector kernels' limit (8) and over it, a catch-up pass, a prompt
    for (int64_t n : { 1, 4, 8 }) {
        CHECK(!eligible(g.layer(n)));
    }
    for (int64_t n : { 9, 64, 128, 512 }) {
        CHECK(eligible(g.layer(n)));
    }
    // each signal alone
    CHECK(eligible(g.layer(512, true,  false, false, false)));
    CHECK(eligible(g.layer(512, false, true,  false, false)));
    CHECK(eligible(g.layer(512, false, false, true,  false)));
    CHECK(!eligible(g.layer(1, true,  false, false, false)));
    CHECK(!eligible(g.layer(1, false, true,  false, false)));
    CHECK(!eligible(g.layer(1, false, false, true,  false)));
    // a plain product never counts, whatever its rows: a prompt's weight product, MLA's absorbed queries of one token
    // ([d, n_head, 1] is a 2-D [d, n_head] tensor)
    CHECK(!eligible(g.layer(512, false, false, false, true)));
    {
        ggml_tensor * q_nope = g.t(16, 128, 1);
        std::vector<ggml_tensor *> nodes = { ggml_mul_mat(g.ctx, g.t(16, 32), q_nope) };
        CHECK(!eligible(nodes));
    }
    // a MUL_MAT_ID of one row per token in its second dimension (a candidates' product over one row) is one token
    {
        ggml_tensor * ids = g.t(32, 1, 1, 1, GGML_TYPE_I32);
        std::vector<ggml_tensor *> nodes = { ggml_mul_mat_id(g.ctx, g.t(64, 1, 4096), g.t(64, 1, 1), ids) };
        CHECK(!eligible(nodes));
    }
    // no products, no tokens to read: left to the warm-up
    CHECK(!eligible({}));
    CHECK(!eligible({ ggml_add(g.ctx, g.t(64, 512), g.t(64, 512)) }));

    // ARGSORT: CUB's segmented path (over 1,024 columns and several rows) keeps the whole graph out
    {
        std::vector<ggml_tensor *> nodes = g.layer(512);
        nodes.push_back(ggml_argsort(g.ctx, g.t(151936, 2), GGML_SORT_ORDER_DESC));
        CHECK(!eligible(nodes));
    }
    // one row (DeviceRadixSort, eager or captured) or up to 1,024 columns (bitonic, in shared memory) do not
    const std::vector<std::pair<int64_t, int64_t>> shapes = { { 151936, 1 }, { 1024, 512 }, { 8, 512 } };
    for (const auto & shape : shapes) {
        std::vector<ggml_tensor *> nodes = g.layer(512);
        nodes.push_back(ggml_argsort(g.ctx, g.t(shape.first, shape.second), GGML_SORT_ORDER_DESC));
        CHECK(eligible(nodes));
    }
    // over 1,024 columns and several rows in another dimension too
    {
        std::vector<ggml_tensor *> nodes = g.layer(512);
        nodes.push_back(ggml_argsort(g.ctx, g.t(2048, 1, 3), GGML_SORT_ORDER_ASC));
        CHECK(!eligible(nodes));
    }
}

// ---- the decision, in a model of ggml_backend_cuda_graph_compute ---------------------------------------------------

// a key's ggml_cuda_graph: its node properties as last recorded, its executables, its warm-up state, its phase sets
struct fake_graph {
    int                                            props = -1; // node_props: -1 before the first record
    std::vector<int>                               instances;
    ggml_cuda_profile_phase_sets<std::vector<int>> phases;
    uint64_t                                       capture_gen     = 0;
    bool                                           warmup_complete = false;
};

// a context's: its keys and the switch's first eager batched graph
struct fake_context {
    std::vector<fake_graph> keys;
    bool                    graphs_force_warm = false;
};

enum { HOW_EAGER, HOW_CAPTURE, HOW_REPLAY };

enum rule { RULE_OFF, RULE_FORCE, RULE_PORT_CONTROL };

// One evaluation of graph `props` on key `key`, as ggml_backend_cuda_graph_compute decides it. forced: the switch
// takes it (a function of its properties); compatible: ggml_cuda_graph_check_compability (no MUL_MAT_ID that needs a
// synchronization), whose failure skips the record of its properties.
static int fake_eval(fake_context & c, int key, int props, bool forced, bool compatible, int phase, int n_phases, int n_chunks,
                     rule r, bool & replay_ok) {
    fake_graph & g = c.keys[key];
    if (n_phases > 0) {
        g.phases.select(g.instances, phase, n_phases);
    }
    bool use = false, update = false;
    if (compatible) {
        const bool changed = props != g.props;
        g.props = props;
        const auto stale = [&g](uint64_t gen) { return g.phases.stale(gen); };
        if (r == RULE_FORCE && forced) {
            const ggml_cuda_graph_force_plan plan = ggml_cuda_graph_force_next(c.graphs_force_warm, g.warmup_complete,
                g.capture_gen, changed, g.instances.empty(), stale);
            use    = plan.use_graph;
            update = plan.capture;
        } else if (r == RULE_PORT_CONTROL && forced && !c.graphs_force_warm) {
            // the control: the port's decision below, behind this branch's eager first batched graph
            c.graphs_force_warm = true;
        } else if (r == RULE_PORT_CONTROL && forced) {
            // 692bbbc9a's forced branch
            if (!g.warmup_complete || changed) {
                g.warmup_complete = true;
                g.capture_gen++;
            }
            use    = true;
            update = changed || g.instances.empty() || g.phases.stale(g.capture_gen);
        } else if (!g.warmup_complete) {
            // the stock branches, as in ggml-cuda.cu
            if (!changed) {
                g.warmup_complete = true;
                g.capture_gen++;
                use = update = true;
            }
        } else if (changed) {
            g.warmup_complete = false;
        } else {
            use    = true;
            update = g.instances.empty() || g.phases.stale(g.capture_gen);
        }
    }
    if (use && update) {
        g.instances.assign(n_chunks, props);
        g.phases.captured(g.capture_gen);
        return HOW_CAPTURE;
    }
    if (use) {
        for (int x : g.instances) {
            replay_ok = replay_ok && x == props;
        }
        replay_ok = replay_ok && !g.instances.empty();
        return HOW_REPLAY;
    }
    return HOW_EAGER;
}

// properties 0..n_props-1; the even ones are batched (the switch takes them), as a function of the properties
static bool props_forced(int props) { return props % 2 == 0; }

struct run_counts {
    long eager = 0, capture = 0, replay = 0;
    long forced_eager = 0;     // evaluations of graphs the switch takes that ran eagerly
    long forced_evals = 0;
    bool replay_ok    = true;
};

// a random run: a few keys sharing a context; each step picks a key and either repeats its last properties (a reused
// graph, the next decode step) or takes others; now and then a graph CUDA cannot capture. Every key starts with a
// decode graph (props 1), as a server's warm-up decode comes before its first prompt
static run_counts random_run(std::mt19937 & rng, rule r, int n_keys, int n_props, int n_steps, int n_phases) {
    fake_context c;
    c.keys.resize(n_keys);
    std::vector<int> last(n_keys, 1);
    run_counts k;
    for (int e = 0; e < n_steps; ++e) {
        const int key = (int) (rng() % n_keys);
        if (rng() % 3 == 0) {
            last[key] = (int) (rng() % n_props);
        }
        const int  props      = last[key];
        const bool compatible = rng() % 17 != 0;
        const bool forced     = props_forced(props);
        const int  chunks     = 1 + props % 3;
        const int  how        = fake_eval(c, key, props, forced, compatible, e % (n_phases > 0 ? n_phases : 1), n_phases,
                                          chunks, r, k.replay_ok);
        k.eager   += how == HOW_EAGER;
        k.capture += how == HOW_CAPTURE;
        k.replay  += how == HOW_REPLAY;
        if (forced && compatible) {
            k.forced_evals++;
            k.forced_eager += how == HOW_EAGER;
        }
    }
    return k;
}

static void test_decisions() {
    // a prompt under pipeline parallelism: each micro-batch's graph changes; the switch captures every one after the
    // context's first, where the stock logic launches each eagerly
    for (rule r : { RULE_OFF, RULE_FORCE }) {
        fake_context c;
        c.keys.resize(1);
        bool ok = true;
        std::vector<int> hows;
        for (int mb = 0; mb < 8; ++mb) {
            hows.push_back(fake_eval(c, 0, 2*mb, true, true, 0, 0, 1, r, ok));
        }
        CHECK(ok);
        for (int mb = 0; mb < 8; ++mb) {
            CHECK(hows[mb] == (r == RULE_OFF || mb == 0 ? HOW_EAGER : HOW_CAPTURE));
        }
    }
    // the same graph again (a reused graph): replayed
    {
        fake_context c;
        c.keys.resize(1);
        bool ok = true;
        CHECK(fake_eval(c, 0, 0, true, true, 0, 0, 1, RULE_FORCE, ok) == HOW_EAGER);
        CHECK(fake_eval(c, 0, 2, true, true, 0, 0, 1, RULE_FORCE, ok) == HOW_CAPTURE);
        CHECK(fake_eval(c, 0, 2, true, true, 0, 0, 1, RULE_FORCE, ok) == HOW_REPLAY);
        CHECK(ok);
    }
    // the context's first batched graph runs eagerly, on whichever key; later ones on any key are captured
    {
        fake_context c;
        c.keys.resize(2);
        bool ok = true;
        CHECK(fake_eval(c, 0, 0, true, true, 0, 0, 1, RULE_FORCE, ok) == HOW_EAGER);
        CHECK(fake_eval(c, 1, 2, true, true, 0, 0, 1, RULE_FORCE, ok) == HOW_CAPTURE);
        CHECK(ok);
    }
    // the case the port's decision gets wrong once some batched graph runs eagerly: two decode steps capture a key's
    // instances from props 1, the context's first batched graph records props 2 eagerly on the same key, then props 2
    // again (a reused graph). The switch captures props 2; the port's decision replays props 1's executables
    for (rule r : { RULE_FORCE, RULE_PORT_CONTROL }) {
        fake_context c;
        c.keys.resize(1);
        bool ok = true;
        CHECK(fake_eval(c, 0, 1, false, true, 0, 0, 1, r, ok) == HOW_EAGER);   // a decode graph, warming up
        CHECK(fake_eval(c, 0, 1, false, true, 0, 0, 1, r, ok) == HOW_CAPTURE); // captured from props 1
        CHECK(fake_eval(c, 0, 2, true, true, 0, 0, 1, r, ok) == HOW_EAGER);    // the first batched graph: eager
        const int how = fake_eval(c, 0, 2, true, true, 0, 0, 1, r, ok);
        CHECK(how == (r == RULE_FORCE ? HOW_CAPTURE : HOW_REPLAY));
        CHECK(ok == (r == RULE_FORCE));
    }
    // decode graphs keep the stock warm-up with the switch on: the same decisions as with it off
    {
        fake_context c_off, c_on;
        c_off.keys.resize(1);
        c_on.keys.resize(1);
        bool ok = true;
        for (int props : { 1, 1, 1, 3, 3, 3, 3, 1, 1 }) {
            CHECK(fake_eval(c_off, 0, props, false, true, 0, 0, 1, RULE_OFF, ok) ==
                  fake_eval(c_on,  0, props, false, true, 0, 0, 1, RULE_FORCE, ok));
        }
        CHECK(ok);
    }

    // long random runs: no replay of stale executables, with and without sample mode's phase sets; graphs the switch
    // takes are eager only once per context; and the control is caught
    std::mt19937 rng(702);
    long replays = 0, captures = 0, forced_eager_max = 0, port_stale_runs = 0;
    for (int t = 0; t < 2000; ++t) {
        const int n_keys   = 1 + (int) (rng() % 3);
        const int n_props  = 2 + (int) (rng() % 5);
        const int n_phases = rng() % 4 == 0 ? 1 + (int) (rng() % 4) : 0;
        const unsigned seed = (unsigned) rng();

        std::mt19937 r_on(seed);
        const run_counts on = random_run(r_on, RULE_FORCE, n_keys, n_props, 400, n_phases);
        CHECK(on.replay_ok);
        CHECK(on.forced_eager <= 1);
        replays          += on.replay;
        captures         += on.capture;
        forced_eager_max  = std::max(forced_eager_max, on.forced_eager);

        std::mt19937 r_off(seed);
        const run_counts off = random_run(r_off, RULE_OFF, n_keys, n_props, 400, n_phases);
        CHECK(off.replay_ok);

        std::mt19937 r_port(seed);
        const run_counts port = random_run(r_port, RULE_PORT_CONTROL, n_keys, n_props, 400, n_phases);
        port_stale_runs += !port.replay_ok;
    }
    CHECK(replays > 0 && captures > 0);
    CHECK(port_stale_runs > 0);
    printf("random runs: 2000 x 400 evaluations; switch on: %ld captures, %ld replays, none stale, at most %ld eager "
           "batched graph a run; the control replayed stale executables in %ld of 2000 runs\n",
           captures, replays, forced_eager_max, port_stale_runs);
}

int main() {
    test_eligible();
    test_decisions();
    printf("test-cuda-graph-force: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
