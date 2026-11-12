// GGML_CUDA_GRAPHS_PATCH's parts that run without a GPU (ggml/src/ggml-cuda/graph-patch.h): which dispatch groups a
// patch re-dispatches, how a capture's chain of CUDA nodes is cut into the groups that made it, and the decision
// before a patch, checked first one by one and then through a model of ggml_cuda_graph_patch and the recording capture
// (ggml-cuda.cu) over long random runs of prompt micro-batches.
//
// The model: a graph of nodes with a fixed topology (ops) and properties that change from evaluation to evaluation as a
// prompt's do (a few input readers rotate through four input copies, a few cache readers grow, now and then a node's
// op changes, the node count changes or an evaluation runs eagerly). The dispatch loop fuses patterns of up to six
// nodes, some of whose conditions read properties of the pattern's later nodes; a group makes one to three CUDA nodes
// (kernels, sometimes a memset) whose parameters are a function of its nodes' properties, a count that may depend on a
// property, and a pool use that one op holds into the next group. A full capture cuts the graph into chunks and records
// each group; a patch follows ggml_cuda_graph_patch step by step. After every evaluation the executables must equal a
// fresh capture of that evaluation's properties: a patch is only ever exact. A control with a window shorter than the
// longest pattern must be caught making executables that differ.
// Usage: test-cuda-graph-patch   (exit status 0 when every check passes)

#include "../ggml/src/ggml-cuda/graph-patch.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

// ---- the pieces ----------------------------------------------------------------------------------------------------

static std::vector<ggml_cuda_graph_patch_group> make_groups(const std::vector<std::pair<int, int>> & spans, std::vector<int32_t> & node_group, int n) {
    std::vector<ggml_cuda_graph_patch_group> groups;
    node_group.assign(n, -1);
    for (const auto & s : spans) {
        ggml_cuda_graph_patch_group g;
        g.start = s.first;
        g.end   = s.second;
        for (int j = s.first; j <= s.second; j++) {
            node_group[j] = (int32_t) groups.size();
        }
        groups.push_back(g);
    }
    return groups;
}

static void test_dirty() {
    // nodes 0..11: groups [0], [1-3], [5], [6-7], [9], [10-11]; 4 and 8 skipped (views)
    std::vector<int32_t> node_group;
    const auto groups = make_groups({ { 0, 0 }, { 1, 3 }, { 5, 5 }, { 6, 7 }, { 9, 9 }, { 10, 11 } }, node_group, 12);
    std::vector<uint8_t> changed(12, 0);
    int n_nodes = -1;

    CHECK(ggml_cuda_graph_patch_dirty(groups, node_group, changed, 8, n_nodes).empty() && n_nodes == 0);

    changed[2] = 1; // inside [1-3]
    auto d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, 0, n_nodes);
    CHECK(d == std::vector<int>({ 1 }) && n_nodes == 3);
    d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, 2, n_nodes);
    CHECK(d == std::vector<int>({ 0, 1 }) && n_nodes == 4); // group 0 starts 2 nodes before node 2

    changed.assign(12, 0);
    changed[8] = 1; // a skipped node: the groups starting before it within the window
    d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, 0, n_nodes);
    CHECK(d.empty() && n_nodes == 0);
    d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, 2, n_nodes);
    CHECK(d == std::vector<int>({ 3 }) && n_nodes == 2); // [6-7] starts at 6; [5] at 5 is 3 before
    d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, 3, n_nodes);
    CHECK(d == std::vector<int>({ 2, 3 }) && n_nodes == 3);

    changed.assign(12, 0);
    changed[11] = 1; // the last node of a group whose start is 1 before
    d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, 0, n_nodes);
    CHECK(d == std::vector<int>({ 5 }) && n_nodes == 2);

    changed.assign(12, 1);
    d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, 32, n_nodes);
    CHECK(d.size() == groups.size() && n_nodes == 10);

    changed.assign(12, 0);
    changed[0] = 1;
    d = ggml_cuda_graph_patch_dirty(groups, node_group, changed, -5, n_nodes); // a negative window counts as 0
    CHECK(d == std::vector<int>({ 0 }) && n_nodes == 1);
}

static void test_ranges() {
    std::vector<ggml_cuda_graph_patch_group> groups(5);
    // chain 10 11 12 13 14; groups make 2, 0, 1, 2, 0 nodes
    const std::vector<int> chain = { 10, 11, 12, 13, 14 };
    CHECK(ggml_cuda_graph_patch_ranges<int>(chain, { 11, 11, 12, 14, 14 }, groups, 0));
    CHECK(groups[0].first == 0 && groups[0].count == 2);
    CHECK(groups[1].first == 2 && groups[1].count == 0);
    CHECK(groups[2].first == 2 && groups[2].count == 1);
    CHECK(groups[3].first == 3 && groups[3].count == 2);
    CHECK(groups[4].first == 5 && groups[4].count == 0);

    // a group without nodes before the first node (tail null), at an offset in the group list
    std::vector<ggml_cuda_graph_patch_group> g2(4);
    CHECK(ggml_cuda_graph_patch_ranges<int>({ 7, 8 }, { 0, 7, 8 }, g2, 1));
    CHECK(g2[1].count == 0 && g2[2].first == 0 && g2[2].count == 1 && g2[3].first == 1 && g2[3].count == 1);

    // failures: a tail not in the chain, tails going backwards, nodes after the last tail, too many tails
    CHECK(!ggml_cuda_graph_patch_ranges<int>(chain, { 11, 99 }, groups, 0));
    CHECK(!ggml_cuda_graph_patch_ranges<int>(chain, { 13, 11, 14 }, groups, 0));
    CHECK(!ggml_cuda_graph_patch_ranges<int>(chain, { 11, 13 }, groups, 0));
    CHECK(!ggml_cuda_graph_patch_ranges<int>(chain, { 11, 12, 13, 14 }, groups, 3));
    // an empty chunk
    CHECK(ggml_cuda_graph_patch_ranges<int>({}, { 0, 0 }, groups, 0) && groups[0].count == 0 && groups[1].count == 0);
}

static void test_precheck() {
    CHECK(ggml_cuda_graph_patch_precheck(true, false, 2, 2, 10, 100, 60) == GGML_CUDA_GRAPH_PATCH_NONE);
    CHECK(ggml_cuda_graph_patch_precheck(false, false, 2, 2, 10, 100, 60) == GGML_CUDA_GRAPH_PATCH_NOT_READY);
    CHECK(ggml_cuda_graph_patch_precheck(true, false, 0, 0, 0, 100, 60) == GGML_CUDA_GRAPH_PATCH_NOT_READY);
    CHECK(ggml_cuda_graph_patch_precheck(true, false, 2, 1, 10, 100, 60) == GGML_CUDA_GRAPH_PATCH_NOT_READY);
    CHECK(ggml_cuda_graph_patch_precheck(true, true, 2, 2, 10, 100, 60) == GGML_CUDA_GRAPH_PATCH_STRUCTURE);
    CHECK(ggml_cuda_graph_patch_precheck(true, false, 2, 2, 60, 100, 60) == GGML_CUDA_GRAPH_PATCH_NONE);
    CHECK(ggml_cuda_graph_patch_precheck(true, false, 2, 2, 61, 100, 60) == GGML_CUDA_GRAPH_PATCH_SHARE);
    CHECK(ggml_cuda_graph_patch_precheck(true, false, 2, 2, 1, 100, 0) == GGML_CUDA_GRAPH_PATCH_SHARE);
    CHECK(ggml_cuda_graph_patch_precheck(true, false, 2, 2, 100, 100, 100) == GGML_CUDA_GRAPH_PATCH_NONE);
    for (int r = 0; r < GGML_CUDA_GRAPH_PATCH_REASON_COUNT; r++) {
        CHECK(ggml_cuda_graph_patch_reason_name(r)[0] != '?');
    }
}

// ---- the model -----------------------------------------------------------------------------------------------------

enum { OP_VIEW = 0, OP_A, OP_B, OP_C, OP_D, OP_E, OP_H, OP_M, OP_K, N_OPS };

struct mnode {
    int op;
    int shape;
    int ptr;
};

struct cuda_node {
    int      kind;   // 0 kernel, 1 memset
    uint64_t params; // the function and its parameters
    bool operator==(const cuda_node & o) const { return kind == o.kind && params == o.params; }
};

static uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h *= 0xbf58476d1ce4e5b9ull;
    return h ^ (h >> 31);
}

static constexpr int LONGEST = 6; // the longest pattern: a fusion decision at i reads nodes up to i + 5

// the dispatch loop's fusion decision at i: the nodes it fuses after i (0: none)
static int model_fuse(const std::vector<mnode> & g, const int i) {
    const int n = (int) g.size();
    const auto op = [&](int j) { return j < n ? g[j].op : -1; };
    // C D D D D E, when E's shape is even (a condition on the pattern's last node)
    if (op(i) == OP_C && op(i + 1) == OP_D && op(i + 2) == OP_D && op(i + 3) == OP_D && op(i + 4) == OP_D &&
            op(i + 5) == OP_E && g[i + 5].shape % 2 == 0) {
        return 5;
    }
    // C D D, when the two Ds have the same shape
    if (op(i) == OP_C && op(i + 1) == OP_D && op(i + 2) == OP_D && g[i + 1].shape == g[i + 2].shape) {
        return 2;
    }
    // A B, when B's shape is under 40
    if (op(i) == OP_A && op(i + 1) == OP_B && g[i + 1].shape < 40) {
        return 1;
    }
    return 0;
}

// the CUDA nodes group [i, i + k] makes: K makes one more kernel when its shape is over 50 (as attention may add a
// combine kernel); M makes a memset first
static std::vector<cuda_node> model_make(const std::vector<mnode> & g, const int i, const int k) {
    uint64_t h = 1469598103934665603ull;
    for (int j = i; j <= i + k; j++) {
        h = mix(mix(mix(h, (uint64_t) g[j].op), (uint64_t) g[j].shape), (uint64_t) g[j].ptr);
    }
    std::vector<cuda_node> out;
    if (g[i].op == OP_M) {
        out.push_back({ 1, mix(h, 7) });
    }
    out.push_back({ 0, h });
    if (g[i].op == OP_K && g[i].shape > 50) {
        out.push_back({ 0, mix(h, 9) });
    }
    return out;
}

struct model_exec {
    std::vector<std::vector<cuda_node>> chunks;
};

struct model_record {
    bool ready = false;
    std::vector<ggml_cuda_graph_patch_group> groups;
    std::vector<int32_t> node_group;
};

static constexpr int CHUNK = 16;

// the dispatch loop as a capture: the chunks' chains, and the groups with their tails when rec is given. H holds pool
// memory into the next group, which starts with a nonzero pool use
static model_exec model_capture(const std::vector<mnode> & g, model_record * rec) {
    model_exec ex;
    ex.chunks.emplace_back();
    int chunk_end = CHUNK;
    bool prev_h = false;
    std::vector<int> tails; // as chain positions + 1 (0: none), per group of the current chunk
    size_t g0 = 0;
    bool ok = rec != nullptr;
    if (rec) {
        rec->groups.clear();
        rec->node_group.assign(g.size(), -1);
        rec->ready = false;
    }
    const auto end_chunk = [&]() {
        if (ok) {
            std::vector<int> chain(ex.chunks.back().size());
            for (size_t c = 0; c < chain.size(); c++) {
                chain[c] = (int) c + 1;
            }
            ok = ggml_cuda_graph_patch_ranges<int>(chain, tails, rec->groups, g0);
            g0 = rec->groups.size();
            tails.clear();
        }
    };
    for (int i = 0; i < (int) g.size(); i++) {
        if (i >= chunk_end) {
            end_chunk();
            ex.chunks.emplace_back();
            chunk_end = i + CHUNK;
        }
        if (g[i].op == OP_VIEW) {
            continue;
        }
        const size_t pool_before = prev_h ? 64 : 0;
        prev_h = g[i].op == OP_H;
        const int k = model_fuse(g, i);
        for (const cuda_node & c : model_make(g, i, k)) {
            ex.chunks.back().push_back(c);
        }
        if (ok) {
            ggml_cuda_graph_patch_group grp;
            grp.start = i;
            grp.end = i + k;
            grp.chunk = (uint32_t) (ex.chunks.size() - 1);
            grp.pool_used = pool_before;
            for (int j = i; j <= i + k; j++) {
                rec->node_group[j] = (int32_t) rec->groups.size();
            }
            rec->groups.push_back(grp);
            tails.push_back((int) ex.chunks.back().size());
        }
        i += k;
    }
    end_chunk();
    if (rec) {
        rec->ready = ok;
    }
    return ex;
}

// the pool use at each node's dispatch in an evaluation of g (as the full loop has it)
static std::vector<size_t> model_pool(const std::vector<mnode> & g) {
    std::vector<size_t> at(g.size(), 0);
    bool prev_h = false;
    for (int i = 0; i < (int) g.size(); i++) {
        if (g[i].op == OP_VIEW) {
            continue;
        }
        at[i] = prev_h ? 64 : 0;
        prev_h = g[i].op == OP_H;
        i += model_fuse(g, i);
    }
    return at;
}

// ggml_cuda_graph_patch on the model: GGML_CUDA_GRAPH_PATCH_NONE and the executable patched, or why not
static int model_patch(const std::vector<mnode> & g, const std::vector<mnode> & prev, const model_record & rec,
        model_exec & ex, const int window, const int max_pct, int & n_dirty_groups) {
    n_dirty_groups = 0;
    const bool structure = g.size() != prev.size() ||
        !std::equal(g.begin(), g.end(), prev.begin(), [](const mnode & a, const mnode & b) { return a.op == b.op; });
    std::vector<uint8_t> changed(g.size(), 0);
    for (size_t i = 0; i < g.size() && i < prev.size(); i++) {
        changed[i] = g[i].op != prev[i].op || g[i].shape != prev[i].shape || g[i].ptr != prev[i].ptr;
    }
    int n_dirty_nodes = 0;
    const std::vector<int> dirty = structure ? std::vector<int>() :
        ggml_cuda_graph_patch_dirty(rec.groups, rec.node_group, changed, window, n_dirty_nodes);
    int r = ggml_cuda_graph_patch_precheck(rec.ready, structure, ex.chunks.size(), ex.chunks.size(), n_dirty_nodes,
        (int) g.size(), max_pct);
    if (r != GGML_CUDA_GRAPH_PATCH_NONE) {
        return r;
    }
    // the scratch capture: each dirty group from its own start, with the pool use the scratch loop has there (nothing
    // held: the groups before it were not dispatched)
    std::vector<cuda_node> scratch;
    std::vector<int> tails;
    for (const int d : dirty) {
        const ggml_cuda_graph_patch_group & grp = rec.groups[d];
        const size_t pool_now = 0;
        if (pool_now != grp.pool_used) {
            return GGML_CUDA_GRAPH_PATCH_POOL;
        }
        const int k = model_fuse(g, grp.start);
        if (k != grp.end - grp.start) {
            return GGML_CUDA_GRAPH_PATCH_FUSE;
        }
        for (const cuda_node & c : model_make(g, grp.start, k)) {
            scratch.push_back(c);
        }
        tails.push_back((int) scratch.size());
    }
    std::vector<int> chain(scratch.size());
    for (size_t c = 0; c < chain.size(); c++) {
        chain[c] = (int) c + 1;
    }
    std::vector<ggml_cuda_graph_patch_group> made(dirty.size());
    if (!ggml_cuda_graph_patch_ranges<int>(chain, tails, made, 0)) {
        return GGML_CUDA_GRAPH_PATCH_NODES;
    }
    for (size_t d = 0; d < dirty.size(); d++) {
        const ggml_cuda_graph_patch_group & grp = rec.groups[dirty[d]];
        if (made[d].count != grp.count) {
            return GGML_CUDA_GRAPH_PATCH_NODES;
        }
        for (uint32_t j = 0; j < grp.count; j++) {
            if (scratch[made[d].first + j].kind != ex.chunks[grp.chunk][grp.first + j].kind) {
                return GGML_CUDA_GRAPH_PATCH_NODES;
            }
        }
    }
    for (size_t d = 0; d < dirty.size(); d++) {
        const ggml_cuda_graph_patch_group & grp = rec.groups[dirty[d]];
        for (uint32_t j = 0; j < grp.count; j++) {
            ex.chunks[grp.chunk][grp.first + j] = scratch[made[d].first + j];
        }
    }
    n_dirty_groups = (int) dirty.size();
    return GGML_CUDA_GRAPH_PATCH_NONE;
}

static std::vector<mnode> random_graph(std::mt19937 & rng, const int n) {
    std::vector<mnode> g(n);
    std::uniform_int_distribution<int> op(0, N_OPS - 1);
    for (int i = 0; i < n; i++) {
        g[i] = { op(rng), 10 + (int) (rng() % 20), (int) (rng() % 1000) };
    }
    // a few long patterns, so that the six-node fusion and its condition on its last node occur
    for (int p = 0; p + 6 < n; p += 30 + (int) (rng() % 21)) {
        g[p] = { OP_C, 10, 1 };
        for (int j = 1; j <= 4; j++) {
            g[p + j] = { OP_D, 20, 2 };
        }
        g[p + 5] = { OP_E, 30 + (int) (rng() % 2), 3 };
    }
    return g;
}

struct run_counts {
    long evals = 0, patched = 0, full = 0, groups = 0, inexact = 0;
    long by_reason[GGML_CUDA_GRAPH_PATCH_REASON_COUNT] = {};
};

// one run of prompt micro-batches: rotating input readers, growing cache readers (those that are E end a long pattern,
// so the six-node fusion turns on and off), sometimes a structure change, another node count or an eager evaluation
static run_counts random_run(std::mt19937 & rng, const int window, const int n_evals) {
    run_counts rc;
    std::vector<mnode> g = random_graph(rng, 40 + (int) (rng() % 120));
    std::vector<int> inputs, cache;
    const unsigned density = 6 + rng() % 60; // one node in this many reads an input copy
    for (int i = 0; i < (int) g.size(); i++) {
        if (rng() % density == 0) {
            inputs.push_back(i);
        } else if (g[i].op == OP_E || (g[i].op == OP_K && rng() % 3 == 0) || rng() % 80 == 0) {
            cache.push_back(i);
        }
    }
    model_record rec;
    model_exec ex;
    std::vector<mnode> prev;
    bool have = false;
    for (int e = 0; e < n_evals; e++) {
        std::vector<mnode> next = g;
        for (const int i : inputs) {
            next[i].ptr = 5000 + (e % 4) * 100 + i; // four input copies
        }
        for (const int i : cache) {
            next[i].shape = g[i].shape + 2 * e + e / 7; // the cache grows every micro-batch, its parity now and then
        }
        const unsigned u = rng() % 1000;
        if (u < 8) {
            next[rng() % next.size()].op = (int) (rng() % N_OPS); // a structure change
            g = next;
        } else if (u < 12) {
            next.push_back({ OP_A, 11, 7 }); // another node count
            g = next;
        }
        const bool eager = u >= 12 && u < 20;
        rc.evals++;
        if (eager) {
            rec.ready = false; // an eager evaluation leaves the executables behind
            prev = next;
            continue;
        }
        int n_groups = 0;
        int r = have ? model_patch(next, prev, rec, ex, window, 60, n_groups) : GGML_CUDA_GRAPH_PATCH_NOT_READY;
        rc.by_reason[r]++;
        if (r == GGML_CUDA_GRAPH_PATCH_NONE) {
            rc.patched++;
            rc.groups += n_groups;
        } else {
            ex = model_capture(next, &rec);
            have = true;
            rc.full++;
        }
        // the executables as a fresh capture of this evaluation's properties would make them
        const model_exec fresh = model_capture(next, nullptr);
        rc.inexact += fresh.chunks != ex.chunks;
        prev = next;
    }
    return rc;
}

static void test_model() {
    // the model's loop: the recorded groups cover every dispatched node and the pool use is the loop's
    {
        std::mt19937 rng(7);
        for (int t = 0; t < 200; t++) {
            const std::vector<mnode> g = random_graph(rng, 60);
            model_record rec;
            const model_exec ex = model_capture(g, &rec);
            CHECK(rec.ready);
            const std::vector<size_t> pool = model_pool(g);
            size_t n_cuda = 0;
            for (const auto & c : ex.chunks) {
                n_cuda += c.size();
            }
            size_t counted = 0;
            for (const auto & grp : rec.groups) {
                counted += grp.count;
                CHECK(grp.pool_used == pool[grp.start]);
                CHECK(grp.first + grp.count <= ex.chunks[grp.chunk].size());
            }
            CHECK(counted == n_cuda);
        }
    }

    long patched = 0, full = 0, groups = 0, evals = 0, ctl_inexact = 0, ctl_runs = 0;
    long reasons[GGML_CUDA_GRAPH_PATCH_REASON_COUNT] = {};
    for (int seed = 1; seed <= 2000; seed++) {
        std::mt19937 r_on(seed);
        const run_counts on = random_run(r_on, LONGEST - 1, 300);
        CHECK(on.inexact == 0);
        patched += on.patched;
        full    += on.full;
        groups  += on.groups;
        evals   += on.evals;
        for (int r = 0; r < GGML_CUDA_GRAPH_PATCH_REASON_COUNT; r++) {
            reasons[r] += on.by_reason[r];
        }
        // the control: a window shorter than the longest pattern misses a fusion that a changed last node turns on
        std::mt19937 r_ctl(seed);
        const run_counts ctl = random_run(r_ctl, 2, 300);
        ctl_inexact += ctl.inexact;
        ctl_runs    += ctl.inexact > 0;
    }
    CHECK(patched > 0 && full > 0);
    for (int r = GGML_CUDA_GRAPH_PATCH_NONE; r < GGML_CUDA_GRAPH_PATCH_REASON_COUNT; r++) {
        if (r != GGML_CUDA_GRAPH_PATCH_SKIP && r != GGML_CUDA_GRAPH_PATCH_API) {
            CHECK(reasons[r] > 0); // the model reaches every reason but the two it has no part for
        }
    }
    CHECK(ctl_runs > 0);
    printf("random runs: 2000 x 300 evaluations (%ld); window %d: %ld patched (%.1f groups each), %ld full captures, "
           "every executable a fresh capture's;", evals, LONGEST - 1, patched, patched ? (double) groups / patched : 0.0, full);
    for (int r = GGML_CUDA_GRAPH_PATCH_NOT_READY; r < GGML_CUDA_GRAPH_PATCH_REASON_COUNT; r++) {
        printf(" %s %ld", ggml_cuda_graph_patch_reason_name(r), reasons[r]);
    }
    printf("; the control (window 2) made other executables in %ld evaluations of %ld runs\n", ctl_inexact, ctl_runs);
}

int main() {
    test_dirty();
    test_ranges();
    test_precheck();
    test_model();
    printf("test-cuda-graph-patch: %d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
