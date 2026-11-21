// test-moe-ids [-m model.gguf] [-ngl N] [-t threads] [--prompt N] [--ubatch N] [--steps N]: every MUL_MAT_ID of a
// model's forward passes gets, in each token column, each expert at most once and only ids below its expert count.
// llama.cpp's CUDA kernels assume both (they group a token's columns by expert); the CPU kernels do not, so a graph
// that repeats an id runs on the CPU and goes out of bounds on CUDA (Q238's first cold-tier graph read slot 0 of a
// tier for every pick of the other tier). This checks on the CPU what a GPU needs: the prompt in micro-batches, then
// single tokens, each MUL_MAT_ID's ids read back after it ran. Before the model, the checker itself must flag a
// repeated id and an id out of range (and pass top-k-like ids). Prints the counts and "N checks, M failed"; exits 1
// on any failure.

#include "llama.h"
#include "llama-cpp.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

struct ids_report {
    int64_t nodes   = 0; // MUL_MAT_ID nodes checked
    int64_t columns = 0; // token columns checked
    int64_t repeats = 0; // columns with an id twice
    int64_t range   = 0; // ids outside [0, n_as)
};

// one MUL_MAT_ID's ids [n_ids, n_cols] with byte strides nb0, nb1 and n_as expert matrices
static void check_ids(const char * data, int64_t n_ids, int64_t n_cols, size_t nb0, size_t nb1, int64_t n_as, ids_report & r) {
    r.nodes++;
    for (int64_t c = 0; c < n_cols; ++c) {
        std::set<int32_t> seen;
        bool repeated = false;
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) (data + c*nb1 + i*nb0);
            if (id < 0 || id >= n_as) {
                r.range++;
            }
            repeated |= !seen.insert(id).second;
        }
        r.columns++;
        r.repeats += repeated;
    }
}

static bool ids_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (t->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    if (ask) {
        return true;
    }
    const ggml_tensor * ids = t->src[2];
    std::vector<char> buf(ggml_nbytes(ids));
    ggml_backend_tensor_get(ids, buf.data(), 0, buf.size());
    check_ids(buf.data(), ids->ne[0], ids->ne[1], ids->nb[0], ids->nb[1], t->src[0]->ne[2], *(ids_report *) user_data);
    return true;
}

static int n_checks = 0;
static int n_failed = 0;

static void expect(bool ok, const char * what) {
    n_checks++;
    if (!ok) {
        n_failed++;
        fprintf(stderr, "FAIL: %s\n", what);
    }
}

// the checker on ids made by hand: 4 slots of 3 tokens out of 8 experts
static void test_checker() {
    auto run = [](const std::vector<int32_t> & v, int64_t n_as) {
        ids_report r;
        check_ids((const char *) v.data(), 4, 3, sizeof(int32_t), 4*sizeof(int32_t), n_as, r);
        return r;
    };
    const ids_report topk = run({ 0, 5, 2, 7,   1, 3, 6, 4,   7, 6, 5, 0 }, 8);
    expect(topk.columns == 3 && topk.repeats == 0 && topk.range == 0, "distinct ids pass");
    // the first tiered graph's hot ids: picks of the other tier read slot 0, next to a real pick of slot 0
    const ids_report tier = run({ 0, 0, 2, 0,   1, 3, 6, 4,   5, 0, 0, 6 }, 8);
    expect(tier.repeats == 2 && tier.range == 0, "a repeated id is flagged in each column that has one");
    const ids_report out = run({ 0, 5, 2, 8,   1, 3, 6, -1,   7, 6, 5, 0 }, 8);
    expect(out.repeats == 0 && out.range == 2, "ids outside [0, n_as) are flagged");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int n_gpu_layers = 0;
    int n_threads    = 4;
    int n_prompt     = 64;
    int n_ubatch     = 32;
    int n_steps      = 4;
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
            n_gpu_layers = atoi(next());
        } else if (a == "-t") {
            n_threads = atoi(next());
        } else if (a == "--prompt") {
            n_prompt = atoi(next());
        } else if (a == "--ubatch") {
            n_ubatch = atoi(next());
        } else if (a == "--steps") {
            n_steps = atoi(next());
        } else {
            fprintf(stderr, "usage: %s [-m model.gguf] [-ngl N] [-t threads] [--prompt N] [--ubatch N] [--steps N]\n", argv[0]);
            return 2;
        }
    }

    test_checker();

    if (model_path) {
        llama_backend_init();
        ggml_backend_load_all();

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = n_gpu_layers;
        llama_model * model = llama_model_load_from_file(model_path, mp);
        expect(model != nullptr, "the model loads");
        if (model) {
            ids_report r;
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx             = n_prompt + n_steps + 16;
            cp.n_batch           = n_prompt;
            cp.n_ubatch          = n_ubatch;
            cp.n_seq_max         = 1;
            cp.n_threads         = n_threads;
            cp.n_threads_batch   = n_threads;
            cp.no_perf           = true;
            cp.cb_eval           = ids_cb;
            cp.cb_eval_user_data = &r;
            llama_context * ctx = llama_init_from_model(model, cp);
            expect(ctx != nullptr, "a context");
            if (ctx) {
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                std::mt19937 rng(238);
                std::uniform_int_distribution<int> tok(1, n_vocab - 1);
                llama_batch_ext_ptr batch(llama_batch_ext_init(ctx)); // holds up to n_batch = n_prompt entries
                int pos = 0;
                auto eval = [&](int n) {
                    llama_batch_ext_clear(batch.get());
                    bool filled = true;
                    for (int i = 0; i < n; ++i) {
                        const llama_pos p   = pos + i;
                        const int32_t   idx = llama_batch_ext_add_token(batch.get(), 0, tok(rng));
                        filled = filled && idx >= 0 && llama_batch_ext_set_pos(batch.get(), idx, &p) &&
                                 llama_batch_ext_set_output_logits(batch.get(), idx, i == n - 1);
                    }
                    pos += n;
                    return filled && llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
                };
                expect(eval(n_prompt), "the prompt decodes");
                for (int s = 0; s < n_steps; ++s) {
                    expect(eval(1), "a single token decodes");
                }
                batch.reset();
                llama_free(ctx);

                printf("test-moe-ids: %lld MUL_MAT_ID nodes, %lld token columns, %lld with a repeated id, %lld ids out of range\n",
                       (long long) r.nodes, (long long) r.columns, (long long) r.repeats, (long long) r.range);
                expect(r.nodes > 0, "the model has MUL_MAT_ID nodes");
                expect(r.repeats == 0, "no token column repeats an id");
                expect(r.range == 0, "every id is below its expert count");
            }
            llama_model_free(model);
        }
        llama_backend_free();
    }

    printf("%d checks, %d failed\n", n_checks, n_failed);
    return n_failed == 0 ? 0 : 1;
}
