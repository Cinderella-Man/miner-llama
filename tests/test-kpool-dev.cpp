// The QSA k-pool tables kept on the devices (LLAMA_QSA_KPOOL_DEV, C100) on a model file (a random qwen4exp from
// test-llama-archs, or a real one), on the CPU backend, where the tables' copy sits in a CPU buffer and the graph
// updates it as it updates a card's.
//
// Three contexts take the same steps: prompts in micro-batches, one-token steps, four-token steps (all outputs) of
// which a rollback keeps one to three, two sequences alone and in one batch, a partial checkpoint restored as the
// server's speculative steps restore one (the recurrent state, then the cells after it removed), a whole sequence's
// state saved and restored, a sequence removed and prompted again, a cleared memory. After every step all three
// contexts' logits must be equal bit for bit:
//   - ref:   without the switch (the tables uploaded whole, as pascal-p106-v4);
//   - check: with it and LLAMA_QSA_KPOOL_DEV_CHECK=1, which compares every ubatch, after a synchronize, each copy as
//            the steps before left it and this ubatch's update with the inputs the host builds without the switch,
//            and aborts on any difference;
//   - dev:   with the switch alone.
// --top-k lowers the indexer's budget (a random qwen4exp keeps every pool), so that the selection drops pools and the
// logits depend on which pools the tables name.
//
// Usage: test-kpool-dev -m <model.gguf> [-t threads] [-s steps] [-u n_ubatch] [--top-k k] [--seed n] [--unified]

#include "../src/llama-model.h"
#include "../src/llama-ext.h"

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_checks = 0;
static int n_fail   = 0;

#define CHECK(cond) do { n_checks++; if (!(cond)) { n_fail++; fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)

struct args_t {
    std::string model;
    int      n_threads = 4;
    int      n_steps   = 240;
    int      n_ubatch  = 16;
    int      top_k     = 0; // 0: the model's
    uint32_t seed      = 1;
    bool     unified   = false;
};

static llama_context * make_ctx(llama_model * model, const args_t & a, const char * dev, const char * check) {
    // the memory reads both when it is created
    if (dev)   { setenv("LLAMA_QSA_KPOOL_DEV", dev, 1); }         else { unsetenv("LLAMA_QSA_KPOOL_DEV"); }
    if (check) { setenv("LLAMA_QSA_KPOOL_DEV_CHECK", check, 1); } else { unsetenv("LLAMA_QSA_KPOOL_DEV_CHECK"); }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 256;
    cp.n_ubatch        = a.n_ubatch;
    cp.n_seq_max       = 2;
    cp.n_rs_seq        = 4; // a verify step's rollback, as the server's speculative decoding
    cp.kv_unified      = a.unified;
    cp.n_threads       = a.n_threads;
    cp.n_threads_batch = a.n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);

    unsetenv("LLAMA_QSA_KPOOL_DEV");
    unsetenv("LLAMA_QSA_KPOOL_DEV_CHECK");
    return ctx;
}

struct lockstep {
    std::vector<llama_context *> ctxs;
    int n_vocab = 0;

    llama_pos pos[2] = { 0, 0 };
    bool      live[2] = { false, false };

    std::mt19937 rng;

    uint64_t n_decodes = 0;
    uint64_t n_logits  = 0;
    uint64_t n_differ  = 0;

    llama_token tok() { return std::uniform_int_distribution<int>(1, n_vocab - 1)(rng); }

    // parts: (sequence, tokens); outputs on every token, or the last of each part
    bool decode(const std::vector<std::pair<llama_seq_id, int>> & parts, bool all_out) {
        int n = 0;
        for (const auto & p : parts) {
            n += p.second;
        }
        llama_batch b = llama_batch_init(n, 0, 1);
        for (const auto & p : parts) {
            for (int i = 0; i < p.second; ++i) {
                const int j = b.n_tokens++;
                b.token[j]     = tok();
                b.pos[j]       = pos[p.first] + i;
                b.n_seq_id[j]  = 1;
                b.seq_id[j][0] = p.first;
                b.logits[j]    = all_out || i == p.second - 1;
            }
        }

        bool ok = true;
        for (llama_context * c : ctxs) {
            const int rc = llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b);
            CHECK(rc == 0);
            ok = ok && rc == 0;
        }
        n_decodes++;

        if (ok) {
            for (int j = 0; j < b.n_tokens; ++j) {
                if (!b.logits[j]) {
                    continue;
                }
                const float * l0 = llama_get_logits_ith(ctxs[0], j);
                for (size_t k = 1; k < ctxs.size(); ++k) {
                    const float * lk = llama_get_logits_ith(ctxs[k], j);
                    if (memcmp(l0, lk, n_vocab*sizeof(float)) != 0) {
                        if (n_differ++ < 8) {
                            fprintf(stderr, "decode %llu: output %d of context %zu differs from the reference\n",
                                    (unsigned long long) n_decodes, j, k);
                        }
                    }
                }
                n_logits++;
            }
            for (const auto & p : parts) {
                pos[p.first] += p.second;
            }
        }

        llama_batch_free(b);
        return ok;
    }

    bool seq_rm(llama_seq_id s, llama_pos p0, llama_pos p1) {
        bool ok = true;
        for (llama_context * c : ctxs) {
            ok = llama_memory_seq_rm(llama_get_memory(c), s, p0, p1) && ok;
        }
        return ok;
    }
};

// a saved state per context
struct saved_t {
    bool      have    = false;
    bool      partial = false;
    llama_pos pos     = 0; // the sequence's next position when saved
    std::vector<std::vector<uint8_t>> data;
};

static void run(llama_model * model, const args_t & a) {
    lockstep ls;
    ls.ctxs = {
        make_ctx(model, a, nullptr, nullptr),
        make_ctx(model, a, "1", "1"),
        make_ctx(model, a, "1", nullptr),
    };
    for (llama_context * c : ls.ctxs) {
        if (c == nullptr) {
            fprintf(stderr, "failed to create a context\n");
            exit(1);
        }
    }
    ls.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    ls.rng.seed(a.seed);

    // the switch must be on in the two contexts that set it
    for (size_t k = 0; k < ls.ctxs.size(); ++k) {
        uint64_t n = 0;
        CHECK(llama_kpool_dev_stats(ls.ctxs[k], &n, nullptr, nullptr, nullptr) == (k > 0));
    }

    const int n_seq_ctx = 1024/2; // keep each sequence in its half of the cache
    saved_t saved[2];

    auto prompt = [&](llama_seq_id s) {
        const int n = std::uniform_int_distribution<int>(20, 90)(ls.rng);
        ls.decode({ { s, n } }, true);
        ls.live[s] = true;
    };
    auto drop = [&](llama_seq_id s) {
        CHECK(ls.seq_rm(s, -1, -1));
        ls.pos[s]  = 0;
        ls.live[s] = false;
        saved[s]   = saved_t();
    };

    int n_rollback = 0, n_both = 0, n_ckpt = 0, n_ckpt_restore = 0, n_state = 0, n_state_restore = 0, n_clear = 0;

    prompt(0);
    for (int step = 0; step < a.n_steps; ++step) {
        const int r = std::uniform_int_distribution<int>(0, 99)(ls.rng);
        llama_seq_id s = (ls.live[1] && std::uniform_int_distribution<int>(0, 1)(ls.rng)) ? 1 : 0;
        if (!ls.live[s]) {
            prompt(s);
            continue;
        }

        if (ls.pos[s] > n_seq_ctx - 96) {
            drop(s);
            prompt(s);
        } else if (r < 30) {
            ls.decode({ { s, 1 } }, true);
        } else if (r < 55) {
            // a verify step, then keep one to three of its four tokens
            const llama_pos p0 = ls.pos[s];
            ls.decode({ { s, 4 } }, true);
            if (std::uniform_int_distribution<int>(0, 9)(ls.rng) < 6) {
                const int keep = std::uniform_int_distribution<int>(1, 3)(ls.rng);
                CHECK(ls.seq_rm(s, p0 + keep, -1));
                ls.pos[s] = p0 + keep;
                n_rollback++;
            }
        } else if (r < 65) {
            if (ls.live[0] && ls.live[1]) {
                ls.decode({ { 0, 2 }, { 1, 2 } }, true);
                n_both++;
            } else {
                prompt(1);
            }
        } else if (r < 75) {
            // the server's speculative checkpoint: the recurrent state only, then the cells after it removed
            auto & sv = saved[s];
            if (!sv.have || !sv.partial) {
                sv = saved_t();
                sv.have = true; sv.partial = true; sv.pos = ls.pos[s];
                for (llama_context * c : ls.ctxs) {
                    const size_t n = llama_state_seq_get_size_ext(c, s, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    sv.data.emplace_back(n);
                    CHECK(llama_state_seq_get_data_ext(c, sv.data.back().data(), n, s, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == n);
                }
                n_ckpt++;
                ls.decode({ { s, std::uniform_int_distribution<int>(1, 4)(ls.rng) } }, true);
            } else {
                for (size_t k = 0; k < ls.ctxs.size(); ++k) {
                    llama_context * c = ls.ctxs[k];
                    CHECK(llama_state_seq_set_data_ext(c, sv.data[k].data(), sv.data[k].size(), s, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == sv.data[k].size());
                    CHECK(llama_memory_seq_rm(llama_get_memory(c), s, sv.pos, -1));
                }
                ls.pos[s] = sv.pos;
                sv = saved_t();
                n_ckpt_restore++;
                ls.decode({ { s, 4 } }, true);
            }
        } else if (r < 82) {
            // a whole sequence's state, saved and later restored into the emptied sequence
            auto & sv = saved[s];
            if (!sv.have || sv.partial) {
                sv = saved_t();
                sv.have = true; sv.pos = ls.pos[s];
                for (llama_context * c : ls.ctxs) {
                    const size_t n = llama_state_seq_get_size(c, s);
                    sv.data.emplace_back(n);
                    CHECK(llama_state_seq_get_data(c, sv.data.back().data(), n, s) == n);
                }
                n_state++;
            } else {
                CHECK(ls.seq_rm(s, -1, -1));
                for (size_t k = 0; k < ls.ctxs.size(); ++k) {
                    CHECK(llama_state_seq_set_data(ls.ctxs[k], sv.data[k].data(), sv.data[k].size(), s) == sv.data[k].size());
                }
                ls.pos[s] = sv.pos;
                sv = saved_t();
                n_state_restore++;
            }
            ls.decode({ { s, 1 } }, true);
        } else if (r < 90) {
            // a short prompt continuation in micro-batches
            ls.decode({ { s, std::uniform_int_distribution<int>(17, 40)(ls.rng) } }, false);
        } else if (r < 96) {
            if (s == 1) {
                drop(1);
            } else {
                ls.decode({ { s, 1 } }, true);
            }
        } else if (r < 98) {
            for (llama_context * c : ls.ctxs) {
                llama_memory_clear(llama_get_memory(c), true);
            }
            ls.pos[0] = ls.pos[1] = 0;
            ls.live[0] = ls.live[1] = false;
            saved[0] = saved[1] = saved_t();
            n_clear++;
            prompt(0);
        } else {
            ls.decode({ { s, 1 } }, true);
        }
    }
    // one more step, so that the check reads the copies the last step left
    ls.decode({ { 0, 1 } }, true);

    printf("%llu decodes, %llu outputs compared, %llu differ; %d rollbacks, %d two-sequence batches, %d/%d checkpoints "
           "saved/restored, %d/%d sequence states saved/restored, %d clears\n",
           (unsigned long long) ls.n_decodes, (unsigned long long) ls.n_logits, (unsigned long long) ls.n_differ,
           n_rollback, n_both, n_ckpt, n_ckpt_restore, n_state, n_state_restore, n_clear);
    CHECK(ls.n_differ == 0);
    CHECK(ls.n_logits > 0);

    uint64_t ub[3] = {}, full[3] = {}, rows[3] = {}, checked[3] = {};
    for (size_t k = 1; k < ls.ctxs.size(); ++k) {
        CHECK(llama_kpool_dev_stats(ls.ctxs[k], &ub[k], &full[k], &rows[k], &checked[k]));
        printf("context %zu: %llu ubatches, %llu full updates, %llu rows uploaded (%.1f a ubatch), %llu checked\n", k,
               (unsigned long long) ub[k], (unsigned long long) full[k], (unsigned long long) rows[k],
               ub[k] ? (double) rows[k]/ub[k] : 0.0, (unsigned long long) checked[k]);
    }
    CHECK(ub[1] > 0 && checked[1] == ub[1]);
    CHECK(ub[2] == ub[1] && checked[2] == 0 && rows[2] == rows[1] && full[2] == full[1]);
    CHECK(full[1] < ub[1]);
    CHECK(n_rollback > 0 && n_ckpt_restore > 0 && n_state_restore > 0);

    for (llama_context * c : ls.ctxs) {
        llama_free(c);
    }
}

int main(int argc, char ** argv) {
    args_t a;
    const char * usage = "usage: %s -m <model.gguf> [-t threads] [-s steps] [-u n_ubatch] [--top-k k] [--seed n] [--unified]\n";
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "-m" && i + 1 < argc) {
            a.model = argv[++i];
        } else if (s == "-t" && i + 1 < argc) {
            a.n_threads = atoi(argv[++i]);
        } else if (s == "-s" && i + 1 < argc) {
            a.n_steps = atoi(argv[++i]);
        } else if (s == "-u" && i + 1 < argc) {
            a.n_ubatch = atoi(argv[++i]);
        } else if (s == "--top-k" && i + 1 < argc) {
            a.top_k = atoi(argv[++i]);
        } else if (s == "--seed" && i + 1 < argc) {
            a.seed = (uint32_t) atoi(argv[++i]);
        } else if (s == "--unified") {
            a.unified = true;
        } else {
            fprintf(stderr, usage, argv[0]);
            return 2;
        }
    }
    if (a.model.empty()) {
        fprintf(stderr, usage, argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = { cpu, nullptr };
    mp.devices = devs;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", a.model.c_str());
        return 1;
    }
    if (model->arch != LLM_ARCH_QWEN4EXP || model->hparams.indexer_kpool == 0) {
        fprintf(stderr, "%s is not a qwen4exp with QSA layers\n", a.model.c_str());
        return 1;
    }
    if (a.top_k > 0) {
        // a budget of whole pools, as load_arch_hparams requires
        GGML_ASSERT(a.top_k % (int) model->hparams.indexer_kpool == 0);
        model->hparams.indexer_top_k = (uint32_t) a.top_k;
    }
    printf("%s: kpool %u, top-k %u, n_ubatch %d, %s KV, seed %u, %d steps\n", a.model.c_str(), model->hparams.indexer_kpool,
           model->hparams.indexer_top_k, a.n_ubatch, a.unified ? "unified" : "per-sequence", a.seed, a.n_steps);

    run(model, a);

    llama_model_free(model);
    llama_backend_free();

    printf("%d checks, %d failed\n", n_checks, n_fail);
    return n_fail == 0 ? 0 : 1;
}
