// recurrent state cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include "../../src/llama-ext.h" // llama_arch_from_string

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// roll back multiple sequences, then replay them in a single batch whose per-seq token count exceeds n_ubatch: each
// seq's replay spans several ubatches while its rollback restore is still pending. Compared against a reference context
// that never advanced past the rollback point and decodes the identical replay batch. Only the first replayed token of
// each seq is compared.
struct case_multi_seq_split_replay : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "replay"; }

    const char * description() const override {
        return "replay rolled-back sequences in a batch wider than n_ubatch, against a reference context";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t  n_seqs     = 2;
        constexpr uint32_t  n_ubatch   = 16;
        constexpr uint32_t  n_prompt   = 20;
        constexpr uint32_t  n_rollback = 3;
        constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
        constexpr llama_pos p0         = n_prompt - n_rollback;

        // both contexts have the same layout, only what they decode differs
        const auto make_ctx_multi = [&]() {
            ctx_opts opts;
            opts.n_ctx      = 512;
            opts.n_batch    = 256;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seqs;
            opts.n_rs_seq   = 8;
            opts.kv_unified = 0;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_roll = make_ctx_multi();
        llama_context_ptr ctx_ref  = make_ctx_multi();
        if (!ctx_roll || !ctx_ref) {
            return case_result::fail();
        }

        if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        const auto tok = [&](uint32_t seq, llama_pos pos) {
            return gen_token(n_vocab, (llama_seq_id) seq, pos);
        };

        bool ok = true;

        // keep the anchor at p0 - 1 when rolling back the tail
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);

            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) p0);

            ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

            // a second partial removal while one is pending must be refused
            ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
        }
        CTX_REQUIRE(ok, "the multi-seq prefill / rollback failed");

        // all seqs replay in a single batch
        const auto decode_replay = [&](llama_context * ctx) {
            common_batch batch(ctx);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                for (uint32_t i = 0; i < n_replay; ++i) {
                    const llama_pos pos = p0 + (llama_pos) i;
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
                }
            }
            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        ok = decode_replay(ctx_roll.get()) && decode_replay(ctx_ref.get());
        CTX_REQUIRE(ok, "the multi-seq replay decode failed");

        // the first replayed token of each seq reads the state right after the rollback, so its
        //   logits should match; the later tokens are not compared because small state
        //   differences at the start grow with each replayed token
        // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
        constexpr float nmse_eps = 1e-4f;

        logits_diff diff;
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t i = s*n_replay; // first replayed token of this seq

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
                return case_result::fail();
            }

            diff.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(diff.nmse() <= nmse_eps,
                "multi-seq split replay logits mismatch (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        // the replay above can leave the 2 contexts in slightly different states, and the tail
        //   below assumes they start from the same state - so copy the roll state into ref
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            std::vector<uint8_t> state;

            ok = ok && get_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, state);
            ok = ok && set_seq_state(ctx_ref.get(),  (llama_seq_id) s, 0, state);

            // do a second round of state save/load to negate any SWA-positioning effects
            if (ok) {
                std::vector<uint8_t> restored;
                ok = ok && get_seq_state(ctx_ref.get(), (llama_seq_id) s, 0, restored);

                // what was loaded into ref must come back out bit-identical
                CTX_REQUIRE(ok && state == restored,
                        "the seq " + std::to_string(s) + " state did not round-trip bit-exact (sizes " +
                        std::to_string(state.size()) + "/" + std::to_string(restored.size()) + ")");

                ok = ok && set_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, restored);
            }
        }
        CTX_REQUIRE(ok, "the multi-seq state copy failed");

        // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
        //   in ctx_ref only, then compare identical seq-1-only continuations
        constexpr uint32_t n_tail = 4;

        {
            common_batch batch_tail(ctx_ref.get());
            for (uint32_t i = 0; i < n_tail; ++i) {
                const llama_pos pos = p0 + (llama_pos) (n_replay + i);
                batch_tail.add(tok(0, pos + 7), pos, 0, false);
            }
            ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
        }

        logits_diff diff_tail;
        for (uint32_t i = 0; i < n_tail && ok; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1) &&
                 decode_one(ctx_ref.get(),  tok(1, pos), pos, 1);
            if (!ok) {
                break;
            }

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing tail logits at step %u\n", __func__, i);
                return case_result::fail();
            }

            diff_tail.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(ok && diff_tail.nmse() <= 0.0f,
                "a seq-1-only decode leaked the state of seq 0 (nmse " + std::to_string(diff_tail.nmse()) + ")");

        TLOG_INF("%s: seq-1-only decode is independent of seq 0 (max diff %g, nmse %g)\n",
                __func__, (double) diff_tail.diff_max, diff_tail.nmse());

        return case_result::pass();
    }
};

// save a rolled-back single-seq state, restore it into a fresh and into a dirty context, and
// verify exact logit matches on replay
struct case_rollback : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback"; }

    const char * description() const override {
        return "a rolled-back recurrent state restores into fresh and dirty contexts with identical replay";
    }

    case_result run(model_run & mr) const override {
        const llama_vocab * vocab   = llama_model_get_vocab(mr.model);
        const int           n_vocab = llama_vocab_n_tokens(vocab);

        const auto make_rs_ctx = [&]() {
            ctx_opts opts;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_src = make_rs_ctx();
        llama_context_ptr ctx_dst = make_rs_ctx();
        if (!ctx_src || !ctx_dst) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        constexpr uint32_t n_rollback = 3;
        if (n_rs_seq < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        llama_tokens tokens;
        if (!mr.has_vocab()) {
            tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        } else {
            tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
        }
        if (tokens.empty()) {
            TLOG_ERR("%s: not enough prompt tokens\n", __func__);
            return case_result::fail();
        }
        tokens.resize(n_rs_seq + 1, tokens.back());

        const llama_pos rollback_pos = (llama_pos) tokens.size() - n_rollback;

        // Decode the full prompt on the source, then roll back three positions.
        // Replaying them crosses DSV4's ratio-4 compressor boundary.
        // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
        if (!decode_tokens(ctx_src.get(), tokens)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: rollback failed\n", __func__);
            return case_result::fail();
        }

        // Save the rolled-back state and restore it into a fresh context.
        common_prompt_checkpoint ckpt;
        ckpt.update_tgt(ctx_src.get(), 0, 0);
        ckpt.load_tgt(ctx_dst.get(), 0, 0);

        // exact match - the same context configuration replaying the same tokens
        constexpr float nmse_eps = 0.0;

        std::vector<std::vector<float>> logits_src_replay(n_rollback);
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing replay logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);

            const logits_diff diff = compare_logits(logits_src, logits_dst, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "replay logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        // the replay ran in single-token ubatches, which hold no snapshot that deep
        CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) &&
                    !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1),
                "a rollback past the snapshots of single-token ubatches was accepted");

        // Repeat the load into a context that already has its own rollback state:
        // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
        // non-zero at load time. The restore must wipe that state and still match.
        llama_context_ptr ctx_dirty = make_rs_ctx();
        if (!ctx_dirty) {
            return case_result::fail();
        }

        llama_tokens noise = tokens;
        for (auto & tok : noise) {
            tok = (tok + 1) % n_vocab;
            if (tok < 0) {
                tok = 0;
            }
        }
        if (!decode_tokens(ctx_dirty.get(), noise)) {
            TLOG_ERR("%s: failed to decode the dirty prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: dirty rollback failed\n", __func__);
            return case_result::fail();
        }

        ckpt.load_tgt(ctx_dirty.get(), 0, 0);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
            if (logits_dirty == nullptr) {
                TLOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(logits_src_replay[i].data(), logits_dirty, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "dirty-ctx logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        TLOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);

        return case_result::pass();
    }
};

// after a ubatch of n tokens, probe rollbacks from deepest to shallowest: every rollback past the
// snapshots that ubatch guarantees is refused and the deepest guaranteed one is accepted
struct case_rollback_bound : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "rollback_bound"; }

    const char * description() const override {
        return "seq_rm accepts exactly the rollbacks whose snapshots the last ubatch wrote";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prefix = 16;

        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));
        const llm_arch arch = llama_arch_from_string(arch_name);

        // the DSV4 cache rebuilds snapshot d from the state before the ubatch, so d <= n is exact;
        // the recurrent memory gets snapshots 0..n-1 from ggml_gated_delta_net, only for n <= n_rs_seq + 1
        const bool from_ubatch_start = arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41;

        const auto make_ctx = [&]() {
            ctx_opts opts;
            opts.n_batch   = 64;
            opts.n_ubatch  = 64;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx = make_ctx();
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        for (const uint32_t n : { 1u, 2u, n_rs_seq + 1, n_rs_seq + 2 }) {
            const uint32_t r_max = from_ubatch_start ? std::min(n, n_rs_seq) : (n <= n_rs_seq + 1 ? n - 1 : 0);

            ctx = make_ctx();
            if (!ctx) {
                return case_result::fail();
            }

            const llama_pos pos_end = n_prefix + (llama_pos) n;

            CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, 0, 0, n_prefix) &&
                        decode_gen(ctx.get(), n_vocab, 0, n_prefix, pos_end),
                    "the prefix / ubatch decode failed");

            for (uint32_t r = n_rs_seq + 1; r > r_max; --r) {
                CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r, -1),
                        "rollback " + std::to_string(r) + " after a " + std::to_string(n) + "-token ubatch was accepted");
            }

            if (r_max > 0) {
                CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r_max, -1),
                        "rollback " + std::to_string(r_max) + " after a " + std::to_string(n) + "-token ubatch was refused");
            }
        }

        TLOG_INF("%s: rollback bounds hold for ubatches of 1, 2, %u and %u tokens\n", __func__, n_rs_seq + 1, n_rs_seq + 2);

        return case_result::pass();
    }
};

// decode a prompt into seq 0, share its cells with a second sequence, then keep decoding both
struct case_shared_seq_reserve : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "shared"; }

    const char * description() const override {
        return "sequences sharing cells keep decoding correctly after a seq_cp";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the shared-seq batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_seqs     = 2;
        constexpr uint32_t n_prompt   = 128;
        constexpr uint32_t n_continue = 32;

        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_batch    = 256;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = n_seqs;
        opts.kv_unified = 1; // only a unified cache shares cells on seq_cp

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        if (!decode_gen(ctx.get(), n_vocab, 0, 0, (llama_pos) n_prompt)) {
            TLOG_ERR("%s: prompt decode failed\n", __func__);
            return case_result::fail();
        }

        // this is what llama-batched-bench does for -pps
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

        for (uint32_t i = 0; i < n_continue; ++i) {
            const llama_pos pos = (llama_pos) (n_prompt + i);

            common_batch batch(ctx.get());
            for (uint32_t s = 0; s < n_seqs; ++s) {
                batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, true);
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "shared-seq decode failed at step " + std::to_string(i));

            for (uint32_t s = 0; s < n_seqs; ++s) {
                const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
                if (logits == nullptr) {
                    TLOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                    return case_result::fail();
                }

                bool finite = true;
                for (int t_ = 0; t_ < n_vocab; ++t_) {
                    finite = finite && std::isfinite(logits[t_]);
                }

                CTX_REQUIRE(finite, "non-finite shared-seq logit at step " + std::to_string(i) + ", seq " + std::to_string(s));
            }
        }

        TLOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);

        return case_result::pass();
    }
};

// numbered device stores (llama_state_seq_get_data_store), which hold llama-server's context checkpoints on the devices
// (--ctx-checkpoints-vram), against the host path that holds them otherwise:
// - a state saved into a store and restored from it continues with the same logits, bit for bit, as the state saved and
//   restored over the host; both follow the live continuation
// - the store's host copy is the host blob, byte for byte, also when it is made on another thread while the context
//   decodes (the server's background move to the host)
// - two stores of one sequence keep their own states; a freed store, or a blob of another fill, restores nothing
// the partial variant is the server's (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY: the recurrent state only, the positions
// after the checkpoint removed from the rest)
struct case_state_store : ctx_test {
    explicit case_state_store(bool partial_) : partial(partial_) {}

    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return partial ? "store_p" : "store"; }

    const char * description() const override {
        return partial ? "partial states in numbered device stores restore as from the host, copy to the host, stay apart"
                       : "whole states in numbered device stores restore as from the host, copy to the host, stay apart";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prompt = 24;
        constexpr llama_pos n_cont   = 6;

        const llama_state_seq_flags flags = partial ? LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY : LLAMA_STATE_SEQ_FLAGS_NONE;

        ctx_opts opts;
        opts.n_ctx     = 256;
        opts.n_seq_max = 1;

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        llama_context * lctx = ctx.get();

        using run_logits = std::vector<std::vector<float>>;

        // decode [p0, p1) of seq 0 one token at a time, with the logits after each
        const auto run_seq = [&](llama_pos p0, llama_pos p1, run_logits & out) {
            out.clear();
            for (llama_pos pos = p0; pos < p1; ++pos) {
                std::vector<float> logits;
                if (!decode_one(lctx, gen_token(n_vocab, 0, pos), pos, 0) || !get_current_logits(lctx, logits)) {
                    return false;
                }
                out.push_back(std::move(logits));
            }
            return true;
        };

        const auto bitwise_equal = [](const run_logits & a, const run_logits & b) {
            if (a.size() != b.size()) {
                return false;
            }
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i].size() != b[i].size() || memcmp(a[i].data(), b[i].data(), a[i].size()*sizeof(float)) != 0) {
                    return false;
                }
            }
            return true;
        };

        const auto nmse = [](const run_logits & a, const run_logits & b) {
            logits_diff diff;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                diff.add(a[i].data(), b[i].data(), std::min(a[i].size(), b[i].size()));
            }
            return diff.nmse();
        };

        // as the server restores a checkpoint at p: its state, then the positions from p on removed
        const auto trim = [&](llama_pos p) {
            return llama_memory_seq_rm(llama_get_memory(lctx), 0, p, -1);
        };

        const auto store_save = [&](int32_t store, std::vector<uint8_t> & meta) {
            meta.resize(llama_state_seq_get_size_ext(lctx, 0, flags | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
            return llama_state_seq_get_data_store(lctx, meta.data(), meta.size(), 0, store, flags) == meta.size();
        };

        const auto store_load = [&](int32_t store, const std::vector<uint8_t> & meta) {
            return llama_state_seq_set_data_store(lctx, meta.data(), meta.size(), 0, store, flags) == meta.size();
        };

        const auto store_copy = [&](int32_t store, const std::vector<uint8_t> & meta, std::vector<uint8_t> & out) {
            out.resize(llama_state_seq_store_get_size_host(lctx, 0, store));
            return !out.empty() &&
                llama_state_seq_store_get_data_host(lctx, meta.data(), meta.size(), out.data(), out.size(), 0, store) == out.size();
        };

        CTX_REQUIRE(decode_gen(lctx, n_vocab, 0, 0, n_prompt), "prompt decode failed");

        // the state at the prompt's end, on the host and in store 0
        std::vector<uint8_t> host0;
        std::vector<uint8_t> meta0;
        CTX_REQUIRE(get_seq_state(lctx, 0, flags, host0), "host save failed");
        CTX_REQUIRE(store_save(0, meta0), "store 0 save failed");
        TLOG_INF("%s: host state %zu bytes, store blob %zu bytes, store %zu bytes\n", __func__,
                host0.size(), meta0.size(), llama_state_seq_store_get_size_host(lctx, 0, 0) - meta0.size());

        std::vector<uint8_t> copy0;
        CTX_REQUIRE(store_copy(0, meta0, copy0), "host copy of store 0 failed");
        CTX_REQUIRE(blobs_equal(host0, copy0), "the host copy of store 0 differs from the host state");

        run_logits ref0;
        run_logits from_host;
        run_logits from_store;
        CTX_REQUIRE(run_seq(n_prompt, n_prompt + n_cont, ref0), "reference decode failed");

        CTX_REQUIRE(set_seq_state(lctx, 0, flags, host0) && trim(n_prompt), "host restore failed");
        CTX_REQUIRE(run_seq(n_prompt, n_prompt + n_cont, from_host), "decode after the host restore failed");

        CTX_REQUIRE(store_load(0, meta0) && trim(n_prompt), "store 0 restore failed");
        CTX_REQUIRE(run_seq(n_prompt, n_prompt + n_cont, from_store), "decode after the store restore failed");

        CTX_REQUIRE(bitwise_equal(from_host, from_store), "the store restore continues with other logits than the host restore");
        TLOG_INF("%s: restore vs the live state, nmse = %.3e\n", __func__, nmse(ref0, from_store));
        CTX_REQUIRE(nmse(ref0, from_store) <= NMSE_THRESHOLD, "the restored state does not follow the live one");

        // the state after the continuation into store 1, while store 0 keeps the prompt's end
        std::vector<uint8_t> host1;
        std::vector<uint8_t> meta1;
        CTX_REQUIRE(get_seq_state(lctx, 0, flags, host1), "host save 1 failed");
        CTX_REQUIRE(store_save(1, meta1), "store 1 save failed");

        run_logits ref1;
        CTX_REQUIRE(run_seq(n_prompt + n_cont, n_prompt + 2*n_cont, ref1), "reference decode 1 failed");

        // store 1's host copy on another thread, while the context restores store 0 and decodes from it
        std::vector<uint8_t> copy1(llama_state_seq_store_get_size_host(lctx, 0, 1));
        size_t n_copy1 = 0;
        std::thread th([&]() {
            n_copy1 = llama_state_seq_store_get_data_host(lctx, meta1.data(), meta1.size(), copy1.data(), copy1.size(), 0, 1);
        });

        run_logits again0;
        const bool ok0 = store_load(0, meta0) && trim(n_prompt) && run_seq(n_prompt, n_prompt + n_cont, again0);
        th.join();

        CTX_REQUIRE(ok0, "store 0 restore after store 1 was filled failed");
        CTX_REQUIRE(bitwise_equal(again0, from_store), "store 0 changed when store 1 was filled");
        CTX_REQUIRE(n_copy1 > 0 && n_copy1 == copy1.size(), "host copy of store 1 on another thread failed");
        CTX_REQUIRE(blobs_equal(host1, copy1), "the host copy of store 1 differs from the host state");

        run_logits from1;
        CTX_REQUIRE(store_load(1, meta1) && trim(n_prompt + n_cont), "store 1 restore failed");
        CTX_REQUIRE(run_seq(n_prompt + n_cont, n_prompt + 2*n_cont, from1), "decode after the store 1 restore failed");
        CTX_REQUIRE(nmse(ref1, from1) <= NMSE_THRESHOLD, "store 1 does not follow the live state");

        // the host copy restores like the host state (the server's checkpoint after its move to the host)
        run_logits from_copy;
        CTX_REQUIRE(set_seq_state(lctx, 0, flags, copy0) && trim(n_prompt), "restore of the host copy failed");
        CTX_REQUIRE(run_seq(n_prompt, n_prompt + n_cont, from_copy), "decode after the host copy restore failed");
        CTX_REQUIRE(bitwise_equal(from_copy, from_host), "the host copy continues with other logits than the host state");

        // a blob of another fill, or a freed store, restores nothing
        CTX_REQUIRE(!store_load(1, meta0), "store 1 accepted the blob of store 0");
        llama_state_seq_store_free(lctx, 0, 0);
        CTX_REQUIRE(llama_state_seq_store_get_size_host(lctx, 0, 0) == 0, "a freed store still has a size");
        CTX_REQUIRE(!store_load(0, meta0), "a freed store restored a state");

        return case_result::pass();
    }

private:
    bool partial;
};

// a recurrent state is cleared for a new sequence; a cell that holds NaN must not stay NaN after the clear
struct case_nan_clear : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "nan_clear"; }

    const char * description() const override {
        return "a new sequence in a cache filled with NaN decodes finite logits";
    }

    // the case sets its own fill
    bool uses_fill() const override { return false; }

    std::string skip_reason(model_run & mr) const override {
        if (!llama_model_is_recurrent(mr.model) || llama_model_is_hybrid(mr.model)) {
            return "a NaN fill would also reach the attention cache";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        ctx_opts opts;
        opts.fill = 0xff; // every float is NaN

        llama_context_ptr ctx = make_ctx(mr.model, mr.params, opts);
        if (!ctx) {
            return case_result::fail();
        }

        CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, 0, 0, 8), "decode failed");
        CTX_REQUIRE(decode_one(ctx.get(), gen_token(n_vocab, 0, 8), 8), "decode failed");

        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        CTX_REQUIRE(logits != nullptr, "missing logits");

        for (int i = 0; i < n_vocab; ++i) {
            CTX_REQUIRE(std::isfinite(logits[i]), "non-finite logit after the state was cleared");
        }

        return case_result::pass();
    }
};

// the shape of a served agent slot with an MTP draft (Q1158): one sequence whose context keeps rollback snapshots for
// the draft's verify batches, and which goes back to a checkpoint of its prompt prefix. With one sequence the slot
// finder has no second cell to trade places with, so it must never move a cell: a moved sequence keeps its final
// state but loses its rollback snapshots (llama.cpp #31656)
struct case_one_slot_no_swap : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return "one_slot"; }

    const char * description() const override {
        return "one sequence with rollback snapshots: verify batches, rollbacks and a prefix restore move no cell";
    }

    std::string skip_reason(model_run & mr) const override {
        if (!llama_model_is_recurrent(mr.model) && !llama_model_is_hybrid(mr.model)) {
            return "no recurrent state";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prefix = 16;
        constexpr uint32_t  n_draft  = 3; // drafted tokens per verify batch

        ctx_opts opts;
        opts.n_batch   = 64;
        opts.n_ubatch  = 64;
        opts.n_seq_max = 1;
        opts.n_rs_seq  = n_draft;

        llama_context_ptr ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }
        if (llama_n_rs_seq(ctx.get()) < n_draft) {
            return case_result::skip("n_rs_seq is disabled");
        }

        llama_memory_t mem = llama_get_memory(ctx.get());

        // the prompt prefix and its checkpoint, as the server takes it (the recurrent state only)
        CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, 0, 0, n_prefix), "the prefix decode failed");

        std::vector<uint8_t> ckpt;
        CTX_REQUIRE(get_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, ckpt), "the checkpoint was not saved");

        // verify batches of the sampled token and n_draft drafted ones, of which the last n_reject are rejected
        const auto verify_steps = [&](llama_pos pos) -> std::string {
            for (const uint32_t n_reject : { 0u, 1u, n_draft, 2u, 0u, n_draft }) {
                const llama_pos pos_end = pos + 1 + (llama_pos) n_draft;
                if (!decode_gen(ctx.get(), n_vocab, 0, pos, pos_end)) {
                    return "a verify batch failed at position " + std::to_string(pos);
                }
                if (n_reject > 0 && !llama_memory_seq_rm(mem, 0, pos_end - (llama_pos) n_reject, -1)) {
                    return "a rollback of " + std::to_string(n_reject) + " tokens was refused at position " + std::to_string(pos);
                }
                pos = pos_end - (llama_pos) n_reject;
            }
            return "";
        };

        std::string err = verify_steps(n_prefix);
        CTX_REQUIRE(err.empty(), err);

        // back to the prefix, as the server restores a checkpoint: its state, then the positions after it removed
        CTX_REQUIRE(set_seq_state(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, ckpt), "the checkpoint was not restored");
        CTX_REQUIRE(llama_memory_seq_rm(mem, 0, n_prefix, -1), "the positions after the checkpoint were not removed");

        err = verify_steps(n_prefix);
        CTX_REQUIRE(err.empty(), err);

        const uint64_t n_swaps = llama_memory_recurrent_n_swaps(ctx.get());
        CTX_REQUIRE(n_swaps == 0, "the slot finder moved " + std::to_string(n_swaps) + " cells of a single sequence");

        return case_result::pass();
    }
};

// C217: a state saved into a device store inside a decode call (llama_state_seq_store_at), at the end of the micro-batch
// that ends where a call would have been cut, is the state of the calls cut there, byte for byte: the call cuts its
// micro-batches as those calls are cut (llama_set_ubatch_cuts, n_keep_tail of the rollback snapshots too), so it computes
// what they compute. Prompt [0, 60) in micro-batches of 16; the cut calls end at 18 (with rollback snapshots for 3 drafted
// tokens: 14 + 4 tokens, the last 4 kept together) and 34
struct case_store_at : ctx_test {
    explicit case_store_at(bool partial_) : partial(partial_) {}

    test_group group() const override { return TEST_GROUP_RS; }

    const char * name() const override { return partial ? "store_at_p" : "store_at"; }

    const char * description() const override {
        return partial ? "a partial state stored inside a call at a cut is the cut calls' state, byte for byte"
                       : "a whole state stored inside a call at a cut is the cut calls' state, byte for byte";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prompt = 60;
        const std::vector<llama_pos> cuts = { 18, 34 };

        const llama_state_seq_flags flags = partial ? LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY : LLAMA_STATE_SEQ_FLAGS_NONE;

        llama_tokens tokens;
        for (llama_pos pos = 0; pos < n_prompt; ++pos) {
            tokens.push_back(gen_token(n_vocab, 0, pos));
        }

        std::string report;

        // without and with rollback snapshots (n_keep_tail = 4 cuts the cut calls' micro-batches differently)
        for (const uint32_t n_rs : { 0u, 3u }) {
            ctx_opts opts;
            opts.n_ctx     = 256;
            opts.n_batch   = 64;
            opts.n_ubatch  = 16;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = n_rs;

            auto ctx_ref = mr.ctx(opts);
            auto ctx_at  = mr.ctx(opts);
            auto ctx_cut = mr.ctx(opts);
            auto ctx_one = mr.ctx(opts);
            if (!ctx_ref || !ctx_at || !ctx_cut || !ctx_one) {
                return case_result::fail();
            }
            if (n_rs > 0 && llama_n_rs_seq(ctx_ref.get()) < n_rs) {
                report += " n_rs_seq " + std::to_string(n_rs) + " off;";
                continue;
            }
            const std::string tag = "n_rs_seq " + std::to_string(n_rs) + ": ";

            const auto pos_range = [](llama_context * ctx) {
                llama_memory_t mem = llama_get_memory(ctx);
                return std::make_pair(llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));
            };

            // the reference: calls cut at each cut, the state and positions after each, the end state and last logits
            std::vector<std::vector<uint8_t>> host;
            std::vector<std::pair<llama_pos, llama_pos>> ranges;
            {
                llama_pos p0 = 0;
                for (const llama_pos c : cuts) {
                    CTX_REQUIRE(decode_seq(ctx_ref.get(), tokens, p0, c, 0, false), tag + "reference decode failed");
                    host.emplace_back();
                    CTX_REQUIRE(get_seq_state(ctx_ref.get(), 0, flags, host.back()), tag + "reference save failed");
                    ranges.push_back(pos_range(ctx_ref.get()));
                    p0 = c;
                }
                CTX_REQUIRE(decode_seq(ctx_ref.get(), tokens, p0, n_prompt, 0, true), tag + "reference decode failed");
            }
            std::vector<uint8_t> host_end;
            std::vector<float>   logits_end;
            CTX_REQUIRE(get_seq_state(ctx_ref.get(), 0, flags, host_end), tag + "reference save failed");
            CTX_REQUIRE(get_current_logits(ctx_ref.get(), logits_end), tag + "reference logits missing");

            // the reference again from its state at the first cut, restored from the host as the server restores a
            // checkpoint (the cells after a restore can sit elsewhere than in the unbroken run, so a restore is compared
            // with a restore)
            std::vector<uint8_t> host_rst;
            std::vector<float>   logits_rst;
            CTX_REQUIRE(set_seq_state(ctx_ref.get(), 0, flags, host[0]) && llama_memory_seq_rm(llama_get_memory(ctx_ref.get()), 0, cuts[0], -1),
                        tag + "the reference's restore failed");
            CTX_REQUIRE(decode_seq(ctx_ref.get(), tokens, cuts[0], cuts[1], 0, false) && decode_seq(ctx_ref.get(), tokens, cuts[1], n_prompt, 0, true),
                        tag + "the reference's decode after its restore failed");
            CTX_REQUIRE(get_seq_state(ctx_ref.get(), 0, flags, host_rst) && get_current_logits(ctx_ref.get(), logits_rst),
                        tag + "the reference's state after its restore was not read");

            const auto same_as = [&](llama_context * ctx, const std::vector<uint8_t> & want_state, const std::vector<float> & want_logits) {
                std::vector<uint8_t> state;
                std::vector<float>   logits;
                return get_seq_state(ctx, 0, flags, state) && get_current_logits(ctx, logits) &&
                    state == want_state && logits.size() == want_logits.size() &&
                    memcmp(logits.data(), want_logits.data(), logits.size()*sizeof(float)) == 0;
            };
            const auto same_end = [&](llama_context * ctx) {
                return same_as(ctx, host_end, logits_end);
            };

            // one call with a store fill at each cut
            llama_context * lctx = ctx_at.get();
            for (size_t k = 0; k < cuts.size(); ++k) {
                llama_state_seq_store_at(lctx, 0, (int32_t) k, (uint32_t) cuts[k], flags);
            }
            CTX_REQUIRE(decode_seq(lctx, tokens, 0, n_prompt, 0, true), tag + "the call with fills failed");

            CTX_REQUIRE(same_end(lctx), tag + "the call with fills ends in another state or other logits than the cut calls");

            std::vector<std::vector<uint8_t>> blobs(cuts.size());
            for (size_t k = 0; k < cuts.size(); ++k) {
                llama_pos pmin = -2;
                llama_pos pmax = -2;
                const size_t n = llama_state_seq_store_at_get(lctx, nullptr, 0, 0, (int32_t) k, &pmin, &pmax);
                CTX_REQUIRE(n > 0, tag + "no fill at " + std::to_string(cuts[k]));
                blobs[k].resize(n);
                CTX_REQUIRE(llama_state_seq_store_at_get(lctx, blobs[k].data(), n, 0, (int32_t) k, nullptr, nullptr) == n, tag + "the blob did not copy");
                CTX_REQUIRE(pmin == ranges[k].first && pmax == ranges[k].second,
                        tag + "positions at " + std::to_string(cuts[k]) + ": " + std::to_string(pmin) + ".." + std::to_string(pmax) +
                        " against " + std::to_string(ranges[k].first) + ".." + std::to_string(ranges[k].second));

                std::vector<uint8_t> copy(llama_state_seq_store_get_size_host(lctx, 0, (int32_t) k));
                CTX_REQUIRE(!copy.empty() &&
                        llama_state_seq_store_get_data_host(lctx, blobs[k].data(), n, copy.data(), copy.size(), 0, (int32_t) k) == copy.size(),
                        tag + "the host copy of the fill at " + std::to_string(cuts[k]) + " failed");
                CTX_REQUIRE(blobs_equal(copy, host[k]), tag + "the fill at " + std::to_string(cuts[k]) + " is not the cut calls' state");
            }

            // the first fill restores, and the cut calls from there end as the reference does
            CTX_REQUIRE(llama_state_seq_set_data_store(lctx, blobs[0].data(), blobs[0].size(), 0, 0, flags) == blobs[0].size() &&
                        llama_memory_seq_rm(llama_get_memory(lctx), 0, cuts[0], -1), tag + "the restore of the first fill failed");
            CTX_REQUIRE(decode_seq(lctx, tokens, cuts[0], cuts[1], 0, false) && decode_seq(lctx, tokens, cuts[1], n_prompt, 0, true),
                        tag + "the decode after the restore failed");
            CTX_REQUIRE(same_as(lctx, host_rst, logits_rst), tag + "the restored fill does not continue as the reference's restored state");

            // a request the next call cannot take leaves no result, and nothing is left for the call after
            CTX_REQUIRE(llama_state_seq_store_at_get(lctx, nullptr, 0, 0, 0, nullptr, nullptr) == 0, tag + "a fill result outlived its call");

            // the cuts alone: the same micro-batches, the same end
            const uint32_t ucuts[] = { (uint32_t) cuts[0], (uint32_t) cuts[1] };
            llama_set_ubatch_cuts(ctx_cut.get(), ucuts, 2);
            CTX_REQUIRE(decode_seq(ctx_cut.get(), tokens, 0, n_prompt, 0, true), tag + "the cut call failed");
            llama_set_ubatch_cuts(ctx_cut.get(), nullptr, 0);
            CTX_REQUIRE(same_end(ctx_cut.get()), tag + "the call with cuts ends in another state than the cut calls");

            // the control: one call without cuts takes other micro-batches (reported, not required)
            CTX_REQUIRE(decode_seq(ctx_one.get(), tokens, 0, n_prompt, 0, true), tag + "the control call failed");
            report += " " + tag + (same_end(ctx_one.get()) ? "an uncut call ends the same;" : "an uncut call ends elsewhere;");
        }

        TLOG_INF("%s:%s\n", name(), report.c_str());

        return case_result::pass();
    }

private:
    bool partial;
};

std::vector<ctx_test_ptr> make_rs_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_rollback>());
    cases.push_back(std::make_unique<case_rollback_bound>());
    cases.push_back(std::make_unique<case_multi_seq_split_replay>());
    cases.push_back(std::make_unique<case_shared_seq_reserve>());
    cases.push_back(std::make_unique<case_nan_clear>());
    cases.push_back(std::make_unique<case_state_store>(false));
    cases.push_back(std::make_unique<case_state_store>(true));
    cases.push_back(std::make_unique<case_one_slot_no_swap>());
    cases.push_back(std::make_unique<case_store_at>(false));
    cases.push_back(std::make_unique<case_store_at>(true));

    return cases;
}
