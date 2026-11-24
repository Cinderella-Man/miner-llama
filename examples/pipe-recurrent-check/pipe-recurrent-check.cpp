// Checks the rollbacks of pipelined decoding that the executors' slot rotation rarely or never produces, on any model
// (meant for the recurrent state groups of hybrid models, llama_memory_recurrent::n_grp): greedy decoding of -n tokens
// after the prompt -p, once with llama_process one token at a time, once through three pipe slots with
//   - every third token submitted to the slot of the token before it, so the state before it moves to the stash;
//   - every fourth token followed by a wrong guess, dropped again with llama_memory_seq_rm;
//   - the token in flight dropped as llama-server's pipe_leave does (llama_memory_seq_rm from its position), then
//     submitted again through another slot, or decoded with llama_process, at steps 0, 4, 9 and 6 of every 12.
// Both runs must give the same tokens, the same logits for every token (FNV-1a hashes), the same sequence state
// (llama_state_seq_get_data) and the same logits for four probe tokens decoded afterwards. Exit status 0 when they do.
//
// Usage: llama-pipe-recurrent-check -m <gguf> -p <prompt> [-n 64] [-ngl 99 -fa on -c 2048 ...]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// one token of sequence 0 at pos, with its logits
static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch_ext_ptr b(llama_batch_ext_init(ctx));
    const int32_t idx = llama_batch_ext_add_token(b.get(), 0, tok);
    return idx >= 0 && llama_batch_ext_set_pos(b.get(), idx, &pos) && llama_batch_ext_set_output_logits(b.get(), idx, true) &&
           llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0;
}

static llama_token argmax(const std::vector<float> & logits) {
    return (llama_token) (std::max_element(logits.begin(), logits.end()) - logits.begin());
}

struct check_run {
    std::vector<llama_token> toks;
    std::vector<uint64_t>    hashes;
    uint64_t                 state = 0;
    uint64_t                 probe = 0;
};

int main(int argc, char ** argv) {
    common_log_flush_guard log_flush_guard;
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const int n_gen = params.n_predict > 0 ? params.n_predict : 64;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);

    const std::vector<llama_token> inp = common_tokenize(ctx, params.prompt, true, true);
    const int n_prompt = (int) inp.size();
    if (n_prompt < 4 || n_prompt + n_gen + 8 > (int) llama_n_ctx(ctx)) {
        LOG_ERR("%s: the prompt (%d tokens, at least 4) and -n %d must fit the context (%u)\n", __func__, n_prompt, n_gen,
                llama_n_ctx(ctx));
        return 1;
    }

    std::vector<float> logits(n_vocab);
    auto take = [&](const float * p) {
        std::copy(p, p + n_vocab, logits.begin());
    };

    // the prompt with llama_process in one call (at most n_batch tokens), from an empty memory; leaves the last
    // position's logits in `logits`
    auto prompt = [&]() -> bool {
        llama_memory_clear(mem, true);
        llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
        for (int i = 0; i < n_prompt; i++) {
            const llama_pos pos = i;
            const int32_t   idx = llama_batch_ext_add_token(batch.get(), 0, inp[i]);
            if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &pos) ||
                    !llama_batch_ext_set_output_logits(batch.get(), idx, i == n_prompt - 1)) {
                LOG_ERR("%s: the prompt (%d tokens) does not fit a batch of n_batch %u\n", __func__, n_prompt,
                        llama_n_batch(ctx));
                return false;
            }
        }
        const bool ok = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        if (ok) {
            take(llama_get_logits_ith(ctx, -1));
        }
        return ok;
    };

    // the sequence's state, then four probe tokens after the generated ones (the last generated token is never fed)
    auto finish = [&](check_run & r) -> bool {
        llama_pipe_drain(ctx);
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
        buf.resize(llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0));
        r.state = fnv1a(buf.data(), buf.size());
        std::vector<uint64_t> ph;
        for (int j = 0; j < 4; j++) {
            if (!decode_one(ctx, inp[j], n_prompt + n_gen - 1 + j)) {
                return false;
            }
            ph.push_back(fnv1a(llama_get_logits_ith(ctx, -1), sizeof(float)*n_vocab));
        }
        r.probe = fnv1a(ph.data(), ph.size()*sizeof(uint64_t));
        return true;
    };

    auto fail = [&](const char * what, int i) {
        LOG_ERR("%s: %s at step %d\n", __func__, what, i);
        return 1;
    };

    // plain decoding
    check_run ref;
    if (!prompt()) {
        return fail("prompt decode failed", -1);
    }
    for (int i = 0; i < n_gen; i++) {
        const llama_token tok = argmax(logits);
        ref.toks.push_back(tok);
        ref.hashes.push_back(fnv1a(logits.data(), sizeof(float)*n_vocab));
        if (i + 1 == n_gen) {
            break;
        }
        if (!decode_one(ctx, tok, n_prompt + i)) {
            return fail("plain decode failed", i);
        }
        take(llama_get_logits_ith(ctx, -1));
    }
    if (!finish(ref)) {
        return fail("plain probe failed", n_gen);
    }

    // pipelined, three slots
    const int S = 3;
    if (llama_pipe_init(ctx, S) != 0) {
        return fail("llama_pipe_init failed", -1);
    }
    check_run pr;
    if (!prompt()) {
        return fail("prompt decode failed", -1);
    }
    int prev = 0; // the slot whose state group holds the newest state (a plain decode writes group 0)
    int n_inplace = 0, n_guess = 0, n_leave_pipe = 0, n_leave_decode = 0;
    for (int i = 0; i < n_gen; i++) {
        const llama_token tok = argmax(logits);
        pr.toks.push_back(tok);
        pr.hashes.push_back(fnv1a(logits.data(), sizeof(float)*n_vocab));
        if (i + 1 == n_gen) {
            break;
        }
        const llama_pos pos = n_prompt + i;

        int s = i % 3 == 0 ? prev : (prev + 1) % S;
        n_inplace += s == prev;
        if (llama_pipe_submit(ctx, s, tok, pos, 0) != 0) {
            return fail("llama_pipe_submit failed", i);
        }
        int sg = -1;
        if (i % 4 == 1) {
            sg = (s + 1) % S;
            n_guess++;
            if (llama_pipe_submit(ctx, sg, (tok + 1) % n_vocab, pos + 1, 0) != 0) {
                return fail("llama_pipe_submit of a guess failed", i);
            }
        }

        const int k     = i % 12;
        const int leave = k == 0 || k == 4 || k == 9 ? 1 : k == 6 ? 2 : 0;
        if (leave > 0) {
            // the token in flight goes, with whatever follows it
            if (!llama_memory_seq_rm(mem, 0, pos, -1)) {
                return fail("llama_memory_seq_rm of the token in flight failed", i);
            }
            llama_pipe_drain(ctx);
            sg = -1;
            if (leave == 1) {
                n_leave_pipe++;
                s = (s + 2) % S;
                if (llama_pipe_submit(ctx, s, tok, pos, 0) != 0) {
                    return fail("llama_pipe_submit after dropping failed", i);
                }
            } else {
                n_leave_decode++;
                if (!decode_one(ctx, tok, pos)) {
                    return fail("llama_process after dropping failed", i);
                }
                take(llama_get_logits_ith(ctx, -1));
                prev = 0;
                continue;
            }
        }

        take(llama_pipe_get_logits(ctx, s));
        prev = s;

        if (sg >= 0 && !llama_memory_seq_rm(mem, 0, pos + 1, -1)) {
            return fail("llama_memory_seq_rm of a guess failed", i);
        }
    }
    if (!finish(pr)) {
        return fail("pipelined probe failed", n_gen);
    }

    const bool same_toks   = pr.toks == ref.toks;
    const bool same_logits = pr.hashes == ref.hashes;
    const bool same_state  = pr.state == ref.state;
    const bool same_probe  = pr.probe == ref.probe;
    int first_diff = -1;
    for (int i = 0; i < n_gen && first_diff < 0; i++) {
        if (pr.hashes[i] != ref.hashes[i]) {
            first_diff = i;
        }
    }

    printf("PIPE-RECURRENT-CHECK n_prompt=%d n_gen=%d inplace=%d guesses=%d leave_pipe=%d leave_decode=%d tokens=%s "
           "logits=%s state=%s probe=%s first_logits_diff=%d\n", n_prompt, n_gen, n_inplace, n_guess, n_leave_pipe,
           n_leave_decode, same_toks ? "same" : "DIFF", same_logits ? "same" : "DIFF", same_state ? "same" : "DIFF",
           same_probe ? "same" : "DIFF", first_diff);

    return same_toks && same_logits && same_state && same_probe ? 0 : 2;
}
