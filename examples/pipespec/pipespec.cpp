// Pipelined speculation for one sequence on a layer split.
//
// With the layers split over several devices, plain decoding keeps one device busy at a time. Here a guessed token
// enters the first device while the later devices still work on the current token: up to D guesses are in flight,
// each an ordinary one-token evaluation (llama_pipe_*). When the oldest token's logits arrive, the next token is
// sampled once, in order, exactly as plain decoding samples it, and compared with the guess that is already in flight
// behind it (sample-and-match):
//   - same token: the guess's evaluation is the right one and keeps going;
//   - different token: every guess behind is dropped, its positions are removed from the KV cache, and the sampled
//     token is submitted instead.
// The output is therefore the output of plain decoding with the same sampler and seed.
//
// One run ("job") is set by the --pipe-* flags; --pipe-jobs FILE runs a list of jobs with one model load, one per line
// as key=value pairs that override the flags (keys: name mode guess depth copy_min miss oracle prompt n spec probe chat
// out state_out trace; oracle=@<name> replays the tokens of an earlier job in the list). Each job prints one PIPESPEC line.
//
//   --pipe-mode plain|pipe  plain decoding with llama_process, one token per call (the reference), or pipelined decoding
//   --pipe-depth D          guesses in flight (0 = the pipe API without guesses)
//   --pipe-guess G          none
//                           copy: the token after the longest earlier match of the history's suffix, if that match is
//                             at least --pipe-copy-min tokens long (history includes the guesses in flight)
//                           cont: the copy lookup if it finds a match of at least --pipe-copy-min, else the most
//                             frequent continuation of the longest 1-8-token context in the request so far (prompt
//                             and committed output) plus, with --pipe-table FILE, a table of other outputs; with
//                             --pipe-store FILE (Q161) the store (common_pipe_store) comes after the request's
//                             continuations of 2-token or longer contexts (job key store=0 leaves it out)
//                           ngram-simple: llama.cpp's ngram-simple lookup on the same history (--spec-ngram-simple-size-n/-m)
//                           spec: any llama.cpp drafter, --pipe-spec TYPE (e.g. ngram-map-k), drafted from committed
//                             tokens only and consumed one guess at a time
//                           oracle: the tokens of a reference run (--pipe-oracle FILE from --pipe-out), with a share
//                             --pipe-miss of the guesses made wrong on purpose (deterministic per position)
//                           staircase: the copy lookup if it finds a match of at least --pipe-copy-min, else the
//                             staircase drafter of --pipe-draft FILE (llama_pipe_draft_*), launched right after each
//                             submission so that its guess is ready when a slot opens; with --pipe-backend-sample its
//                             guess is drawn with the sampler's own uniform for that position, else it is the head's
//                             most likely token
//   --pipe-prompt FILE      prompt file (default: -p / -f)
//   --pipe-chat             wrap the prompt as one user message with the model's chat template
//   --pipe-out FILE         one line per generated token: index, token id, FNV-1a hash of the logits it was sampled from
//   --pipe-probe N          after generation, decode N more tokens with llama_process; their logits hash checks every KV
//                           cell that is still visible
//   --pipe-state-out FILE   the sequence's KV state (llama_state_seq_get_data) after generation
//   --pipe-jsonl FILE       append {id, category, prompt_tokens, gen_tokens} (bench/ngram_replay's transcript format, for
//                           its pipeline simulator); --pipe-cat sets the category (default chat)
//   --pipe-trace FILE       one line per sampled token: microseconds since the first submit, guesses in flight, outcome
//                           (h right guess behind it, m wrong, n none, p plain, r a raced sample's token, f the next
//                           token already in flight, fed on the devices)
//   --pipe-backend-greedy   greedy sampling on the GPU (llama_set_sampler) in both modes: only the token id comes back
//                           instead of the logits; the logits hashes then hash the token ids
//   --pipe-backend-sample   the --temp/--top-k/--top-p/--min-p chain on the GPU, one per sequence, each draw keyed by
//                           (seed, position) with llama_sampler_set_pos
//   --pipe-streams N        N conversations staggered through the cards (job key streams): one KV stream each (sequence
//                           c, the context split into N streams), prompts from --pipe-prompts a,b,... (cycled), seed
//                           seed + c; prompts enter as chunks of --pipe-chunk tokens, one token of every generating
//                           conversation stays in flight, and guesses (depth, guess) use only what --pipe-budget leaves;
//                           conversation c joins after --pipe-arrive * c generated tokens. One PIPESPEC line per
//                           conversation (<name>-c<c>), comparable with a plain job of that prompt, seed and chunk size
//   --pipe-chunk C          prompt chunk size (job key chunk): plain mode decodes the prompt in chunks of C tokens too
//   --pipe-chunks K         streams: prompt chunks in flight at once (job key chunks, default 2; the context gets the
//                           largest K of the jobs as chunk slots), several of one prompt if it has them, prompts in
//                           order of arrival
//   --pipe-chunks-gen G     streams: chunks in flight while any conversation generates (job key chunks_gen, default K);
//                           on one stream per card every token queued behind a chunk waits for it on every card
//   --pipe-seed S           the job's base seed (job key seed; default --seed)
//   --pipe-race K           race K samples of the request (sequences 0..K-1 share the prompt's KV, seeds seed..seed+K-1)
//                           until one of them emits --pipe-race-stop (default "</think>"), then continue that one alone
//                           (K = 1: one sample, timed to the stop text the same way);
//                           plain mode decodes the samples round robin, pipe mode keeps one token of each in flight
//                           (needs host sampling; the context gets K sequences and a unified KV cache)
//   --pipe-floor F          (job key floor) pipe mode: when the guesser has nothing, fill the window up to F guesses
//                           with the last token in it (almost always wrong, but idle cards drop to their base clock)
//   --pipe-feed             (job key feed) pipe mode at depth 0 and streams mode without guesses, with a sampler on the
//                           device: a sequence's next token is submitted before its current token is sampled, fed on
//                           the devices from that token's sampled output (llama_pipe_submit_fed; run with
//                           LLAMA_INPUT_ON_GPU=1), so the host stays one token ahead instead of waiting in between
// Staircase drafter options (one drafter per run): --pipe-draft FILE, --pipe-draft-dev N (GPU index; default: the first
// layer's), --pipe-draft-ctx N (committed positions it attends to, 4096), --pipe-draft-rows N (rows per guess, default
// 8 or depth + 2 rounded up to 8), --pipe-draft-alpha A, --pipe-draft-topk K, --pipe-draft-stair L9,L17,... (depths of
// the tokens in flight, newest first; default: from the layer split), --pipe-draft-kind stair|pre (the head starts after
// its device's stage, or with the cycle on a spare device; default: by whether its device holds layers),
// --pipe-draft-top1 (the head's most likely token, no draw).
// Pipe and streams jobs also report the host time per generated token: inside the submit calls (host_submit_us), inside
// the waits for results (host_wait_us) and the rest (host_rest_us, the executor's own work).
// LLAMA_PIPE_UPLOAD=0|1|2 sets how the slots upload their host inputs (llama-context; 2, one packed asynchronous copy
// per card and token, is the default). LLAMA_PIPE_CHUNK_STREAM=1 runs the prompt chunks on a second stream per card,
// below the priority of the tokens' streams, so that tokens overtake the chunks queued before them (llama-context).

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ngram-map.h"
#include "pipe-store.h"
#include "speculative.h"

#include <algorithm>
#include <clocale>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// sampling on the device that computes the logits: greedy (--pipe-backend-greedy) or the temperature chain
// (--pipe-backend-sample)
static bool g_backend_greedy = false;
static bool g_backend_sample = false;

static bool backend_sampling() {
    return g_backend_greedy || g_backend_sample;
}

// the device sampler chain of each sequence (index = sequence id)
static std::vector<llama_sampler *> g_backend_chains;

// slots given to llama_pipe_init_ext: one-token slots first, then the chunk slots
static int g_n_token_slots = 0;
static int g_n_chunk_slots = 0;

// streams: handle the newest complete item first (PIPESPEC_PICK=newest, a test)
static const bool g_pick_newest = getenv("PIPESPEC_PICK") != nullptr && std::string(getenv("PIPESPEC_PICK")) == "newest";

// token file of other outputs for the continuation guesser (--pipe-table), one output per line
static std::string g_table_path;

// the staircase drafter (--pipe-draft*)
static llama_pipe_draft_params g_draft = llama_pipe_draft_default_params();
static std::string g_draft_path;
static std::string g_draft_stair;
static std::string g_draft_kind;
static bool        g_draft_rows_set = false;
static bool        g_draft_top1     = false;

// continuation store for the continuation guesser (--pipe-store)
static std::string         g_store_path;
static common_pipe_store * g_store = nullptr;

// host time in the current job: inside the submit calls (building, uploading and launching an evaluation) and inside
// the waits for results
static int64_t g_t_submit_us = 0;
static int64_t g_t_wait_us   = 0;

struct host_timer {
    int64_t & acc;
    int64_t   t0;
    explicit host_timer(int64_t & a) : acc(a), t0(ggml_time_us()) {}
    ~host_timer() { acc += ggml_time_us() - t0; }
};

struct pipe_job {
    std::string name     = "run";
    std::string mode     = "pipe";
    std::string guess    = "none";
    int         depth    = 0;
    int         copy_min = 4;
    double      miss     = 0.0;
    std::string oracle;            // file written by --pipe-out, or @<job name>
    std::string prompt;            // file; empty = -p / -f
    int         n_predict = 256;
    std::string spec     = "ngram-map-k";
    int         probe    = 0;
    bool        chat     = false;
    std::string out;
    std::string state_out;
    std::string trace;
    int         race     = 0;
    std::string race_stop = "</think>";
    std::string jsonl;
    std::string cat      = "chat";
    int         streams  = 0;
    std::string prompts;           // comma-separated prompt files for streams
    int         chunk    = 0;      // prompt chunk size (0 = n_batch for plain prompts, 64 for streams)
    int64_t     seed     = -1;     // base seed (-1 = --seed)
    int         arrive   = 0;      // streams: conversation c joins after arrive * c generated tokens
    int         budget   = 0;      // streams: items in flight before guesses stop (0 = no limit)
    int         seq      = 0;      // the sequence (KV stream) a plain or pipe job generates in
    bool        feed     = false;  // submit each next token fed from the device (llama_pipe_submit_fed)
    int         floor    = 0;      // pipe mode: guesses kept in flight when the guesser has none (a filler token)
    int         chunks   = 0;      // streams: prompt chunks in flight (0 = 2)
    int         chunks_gen = -1;   // streams: prompt chunks in flight while a conversation generates (-1 = chunks)
    bool        store    = true;   // cont: use the --pipe-store store, if given
};

static bool job_set(pipe_job & j, const std::string & k, const std::string & v) {
    if      (k == "name")      { j.name      = v; }
    else if (k == "mode")      { j.mode      = v; }
    else if (k == "guess")     { j.guess     = v; }
    else if (k == "depth")     { j.depth     = atoi(v.c_str()); }
    else if (k == "copy_min")  { j.copy_min  = atoi(v.c_str()); }
    else if (k == "miss")      { j.miss      = atof(v.c_str()); }
    else if (k == "oracle")    { j.oracle    = v; }
    else if (k == "prompt")    { j.prompt    = v; }
    else if (k == "n")         { j.n_predict = atoi(v.c_str()); }
    else if (k == "spec")      { j.spec      = v; }
    else if (k == "probe")     { j.probe     = atoi(v.c_str()); }
    else if (k == "chat")      { j.chat      = v != "0"; }
    else if (k == "out")       { j.out       = v; }
    else if (k == "state_out") { j.state_out = v; }
    else if (k == "trace")     { j.trace     = v; }
    else if (k == "race")      { j.race      = atoi(v.c_str()); }
    else if (k == "race_stop") { j.race_stop = v; }
    else if (k == "jsonl")     { j.jsonl     = v; }
    else if (k == "cat")       { j.cat       = v; }
    else if (k == "streams")   { j.streams   = atoi(v.c_str()); }
    else if (k == "prompts")   { j.prompts   = v; }
    else if (k == "chunk")     { j.chunk     = atoi(v.c_str()); }
    else if (k == "seed")      { j.seed      = atoll(v.c_str()); }
    else if (k == "arrive")    { j.arrive    = atoi(v.c_str()); }
    else if (k == "budget")    { j.budget    = atoi(v.c_str()); }
    else if (k == "seq")       { j.seq       = atoi(v.c_str()); }
    else if (k == "feed")      { j.feed      = v != "0"; }
    else if (k == "floor")     { j.floor     = atoi(v.c_str()); }
    else if (k == "chunks")    { j.chunks    = atoi(v.c_str()); }
    else if (k == "chunks_gen") { j.chunks_gen = atoi(v.c_str()); }
    else if (k == "store")     { j.store     = v != "0"; }
    else {
        return false;
    }
    return true;
}

// removes the --pipe-* arguments so that the common parser sees only its own
static bool parse_pipe_args(int & argc, char ** argv, pipe_job & def, std::string & jobs_file, int & n_slots) {
    static const std::map<std::string, std::string> keys = {
        {"--pipe-name", "name"}, {"--pipe-mode", "mode"}, {"--pipe-guess", "guess"}, {"--pipe-depth", "depth"},
        {"--pipe-copy-min", "copy_min"}, {"--pipe-miss", "miss"}, {"--pipe-oracle", "oracle"},
        {"--pipe-prompt", "prompt"}, {"--pipe-spec", "spec"}, {"--pipe-probe", "probe"}, {"--pipe-out", "out"},
        {"--pipe-state-out", "state_out"}, {"--pipe-trace", "trace"}, {"--pipe-race", "race"},
        {"--pipe-race-stop", "race_stop"}, {"--pipe-jsonl", "jsonl"}, {"--pipe-cat", "cat"},
        {"--pipe-streams", "streams"}, {"--pipe-prompts", "prompts"}, {"--pipe-chunk", "chunk"}, {"--pipe-seed", "seed"},
        {"--pipe-arrive", "arrive"}, {"--pipe-budget", "budget"}, {"--pipe-floor", "floor"},
        {"--pipe-chunks", "chunks"}, {"--pipe-chunks-gen", "chunks_gen"},
    };
    int j = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a.rfind("--pipe-", 0) != 0) {
            argv[j++] = argv[i];
            continue;
        }
        if (a == "--pipe-chat") {
            def.chat = true;
            continue;
        }
        if (a == "--pipe-feed") {
            def.feed = true;
            continue;
        }
        if (a == "--pipe-backend-greedy") {
            g_backend_greedy = true;
            continue;
        }
        if (a == "--pipe-backend-sample") {
            g_backend_sample = true;
            continue;
        }
        if (a == "--pipe-draft-top1") {
            g_draft_top1 = true;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", a.c_str());
            return false;
        }
        const std::string v = argv[++i];
        if (a == "--pipe-table") {
            g_table_path = v;
        } else if (a == "--pipe-draft") {
            g_draft_path = v;
        } else if (a == "--pipe-draft-dev") {
            g_draft.device = atoi(v.c_str());
        } else if (a == "--pipe-draft-ctx") {
            g_draft.n_ctx = atoi(v.c_str());
        } else if (a == "--pipe-draft-rows") {
            g_draft.n_rows = atoi(v.c_str());
            g_draft_rows_set = true;
        } else if (a == "--pipe-draft-alpha") {
            g_draft.alpha = (float) atof(v.c_str());
        } else if (a == "--pipe-draft-topk") {
            g_draft.top_k = atoi(v.c_str());
        } else if (a == "--pipe-draft-stair") {
            g_draft_stair = v;
        } else if (a == "--pipe-draft-kind") {
            g_draft_kind = v;
        } else if (a == "--pipe-store") {
            g_store_path = v;
        } else if (a == "--pipe-jobs") {
            jobs_file = v;
        } else if (a == "--pipe-slots") {
            n_slots = atoi(v.c_str());
        } else if (keys.count(a)) {
            job_set(def, keys.at(a), v);
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return false;
        }
    }
    argc = j;
    return true;
}

static uint64_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// deterministic per-position coin for the oracle's injected misses
static double unit_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x =  x ^ (x >> 31);
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

// the sampler chain used by both modes, applied to a raw logits row
struct row_sampler {
    llama_sampler * chain = nullptr;
    std::vector<llama_token_data> cur;

    row_sampler(const common_params_sampling & sp, int n_vocab) : cur(n_vocab) {
        chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (sp.temp <= 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temp));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(sp.seed));
        }
    }

    ~row_sampler() { llama_sampler_free(chain); }

    llama_token sample(const float * logits) {
        for (size_t i = 0; i < cur.size(); i++) {
            cur[i] = { (llama_token) i, logits[i], 0.0f };
        }
        llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(chain, &arr);
        GGML_ASSERT(arr.selected >= 0 && arr.selected < (int64_t) arr.size);
        const llama_token id = arr.data[arr.selected].id;
        llama_sampler_accept(chain, id);
        return id;
    }
};

// the token after the longest earlier occurrence of the history's suffix, up to 12 tokens long (the most recent one on
// ties), or -1 if that occurrence is shorter than min_match; the same lookup as bench/ngram_replay's simulator
static llama_token copy_guess(const std::vector<llama_token> & h, int min_match) {
    const int n = (int) h.size();
    if (n < 2) {
        return -1;
    }
    const int max_len = 12;
    int best_len = 0;
    int best_end = -1;
    for (int i = n - 2; i >= 0; --i) {
        if (h[i] != h[n - 1]) {
            continue;
        }
        int len = 1;
        while (len < max_len && i - len >= 0 && h[i - len] == h[n - 1 - len]) {
            len++;
        }
        if (len > best_len) {
            best_len = len;
            best_end = i;
            if (len == max_len) {
                break;
            }
        }
    }
    if (best_len < min_match) {
        return -1;
    }
    return h[best_end + 1];
}

// counts of the token after every 1..8-token context of a token history
struct cont_counts {
    static constexpr int max_ctx = 8;

    std::unordered_map<uint64_t, std::vector<std::pair<llama_token, int32_t>>> counts;
    size_t n_counted = 0; // tokens of the history already counted

    static uint64_t key(const llama_token * p, int n) {
        return fnv1a(p, n*sizeof(llama_token)) ^ (uint64_t) n;
    }

    // count the tokens of h from n_counted on
    void update(const std::vector<llama_token> & h) {
        for (size_t j = std::max<size_t>(n_counted, 1); j < h.size(); j++) {
            for (int ln = 1; ln <= max_ctx && ln <= (int) j; ln++) {
                auto & v = counts[key(&h[j - ln], ln)];
                bool found = false;
                for (auto & e : v) {
                    if (e.first == h[j]) {
                        e.second++;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    v.push_back({ h[j], 1 });
                }
            }
        }
        n_counted = h.size();
    }

    const std::vector<std::pair<llama_token, int32_t>> * get(uint64_t k) const {
        const auto it = counts.find(k);
        return it == counts.end() ? nullptr : &it->second;
    }
};

// the most frequent continuation (summed over the request's counts and the table's; the first seen on ties) of the
// longest context of at least min_len tokens at the end of ctx that either has seen, or -1
static llama_token cont_guess(const cont_counts & req, const cont_counts * table, const std::vector<llama_token> & ctx,
                              int min_len = 1) {
    const int n = (int) ctx.size();
    for (int ln = std::min(cont_counts::max_ctx, n); ln >= min_len; ln--) {
        const uint64_t k = cont_counts::key(&ctx[n - ln], ln);
        const auto * t = table ? table->get(k) : nullptr;
        const auto * r = req.get(k);
        if (!t && !r) {
            continue;
        }
        std::vector<std::pair<llama_token, int32_t>> sum;
        for (const auto * v : { t, r }) {
            if (!v) {
                continue;
            }
            for (const auto & e : *v) {
                bool found = false;
                for (auto & x : sum) {
                    if (x.first == e.first) {
                        x.second += e.second;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    sum.push_back(e);
                }
            }
        }
        llama_token best = -1;
        int32_t     bc   = 0;
        for (const auto & x : sum) {
            if (x.second > bc) {
                best = x.first;
                bc   = x.second;
            }
        }
        return best;
    }
    return -1;
}

// the continuation table of --pipe-table: counts within each line (output) of the file
static const cont_counts * cont_table() {
    static cont_counts table;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        std::ifstream f(g_table_path);
        std::string line;
        size_t n_tok = 0;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::vector<llama_token> toks;
            long t;
            while (ss >> t) {
                toks.push_back((llama_token) t);
            }
            table.n_counted = 0;
            table.update(toks);
            n_tok += toks.size();
        }
        LOG_INF("%s: continuation table of %zu tokens, %zu contexts\n", __func__, n_tok, table.counts.size());
    }
    return g_table_path.empty() ? nullptr : &table;
}

// cont after the copy lookup: with the store (--pipe-store, store=1), the request's continuations of contexts of 2 tokens
// or more, else the store's guess (counted in *n_store), else the request's continuations of any context
static llama_token cont_chain(const pipe_job & job, const cont_counts & req, const std::vector<llama_token> & h,
                              int * n_store) {
    if (g_store && job.store) {
        const llama_token r = cont_guess(req, cont_table(), h, 2);
        if (r >= 0) {
            return r;
        }
        const llama_token s = common_pipe_store_guess(g_store, h.data(), (int) h.size());
        if (s >= 0) {
            (*n_store)++;
            return s;
        }
    }
    return cont_guess(req, cont_table(), h);
}

static std::vector<llama_token> read_tokens(const std::string & path) {
    // the --pipe-out format: "<index> <token> <hash>" per line
    std::vector<llama_token> res;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        long idx; long tok;
        if (ss >> idx >> tok) {
            res.push_back((llama_token) tok);
        }
    }
    return res;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct job_result {
    std::vector<llama_token> gen;    // gen[j] sits at position n_prompt + j
    std::vector<uint64_t>    hashes; // logits hash of each sampled token
};

// one token of one sequence through llama_process, with its logits
static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq) {
    llama_batch_ext_ptr b(llama_batch_ext_init(ctx));
    const int32_t idx = llama_batch_ext_add_token(b.get(), seq, tok);
    return idx >= 0 && llama_batch_ext_set_pos(b.get(), idx, &pos) && llama_batch_ext_set_output_logits(b.get(), idx, true) &&
           llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0;
}

// one device sampler chain per sequence (seeds[i] for sequence i); the other sequences get none
static bool set_backend_samplers(llama_context * ctx, const common_params_sampling & sp, const std::vector<uint32_t> & seeds) {
    const int n_seq = (int) llama_n_seq_max(ctx);
    std::vector<llama_sampler *> old = g_backend_chains;
    g_backend_chains.assign(n_seq, nullptr);
    for (int i = 0; i < n_seq; i++) {
        llama_sampler * chain = nullptr;
        if (i < (int) seeds.size()) {
            chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (g_backend_greedy) {
                llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            } else {
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, 1));
                llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, 1));
                llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temp));
                llama_sampler_chain_add(chain, llama_sampler_init_dist(seeds[i]));
            }
        }
        if (!llama_set_sampler(ctx, i, chain) && chain) {
            LOG_ERR("%s: the sampler chain cannot run on the device\n", __func__);
            return false;
        }
        g_backend_chains[i] = chain;
    }
    for (auto * c : old) {
        if (c) {
            llama_sampler_free(c);
        }
    }
    return true;
}

// before a token of sequence seq at pos is decoded or submitted: its device draw depends on (seed, pos) only
static void backend_pos(llama_seq_id seq, llama_pos pos) {
    if (g_backend_sample && seq < (llama_seq_id) g_backend_chains.size() && g_backend_chains[seq]) {
        llama_sampler_set_pos(g_backend_chains[seq], pos);
    }
}

// a prompt file (or -p) as tokens, in the model's chat template when chat is set
static std::vector<llama_token> make_prompt(const std::string & file, bool chat, const common_params & params,
                                            llama_model * model, llama_context * ctx) {
    std::string prompt = file.empty() ? params.prompt : read_file(file);
    if (chat) {
        auto tmpls = common_chat_templates_init(model, params.chat_template);
        common_chat_templates_inputs inputs;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = prompt;
        inputs.messages = { msg };
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt.str();
    }
    return common_tokenize(ctx, prompt, true, true);
}

static bool run_job(const pipe_job & job, common_params & params, llama_model * model, llama_context * ctx,
                    const std::map<std::string, job_result> & done_jobs, job_result & res) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t      mem     = llama_get_memory(ctx);
    llama_seq_id        seq     = job.seq;  // the sequence that is generated (the race's winner)

    llama_memory_clear(mem, false);

    // the drafter follows this job's sequence (its prompt included) only when it guesses for it
    const bool stair = job.mode == "pipe" && job.guess == "staircase";
    if (stair && (!llama_pipe_draft_ready(ctx) || job.race >= 1 || job.feed)) {
        LOG_ERR("%s: %s: guess staircase needs --pipe-draft, no race and no feed\n", __func__, job.name.c_str());
        return false;
    }
    if (llama_pipe_draft_ready(ctx)) {
        llama_pipe_draft_reset(ctx, stair ? job.seq : -1, 0);
    }

    const uint32_t seed0 = job.seed >= 0 ? (uint32_t) job.seed : params.sampling.seed;
    if (job.seq != 0 && job.race >= 1) {
        LOG_ERR("%s: %s: a race starts in sequence 0\n", __func__, job.name.c_str());
        return false;
    }
    if (job.seq < 0 || job.seq >= (int) llama_n_seq_max(ctx)) {
        LOG_ERR("%s: %s: sequence %d does not exist\n", __func__, job.name.c_str(), job.seq);
        return false;
    }
    if (backend_sampling()) {
        // the job's chain on its sequence
        std::vector<uint32_t> seeds(job.seq + 1, seed0);
        if (!set_backend_samplers(ctx, params.sampling, seeds)) {
            return false;
        }
    }

    const std::vector<llama_token> inp = make_prompt(job.prompt, job.chat, params, model, ctx);
    const int n_prompt  = (int) inp.size();
    const int n_predict = job.n_predict;

    const int n_race = std::max(1, job.race);
    if (n_race > (int) llama_n_seq_max(ctx)) {
        LOG_ERR("%s: %s: race of %d needs as many sequences, the context has %u\n", __func__, job.name.c_str(), n_race,
                llama_n_seq_max(ctx));
        return false;
    }

    if (n_prompt + n_race*(n_predict + job.depth + 8) + job.probe > (int) llama_n_ctx(ctx)) {
        LOG_ERR("%s: %s: prompt (%d) + n_predict (%d) do not fit the context (%d)\n", __func__, job.name.c_str(),
                n_prompt, n_predict, llama_n_ctx(ctx));
        return false;
    }

    // prompt: identical in both modes, in calls of job.chunk tokens (n_batch without one), logits for the last token
    {
        backend_pos(seq, n_prompt - 1);
        const int step = job.chunk > 0 ? job.chunk : (int) llama_n_batch(ctx);
        llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
        for (int i0 = 0; i0 < n_prompt; i0 += step) {
            const int n = std::min(step, n_prompt - i0);
            llama_batch_ext_clear(batch.get());
            bool filled = true;
            for (int i = i0; i < i0 + n && filled; i++) {
                const llama_pos pos = i;
                const int32_t   idx = llama_batch_ext_add_token(batch.get(), seq, inp[i]);
                filled = idx >= 0 && llama_batch_ext_set_pos(batch.get(), idx, &pos) &&
                         llama_batch_ext_set_output_logits(batch.get(), idx, i == n_prompt - 1);
            }
            if (!filled) {
                LOG_ERR("%s: a prompt call of %d tokens does not fit a batch of n_batch %u\n", __func__, n,
                        llama_n_batch(ctx));
                return false;
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: prompt decode failed\n", __func__);
                return false;
            }
        }
    }

    // one sampler per raced sample, seeds seed, seed + 1, ...
    std::vector<std::unique_ptr<row_sampler>> smpls;
    for (int s = 0; s < n_race; s++) {
        common_params_sampling sp = params.sampling;
        sp.seed = seed0 + s;
        smpls.emplace_back(new row_sampler(sp, n_vocab));
    }
    row_sampler * smpl = smpls[0].get();

    auto & gen    = res.gen;
    auto & hashes = res.hashes;
    gen.clear();
    hashes.clear();

    auto take = [&](const float * logits) {
        const llama_token id = smpl->sample(logits);
        gen.push_back(id);
        hashes.push_back(fnv1a(logits, sizeof(float)*n_vocab));
        return id;
    };
    auto take_id = [&](llama_token id) {
        gen.push_back(id);
        hashes.push_back(fnv1a(&id, sizeof(id)));
        return id;
    };

    // stats
    int n_submit = 0, n_guess = 0, n_hit = 0, n_miss = 0, n_wasted = 0, n_store = 0;
    // staircase: guesses and right guesses of the copy lookup and of the drafter
    int n_copy_guess = 0, n_copy_hit = 0, n_draft_guess = 0, n_draft_hit = 0;

    struct trace_row { int64_t t_us; int inflight; char outcome; };
    std::vector<trace_row> trace;

    // the prompt's last evaluations may still run on the cards (llama_process returns before they finish): wait for them
    // here, so that the generation time starts at the prompt's end and not at a varying point inside it
    llama_synchronize(ctx);

    // generation time includes the race; t_race is the time until the race ended
    const int64_t t_gen_start = ggml_time_us();
    double        t_race      = 0.0;
    int           n_fed       = 0;
    g_t_submit_us = 0;
    g_t_wait_us   = 0;

    // race: K samples of the request, one sequence each, until one emits the stop text
    int      race_winner = -1;
    int      race_tokens = 0;
    uint64_t race_hash   = 0;

    if (job.race >= 1) {
        if (backend_sampling()) {
            LOG_ERR("%s: a race needs host sampling\n", __func__);
            return false;
        }

        std::vector<std::vector<llama_token>> rgen(n_race);
        std::vector<std::vector<uint64_t>>    rhash(n_race);

        // the stop text as one token if it is one, otherwise matched on the detokenized output
        const auto        stop_toks = common_tokenize(ctx, job.race_stop, false, true);
        const llama_token stop_id   = stop_toks.size() == 1 ? stop_toks[0] : LLAMA_TOKEN_NULL;
        std::vector<std::string> tail(n_race);

        auto rtake = [&](int s, const float * logits) {
            const llama_token id = smpls[s]->sample(logits);
            rgen[s].push_back(id);
            rhash[s].push_back(fnv1a(logits, sizeof(float)*n_vocab));
            race_tokens++;
            if (!job.trace.empty()) {
                trace.push_back({ ggml_time_us() - t_gen_start, n_race - 1, 'r' });
            }
            return id;
        };
        // true if sample s is done: the stop text, end of generation or the token limit
        auto race_over = [&](int s, llama_token id) {
            if (llama_vocab_is_eog(vocab, id) || (int) rgen[s].size() >= n_predict) {
                return true;
            }
            if (stop_id != LLAMA_TOKEN_NULL) {
                return id == stop_id;
            }
            tail[s] += common_token_to_piece(ctx, id);
            if (tail[s].size() > 256) {
                tail[s].erase(0, tail[s].size() - 256);
            }
            return tail[s].size() >= job.race_stop.size() &&
                   tail[s].compare(tail[s].size() - job.race_stop.size(), std::string::npos, job.race_stop) == 0;
        };

        // every sample starts from the prompt: its cells are shared, and each samples its first token from its logits
        for (int s = 1; s < n_race; s++) {
            llama_memory_seq_cp(mem, 0, s, -1, -1);
        }
        {
            const float * logits = llama_get_logits_ith(ctx, -1);
            for (int s = 0; s < n_race && race_winner < 0; s++) {
                if (race_over(s, rtake(s, logits))) {
                    race_winner = s;
                }
            }
        }

        if (job.mode == "plain") {
            // round robin, one token of one sample per llama_process call
            while (race_winner < 0) {
                for (int s = 0; s < n_race && race_winner < 0; s++) {
                    const llama_pos pos = n_prompt + (llama_pos) rgen[s].size() - 1;
                    if (!decode_one(ctx, rgen[s].back(), pos, s)) {
                        LOG_ERR("%s: decode failed at pos %d of sample %d\n", __func__, pos, s);
                        return false;
                    }
                    n_submit++;
                    if (race_over(s, rtake(s, llama_get_logits_ith(ctx, -1)))) {
                        race_winner = s;
                    }
                }
            }
        } else if (race_winner < 0) {
            // one token of every sample in flight, staggered through the cards; the same submission order as the
            // round robin above, so the same KV cells
            const int n_slots = llama_pipe_n_slots(ctx);
            if (n_slots < n_race + 1) {
                LOG_ERR("%s: a race of %d needs at least %d slots, the context has %d\n", __func__, n_race, n_race + 1, n_slots);
                return false;
            }
            struct rentry { int slot; int s; };
            std::deque<rentry> fifo;
            int next_slot = 0;
            auto rsubmit = [&](int s) {
                for (;;) {
                    const int sl = next_slot;
                    next_slot = (next_slot + 1) % n_slots;
                    bool used = false;
                    for (const auto & e : fifo) {
                        used = used || e.slot == sl;
                    }
                    if (used) {
                        continue;
                    }
                    const llama_pos pos = n_prompt + (llama_pos) rgen[s].size() - 1;
                    int ret;
                    {
                        host_timer ht(g_t_submit_us);
                        ret = llama_pipe_submit(ctx, sl, rgen[s].back(), pos, s);
                    }
                    if (ret != 0) {
                        LOG_ERR("%s: llama_pipe_submit failed at pos %d of sample %d\n", __func__, pos, s);
                        return false;
                    }
                    fifo.push_back({ sl, s });
                    n_submit++;
                    return true;
                }
            };
            for (int s = 0; s < n_race; s++) {
                if (!rsubmit(s)) {
                    return false;
                }
            }
            while (race_winner < 0) {
                const rentry e = fifo.front();
                fifo.pop_front();
                const float * logits;
                {
                    host_timer ht(g_t_wait_us);
                    logits = llama_pipe_get_logits(ctx, e.slot);
                }
                if (race_over(e.s, rtake(e.s, logits))) {
                    race_winner = e.s;
                    break;
                }
                if (!rsubmit(e.s)) {
                    return false;
                }
            }
            // the other samples' tokens still in flight finish and are ignored
            llama_pipe_drain(ctx);
        }

        // keep the winner, drop the others (their cells, the shared prompt keeps the winner's id)
        for (int s = 0; s < n_race; s++) {
            if (s != race_winner) {
                llama_memory_seq_rm(mem, s, -1, -1);
            }
        }

        std::vector<uint64_t> all;
        for (int s = 0; s < n_race; s++) {
            all.push_back(fnv1a(rgen[s].data(), rgen[s].size()*sizeof(llama_token)));
            all.push_back(fnv1a(rhash[s].data(), rhash[s].size()*sizeof(uint64_t)));
        }
        race_hash = fnv1a(all.data(), all.size()*sizeof(uint64_t));
        t_race    = (ggml_time_us() - t_gen_start)/1e6;

        seq    = race_winner;
        smpl   = smpls[seq].get();
        gen    = rgen[seq];
        hashes = rhash[seq];
    } else if (backend_sampling()) {
        take_id(llama_get_sampled_token_ith(ctx, -1));
    } else {
        take(llama_get_logits_ith(ctx, -1));
    }

    double depth_sum = 0.0;
    int    depth_n   = 0;


    auto done = [&]() {
        return (int) gen.size() >= n_predict || llama_vocab_is_eog(vocab, gen.back());
    };

    if (job.mode == "plain") {
        while (!done()) {
            const llama_pos pos = n_prompt + (int) gen.size() - 1;
            backend_pos(seq, pos);
            if (!decode_one(ctx, gen.back(), pos, seq)) {
                LOG_ERR("%s: decode failed at pos %d\n", __func__, pos);
                return false;
            }
            n_submit++;
            if (backend_sampling()) {
                take_id(llama_get_sampled_token_ith(ctx, -1));
            } else {
                take(llama_get_logits_ith(ctx, -1));
            }
            if (!job.trace.empty()) {
                trace.push_back({ ggml_time_us() - t_gen_start, 0, 'p' });
            }
        }
    } else if (job.mode == "pipe") {
        const int D       = std::max(0, job.depth);
        const int n_slots = llama_pipe_n_slots(ctx);
        if (n_slots < D + 2) {
            LOG_ERR("%s: depth %d needs at least %d slots, the context has %d\n", __func__, D, D + 2, n_slots);
            return false;
        }
        if (job.feed && (D != 0 || !backend_sampling())) {
            LOG_ERR("%s: %s: feed needs depth 0 and a sampler on the device\n", __func__, job.name.c_str());
            return false;
        }

        std::vector<llama_token> oracle;
        if (job.guess == "oracle") {
            if (!job.oracle.empty() && job.oracle[0] == '@') {
                auto it = done_jobs.find(job.oracle.substr(1));
                if (it != done_jobs.end()) {
                    oracle = it->second.gen;
                }
            } else {
                oracle = read_tokens(job.oracle);
            }
            if (oracle.empty()) {
                LOG_ERR("%s: no oracle tokens in '%s'\n", __func__, job.oracle.c_str());
                return false;
            }
        }

        // the drafter keeps one state (its index 0) for whichever sequence is generated
        common_speculative * spec = nullptr;
        if (job.guess == "spec") {
            params.speculative.types = common_speculative_types_from_names({ job.spec });
            spec = common_speculative_init(params.speculative, 1);
            if (!spec) {
                LOG_ERR("%s: failed to initialize the drafter '%s'\n", __func__, job.spec.c_str());
                return false;
            }
        }
        const common_ngram_simple_config simple_cfg = {
            params.speculative.ngram_simple.size_n, params.speculative.ngram_simple.size_m };

        struct entry { int slot; llama_token tok; llama_pos pos; char src; };
        std::deque<entry> win; // win[0]: the newest committed token, in flight; win[1..]: guesses
        int next_slot = 0;
        bool failed = false;
        char guess_src = 0;    // who made the guess being submitted: 'c' the copy lookup, 'd' the drafter

        // src_slot >= 0: the token is the sampled output of that slot, fed on the devices (tok is not known yet)
        auto submit = [&](llama_token tok, llama_pos pos, int src_slot = -1) {
            // round robin: the least recently used slot that is not in the window
            for (;;) {
                const int s = next_slot;
                next_slot = (next_slot + 1) % n_slots;
                bool used = false;
                for (const auto & e : win) {
                    used = used || e.slot == s;
                }
                if (used) {
                    continue;
                }
                backend_pos(seq, pos);
                int ret;
                {
                    host_timer ht(g_t_submit_us);
                    ret = src_slot >= 0 ? llama_pipe_submit_fed(ctx, s, src_slot, pos, seq)
                                        : llama_pipe_submit(ctx, s, tok, pos, seq);
                }
                if (ret != 0) {
                    LOG_ERR("%s: llama_pipe_submit%s failed (%d) at pos %d\n", __func__, src_slot >= 0 ? "_fed" : "", ret, pos);
                    failed = true;
                    return;
                }
                win.push_back({ s, tok, pos, guess_src });
                guess_src = 0;
                n_submit++;
                return;
            }
        };

        // committed history for the guessers: prompt + generated tokens
        std::vector<llama_token> hist(inp);
        hist.insert(hist.end(), gen.begin(), gen.end());

        std::deque<llama_token> chain;  // drafted but not yet submitted (guess "spec")
        int  chain_hits = 0;
        bool chain_live = false;

        if (spec) {
            std::vector<llama_token> prompt_tgt(hist.begin(), hist.end() - 1);
            common_speculative_begin(spec, 0, prompt_tgt);
        }

        // the request's own continuation counts (guess "cont"), over committed tokens only
        cont_counts req_counts;

        // history including the guesses in flight
        auto hist_inflight = [&]() {
            std::vector<llama_token> h(hist);
            for (size_t i = 1; i < win.size(); i++) {
                h.push_back(win[i].tok);
            }
            return h;
        };

        // staircase: the position the drafter's pending launch guesses (-1: none), and the uniform the sampler draws at
        // the position before the guessed one (its output is the guessed token)
        llama_pos draft_pending = -1;
        auto draft_launch = [&]() {
            std::vector<llama_pipe_draft_row> rows;
            for (const auto & e : win) {
                rows.push_back({ e.tok, e.pos, e.slot });
            }
            if (llama_pipe_draft_launch(ctx, rows.data(), (int32_t) rows.size()) == 0) {
                draft_pending = win.back().pos + 1;
            }
        };
        auto draft_u = [&](llama_pos pos_guess) -> float {
            if (g_draft_top1 || params.sampling.temp <= 0.0f || !g_backend_sample || seq >= (llama_seq_id) g_backend_chains.size() ||
                    !g_backend_chains[seq]) {
                return -1.0f;
            }
            return llama_sampler_pos_uniform(g_backend_chains[seq], pos_guess - 1);
        };
        // after a submission: start the drafter on the next position unless the copy lookup has it
        auto draft_ahead = [&]() {
            if (!stair || failed || win.empty() || win.back().pos + 1 >= n_prompt + n_predict) {
                return;
            }
            if (copy_guess(hist_inflight(), job.copy_min) < 0) {
                draft_launch();
            }
        };

        // the next guess, for the position after win.back(), or -1
        auto next_guess = [&]() -> llama_token {
            const llama_pos pos = win.back().pos + 1;
            if (job.guess == "oracle") {
                const int j = pos - n_prompt;
                if (j < 0 || j >= (int) oracle.size()) {
                    return -1;
                }
                llama_token g = oracle[j];
                if (job.miss > 0.0 && unit_hash((uint64_t) j) < job.miss) {
                    g = (g + 1) % n_vocab;
                }
                return g;
            }
            if (job.guess == "copy") {
                return copy_guess(hist_inflight(), job.copy_min);
            }
            if (stair) {
                const llama_token g = copy_guess(hist_inflight(), job.copy_min);
                if (g >= 0) {
                    guess_src = 'c';
                    return g;
                }
                if (draft_pending != pos) {
                    draft_launch();
                }
                draft_pending = -1;
                const llama_token d = llama_pipe_draft_get(ctx, draft_u(pos), nullptr);
                if (d >= 0) {
                    guess_src = 'd';
                }
                return d;
            }
            if (job.guess == "cont") {
                const std::vector<llama_token> h = hist_inflight();
                const llama_token g = copy_guess(h, job.copy_min);
                if (g >= 0) {
                    return g;
                }
                req_counts.update(hist);
                return cont_chain(job, req_counts, h, &n_store);
            }
            if (job.guess == "ngram-simple") {
                std::vector<llama_token> h = hist_inflight();
                const llama_token last = h.back();
                h.pop_back();
                const auto d = common_ngram_simple_draft(simple_cfg, h, last);
                return d.empty() ? -1 : d[0];
            }
            if (job.guess == "spec") {
                if (chain.empty() && win.size() == 1) {
                    if (chain_live) {
                        common_speculative_accept(spec, 0, chain_hits);
                    }
                    std::vector<llama_token> prompt_tgt(hist.begin(), hist.end() - 1);
                    std::vector<llama_token> draft;
                    common_speculative_get_draft_params(spec, 0) = {
                        /* .drafting = */ true,
                        /* .n_max    = */ std::max(1, n_predict - (int) gen.size()),
                        /* .pos0     = */ (llama_pos) (hist.size() - 1),
                        /* .id_last  = */ hist.back(),
                        /* .prompt   = */ &prompt_tgt,
                        /* .result   = */ &draft,
                    };
                    common_speculative_draft(spec);
                    chain.assign(draft.begin(), draft.end());
                    chain_hits = 0;
                    chain_live = !draft.empty();
                }
                if (chain.empty()) {
                    return -1;
                }
                const llama_token g = chain.front();
                chain.pop_front();
                return g;
            }
            return -1;
        };

        submit(gen.back(), n_prompt + (int) gen.size() - 1);
        draft_ahead();

        while (!failed && !done()) {
            // keep up to D guesses in flight behind the newest committed token
            while (!failed && (int) win.size() < D + 1 && win.back().pos + 1 < n_prompt + n_predict) {
                llama_token g = next_guess();
                if (g < 0 && (int) win.size() - 1 < job.floor && win.back().tok >= 0) {
                    // a filler: almost always wrong, but it keeps the cards busy, so their clocks stay up
                    g = win.back().tok;
                }
                if (g < 0) {
                    guess_src = 0;
                    break;
                }
                submit(g, win.back().pos + 1);
                n_guess++;
                draft_ahead();
            }
            if (failed) {
                break;
            }

            const int inflight = (int) win.size() - 1;
            depth_sum += inflight;
            depth_n++;

            // feed: the next token goes in before this one is sampled, fed on the devices from its sampled output
            if (job.feed && win.size() == 1 && win.back().pos + 1 < n_prompt + n_predict) {
                submit(-1, win.back().pos + 1, win.back().slot);
                n_fed++;
                if (failed) {
                    break;
                }
            }

            const entry cur = win.front();
            win.pop_front();
            if (stair) {
                // its output is about to be sampled: its row is a committed one from now on
                llama_pipe_draft_commit(ctx, cur.tok, cur.pos, cur.slot);
            }

            llama_token id;
            if (backend_sampling()) {
                llama_token sid;
                {
                    host_timer ht(g_t_wait_us);
                    sid = llama_pipe_get_sampled(ctx, cur.slot);
                }
                id = take_id(sid);
            } else {
                const float * logits;
                {
                    host_timer ht(g_t_wait_us);
                    logits = llama_pipe_get_logits(ctx, cur.slot);
                }
                id = take(logits);
            }
            hist.push_back(id);

            char outcome = 'n'; // no guess in flight
            const bool fed_next = job.feed && !win.empty();
            if (fed_next) {
                win.front().tok = id; // the fed token is the one just sampled
                outcome = 'f';
            } else if (!win.empty()) {
                outcome = win.front().tok == id ? 'h' : 'm';
            }
            if (!job.trace.empty()) {
                trace.push_back({ ggml_time_us() - t_gen_start, inflight, outcome });
            }

            if (done()) {
                break;
            }

            if (fed_next) {
                continue;
            }

            if (outcome == 'h' || outcome == 'm') {
                const char src = win.front().src;
                n_copy_guess  += src == 'c';
                n_draft_guess += src == 'd';
                n_copy_hit    += src == 'c' && outcome == 'h';
                n_draft_hit   += src == 'd' && outcome == 'h';
            }

            if (outcome == 'h') {
                n_hit++;
                chain_hits++;
                continue;
            }

            if (outcome == 'm') {
                // wrong guess: drop it and everything behind it
                n_miss++;
                draft_pending = -1;
                n_wasted += (int) win.size();
                if (!llama_memory_seq_rm(mem, seq, win.front().pos, -1)) {
                    LOG_ERR("%s: llama_memory_seq_rm failed\n", __func__);
                    failed = true;
                    break;
                }
                win.clear();
                chain.clear();
            }

            submit(id, cur.pos + 1);
            draft_ahead();
        }

        // drop whatever is still in flight beyond the last fed token: plain decoding never fed it
        llama_pipe_drain(ctx);
        llama_memory_seq_rm(mem, seq, n_prompt + (llama_pos) gen.size() - 1, -1);

        if (spec) {
            common_speculative_free(spec);
        }
        if (failed) {
            return false;
        }
    } else {
        LOG_ERR("%s: unknown mode '%s'\n", __func__, job.mode.c_str());
        return false;
    }

    const double t_gen = (ggml_time_us() - t_gen_start)/1e6;

    {
        std::string text;
        for (auto id : gen) {
            text += common_token_to_piece(ctx, id);
        }
        LOG("\n[%s]\n%s\n\n", job.name.c_str(), text.c_str());
    }

    if (!job.out.empty()) {
        FILE * f = fopen(job.out.c_str(), "w");
        for (size_t i = 0; i < gen.size(); i++) {
            fprintf(f, "%zu %d %016" PRIx64 "\n", i, gen[i], hashes[i]);
        }
        fclose(f);
    }

    if (!job.jsonl.empty()) {
        FILE * f = fopen(job.jsonl.c_str(), "a");
        auto list = [&](const std::vector<llama_token> & v) {
            std::string r = "[";
            for (size_t i = 0; i < v.size(); i++) {
                r += (i ? ", " : "") + std::to_string(v[i]);
            }
            return r + "]";
        };
        fprintf(f, "{\"id\": \"%s\", \"category\": \"%s\", \"prompt_tokens\": %s, \"gen_tokens\": %s}\n",
                job.name.c_str(), job.cat.c_str(), list(inp).c_str(), list(gen).c_str());
        fclose(f);
    }

    if (!job.trace.empty()) {
        FILE * f = fopen(job.trace.c_str(), "w");
        fprintf(f, "t_us\tinflight\toutcome\n");
        for (const auto & r : trace) {
            fprintf(f, "%" PRId64 "\t%d\t%c\n", r.t_us, r.inflight, r.outcome);
        }
        fclose(f);
    }

    uint64_t h_state = 0;
    {
        const size_t n = llama_state_seq_get_size(ctx, seq);
        std::vector<uint8_t> buf(n);
        const size_t n_written = llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq);
        h_state = fnv1a(buf.data(), n_written);
        if (!job.state_out.empty()) {
            FILE * f = fopen(job.state_out.c_str(), "wb");
            fwrite(buf.data(), 1, n_written, f);
            fclose(f);
        }
    }

    uint64_t h_probe = 0;
    if (job.probe > 0) {
        // continue with the prompt's first tokens: their logits read every KV cell that is still visible
        std::vector<uint64_t> ph;
        for (int i = 0; i < job.probe; i++) {
            const llama_pos pos = n_prompt + (llama_pos) gen.size() - 1 + i;
            backend_pos(seq, pos);
            if (!decode_one(ctx, inp[i % n_prompt], pos, seq)) {
                LOG_ERR("%s: probe decode failed\n", __func__);
                return false;
            }
            if (backend_sampling()) {
                const llama_token pid = llama_get_sampled_token_ith(ctx, -1);
                ph.push_back(fnv1a(&pid, sizeof(pid)));
            } else {
                ph.push_back(fnv1a(llama_get_logits_ith(ctx, -1), sizeof(float)*n_vocab));
            }
        }
        h_probe = fnv1a(ph.data(), ph.size()*sizeof(uint64_t));
    }

    const int n_gen = (int) gen.size();
    LOG_INF("PIPESPEC name=%s mode=%s guess=%s depth=%d n_prompt=%d n_gen=%d t_gen=%.3f tok_s=%.3f submits=%d guesses=%d "
            "hits=%d misses=%d wasted=%d avg_inflight_guesses=%.3f tokens=%016" PRIx64 " logits=%016" PRIx64 " state=%016" PRIx64
            " probe=%016" PRIx64 " race=%d winner=%d race_tokens=%d t_race=%.3f race_hash=%016" PRIx64
            " fed=%d store_guesses=%d host_submit_us=%.1f host_wait_us=%.1f host_rest_us=%.1f"
            " copy_guesses=%d copy_hits=%d draft_guesses=%d draft_hits=%d\n",
            job.name.c_str(), job.mode.c_str(), job.mode == "plain" ? "-" : job.guess.c_str(), job.mode == "plain" ? 0 : job.depth,
            n_prompt, n_gen, t_gen, (n_gen - 1)/t_gen, n_submit, n_guess, n_hit, n_miss, n_wasted,
            depth_n ? depth_sum/depth_n : 0.0,
            fnv1a(gen.data(), gen.size()*sizeof(llama_token)), fnv1a(hashes.data(), hashes.size()*sizeof(uint64_t)),
            h_state, h_probe, n_race, race_winner, race_tokens, t_race, race_hash, n_fed, n_store,
            n_gen ? (double) g_t_submit_us/n_gen : 0.0, n_gen ? (double) g_t_wait_us/n_gen : 0.0,
            n_gen ? (t_gen*1e6 - g_t_submit_us - g_t_wait_us)/n_gen : 0.0,
            n_copy_guess, n_copy_hit, n_draft_guess, n_draft_hit);

    return true;
}

// N conversations staggered through the cards. Conversation c is sequence c with a KV stream of its own, its own prompt
// (job.prompts, cycled), sampler and seed (seed + c). Work items, in submission order:
//   - a conversation's next token, submitted the moment its previous token is sampled (first);
//   - prompt chunks of job.chunk tokens on the shared chunk slots (a new prompt joins the pipeline instead of draining
//     it through llama_process), prompts in order of arrival, up to job.chunks in flight (job.chunks_gen while a
//     conversation generates), several of one prompt: each card runs a prompt's chunks in submission order;
//   - guesses (job.guess, up to job.depth per conversation), only while fewer than job.budget items are in flight:
//     copy-lookup guesses of any conversation before the other kinds.
// Every card runs its items in submission order, so on one stream per card items complete in that order. With
// LLAMA_PIPE_CHUNK_STREAM=1 the chunks have streams of their own, below the tokens' priority, and a token can complete
// before an earlier chunk: the loop handles the oldest complete item among each conversation's oldest (a conversation's
// items still complete in order), and when none is complete waits for the oldest token or guess, else the oldest chunk.
static bool run_streams(const pipe_job & job, common_params & params, llama_model * model, llama_context * ctx,
                        std::map<std::string, job_result> & done_jobs) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t      mem     = llama_get_memory(ctx);

    const int      N      = job.streams;
    const int      D      = std::max(0, job.depth);
    const int      C      = job.chunk > 0 ? job.chunk : 64;
    const int      P      = 2*(D + 1);                       // one-token slots per conversation
    const uint32_t seed0  = job.seed >= 0 ? (uint32_t) job.seed : params.sampling.seed;
    const int      K      = job.chunks > 0 ? job.chunks : 2;
    const int      K_gen  = job.chunks_gen >= 0 ? std::min(job.chunks_gen, K) : K;
    const int      budget = job.budget > 0 ? job.budget : N*(1 + D) + K;

    if (job.feed && (D != 0 || !backend_sampling())) {
        LOG_ERR("%s: %s: feed needs depth 0 and a sampler on the device\n", __func__, job.name.c_str());
        return false;
    }

    if (N > (int) llama_n_seq_max(ctx) || N*P > g_n_token_slots || g_n_chunk_slots < K) {
        LOG_ERR("%s: %s: %d conversations at depth %d need %d sequences, %d one-token slots and %d chunk slots; the "
                "context has %u, %d and %d\n", __func__, job.name.c_str(), N, D, N, N*P, K, llama_n_seq_max(ctx),
                g_n_token_slots, g_n_chunk_slots);
        return false;
    }

    llama_memory_clear(mem, false);

    std::vector<uint32_t> seeds;
    for (int c = 0; c < N; c++) {
        seeds.push_back(seed0 + c);
    }
    if (backend_sampling() && !set_backend_samplers(ctx, params.sampling, seeds)) {
        return false;
    }

    std::vector<std::string> files;
    {
        std::stringstream ss(job.prompts.empty() ? job.prompt : job.prompts);
        std::string f;
        while (std::getline(ss, f, ',')) {
            files.push_back(f);
        }
        if (files.empty()) {
            files.push_back("");
        }
    }

    struct conv_state {
        llama_seq_id             seq;
        std::vector<llama_token> inp;
        int                      n_prefilled    = 0;     // prompt tokens submitted
        bool                     started        = false; // its prompt may be submitted
        bool                     generating     = false; // the prompt is done
        bool                     done           = false;
        std::vector<llama_token> gen;
        std::vector<uint64_t>    hashes;
        std::vector<llama_token> hist;                   // prompt + committed output, for the guessers
        std::deque<uint64_t>     win;                    // ids of its token in flight and its guesses behind it
        std::unique_ptr<row_sampler> smpl;
        cont_counts              req;
        int                      next_slot = 0;
        int n_submit = 0, n_guess = 0, n_hit = 0, n_miss = 0, n_wasted = 0, n_chunks = 0, n_store = 0;
        int64_t                  t_join = 0, t_first = 0, t_done = 0;
    };

    std::vector<conv_state> cv(N);
    const int n_ctx_seq = (int) llama_n_ctx_seq(ctx);
    for (int c = 0; c < N; c++) {
        cv[c].seq = c;
        cv[c].inp = make_prompt(files[c % files.size()], job.chat, params, model, ctx);
        cv[c].hist = cv[c].inp;
        common_params_sampling sp = params.sampling;
        sp.seed = seeds[c];
        cv[c].smpl.reset(new row_sampler(sp, n_vocab));
        if ((int) cv[c].inp.size() + job.n_predict + D + 8 + job.probe > n_ctx_seq) {
            LOG_ERR("%s: conversation %d: prompt (%zu) + n_predict (%d) do not fit its stream (%d)\n", __func__, c,
                    cv[c].inp.size(), job.n_predict, n_ctx_seq);
            return false;
        }
    }

    struct item {
        uint64_t    id;
        int         slot;
        int         conv;
        char        kind;      // 'c' prompt chunk, 't' a conversation's token, 'g' a guess
        llama_pos   pos;       // of its first token
        int         n_tok;
        llama_token tok;       // the last token
        bool        live;
        bool        last = false; // a prompt's last chunk: its output is the first generated token
    };
    std::deque<item> fifo;
    uint64_t next_id = 1;
    int      n_total = 0;      // generated tokens over all conversations
    int      next_chunk_slot = 0;
    bool     failed = false;

    const int64_t t_start = ggml_time_us();
    int n_fed = 0;
    g_t_submit_us = 0;
    g_t_wait_us   = 0;

    auto slot_in_use = [&](int s) {
        for (const auto & it : fifo) {
            if (it.slot == s) {
                return true;
            }
        }
        return false;
    };

    // submit tokens of conversation c into slot s; src_slot >= 0: one token, the sampled output of that slot, fed on
    // the devices (toks is not read)
    auto submit = [&](int c, int s, char kind, const llama_token * toks, int n, llama_pos pos, int src_slot = -1) {
        auto & v = cv[c];
        backend_pos(v.seq, pos + n - 1);
        int ret;
        {
            host_timer ht(g_t_submit_us);
            ret = src_slot >= 0 ? llama_pipe_submit_fed(ctx, s, src_slot, pos, v.seq)
                                : llama_pipe_submit_tokens(ctx, s, toks, n, pos, v.seq);
        }
        if (ret != 0) {
            LOG_ERR("%s: llama_pipe_submit%s failed (%d) for conversation %d at pos %d\n", __func__,
                    src_slot >= 0 ? "_fed" : "_tokens", ret, c, pos);
            failed = true;
            return (uint64_t) 0;
        }
        const uint64_t id = next_id++;
        fifo.push_back({ id, s, c, kind, pos, n, src_slot >= 0 ? (llama_token) -1 : toks[n - 1], true });
        v.n_submit++;
        return id;
    };

    // a free one-token slot of conversation c (its own range: the slot's graph is reused for the same sequence)
    auto token_slot = [&](int c) {
        auto & v = cv[c];
        for (int k = 0; k < P; k++) {
            const int s = c*P + (v.next_slot + k) % P;
            if (!slot_in_use(s)) {
                v.next_slot = (v.next_slot + k + 1) % P;
                return s;
            }
        }
        return -1;
    };

    auto chunk_slot = [&]() {
        for (int k = 0; k < g_n_chunk_slots; k++) {
            const int s = g_n_token_slots + (next_chunk_slot + k) % g_n_chunk_slots;
            if (!slot_in_use(s)) {
                next_chunk_slot = (next_chunk_slot + k + 1) % g_n_chunk_slots;
                return s;
            }
        }
        return -1;
    };

    auto find = [&](uint64_t id) -> item * {
        for (auto & it : fifo) {
            if (it.id == id) {
                return &it;
            }
        }
        return nullptr;
    };

    // history of conversation c including its guesses in flight
    auto hist_inflight = [&](int c) {
        auto & v = cv[c];
        std::vector<llama_token> h(v.hist);
        for (size_t i = 1; i < v.win.size(); i++) {
            h.push_back(find(v.win[i])->tok);
        }
        return h;
    };

    // the next guess of conversation c; copy_only: only a copy-lookup guess
    auto next_guess = [&](int c, bool copy_only) -> llama_token {
        auto & v = cv[c];
        if (job.guess != "copy" && job.guess != "cont") {
            return -1;
        }
        const std::vector<llama_token> h = hist_inflight(c);
        const llama_token g = copy_guess(h, job.copy_min);
        if (g >= 0 || copy_only || job.guess == "copy") {
            return g;
        }
        v.req.update(v.hist);
        return cont_chain(job, v.req, h, &v.n_store);
    };

    auto conv_done = [&](conv_state & v) {
        return (int) v.gen.size() >= job.n_predict || llama_vocab_is_eog(vocab, v.gen.back());
    };

    auto top_up = [&]() {
        // 1. every generating conversation without a token in flight gets its next one
        for (int c = 0; c < N && !failed; c++) {
            auto & v = cv[c];
            if (v.generating && !v.done && v.win.empty()) {
                const int s = token_slot(c);
                GGML_ASSERT(s >= 0);
                const llama_token t = v.gen.back();
                const uint64_t id = submit(c, s, 't', &t, 1, (llama_pos) (v.inp.size() + v.gen.size() - 1));
                if (id) {
                    v.win.push_back(id);
                }
            }
        }
        // 2. arrivals: conversation c joins after arrive * c generated tokens, or when nothing else is in flight
        for (int c = 0; c < N; c++) {
            if (!cv[c].started && (n_total >= job.arrive*c || fifo.empty())) {
                cv[c].started = true;
                cv[c].t_join  = ggml_time_us();
                if (fifo.empty()) {
                    break;
                }
            }
        }
        // 3. prompt chunks in order of arrival on free chunk slots, several of one prompt: up to K in flight, K_gen while
        //    a conversation generates
        int  n_chunks_inflight = 0;
        bool any_generating    = false;
        for (const auto & it : fifo) {
            n_chunks_inflight += it.kind == 'c';
        }
        for (const auto & v : cv) {
            any_generating = any_generating || (v.generating && !v.done);
        }
        const int n_chunks_max = any_generating ? K_gen : K;
        for (int c = 0; c < N && !failed && n_chunks_inflight < n_chunks_max; c++) {
            auto & v = cv[c];
            while (v.started && v.n_prefilled < (int) v.inp.size() && n_chunks_inflight < n_chunks_max && !failed) {
                const int s = chunk_slot();
                if (s < 0) {
                    break;
                }
                const int n = std::min(C, (int) v.inp.size() - v.n_prefilled);
                if (!submit(c, s, 'c', v.inp.data() + v.n_prefilled, n, v.n_prefilled)) {
                    break;
                }
                v.n_prefilled += n;
                fifo.back().last = v.n_prefilled == (int) v.inp.size();
                v.n_chunks++;
                n_chunks_inflight++;
            }
        }
        // 4. guesses in the slack: copy-lookup guesses first, then the other kinds
        for (int pass = 0; pass < 2 && !failed; pass++) {
            bool progress = true;
            while (progress && (int) fifo.size() < budget && !failed) {
                progress = false;
                for (int c = 0; c < N && (int) fifo.size() < budget; c++) {
                    auto & v = cv[c];
                    if (v.win.empty() || v.done || (int) v.win.size() - 1 >= D) {
                        continue;
                    }
                    const item * last = find(v.win.back());
                    if ((int) (last->pos + 1 - v.inp.size()) >= job.n_predict) {
                        continue;
                    }
                    const llama_token g = next_guess(c, pass == 0);
                    if (g < 0) {
                        continue;
                    }
                    const int s = token_slot(c);
                    if (s < 0) {
                        continue;
                    }
                    const uint64_t id = submit(c, s, 'g', &g, 1, last->pos + 1);
                    if (id) {
                        v.win.push_back(id);
                        v.n_guess++;
                        progress = true;
                    }
                }
            }
        }
    };

    // the sampled token after an item of conversation c
    auto sample = [&](int c, const item & it) {
        auto & v = cv[c];
        llama_token id;
        if (backend_sampling()) {
            {
                host_timer ht(g_t_wait_us);
                id = llama_pipe_get_sampled(ctx, it.slot);
            }
            v.hashes.push_back(fnv1a(&id, sizeof(id)));
        } else {
            const float * logits;
            {
                host_timer ht(g_t_wait_us);
                logits = llama_pipe_get_logits(ctx, it.slot);
            }
            id = v.smpl->sample(logits);
            v.hashes.push_back(fnv1a(logits, sizeof(float)*n_vocab));
        }
        v.gen.push_back(id);
        v.hist.push_back(id);
        n_total++;
        return id;
    };

    auto all_done = [&]() {
        for (const auto & v : cv) {
            if (!v.done) {
                return false;
            }
        }
        return true;
    };

    while (!failed && !all_done()) {
        top_up();
        if (failed) {
            break;
        }
        GGML_ASSERT(!fifo.empty());

        // the item to handle next: the oldest complete one among each conversation's oldest, else the oldest token or
        // guess (the chunks' streams run below the tokens' priority), else the oldest chunk. PIPESPEC_PICK=newest
        // (a test of the bookkeeping on the CPU, where every item is complete) takes the newest complete one instead
        size_t i_next = 0;
        {
            std::vector<bool> seen(N, false);
            size_t i_tok = fifo.size();
            bool   found = false;
            for (size_t i = 0; i < fifo.size() && (!found || g_pick_newest); i++) {
                const item & q = fifo[i];
                if (seen[q.conv]) {
                    continue;
                }
                seen[q.conv] = true;
                if (llama_pipe_ready(ctx, q.slot)) {
                    i_next = i;
                    found  = true;
                } else if (q.kind != 'c' && i_tok == fifo.size()) {
                    i_tok = i;
                }
            }
            if (!found && i_tok < fifo.size()) {
                i_next = i_tok;
            }
        }

        // feed: the next item's conversation gets its next token before that item is sampled, fed on the devices
        // from its sampled output; it enters the cards after every item already in flight, as it would after the wait
        if (job.feed) {
            const item & front = fifo[i_next];
            auto & v = cv[front.conv];
            if (front.live && front.kind == 't' && v.win.size() == 1 && v.win.front() == front.id &&
                    (int) (front.pos + 1 - v.inp.size()) < job.n_predict) {
                const int s = token_slot(front.conv);
                GGML_ASSERT(s >= 0);
                const uint64_t id = submit(front.conv, s, 't', nullptr, 1, front.pos + 1, front.slot);
                if (!id) {
                    break;
                }
                v.win.push_back(id);
                n_fed++;
            }
        }

        const item it = fifo[i_next];
        fifo.erase(fifo.begin() + i_next);
        auto & v = cv[it.conv];

        if (!it.live) {
            host_timer ht(g_t_wait_us);
            llama_pipe_get_logits(ctx, it.slot); // a dropped guess: wait so its slot is free
            continue;
        }

        if (it.kind == 'c') {
            if (!it.last) {
                host_timer ht(g_t_wait_us);
                llama_pipe_get_logits(ctx, it.slot);
                continue;
            }
            // the last chunk: its output is the first generated token
            sample(it.conv, it);
            v.generating = true;
            v.t_first    = ggml_time_us();
            if (conv_done(v)) {
                v.done   = true;
                v.t_done = ggml_time_us();
            }
            continue;
        }

        // the conversation's committed token (a guess that was promoted after hits counts the same)
        GGML_ASSERT(!v.win.empty() && v.win.front() == it.id);
        v.win.pop_front();
        const llama_token id = sample(it.conv, it);

        if (conv_done(v)) {
            for (uint64_t g : v.win) {
                find(g)->live = false;
                v.n_wasted++;
            }
            v.win.clear();
            v.done   = true;
            v.t_done = ggml_time_us();
            continue;
        }

        if (!v.win.empty()) {
            item * nx = find(v.win.front());
            if (job.feed) {
                nx->tok = id; // the fed token is the one just sampled
            } else if (nx->tok == id) {
                v.n_hit++;
            } else {
                v.n_miss++;
                v.n_wasted += (int) v.win.size();
                if (!llama_memory_seq_rm(mem, v.seq, nx->pos, -1)) {
                    LOG_ERR("%s: llama_memory_seq_rm failed\n", __func__);
                    failed = true;
                    break;
                }
                for (uint64_t g : v.win) {
                    find(g)->live = false;
                }
                v.win.clear();
            }
        }
    }

    llama_pipe_drain(ctx);
    const double t_all = (ggml_time_us() - t_start)/1e6;
    if (failed) {
        return false;
    }

    // drop whatever was still in flight beyond each conversation's last fed token
    for (auto & v : cv) {
        llama_memory_seq_rm(mem, v.seq, (llama_pos) (v.inp.size() + v.gen.size() - 1), -1);
    }

    for (int c = 0; c < N; c++) {
        auto & v = cv[c];
        const int n_prompt = (int) v.inp.size();

        uint64_t h_state = 0;
        {
            const size_t n = llama_state_seq_get_size(ctx, v.seq);
            std::vector<uint8_t> buf(n);
            const size_t n_written = llama_state_seq_get_data(ctx, buf.data(), buf.size(), v.seq);
            h_state = fnv1a(buf.data(), n_written);
        }

        uint64_t h_probe = 0;
        if (job.probe > 0) {
            std::vector<uint64_t> ph;
            for (int i = 0; i < job.probe; i++) {
                const llama_pos pos = n_prompt + (llama_pos) v.gen.size() - 1 + i;
                backend_pos(v.seq, pos);
                if (!decode_one(ctx, v.inp[i % n_prompt], pos, v.seq)) {
                    LOG_ERR("%s: probe decode failed\n", __func__);
                    return false;
                }
                if (backend_sampling()) {
                    const llama_token pid = llama_get_sampled_token_ith(ctx, -1);
                    ph.push_back(fnv1a(&pid, sizeof(pid)));
                } else {
                    ph.push_back(fnv1a(llama_get_logits_ith(ctx, -1), sizeof(float)*n_vocab));
                }
            }
            h_probe = fnv1a(ph.data(), ph.size()*sizeof(uint64_t));
        }

        const std::string name = job.name + "-c" + std::to_string(c);
        const int    n_gen = (int) v.gen.size();
        const double t_gen = (v.t_done - v.t_first)/1e6;
        LOG_INF("PIPESPEC name=%s mode=streams guess=%s depth=%d n_prompt=%d n_gen=%d t_gen=%.3f tok_s=%.3f t_first=%.3f "
                "submits=%d chunks=%d guesses=%d hits=%d misses=%d wasted=%d store_guesses=%d tokens=%016" PRIx64
                " logits=%016" PRIx64 " state=%016" PRIx64 " probe=%016" PRIx64 "\n",
                name.c_str(), job.guess.c_str(), D, n_prompt, n_gen, t_gen, t_gen > 0 ? (n_gen - 1)/t_gen : 0.0,
                (v.t_first - v.t_join)/1e6, v.n_submit, v.n_chunks, v.n_guess, v.n_hit, v.n_miss, v.n_wasted, v.n_store,
                fnv1a(v.gen.data(), v.gen.size()*sizeof(llama_token)), fnv1a(v.hashes.data(), v.hashes.size()*sizeof(uint64_t)),
                h_state, h_probe);

        job_result r;
        r.gen    = v.gen;
        r.hashes = v.hashes;
        done_jobs[name] = std::move(r);
    }

    const char * chunk_stream = getenv("LLAMA_PIPE_CHUNK_STREAM");
    LOG_INF("PIPESPEC_STREAMS name=%s n_conv=%d chunk=%d chunks=%d chunks_gen=%d chunk_stream=%d depth=%d budget=%d "
            "total_tokens=%d t_all=%.3f total_tok_s=%.3f fed=%d host_submit_us=%.1f host_wait_us=%.1f host_rest_us=%.1f\n",
            job.name.c_str(), N, C, K, K_gen, chunk_stream ? atoi(chunk_stream) : 0, D, budget, n_total, t_all,
            n_total/t_all, n_fed,
            n_total ? (double) g_t_submit_us/n_total : 0.0, n_total ? (double) g_t_wait_us/n_total : 0.0,
            n_total ? (t_all*1e6 - g_t_submit_us - g_t_wait_us)/n_total : 0.0);

    return true;
}

int main(int argc, char ** argv) {
    common_log_flush_guard log_flush_guard;
    std::setlocale(LC_NUMERIC, "C");

    pipe_job    def;
    std::string jobs_file;
    int         n_slots = 0;
    if (!parse_pipe_args(argc, argv, def, jobs_file, n_slots)) {
        return 1;
    }

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    if (params.n_predict > 0) {
        def.n_predict = params.n_predict;
    }

    std::vector<pipe_job> jobs;
    if (jobs_file.empty()) {
        jobs.push_back(def);
    } else {
        std::ifstream f(jobs_file);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            pipe_job j = def;
            std::istringstream ss(line);
            std::string kv;
            while (ss >> kv) {
                const auto eq = kv.find('=');
                if (eq == std::string::npos || !job_set(j, kv.substr(0, eq), kv.substr(eq + 1))) {
                    fprintf(stderr, "bad job field '%s' in: %s\n", kv.c_str(), line.c_str());
                    return 1;
                }
            }
            jobs.push_back(j);
        }
    }

    int depth_max   = -1;
    int race_max    = 1;
    int streams_max = 0;
    int streams_slots = 0;
    int chunk_max   = 0;
    int chunks_max  = 0;
    for (const auto & j : jobs) {
        if (j.mode == "pipe" || j.streams > 0) {
            depth_max = std::max(depth_max, j.depth);
        }
        race_max = std::max(race_max, j.race);
        if (j.streams > 0) {
            streams_max   = std::max(streams_max, j.streams);
            streams_slots = std::max(streams_slots, j.streams*2*(std::max(0, j.depth) + 1));
            chunk_max     = std::max(chunk_max, j.chunk > 0 ? j.chunk : 64);
            chunks_max    = std::max(chunks_max, j.chunks > 0 ? j.chunks : 2);
        }
    }
    if (race_max > 1 && streams_max > 0) {
        LOG_ERR("%s: races (one KV stream for all samples) and streams (one per conversation) need separate runs\n", __func__);
        return 1;
    }
    if (race_max > 1) {
        // raced samples share the prompt's cells
        params.n_parallel = std::max(params.n_parallel, race_max);
        params.kv_unified = true;
    }
    if (streams_max > 0) {
        // one KV stream per conversation: the context is split into streams_max streams
        params.n_parallel = std::max(params.n_parallel, streams_max);
        params.kv_unified = false;
    }

    if (!g_store_path.empty()) {
        g_store = common_pipe_store_load(g_store_path);
        if (!g_store) {
            LOG_ERR("%s: cannot read the continuation store %s\n", __func__, g_store_path.c_str());
            return 1;
        }
        LOG_INF("continuation store %s: %lld tokens\n", g_store_path.c_str(), (long long) common_pipe_store_size(g_store));
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }

    // the sliding-window cache is a ring of n_swa + n_ubatch cells per sequence, padded to 256 (check_swa_wrap.sh)
    LOG_INF("window cache: n_swa = %d, n_ubatch = %u, n_ctx = %u, n_seq_max = %u\n", llama_model_n_swa(model),
            llama_n_ubatch(ctx), llama_n_ctx(ctx), llama_n_seq_max(ctx));

    if (g_backend_greedy && params.sampling.temp > 0.0f) {
        LOG_ERR("%s: --pipe-backend-greedy needs --temp 0\n", __func__);
        return 1;
    }
    if (g_backend_greedy && g_backend_sample) {
        LOG_ERR("%s: --pipe-backend-greedy or --pipe-backend-sample, not both\n", __func__);
        return 1;
    }
    // a device sampler for sequence 0 before the slots are reserved, so that their buffers include it
    if (backend_sampling() && !set_backend_samplers(ctx, params.sampling, { params.sampling.seed })) {
        return 1;
    }

    if (!g_draft_path.empty()) {
        g_draft.path  = g_draft_path.c_str();
        g_draft.stair = g_draft_stair.empty() ? nullptr : g_draft_stair.c_str();
        g_draft.kind  = g_draft_kind.empty() ? nullptr : g_draft_kind.c_str();
        if (!g_draft_rows_set) {
            // the tokens in flight plus a committed one
            g_draft.n_rows = std::max(8, ((std::max(0, depth_max) + 2 + 7)/8)*8);
        }
        if (llama_pipe_draft_init(ctx, g_draft) != 0) {
            LOG_ERR("%s: llama_pipe_draft_init failed\n", __func__);
            return 1;
        }
    }

    if (depth_max >= 0) {
        g_n_token_slots = n_slots > 0 ? n_slots : std::max({ 2*(depth_max + 1), 2*race_max, streams_slots });
        g_n_chunk_slots = streams_max > 0 ? chunks_max : 0;
        if (llama_pipe_init_ext(ctx, g_n_token_slots, g_n_chunk_slots, chunk_max) != 0) {
            LOG_ERR("%s: llama_pipe_init failed\n", __func__);
            return 1;
        }
    }

    std::map<std::string, job_result> done_jobs;
    int n_failed = 0;
    for (const auto & j : jobs) {
        if (j.streams > 0) {
            if (!run_streams(j, params, model, ctx, done_jobs)) {
                LOG_ERR("%s: job %s failed\n", __func__, j.name.c_str());
                n_failed++;
            }
            continue;
        }
        job_result res;
        if (run_job(j, params, model, ctx, done_jobs, res)) {
            done_jobs[j.name] = std::move(res);
        } else {
            LOG_ERR("%s: job %s failed\n", __func__, j.name.c_str());
            n_failed++;
        }
    }

    llama_backend_free();

    common_pipe_store_free(g_store);

    for (auto * c : g_backend_chains) {
        if (c) {
            llama_sampler_free(c);
        }
    }

    return n_failed == 0 ? 0 : 1;
}
