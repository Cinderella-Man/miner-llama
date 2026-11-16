#pragma once

// LLAMA_NGRAM_PREFETCH (C761): the n-gram table's row prefetch (llama_prefetch_rows, madvise(MADV_WILLNEED) on the rows
// the n-gram input reads from the mapped table) for a decode call's later micro-batches, issued from a helper thread
// while the earlier micro-batches compute, instead of inside the n-gram input's set_input just before card 0.
//
// - The call's first micro-batch issues its rows in set_input as before. Once that is done, the helper takes the call's
//   other tokens in order, a chunk of tokens at a time, computes each token's rows exactly as set_input will (the same
//   function, llama_ngram_prefetch::token_rows), and prefetches them, at most LLAMA_NGRAM_PREFETCH_MB (default 48) MiB
//   of pages ahead of the micro-batches that set_input has reached.
// - set_input still computes its rows and still uploads them: the rows read are the same bytes. For each micro-batch it
//   issues inline only the tokens the helper has not claimed, and the tokens whose helper rows differ from its own (none
//   by construction; counted). The switch changes only when the reads are issued.
// - Planned calls: token batches of one sequence at consecutive positions, more than one micro-batch, in a context
//   whose graph has the n-gram input of a mapped (lazy) table (qwen4exp); anything else issues inline as before.
// - With LLAMA_UBATCH_TRACE or LLAMA_NGRAM_PREFETCH_STATS=1 the rows, pages and digest of every call are counted (with
//   the switch off too, for the inline path) and a "ngram-prefetch:" line is logged per call.

#include "llama.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_tensor;
struct llama_hparams;
struct llama_ubatch;
struct llama_batch;

// one decode call's counts (stats mode)
struct llama_ngram_prefetch_call {
    uint32_t n_tokens      = 0;
    uint32_t n_ubatch      = 0; // micro-batches whose n-gram rows were set
    bool     planned       = false;
    uint64_t rows          = 0; // rows issued, helper and inline (each token's n_heads rows, repeats included)
    uint64_t rows_helper   = 0;
    uint64_t rows_inline   = 0;
    uint64_t rows_mismatch = 0; // rows of tokens whose helper rows differed from set_input's (issued inline as well)
    uint64_t pages         = 0; // the distinct pages of the call's issued rows
    uint64_t madv_bytes    = 0; // the bytes of the ranges passed to madvise (each chunk's and micro-batch's own pages)
    uint64_t digest        = 0; // order-free digest of the issued rows (a sum of mixed row ids)
    uint64_t window_max    = 0; // the most bytes (page spans of the rows) the helper had issued ahead of set_input
    int64_t  helper_us     = 0; // the helper's time in its prefetch calls
    int64_t  inline_us     = 0; // set_input's time in its prefetch calls
};

struct llama_ngram_prefetch_info {
    bool     on    = false;
    bool     stats = false;
    uint64_t cap   = 0;   // bytes
    uint64_t page  = 0;
    uint64_t calls_planned = 0;
    uint64_t rows_helper   = 0;
    uint64_t rows_inline   = 0;
    uint64_t rows_mismatch = 0;
    uint64_t window_max    = 0;
};

class llama_ngram_prefetch {
public:
    // on: the switch; cap: bytes; stats: count every call
    llama_ngram_prefetch(const llama_hparams & hparams, const ggml_tensor * table, bool on, uint64_t cap, bool stats);
    ~llama_ngram_prefetch();

    // the rows of one token (hparams.ple_n_heads of them) from the token and its n_gram - 1 predecessors (oldest first,
    // LLAMA_TOKEN_NULL where missing), as the n-gram input sets them
    static void token_rows(const llama_hparams & hparams, llama_token tok, const llama_token * prev, int32_t * out);

    bool on()    const { return enabled; }
    bool stats() const { return stats_on; }

    // the n-gram input has set rows in this context: calls may be planned from now on
    void attach() { attached = true; }

    // decode, after the memory module split the batch and before the first micro-batch: the call's tokens (one sequence,
    // consecutive positions), the first micro-batch's size and the n_gram - 1 tokens before the call (oldest first,
    // LLAMA_TOKEN_NULL where missing). Returns whether the helper takes part.
    bool begin_call(const llama_batch & batch, uint32_t n_tokens, uint32_t n_first, const std::vector<llama_token> & hist);

    // decode, after the last micro-batch (stats mode: waits for the helper and logs the call's line)
    void end_call();

    // the n-gram input's set_input: the micro-batch's rows (n_heads per token); issues what the helper did not
    void on_rows(const llama_ubatch & ubatch, const int32_t * idx, size_t n_idx);

    // since the context was made
    llama_ngram_prefetch_info info() const;
    std::vector<llama_ngram_prefetch_call> calls() const;

    // the n-gram input's time in its inline prefetch of the last micro-batch, and its rows (inline, helper)
    int64_t  last_inline_us   = 0;
    uint64_t last_rows_inline = 0;
    uint64_t last_rows_helper = 0;

private:
    void helper_main();
    uint64_t row_span(int32_t row) const; // the bytes of the pages the row touches
    void add_pages(std::vector<uint64_t> & dst, const int32_t * rows, size_t n) const;
    void issue(const int32_t * rows, size_t n) const;
    void count_rows(uint64_t & digest, const int32_t * rows, size_t n) const;
    void finish_call_locked(std::unique_lock<std::mutex> & lk);

    const llama_hparams & hparams;
    const ggml_tensor *   table;
    const bool            enabled;
    const uint64_t        cap;
    const bool            stats_on;
    const int64_t         n_heads;
    const int64_t         n_prev;
    const uint64_t        page;
    bool                  attached = false;

    // the current call; tokens, hist and rows_h are sized while the helper is idle and fixed for the call
    std::vector<llama_token> toks;
    std::vector<llama_token> hist;
    std::vector<int32_t>     rows_h;   // the helper's rows, n_heads per token (valid for the tokens it claimed)
    std::vector<uint64_t>    bound_h;  // per token: the page spans of its rows
    llama_pos    p0     = 0;
    llama_seq_id seq    = 0;
    int64_t      n_call = 0;
    int64_t      start  = 0;           // the first token the helper takes (the first micro-batch is set_input's)

    mutable std::mutex      mu;
    std::condition_variable cv;
    std::thread             worker;
    bool     quit     = false;
    bool     active   = false;         // a planned call is running
    bool     busy     = false;         // the helper is between claiming a chunk and the end of its prefetch
    bool     gave_up  = false;         // a token's rows alone exceed the cap
    uint64_t gen      = 0;             // the call's number
    int64_t  next     = 0;             // tokens below are claimed (by the helper or by set_input)
    int64_t  consumed = 0;             // set_input has reached this token
    uint64_t win      = 0;             // bytes claimed by the helper for tokens set_input has not reached
    std::deque<std::pair<int64_t, uint64_t>> chunks; // (end token, bytes) of the helper's chunks in the window

    // the current call's counts (stats mode); the helper's part is written under mu
    llama_ngram_prefetch_call cur;
    bool                      cur_open = false;
    std::vector<uint64_t>     cur_pages;

    // since the context was made
    llama_ngram_prefetch_info               tot;
    std::vector<llama_ngram_prefetch_call>  hist_calls;
};
