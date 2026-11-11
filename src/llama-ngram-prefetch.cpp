#include "llama-ngram-prefetch.h"

#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

// tokens per helper chunk: the claim and madvise granularity
static constexpr int64_t NGRAM_PF_CHUNK = 32;

// the calls kept for llama_ngram_prefetch_calls
static constexpr size_t NGRAM_PF_KEEP = 4096;

static uint64_t ngram_pf_mix(uint64_t x) {
    // splitmix64's finalizer
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static uint64_t ngram_pf_page_size() {
#if defined(_SC_PAGESIZE)
    const long p = sysconf(_SC_PAGESIZE);
    if (p > 0) {
        return (uint64_t) p;
    }
#endif
    return 4096;
}

llama_ngram_prefetch::llama_ngram_prefetch(
        const llama_hparams & hparams, const ggml_tensor * table, bool on, uint64_t cap, bool stats) :
    hparams(hparams), table(table), enabled(on), cap(cap), stats_on(stats),
    n_heads(hparams.ple_n_heads), n_prev((int64_t) hparams.ple_ngram_size - 1), page(ngram_pf_page_size()) {
    GGML_ASSERT(hparams.ple_ngram_size >= 2 && hparams.ple_ngram_size <= LLAMA_MAX_PLE_NGRAM);
    GGML_ASSERT(hparams.ple_n_heads > 0 && hparams.ple_n_heads <= LLAMA_MAX_PLE_HEADS);
    GGML_ASSERT(table != nullptr && ggml_is_matrix(table));

    tot.on    = enabled;
    tot.stats = stats_on;
    tot.cap   = cap;
    tot.page  = page;

    if (enabled) {
        worker = std::thread([this] { helper_main(); });
    }
}

llama_ngram_prefetch::~llama_ngram_prefetch() {
    {
        std::lock_guard<std::mutex> lk(mu);
        quit   = true;
        active = false;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

// the loop of llm_graph_input_qwen4exp_ple::set_input for one token (the n-gram hashes of the token and its
// predecessors); both set_input and the helper call it, so both compute the same rows
void llama_ngram_prefetch::token_rows(const llama_hparams & hparams, llama_token tok, const llama_token * prev, int32_t * out) {
    const int64_t n_gram   = hparams.ple_ngram_size;
    const int64_t per_gram = hparams.ple_heads_per_ngram;
    const int64_t eos      = hparams.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    // an EOS in the window resets everything at or before it
    // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
    // the EOS of the token itself does not cut its own context, as in the reference
    int64_t ctx[LLAMA_MAX_PLE_NGRAM];
    ctx[0] = tok;
    bool cut = false;
    for (int64_t s = 1; s < n_gram; ++s) {
        // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
        const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[n_prev - s];
        cut = cut || t < 0 || t == eos;
        ctx[s] = cut ? eos : t;
    }

    for (int64_t n = 2; n <= n_gram; ++n) {
        uint64_t mixed = (uint64_t) ctx[0] * hparams.ple_layer_multipliers[0];
        for (int64_t j = 1; j < n; ++j) {
            mixed ^= (uint64_t) ctx[j] * hparams.ple_layer_multipliers[j];
        }
        const int64_t base = (n - 2) * per_gram;
        for (int64_t g = 0; g < per_gram; ++g) {
            const int64_t h_i = base + g;
            out[h_i] = (int32_t) (mixed % hparams.ple_head_vocab_sizes[h_i] + hparams.ple_head_offsets[h_i]);
        }
    }
}

uint64_t llama_ngram_prefetch::row_span(int32_t row) const {
    const uint64_t row_bytes = ggml_row_size(table->type, table->ne[0]);
    const uint64_t addr  = (uint64_t) (uintptr_t) table->data + (uint64_t) row * table->nb[1];
    const uint64_t first = addr / page;
    const uint64_t last  = (addr + row_bytes - 1) / page;
    return (last - first + 1) * page;
}

void llama_ngram_prefetch::add_pages(std::vector<uint64_t> & dst, const int32_t * rows, size_t n) const {
    const uint64_t row_bytes = ggml_row_size(table->type, table->ne[0]);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t addr = (uint64_t) (uintptr_t) table->data + (uint64_t) rows[i] * table->nb[1];
        for (uint64_t p = addr / page; p <= (addr + row_bytes - 1) / page; ++p) {
            dst.push_back(p);
        }
    }
}

void llama_ngram_prefetch::issue(const int32_t * rows, size_t n) const {
    if (n > 0) {
        llama_prefetch_rows(table, rows, n);
    }
}

void llama_ngram_prefetch::count_rows(uint64_t & digest, const int32_t * rows, size_t n) const {
    for (size_t i = 0; i < n; ++i) {
        digest += ngram_pf_mix((uint64_t) (uint32_t) rows[i]);
    }
}

static uint64_t ngram_pf_distinct(std::vector<uint64_t> & v) {
    std::sort(v.begin(), v.end());
    return (uint64_t) (std::unique(v.begin(), v.end()) - v.begin());
}

bool llama_ngram_prefetch::begin_call(const llama_batch & batch, uint32_t n_tokens, uint32_t n_first, const std::vector<llama_token> & hist_in) {
    std::unique_lock<std::mutex> lk(mu);

    // the previous call's last chunk must end before the call's buffers change
    active = false;
    cv.wait(lk, [&] { return !busy; });
    if (cur_open) {
        finish_call_locked(lk); // a call that ended on an error path
    }

    chunks.clear();
    win      = 0;
    gave_up  = false;
    next     = 0;
    consumed = 0;
    n_call   = 0;
    ++gen;

    if (stats_on) {
        cur          = {};
        cur.n_tokens = n_tokens;
        cur_open     = true;
        cur_pages.clear();
    }

    bool plan = enabled && attached && n_first > 0 && n_tokens > n_first && batch.token != nullptr && batch.embd == nullptr &&
        batch.pos != nullptr && batch.seq_id != nullptr && batch.n_seq_id != nullptr && (int64_t) hist_in.size() == n_prev;
    for (uint32_t i = 0; plan && i < n_tokens; ++i) {
        plan = batch.n_seq_id[i] == 1 && batch.seq_id[i][0] == batch.seq_id[0][0] && batch.pos[i] == batch.pos[0] + (llama_pos) i;
    }
    if (!plan) {
        return false;
    }

    toks.assign(batch.token, batch.token + n_tokens);
    hist   = hist_in;
    p0     = batch.pos[0];
    seq    = batch.seq_id[0][0];
    n_call = n_tokens;
    start  = n_first;
    rows_h.resize((size_t) n_tokens * n_heads);
    bound_h.resize(n_tokens);
    active = true;

    tot.calls_planned++;
    if (stats_on) {
        cur.planned = true;
    }

    lk.unlock();
    cv.notify_all();

    return true;
}

void llama_ngram_prefetch::end_call() {
    std::unique_lock<std::mutex> lk(mu);
    active = false;
    if (stats_on && cur_open) {
        cv.wait(lk, [&] { return !busy; });
        finish_call_locked(lk);
    }
    lk.unlock();
    cv.notify_all(); // a helper waiting for room in the window goes back to sleep
}

void llama_ngram_prefetch::finish_call_locked(std::unique_lock<std::mutex> & lk) {
    GGML_UNUSED(lk);
    cur_open = false;
    if (cur.n_ubatch == 0) {
        return; // the n-gram input did not run (another context of the model, an encoder pass)
    }
    cur.pages = ngram_pf_distinct(cur_pages);
    cur_pages.clear();
    cur.rows  = cur.rows_helper + cur.rows_inline;

    LLAMA_LOG_WARN("ngram-prefetch: call n_tokens = %u ubatches = %u %s rows = %" PRIu64 " helper = %" PRIu64 " inline = %" PRIu64
            " mismatch = %" PRIu64 " pages = %" PRIu64 " bytes = %" PRIu64 " madv_bytes = %" PRIu64 " digest = %016" PRIx64
            " window_max = %" PRIu64 " cap = %" PRIu64 " helper_us = %" PRId64 " inline_us = %" PRId64 "\n",
            cur.n_tokens, cur.n_ubatch, !enabled ? "off" : cur.planned ? "planned" : "inline", cur.rows, cur.rows_helper,
            cur.rows_inline, cur.rows_mismatch, cur.pages, cur.pages * page, cur.madv_bytes, cur.digest, cur.window_max, cap,
            cur.helper_us, cur.inline_us);

    if (hist_calls.size() >= NGRAM_PF_KEEP) {
        hist_calls.erase(hist_calls.begin());
    }
    hist_calls.push_back(cur);
}

void llama_ngram_prefetch::on_rows(const llama_ubatch & ub, const int32_t * idx, size_t n_idx) {
    const int64_t nt = ub.n_tokens;
    GGML_ASSERT((int64_t) n_idx == nt * n_heads);

    // the rows to issue here: a slice of idx when only the tail is ours, otherwise gathered
    const int32_t * own   = idx;
    size_t          n_own = n_idx;
    std::vector<int32_t> gathered;
    uint64_t n_helper   = 0;
    uint64_t n_mismatch = 0;
    int64_t  b          = -1; // the end of this micro-batch in the planned call

    std::unique_lock<std::mutex> lk(mu);

    bool in_plan = active;
    int64_t a = 0;
    if (in_plan) {
        a = (int64_t) ub.pos[0] - p0;
        in_plan = ub.token != nullptr && !ub.is_mixed() && a == consumed && a + nt <= n_call;
        for (int64_t i = 0; in_plan && i < nt; ++i) {
            in_plan = ub.pos[i] == p0 + (llama_pos) (a + i) && ub.n_seq_id[i] == 1 && ub.seq_id[i][0] == seq &&
                ub.token[i] == toks[a + i];
        }
        if (!in_plan) {
            // not the planned call's next micro-batch: stop the helper, every later row is issued here
            active = false;
            LLAMA_LOG_WARN("%s: a micro-batch outside the planned call (pos %d, %" PRId64 " tokens): the rest is issued inline\n",
                    __func__, ub.pos[0], nt);
        }
    }

    if (in_plan) {
        b = a + nt;
        const int64_t old_next = next;
        const int64_t hb       = std::min(b, std::max(a, old_next)); // [a, hb) claimed by the helper
        next = std::max(next, b);

        // the helper's rows of its tokens against set_input's; a token that differs is issued here as well
        for (int64_t t = a; t < hb; ++t) {
            const int32_t * mine = idx + (t - a) * n_heads;
            if (memcmp(&rows_h[t * n_heads], mine, n_heads * sizeof(int32_t)) != 0) {
                gathered.insert(gathered.end(), mine, mine + n_heads);
                n_mismatch += n_heads;
            }
        }
        n_helper = (uint64_t) (hb - a) * n_heads;

        if (gathered.empty()) {
            own   = idx + (hb - a) * n_heads;
            n_own = (size_t) (b - hb) * n_heads;
        } else {
            gathered.insert(gathered.end(), idx + (hb - a) * n_heads, idx + n_idx);
            own   = gathered.data();
            n_own = gathered.size();
        }
    }
    lk.unlock();

    // this micro-batch's own rows: all of the call's first micro-batch, and what the helper has not claimed
    const int64_t t0 = ggml_time_us();
    issue(own, n_own);
    const int64_t dt = ggml_time_us() - t0;

    last_inline_us   = dt;
    last_rows_inline = n_own;
    last_rows_helper = n_helper;

    std::vector<uint64_t> pages;
    uint64_t digest = 0;
    if (stats_on) {
        add_pages(pages, own, n_own);
        count_rows(digest, own, n_own);
    }

    lk.lock();
    if (b >= 0 && active) {
        // set_input has reached b, once its own reads are issued (so the helper's first chunk, the call's second
        // micro-batch, does not share the disk with the first micro-batch's reads): the helper's chunks that end there
        // leave the window
        consumed = b;
        while (!chunks.empty() && chunks.front().first <= consumed) {
            win -= chunks.front().second;
            chunks.pop_front();
        }
        if (consumed >= n_call) {
            active = false;
        }
    }
    tot.rows_inline   += n_own;
    tot.rows_mismatch += n_mismatch;
    if (stats_on && cur_open) {
        cur.n_ubatch++;
        cur.rows_inline   += n_own;
        cur.rows_mismatch += n_mismatch;
        cur.inline_us     += dt;
        cur.digest        += digest;
        // the bytes passed to madvise: this micro-batch's own distinct pages
        cur_pages.insert(cur_pages.end(), pages.begin(), pages.end());
        cur.madv_bytes += ngram_pf_distinct(pages) * page;
    }
    lk.unlock();
    cv.notify_all();
}

void llama_ngram_prefetch::helper_main() {
    std::vector<llama_token> prev(n_prev);
    std::vector<uint64_t>    pages;

    uint64_t comp_gen = 0; // rows_h holds the rows of tokens [comp_beg, comp_end) of call comp_gen
    int64_t  comp_beg = 0;
    int64_t  comp_end = 0;

    std::unique_lock<std::mutex> lk(mu);
    while (true) {
        cv.wait(lk, [&] { return quit || (active && !gave_up && next < n_call && consumed >= start && next >= start); });
        if (quit) {
            return;
        }

        const uint64_t g  = gen;
        const int64_t  t0 = next;
        const int64_t  t1 = std::min<int64_t>(n_call, t0 + NGRAM_PF_CHUNK);

        if (comp_gen != g || t0 < comp_beg || t0 > comp_end) {
            comp_gen = g;
            comp_beg = comp_end = t0;
        }
        const int64_t c0 = comp_end;

        // the rows of tokens [c0, t1): only this thread writes them, and set_input reads a token's rows only after the
        // helper claimed it below; busy keeps begin_call from changing the call's buffers meanwhile
        busy = true;
        lk.unlock();
        for (int64_t t = c0; t < t1; ++t) {
            for (int64_t s = 1; s <= n_prev; ++s) {
                const int64_t k = t - s; // the predecessor's index in the call; below 0: before the call
                prev[n_prev - s] = k >= 0 ? toks[k] : hist[n_prev + k];
            }
            int32_t * out = &rows_h[t * n_heads];
            token_rows(hparams, toks[t], prev.data(), out);
            uint64_t bound = 0;
            for (int64_t h = 0; h < n_heads; ++h) {
                bound += row_span(out[h]);
            }
            bound_h[t] = bound;
        }
        lk.lock();
        busy     = false;
        comp_end = std::max(comp_end, t1);
        cv.notify_all();

        if (quit) {
            return;
        }
        if (gen != g || !active || next != t0) {
            continue; // the call ended, or set_input took these tokens
        }

        // the tokens that fit in the window
        const uint64_t room = cap > win ? cap - win : 0;
        int64_t  e     = t0;
        uint64_t bound = 0;
        while (e < t1 && bound + bound_h[e] <= room) {
            bound += bound_h[e];
            ++e;
        }
        if (e == t0) {
            if (win == 0) {
                gave_up = true; // one token's rows exceed the cap: set_input issues the rest
                LLAMA_LOG_WARN("%s: a token's rows (%" PRIu64 " bytes) exceed the cap (%" PRIu64 " bytes)\n", __func__, bound_h[t0], cap);
                continue;
            }
            const uint64_t need = bound_h[t0];
            cv.wait(lk, [&] { return quit || gen != g || !active || next != t0 || win + need <= cap; });
            continue;
        }

        next = e;
        win += bound;
        chunks.emplace_back(e, bound);
        tot.window_max = std::max(tot.window_max, win);
        if (stats_on && cur_open) {
            cur.window_max = std::max(cur.window_max, win);
        }
        busy = true;
        lk.unlock();

        const int32_t * rows = &rows_h[t0 * n_heads];
        const size_t    n    = (size_t) (e - t0) * n_heads;

        const int64_t ts = ggml_time_us();
        issue(rows, n);
        const int64_t dt = ggml_time_us() - ts;

        uint64_t digest = 0;
        uint64_t madv   = 0;
        if (stats_on) {
            pages.clear();
            add_pages(pages, rows, n);
            count_rows(digest, rows, n);
            std::vector<uint64_t> own = pages;
            madv = ngram_pf_distinct(own) * page;
        }

        lk.lock();
        busy = false;
        tot.rows_helper += n;
        if (stats_on && cur_open && gen == g) {
            cur.rows_helper += n;
            cur.helper_us   += dt;
            cur.digest      += digest;
            cur.madv_bytes  += madv;
            cur_pages.insert(cur_pages.end(), pages.begin(), pages.end());
        }
        cv.notify_all();
    }
}

llama_ngram_prefetch_info llama_ngram_prefetch::info() const {
    std::lock_guard<std::mutex> lk(mu);
    return tot;
}

std::vector<llama_ngram_prefetch_call> llama_ngram_prefetch::calls() const {
    std::lock_guard<std::mutex> lk(mu);
    return hist_calls;
}
