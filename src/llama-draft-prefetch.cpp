#include "llama-draft-prefetch.h"

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

struct dp_config {
    bool     on    = false; // LLAMA_DRAFT_PREFETCH=1
    uint64_t cap   = 0;     // bytes of pages a window may issue (LLAMA_DRAFT_PREFETCH_KB)
    bool     stats = false; // LLAMA_DRAFT_PREFETCH_STATS=1 or LLAMA_UBATCH_TRACE
};

const dp_config & dp_cfg() {
    static const dp_config cfg = [] {
        dp_config c;
        const char * on = getenv("LLAMA_DRAFT_PREFETCH");
        c.on = on != nullptr && atoi(on) != 0;
        const char * kb = getenv("LLAMA_DRAFT_PREFETCH_KB");
        const long long n_kb = kb != nullptr ? atoll(kb) : 1024;
        c.cap = (uint64_t) std::max(0LL, n_kb) * 1024;
        const char * st = getenv("LLAMA_DRAFT_PREFETCH_STATS");
        c.stats = (st != nullptr && atoi(st) != 0) || llama_ubatch_trace() > 0;
        if (c.on || c.stats) {
            LLAMA_LOG_INFO("llama_draft_prefetch: LLAMA_DRAFT_PREFETCH %s, cap %" PRIu64 " KiB a window, counts %s\n",
                    c.on ? "on" : "off", c.cap / 1024, c.stats ? "on" : "off");
        }
        return c;
    }();
    return cfg;
}

// one context's state: its model's table, the window since the last evaluation that set the table's rows, and the
// totals since the context was made
struct dp_state {
    const llama_model & model;
    const ggml_tensor * table;
    uint64_t page;
    size_t   row_bytes;
    int64_t  n_heads;
    int64_t  n_prev;

    // the window
    std::unordered_set<int32_t>  w_rows;  // rows issued
    std::unordered_set<uint64_t> w_pages; // the pages they span (address / page)
    uint64_t w_calls  = 0;
    uint64_t w_tokens = 0;                // tokens whose rows were issued
    uint64_t w_issued = 0;                // rows issued
    uint64_t w_dup    = 0;                // rows the window had issued already
    uint64_t w_capped = 0;                // tokens not issued: their pages would have passed the cap
    uint64_t w_bytes  = 0;                // page bytes issued
    int64_t  w_us     = 0;                // time in the calls

    // since the context was made
    uint64_t calls     = 0;
    uint64_t tokens    = 0;
    uint64_t issued    = 0;
    uint64_t dup       = 0;
    uint64_t capped    = 0;
    uint64_t bytes     = 0;
    uint64_t bytes_max = 0;               // the most page bytes one window issued
    int64_t  us        = 0;
    uint64_t evals     = 0;               // evaluations whose n-gram input set rows
    uint64_t ev_rows   = 0;
    uint64_t ev_ahead  = 0;               // of them, rows the window had issued
    uint64_t ev_pages  = 0;               // distinct pages per evaluation (stats)
    uint64_t ev_chk    = 0;               // pages checked for residency (stats, evaluations of up to 64 tokens)
    uint64_t ev_res    = 0;               // of them, in the page cache
    uint64_t sm_evals  = 0;               // evaluations of up to 64 tokens (verify passes)
    uint64_t sm_rows   = 0;
    uint64_t sm_ahead  = 0;

    // scratch
    std::vector<llama_token> prev;
    std::vector<int32_t>     rows;
    std::vector<int32_t>     fresh;
    std::vector<uint64_t>    fresh_pages;
    std::vector<int32_t>     issue;
    std::vector<uint64_t>    pages;

    dp_state(const llama_model & model) : model(model), table(model.per_layer_tok_embd) {
#if defined(__linux__)
        const long ps = sysconf(_SC_PAGESIZE);
        page = ps > 0 ? (uint64_t) ps : 4096;
#else
        page = 4096;
#endif
        row_bytes = ggml_row_size(table->type, table->ne[0]);
        n_heads   = model.hparams.ple_n_heads;
        n_prev    = (int64_t) model.hparams.ple_ngram_size - 1;
        prev.resize(std::max<int64_t>(n_prev, 1));
        rows.resize(n_heads);
    }

    // the pages row r of the table spans, first and last
    std::pair<uint64_t, uint64_t> span(int32_t r) const {
        const uintptr_t a = (uintptr_t) table->data + (size_t) r * table->nb[1];
        return { a / page, (a + row_bytes - 1) / page };
    }
};

std::mutex g_mu;
std::map<const llama_kv_cache *, std::unique_ptr<dp_state>> g_states;

// the attention KV cache of a context's memory: its cells hold the n-gram input's predecessor tokens
const llama_kv_cache * dp_attn_kv(const llama_memory_i * mem) {
    if (mem == nullptr) {
        return nullptr;
    }
    if (const auto * hyb = dynamic_cast<const llama_memory_hybrid *>(mem)) {
        return hyb->get_mem_attn();
    }
    return dynamic_cast<const llama_kv_cache *>(mem);
}

// qwen4exp's n-gram table, mapped (lazy): its rows are read from the disk on demand
bool dp_supported(const llama_model & model) {
    return model.arch == LLM_ARCH_QWEN4EXP && model.hparams.ple_n_heads > 0 && model.hparams.ple_ngram_size >= 2 &&
           model.per_layer_tok_embd != nullptr && model.per_layer_tok_embd->data != nullptr &&
           model.can_prefetch.count(model.per_layer_tok_embd) > 0;
}

dp_state * dp_get(const llama_kv_cache * kv, const llama_model * model, bool create) {
    if (kv == nullptr || model == nullptr) {
        return nullptr;
    }
    auto it = g_states.find(kv);
    if (it != g_states.end()) {
        return it->second.get();
    }
    if (!create || !dp_supported(*model)) {
        return nullptr;
    }
    return g_states.emplace(kv, std::make_unique<dp_state>(*model)).first->second.get();
}

// the rows of one token from the token and its n_gram - 1 predecessors (oldest first, LLAMA_TOKEN_NULL where missing):
// the loop body of llm_graph_input_qwen4exp_ple::set_input (src/models/qwen4exp.cpp), line for line
void dp_token_rows(const llama_hparams & hparams, llama_token tok, const llama_token * prev, int32_t * idx) {
    const int64_t n_gram   = hparams.ple_ngram_size;
    const int64_t per_gram = hparams.ple_heads_per_ngram;
    const int64_t eos      = hparams.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    // an EOS in the window resets everything at or before it
    // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
    // the EOS of the token itself does not cut its own context, as in the reference
    std::array<int64_t, LLAMA_MAX_PLE_NGRAM> ctx;
    ctx[0] = tok;
    bool cut = false;
    for (int64_t s = 1; s < n_gram; ++s) {
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
            idx[h_i] = (int32_t) (mixed % hparams.ple_head_vocab_sizes[h_i] + hparams.ple_head_offsets[h_i]);
        }
    }
}

// how many of the pages are in the page cache (mincore), -1 where it cannot tell
int64_t dp_resident(const std::vector<uint64_t> & pages, uint64_t page) {
#if defined(__linux__)
    int64_t n = 0;
    for (const uint64_t p : pages) {
        unsigned char v = 0;
        if (mincore((void *) (uintptr_t) (p * page), page, &v) != 0) {
            return -1;
        }
        n += v & 1;
    }
    return n;
#else
    GGML_UNUSED(pages);
    GGML_UNUSED(page);
    return -1;
#endif
}

} // namespace

bool llama_draft_prefetch_enabled(const struct llama_context * ctx) {
    if (!dp_cfg().on || ctx == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    return dp_get(dp_attn_kv(llama_get_memory(ctx)), llama_get_model(ctx), true) != nullptr;
}

int32_t llama_draft_prefetch(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos0, const llama_token * tokens,
                             int32_t n_tokens, int32_t i_first) {
    const auto & cfg = dp_cfg();
    if (!cfg.on || ctx == nullptr || tokens == nullptr || n_tokens <= 0 || i_first < 0 || i_first >= n_tokens ||
        seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(ctx) || pos0 < 0) {
        return 0;
    }
    const int64_t t0 = ggml_time_us();

    std::lock_guard<std::mutex> lk(g_mu);
    const llama_kv_cache * kv = dp_attn_kv(llama_get_memory(ctx));
    dp_state * st = dp_get(kv, llama_get_model(ctx), true);
    if (st == nullptr) {
        return 0;
    }

    const llama_hparams & hparams = st->model.hparams;
    const llama_kv_cells & cells = kv->get_cells(seq_id);

    st->issue.clear();
    st->w_calls++;

    for (int32_t i = i_first; i < n_tokens; ++i) {
        // the predecessors, oldest first: the call's own tokens from pos0 on, the cells before (as get_prev_tokens)
        for (int64_t j = 0; j < st->n_prev; ++j) {
            const llama_pos p = pos0 + i - (llama_pos) (st->n_prev - j);
            st->prev[j] = p >= pos0 ? tokens[p - pos0] : (p >= 0 ? cells.seq_pos_tok_le(seq_id, p) : LLAMA_TOKEN_NULL);
        }
        dp_token_rows(hparams, tokens[i], st->prev.data(), st->rows.data());

        // the rows the window has not issued, and the pages of theirs it has not
        st->fresh.clear();
        st->fresh_pages.clear();
        for (int64_t h = 0; h < st->n_heads; ++h) {
            const int32_t r = st->rows[h];
            if (st->w_rows.count(r) || std::find(st->fresh.begin(), st->fresh.end(), r) != st->fresh.end()) {
                continue;
            }
            st->fresh.push_back(r);
            const auto [p0, p1] = st->span(r);
            for (uint64_t p = p0; p <= p1; ++p) {
                if (!st->w_pages.count(p) &&
                        std::find(st->fresh_pages.begin(), st->fresh_pages.end(), p) == st->fresh_pages.end()) {
                    st->fresh_pages.push_back(p);
                }
            }
        }

        const uint64_t add = st->fresh_pages.size() * st->page;
        if (st->w_bytes + add > cfg.cap) {
            // this token and the call's later ones stay with the n-gram input
            st->w_capped += n_tokens - i;
            break;
        }

        st->w_rows.insert(st->fresh.begin(), st->fresh.end());
        st->w_pages.insert(st->fresh_pages.begin(), st->fresh_pages.end());
        st->w_bytes  += add;
        st->w_tokens += 1;
        st->w_dup    += (uint64_t) (st->n_heads - (int64_t) st->fresh.size());
        st->issue.insert(st->issue.end(), st->fresh.begin(), st->fresh.end());
    }

    if (!st->issue.empty()) {
        llama_prefetch_rows(st->table, st->issue.data(), st->issue.size());
    }
    st->w_issued += st->issue.size();
    st->w_us     += ggml_time_us() - t0;

    return (int32_t) st->issue.size();
}

void llama_draft_prefetch_on_rows(const llama_kv_cache * kv, const llama_model & model, const int32_t * rows, size_t n_rows,
                                  uint32_t n_tokens) {
    const auto & cfg = dp_cfg();
    if (!cfg.on && !cfg.stats) {
        return;
    }
    const int64_t t0 = ggml_time_us();

    std::lock_guard<std::mutex> lk(g_mu);
    dp_state * st = dp_get(kv, &model, cfg.stats);
    if (st == nullptr) {
        return;
    }

    uint64_t ahead = 0;
    if (!st->w_rows.empty()) {
        for (size_t i = 0; i < n_rows; ++i) {
            ahead += st->w_rows.count(rows[i]);
        }
    }

    st->evals    += 1;
    st->ev_rows  += n_rows;
    st->ev_ahead += ahead;
    if (n_tokens <= 64) {
        st->sm_evals += 1;
        st->sm_rows  += n_rows;
        st->sm_ahead += ahead;
    }

    st->calls  += st->w_calls;
    st->tokens += st->w_tokens;
    st->issued += st->w_issued;
    st->dup    += st->w_dup;
    st->capped += st->w_capped;
    st->bytes  += st->w_bytes;
    st->us     += st->w_us;
    st->bytes_max = std::max(st->bytes_max, st->w_bytes);

    if (cfg.stats) {
        // the evaluation's distinct pages, and for verify-sized ones how many were in the page cache once the input had
        // set its rows (its own madvise ran just before, and a read from the disk takes longer than this gap)
        st->pages.clear();
        for (size_t i = 0; i < n_rows; ++i) {
            const auto [p0, p1] = st->span(rows[i]);
            for (uint64_t p = p0; p <= p1; ++p) {
                st->pages.push_back(p);
            }
        }
        std::sort(st->pages.begin(), st->pages.end());
        st->pages.erase(std::unique(st->pages.begin(), st->pages.end()), st->pages.end());
        st->ev_pages += st->pages.size();

        int64_t res = -1;
        if (n_tokens <= 64) {
            res = dp_resident(st->pages, st->page);
            if (res >= 0) {
                st->ev_chk += st->pages.size();
                st->ev_res += (uint64_t) res;
            }
        }

        LLAMA_LOG_WARN("draft-prefetch: n_tokens = %u rows = %zu ahead = %" PRIu64 " pages = %zu resident = %" PRId64
                " | calls = %" PRIu64 " tokens = %" PRIu64 " issued = %" PRIu64 " dup = %" PRIu64 " capped = %" PRIu64
                " bytes = %" PRIu64 " call_us = %" PRId64 " | us = %" PRId64 "\n",
                n_tokens, n_rows, ahead, st->pages.size(), res, st->w_calls, st->w_tokens, st->w_issued, st->w_dup,
                st->w_capped, st->w_bytes, st->w_us, ggml_time_us() - t0);
    }

    // a new window
    st->w_rows.clear();
    st->w_pages.clear();
    st->w_calls  = 0;
    st->w_tokens = 0;
    st->w_issued = 0;
    st->w_dup    = 0;
    st->w_capped = 0;
    st->w_bytes  = 0;
    st->w_us     = 0;
}

void llama_draft_prefetch_release(const llama_memory_i * mem) {
    const auto & cfg = dp_cfg();
    if (!cfg.on && !cfg.stats) {
        return;
    }

    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_states.find(dp_attn_kv(mem));
    if (it == g_states.end()) {
        return;
    }
    const dp_state & st = *it->second;
    // a window not closed by an evaluation (the last draft of a run) counts as issued
    const uint64_t calls  = st.calls  + st.w_calls;
    const uint64_t issued = st.issued + st.w_issued;
    LLAMA_LOG_WARN("C615: draft prefetch %s, cap %" PRIu64 " KiB: %" PRIu64 " calls issued the rows of %" PRIu64
            " tokens (%" PRIu64 " rows, %" PRIu64 " already issued, %" PRIu64 " tokens over the cap, most %" PRIu64
            " B of pages in a window, %.3f ms in the calls); %" PRIu64 " evaluations set %" PRIu64 " rows, %" PRIu64
            " issued ahead; %" PRIu64 " of up to 64 tokens set %" PRIu64 " rows, %" PRIu64 " issued ahead; pages %" PRIu64
            ", %" PRIu64 " of %" PRIu64 " checked in the page cache\n",
            cfg.on ? "on" : "off", cfg.cap / 1024, calls, st.tokens + st.w_tokens, issued, st.dup + st.w_dup,
            st.capped + st.w_capped, std::max(st.bytes_max, st.w_bytes), (st.us + st.w_us) / 1000.0, st.evals,
            st.ev_rows, st.ev_ahead, st.sm_evals, st.sm_rows, st.sm_ahead, st.ev_pages, st.ev_res, st.ev_chk);
    g_states.erase(it);
}
