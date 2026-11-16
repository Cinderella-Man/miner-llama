#pragma once

// LLAMA_DRAFT_PREFETCH (C615): the n-gram table's rows of the tokens a drafter samples, prefetched as each one is sampled
// (llama_prefetch_rows: madvise(MADV_WILLNEED) on the rows of the mapped table), so that the reads from the disk run
// while the drafter works instead of when the verify pass's n-gram input sets its rows just before the cards start.
//
// - The public calls are llama_draft_prefetch_enabled and llama_draft_prefetch (llama-ext.h). A token's rows are
//   computed from the token and its n_gram - 1 predecessors exactly as qwen4exp's n-gram input computes them; the
//   predecessors before the call's first position come from the context's attention cells, as the input reads them.
// - The window: the rows issued since the last evaluation whose n-gram input set the context's rows. A row issued in the
//   window is not issued again, and at most LLAMA_DRAFT_PREFETCH_KB (default 1024) KiB of pages (each row's page span)
//   are issued in one window: a token whose new pages would pass the cap is not issued, nor any later one of that call.
// - The rows read stay the same bytes: the input still computes, prefetches and uploads its own rows. Only when the
//   reads are issued changes.
// - Counts (LLAMA_DRAFT_PREFETCH_STATS=1 or LLAMA_UBATCH_TRACE, the switch off too): each evaluation's rows, how many of
//   them the window had issued, their distinct pages and, for evaluations of up to 64 tokens, how many of those pages
//   were in the page cache when the input had set its rows (mincore), on a "draft-prefetch:" line; a closing "C615:" line
//   per context.
// - One state per context with a mapped (lazy) n-gram table of a qwen4exp model, keyed by its attention KV cache.

#include "llama.h"

#include <cstddef>
#include <cstdint>

struct llama_model;
class llama_kv_cache;
struct llama_memory_i;

// the n-gram input's set_input, after it set its rows: the micro-batch's rows (n_heads per token) for the context whose
// attention cache is kv; counts them against the window, logs them (stats) and starts a new window
void llama_draft_prefetch_on_rows(const llama_kv_cache * kv, const llama_model & model, const int32_t * rows, size_t n_rows,
                                  uint32_t n_tokens);

// a context is freed: its closing line (when anything was issued or counted) and its state dropped
void llama_draft_prefetch_release(const llama_memory_i * mem);
