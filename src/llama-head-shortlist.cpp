#include "llama-head-shortlist.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>

bool llama_head_shortlist::init(uint32_t n_vocab, uint32_t k_low, uint32_t m_cap) {
    if (n_vocab == 0 || k_low == 0 || k_low >= n_vocab || m_cap > 65535) {
        return false;
    }

    this->n_vocab = n_vocab;
    this->k_low   = k_low;
    this->m_cap   = std::min(m_cap, n_vocab - k_low);

    extra.clear();
    extra.reserve(this->m_cap);

    member.assign(n_vocab, 0);
    std::fill(member.begin(), member.begin() + k_low, 1);

    n_dev = 0;

    return true;
}

bool llama_head_shortlist::alloc(ggml_backend_buffer_type_t buft) {
    ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ids     = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, std::max<uint32_t>(m_cap, 1));
    neg_inf = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_vocab - k_low, 1);
    ggml_set_name(ids,     "head_short_ids");
    ggml_set_name(neg_inf, "head_short_neg_inf");

    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buf) {
        return false;
    }
    // placed like weights: the ops that read them run where they are
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    {
        std::vector<float> v(n_vocab - k_low, -INFINITY);
        ggml_backend_tensor_set(neg_inf, v.data(), 0, ggml_nbytes(neg_inf));
    }

    n_dev = 0;
    sync();

    return true;
}

uint32_t llama_head_shortlist::add(const int32_t * ids, size_t n) {
    uint32_t n_added = 0;
    for (size_t i = 0; i < n && extra.size() < m_cap; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || (uint32_t) id >= n_vocab || member[id]) {
            continue;
        }
        member[id] = 1;
        extra.push_back(id);
        n_added++;
    }
    return n_added;
}

void llama_head_shortlist::sync() {
    if (this->ids == nullptr || n_dev >= extra.size()) {
        return;
    }
    ggml_backend_tensor_set(this->ids, extra.data() + n_dev, n_dev*sizeof(int32_t), (extra.size() - n_dev)*sizeof(int32_t));
    n_dev = extra.size();
}

ggml_tensor * llama_head_shortlist_logits(ggml_context * ctx, const llama_head_shortlist & sl, ggml_tensor * output, ggml_tensor * h,
        uint32_t n_extra_plan) {
    const int64_t n_embd  = output->ne[0];
    const int64_t n_vocab = output->ne[1];

    GGML_ASSERT(n_vocab == (int64_t) sl.n_vocab && h->ne[0] == n_embd && ggml_nrows(h) == 1);

    if (!ggml_is_contiguous(h)) {
        h = ggml_cont(ctx, h);
    }

    // the lowest ids: the first k_low rows are a plain matrix, so they take the head's usual matrix-vector kernel
    ggml_tensor * low = ggml_mul_mat(ctx, ggml_view_2d(ctx, output, n_embd, sl.k_low, output->nb[1], 0), h); // [k_low, 1]
    ggml_tensor * out = ggml_concat(ctx, low, sl.neg_inf, 0);                                                  // [n_vocab, 1]

    GGML_ASSERT(n_extra_plan <= std::max<uint32_t>(sl.m_cap, 1));
    const int64_t n_extra = n_extra_plan > 0 ? (int64_t) n_extra_plan : (int64_t) sl.n_extra();
    if (n_extra > 0) {
        // the further ids: every row of the head is one "expert" of a mul_mat_id, as in the screened head
        ggml_tensor * ids  = ggml_view_1d(ctx, sl.ids, n_extra, 0);                                             // I32 [n_extra]
        ggml_tensor * rows = ggml_view_3d(ctx, output, n_embd, 1, n_vocab, output->nb[1], output->nb[1], 0);
        ggml_tensor * ex   = ggml_mul_mat_id(ctx, rows, ggml_reshape_3d(ctx, h, n_embd, 1, 1),
                ggml_reshape_2d(ctx, ids, n_extra, 1));                                                          // [1, n_extra, 1]

        out = ggml_set_rows(ctx, ggml_reshape_3d(ctx, out, 1, n_vocab, 1), ex, ids);
        out = ggml_cont_2d(ctx, out, n_vocab, 1);
    }

    return out;
}
