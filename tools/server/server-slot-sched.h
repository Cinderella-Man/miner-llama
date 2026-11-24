#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;

    // the live state and the checkpoints are useful when their pos_min is below this
    llama_pos pos_min_thold = 0;
};

// a checkpoint does not change after it is created: a copy of a prompt (the prompt cache's entry, a child slot) can
// share it instead of copying its data (#30137, --ctx-checkpoints-share); a checkpoint that does change (one in a device
// store moving to the host) is replaced by a new one, and a store's checkpoint is never shared
using server_prompt_checkpoint_ptr = std::shared_ptr<const common_prompt_checkpoint>;
using server_prompt_checkpoints    = std::list<server_prompt_checkpoint_ptr>;

// the checkpoints for a copy of a prompt: the same ones (share), or copies of their data
static inline server_prompt_checkpoints server_prompt_checkpoints_copy(const server_prompt_checkpoints & cks, bool share) {
    if (share) {
        return cks;
    }
    server_prompt_checkpoints res;
    for (const auto & ck : cks) {
        res.push_back(std::make_shared<common_prompt_checkpoint>(*ck));
    }
    return res;
}

struct server_prompt {
    server_tokens tokens;

    server_prompt_checkpoints checkpoints;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    // share: the copy shares the checkpoints (--ctx-checkpoints-share), else it gets copies of their data
    server_prompt clone(bool share) const {
        return server_prompt {
            tokens.clone(),
            server_prompt_checkpoints_copy(checkpoints, share),
            lora,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // n_swa: 0 if a state must end right before the next token (no SWA, or recurrent)
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    // ring_past: a ring checkpoint (common_prompt_checkpoint::ring) past the divergence may serve it
    //            (LLAMA_SERVER_CKPT_RING_PAST with the sliding-window trim); false = upstream's rule
    // resume_next: the prompt resumes right after the checkpoint's pos_max instead of at it
    //            (LLAMA_SERVER_CKPT_RESUME_NEXT with the sliding-window trim); false = upstream's rule
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min,
            bool ring_past = false, bool resume_next = false) const;
};

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // an entry shares the saved prompt's checkpoints instead of copying them (--ctx-checkpoints-share); the size counts
    // a shared checkpoint in every entry that holds it
    bool share_checkpoints = false;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // lora_new: the lora config of the new prompt, the states made with another config are skipped
    bool load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    // lora: the lora config of the task
    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora) const;
};
