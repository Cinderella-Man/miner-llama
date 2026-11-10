#pragma once

#include "llama.h"

#include "llama-arch.h"
#include "llama-cparams.h"

#include <array>
#include <vector>
#include <set>
#include <bitset>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

// keep this struct lightweight
struct llama_ubatch {
    bool equal_seqs() const {
        return b_equal_seqs != 0;
    }

    // typical for M-RoPE cases:
    //   0 - sequential position of the tokens/embeddings in the sequence
    //   1 - y position in the image
    //   2 - x position in the image
    //   3 - other
    bool is_pos_2d() const {
        // TODO @ngxson : we may need to check for model arch when more models use >1 positions
        return n_pos >= 3;
    }

    // mixed: type picks token or embd per row, pos has n_pos sections for all rows
    bool is_mixed() const {
        return type != nullptr;
    }

    uint32_t b_equal_seqs; // note: this is a boolean, but we use an int32_t for alignment
                           //       otherwise address sanitizer complains
    // TODO: whole_seqs for embeddings?

    uint32_t n_tokens;     // total tokens (n_seq_tokens * n_seqs)
    uint32_t n_seq_tokens; // tokens per sequence set
    uint32_t n_seqs;       // sequence sets in the ubatch
    uint32_t n_seqs_unq;   // unique sequence ids in the ubatch
    uint32_t n_pos;        // number of position inputs for each token/embedding

    // seq_id_unq: unique sequence ids in the ubatch
    // seq_idx:    indices of the unique sequence ids in the ubatch in [0, n_seqs_unq)
    //             used for extracting sequence pooled embeddings

    //                          // size                     | idx | val
    llama_token  *  token;      // [n_tokens]               | i   | id, token
    float        *  embd;       // [n_embd, n_tokens]       | i   | embd
    float        *  embd_state; // [n_embd_state, n_tokens] | i   | hidden state carried over from a previous stage (e.g. MTP)
    int32_t      *  embd_row;   // [n_tokens]               | i   | embd is this row of ctx_other's layer input device buffer
    llama_pos    *  pos;        // [n_tokens*n_pos]         | i   | pos
    int32_t      *  n_seq_id;   // [n_tokens]         | i   | -
    llama_seq_id ** seq_id;     // [n_tokens]         | s   | s0, s1, seq_id
    llama_seq_id *  seq_id_unq; // [n_seqs_unq]       | s   | seq_id
    int32_t      *  seq_idx;    // [LLAMA_MAX_SEQ]    | -   | seq_idx
    int8_t       *  output;     // [n_tokens]         | i   | -
    int8_t       *  type;       // [n_tokens]         | i   | -     (mixed ubatch only, 0 - token, 1 - embd)
    int32_t      *  decision_order; // [n_tokens], NULL if no entry has one, see llama_batch_ext_set_decision_order()

    struct data_t {
        std::vector<llama_token>    token;
        std::vector<float>          embd;
        std::vector<float>          embd_state;
        std::vector<int32_t>        embd_row;
        std::vector<int32_t>        embd_state_row; // the state is this row of ctx_state_src's h_nextn device buffer, empty if none
        std::vector<llama_pos>      pos;
        std::vector<int32_t>        n_seq_id;
        std::vector<llama_seq_id *> seq_id;      // these point into the seq_id_data below
        std::vector<llama_seq_id>   seq_id_unq;
        std::vector<int32_t>        seq_idx;
        std::vector<int8_t>         output;
        std::vector<int8_t>         type;
        std::vector<int32_t>        batch_idxs;  // original batch index for each token
        std::vector<int32_t>        decision_order;

        std::vector<llama_seq_id> seq_id_data;
    };

    // the llama_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
    std::shared_ptr<data_t> data;
};

// crash if a mixed ubatch reaches code that expects only tokens or only embd
#define ASSERT_EMBD_OR_TOKEN(ubatch) GGML_ASSERT(!(ubatch).is_mixed() && "mixed token/embd ubatch is not supported here")

struct llama_hparams;

// DFlash batches carry the fused target features at the encoder input width (n_embd_inp_enc size).
// Other batches carry token embeddings (n_embd_inp size).
size_t llama_batch_ext_select_n_embd_inp(llama_context_type ctx_type, llm_arch arch, const llama_hparams & hparams);

// MTP contexts also take the target model's hidden state (n_embd_out size), 0 = no state input
size_t llama_batch_ext_select_n_embd_state(llama_context_type ctx_type, const llama_hparams & hparams);

struct llama_batch_ext {
    const size_t n_tokens_max;     // max number of tokens that can be stored in the batch
    const size_t n_embd_inp;       // decoder embd row width
    const size_t n_embd_inp_enc;   // encoder embd row width (e.g. eagle3/dflash extracted features)
    const size_t n_embd_state;     // state embd row width, 0 if the context takes no state
    const llama_seq_id n_seq_max;  // max number of sequences
    llama_memory_i * mem;          // memory for position inference
    const llama_token n_vocab;     // max token ID that we accept
    const size_t n_pos_per_embd;

    // actual embd row width of this batch, set by the first set_token_embd()
    // must be either n_embd_inp or n_embd_inp_enc; encode/decode verify it against the graph input
    size_t n_embd = 0;

    struct token {
        llama_token  id = LLAMA_TOKEN_NULL;
        bool         has_embd = false; // whether embd_off is set
        size_t       embd_off = 0; // index offset in the embd array
        bool         has_state = false; // whether state_off is set
        size_t       state_off = 0; // index offset in the state array
        int32_t      embd_row = -1; // row of ctx_other's layer input device buffer used as embd, -1 if none
        int32_t      state_row = -1; // row of ctx_state_src's h_nextn device buffer used as state, -1 if none
        bool         output = false; // TODO: have dedicated output flags
        int32_t      decision_order = 0; // see llama_batch_ext_set_decision_order()
        std::unordered_set<llama_seq_id> seq_ids;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos = {0, 0, 0, 0};
    };
    std::vector<token> tokens;
    std::vector<float> embd;
    std::vector<float> state;

    llama_batch_ext(llama_context * ctx);

    // build without a llama_context, used by tests
    llama_batch_ext(
            size_t n_tokens_max,
            size_t n_embd_inp,
            size_t n_embd_inp_enc,
            llama_seq_id n_seq_max,
            llama_memory_i * mem,
            llama_token n_vocab,
            size_t n_pos_per_embd);

    void clear();

    // add an entry with an undefined position
    // the caller must set it explicitly via set_token_pos()
    int32_t add_token(llama_seq_id seq_id);

    bool add_seq(int32_t idx, llama_seq_id seq_id);
    bool set_token_id(int32_t idx, llama_token id);
    bool set_token_embd(int32_t idx, llama_embd embd_in);
    bool set_token_state(int32_t idx, llama_embd state_in);
    bool set_token_embd_row(int32_t idx, int32_t row);
    bool set_token_state_row(int32_t idx, int32_t row);
    bool set_token_pos(int32_t idx, const llama_pos * pos_in);
    bool set_output(int32_t idx, bool output_last);
    bool set_decision_order(int32_t idx, int32_t order);
};

// Micro-batch taper (Q771, LLAMA_UBATCH_TAPER, llama_set_ubatch_taper): with layer-split pipeline parallelism a
// micro-batch crosses the devices one after another, so the first micro-batches of a call that starts on idle devices
// fill the pipeline and the last ones before the caller waits drain it. The taper gives those ends of a call smaller
// micro-batches and leaves the rest at n_ubatch. Sizes never exceed n_ubatch, so the reserved compute buffers hold.
struct llama_ubatch_taper {
    uint32_t head_n  = 0; // the first head_n tokens of a call that starts with nothing in flight,
    uint32_t head_s  = 0; //     in micro-batches of head_s tokens (the remainder first)
    uint32_t tail_n  = 0; // the last tail_n tokens of a call that the caller waits for,
    uint32_t tail_s  = 0; //     in micro-batches of tail_s tokens (the remainder last)
    uint32_t short_n = 0; // a call that is both and holds short_n tokens or fewer: all of it
    uint32_t short_s = 0; //     in micro-batches of short_s tokens (the remainder last)
    uint32_t min_s   = 9; // no planned micro-batch below min_s tokens (1-8 tokens take the matrix-vector products)

    bool enabled() const {
        return (head_n > 0 && head_s > 0) || (tail_n > 0 && tail_s > 0) || (short_n > 0 && short_s > 0);
    }

    // "", "0" or "off": off; "1" or "on": head=256:64,tail=512:128,short=600:64; otherwise any of head=N:S, tail=N:S,
    // short=N:S and min=N, comma-separated (a rule left out is off, min 9); false with a message in err on a bad spec
    bool parse(const char * spec, std::string & err);

    // the spec of this schedule, "off" when off
    std::string str() const;
};

// the ends (token offsets) of the micro-batches of a call of n_tokens under the taper; head: the call starts with
// nothing in flight, tail: the caller waits after it. Empty when the taper leaves the call to n_ubatch (the split the
// memory modules make without a plan).
std::vector<uint32_t> llama_ubatch_taper_plan(
        const llama_ubatch_taper & taper, uint32_t n_tokens, uint32_t n_ubatch, bool head, bool tail);

// a helper for sanitizing, fulfilling and splitting a batch
class llama_batch_allocr {
public:
    llama_batch_allocr(uint32_t n_pos_per_embd, bool allow_mixed = false);

    // convert a llama_batch_ext to internal llama_batch and sanitize it
    bool init(
            const llama_batch_ext & batch_inp,
            const llama_vocab & vocab,
            bool output_all);

    const llama_batch & get_batch() const;

    uint32_t get_n_tokens()  const;
    uint32_t get_n_outputs() const;
    uint32_t get_n_used()    const;

    // the array of output indices in the order they were encountered during the ubatch splitting
    std::vector<int32_t> & get_out_ids();

    // min/max positions of each sequence in the current ubatch
    llama_pos seq_pos_min(llama_seq_id seq_id) const;
    llama_pos seq_pos_max(llama_seq_id seq_id) const;

    // call once before splitting the batch to reset the internal state
    void split_reset();

    // simple split, unknown number of sequence sets of unequal lengths
    llama_ubatch split_simple(uint32_t n_ubatch);

    // make ubatches of equal-length sequences sets
    // if sequential == true, the tokens in the ubatch will have increasing sequential sequence ids
    // n_keep_tail = minimum trailing tokens of a seq that must land in the same ubatch
    llama_ubatch split_equal(uint32_t n_ubatch, bool sequential, uint32_t n_keep_tail);

    // sequence-set-wise split - each ubatch contains a single sequence-set
    // TODO: extend to support multiple _full_ sequences in the ubatch
    llama_ubatch split_seq(uint32_t n_ubatch);

    // a helper method for creating a well-defined ubatch of tokens
    // TODO: support embeddings if needed in the future
    llama_ubatch ubatch_reserve(uint32_t n_seq_tokens, uint32_t n_seqs);

    // Q771: the ends (token offsets) of the micro-batches planned for the current batch (llama_ubatch_taper_plan);
    // while set, a split takes at most the tokens up to the next planned end. init() clears it, empty = no plan
    void set_ubatch_plan(std::vector<uint32_t> ends);

    // the batches the taper applies to: token ids only (no embeddings, states, embd rows or decision orders), all
    // of one sequence
    bool is_one_seq_tokens() const;

    // C217: cuts (token offsets) in the current batch: a split stops at the next cut and treats it as the end of the
    // batch (the trailing tokens of split_equal's n_keep_tail too), so the batch splits into the micro-batches of
    // separate calls cut there. For batches of one sequence (is_one_seq); init() clears them, empty = no cuts
    void set_ubatch_cuts(std::vector<uint32_t> cuts);

    // all tokens of one sequence (the cuts apply)
    bool is_one_seq() const;

private:
    void clear();

    // the most tokens the next split may take under the plan, n_ubatch without one
    uint32_t plan_cap(uint32_t n_ubatch) const;

    // C217: the first cut after the tokens used so far, UINT32_MAX without one
    uint32_t cut_next() const;

    std::vector<uint32_t> ubatch_plan;
    std::vector<uint32_t> ubatch_cuts;

    // create the next ubatch based on the provided batch indices (idxs) and the number of sequence sets (n_seqs)
    // return llama_ubatch.n_tokens == 0 if the entire batch was consumed
    llama_ubatch ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs);

    // for debugging, start with LLAMA_BATCH_DEBUG=2
    void ubatch_print(const llama_ubatch & ubatch, int debug);

    llama_batch batch;

    // only for debugging purposes
    const llama_vocab * vocab;

    // TODO: this is more of a temporary solution until we have a better way to handle multiple positions per token/embd
    //       ref: https://github.com/ggml-org/llama.cpp/issues/13694#issuecomment-2983871762
    const uint32_t n_pos_per_embd;

    const bool allow_mixed;

    uint32_t n_embd;
    uint32_t n_embd_state;
    uint32_t n_seq_max;
    uint32_t n_outputs;

    std::vector<llama_token>    token_vec;    // owned token IDs built from llama_batch_ext
    std::vector<float>          embd_vec;     // owned embeddings built from llama_batch_ext
    std::vector<int8_t>         is_embd_vec;  // mixed batch only (= 1 if embd, 0 if text token)
    std::vector<float>          state_vec;    // owned state embeddings built from llama_batch_ext, llama_batch has no slot for them
    std::vector<int32_t>        embd_row_vec; // owned embd rows built from llama_batch_ext, llama_batch has no slot for them
    std::vector<int32_t>        state_row_vec; // owned state rows built from llama_batch_ext, llama_batch has no slot for them
    std::vector<llama_seq_id>   seq_id_data;  // flat storage for seq_id pointers below

    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id>   seq_id_unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         output;
    std::vector<int32_t>        decision_order; // empty if no entry has one

    using pos_set_t = std::set<llama_pos>;
    using seq_cpl_t = std::vector<bool>;

    // helper flag to quickly determine if there are any coupled sequences in the batch
    bool has_cpl = false;

    std::vector<pos_set_t> seq_pos; // seq_pos[s]: the set of positions in sequence s
    std::vector<seq_cpl_t> seq_cpl; // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1

    using idx_vec_t = std::vector<int32_t>;
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    std::vector<seq_set_t> seq_set; // seq_set[i]: the sequence set of token i

    std::unordered_map<seq_set_t, idx_vec_t> seq_set_map; // the indices at which the sequence set appears

    // batch indices of the output
    std::vector<int32_t> out_ids;

    uint32_t n_used;

    // used[i] indicates if token i has already been used in a previous ubatch
    std::vector<bool> used;

    int debug;
};

// RAII translation layer: converts a llama_batch (old API) into a llama_batch_ext
struct llama_batch_compat {
    llama_batch_ext * batch_ext;

    // n_embd_row is the embd row width of batch_inp, 0 = use the decoder width
    llama_batch_compat(llama_context * ctx, const llama_batch & batch_inp, size_t n_embd_row = 0);
    ~llama_batch_compat();

    // fill an existing llama_batch_ext from a llama_batch (old API)
    // note: this is called directly by the tests, skipping llama_context creation
    static void init(llama_batch_ext & batch_ext, const llama_batch & batch_inp, size_t n_embd_row = 0);
};
