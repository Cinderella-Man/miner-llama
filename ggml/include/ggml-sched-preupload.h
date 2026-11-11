#pragma once

// C738: one call of the backend scheduler, in its own header so that the backends' sources (which include
// ggml-backend.h) are not rebuilt for it

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // Copy now, as the next evaluation would, the graph inputs (GGML_TENSOR_FLAG_INPUT, in host memory) of the
    // allocated graph's first split on a backend other than the CPU. The scheduler keeps a copy of the bytes each upload
    // sent, and the next evaluation skips an input whose host bytes are still the same (so an input set again in between
    // is copied as usual). The caller must not run that split's backend on the copies before the evaluation. Returns the
    // inputs uploaded (0 without such a split, or if its inputs are packed) and their bytes in nbytes (may be NULL); a
    // reset, a new allocation and the next evaluation drop them
    GGML_API int ggml_backend_sched_preupload_inputs(ggml_backend_sched_t sched, size_t * nbytes);

#ifdef  __cplusplus
}
#endif
