#include "range-mask.cuh"
#include "convert.cuh"

#include <cstdlib>

// One block row per mask row (blockIdx.y), threads over its columns, each reading its row's two ints.
// fault (GGML_CUDA_RANGE_MASK_FAULT=1, for testing the tests): one more column visible after each range.
template <typename T>
static __global__ void range_mask_kernel(
        const int * __restrict__ ranges, T * __restrict__ dst, const int n, const int64_t nb1, const int fault) {
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (j >= n) {
        return;
    }
    const int j0 = ranges[2*r + 0];
    const int j1 = ranges[2*r + 1] + fault;
    T * row = (T *) ((char *) dst + r*nb1);
    row[j] = ggml_cuda_cast<T>(j >= j0 && j < j1 ? 0.0f : -INFINITY);
}

void ggml_cuda_op_range_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ranges = dst->src[0];
    GGML_ASSERT(ranges->type == GGML_TYPE_I32 && ggml_is_contiguous(ranges) && ggml_is_contiguous(dst));

    static const int fault = [] {
        const char * env = getenv("GGML_CUDA_RANGE_MASK_FAULT");
        return env != nullptr && atoi(env) != 0 ? 1 : 0;
    }();

    const int     n  = (int) dst->ne[0];
    const int64_t nr = ggml_nrows(dst);
    GGML_ASSERT(nr <= 65535);

    const dim3 grid((n + CUDA_RANGE_MASK_BLOCK_SIZE - 1) / CUDA_RANGE_MASK_BLOCK_SIZE, (unsigned int) nr, 1);
    cudaStream_t stream = ctx.stream();
    if (dst->type == GGML_TYPE_F16) {
        range_mask_kernel<half><<<grid, CUDA_RANGE_MASK_BLOCK_SIZE, 0, stream>>>(
            (const int *) ranges->data, (half *) dst->data, n, dst->nb[1], fault);
    } else {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        range_mask_kernel<float><<<grid, CUDA_RANGE_MASK_BLOCK_SIZE, 0, stream>>>(
            (const int *) ranges->data, (float *) dst->data, n, dst->nb[1], fault);
    }
}
