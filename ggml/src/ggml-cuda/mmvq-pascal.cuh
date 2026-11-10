#pragma once

#include "common.cuh"

// Quantized matrix-vector products of 1 to 4 tokens (Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, IQ4_XS, Q6_K, Q2_0, IQ2_S,
// IQ3_XXS, IQ2_XXS, IQ2_XS, IQ3_S) tuned for compute capability 6.1 (Pascal with dp4a). On small Pascal cards such as
// the 5-SM P106-090, mul_mat_vec_q runs out of instruction issue and load throughput long before memory bandwidth; these
// kernels give each row a lane group sized to the row width, let several rows share one load of the activations, load
// several weight blocks per lane with coalesced wide loads and keep the addressing out of the loop. They compute the
// activation block sums in FP32 from the int8 values instead of reading the FP16 sum of block_q8_1, so activations whose
// blocks add up past FP16's range stay finite. With 2 to 4 tokens each weight row is read once for all of them, and for
// MUL_MAT_ID each distinct expert once for all tokens that picked it.
//
// Takes the same arguments as mul_mat_vec_q_switch_type (strides in blocks, Q8_1 activations) and returns true if it
// ran the product; false leaves it to mul_mat_vec_q. GGML_CUDA_MMVQ_PASCAL: 0 off, 1 compute capability 6.1 only
// (default), 2 any NVIDIA GPU (for testing). GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1: with 2 to 4 tokens, stage each token's
// activations through shared memory too (same results, a different load pattern). GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K:
// leave the listed weight types (ggml type names) to mul_mat_vec_q. GGML_CUDA_MMVQ_PASCAL_Q2_0=G,P,T: Q2_0's rows per lane
// group with and without a fused gate (1 or 2) and its tile staging (0 off, 1 where segments can't be staged, 2 wherever
// it fits), for measuring the alternatives.
bool ggml_cuda_mmvq_pascal(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst, int64_t stride_row_x, int64_t stride_col_y,
        int64_t stride_col_dst, int64_t nchannels_x, int64_t nchannels_y, int64_t nchannels_dst, int64_t stride_channel_x,
        int64_t stride_channel_y, int64_t stride_channel_dst, int64_t nsamples_x, int64_t nsamples_dst, int64_t stride_sample_x,
        int64_t stride_sample_y, int64_t stride_sample_dst, int64_t ids_stride, cudaStream_t stream);

// Whether the kernels run on a device of compute capability cc under GGML_CUDA_MMVQ_PASCAL.
bool ggml_cuda_mmvq_pascal_active(int cc);

// Whether GGML_CUDA_MMVQ_PASCAL_SKIP leaves the weight type to mul_mat_vec_q.
bool ggml_cuda_mmvq_pascal_skips(ggml_type type);

// One token: the three products of an attention block (Q, K, V) with the same Q8_1 activations vy, in one launch. Each
// matrix is contiguous with ncols columns; K and V share a type. Every row comes out exactly as ggml_cuda_mmvq_pascal
// computes it. Returns false (nothing launched) unless the kernels are active on this device, the types are a supported
// combination and every product would run a staged shape there.
struct ggml_cuda_mmvq_pascal_qkv_args {
    const void * vx[3];    // weights: Q, K, V
    ggml_type    type[3];
    int64_t      nrows[3];
    float      * dst[3];
    int64_t      ncols;
    const void * vy;       // Q8_1 activations, ncols padded to MATRIX_ROW_PADDING
};
bool ggml_cuda_mmvq_pascal_qkv(const ggml_cuda_mmvq_pascal_qkv_args & args, cudaStream_t stream);

// One token of a contiguous F16 matrix (ncols columns, rows stride_row halves apart) with FP32 activations y, as
// mul_mat_vec_f computes it with FP32 accumulation, e.g. a MoE router kept in F16. Returns false (nothing launched)
// unless GGML_CUDA_MMVQ_PASCAL_F16=1 (off by default), the Pascal kernels are active on this device
// (GGML_CUDA_MMVQ_PASCAL) and F16 is not in GGML_CUDA_MMVQ_PASCAL_SKIP, ncols is a multiple of 8, and x, y and the rows
// are 16-byte aligned.
bool ggml_cuda_mmvf_pascal_f16(const void * x, const float * y, float * dst, int64_t ncols, int64_t nrows, int64_t stride_row,
                               cudaStream_t stream);

// Whether ggml_cuda_mmvf_pascal_f16 would take a matrix x (activations 16-byte aligned), so that the persistent MoE kernel
// can sum its router rows in the same order as the per-op path.
bool ggml_cuda_mmvf_pascal_f16_applies(const void * x, int64_t ncols, int64_t stride_row);
