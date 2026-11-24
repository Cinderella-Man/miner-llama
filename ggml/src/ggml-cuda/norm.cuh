#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx,
                       ggml_tensor *               dst,
                       ggml_tensor *               mul = nullptr,
                       ggml_tensor *               add = nullptr);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// GGML_CUDA_NORM_Q8 (default on): fused norms of a single row also write its Q8_1 copy for mul_mat_vec_q.
bool ggml_cuda_norm_q8_enabled();

// RMS_NORM + MUL (rms_norm->src[0] == add when add is set, then ADD + RMS_NORM + MUL) of one row, also writing the row's
// Q8_1 blocks into ctx.q8_1_row. Returns false, having done nothing, when the tensors are not covered.
bool ggml_cuda_op_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
