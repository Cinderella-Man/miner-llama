#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_multi(const T *            x,
                                  D *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const int64_t *      row_indices,
                                  const int            set_rows_stride,
                                  const bool           inplace) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = ggml_cuda_cast<D>(x[ix + i0/2 + 0]);
        dst[idst + i0/2 + 1] = ggml_cuda_cast<D>(x[ix + i0/2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]        = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
    dst[idst + n_offs/2 + n_dims/2] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_multi_cuda(const T *            x,
                            D *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const int64_t *      row_indices,
                            const int            set_rows_stride,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(dst), corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_multi_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                   s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                   ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                   row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                  row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                 row_indices, set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
template <int block_size, bool has_ff, typename D>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc = x + sample*s03 + channel*s02 + row*s01;

            const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
            const float * mulc = mul + mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

            float tmp = 0.0f;

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            const float scale = rsqrtf(tmp/ncols + eps);

            int64_t idst = sample*s3 + channel*s2 + row*s1;
            if (set_rows_stride != 0) {
                idst = row*s1 + row_indices[channel]*set_rows_stride;
            }
            D * dstc = dst + idst;

            for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
                int ix0;
                int ix1;
                if (is_neox && i0 < n_dims) {
                    ix0 = i0/2;
                    ix1 = i0/2 + n_dims/2;
                } else {
                    ix0 = i0 + 0;
                    ix1 = i0 + 1;
                }

                const float x0 = scale * xc[ix0] * mulc[fastmodulo(ix0, mul_ncols_packed)];
                const float x1 = scale * xc[ix1] * mulc[fastmodulo(ix1, mul_ncols_packed)];

                if (i0 >= n_dims) {
                    dstc[ix0] = ggml_cuda_cast<D>(x0);
                    dstc[ix1] = ggml_cuda_cast<D>(x1);
                    continue;
                }

                const float theta_base  = pos[channel]*powf(theta_scale, i0/2.0f);
                const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

                float cos_theta;
                float sin_theta;
                rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

                dstc[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
                dstc[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);

    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox);
        }
    }
}

// the arguments of rms_norm_mul_rope_cuda for RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
struct rms_norm_mul_rope_params {
    const float   * x;
    void          * dst;
    ggml_type       dst_type;
    int64_t         ne[4];
    int64_t         s01, s02, s03, s1, s2, s3;
    float           eps;
    const float   * mul;
    int64_t         mul_s01, mul_s02, mul_s03;
    int64_t         mul_ne[4];
    int             n_dims;
    const int32_t * pos;
    float           freq_scale, freq_base, ext_factor, attn_factor;
    rope_corr_dims  corr_dims;
    const float   * freq_factors;
    const int64_t * row_indices;
    int             set_rows_stride;
    bool            is_neox;
};

static rms_norm_mul_rope_params rms_norm_mul_rope_get_params(
        const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * rope, const ggml_tensor * set_rows) {
    rms_norm_mul_rope_params p;

    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    p.dst             = rope->data;
    p.dst_type        = rope->type;
    p.row_indices     = nullptr;
    p.set_rows_stride = 0;

    if (set_rows != nullptr) {
        p.dst             = set_rows->data;
        p.dst_type        = set_rows->type;
        p.row_indices     = (const int64_t *) set_rows->src[1]->data;
        p.set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    p.n_dims             = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float beta_fast;
    float beta_slow;

    memcpy(&p.freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&p.freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&p.ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&p.attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,     (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,     (const int32_t *) rope->op_params + 10, sizeof(float));

    p.is_neox = mode & GGML_ROPE_TYPE_NEOX;

    p.pos = (const int32_t *) rope->src[1]->data;

    p.freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    ggml_rope_yarn_corr_dims(p.n_dims, n_ctx_orig, p.freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(rope), p.corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    p.s01 = x->nb[1] / ts0;
    p.s02 = x->nb[2] / ts0;
    p.s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    p.mul_s01 = mul_src->nb[1] / ts_mul;
    p.mul_s02 = mul_src->nb[2] / ts_mul;
    p.mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    p.s1 = rope->nb[1] / ts_dst;
    p.s2 = rope->nb[2] / ts_dst;
    p.s3 = rope->nb[3] / ts_dst;

    p.x   = (const float *) x->data;
    p.mul = (const float *) mul_src->data;
    for (int i = 0; i < 4; ++i) {
        p.ne[i]     = x->ne[i];
        p.mul_ne[i] = mul_src->ne[i];
    }
    p.eps = eps;
    return p;
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
    const rms_norm_mul_rope_params p = rms_norm_mul_rope_get_params(rms_norm, mul, rope, set_rows);

    cudaStream_t stream = ctx.stream();

    if (p.dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda(p.x, (float *) p.dst,
            p.ne[0], p.ne[1], p.ne[2], p.ne[3], p.s01, p.s02, p.s03, p.s1, p.s2, p.s3, p.eps,
            p.mul, p.mul_s01, p.mul_s02, p.mul_s03,
            p.mul_ne[0], p.mul_ne[1], p.mul_ne[2], p.mul_ne[3],
            p.n_dims, p.pos, p.freq_scale, p.freq_base, p.ext_factor, p.attn_factor, p.corr_dims,
            p.freq_factors, p.row_indices, p.set_rows_stride, p.is_neox, stream);
    } else if (p.dst_type == GGML_TYPE_F16) {
        rms_norm_mul_rope_cuda(p.x, (half *) p.dst,
            p.ne[0], p.ne[1], p.ne[2], p.ne[3], p.s01, p.s02, p.s03, p.s1, p.s2, p.s3, p.eps,
            p.mul, p.mul_s01, p.mul_s02, p.mul_s03,
            p.mul_ne[0], p.mul_ne[1], p.mul_ne[2], p.mul_ne[3],
            p.n_dims, p.pos, p.freq_scale, p.freq_base, p.ext_factor, p.attn_factor, p.corr_dims,
            p.freq_factors, p.row_indices, p.set_rows_stride, p.is_neox, stream);
    } else {
        GGML_ABORT("fatal error");
    }
}

// One token of an attention block after its Q/K/V products: Q's RMS_NORM + MUL + ROPE, K's RMS_NORM + MUL + ROPE + VIEW
// + SET_ROWS and V's VIEW + SET_ROWS in one launch. Blocks [0, q.nrows) are Q's rows (heads), the next k.nrows K's, the
// last one writes V; every value is computed as rms_norm_mul_rope_f32 (256 threads per row) and k_set_rows compute it
// (attn_epilogue_row is rms_norm_mul_rope_f32's body for one row; the kernel itself stays as it is).
struct attn_epilogue_rows {
    const float   * x;
    void          * dst;
    int             ncols;
    int             nrows;
    int64_t         s01, s02, s03, s1, s2, s3;
    float           eps;
    const float   * mul;
    int64_t         mul_s01, mul_s02, mul_s03;
    uint3           mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed;
    int             n_dims;
    const int32_t * pos;
    float           freq_scale, ext_factor, attn_factor;
    rope_corr_dims  corr_dims;
    float           theta_scale;
    const float   * freq_factors;
    const int64_t * row_indices;
    int             set_rows_stride;
    bool            is_neox;
};

struct attn_epilogue_args {
    attn_epilogue_rows q; // F32 result
    attn_epilogue_rows k; // F16, into the K cache
    const float   * v;    // the token's V row, nv values, into row v_idx[0] of the V cache as F16
    half          * v_dst;
    const int64_t * v_idx;
    int64_t         v_row_stride;
    int             nv;
};

static attn_epilogue_rows attn_epilogue_get_rows(const rms_norm_mul_rope_params & p) {
    attn_epilogue_rows r;
    r.x                    = p.x;
    r.dst                  = p.dst;
    r.ncols                = (int) p.ne[0];
    r.nrows                = (int) p.ne[1];
    r.s01                  = p.s01;
    r.s02                  = p.s02;
    r.s03                  = p.s03;
    r.s1                   = p.s1;
    r.s2                   = p.s2;
    r.s3                   = p.s3;
    r.eps                  = p.eps;
    r.mul                  = p.mul;
    r.mul_s01              = p.mul_s01;
    r.mul_s02              = p.mul_s02;
    r.mul_s03              = p.mul_s03;
    r.mul_ncols_packed     = init_fastdiv_values((uint32_t) p.mul_ne[0]);
    r.mul_nrows_packed     = init_fastdiv_values((uint32_t) p.mul_ne[1]);
    r.mul_nchannels_packed = init_fastdiv_values((uint32_t) p.mul_ne[2]);
    r.mul_nsamples_packed  = init_fastdiv_values((uint32_t) p.mul_ne[3]);
    r.n_dims               = p.n_dims;
    r.pos                  = p.pos;
    r.freq_scale           = p.freq_scale;
    r.ext_factor           = p.ext_factor;
    r.attn_factor          = p.attn_factor;
    r.corr_dims            = p.corr_dims;
    r.theta_scale          = powf(p.freq_base, -2.0f/p.n_dims);
    r.freq_factors         = p.freq_factors;
    r.row_indices          = p.row_indices;
    r.set_rows_stride      = p.set_rows_stride;
    r.is_neox              = p.is_neox;
    return r;
}

// one row (head) of channel 0 and sample 0 with 256 threads, as rms_norm_mul_rope_f32<256> computes it (the same code)
template <bool has_ff, typename D>
static __device__ __forceinline__ void attn_epilogue_row(const attn_epilogue_rows & r, const int row) {
    constexpr int block_size = 256;
    const int tid     = threadIdx.x;
    const int channel = 0;
    const int sample  = 0;

    extern __shared__ float s_sum[];

    const float * xc = r.x + sample*r.s03 + channel*r.s02 + row*r.s01;

    const uint32_t mul_row     = fastmodulo(row,     r.mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, r.mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample,  r.mul_nsamples_packed);
    const float * mulc = r.mul + mul_sample*r.mul_s03 + mul_channel*r.mul_s02 + mul_row*r.mul_s01;

    float tmp = 0.0f;

    for (int col = tid; col < r.ncols; col += block_size) {
        const float xi = xc[col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float scale = rsqrtf(tmp/r.ncols + r.eps);

    int64_t idst = sample*r.s3 + channel*r.s2 + row*r.s1;
    if (r.set_rows_stride != 0) {
        idst = row*r.s1 + r.row_indices[channel]*r.set_rows_stride;
    }
    D * dstc = (D *) r.dst + idst;

    for (int i0 = 2*tid; i0 < r.ncols; i0 += 2*block_size) {
        int ix0;
        int ix1;
        if (r.is_neox && i0 < r.n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + r.n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }

        const float x0 = scale * xc[ix0] * mulc[fastmodulo(ix0, r.mul_ncols_packed)];
        const float x1 = scale * xc[ix1] * mulc[fastmodulo(ix1, r.mul_ncols_packed)];

        if (i0 >= r.n_dims) {
            dstc[ix0] = ggml_cuda_cast<D>(x0);
            dstc[ix1] = ggml_cuda_cast<D>(x1);
            continue;
        }

        const float theta_base  = r.pos[channel]*powf(r.theta_scale, i0/2.0f);
        const float freq_factor = has_ff ? r.freq_factors[i0/2] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn<true>(theta_base/freq_factor, r.freq_scale, r.corr_dims, i0, r.ext_factor, r.attn_factor, cos_theta, sin_theta);

        dstc[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
        dstc[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
    }
}

template <bool has_ff>
__launch_bounds__(256, 1)
static __global__ void attn_epilogue_f32(const attn_epilogue_args a) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int b = blockIdx.x;
    if (b < a.q.nrows) {
        attn_epilogue_row<has_ff, float>(a.q, b);
    } else if (b < a.q.nrows + a.k.nrows) {
        attn_epilogue_row<has_ff, half>(a.k, b - a.q.nrows);
    } else {
        half * dst = a.v_dst + a.v_idx[0]*a.v_row_stride;
        for (int i = threadIdx.x; i < a.nv; i += blockDim.x) {
            dst[i] = ggml_cuda_cast<half>(a.v[i]);
        }
    }
}

// the epilogue's arguments for these nodes, if it covers them
static bool attn_epilogue_prepare(
        const ggml_tensor * q_norm, const ggml_tensor * q_mul, const ggml_tensor * q_rope,
        const ggml_tensor * k_norm, const ggml_tensor * k_mul, const ggml_tensor * k_rope, const ggml_tensor * k_set_rows,
        const ggml_tensor * v_set_rows, attn_epilogue_args & a) {
    const rms_norm_mul_rope_params pq = rms_norm_mul_rope_get_params(q_norm, q_mul, q_rope, nullptr);
    const rms_norm_mul_rope_params pk = rms_norm_mul_rope_get_params(k_norm, k_mul, k_rope, k_set_rows);

    // one token; contiguous rows of fewer than 1024 values, which rms_norm_mul_rope_cuda gives 256 threads; F16 caches
    for (const rms_norm_mul_rope_params * p : { &pq, &pk }) {
        if (p->ne[0] >= 1024 || p->ne[0] % 2 != 0 || p->ne[2] != 1 || p->ne[3] != 1 || p->s01 != p->ne[0]) {
            return false;
        }
    }
    if (pq.dst_type != GGML_TYPE_F32 || pk.dst_type != GGML_TYPE_F16 ||
            (pq.freq_factors == nullptr) != (pk.freq_factors == nullptr)) {
        return false;
    }
    const ggml_tensor * v     = v_set_rows->src[0];
    const ggml_tensor * v_idx = v_set_rows->src[1];
    if (v->type != GGML_TYPE_F32 || !ggml_is_contiguous(v) || ggml_nrows(v) != 1 || v_idx->type != GGML_TYPE_I64 ||
            ggml_nelements(v_idx) != 1 || v_set_rows->type != GGML_TYPE_F16 || v->ne[0] != v_set_rows->ne[0]) {
        return false;
    }

    a.q            = attn_epilogue_get_rows(pq);
    a.k            = attn_epilogue_get_rows(pk);
    a.v            = (const float *) v->data;
    a.v_dst        = (half *) v_set_rows->data;
    a.v_idx        = (const int64_t *) v_idx->data;
    a.v_row_stride = v_set_rows->nb[1] / ggml_type_size(v_set_rows->type);
    a.nv           = (int) v->ne[0];
    return true;
}

bool ggml_cuda_attn_epilogue_supported(
        const ggml_tensor * q_norm, const ggml_tensor * q_mul, const ggml_tensor * q_rope,
        const ggml_tensor * k_norm, const ggml_tensor * k_mul, const ggml_tensor * k_rope, const ggml_tensor * k_set_rows,
        const ggml_tensor * v_set_rows) {
    attn_epilogue_args a;
    return attn_epilogue_prepare(q_norm, q_mul, q_rope, k_norm, k_mul, k_rope, k_set_rows, v_set_rows, a);
}

void ggml_cuda_op_attn_epilogue(ggml_backend_cuda_context & ctx,
        const ggml_tensor * q_norm, const ggml_tensor * q_mul, const ggml_tensor * q_rope,
        const ggml_tensor * k_norm, const ggml_tensor * k_mul, const ggml_tensor * k_rope, const ggml_tensor * k_set_rows,
        const ggml_tensor * v_set_rows, const float * q, const float * k, const float * v) {
    attn_epilogue_args a;
    GGML_ASSERT(attn_epilogue_prepare(q_norm, q_mul, q_rope, k_norm, k_mul, k_rope, k_set_rows, v_set_rows, a));
    // Q, K and V as their products laid out contiguously (one row per head) somewhere else
    a.q.x = q;
    a.k.x = k;
    a.v   = v;

    const dim3 blocks(a.q.nrows + a.k.nrows + 1, 1, 1);
    const dim3 threads(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = {blocks, threads, 32*sizeof(float), ctx.stream()};
    if (a.q.freq_factors == nullptr) {
        ggml_cuda_kernel_launch(attn_epilogue_f32<false>, launch_params, a);
    } else {
        ggml_cuda_kernel_launch(attn_epilogue_f32<true>, launch_params, a);
    }
}
