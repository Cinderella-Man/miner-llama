#pragma once

#include "common.cuh"

// Matrix products of 1 to 8 tokens (columns) for compute capability 6.1 (Pascal), tuned for the 5-SM P106-090: each
// weight is read once and decoded once per launch for all columns, and the activations are quantized into records that
// every block stages in shared memory once (the design is at the top of mmvq-pascal-kcol-impl.cuh). They are for the
// k-token passes of speculative decoding, where kernels that loop over the columns cost 2.3x (BF16) to 2.8x (Q3_K,
// IQ4_XS) a token at 4 tokens on Qwen3.8-Flash-Next's dense products. Types: Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K,
// IQ4_XS, and Q4_0, Q5_0 and IQ4_NL in rows of whole 64-weight pairs of blocks (the shared expert's down projections);
// since Q713 the grid types IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_M (Swift's IQ attention, shared experts and
// ssm_out at 2 to 8 tokens; one token keeps their one-token Pascal kernels, IQ1_M llama.cpp's);
// plain 2-D products without fusion (MUL_MAT; the experts' MUL_MAT_ID stays with mmvq-pascal.cu).
//
// GGML_CUDA_MMVQ_PASCAL_KCOL: 0 off; 1 (default) 2 to 8 columns, and 1 column for Q3_K and IQ4_NL (which have no other
// Pascal kernel) and for Q8_0 rows whose 64-weight chunks aren't a multiple of 4 (the one-token kernel loads those 32
// bits at a time); 2 every product of 1 to 8 columns of these types. GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_K,q6_K: types
// left to the other kernels. GGML_CUDA_MMVQ_PASCAL_KCOL_R=1|2 rows per lane group, _KCOL_ITT=n iterations per activation
// tile, _KCOL_SMEM=bytes of shared memory per block (at most 49152), _KCOL_NSM=n the SM count and _KCOL_SMSM=bytes the
// shared memory per SM the shape is chosen for (=5 and =98304: a P106's shapes on another card): launch shapes for
// tuning. Any of R, ITT and SMEM selects the shape as before Q823 with those values (_KCOL_SMEM=49152:
// the default before Q823); without them the shape follows kcol::pick_shape's defaults (Q823: a third of the SM's shared
// memory per block where the registers allow a third block). In mode 1 a dispatch table (Q824, mmvq-pascal-kcol.cu)
// leaves the products of 2 to 8 columns that the other kernels run faster on a P106 to them, by type, row width and
// column count; GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 turns it off. GGML_CUDA_MMVQ_PASCAL=0 turns these kernels off with the
// other Pascal kernels, =2 runs them on any NVIDIA GPU (testing).
// GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1 (C805; 0 by default): the products of 2 to 8 columns of Q3_K, IQ4_NL, Q2_0 and Q8_0 in
// rows of chunks not a multiple of 4 (the types whose one-token sums are these kernels') are never left to the other
// kernels, whose sums differ: not by the table, and not where the columns' activations pass the shared-memory budget,
// where they run within 48 KB (in one launch where all columns fit, else in launches of fewer columns); =2 tries
// launches within the budget first. _KCOL_EXACT_FAULT=1 (checks): a split product's later launches read the first one's
// activations.

// Runs dst = src0 * src1 if these kernels take it; returns false (nothing launched) otherwise.
bool ggml_cuda_mmvq_pascal_kcol(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst);

// Whether the k-column kernels are on for weights of type (GGML_CUDA_MMVQ_PASCAL_KCOL not 0, the type one of theirs and
// not in GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP); the caller checks GGML_CUDA_MMVQ_PASCAL. For kernels that keep their sums
// (mmvq-pascal-q3kmt.cu: Q3_K's one-token kernel is the k-column kernel).
bool ggml_cuda_mmvq_pascal_kcol_on(ggml_type type);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace kcol { struct params; }

// The launchers per type (mmvq-pascal-kcol-<type>.cu, one file per type so that they compile in parallel).
void ggml_cuda_kcol_launch_q8_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q2_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q3_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q4_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q5_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q6_K(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq4_xs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q4_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_q5_0(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq4_nl(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq2_xxs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq2_xs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq2_s(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq3_xxs(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq3_s(const kcol::params & p, cudaStream_t stream);
void ggml_cuda_kcol_launch_iq1_m(const kcol::params & p, cudaStream_t stream);

// Per type: the resident blocks per SM that the registers of its kernel for ncols columns and R rows per lane group allow
// (kcol::kernel_blocks; the launch shape's blocks_fn).
int ggml_cuda_kcol_blocks_q8_0(int ncols, int R);
int ggml_cuda_kcol_blocks_q2_0(int ncols, int R);
int ggml_cuda_kcol_blocks_q3_K(int ncols, int R);
int ggml_cuda_kcol_blocks_q4_K(int ncols, int R);
int ggml_cuda_kcol_blocks_q5_K(int ncols, int R);
int ggml_cuda_kcol_blocks_q6_K(int ncols, int R);
int ggml_cuda_kcol_blocks_iq4_xs(int ncols, int R);
int ggml_cuda_kcol_blocks_q4_0(int ncols, int R);
int ggml_cuda_kcol_blocks_q5_0(int ncols, int R);
int ggml_cuda_kcol_blocks_iq4_nl(int ncols, int R);
int ggml_cuda_kcol_blocks_iq2_xxs(int ncols, int R);
int ggml_cuda_kcol_blocks_iq2_xs(int ncols, int R);
int ggml_cuda_kcol_blocks_iq2_s(int ncols, int R);
int ggml_cuda_kcol_blocks_iq3_xxs(int ncols, int R);
int ggml_cuda_kcol_blocks_iq3_s(int ncols, int R);
int ggml_cuda_kcol_blocks_iq1_m(int ncols, int R);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
