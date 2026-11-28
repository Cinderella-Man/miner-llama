#pragma once

#include "common.cuh"

// ---- card hops: through NCCL's point-to-point send and receive (GGML_CUDA_HOP_NCCL, 2026-11-28) ---------------------
//
// A copy between two cards of a layer split normally takes the driver's peer copy, the staged ring
// (GGML_CUDA_HOP_STAGED) or the mailbox (GGML_CUDA_HOP_MAILBOX). This path hands it to NCCL instead: one ncclSend on
// the source card's stream and one ncclRecv on the destination card's, issued together as one NCCL group, so NCCL picks
// its own transport for the pair: peer-to-peer where the cards reach each other, its shared host memory transport where
// they do not (a board whose switches and chipset drop traffic between cards), or a transport an NCCL build adds. The
// communicators (one per physical card, ranks = CUDA's device numbers) are made with ncclCommInitAll on the first hop.
// The destination's stream is ordered after the copy because the receive runs on it.
//   GGML_CUDA_HOP_NCCL   0 (default): off; 1: copies between cards without peer access; 2: every copy between two
//                        physical cards (for checks on boards that have peer access). It takes precedence over the
//                        mailbox and the staged ring for the copies it takes.
// Only in a build with NCCL (GGML_USE_NCCL: -DGGML_CUDA_NCCL=ON with NCCL found); in other builds the switch is read,
// logged once and ignored. NCCL's own variables (NCCL_BUFFSIZE, NCCL_MAX_NCHANNELS, NCCL_DEBUG ...) apply as usual.

// whether the switch asks for this pair (the physical devices); reads the switch once
bool ggml_cuda_hop_nccl_takes(int src_physical, int dst_physical);
// enqueues the send on stream_src (on src_physical) and the receive on stream_dst (on dst_physical) as one group;
// returns false, having enqueued nothing, when the build has no NCCL or the communicators could not be made
bool ggml_cuda_hop_nccl_copy(int src_physical, cudaStream_t stream_src, int dst_physical, cudaStream_t stream_dst,
        void * dst, const void * src, size_t nbytes);
// logs the hops it made and destroys the communicators; called from a CUDA context's destructor (once)
void ggml_cuda_hop_nccl_report(void);
