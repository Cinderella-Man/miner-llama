#pragma once

#include "common.cuh"

// ---- card hops: each hop's time in the engine (GGML_CUDA_HOP_TIMING, 2026-11-28) ------------------------------------
//
// What a hop between two cards costs inside the running engine, without CUDA timing events (on Pascal cards with a
// host-staged path they change the token: Miner Llama's registry row Q49) and without host callbacks on the streams.
// Per hop between two physical cards:
//   - the host's clock when it enqueues the hop (H);
//   - a one-thread kernel on the source card's stream, just before the hop, writes the card's %globaltimer into mapped
//     pinned host memory (S: the source card has finished the work that wrote the tensor);
//   - the same kernel on the destination card's stream, just after the hop is ordered there, writes that card's clock
//     (D: the destination may go on).
// Each card's clock is related to the host's CLOCK_MONOTONIC by a short calibration (a stamp kernel on a side stream,
// the host polling for it; the best of 16 round trips) on the first hop that touches the card and again after 5 s,
// so D - S is the hop's time from the source's last kernel to the destination's go, and S - H is how far ahead the host
// enqueued it (near the launch latency: the card waited for the host).
// Records are kept in a ring of 16,384 and summarised 8,192 hops after they were made (long complete by then), per
// boundary, per size class (decode <= 16 KiB, verify <= 64 KiB, short prompt <= 1 MiB, prompt) and per path (the
// driver's peer copy, the staged ring, the mailbox); the summary is logged when the first CUDA context is freed,
// at warning level, so that a server's default log keeps it.
//   GGML_CUDA_HOP_TIMING       0 (default): off, nothing is enqueued; 1: on
//   GGML_CUDA_HOP_TIMING_FILE  a CSV of every hop (src, dst, bytes, path, host ns at enqueue, S and D in host ns);
//                              a "%p" in the name becomes the process id
// Cost when on: two one-thread kernel launches per hop on the host's submit thread, their few microseconds on each
// card's stream, and the calibrations' host polls (~0.3 ms per card every 5 s): measure it as its own arm.

// returns the hop's record, or -1 when off; call before enqueuing the hop, with each side's current stream
int  ggml_cuda_hop_timing_begin(int dev_src, cudaStream_t stream_src, int dev_dst, size_t nbytes);
// path: 0 the driver's peer copy, 1 the staged ring, 2 the mailbox (with its gate); call once the destination's
// stream is ordered after the hop
void ggml_cuda_hop_timing_end(int rec, int dev_dst, cudaStream_t stream_dst, int path);
// logs the summary (and closes the file) once; called from a CUDA context's destructor
void ggml_cuda_hop_timing_report(void);
