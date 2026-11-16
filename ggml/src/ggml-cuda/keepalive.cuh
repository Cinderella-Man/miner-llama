#pragma once

// GGML_CUDA_KEEPALIVE: a small block per card that holds the card's boost clock while a layer split decodes. The
// settings and the controller are in keepalive-policy.h; this is what the rest of the CUDA backend calls. Every call
// returns at once while the switch is off.

#include "keepalive-policy.h"

// devices are physical CUDA devices (ggml_cuda_get_physical_device), the cards the blocks run on

// before a graph's work is enqueued on the device; capture: a CUDA graph capture follows
void ggml_cuda_keepalive_graph_begin(int device, bool capture);

// after a graph's work is enqueued on the device
void ggml_cuda_keepalive_graph_end(int device);

// the same, with the stream (cudaStream_t) the graph ran on: a start GGML_CUDA_KEEPALIVE_EAGER holds waits for the
// stream's work so far
void ggml_cuda_keepalive_graph_end_after(int device, void * stream);

// GGML_CUDA_KEEPALIVE_EAGER (C811): its level while the keep-alive is on (keepalive-policy.h), else -1; returns -1 at
// once while the switch is unset
int ggml_cuda_keepalive_eager_level();

// before an evaluation that runs kernel by kernel on the device (not a capture, not a replay); shape: its shapes as one
// number (unused below level 1)
void ggml_cuda_keepalive_eager_begin(int device, uint64_t shape);

// before a set, get, memset or copy the backend runs on the device outside a graph (ggml_cuda_keepalive_outside_kind)
void ggml_cuda_keepalive_outside(int device, int kind, size_t size);

// the counts since the start as one line, written into buf; 0 (nothing written) while the switch is unset or the
// keep-alive off. The CUDA registry's "ggml_backend_cuda_keepalive_counts", which llama-server prints after each reply
size_t ggml_backend_cuda_keepalive_counts(char * buf, size_t size);

// before anything that waits for every stream of a card or of every card (cudaFree, cudaFreeHost, cudaHostAlloc,
// cudaMallocHost, cudaHostRegister/Unregister, cuMemUnmap, cudaDeviceSynchronize): every block ended
void ggml_cuda_keepalive_quiesce();

// at a backend's free: the card's runs and the SM clock they measured, logged and reset
void ggml_cuda_keepalive_log(int device);

// llama-server took a request (GGML_CUDA_KEEPALIVE=request); the CUDA registry's "ggml_backend_cuda_keepalive_kick"
void ggml_backend_cuda_keepalive_kick(void);
