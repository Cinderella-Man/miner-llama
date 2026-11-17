#include "hop-mailbox.cuh"
#include "keepalive.cuh"

#include <cinttypes>
#include <cstring>

// The ring's control words: per slot a flag line (the sequence number of the chunk the slot holds, written by the
// source card) and an acknowledgement line (the sequence number of the chunk last read from it, written by this card),
// each in its own 128-byte line, then a line that a kernel which timed out fills (what, the sequence number it waited
// for, the value it saw). Chunk q goes into slot q & (n_slots - 1); sequence numbers wrap at 2^32 and are compared by
// their signed difference.
#define HOP_MB_LINE 32

struct ggml_cuda_hop_mailbox {
    int    dev_src     = -1; // the backends' (possibly virtual) devices
    int    dev_dst     = -1;
    int    n_chunks    = 8;
    int    n_slots     = 16;
    size_t slot_size   = 0;
    size_t max_bytes   = 0;
    int    stream_mode = 1;
    int    threads     = 256;

    long long          poll_src   = 0; // the longest pause between two polls, in each card's clock cycles
    long long          poll_dst   = 0;
    unsigned long long timeout_ns = 0;

    char     * ring  = nullptr; // n_slots * slot_size bytes, pinned, mapped, portable
    unsigned * words = nullptr; // (2 n_slots + 1) lines: the flags, the acknowledgements, the timeout line

    // the same memory as each card addresses it
    char     * ring_src  = nullptr;
    char     * ring_dst  = nullptr;
    unsigned * words_src = nullptr;
    unsigned * words_dst = nullptr;

    uint32_t next_seq = 1; // the sequence number of the next chunk

    cudaStream_t copy_stream = nullptr; // on dev_dst, made on first use
    cudaEvent_t  copy_done   = nullptr; // on dev_dst: the copy stream's reader of a hop is done

    int64_t n_hops = 0, n_bytes = 0, n_chunk_count = 0, n_on_copy_stream = 0, n_guarded = 0;

    // GGML_CUDA_HOP_MAILBOX_GATE_KB: hops above gate_bytes go through gate buffers ordered by events, no polling
    size_t gate_bytes = 0;
    size_t gate_size  = 0;                  // bytes per gate buffer
    char * gate_host  = nullptr;            // HOP_MB_GATE_BUFS * gate_size bytes, pinned, mapped, portable
    char * gate_src   = nullptr;            // the same memory as each card addresses it
    char * gate_dst   = nullptr;
    int    gate_next  = 0;
    cudaEvent_t gate_written[4] = {};       // on dev_src: the writer into the buffer is done
    cudaEvent_t gate_read[4]    = {};       // on dev_dst: the reader out of the buffer is done
    bool        gate_used[4]    = {};
    int64_t n_gated = 0, n_gate_too_big = 0;

    unsigned * timeout_line() const { return words + (size_t) 2*n_slots*HOP_MB_LINE; }
};

#define HOP_MB_GATE_BUFS 4

struct hop_mb_args {
    char       * ring;
    unsigned   * flags;
    unsigned   * acks;
    unsigned   * err;
    const char * src;   // the writer's source tensor
    char       * dst;   // the reader's destination tensor
    size_t       nbytes;
    size_t       slot_size;
    unsigned     chunk; // bytes, a multiple of 256
    unsigned     n;     // chunks in this hop
    unsigned     mask;  // n_slots - 1
    unsigned     seq0;  // the sequence number of the hop's first chunk

    long long          poll;
    unsigned long long timeout_ns;
};

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ unsigned long long hop_mb_time_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// a load from mapped host memory that goes over PCIe every time
static __device__ __forceinline__ unsigned hop_mb_peek(const unsigned * p) {
    return *(const volatile unsigned *) p;
}

// (one thread) waits until the word at p reaches want in sequence order: one poll at once, then polls with pauses that
// double up to a.poll cycles. A wait longer than a.timeout_ns fills the timeout line and traps
static __device__ void hop_mb_wait(const unsigned * p, const unsigned want, const hop_mb_args & a, const unsigned what) {
    unsigned v = hop_mb_peek(p);
    if ((int) (v - want) >= 0) {
        return;
    }
    const unsigned long long t0 = hop_mb_time_ns();
    long long pause = 32;
    for (;;) {
        const long long c0 = clock64();
        while (clock64() - c0 < pause) {
        }
        v = hop_mb_peek(p);
        if ((int) (v - want) >= 0) {
            return;
        }
        pause = min(2*pause, a.poll);
        if (hop_mb_time_ns() - t0 > a.timeout_ns) {
            volatile unsigned * e = a.err;
            e[1] = want;
            e[2] = v;
            e[0] = what;
            __threadfence_system();
            printf("ggml-cuda hop mailbox: the %s kernel timed out waiting for sequence number %u (the word holds %u)\n",
                   what == 1 ? "writing" : "reading", want, v);
            __trap();
        }
    }
}

// on the source card: writes the hop's chunks into the ring, each after its slot's previous chunk was read, and raises
// each chunk's flag after its bytes
static __global__ void __launch_bounds__(1024) k_hop_mb_send(const hop_mb_args a) {
    __shared__ int lap0_free;
    const unsigned n_slots = a.mask + 1;
    const unsigned lap0    = min(a.n, n_slots);

    // the slots of the first lap were last filled by earlier hops: their acknowledgements in one round trip
    if (threadIdx.x == 0) {
        lap0_free = 1;
    }
    __syncthreads();
    for (unsigned c = threadIdx.x; c < lap0; c += blockDim.x) {
        const unsigned q = a.seq0 + c;
        if ((int) (hop_mb_peek(a.acks + (q & a.mask)*HOP_MB_LINE) - (q - n_slots)) < 0) {
            lap0_free = 0;
        }
    }
    __syncthreads();
    const bool wait_lap0 = !lap0_free;

    for (unsigned c = 0; c < a.n; ++c) {
        const unsigned q = a.seq0 + c;
        const unsigned s = q & a.mask;
        if (c >= n_slots || wait_lap0) {
            if (threadIdx.x == 0) {
                hop_mb_wait(a.acks + s*HOP_MB_LINE, q - n_slots, a, 1);
            }
            __syncthreads();
        }
        const size_t   off  = (size_t) c * a.chunk;
        const unsigned n16  = (unsigned) (min((size_t) a.chunk, a.nbytes - off) / 16);
        const uint4  * from = (const uint4 *) (a.src + off);
        uint4        * to   = (uint4 *) (a.ring + (size_t) s * a.slot_size);
        for (unsigned i = threadIdx.x; i < n16; i += blockDim.x) {
            to[i] = from[i];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            *(volatile unsigned *) (a.flags + s*HOP_MB_LINE) = q;
        }
    }
}

// on the destination card: waits for the next chunk, reads it and every later chunk already in the ring (warp 0 looks
// at their flags in one round trip) straight from the ring, then acknowledges their slots
static __global__ void __launch_bounds__(1024) k_hop_mb_recv(const hop_mb_args a) {
    __shared__ unsigned batch;
    const unsigned n_slots = a.mask + 1;
    const unsigned chunk16 = a.chunk / 16;

    for (unsigned c = 0; c < a.n; ) {
        if (threadIdx.x < WARP_SIZE) {
            const unsigned lane = threadIdx.x;
            if (lane == 0) {
                hop_mb_wait(a.flags + ((a.seq0 + c) & a.mask)*HOP_MB_LINE, a.seq0 + c, a, 2);
            }
            __syncwarp();
            // only within a lap: chunk c + n_slots shares chunk c's slot
            bool here = lane == 0;
            const unsigned j = c + lane;
            if (lane > 0 && j < a.n && lane < n_slots) {
                const unsigned q = a.seq0 + j;
                here = (int) (hop_mb_peek(a.flags + (q & a.mask)*HOP_MB_LINE) - q) >= 0;
            }
            const unsigned ready = __ballot_sync(0xffffffff, here);
            __threadfence_system();
            if (lane == 0) {
                batch = ~ready == 0 ? WARP_SIZE : __ffs(~ready) - 1;
            }
        }
        __syncthreads();
        const unsigned k     = batch;
        const unsigned units = k * chunk16;
        for (unsigned u = threadIdx.x; u < units; u += blockDim.x) {
            const unsigned jj  = u / chunk16;
            const unsigned w   = u - jj*chunk16;
            const size_t   off = (size_t) (c + jj) * a.chunk + (size_t) w*16;
            if (off >= a.nbytes) {
                continue; // the hop's last chunk may be short
            }
            const unsigned s = (a.seq0 + c + jj) & a.mask;
            const uint4 * from = (const uint4 *) (a.ring + (size_t) s * a.slot_size) + w;
            *(uint4 *) (a.dst + off) = __ldcv(from);
        }
        // every thread's loads have returned (their values are stored): the slots may be refilled
        __syncthreads();
        if (threadIdx.x < k) {
            const unsigned q = a.seq0 + c + threadIdx.x;
            *(volatile unsigned *) (a.acks + (q & a.mask)*HOP_MB_LINE) = q;
        }
        c += k;
    }
}

// GGML_CUDA_HOP_MAILBOX_GATE_KB: a plain copy of n16 16-byte units, device <-> mapped host memory. Nothing waits in it:
// events order it after the data it reads is complete (volatile loads so that a reader sees the host memory)
static __global__ void k_hop_mb_gate_copy(const uint4 * from, uint4 * to, const size_t n16) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x*blockDim.x) {
        to[i] = __ldcv(from + i);
    }
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static int hop_mb_env(const char * name, int def, int lo, int hi) {
    const char * env = getenv(name);
    if (env == nullptr || *env == '\0') {
        return def;
    }
    return std::min(std::max(atoi(env), lo), hi);
}

// the current card's clock in kHz, for the poll pause in cycles
static int hop_mb_clock_khz() {
    int dev = 0;
    int khz = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
    return khz > 0 ? khz : 1000000;
}

static ggml_cuda_hop_mailbox * hop_mb_get(ggml_backend_cuda_context * ctx_src, ggml_backend_cuda_context * ctx_dst) {
    ggml_cuda_hop_mailbox *& mb = ctx_dst->hop_mailboxes[ctx_src->device];
    if (mb != nullptr) {
        return mb;
    }
    mb = new ggml_cuda_hop_mailbox;
    mb->dev_src     = ctx_src->device;
    mb->dev_dst     = ctx_dst->device;
    mb->n_chunks    = hop_mb_env("GGML_CUDA_HOP_MAILBOX_CHUNKS",  8,   1,   256);
    mb->n_slots     = hop_mb_env("GGML_CUDA_HOP_MAILBOX_SLOTS",  16,   2,   256);
    mb->slot_size   = (size_t) hop_mb_env("GGML_CUDA_HOP_MAILBOX_SLOT_KB", 32, 1, 4096) * 1024;
    mb->max_bytes   = (size_t) hop_mb_env("GGML_CUDA_HOP_MAILBOX_MAX_KB",   0, 0, 1 << 22) * 1024;
    mb->gate_bytes  = (size_t) hop_mb_env("GGML_CUDA_HOP_MAILBOX_GATE_KB",  0, 0, 1 << 22) * 1024;
    mb->gate_size   = (size_t) hop_mb_env("GGML_CUDA_HOP_MAILBOX_GATE_BUF_KB", 4096, 64, 1 << 20) * 1024;
    mb->stream_mode = hop_mb_env("GGML_CUDA_HOP_MAILBOX_STREAM",  1,   0,     2);
    mb->threads     = hop_mb_env("GGML_CUDA_HOP_MAILBOX_THREADS", 256, 64, 1024) / WARP_SIZE * WARP_SIZE;
    const int poll_us    = hop_mb_env("GGML_CUDA_HOP_MAILBOX_POLL_US",    4, 0, 1000);
    const int timeout_ms = hop_mb_env("GGML_CUDA_HOP_MAILBOX_TIMEOUT_MS", 60000, 1, 3600000);
    mb->timeout_ns  = (unsigned long long) timeout_ms * 1000000ull;
    while ((mb->n_slots & (mb->n_slots - 1)) != 0) {
        mb->n_slots &= mb->n_slots - 1; // a power of two: the slot is the sequence number's low bits
    }

    const size_t n_words = (size_t) (2*mb->n_slots + 1) * HOP_MB_LINE;
    ggml_cuda_set_device(mb->dev_dst);
    ggml_cuda_keepalive_quiesce();
    CUDA_CHECK(cudaHostAlloc((void **) &mb->ring, (size_t) mb->n_slots * mb->slot_size,
            cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &mb->words, n_words * sizeof(unsigned), cudaHostAllocPortable | cudaHostAllocMapped));
    memset(mb->words, 0, n_words * sizeof(unsigned));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &mb->ring_dst,  mb->ring,  0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &mb->words_dst, mb->words, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&mb->copy_done, cudaEventDisableTiming));
    mb->poll_dst = (long long) poll_us * hop_mb_clock_khz() / 1000;
    ggml_cuda_set_device(mb->dev_src);
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &mb->ring_src,  mb->ring,  0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &mb->words_src, mb->words, 0));
    mb->poll_src = (long long) poll_us * hop_mb_clock_khz() / 1000;

    GGML_LOG_INFO("%s: card hop %s -> %s by mailbox: %d slots of %zu KiB in mapped host memory, %d chunks, %d threads, "
            "polls at most %d us apart, copy stream mode %d%s\n", __func__, ctx_src->name.c_str(), ctx_dst->name.c_str(),
            mb->n_slots, mb->slot_size / 1024, mb->n_chunks, mb->threads, poll_us, mb->stream_mode,
            mb->max_bytes ? (", hops up to " + std::to_string(mb->max_bytes / 1024) + " KiB").c_str() : "");
    return mb;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// GGML_CUDA_HOP_MAILBOX_GATE_KB (2026-11-16): a large hop (a prompt micro-batch) without polling. A polling reader is
// enqueued long before its source card gets there (the scheduler runs up to its copies ahead), so its wait grows with
// the context and at ~50k+ it passed the 1 s timeout and trapped. Here each card's kernel only runs once its
// data is complete: the source card copies into one of HOP_MB_GATE_BUFS pinned mapped buffers (after that buffer's
// previous reader is done, by event), records an event; this card's copy stream waits on it (and on the scheduler's
// guard) and copies out; this card's compute stream waits on that. Nothing waits inside a kernel, so no timeout exists.
// Returns false (nothing enqueued) for a hop above the buffer size.
static bool hop_mb_gated_copy(ggml_cuda_hop_mailbox * mb, ggml_backend_cuda_context * ctx_src,
        ggml_backend_cuda_context * ctx_dst, char * dst, const char * src, const size_t nbytes) {
    if (nbytes > mb->gate_size) {
        mb->n_gate_too_big++;
        return false;
    }
    if (mb->gate_host == nullptr) {
        ggml_cuda_set_device(mb->dev_dst);
        ggml_cuda_keepalive_quiesce();
        CUDA_CHECK(cudaHostAlloc((void **) &mb->gate_host, HOP_MB_GATE_BUFS * mb->gate_size,
                cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &mb->gate_dst, mb->gate_host, 0));
        for (int b = 0; b < HOP_MB_GATE_BUFS; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&mb->gate_read[b], cudaEventDisableTiming));
        }
        if (mb->copy_stream == nullptr) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&mb->copy_stream, cudaStreamNonBlocking));
        }
        ggml_cuda_set_device(mb->dev_src);
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &mb->gate_src, mb->gate_host, 0));
        for (int b = 0; b < HOP_MB_GATE_BUFS; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&mb->gate_written[b], cudaEventDisableTiming));
        }
        GGML_LOG_WARN("%s: card hop CUDA%d -> %s: hops above %zu KiB through %d gate buffers of %zu KiB, ordered by events\n",
                __func__, mb->dev_src, ctx_dst->name.c_str(), mb->gate_bytes / 1024, HOP_MB_GATE_BUFS, mb->gate_size / 1024);
    }
    const int b = mb->gate_next;
    mb->gate_next = (b + 1) % HOP_MB_GATE_BUFS;
    const size_t n16     = nbytes / 16;
    const int    threads = 256;
    const int    blocks  = (int) std::min<size_t>((n16 + threads - 1) / threads, 64);

    // the source card, after the work that wrote src and after the buffer's previous reader
    ggml_cuda_set_device(mb->dev_src);
    if (mb->gate_used[b]) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx_src->stream(), mb->gate_read[b], 0));
    }
    k_hop_mb_gate_copy<<<blocks, threads, 0, ctx_src->stream()>>>((const uint4 *) src,
            (uint4 *) (mb->gate_src + (size_t) b * mb->gate_size), n16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(mb->gate_written[b], ctx_src->stream()));

    // this card's copy stream, after the scheduler's guard for the input copy it writes and after the writer
    ggml_cuda_set_device(mb->dev_dst);
    cudaStream_t s = mb->copy_stream;
    if (ctx_dst->hop_guard != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(s, ctx_dst->hop_guard, 0));
        mb->n_guarded++;
    }
    ctx_dst->hop_guard = nullptr;
    CUDA_CHECK(cudaStreamWaitEvent(s, mb->gate_written[b], 0));
    k_hop_mb_gate_copy<<<blocks, threads, 0, s>>>((const uint4 *) (mb->gate_dst + (size_t) b * mb->gate_size),
            (uint4 *) dst, n16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(mb->gate_read[b], s));
    CUDA_CHECK(cudaStreamWaitEvent(ctx_dst->stream(), mb->gate_read[b], 0));
    mb->gate_used[b] = true;

    mb->n_hops++;
    mb->n_gated++;
    mb->n_bytes += nbytes;
    return true;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_hop_mailbox_copy(ggml_backend_cuda_context * ctx_src, ggml_backend_cuda_context * ctx_dst, char * dst,
        const char * src, const size_t nbytes) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx_src, ctx_dst, dst, src, nbytes);
    return false;
#else
    if (nbytes == 0 || nbytes % 16 != 0 || ((uintptr_t) dst | (uintptr_t) src) % 16 != 0) {
        return false;
    }
    ggml_cuda_hop_mailbox * mb = hop_mb_get(ctx_src, ctx_dst);
    if (mb->gate_bytes != 0 && nbytes > mb->gate_bytes) {
        return hop_mb_gated_copy(mb, ctx_src, ctx_dst, dst, src, nbytes);
    }
    if (mb->max_bytes != 0 && nbytes > mb->max_bytes) {
        return false;
    }
    const volatile unsigned * timeout_line = mb->timeout_line();
    if (timeout_line[0] != 0) {
        GGML_ABORT("card hop CUDA%d -> %s by mailbox: the %s kernel timed out waiting for sequence number %u (saw %u)",
                mb->dev_src, ctx_dst->name.c_str(), timeout_line[0] == 1 ? "writing" : "reading", timeout_line[1],
                timeout_line[2]);
    }

    size_t chunk = (nbytes + mb->n_chunks - 1) / mb->n_chunks;
    chunk = std::min(GGML_PAD(chunk, 256), mb->slot_size);
    const size_t n = (nbytes + chunk - 1) / chunk;
    GGML_ASSERT(n < (1u << 30));

    hop_mb_args a;
    a.src        = src;
    a.dst        = dst;
    a.nbytes     = nbytes;
    a.slot_size  = mb->slot_size;
    a.chunk      = (unsigned) chunk;
    a.n          = (unsigned) n;
    a.mask       = (unsigned) mb->n_slots - 1;
    a.seq0       = mb->next_seq;
    a.timeout_ns = mb->timeout_ns;
    mb->next_seq += (uint32_t) n;

    const bool on_copy_stream = mb->stream_mode == 2 || (mb->stream_mode == 1 && n > (size_t) mb->n_slots);

    // the source card's kernel, after the work that wrote src
    ggml_cuda_set_device(mb->dev_src);
    a.ring  = mb->ring_src;
    a.flags = mb->words_src;
    a.acks  = mb->words_src + (size_t) mb->n_slots*HOP_MB_LINE;
    a.err   = mb->words_src + (size_t) 2*mb->n_slots*HOP_MB_LINE;
    a.poll  = mb->poll_src;
    k_hop_mb_send<<<1, mb->threads, 0, ctx_src->stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());

    // this card's
    ggml_cuda_set_device(mb->dev_dst);
    cudaStream_t s_dst = ctx_dst->stream();
    if (on_copy_stream) {
        if (mb->copy_stream == nullptr) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&mb->copy_stream, cudaStreamNonBlocking));
        }
        s_dst = mb->copy_stream;
        // the copy stream must not write the input copy while an earlier evaluation on this card still reads it: the
        // scheduler made the compute stream wait on that guard just before this copy
        if (ctx_dst->hop_guard != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(s_dst, ctx_dst->hop_guard, 0));
            mb->n_guarded++;
        }
        mb->n_on_copy_stream++;
    }
    ctx_dst->hop_guard = nullptr;
    a.ring  = mb->ring_dst;
    a.flags = mb->words_dst;
    a.acks  = mb->words_dst + (size_t) mb->n_slots*HOP_MB_LINE;
    a.err   = mb->words_dst + (size_t) 2*mb->n_slots*HOP_MB_LINE;
    a.poll  = mb->poll_dst;
    k_hop_mb_recv<<<1, mb->threads, 0, s_dst>>>(a);
    CUDA_CHECK(cudaGetLastError());
    if (on_copy_stream) {
        CUDA_CHECK(cudaEventRecord(mb->copy_done, s_dst));
        CUDA_CHECK(cudaStreamWaitEvent(ctx_dst->stream(), mb->copy_done, 0));
    }

    mb->n_hops++;
    mb->n_bytes += nbytes;
    mb->n_chunk_count += n;
    return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

void ggml_cuda_hop_mailbox_free(ggml_cuda_hop_mailbox * mb, ggml_backend_cuda_context * ctx_dst) {
    GGML_LOG_INFO("%s: card hop CUDA%d -> %s by mailbox: %" PRId64 " hops, %.2f MiB, %" PRId64 " chunks, %" PRId64
            " on the copy stream (%" PRId64 " behind the scheduler's guard)\n", __func__, mb->dev_src,
            ctx_dst->name.c_str(), mb->n_hops, mb->n_bytes / 1048576.0, mb->n_chunk_count, mb->n_on_copy_stream,
            mb->n_guarded);
    // the reading kernels run on this card's streams and wait for every chunk the writing kernels raise: once they are
    // done, neither card touches the ring
    ggml_cuda_set_device(mb->dev_dst);
    for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
        if (ctx_dst->streams[mb->dev_dst][i] != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(ctx_dst->streams[mb->dev_dst][i]));
        }
    }
    if (mb->copy_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(mb->copy_stream));
        CUDA_CHECK(cudaStreamDestroy(mb->copy_stream));
    }
    CUDA_CHECK(cudaEventDestroy(mb->copy_done));
    if (mb->gate_host != nullptr) {
        GGML_LOG_WARN("%s: card hop CUDA%d -> %s: %" PRId64 " hops through the gate buffers, %" PRId64 " too big for them\n",
                __func__, mb->dev_src, ctx_dst->name.c_str(), mb->n_gated, mb->n_gate_too_big);
        for (int b = 0; b < HOP_MB_GATE_BUFS; ++b) {
            CUDA_CHECK(cudaEventDestroy(mb->gate_read[b]));
        }
        ggml_cuda_set_device(mb->dev_src);
        CUDA_CHECK(cudaDeviceSynchronize()); // the writers into the gate buffers
        for (int b = 0; b < HOP_MB_GATE_BUFS; ++b) {
            CUDA_CHECK(cudaEventDestroy(mb->gate_written[b]));
        }
        ggml_cuda_set_device(mb->dev_dst);
    }
    const volatile unsigned * timeout_line = mb->timeout_line();
    if (timeout_line[0] != 0) {
        GGML_LOG_ERROR("%s: card hop CUDA%d -> %s by mailbox: the %s kernel timed out waiting for sequence number %u "
                "(saw %u)\n", __func__, mb->dev_src, ctx_dst->name.c_str(), timeout_line[0] == 1 ? "writing" : "reading",
                timeout_line[1], timeout_line[2]);
    }
    ggml_cuda_keepalive_quiesce();
    CUDA_CHECK(cudaFreeHost(mb->words));
    CUDA_CHECK(cudaFreeHost(mb->ring));
    if (mb->gate_host != nullptr) {
        CUDA_CHECK(cudaFreeHost(mb->gate_host));
    }
    delete mb;
}
