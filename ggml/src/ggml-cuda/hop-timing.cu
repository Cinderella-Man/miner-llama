#include "hop-timing.cuh"
#include "keepalive.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <mutex>
#include <string>
#include <time.h>
#include <tuple>
#include <unistd.h>
#include <vector>

// GGML_CUDA_HOP_TIMING: see hop-timing.cuh

namespace {

struct hop_rec {
    unsigned long long s_gt;  // the source card's %globaltimer before the hop (written by the card)
    unsigned long long d_gt;  // the destination card's after it (written by the card)
    long long          h_ns;  // the host's CLOCK_MONOTONIC at enqueue
    unsigned int       bytes;
    short              src, dst;
    int                path;
    int                pad;
};

constexpr int       HOP_T_N    = 1 << 14;  // records in the ring
constexpr int       HOP_T_LAG  = 1 << 13;  // a record is summarised this many hops after it was made
constexpr int       HOP_T_CALS = 16;       // round trips per calibration
constexpr long long HOP_T_RECAL_NS = 5000000000ll;

struct hop_key_vals {
    std::vector<float> hop_us, lag_us;
};

struct hop_timing_state {
    std::mutex   mu;
    int          mode = -1;  // -1 not read yet, 0 off, 1 on
    bool         reported = false;
    hop_rec *    recs = nullptr;                  // pinned, mapped, portable
    unsigned long long * cal_word = nullptr;      // pinned, mapped, portable
    unsigned long long   seq = 0;
    long long    off[GGML_CUDA_MAX_DEVICES] = {};      // host ns = card ns + off
    long long    err[GGML_CUDA_MAX_DEVICES] = {};      // half the best round trip
    long long    cal_at[GGML_CUDA_MAX_DEVICES] = {};   // host ns of the last calibration, 0: never
    int          n_cal[GGML_CUDA_MAX_DEVICES] = {};
    cudaStream_t cal_stream[GGML_CUDA_MAX_DEVICES] = {};
    FILE *       out = nullptr;
    unsigned long long n_done = 0, n_incomplete = 0;
    std::map<std::tuple<int, int, int, int>, hop_key_vals> agg;  // src, dst, size class, path
};

static hop_timing_state & hop_t() {
    static hop_timing_state st;
    return st;
}

static long long now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long) t.tv_sec * 1000000000ll + t.tv_nsec;
}

static int size_class(unsigned int bytes) {
    return bytes <= 16384u ? 0 : bytes <= 65536u ? 1 : bytes <= (1u << 20) ? 2 : 3;
}

static const char * class_name(int c) {
    static const char * names[4] = {"decode", "verify", "short-prompt", "prompt"};
    return names[c];
}

static const char * path_name(int p) {
    static const char * names[3] = {"peer-copy", "staged", "mailbox"};
    return p >= 0 && p < 3 ? names[p] : "?";
}

} // namespace

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __global__ void k_hop_stamp(unsigned long long * p) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *(volatile unsigned long long *) p = t;
    __threadfence_system();
}
#endif

// relate one card's clock to the host's: the best (shortest) of HOP_T_CALS round trips
static void hop_t_calibrate(hop_timing_state & st, int dev) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_cuda_set_device(dev);
    if (st.cal_stream[dev] == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st.cal_stream[dev], cudaStreamNonBlocking));
    }
    volatile unsigned long long * w = st.cal_word;
    long long best_err = LLONG_MAX, best_off = 0;
    for (int i = 0; i < HOP_T_CALS; ++i) {
        *w = 0;
        const long long t0 = now_ns();
        k_hop_stamp<<<1, 1, 0, st.cal_stream[dev]>>>((unsigned long long *) st.cal_word);
        CUDA_CHECK(cudaGetLastError());
        long long t1;
        do {
            t1 = now_ns();  // the stamp landed at or before the check that follows the last of these
        } while (*w == 0 && t1 - t0 < 100000000ll);
        const unsigned long long gt = *w;
        CUDA_CHECK(cudaStreamSynchronize(st.cal_stream[dev]));
        if (gt == 0) {
            continue;
        }
        const long long e = (t1 - t0) / 2;
        if (e < best_err) {
            best_err = e;
            best_off = t0 + e - (long long) gt;
        }
    }
    if (best_err != LLONG_MAX) {
        st.off[dev] = best_off;
        st.err[dev] = best_err;
    }
    st.cal_at[dev] = now_ns();
    st.n_cal[dev]++;
    if (st.n_cal[dev] == 1) {
        GGML_LOG_WARN("%s: CUDA%d's clock related to the host's to within %lld ns\n", __func__, dev, best_err);
    }
#else
    GGML_UNUSED_VARS(st, dev);
#endif
}

// summarise one record if both stamps are in, then clear it for reuse
static void hop_t_take(hop_timing_state & st, hop_rec & r) {
    if (r.bytes == 0) {
        return;
    }
    if (r.s_gt == 0 || r.d_gt == 0) {
        st.n_incomplete++;
    } else {
        const long long s = (long long) r.s_gt + st.off[r.src];
        const long long d = (long long) r.d_gt + st.off[r.dst];
        const float hop_us = (float) (d - s) / 1000.0f;
        const float lag_us = (float) (s - r.h_ns) / 1000.0f;
        hop_key_vals & kv = st.agg[std::make_tuple((int) r.src, (int) r.dst, size_class(r.bytes), r.path)];
        kv.hop_us.push_back(hop_us);
        kv.lag_us.push_back(lag_us);
        st.n_done++;
        if (st.out != nullptr) {
            fprintf(st.out, "%d,%d,%u,%s,%lld,%lld,%lld\n", r.src, r.dst, r.bytes, path_name(r.path), r.h_ns, s, d);
        }
    }
    r.s_gt = r.d_gt = 0;
    r.bytes = 0;
}

int ggml_cuda_hop_timing_begin(int dev_src, cudaStream_t stream_src, int dev_dst, size_t nbytes) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(dev_src, stream_src, dev_dst, nbytes);
    return -1;
#else
    hop_timing_state & st = hop_t();
    if (st.mode == 0) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(st.mu);
    if (st.mode < 0) {
        const char * e = getenv("GGML_CUDA_HOP_TIMING");
        st.mode = (e != nullptr && atoi(e) != 0) ? 1 : 0;
        if (st.mode == 0) {
            return -1;
        }
        int cur = 0;
        CUDA_CHECK(cudaGetDevice(&cur));
        ggml_cuda_keepalive_quiesce();
        CUDA_CHECK(cudaHostAlloc((void **) &st.recs, HOP_T_N * sizeof(hop_rec), cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc((void **) &st.cal_word, 64, cudaHostAllocPortable | cudaHostAllocMapped));
        memset((void *) st.recs, 0, HOP_T_N * sizeof(hop_rec));
        // the file's name: a "%p" in it becomes the process id, so repeated server starts with one setting keep their files
        std::string path;
        if (const char * f = getenv("GGML_CUDA_HOP_TIMING_FILE"); f != nullptr && *f != '\0') {
            path = f;
            const size_t at = path.find("%p");
            if (at != std::string::npos) {
                path.replace(at, 2, std::to_string((long) getpid()));
            }
            st.out = fopen(path.c_str(), "w");
            if (st.out != nullptr) {
                fprintf(st.out, "src,dst,bytes,path,host_ns,src_done_ns,dst_go_ns\n");
            } else {
                GGML_LOG_WARN("%s: cannot open %s\n", __func__, path.c_str());
            }
        }
        GGML_LOG_WARN("%s: timing every card hop (a stamp kernel on each side, %d records)%s%s\n", __func__, HOP_T_N,
                st.out ? ", every hop to " : "", st.out ? path.c_str() : "");
        ggml_cuda_set_device(cur);
    }
    if (dev_src < 0 || dev_src >= GGML_CUDA_MAX_DEVICES || dev_dst < 0 || dev_dst >= GGML_CUDA_MAX_DEVICES ||
            nbytes == 0 || nbytes > 0xffffffffu) {
        return -1;
    }
    int cur = 0;
    CUDA_CHECK(cudaGetDevice(&cur));
    const long long t = now_ns();
    for (const int dev : {dev_src, dev_dst}) {
        if (st.cal_at[dev] == 0 || t - st.cal_at[dev] > HOP_T_RECAL_NS) {
            hop_t_calibrate(st, dev);
            break;  // at most one calibration per hop, so a late one does not stall a token for every card
        }
    }
    if (st.seq >= (unsigned long long) HOP_T_LAG) {
        hop_t_take(st, st.recs[(st.seq - HOP_T_LAG) % HOP_T_N]);
    }
    const int i = (int) (st.seq % HOP_T_N);
    st.seq++;
    hop_rec & r = st.recs[i];
    if (r.bytes != 0) {
        hop_t_take(st, r);  // only after a wrap with an unsummarised record: cannot happen with LAG < N
    }
    r.s_gt  = 0;
    r.d_gt  = 0;
    r.h_ns  = now_ns();
    r.bytes = (unsigned int) nbytes;
    r.src   = (short) dev_src;
    r.dst   = (short) dev_dst;
    r.path  = -1;
    ggml_cuda_set_device(dev_src);
    k_hop_stamp<<<1, 1, 0, stream_src>>>(&r.s_gt);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_set_device(cur);
    return i;
#endif
}

void ggml_cuda_hop_timing_end(int rec, int dev_dst, cudaStream_t stream_dst, int path) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(rec, dev_dst, stream_dst, path);
#else
    if (rec < 0) {
        return;
    }
    hop_timing_state & st = hop_t();
    std::lock_guard<std::mutex> lock(st.mu);
    hop_rec & r = st.recs[rec];
    r.path = path;
    int cur = 0;
    CUDA_CHECK(cudaGetDevice(&cur));
    ggml_cuda_set_device(dev_dst);
    k_hop_stamp<<<1, 1, 0, stream_dst>>>(&r.d_gt);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_set_device(cur);
#endif
}

void ggml_cuda_hop_timing_report(void) {
    hop_timing_state & st = hop_t();
    if (st.mode != 1) {
        return;
    }
    std::lock_guard<std::mutex> lock(st.mu);
    if (st.reported) {
        return;
    }
    st.reported = true;
    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    for (int d = 0; d < n_dev && d < GGML_CUDA_MAX_DEVICES; ++d) {
        if (st.cal_at[d] != 0) {
            ggml_cuda_set_device(d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    const unsigned long long first = st.seq > (unsigned long long) HOP_T_N ? st.seq - HOP_T_N : 0;
    for (unsigned long long s = first; s < st.seq; ++s) {
        hop_t_take(st, st.recs[s % HOP_T_N]);
    }
    GGML_LOG_WARN("%s: %llu card hops timed, %llu without both stamps; per boundary, size class and path: count, hop "
            "(source done -> destination go) median / p10 / p90 us, host ahead (enqueue -> source done) median us\n",
            __func__, st.n_done, st.n_incomplete);
    for (auto & kv : st.agg) {
        std::vector<float> & h = kv.second.hop_us;
        std::vector<float> & l = kv.second.lag_us;
        std::sort(h.begin(), h.end());
        std::sort(l.begin(), l.end());
        auto q = [](const std::vector<float> & v, double p) { return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))]; };
        GGML_LOG_WARN("%s:   CUDA%d -> CUDA%d %-12s %-9s n %7zu  hop %8.1f / %8.1f / %8.1f  host ahead %9.1f\n", __func__,
                std::get<0>(kv.first), std::get<1>(kv.first), class_name(std::get<2>(kv.first)),
                path_name(std::get<3>(kv.first)), h.size(), q(h, 0.5), q(h, 0.1), q(h, 0.9), q(l, 0.5));
    }
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        if (st.n_cal[d] != 0) {
            GGML_LOG_WARN("%s:   CUDA%d: %d calibrations, the last to within %lld ns\n", __func__, d, st.n_cal[d], st.err[d]);
        }
    }
    if (st.out != nullptr) {
        fclose(st.out);
        st.out = nullptr;
    }
}
