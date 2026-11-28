#include "hop-nccl.cuh"
#include "keepalive.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif

// GGML_CUDA_HOP_NCCL: see hop-nccl.cuh

namespace {

struct hop_nccl_state {
    std::mutex mu;
    int        mode       = -1;  // -1 not read yet, 0 off, 1 pairs without peer access, 2 every pair
    bool       init_tried = false;
    bool       ok         = false;
    bool       reported   = false;
    int        n_dev      = 0;
#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t> comms;
#endif
    signed char peer[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};  // 0 unknown, 1 no peer access, 2 peer access
    int64_t    n_hops = 0, n_bytes = 0;
};

} // namespace

static hop_nccl_state & hop_n() {
    static hop_nccl_state st;
    return st;
}

static int hop_nccl_mode(hop_nccl_state & st) {
    if (st.mode < 0) {
        const char * e = getenv("GGML_CUDA_HOP_NCCL");
        st.mode = e != nullptr ? std::min(std::max(atoi(e), 0), 2) : 0;
#ifndef GGML_USE_NCCL
        if (st.mode != 0) {
            GGML_LOG_WARN("%s: GGML_CUDA_HOP_NCCL=%d ignored: this build has no NCCL (-DGGML_CUDA_NCCL=ON)\n", __func__, st.mode);
            st.mode = 0;
        }
#endif
    }
    return st.mode;
}

bool ggml_cuda_hop_nccl_takes(int src_physical, int dst_physical) {
    hop_nccl_state & st = hop_n();
    if (st.mode == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(st.mu);
    const int mode = hop_nccl_mode(st);
    if (mode == 0 || src_physical < 0 || dst_physical < 0 || src_physical >= GGML_CUDA_MAX_DEVICES ||
            dst_physical >= GGML_CUDA_MAX_DEVICES || src_physical == dst_physical) {
        return false;
    }
    if (mode == 2) {
        return true;
    }
    signed char & p = st.peer[src_physical][dst_physical];
    if (p == 0) {
        int can = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can, dst_physical, src_physical));
        p = can ? 2 : 1;
    }
    return p == 1;
}

bool ggml_cuda_hop_nccl_copy(int src_physical, cudaStream_t stream_src, int dst_physical, cudaStream_t stream_dst,
        void * dst, const void * src, size_t nbytes) {
#ifndef GGML_USE_NCCL
    GGML_UNUSED_VARS(src_physical, stream_src, dst_physical, stream_dst, dst, src, nbytes);
    return false;
#else
    hop_nccl_state & st = hop_n();
    std::lock_guard<std::mutex> lock(st.mu);
    int cur = 0;
    CUDA_CHECK(cudaGetDevice(&cur));
    if (!st.init_tried) {
        st.init_tried = true;
        CUDA_CHECK(cudaGetDeviceCount(&st.n_dev));
        std::vector<int> devs(st.n_dev);
        for (int i = 0; i < st.n_dev; ++i) {
            devs[i] = i;
        }
        st.comms.resize(st.n_dev);
        ggml_cuda_keepalive_quiesce();
        const ncclResult_t rc = ncclCommInitAll(st.comms.data(), st.n_dev, devs.data());
        st.ok = rc == ncclSuccess;
        int version = 0;
        ncclGetVersion(&version);
        if (st.ok) {
            GGML_LOG_WARN("%s: card hops through NCCL %d (communicators for %d cards)\n", __func__, version, st.n_dev);
        } else {
            st.comms.clear();
            GGML_LOG_WARN("%s: ncclCommInitAll failed (%s): card hops take the other paths\n", __func__, ncclGetErrorString(rc));
        }
        ggml_cuda_set_device(cur);
    }
    if (!st.ok || src_physical >= st.n_dev || dst_physical >= st.n_dev) {
        return false;
    }
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(src, nbytes, ncclUint8, dst_physical, st.comms[src_physical], stream_src));
    NCCL_CHECK(ncclRecv(dst, nbytes, ncclUint8, src_physical, st.comms[dst_physical], stream_dst));
    NCCL_CHECK(ncclGroupEnd());
    ggml_cuda_set_device(cur);
    st.n_hops++;
    st.n_bytes += (int64_t) nbytes;
    return true;
#endif
}

void ggml_cuda_hop_nccl_report(void) {
#ifdef GGML_USE_NCCL
    hop_nccl_state & st = hop_n();
    if (st.mode <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(st.mu);
    if (st.reported || !st.ok) {
        return;
    }
    st.reported = true;
    GGML_LOG_WARN("%s: %lld card hops through NCCL, %.2f MiB\n", __func__, (long long) st.n_hops, st.n_bytes / 1048576.0);
    for (int i = 0; i < st.n_dev; ++i) {
        ggml_cuda_set_device(i);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (ncclComm_t c : st.comms) {
        ncclCommDestroy(c);
    }
    st.comms.clear();
    st.ok = false;
#endif
}
