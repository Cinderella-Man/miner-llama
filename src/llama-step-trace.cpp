#include "llama-step-trace.h"

#include "llama-ext.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// The file's first line is a header with the steady clock's time of the trace's 0 ms; then one JSON line per step:
//   {"step":<n>,"write_us":<the previous line's write>,"n":{<count>:<value>,...},"t":[[<mark>,<ms>],...]}
// in the order the marks were made, ms since the trace's 0 at microsecond resolution. GGML_CUDA_PROFILE_HOST's log line
// gives its own 0 ms on the same clock, so a profile time p is at p + t0_profile - t0_trace on this trace's scale.

namespace {

struct step_trace {
    struct mark {
        char         role; // 0: the name is printed as given
        const char * name;
        int64_t      t_ns;
    };

    FILE * f = nullptr;
    std::mutex mutex;
    std::chrono::steady_clock::time_point t0;
    std::vector<mark> marks;
    std::vector<std::pair<const char *, int64_t>> counts;
    bool    decoded  = false; // a context marked a stage since the step began
    int64_t n_step   = 0;
    int64_t write_us = -1;
    std::string line;

    int64_t now_ns() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }
};

step_trace * step_trace_get() {
    static step_trace * st = []() -> step_trace * {
        const char * path = getenv("LLAMA_STEP_TRACE");
        if (path == nullptr || path[0] == '\0') {
            return nullptr;
        }
        FILE * f = fopen(path, "w");
        if (f == nullptr) {
            LLAMA_LOG_ERROR("%s: cannot write LLAMA_STEP_TRACE=%s, no step trace\n", __func__, path);
            return nullptr;
        }
        auto * s = new step_trace;
        s->f  = f;
        s->t0 = std::chrono::steady_clock::now();
        s->marks.reserve(1024);
        s->line.reserve(16384);
        const double t0_ms = std::chrono::duration<double, std::milli>(s->t0.time_since_epoch()).count();
        fprintf(f, "{\"trace\":\"llama-step\",\"clock\":\"steady\",\"t0_steady_ms\":%.3f}\n", t0_ms);
        fflush(f);
        LLAMA_LOG_INFO("%s: host timestamps of every step into %s (LLAMA_STEP_TRACE; its 0 ms is %.3f ms on the "
                "steady clock)\n", __func__, path, t0_ms);
        return s;
    }();
    return st;
}

void step_trace_append(std::string & line, const char * fmt, ...) {
    char buf[128];
    va_list args;
    va_start(args, fmt);
    const int n = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (n > 0) {
        line.append(buf, std::min<size_t>((size_t) n, sizeof(buf) - 1));
    }
}

} // namespace

bool llama_step_trace_active() {
    return step_trace_get() != nullptr;
}

void llama_step_trace_put(char role, const char * name) {
    step_trace * st = step_trace_get();
    if (st == nullptr) {
        return;
    }
    const int64_t t = st->now_ns();
    std::lock_guard<std::mutex> lock(st->mutex);
    st->marks.push_back({ role, name, t });
    st->decoded = st->decoded || role != 0;
}

bool llama_step_trace_on(void) {
    return step_trace_get() != nullptr;
}

void llama_step_trace_begin(const char * name) {
    step_trace * st = step_trace_get();
    if (st == nullptr) {
        return;
    }
    const int64_t t = st->now_ns();
    std::lock_guard<std::mutex> lock(st->mutex);
    st->marks.clear();
    st->counts.clear();
    st->decoded = false;
    st->marks.push_back({ 0, name, t });
}

void llama_step_trace_mark(const char * name) {
    llama_step_trace_put(0, name);
}

void llama_step_trace_count(const char * name, int64_t value) {
    step_trace * st = step_trace_get();
    if (st == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(st->mutex);
    for (auto & c : st->counts) {
        if (c.first == name) {
            c.second += value;
            return;
        }
    }
    st->counts.emplace_back(name, value);
}

void llama_step_trace_end(void) {
    step_trace * st = step_trace_get();
    if (st == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(st->mutex);
    if (st->decoded) {
        const int64_t t_w = st->now_ns();
        std::string & l = st->line;
        l.clear();
        step_trace_append(l, "{\"step\":%" PRId64 ",\"write_us\":%" PRId64 ",\"n\":{", st->n_step, st->write_us);
        for (size_t i = 0; i < st->counts.size(); ++i) {
            step_trace_append(l, "%s\"%s\":%" PRId64, i ? "," : "", st->counts[i].first, st->counts[i].second);
        }
        l += "},\"t\":[";
        for (size_t i = 0; i < st->marks.size(); ++i) {
            const auto & m = st->marks[i];
            const double ms = (double) m.t_ns / 1e6;
            if (m.role != 0) {
                step_trace_append(l, "%s[\"%c.%s\",%.3f]", i ? "," : "", m.role, m.name, ms);
            } else {
                step_trace_append(l, "%s[\"%s\",%.3f]", i ? "," : "", m.name, ms);
            }
        }
        l += "]}\n";
        fwrite(l.data(), 1, l.size(), st->f);
        st->n_step++;
        st->write_us = (st->now_ns() - t_w) / 1000;
    }
    st->marks.clear();
    st->counts.clear();
    st->decoded = false;
}
