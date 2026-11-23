// Record and replay of test-backend-ops cases (C372, Q931): the files and their checksums.
//
// A record of one case is three files in <dir>/cases/:
//   <id>.in    the bytes of every tensor the case allocates, as its initialize_tensors left them (the inputs, the
//              sentinels, the outputs' initial contents), one after the other in context order
//   <id>.ref   the reference backend's result of every node the live check compares, in graph order: the node's bytes
//              first for nodes without an op (sentinels and parameters, which must stay unchanged), then its values as
//              floats after the case's host_reference
//   <id>.meta  text, one "key<TAB>fields" line each: the case string, its key, its class, the threshold, the md5 and
//              size of both blobs, each tensor (name, type, shape, bytes) and each compared node (graph index, op, sizes)
// <id> is the first 20 hex digits of the md5 of the case's key: its class, its parameters and how many cases before it
// in the full case list have both. A record therefore replays on any build whose case list holds the same case, also
// after a branch adds or removes others; replay checks the case string, the tensor layout and both md5s before it
// trusts a record.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#define mkdir(p, m) _mkdir(p)
#endif

// MD5 (RFC 1321)
struct tbo_md5 {
    uint32_t h[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };
    uint8_t  buf[64];
    uint64_t len = 0;

    static uint32_t rotl(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

    void block(const uint8_t * p) {
        static const uint32_t K[64] = {
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
            0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
            0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
            0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
            0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
            0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
            0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
        };
        static const int R[64] = {
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
            5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
            4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        };
        uint32_t w[16];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t) p[4*i] | ((uint32_t) p[4*i + 1] << 8) | ((uint32_t) p[4*i + 2] << 16) | ((uint32_t) p[4*i + 3] << 24);
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        for (int i = 0; i < 64; i++) {
            uint32_t f;
            int      g;
            if (i < 16)      { f = (b & c) | (~b & d); g = i; }
            else if (i < 32) { f = (d & b) | (~d & c); g = (5*i + 1) % 16; }
            else if (i < 48) { f = b ^ c ^ d;          g = (3*i + 5) % 16; }
            else             { f = c ^ (b | ~d);       g = (7*i) % 16; }
            const uint32_t t = d;
            d = c;
            c = b;
            b = b + rotl(a + f + K[i] + w[g], R[i]);
            a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    }

    void update(const void * data, size_t n) {
        if (n == 0) {
            return;
        }
        const uint8_t * p = (const uint8_t *) data;
        size_t fill = len % 64;
        len += n;
        if (fill) {
            const size_t take = std::min<size_t>(64 - fill, n);
            memcpy(buf + fill, p, take);
            p += take; n -= take; fill += take;
            if (fill < 64) {
                return;
            }
            block(buf);
        }
        for (; n >= 64; p += 64, n -= 64) {
            block(p);
        }
        memcpy(buf, p, n);
    }

    std::string hex() {
        const uint64_t bits = len * 8;
        const uint8_t  pad  = 0x80;
        const uint8_t  zero = 0;
        update(&pad, 1);
        while (len % 64 != 56) {
            update(&zero, 1);
        }
        uint8_t lb[8];
        for (int i = 0; i < 8; i++) {
            lb[i] = (uint8_t) (bits >> (8*i));
        }
        update(lb, 8);
        char out[33];
        for (int i = 0; i < 16; i++) {
            snprintf(out + 2*i, 3, "%02x", (h[i/4] >> (8*(i % 4))) & 0xff);
        }
        return std::string(out, 32);
    }
};

static bool tbo_mkdirs(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        if (i < path.size()) {
            cur += path[i];
        }
    }
    return true;
}

static bool tbo_file_exists(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// a file written in one go: to <path>.tmp, renamed when complete, so a file that exists is whole
struct tbo_writer {
    std::string path;
    FILE *      f   = nullptr;
    tbo_md5     md5;
    size_t      n   = 0;
    bool        ok  = false;

    explicit tbo_writer(const std::string & p) : path(p) {
        f  = fopen((path + ".tmp").c_str(), "wb");
        ok = f != nullptr;
    }
    void write(const void * data, size_t size) {
        if (!ok || size == 0) {
            return;
        }
        ok = fwrite(data, 1, size, f) == size;
        md5.update(data, size);
        n += size;
    }
    bool close() {
        if (f) {
            ok = fclose(f) == 0 && ok;
            f  = nullptr;
        }
        ok = ok && rename((path + ".tmp").c_str(), path.c_str()) == 0;
        return ok;
    }
    ~tbo_writer() {
        if (f) {
            fclose(f);
            remove((path + ".tmp").c_str());
        }
    }
};

// <index>.meta, parsed
struct tbo_meta {
    struct tensor { std::string name, type, shape; size_t nbytes = 0; };
    struct node   { int graph_index = 0; std::string op, name; size_t raw = 0, n = 0; };

    std::string name;     // OP(vars), the case string test-backend-ops prints
    std::string cls;      // the test case's class (its err() and thresholds)
    std::string mode;     // nodes (each node computed and compared alone) or graph (the whole graph, then its test nodes)
    double      max_err = 0;
    size_t      in_bytes = 0, ref_bytes = 0;
    std::string in_md5, ref_md5;
    std::vector<tensor> tensors;
    std::vector<node>   nodes;

    static std::vector<std::string> split(const std::string & line) {
        std::vector<std::string> f;
        size_t s = 0;
        for (size_t i = 0; i <= line.size(); i++) {
            if (i == line.size() || line[i] == '\t') {
                f.push_back(line.substr(s, i - s));
                s = i + 1;
            }
        }
        return f;
    }

    bool read(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) {
            return false;
        }
        std::string text;
        char chunk[65536];
        size_t r;
        while ((r = fread(chunk, 1, sizeof(chunk), f)) > 0) {
            text.append(chunk, r);
        }
        fclose(f);
        bool magic = false;
        size_t s = 0;
        while (s < text.size()) {
            size_t e = text.find('\n', s);
            if (e == std::string::npos) {
                e = text.size();
            }
            const std::vector<std::string> v = split(text.substr(s, e - s));
            s = e + 1;
            const std::string & k = v[0];
            if (k == "tbo-record" && v.size() >= 2 && v[1] == "1") {
                magic = true;
            } else if (k == "name" && v.size() >= 2) {
                name = v[1];
            } else if (k == "class" && v.size() >= 2) {
                cls = v[1];
            } else if (k == "mode" && v.size() >= 2) {
                mode = v[1];
            } else if (k == "max_err" && v.size() >= 2) {
                max_err = strtod(v[1].c_str(), nullptr);
            } else if (k == "in" && v.size() >= 3) {
                in_bytes = strtoull(v[1].c_str(), nullptr, 10);
                in_md5   = v[2];
            } else if (k == "ref" && v.size() >= 3) {
                ref_bytes = strtoull(v[1].c_str(), nullptr, 10);
                ref_md5   = v[2];
            } else if (k == "tensor" && v.size() >= 6) {
                tensors.push_back({ v[2], v[3], v[4], (size_t) strtoull(v[5].c_str(), nullptr, 10) });
            } else if (k == "node" && v.size() >= 7) {
                nodes.push_back({ atoi(v[2].c_str()), v[3], v[4], (size_t) strtoull(v[5].c_str(), nullptr, 10),
                                  (size_t) strtoull(v[6].c_str(), nullptr, 10) });
            }
        }
        return magic && !name.empty() && !in_md5.empty() && !ref_md5.empty();
    }
};
