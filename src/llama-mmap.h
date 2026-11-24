#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <cstdio>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define LLAMA_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

// number of pinned staging buffers for direct I/O uploads to a device
#define LLAMA_DIRECT_IO_BUFFER_COUNT 4

// The direct-I/O staging as set by LLAMA_DIO_STAGING=<count>x<MiB> (e.g. 2x16; either part may be left out: "x8",
// "2"), else the defaults above: the size of each pinned upload buffer and of the bounce buffer of reads into host
// memory, and the number of pinned upload buffers. On a small host the pinned staging (4 x 64 MiB by default) is
// allocated again for every buffer a model loads, while earlier contexts already hold their own pinned buffers.
size_t llama_direct_io_buffer_size();
size_t llama_direct_io_buffer_count();

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_memory_range {
    const void * addr;
    size_t size;
};

using llama_memory_ranges = std::vector<llama_memory_range>;

// Prefetch the host pages covering these memory ranges.
void llama_prefetch(llama_memory_ranges mr);
