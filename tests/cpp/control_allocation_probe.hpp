#pragma once
#include <cstdlib>
#include <new>
#if defined(_MSC_VER)
#include <malloc.h>
#endif
namespace allocation_probe {

thread_local bool enabled = false;
thread_local std::size_t calls = 0;
thread_local std::size_t bytes = 0;

void record(std::size_t size) noexcept {
    if (enabled) {
        ++calls;
        bytes += size;
    }
}

struct Report final {
    std::size_t calls;
    std::size_t bytes;
};

void begin() noexcept {
    calls = 0;
    bytes = 0;
    enabled = true;
}

[[nodiscard]] Report end() noexcept {
    enabled = false;
    return {.calls = calls, .bytes = bytes};
}

}  // namespace allocation_probe

// GCC cannot see that these replacement operators are a matched pair: it classifies the
// replaced `operator new` as new-like and `std::free` as malloc-like, then reports
// -Wmismatched-new-delete once the vector teardown below gets inlined into the benchmark.
// Every allocation here is malloc/aligned_alloc and every deallocation is the matching
// free, so the pairing is correct and the diagnostic is a false positive.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void* operator new(std::size_t size) {
    const std::size_t requested = size == 0 ? 1 : size;
    if (void* pointer = std::malloc(requested)) {
        allocation_probe::record(requested);
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    ::operator delete(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    ::operator delete[](pointer);
}

#if defined(__cpp_aligned_new)
void* operator new(std::size_t size, std::align_val_t alignment) {
    const std::size_t requested = size == 0 ? 1 : size;
    const std::size_t alignment_bytes = static_cast<std::size_t>(alignment);
#if defined(_MSC_VER)
    void* pointer = _aligned_malloc(requested, alignment_bytes);
#else
    const std::size_t padded = ((requested + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
    void* pointer = std::aligned_alloc(alignment_bytes, padded);
#endif
    if (pointer == nullptr) {
        throw std::bad_alloc();
    }
    allocation_probe::record(requested);
    return pointer;
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}

void operator delete(void* pointer, std::align_val_t) noexcept {
#if defined(_MSC_VER)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

void operator delete[](void* pointer, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

void operator delete(void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

void operator delete[](void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete[](pointer, alignment);
}
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
