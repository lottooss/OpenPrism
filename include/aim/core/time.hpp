// include/aim/core/time.hpp
#pragma once

#include <cstdint>
#include "aim/core/types.hpp"

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace aim {

/// @brief Overflow-safe QPC to monotonic nanoseconds conversion using Euclidean decomposition.
/// Guaranteed overflow-free for over 580 years of continuous uptime on 64-bit platforms.
[[nodiscard]] constexpr MonotonicNs qpc_to_ns_euclidean(std::uint64_t qpc, std::uint64_t qpf) noexcept {
    if (qpf == 0) [[unlikely]] {
        return 0;
    }
    if (qpf == 10'000'000ULL) [[likely]] {
        return static_cast<MonotonicNs>(qpc * 100ULL);
    }
    const std::uint64_t q = qpc / qpf;
    const std::uint64_t r = qpc % qpf;
    return static_cast<MonotonicNs>((q * 1'000'000'000ULL) + ((r * 1'000'000'000ULL) / qpf));
}

/// @brief Hardware-accelerated 128-bit QPC to nanoseconds conversion (MSVC x64).
[[nodiscard]] inline MonotonicNs qpc_to_ns_128(std::uint64_t qpc, std::uint64_t qpf) noexcept {
    if (qpf == 0) [[unlikely]] {
        return 0;
    }
#if defined(_MSC_VER) && defined(_M_X64)
    if (qpf == 10'000'000ULL) [[likely]] {
        return static_cast<MonotonicNs>(qpc * 100ULL); // 1-cycle fast path
    }
    std::uint64_t high = 0;
    const std::uint64_t low = _umul128(qpc, 1'000'000'000ULL, &high);
    std::uint64_t remainder = 0;
    return static_cast<MonotonicNs>(_udiv128(high, low, qpf, &remainder));
#elif defined(__SIZEOF_INT128__)
    if (qpf == 10'000'000ULL) [[likely]] {
        return static_cast<MonotonicNs>(qpc * 100ULL);
    }
    __extension__ typedef unsigned __int128 uint128_t;
    return static_cast<MonotonicNs>((static_cast<uint128_t>(qpc) * 1'000'000'000ULL) / qpf);
#else
    return qpc_to_ns_euclidean(qpc, qpf);
#endif
}

/// @brief Convert nanoseconds duration to floating point milliseconds.
[[nodiscard]] constexpr double ns_to_ms(MonotonicNs ns) noexcept {
    return static_cast<double>(ns) / 1'000'000.0;
}

/// @brief Convert milliseconds to nanoseconds.
[[nodiscard]] constexpr MonotonicNs ms_to_ns(double ms) noexcept {
    return static_cast<MonotonicNs>(ms * 1'000'000.0);
}

} // namespace aim
