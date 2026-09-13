// include/aim/core/clock.hpp
#pragma once

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

#if defined(_MSC_VER) || defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace aim {

/// @brief Abstract interface for monotonic clock providers.
class IClock {
public:
    virtual ~IClock() = default;
    [[nodiscard]] virtual MonotonicNs now_ns() noexcept = 0;
};

/// @brief Production Windows QPC Clock with sub-15ns conversion and monotonic clamp.
class QpcClock final : public IClock {
public:
    QpcClock() noexcept {
#if defined(_MSC_VER) || defined(_WIN32)
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        qpf_ = static_cast<std::uint64_t>(freq.QuadPart);
        is_10mhz_ = (qpf_ == 10'000'000ULL);
#else
        qpf_ = 10'000'000ULL;
        is_10mhz_ = true;
#endif
    }

    [[nodiscard]] MonotonicNs now_ns() noexcept override {
#if defined(_MSC_VER) || defined(_WIN32)
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        const std::uint64_t raw_qpc = static_cast<std::uint64_t>(counter.QuadPart);
        const MonotonicNs sample_ns = is_10mhz_
            ? static_cast<MonotonicNs>(raw_qpc * 100ULL)
            : qpc_to_ns_128(raw_qpc, qpf_);
#else
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const MonotonicNs sample_ns = static_cast<MonotonicNs>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()
        );
#endif
        // Monotonic non-decreasing clamp
        MonotonicNs prev = last_ns_.load(std::memory_order_relaxed);
        while (sample_ns > prev) {
            if (last_ns_.compare_exchange_weak(prev, sample_ns, std::memory_order_relaxed, std::memory_order_relaxed)) {
                return sample_ns;
            }
        }
        return prev;
    }

    [[nodiscard]] std::uint64_t qpf() const noexcept { return qpf_; }
    [[nodiscard]] bool is_10mhz() const noexcept { return is_10mhz_; }

private:
    std::uint64_t qpf_{10'000'000ULL};
    bool is_10mhz_{true};
    std::atomic<MonotonicNs> last_ns_{0};
};

/// @brief Deterministic fake clock for reproducible unit tests, replays, and benchmarks.
class FakeClock final : public IClock {
public:
    constexpr explicit FakeClock(MonotonicNs initial_ns = 0) noexcept : current_ns_(initial_ns) {}

    [[nodiscard]] MonotonicNs now_ns() noexcept override {
        return current_ns_.load(std::memory_order_relaxed);
    }

    void advance_ns(MonotonicNs delta_ns) noexcept {
        current_ns_.fetch_add(delta_ns, std::memory_order_relaxed);
    }

    void advance_ms(double delta_ms) noexcept {
        advance_ns(ms_to_ns(delta_ms));
    }

    void set_ns(MonotonicNs target_ns) noexcept {
        current_ns_.store(target_ns, std::memory_order_relaxed);
    }

private:
    std::atomic<MonotonicNs> current_ns_{0};
};

} // namespace aim
