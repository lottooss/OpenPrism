// include/aim/core/stage_timer.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include "aim/core/clock.hpp"
#include "aim/core/types.hpp"

namespace aim {

struct StageTimestampEvent {
    CorrelationId correlation_id;
    PipelineStage stage{PipelineStage::capture_arrival};
    MonotonicNs start_ns{0};
    MonotonicNs end_ns{0};

    [[nodiscard]] constexpr MonotonicNs duration_ns() const noexcept {
        return (end_ns >= start_ns) ? (end_ns - start_ns) : 0;
    }

    [[nodiscard]] constexpr double duration_ms() const noexcept {
        return static_cast<double>(duration_ns()) / 1'000'000.0;
    }
};

/// @brief Allocation-free, single-producer thread-confined fixed-capacity buffer for stage timing telemetry.
template <std::size_t Capacity = 1024>
class FixedTelemetryBuffer {
public:
    constexpr FixedTelemetryBuffer() noexcept = default;

    bool record(const StageTimestampEvent& event) noexcept {
        const std::size_t idx = count_.load(std::memory_order_relaxed);
        if (idx < Capacity) {
            events_[idx] = event;
            count_.store(idx + 1, std::memory_order_release);
            return true;
        }
        return false; // Ring buffer full: drop without blocking hot path
    }

    [[nodiscard]] std::span<const StageTimestampEvent> events() const noexcept {
        return std::span<const StageTimestampEvent>(events_.data(), count_.load(std::memory_order_acquire));
    }

    void reset() noexcept {
        count_.store(0, std::memory_order_release);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return count_.load(std::memory_order_acquire);
    }

    [[nodiscard]] constexpr std::size_t capacity() const noexcept {
        return Capacity;
    }

private:
    std::array<StageTimestampEvent, Capacity> events_{};
    std::atomic<std::size_t> count_{0};
};

/// @brief RAII zero-allocation stage timer for measuring pipeline stage durations.
template <std::size_t TelemetryCap = 1024>
class ScopedStageTimer {
public:
    ScopedStageTimer(PipelineStage stage,
                     const CorrelationId& cid,
                     IClock& clock,
                     FixedTelemetryBuffer<TelemetryCap>& buffer) noexcept
        : stage_(stage), cid_(cid), clock_(clock), buffer_(buffer), start_ns_(clock.now_ns()) {}

    ~ScopedStageTimer() noexcept {
        const MonotonicNs end_ns = clock_.now_ns();
        const StageTimestampEvent event{
            .correlation_id = cid_,
            .stage = stage_,
            .start_ns = start_ns_,
            .end_ns = end_ns
        };
        buffer_.record(event);
    }

    ScopedStageTimer(const ScopedStageTimer&) = delete;
    ScopedStageTimer& operator=(const ScopedStageTimer&) = delete;

private:
    PipelineStage stage_;
    CorrelationId cid_;
    IClock& clock_;
    FixedTelemetryBuffer<TelemetryCap>& buffer_;
    MonotonicNs start_ns_;
};

} // namespace aim
