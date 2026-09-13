// include/aim/core/telemetry.hpp
#pragma once

#include <cstdint>
#include "aim/core/stage_timer.hpp"
#include "aim/core/types.hpp"

namespace aim {

/// @brief End-to-end hot-path cycle summary event.
struct HotLoopTelemetryEvent {
    CorrelationId correlation_id;
    MonotonicNs capture_arrival_ns{0};
    MonotonicNs preprocess_done_ns{0};
    MonotonicNs inference_done_ns{0};
    MonotonicNs tracking_done_ns{0};
    MonotonicNs policy_done_ns{0};
    MonotonicNs dispatch_done_ns{0};
    std::uint32_t detected_targets{0};
    std::uint32_t tracked_targets{0};
    bool dropped_or_stale{false};

    [[nodiscard]] constexpr MonotonicNs total_latency_ns() const noexcept {
        return (dispatch_done_ns >= capture_arrival_ns) ? (dispatch_done_ns - capture_arrival_ns) : 0;
    }

    [[nodiscard]] constexpr double total_latency_ms() const noexcept {
        return static_cast<double>(total_latency_ns()) / 1'000'000.0;
    }
};

} // namespace aim
