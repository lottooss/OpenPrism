// include/aim/core/types.hpp
#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace aim {

using SequenceId = std::uint64_t;
using TrackId = std::uint64_t;
using MonotonicNs = std::int64_t;

struct PixelPoint { float x; float y; };          // origin top-left; +x right; +y down
struct NormalizedPoint { float x; float y; };     // [-1, 1], same axis directions
struct PixelVelocity { float x_per_s; float y_per_s; };
struct PixelAcceleration { float x_per_s2; float y_per_s2; };
struct BoundingBox { float left; float top; float right; float bottom; };
struct Covariance2D { float xx; float xy; float yy; }; // pixel^2

enum class Visibility : std::uint8_t { visible = 0, partial = 1, predicted = 2 };

enum class CorrelationFlags : std::uint32_t {
    none         = 0,
    synthetic    = 1 << 0,
    warmup       = 1 << 1,
    dropped      = 1 << 2,
    trace_verbose= 1 << 3
};

struct alignas(16) CorrelationId {
    SequenceId sequence_id{0};           // Monotonic frame/event sequence
    MonotonicNs source_timestamp_ns{0};  // Capture-surface arrival monotonic timestamp
    std::uint32_t pipeline_run_id{0};    // Unique execution session ID
    std::uint32_t flags{0};              // Diagnostic / synthetic flags

    [[nodiscard]] constexpr bool is_synthetic() const noexcept {
        return (flags & static_cast<std::uint32_t>(CorrelationFlags::synthetic)) != 0;
    }
};

enum class PipelineStage : std::uint8_t {
    capture_arrival    = 0,
    preprocess         = 1,
    perception_infer   = 2,
    tracking_kalman    = 3,
    aim_policy_predict = 4,
    trajectory_plan    = 5,
    actuation_dispatch = 6,
    stage_count        = 7
};

} // namespace aim
