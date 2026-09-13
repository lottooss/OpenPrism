// include/aim/interfaces/trajectory_planner.hpp
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include "aim/bus/bus_traits.hpp"
#include "aim/core/actuator.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim {

constexpr std::size_t kMaxTrajectoryPoints = 32;

struct TrajectoryPoint {
    MonotonicNs target_time_ns{0};
    PixelPoint position_px{};
    PixelVelocity velocity_px_s{};
    std::int32_t step_delta_x_counts{0};
    std::int32_t step_delta_y_counts{0};
};

struct TrajectoryPlan {
    SequenceId plan_id{0};
    CorrelationId correlation_id{};
    MonotonicNs start_time_ns{0};
    MonotonicNs end_time_ns{0};
    std::uint32_t point_count{0};
    std::array<TrajectoryPoint, kMaxTrajectoryPoints> points{};

    [[nodiscard]] std::span<const TrajectoryPoint> items() const noexcept {
        return {points.data(), std::min<std::size_t>(point_count, points.size())};
    }
};

struct ControlState {
    PixelPoint current_crosshair_px{960.0f, 540.0f};
    PixelVelocity current_velocity_px_s{0.0f, 0.0f};
};

struct PlannerLimits {
    float max_velocity_px_s{5000.0f};
    float max_accel_px_s2{500000.0f};
    float max_jerk_px_s3{100000000.0f};
    float small_error_threshold_px{6.0f};
};

/// @brief Primary trajectory planner generating jerk-limited control paths and micro-corrections.
class ITrajectoryPlanner {
public:
    virtual ~ITrajectoryPlanner() = default;
    virtual bool plan(const bus::AimIntent& intent,
                      const ControlState& state,
                      const PlannerLimits& limits,
                      TrajectoryPlan& out_plan) noexcept = 0;
    virtual void cancel() noexcept = 0;
    virtual void reset() noexcept = 0;
};

} // namespace aim
