// include/aim/trajectory/trajectory_planner.hpp
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/interfaces/trajectory_planner.hpp"

namespace aim::trajectory {

struct PlannerConfig {
    float min_movement_time_s{0.005f};     // 5 ms minimum trajectory duration
    float max_movement_time_s{0.032f};     // 32 ms maximum planning horizon (32 microsteps @ 1kHz)
    float time_step_s{0.001f};             // 1.0 ms microstep period (1,000 Hz)
    float pd_kp{1.0f};                     // Proportional gain for terminal correction
    float pd_kd{2.0f};                     // Derivative gain (2*sqrt(kp) for critical damping)
    float counts_per_pixel{1.0f};          // Standalone reference only; pipeline maps geometry through CalibrationModel.
};

/// @brief Production zero-allocation hybrid trajectory planner.
/// Implements direct feed-forward, minimum-jerk profile, and critically damped PD terminal correction.
class HybridTrajectoryPlanner : public ITrajectoryPlanner {
public:
    explicit HybridTrajectoryPlanner(PlannerConfig config = {})
        : config_(config) {}

    bool plan(const bus::AimIntent& intent,
              const ControlState& state,
              const PlannerLimits& limits,
              TrajectoryPlan& out_plan) noexcept override;

    void cancel() noexcept override {
        is_canceled_ = true;
        curve_active_ = false;
    }

    void reset() noexcept override {
        is_canceled_ = false;
        next_plan_id_ = 1;
        curve_active_ = false;
    }

    [[nodiscard]] const PlannerConfig& config() const noexcept { return config_; }
    void set_config(const PlannerConfig& config) noexcept { config_ = config; }

private:
    PlannerConfig config_;
    SequenceId next_plan_id_{1};
    bool is_canceled_{false};
    bool curve_active_{false};
    std::uint64_t curve_track_id_{0};
    MonotonicNs curve_start_ns_{0};
    MonotonicNs last_plan_time_ns_{0};
    float curve_duration_s_{0.0f};
    float curve_dx_{0.0f};
    float curve_dy_{0.0f};
};

using TrajectoryPlanner = HybridTrajectoryPlanner;

} // namespace aim::trajectory
