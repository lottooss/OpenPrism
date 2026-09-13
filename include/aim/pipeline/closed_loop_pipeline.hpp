// include/aim/pipeline/closed_loop_pipeline.hpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "aim/bus/bus_traits.hpp"
#include "aim/actuation/actuator_scheduler.hpp"
#include "aim/calibration/calibration_model.hpp"
#include "aim/core/actuator.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/interfaces/aim_policy.hpp"
#include "aim/interfaces/safety_supervisor.hpp"
#include "aim/interfaces/scenario_adapter.hpp"
#include "aim/interfaces/tracking_engine.hpp"
#include "aim/interfaces/trajectory_planner.hpp"
#include "aim/policy/shot_gate.hpp"

namespace aim::pipeline {

struct ClosedLoopPipelineConfig {
    MonotonicNs max_internal_latency_ns{10'000'000LL}; // 10 ms hard deadline cutoff
    MonotonicNs command_lead_time_ns{2'000'000LL};     // Predicted actuation effect lead (2 ms)
    bool enforce_foreground_authorization{true};
    bool drop_stale_observations{true};
    PlannerLimits planner_limits{};
    policy::ShotGateConfig shot_gate_config{};
};

struct PipelineTelemetryStats {
    std::uint64_t total_frames_processed{0};
    std::uint64_t total_commands_dispatched{0};
    std::uint64_t total_stale_dropped{0};
    std::uint64_t total_shots_authorized{0};
    std::uint64_t total_shots_rejected{0};
    std::uint64_t total_safety_rejections{0};

    // Internal loop stage latencies in nanoseconds
    MonotonicNs last_tracking_latency_ns{0};
    MonotonicNs last_policy_latency_ns{0};
    MonotonicNs last_planner_latency_ns{0};
    MonotonicNs last_total_internal_latency_ns{0};
};

/// @brief Unified End-to-End Closed-Loop Aiming Pipeline (Milestone M7-02).
/// Connects observations through tracking, policy, uncertainty shot gate,
/// trajectory planning, calibration, and actuation with correlation ID propagation.
class ClosedLoopPipeline {
public:
    ClosedLoopPipeline(
        ITrackingEngine* tracker,
        IAimPolicy* policy,
        ITrajectoryPlanner* planner,
        IActuator* actuator,
        ISafetySupervisor* safety_supervisor,
        IScenarioAdapter* scenario_adapter,
        const calibration::CalibrationModel& calibration,
        const ClosedLoopPipelineConfig& config = {}) noexcept;

    /// @brief Execute one synchronous pipeline tick from an observation batch.
    /// @return true if tick processed and command was safely issued or held; false if dropped.
    bool process_frame(
        const bus::TargetObservationBatch& observations,
        const ObservableScenarioState& scenario_state,
        MonotonicNs now_ns) noexcept;

    /// Run at 1 kHz on the same owner thread as process_frame, with current
    /// foreground evidence. Future points remain pending; faults cancel them.
    bool tick_actuation(const ObservableScenarioState& scenario_state,
                        MonotonicNs now_ns) noexcept;

    [[nodiscard]] PipelineTelemetryStats stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = {}; }

    [[nodiscard]] const ActuationCommand& last_command() const noexcept { return last_command_; }
    [[nodiscard]] const bus::AimIntent& last_intent() const noexcept { return last_intent_; }
    [[nodiscard]] const policy::ShotGateResult& last_shot_result() const noexcept { return last_shot_result_; }
    [[nodiscard]] const bus::TrackedTargetBatch& last_tracks() const noexcept { return tracks_; }
    [[nodiscard]] const TrajectoryPlan& last_plan() const noexcept { return plan_; }

private:
    ITrackingEngine* tracker_{nullptr};
    IAimPolicy* policy_{nullptr};
    ITrajectoryPlanner* planner_{nullptr};
    IActuator* actuator_{nullptr};
    ISafetySupervisor* safety_{nullptr};
    IScenarioAdapter* scenario_{nullptr};
    calibration::CalibrationModel calibration_;
    ClosedLoopPipelineConfig config_{};
    actuation::ActuatorScheduler scheduler_;
    calibration::PixelDisplacement calibration_residual_{};
    std::array<calibration::PixelDisplacement, kMaxTrajectoryPoints> point_residuals_{};
    std::uint64_t planned_track_id_{0};

    policy::ShotGate shot_gate_;
    bus::TrackedTargetBatch tracks_{};
    bus::AimIntent last_intent_{};
    TrajectoryPlan plan_{};
    ActuationCommand last_command_{};
    policy::ShotGateResult last_shot_result_{};
    PipelineTelemetryStats stats_{};
    void cancel_pending(bool preserve_motion_phase = false) noexcept;
};

} // namespace aim::pipeline
