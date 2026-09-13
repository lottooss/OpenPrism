// src/pipeline/closed_loop_pipeline.cpp
#include "aim/pipeline/closed_loop_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace aim::pipeline {

void ClosedLoopPipeline::cancel_pending(bool preserve_motion_phase) noexcept {
    scheduler_.cancel_active_plan();
    if (!preserve_motion_phase) {
        if (planner_) planner_->cancel();
        calibration_residual_ = {};
        planned_track_id_ = 0;
    }
    plan_ = {};
    last_intent_ = {};
    last_command_ = {};
    last_shot_result_.authorize_fire = false;
    last_shot_result_.button_transition = {};
    if (last_shot_result_.decision == policy::ShotGateDecision::authorized) {
        last_shot_result_.decision = policy::ShotGateDecision::rejected_invalid_input;
    }
    // Retain shot receipts across focus loss and failed ticks; faults must not
    // reset cadence and permit another immediate shot on recovery.
}

ClosedLoopPipeline::ClosedLoopPipeline(
    ITrackingEngine* tracker,
    IAimPolicy* policy,
    ITrajectoryPlanner* planner,
    IActuator* actuator,
    ISafetySupervisor* safety_supervisor,
    IScenarioAdapter* scenario_adapter,
    const calibration::CalibrationModel& calibration,
    const ClosedLoopPipelineConfig& config) noexcept
    : tracker_(tracker),
      policy_(policy),
      planner_(planner),
      actuator_(actuator),
      safety_(safety_supervisor),
      scenario_(scenario_adapter),
      calibration_(calibration),
      config_(config),
      scheduler_(actuator),
      shot_gate_(config.shot_gate_config) {
    actuation::SchedulerConfig scheduler_config;
    scheduler_config.max_source_age_ns = config.max_internal_latency_ns;
    (void)scheduler_.initialize(scheduler_config, actuator, safety_supervisor);
}

bool ClosedLoopPipeline::tick_actuation(const ObservableScenarioState& scenario_state,
                                        MonotonicNs now_ns) noexcept {
    if (now_ns <= 0 || !calibration_.is_valid() ||
        (config_.enforce_foreground_authorization && scenario_ &&
         !scenario_->foreground_authorized(scenario_state)) ||
        (safety_ && safety_->is_latched())) {
        ++stats_.total_safety_rejections;
        cancel_pending();
        return false;
    }
    ActuationCommand command{};
    std::uint32_t index{};
    const auto result = scheduler_.tick(now_ns, command, index);
    if (result == actuation::SchedulerTickResult::expired) {
        // Cancel all executable work, retaining only motion/quantization history
        // for a future fresh observation. Capture gaps are not target changes.
        cancel_pending(true);
        return false;
    }
    if (result == actuation::SchedulerTickResult::rejected) {
        ++stats_.total_safety_rejections;
        cancel_pending();
        return false;
    }
    if (result == actuation::SchedulerTickResult::submitted ||
        result == actuation::SchedulerTickResult::zero_step) {
        calibration_residual_ = point_residuals_[index];
    }
    if (result == actuation::SchedulerTickResult::submitted) {
        last_command_ = command;
        ++stats_.total_commands_dispatched;
    }
    return true;
}

bool ClosedLoopPipeline::process_frame(
    const bus::TargetObservationBatch& observations,
    const ObservableScenarioState& scenario_state,
    MonotonicNs now_ns) noexcept {

    ++stats_.total_frames_processed;
    last_shot_result_ = {};
    if (observations.target_count > bus::kMaxObservations || now_ns <= 0 ||
        config_.command_lead_time_ns < 0 || config_.command_lead_time_ns > 100'000'000LL ||
        config_.shot_gate_config.max_prediction_lead_ns <= 0 ||
        config_.shot_gate_config.max_prediction_lead_ns > 100'000'000LL ||
        config_.shot_gate_config.max_prediction_lead_ns < config_.command_lead_time_ns ||
        now_ns > (std::numeric_limits<MonotonicNs>::max)() -
            (std::max)(MonotonicNs{10'000'000LL}, config_.command_lead_time_ns) ||
        !calibration_.is_valid()) {
        cancel_pending();
        return false;
    }

    // Gate 1: Scenario / Foreground authorization
    if (config_.enforce_foreground_authorization && scenario_) {
        if (!scenario_->foreground_authorized(scenario_state)) {
            cancel_pending();
            return false;
        }
    }

    // Gate 2: Safety Supervisor latch
    if (safety_ && safety_->is_latched()) {
        ++stats_.total_safety_rejections;
        cancel_pending();
        return false;
    }

    // Gate 3: Hard 10 ms freshness deadline cutoff
    {
        if (observations.captured_at_ns <= 0 || now_ns < observations.captured_at_ns ||
            config_.max_internal_latency_ns <= 0 || config_.max_internal_latency_ns > 10'000'000LL ||
            now_ns - observations.captured_at_ns > config_.max_internal_latency_ns) {
            ++stats_.total_stale_dropped;
            cancel_pending(true);
            return false;
        }
    }
    CorrelationId source = observations.header;
    // Native fixtures may omit the optional header timestamp, but capture time
    // is mandatory. Never replace an existing (possibly older) source time.
    if (source.source_timestamp_ns == 0) source.source_timestamp_ns = observations.captured_at_ns;
    if (source.source_timestamp_ns <= 0 || source.source_timestamp_ns > now_ns ||
        now_ns - source.source_timestamp_ns > config_.max_internal_latency_ns) {
        ++stats_.total_stale_dropped;
        cancel_pending(true);
        return false;
    }

    // Resolve crosshair context
    PixelPoint crosshair_px{960.0f, 540.0f};
    if (scenario_ && scenario_->capabilities().provides_crosshair_context) {
        crosshair_px = scenario_->crosshair_context().center_px;
    }

    const auto t_start = std::chrono::steady_clock::now();

    // Stage 1: Predict to measured command effect; source age above remains
    // bounded to 10 ms regardless of this causal prediction horizon.
    const MonotonicNs prediction_time_ns = now_ns + config_.command_lead_time_ns;
    tracks_.clear();
    const auto t_track_start = std::chrono::steady_clock::now();
    if (!tracker_ || !tracker_->update(observations, prediction_time_ns, tracks_) ||
        tracks_.track_count > bus::kMaxTrackedTargets) {
        cancel_pending();
        return false;
    }
    const auto t_track_end = std::chrono::steady_clock::now();
    stats_.last_tracking_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_track_end - t_track_start
    ).count();

    // Stage 2: Aim Policy Target Selection
    PolicyInput p_input{};
    p_input.correlation_id = observations.header;
    p_input.decision_time_ns = now_ns;
    p_input.crosshair.center_px = crosshair_px;
    p_input.tracks = &tracks_;

    const auto t_policy_start = std::chrono::steady_clock::now();
    last_intent_ = {};
    if (!policy_ || !policy_->choose(p_input, last_intent_)) {
        // Policies conventionally return false when there is no target.
        // Never reuse an old or partially written intent from a failed call.
        last_intent_ = {};
    }
    const auto t_policy_end = std::chrono::steady_clock::now();
    stats_.last_policy_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_policy_end - t_policy_start
    ).count();

    // Ensure correlation ID is carried
    last_intent_.header = source;
    last_intent_.command_deadline_ns = now_ns + 10'000'000LL;
    if (!std::isfinite(last_intent_.target_aim_px.x) || !std::isfinite(last_intent_.target_aim_px.y)) {
        cancel_pending();
        return false;
    }

    // Stage 3: Uncertainty-Aware Shot Gating
    const bus::TrackedTarget* active_target = nullptr;
    for (const auto& trk : tracks_.items()) {
        if (trk.track_id == last_intent_.target_track_id) {
            active_target = &trk;
            break;
        }
    }

    const bool policy_allows_fire = last_intent_.authorize_fire;
    last_shot_result_ = shot_gate_.evaluate(active_target, crosshair_px, now_ns, safety_);
    if (last_shot_result_.authorize_fire && !policy_allows_fire) {
        last_shot_result_.authorize_fire = false;
        last_shot_result_.button_transition = {};
        last_shot_result_.decision = policy::ShotGateDecision::rejected_policy_veto;
    }
    if (last_shot_result_.authorize_fire) {
        last_intent_.authorize_fire = true;
    } else {
        last_intent_.authorize_fire = false;
        ++stats_.total_shots_rejected;
    }

    // Stage 4: Trajectory Planning
    ControlState c_state{};
    c_state.current_crosshair_px = crosshair_px;

    const auto t_plan_start = std::chrono::steady_clock::now();
    plan_ = {};
    if (!planner_ || !planner_->plan(last_intent_, c_state, config_.planner_limits, plan_) ||
        plan_.point_count > kMaxTrajectoryPoints) {
        cancel_pending();
        return false;
    }
    const auto t_plan_end = std::chrono::steady_clock::now();
    stats_.last_planner_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_plan_end - t_plan_start
    ).count();

    plan_.correlation_id = source;
    // A planner's start time is planning time; source age remains capture age.
    plan_.start_time_ns = now_ns;

    // Stage 5: Calibration Mapping to Motor Counts
    if (planned_track_id_ != last_intent_.target_track_id) calibration_residual_ = {};
    planned_track_id_ = last_intent_.target_track_id;
    auto residual = calibration_residual_;
    auto previous_px = crosshair_px;
    for (std::uint32_t i = 0; i < plan_.point_count; ++i) {
        auto& point = plan_.points[i];
        calibration::MouseCounts counts{};
        calibration::PixelDisplacement realized{};
        const float dx = point.position_px.x - previous_px.x + residual.dx_px;
        const float dy = point.position_px.y - previous_px.y + residual.dy_px;
        if (!calibration_.try_pixels_to_counts(dx, dy, counts) ||
            !calibration_.try_counts_to_pixels(counts.counts_x, counts.counts_y, realized)) {
            cancel_pending();
            return false;
        }
        point.step_delta_x_counts = counts.counts_x;
        point.step_delta_y_counts = counts.counts_y;
        residual = {dx - realized.dx_px, dy - realized.dy_px};
        point_residuals_[i] = residual;
        previous_px = point.position_px;
    }
    if (!scheduler_.try_submit_plan(plan_)) {
        cancel_pending();
        return false;
    }

    // Stage 6: Build & Dispatch Actuation Command
    ActuationCommand cmd{};
    cmd.sequence_id = scheduler_.reserve_command_sequence();
    cmd.correlation_id = source;
    cmd.generated_at_ns = source.source_timestamp_ns;
    cmd.desired_apply_time_ns = now_ns;
    cmd.button_transition = last_shot_result_.button_transition;

    // Safety Supervisor Check
    if (safety_ && cmd.button_transition.action != ButtonAction::none) {
        if (!safety_->check_actuation_safety(cmd, now_ns)) {
            ++stats_.total_safety_rejections;
            cancel_pending();
            return false;
        }
    }

    // Dispatch to Actuator only if there is movement or button transition
    if (cmd.button_transition.action != ButtonAction::none) {
        if (!actuator_ || actuator_->submit_latest(cmd) != SubmitResult::submitted) {
            ++stats_.total_safety_rejections;
            cancel_pending();
            return false;
        }
        last_command_ = cmd;
        ++stats_.total_commands_dispatched;
        if (last_shot_result_.authorize_fire && active_target) {
            shot_gate_.record_dispatched_shot(*active_target, now_ns);
            ++stats_.total_shots_authorized;
        }
    } else if (last_intent_.target_track_id == 0) {
        cancel_pending();
    }

    const auto t_end = std::chrono::steady_clock::now();
    stats_.last_total_internal_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        t_end - t_start
    ).count();

    return true;
}

} // namespace aim::pipeline
