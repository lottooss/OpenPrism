#include "control_allocation_probe.hpp"
// tests/cpp/test_closed_loop_pipeline.cpp
#include <iostream>
#include <cmath>
#include <array>
#include <algorithm>
#include <chrono>

#include "aim/core/actuator.hpp"
#include "aim/pipeline/closed_loop_pipeline.hpp"
#include "aim/policy/utility_policy.hpp"
#include "aim/safety/safety_supervisor.hpp"
#include "aim/scenario/scenario_adapter.hpp"
#include "aim/tracking/tracking_engine.hpp"
#include "aim/trajectory/trajectory_planner.hpp"

using namespace aim;
using namespace aim::pipeline;
using namespace aim::policy;
using namespace aim::safety;
using namespace aim::scenario;
using namespace aim::tracking;
using namespace aim::trajectory;
using namespace aim::calibration;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

void test_end_to_end_correlation_tracing() {
    std::cout << "[Test 1] End-to-end correlation ID tracing & nominal hit..." << std::endl;

    TrackingEngine tracker{};
    aim::TrackerConfig t_cfg{};
    TEST_ASSERT(tracker.initialize(t_cfg));

    UtilityAimPolicy policy{};
    PolicyManifest p_man{};
    TEST_ASSERT(policy.initialize(p_man));

    TrajectoryPlanner planner{};
    NullActuator actuator{};
    ActuatorConfig a_cfg{};
    TEST_ASSERT(actuator.initialize(a_cfg));
    TEST_ASSERT(actuator.start());

    SafetySupervisor safety{};
    safety.set_focus_callback([] { return true; });
    safety.set_calibration_callback([] { return true; });
    ScenarioAdapter scenario{};
    TEST_ASSERT(scenario.load_profile(make_generic_profile()) == ScenarioProfileStatus::ok);

    CalibrationProfile cal_prof{};
    cal_prof.counts_per_pixel_x = 1.0f;
    cal_prof.counts_per_pixel_y = 1.0f;
    CalibrationModel cal_model{cal_prof};

    ClosedLoopPipeline pipeline{
        &tracker, &policy, &planner, &actuator, &safety, &scenario, cal_model
    };

    MonotonicNs sim_time_ns = 1'000'000'000LL;

    // Create observation with unique CorrelationId
    bus::TargetObservationBatch batch{};
    batch.header.sequence_id = 42;
    batch.header.source_timestamp_ns = sim_time_ns;
    batch.header.pipeline_run_id = 999;
    batch.captured_at_ns = sim_time_ns;

    bus::TargetObservation target{};
    target.source_id = 101;
    target.center_px = PixelPoint{962.0f, 541.0f}; // 2.2 px from crosshair (960, 540)
    target.effective_radius_px = 15.0f;
    target.confidence = 0.95f;
    target.covariance_px2 = Covariance2D{4.0f, 0.0f, 4.0f}; // sigma = 2.0 px << 15.0 px
    batch.add_target(target);

    ObservableScenarioState state{};
    state.foreground_confirmed = true;

    // Process frame
    TEST_ASSERT(pipeline.process_frame(batch, state, sim_time_ns + 1'000'000LL)); // 1 ms latency

    const auto& cmd = pipeline.last_command();
    // Verify exact correlation ID provenance trace!
    TEST_ASSERT(cmd.correlation_id.sequence_id == 42);
    TEST_ASSERT(cmd.correlation_id.source_timestamp_ns == sim_time_ns);
    TEST_ASSERT(cmd.correlation_id.pipeline_run_id == 999);
    TEST_ASSERT(pipeline.last_tracks().track_count == 1);
    TEST_ASSERT(pipeline.last_plan().correlation_id.sequence_id == 42);

    // Verify shot gating authorized fire!
    const auto& shot = pipeline.last_shot_result();
    TEST_ASSERT(shot.authorize_fire);
    TEST_ASSERT(shot.decision == ShotGateDecision::authorized);
    TEST_ASSERT(cmd.button_transition.button == MouseButton::left);

    std::cout << "  -> Correlation ID tracing & nominal hit passed." << std::endl;
}

void test_uncertainty_and_alignment_shot_gating() {
    std::cout << "[Test 2] Uncertainty and alignment shot gating rejection..." << std::endl;

    TrackingEngine tracker{};
    UtilityAimPolicy policy{};
    TrajectoryPlanner planner{};
    NullActuator actuator{};
    SafetySupervisor safety{};
    safety.set_focus_callback([] { return true; });
    safety.set_calibration_callback([] { return true; });
    ScenarioAdapter scenario{};
    TEST_ASSERT(scenario.load_profile(make_generic_profile()) == ScenarioProfileStatus::ok);

    CalibrationProfile cal_prof{};
    CalibrationModel cal_model{cal_prof};

    ClosedLoopPipeline pipeline{
        &tracker, &policy, &planner, &actuator, &safety, &scenario, cal_model
    };

    MonotonicNs sim_time_ns = 2'000'000'000LL;
    ObservableScenarioState state{};
    state.foreground_confirmed = true;

    // Case A: Excessive uncertainty (covariance sigma > radius)
    {
        bus::TargetObservationBatch batch{};
        batch.captured_at_ns = sim_time_ns;
        bus::TargetObservation target{};
        target.source_id = 201;
        target.center_px = PixelPoint{960.0f, 540.0f}; // Perfectly centered
        target.effective_radius_px = 10.0f;
        target.confidence = 0.95f;
        target.covariance_px2 = Covariance2D{400.0f, 0.0f, 400.0f}; // sigma = 20 px > 10 px radius!
        batch.add_target(target);

        pipeline.process_frame(batch, state, sim_time_ns + 1'000'000LL);
        TEST_ASSERT(!pipeline.last_shot_result().authorize_fire);
        TEST_ASSERT(pipeline.last_shot_result().decision == ShotGateDecision::rejected_uncertainty_exceeded);
    }

    // Case B: Alignment miss (crosshair far from target)
    {
        bus::TargetObservationBatch batch{};
        batch.captured_at_ns = sim_time_ns;
        bus::TargetObservation target{};
        target.source_id = 202;
        target.center_px = PixelPoint{1200.0f, 700.0f}; // 280 px away!
        target.effective_radius_px = 15.0f;
        target.confidence = 0.95f;
        target.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
        batch.add_target(target);

        pipeline.process_frame(batch, state, sim_time_ns + 1'000'000LL);
        TEST_ASSERT(!pipeline.last_shot_result().authorize_fire);
        TEST_ASSERT(pipeline.last_shot_result().decision == ShotGateDecision::rejected_alignment_miss);
    }

    // Case C: Low confidence (< 0.80)
    {
        bus::TargetObservationBatch batch{};
        batch.captured_at_ns = sim_time_ns;
        bus::TargetObservation target{};
        target.source_id = 203;
        target.center_px = PixelPoint{960.0f, 540.0f};
        target.effective_radius_px = 15.0f;
        target.confidence = 0.50f; // low confidence
        target.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
        batch.add_target(target);

        pipeline.process_frame(batch, state, sim_time_ns + 1'000'000LL);
        TEST_ASSERT(!pipeline.last_shot_result().authorize_fire);
        TEST_ASSERT(pipeline.last_shot_result().decision == ShotGateDecision::rejected_low_confidence);
    }

    std::cout << "  -> Shot gating rejection tests passed." << std::endl;
}

void test_stale_deadline_and_safety_latches() {
    std::cout << "[Test 3] Stale deadline and safety latch containment..." << std::endl;

    TrackingEngine tracker{};
    UtilityAimPolicy policy{};
    TrajectoryPlanner planner{};
    NullActuator actuator{};
    SafetySupervisor safety{};
    safety.set_focus_callback([] { return true; });
    safety.set_calibration_callback([] { return true; });
    ScenarioAdapter scenario{};
    TEST_ASSERT(scenario.load_profile(make_generic_profile()) == ScenarioProfileStatus::ok);

    CalibrationProfile cal_prof{};
    CalibrationModel cal_model{cal_prof};

    ClosedLoopPipeline pipeline{
        &tracker, &policy, &planner, &actuator, &safety, &scenario, cal_model
    };

    ObservableScenarioState state{};
    state.foreground_confirmed = true;

    // Case 1: Stale observation (> 10ms old)
    bus::TargetObservationBatch batch{};
    batch.captured_at_ns = 1'000'000'000LL;
    bus::TargetObservation target{};
    target.source_id = 301;
    target.center_px = PixelPoint{960.0f, 540.0f};
    target.effective_radius_px = 15.0f;
    target.confidence = 0.95f;
    batch.add_target(target);

    // Evaluate at 25ms later (> 10ms deadline!)
    TEST_ASSERT(!pipeline.process_frame(batch, state, 1'025'000'000LL));
    TEST_ASSERT(pipeline.stats().total_stale_dropped == 1);

    // Case 2: Emergency Stop Latch
    safety.trigger_emergency_stop(SafetyReason::emergency_stop_triggered);
    TEST_ASSERT(!pipeline.process_frame(batch, state, 1'002'000'000LL));
    TEST_ASSERT(pipeline.stats().total_safety_rejections == 1);

    std::cout << "  -> Stale deadline and safety latch containment passed." << std::endl;
}

struct ReplayTracker final : ITrackingEngine {
    bool initialize(const aim::TrackerConfig&) noexcept override { return true; }
    void reset() noexcept override {}
    TrackerHealth health() const noexcept override { return {}; }
    bool update(const bus::TargetObservationBatch& observations, MonotonicNs now,
                bus::TrackedTargetBatch& tracks) noexcept override {
        tracks.clear();
        if (observations.target_count == 0) return true;
        bus::TrackedTarget track{};
        track.track_id = 1;
        track.predicted_center_px = observations.targets[0].center_px;
        track.prediction_time_ns = now;
        track.confidence = 0.99f;
        track.effective_radius_px = 1.0f;
        track.total_visible_frames = static_cast<std::uint32_t>(observations.header.sequence_id);
        return tracks.add_track(track);
    }
};
struct ReplayPolicy final : IAimPolicy {
    bool initialize(const PolicyManifest&) noexcept override { return true; }
    void reset() noexcept override {}
    PolicyContract contract() const noexcept override { return {}; }
    bool choose(const PolicyInput& input, bus::AimIntent& intent) noexcept override {
        intent = {};
        if (!input.tracks || input.tracks->track_count == 0) return false;
        intent.target_track_id = 1;
        intent.target_aim_px = input.tracks->tracks[0].predicted_center_px;
        return true; // Motion-only replay; never requests a click.
    }
};

void test_calibrated_full_plan_convergence() {
    std::array<MonotonicNs, 8192> timings{};
    std::size_t samples = 0;
    float worst_error = 0.0f;
    float worst_overshoot = 0.0f;
    for (const unsigned refresh_ms : {1u, 7u}) {
        for (const float distance : {7.0f, 15.0f, 50.0f, 100.0f, 200.0f, 300.0f}) {
            ReplayTracker tracker;
            ReplayPolicy policy;
            TrajectoryPlanner planner;
            NullActuator actuator;
            TEST_ASSERT(actuator.initialize({}));
            TEST_ASSERT(actuator.start());
            CalibrationProfile profile;
            profile.counts_per_pixel_x = 2.0f;
            profile.counts_per_pixel_y = 3.0f;
            profile.cross_coupling_xy = 0.2f;
            CalibrationModel calibration{profile};
            ClosedLoopPipeline pipeline{&tracker, &policy, &planner, &actuator,
                                        nullptr, nullptr, calibration};
            PixelPoint error{distance * 0.8f, -distance * 0.6f};
            bus::TargetObservationBatch observations{};
            bus::TargetObservation target{};
            target.confidence = 0.99f;
            target.effective_radius_px = 1.0f;
            observations.add_target(target);
            const MonotonicNs base = 1'000'000'000LL;
            for (unsigned tick = 0; tick < 400; ++tick) {
                const auto now = base + static_cast<MonotonicNs>(tick) * 1'000'000LL;
                if (tick >= 20) allocation_probe::begin();
                const bool tick_ok = pipeline.tick_actuation({}, now);
                const auto tick_allocations = allocation_probe::end();
                TEST_ASSERT(tick_ok);
                TEST_ASSERT(tick_allocations.calls == 0);
                for (const auto& command : actuator.recorded_commands()) {
                    TEST_ASSERT(command.button_transition.action == ButtonAction::none);
                    TEST_ASSERT(command.generated_at_ns == command.correlation_id.source_timestamp_ns);
                    TEST_ASSERT(command.desired_apply_time_ns <= now);
                    TEST_ASSERT(now - command.generated_at_ns <= 10'000'000LL);
                    PixelDisplacement displacement{};
                    TEST_ASSERT(calibration.try_counts_to_pixels(command.delta_x_counts, command.delta_y_counts, displacement));
                    error.x -= displacement.dx_px;
                    error.y -= displacement.dy_px;
                }
                actuator.clear_recorded_commands();
                worst_overshoot = std::max(worst_overshoot, std::max(-error.x, error.y));
                if (tick % refresh_ms != 0) continue;
                observations.header = {tick + 1, now, 77, 0};
                observations.captured_at_ns = now;
                observations.targets[0].center_px = {960.0f + error.x, 540.0f + error.y};
                const auto start = std::chrono::steady_clock::now();
                if (tick >= 20) allocation_probe::begin();
                const bool frame_ok = pipeline.process_frame(observations, {}, now);
                const auto frame_allocations = allocation_probe::end();
                TEST_ASSERT(frame_ok);
                TEST_ASSERT(frame_allocations.calls == 0);
                const auto stop = std::chrono::steady_clock::now();
                if (tick >= 20 && samples < timings.size()) {
                    timings[samples++] = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                }
                for (const auto& point : pipeline.last_plan().items()) {
                    TEST_ASSERT(std::hypot(point.velocity_px_s.x_per_s, point.velocity_px_s.y_per_s)
                                <= PlannerLimits{}.max_velocity_px_s * 1.01f);
                }
            }
            const float final_error = std::hypot(error.x, error.y);
            std::cout << "Calibration replay error=" << distance << "px, refresh=" << refresh_ms
                      << "ms, final=" << final_error << "px\n";
            TEST_ASSERT(final_error <= 0.75f);
            worst_error = std::max(worst_error, final_error);
            TEST_ASSERT(pipeline.stats().total_commands_dispatched > 0);
            TEST_ASSERT(pipeline.stats().total_safety_rejections == 0);
            TEST_ASSERT(pipeline.stats().total_shots_authorized == 0);
        }
    }
    std::sort(timings.begin(), timings.begin() + samples);
    std::cout << "Synthetic CPU observation-to-plan, NullActuator, no game/GPU; warmup=20 ticks/case, samples="
              << samples << ", p50=" << timings[samples * 50 / 100]
              << "ns p95=" << timings[samples * 95 / 100] << "ns p99=" << timings[samples * 99 / 100]
              << "ns max=" << timings[samples - 1] << "ns; worst final error=" << worst_error
              << "px, warmed C++ allocations=0, overshoot=" << worst_overshoot << "px\n";
}

void test_fresh_replans_across_capture_gaps() {
    ReplayTracker tracker;
    ReplayPolicy policy;
    TrajectoryPlanner planner;
    NullActuator actuator;
    TEST_ASSERT(actuator.initialize({}));
    TEST_ASSERT(actuator.start());
    CalibrationModel calibration;
    ClosedLoopPipeline pipeline{&tracker, &policy, &planner, &actuator,
                                nullptr, nullptr, calibration};
    PixelPoint error{160.0f, -120.0f};
    bus::TargetObservationBatch observations{};
    bus::TargetObservation target{};
    target.confidence = 0.99f;
    target.effective_radius_px = 1.0f;
    observations.add_target(target);
    for (unsigned tick = 0; tick < 1600; ++tick) {
        const auto now = 1'000'000'000LL + static_cast<MonotonicNs>(tick) * 1'000'000LL;
        (void)pipeline.tick_actuation({}, now);
        for (const auto& command : actuator.recorded_commands()) {
            TEST_ASSERT(now - command.correlation_id.source_timestamp_ns <= 10'000'000LL);
            TEST_ASSERT(command.button_transition.action == ButtonAction::none);
            PixelDisplacement displacement{};
            TEST_ASSERT(calibration.try_counts_to_pixels(command.delta_x_counts, command.delta_y_counts, displacement));
            error.x -= displacement.dx_px;
            error.y -= displacement.dy_px;
        }
        actuator.clear_recorded_commands();
        if (tick % 21 != 0) continue;
        // Real capture cadence leaves only three fresh 1 kHz steps per frame.
        const auto captured = now - 7'000'000LL;
        observations.header = {tick + 1, captured, 77, 0};
        observations.captured_at_ns = captured;
        observations.targets[0].center_px = {960.0f + error.x, 540.0f + error.y};
        TEST_ASSERT(pipeline.process_frame(observations, {}, now));
    }
    std::cout << "Gapped capture replay final error=" << std::hypot(error.x, error.y) << "px\n";
    TEST_ASSERT(std::hypot(error.x, error.y) <= 1.0f);
    TEST_ASSERT(pipeline.stats().total_shots_authorized == 0);
}

void test_due_safety_recheck_releases_buttons() {
    for (const bool fail_focus : {false, true}) {
        ReplayTracker tracker;
        ReplayPolicy policy;
        TrajectoryPlanner planner;
        NullActuator actuator;
        TEST_ASSERT(actuator.initialize({}));
        TEST_ASSERT(actuator.start());
        SafetySupervisor safety;
        safety.set_focus_callback([] { return true; });
        safety.set_calibration_callback([] { return true; });
        CalibrationModel calibration;
        ClosedLoopPipeline pipeline{&tracker, &policy, &planner, &actuator, &safety, nullptr, calibration};
        bus::TargetObservationBatch observations{};
        observations.captured_at_ns = 1'000'000'000LL;
        observations.header = {1, observations.captured_at_ns, 1, 0};
        bus::TargetObservation target{};
        target.center_px = {965.0f, 540.0f};
        observations.add_target(target);
        TEST_ASSERT(pipeline.process_frame(observations, {}, observations.captured_at_ns));
        ActuationCommand held{};
        held.button_transition = {MouseButton::left, ButtonAction::press};
        TEST_ASSERT(actuator.submit_latest(held) == SubmitResult::submitted);
        if (fail_focus) safety.set_focus_callback([] { return false; });
        else safety.set_calibration_callback({});
        TEST_ASSERT(!pipeline.tick_actuation({}, observations.captured_at_ns + 1'000'000LL));
        TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
        TEST_ASSERT(pipeline.stats().total_commands_dispatched == 0);
        TEST_ASSERT(pipeline.stats().total_shots_authorized == 0);
        TEST_ASSERT(pipeline.last_plan().point_count == 0);
    }
}
} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M7-02 Closed-Loop & Shot Gate Unit Tests     " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_calibrated_full_plan_convergence();
    test_fresh_replans_across_capture_gaps();
    test_due_safety_recheck_releases_buttons();
    test_end_to_end_correlation_tracing();
    test_uncertainty_and_alignment_shot_gating();
    test_stale_deadline_and_safety_latches();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M7-02 Closed-Loop & Shot Gate Tests Passed Successfully!   " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
