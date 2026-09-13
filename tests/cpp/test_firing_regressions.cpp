#include <cstdlib>
#include <iostream>
#include <limits>

#include "aim/pipeline/closed_loop_pipeline.hpp"
#include "aim/safety/safety_supervisor.hpp"
#include "aim/scenario/scenario_adapter.hpp"
#include "aim/tracking/tracking_engine.hpp"
#include "aim/trajectory/trajectory_planner.hpp"

using namespace aim;
using namespace aim::policy;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Failed: " #condition << " at " << __LINE__ << '\n';        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

namespace {
constexpr MonotonicNs start_ns = 1'000'000'000LL;

bus::TrackedTarget target_at(MonotonicNs time = start_ns) {
  bus::TrackedTarget target{};
  target.track_id = 1;
  target.prediction_time_ns = time;
  target.predicted_center_px = {960.0f, 540.0f};
  target.effective_radius_px = 10.0f;
  target.confidence = 0.95f;
  target.covariance_px2 = {1.0f, 0.0f, 1.0f};
  return target;
}

void test_cadence_and_fresh_receipts() {
  ShotGate gate;
  auto target = target_at();
  const PixelPoint crosshair{960.0f, 540.0f};
  CHECK(gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  // Merely evaluating or rejecting a dispatch cannot consume a shot.
  CHECK(gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  gate.record_dispatched_shot(target, start_ns);
  target.prediction_time_ns += 400'000'000LL;
  CHECK(gate.evaluate(&target, crosshair, target.prediction_time_ns).decision ==
        ShotGateDecision::rejected_no_fresh_observation);
  ++target.total_visible_frames;
  CHECK(gate.evaluate(&target, crosshair, target.prediction_time_ns)
            .authorize_fire);
  target.track_id = 2;
  target.prediction_time_ns = start_ns + 1'000'000LL;
  CHECK(gate.evaluate(&target, crosshair, target.prediction_time_ns).decision ==
        ShotGateDecision::rejected_cooldown);
  target.track_id = 1;
  target.prediction_time_ns = start_ns + 200'000'000LL;
  CHECK(gate.evaluate(&target, crosshair, target.prediction_time_ns).decision ==
        ShotGateDecision::rejected_cooldown);

  gate.reset();
  unsigned shots = 0;
  for (unsigned frame = 0; frame < 1000; ++frame) {
    target =
        target_at(start_ns + static_cast<MonotonicNs>(frame) * 1'000'000LL);
    target.total_visible_frames = frame + 1;
    if (gate.evaluate(&target, crosshair, target.prediction_time_ns)
            .authorize_fire) {
      gate.record_dispatched_shot(target, target.prediction_time_ns);
      ++shots;
    }
  }
  CHECK(shots == 3); // 0, 350 and 700 ms; old behavior generated 1,000 clicks.
  std::cout << "Replay: 1000 fresh aligned frames / 1 simulated second -> "
            << shots << " accepted shots\n";
}

void test_alignment_and_invalid_observations() {
  ShotGate gate;
  const PixelPoint crosshair{960.0f, 540.0f};
  auto target = target_at();
  target.predicted_center_px.x +=
      10.5f; // Previously authorized outside the radius.
  CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  target.predicted_center_px.x =
      967.5f; // Center inside, uncertainty crosses safe interior.
  CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  target.predicted_center_px.x = 967.0f;
  CHECK(gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    target = target_at();
    target.confidence = bad;
    CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
    target = target_at();
    target.predicted_center_px.x = bad;
    CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
    target = target_at();
    target.effective_radius_px = bad;
    CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
    target = target_at();
    target.covariance_px2.xy = bad;
    CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  }
  target = target_at();
  target.covariance_px2 = {1.0f, 2.0f, 1.0f};
  CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  target = target_at();
  target.state = bus::TrackState::occluded;
  CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  target = target_at();
  target.state = bus::TrackState::tentative;
  CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  target = target_at();
  target.prediction_time_ns = 0;
  CHECK(!gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  target = target_at();
  CHECK(!gate.evaluate(&target, crosshair, start_ns + 10'000'001LL)
             .authorize_fire);
  ShotGateConfig unsafe{};
  unsafe.max_alignment_radius_ratio = 1.1f;
  CHECK(
      !ShotGate{unsafe}.evaluate(&target, crosshair, start_ns).authorize_fire);
}

void test_bounded_measured_prediction_horizon() {
  const PixelPoint crosshair{960.0f, 540.0f};
  auto target = target_at(start_ns + 15'000'000LL);
  CHECK(ShotGate{}.evaluate(&target, crosshair, start_ns).decision ==
        ShotGateDecision::rejected_deadline_expired);

  ShotGateConfig measured{};
  measured.max_prediction_lead_ns = 15'000'000LL;
  ShotGate gate{measured};
  CHECK(gate.evaluate(&target, crosshair, start_ns).authorize_fire);
  ++target.prediction_time_ns;
  CHECK(gate.evaluate(&target, crosshair, start_ns).decision ==
        ShotGateDecision::rejected_deadline_expired);
  target.prediction_time_ns = start_ns - 11'000'000LL;
  CHECK(gate.evaluate(&target, crosshair, start_ns).decision ==
        ShotGateDecision::rejected_deadline_expired);

  measured.max_prediction_lead_ns = 100'000'000LL;
  target.prediction_time_ns = start_ns + measured.max_prediction_lead_ns;
  CHECK(ShotGate{measured}.evaluate(&target, crosshair, start_ns).authorize_fire);
  for (const MonotonicNs invalid : {MonotonicNs{0}, MonotonicNs{100'000'001LL}}) {
    measured.max_prediction_lead_ns = invalid;
    CHECK(ShotGate{measured}.evaluate(&target, crosshair, start_ns).decision ==
          ShotGateDecision::rejected_invalid_input);
  }
}

struct Tracker final : ITrackingEngine {
  bool succeeds{true};
  bool initialize(const TrackerConfig &) noexcept override { return true; }
  void reset() noexcept override {}
  TrackerHealth health() const noexcept override { return {}; }
  bool update(const bus::TargetObservationBatch &batch, MonotonicNs time,
              bus::TrackedTargetBatch &out) noexcept override {
    out.clear();
    auto target = target_at(time);
    target.total_visible_frames =
        static_cast<std::uint32_t>(batch.header.sequence_id);
    out.add_track(target);
    return succeeds;
  }
};

struct Policy final : IAimPolicy {
  bool permits{true};
  bool succeeds{true};
  float offset_px{0.0f};
  PolicyContract contract() const noexcept override { return {}; }
  bool initialize(const PolicyManifest &) noexcept override { return true; }
  void reset() noexcept override {}
  bool choose(const PolicyInput &input, bus::AimIntent &out) noexcept override {
    out = {};
    out.target_track_id = 1;
    out.target_aim_px = input.crosshair.center_px;
    out.target_aim_px.x += offset_px;
    out.authorize_fire = permits;
    return succeeds; // Deliberately leaves a plausible partial result on
                     // failure.
  }
};

struct Planner final : ITrajectoryPlanner {
  bool succeeds{true};
  bool plan(const bus::AimIntent &, const ControlState &, const PlannerLimits &,
            TrajectoryPlan &out) noexcept override {
    out = {};
    return succeeds;
  }
  void cancel() noexcept override {}
  void reset() noexcept override {}
};

void test_measured_prediction_preserves_source_freshness() {
  Tracker tracker;
  Policy policy;
  Planner planner;
  NullActuator actuator;
  CHECK(actuator.initialize({}));
  CHECK(actuator.start());
  calibration::CalibrationModel calibration;
  pipeline::ClosedLoopPipelineConfig config{};
  config.command_lead_time_ns = 15'000'000LL;
  bus::TargetObservationBatch batch{};
  batch.header.sequence_id = 1;
  batch.captured_at_ns = start_ns;
  batch.header.source_timestamp_ns = start_ns;

  // The host must explicitly supply a matching measured shot-gate horizon.
  pipeline::ClosedLoopPipeline inconsistent{
      &tracker, &policy, &planner, &actuator, nullptr, nullptr, calibration,
      config};
  CHECK(!inconsistent.process_frame(batch, {}, start_ns));
  CHECK(actuator.health().total_commands_submitted == 0);

  config.shot_gate_config.max_prediction_lead_ns = 15'000'000LL;
  pipeline::ClosedLoopPipeline pipeline{
      &tracker, &policy, &planner, &actuator, nullptr, nullptr, calibration,
      config};
  CHECK(pipeline.process_frame(batch, {}, start_ns));
  CHECK(pipeline.last_tracks().tracks[0].prediction_time_ns ==
        start_ns + 15'000'000LL);
  CHECK(pipeline.last_shot_result().authorize_fire);
  CHECK(pipeline.last_command().generated_at_ns == start_ns);
  CHECK(pipeline.last_command().correlation_id.source_timestamp_ns == start_ns);
  CHECK(actuator.health().total_commands_submitted == 1);

  ++batch.header.sequence_id;
  CHECK(!pipeline.process_frame(batch, {}, start_ns + 11'000'000LL));
  CHECK(pipeline.stats().total_stale_dropped == 1);
  // Fresh capture metadata must not conceal an older original source time.
  batch.captured_at_ns = start_ns + 11'000'000LL;
  CHECK(!pipeline.process_frame(batch, {}, batch.captured_at_ns));
  CHECK(pipeline.stats().total_stale_dropped == 2);
  CHECK(actuator.health().total_commands_submitted == 1);
  CHECK(!pipeline.last_shot_result().authorize_fire);

  const MonotonicNs too_late =
      (std::numeric_limits<MonotonicNs>::max)() - 14'000'000LL;
  batch.captured_at_ns = too_late;
  batch.header.source_timestamp_ns = too_late;
  CHECK(!pipeline.process_frame(batch, {}, too_late));
  CHECK(actuator.health().total_commands_submitted == 1);

  config.command_lead_time_ns = 100'000'001LL;
  config.shot_gate_config.max_prediction_lead_ns = config.command_lead_time_ns;
  pipeline::ClosedLoopPipeline unbounded{
      &tracker, &policy, &planner, &actuator, nullptr, nullptr, calibration,
      config};
  batch.captured_at_ns = start_ns;
  batch.header.source_timestamp_ns = start_ns;
  CHECK(!unbounded.process_frame(batch, {}, start_ns));
  CHECK(actuator.health().total_commands_submitted == 1);
}

void test_pipeline_veto_failure_and_cancellation() {
  Tracker tracker;
  Policy policy;
  Planner planner;
  NullActuator actuator;
  CHECK(actuator.initialize({}));
  CHECK(actuator.start());
  calibration::CalibrationModel calibration;
  pipeline::ClosedLoopPipeline pipeline{
      &tracker, &policy, &planner, &actuator, nullptr, nullptr, calibration};
  bus::TargetObservationBatch batch{};
  batch.header.sequence_id = 1;
  batch.captured_at_ns = start_ns;
  policy.permits = false;
  policy.offset_px = 100.0f;
  CHECK(pipeline.process_frame(batch, {}, start_ns));
  CHECK(pipeline.last_shot_result().decision ==
        ShotGateDecision::rejected_policy_veto);
  CHECK(actuator.health().total_commands_submitted == 0);
  // An empty successful plan must not silently become a full-error movement.
  policy.offset_px = 0.0f;
  policy.permits = true;
  policy.succeeds = false;
  CHECK(pipeline.process_frame(batch, {}, start_ns));
  CHECK(actuator.health().total_commands_submitted == 0);
  policy.succeeds = true;
  planner.succeeds = false;
  CHECK(!pipeline.process_frame(batch, {}, start_ns));
  CHECK(actuator.health().total_commands_submitted == 0);
  CHECK(!pipeline.last_shot_result().authorize_fire);
  CHECK(pipeline.last_shot_result().button_transition.action ==
        ButtonAction::none);
  planner.succeeds = true;
  actuator
      .shutdown(); // A rejected shot is not reported as dispatched or consumed.
  CHECK(!pipeline.process_frame(batch, {}, start_ns));
  CHECK(pipeline.stats().total_commands_dispatched == 0);
  CHECK(pipeline.stats().total_shots_authorized == 0);
  CHECK(!pipeline.last_shot_result().authorize_fire);
  CHECK(pipeline.last_shot_result().button_transition.action ==
        ButtonAction::none);
  CHECK(actuator.start());
  CHECK(pipeline.process_frame(batch, {}, start_ns));
  CHECK(pipeline.stats().total_shots_authorized == 1);

  ActuationCommand held{};
  held.button_transition = {MouseButton::left, ButtonAction::press};
  CHECK(actuator.submit_latest(held) == SubmitResult::submitted);
  CHECK(actuator.health().pressed_buttons_mask != 0);
  CHECK(!pipeline.process_frame(batch, {}, start_ns + 11'000'000LL));
  CHECK(actuator.health().pressed_buttons_mask == 0);
  tracker.succeeds = false;
  CHECK(!pipeline.process_frame(batch, {}, start_ns));
  CHECK(pipeline.last_intent().target_track_id == 0);
}

void test_no_target_no_motion() {
  trajectory::HybridTrajectoryPlanner planner;
  TrajectoryPlan plan{};
  plan.point_count = 10;
  CHECK(planner.plan({}, {}, {}, plan));
  CHECK(plan.point_count == 0);
}

void test_focus_stop_and_cooldown_recovery() {
  Tracker tracker;
  Policy policy;
  Planner planner;
  NullActuator actuator;
  CHECK(actuator.initialize({}));
  CHECK(actuator.start());
  scenario::ScenarioAdapter scenario;
  CHECK(scenario.load_profile(scenario::make_generic_profile()) ==
        ScenarioProfileStatus::ok);
  safety::SafetySupervisor safety;
  safety.set_focus_callback([] { return true; });
  safety.set_calibration_callback([] { return true; });
  calibration::CalibrationModel calibration;
  pipeline::ClosedLoopPipeline pipeline{
      &tracker, &policy, &planner, &actuator, &safety, &scenario, calibration};
  bus::TargetObservationBatch batch{};
  batch.header.sequence_id = 1;
  batch.captured_at_ns = start_ns;
  ObservableScenarioState state{};
  state.foreground_confirmed = true;
  CHECK(pipeline.process_frame(batch, state, start_ns));
  CHECK(pipeline.stats().total_shots_authorized == 1);
  ActuationCommand held{};
  held.button_transition = {MouseButton::left, ButtonAction::press};
  CHECK(actuator.submit_latest(held) == SubmitResult::submitted);
  state.foreground_confirmed = false;
  CHECK(!pipeline.process_frame(batch, state, start_ns));
  CHECK(actuator.health().pressed_buttons_mask == 0);
  state.foreground_confirmed = true;
  ++batch.header.sequence_id;
  batch.captured_at_ns += 7'000'000LL;
  CHECK(pipeline.process_frame(batch, state, batch.captured_at_ns));
  CHECK(pipeline.last_shot_result().decision ==
        ShotGateDecision::rejected_cooldown);
  CHECK(pipeline.stats().total_shots_authorized == 1);
  CHECK(actuator.submit_latest(held) == SubmitResult::submitted);
  safety.trigger_emergency_stop();
  CHECK(!pipeline.process_frame(batch, state, batch.captured_at_ns));
  CHECK(actuator.health().pressed_buttons_mask == 0);
  CHECK(!pipeline.last_shot_result().authorize_fire);
}

void test_reacquisition_and_target_switching() {
  tracking::TrackingEngine tracker;
  CHECK(tracker.initialize(aim::TrackerConfig{}));
  bus::TargetObservationBatch batch{};
  batch.captured_at_ns = start_ns;
  bus::TargetObservation observation{};
  observation.center_px = {960.0f, 540.0f};
  observation.confidence = 0.98f;
  observation.effective_radius_px = 20.0f;
  observation.covariance_px2 = {1.0f, 0.0f, 1.0f};
  batch.add_target(observation);
  bus::TrackedTargetBatch tracks{};
  CHECK(tracker.update(batch, start_ns, tracks));
  CHECK(tracks.track_count == 1);
  const auto id = tracks.tracks[0].track_id;
  ShotGate gate;
  CHECK(gate.evaluate(&tracks.tracks[0], {960.0f, 540.0f}, start_ns)
            .authorize_fire);
  gate.record_dispatched_shot(tracks.tracks[0], start_ns);
  batch.target_count = 0;
  batch.captured_at_ns += 7'000'000LL;
  CHECK(tracker.update(batch, batch.captured_at_ns, tracks));
  CHECK(
      !gate.evaluate(&tracks.tracks[0], {960.0f, 540.0f}, batch.captured_at_ns)
           .authorize_fire);
  batch.add_target(observation);
  batch.captured_at_ns = start_ns + 14'000'000LL;
  CHECK(tracker.update(batch, batch.captured_at_ns, tracks));
  CHECK(tracks.tracks[0].track_id == id);
  CHECK(tracks.tracks[0].total_missed_frames == 1);
  // Keep source cadence realistic while waiting out the accepted shot receipt.
  for (unsigned frame = 0; frame < 60; ++frame) {
    batch.captured_at_ns += 7'000'000LL;
    CHECK(tracker.update(batch, batch.captured_at_ns, tracks));
  }
  CHECK(tracks.tracks[0].track_id == id);
  CHECK(tracks.tracks[0].total_missed_frames == 1);
  CHECK(gate.evaluate(&tracks.tracks[0], {960.0f, 540.0f}, batch.captured_at_ns)
            .authorize_fire);

  gate.reset();
  auto target = target_at();
  gate.record_dispatched_shot(target, start_ns);
  target.track_id = 2;
  target.prediction_time_ns += 150'000'000LL;
  CHECK(gate.evaluate(&target, {960.0f, 540.0f}, target.prediction_time_ns)
            .authorize_fire);
  gate.record_dispatched_shot(target, target.prediction_time_ns);
  target.track_id = 1;
  ++target.total_visible_frames;
  target.prediction_time_ns += 150'000'000LL;
  CHECK(gate.evaluate(&target, {960.0f, 540.0f}, target.prediction_time_ns)
            .decision == ShotGateDecision::rejected_cooldown);
}
} // namespace

int main() {
  test_cadence_and_fresh_receipts();
  test_alignment_and_invalid_observations();
  test_bounded_measured_prediction_horizon();
  test_measured_prediction_preserves_source_freshness();
  test_pipeline_veto_failure_and_cancellation();
  test_no_target_no_motion();
  test_reacquisition_and_target_switching();
  test_focus_stop_and_cooldown_recovery();
  std::cout << "Firing regressions passed (no OS input).\n";
}
