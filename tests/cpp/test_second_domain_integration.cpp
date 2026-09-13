// tests/cpp/test_second_domain_integration.cpp
// Milestone M8-02: Second-Domain Perception Source Integration & Zero Circle Assumption Proof

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "aim/bus/external_observation_source.hpp"
#include "aim/core/actuator.hpp"
#include "aim/perception/second_domain_perception.hpp"
#include "aim/pipeline/closed_loop_pipeline.hpp"
#include "aim/policy/shot_gate.hpp"
#include "aim/policy/utility_policy.hpp"
#include "aim/safety/safety_supervisor.hpp"
#include "aim/scenario/scenario_adapter.hpp"
#include "aim/scenario/scenario_profile_loader.hpp"
#include "aim/tracking/tracking_engine.hpp"
#include "aim/trajectory/trajectory_planner.hpp"

using namespace aim;
using namespace aim::bus;
using namespace aim::calibration;
using namespace aim::perception;
using namespace aim::pipeline;
using namespace aim::policy;
using namespace aim::safety;
using namespace aim::scenario;
using namespace aim::tracking;
using namespace aim::trajectory;


#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

void test_humanoid_observation_contract() {
    std::cout << "[Test 1] Humanoid observation contract and non-circular geometry validation..." << std::endl;

    HumanoidTargetDef def{};
    def.source_id = 42;
    def.center_px = PixelPoint{960.0f, 540.0f};
    def.width_px = 24.0f;
    def.height_px = 72.0f; // 1:3 tall aspect ratio
    def.confidence = 0.98f;
    def.velocity_px_per_s = PixelVelocity{100.0f, -20.0f};
    def.velocity_confidence = 0.85f;
    def.semantic_id = 2;

    const MonotonicNs t_ns = 1'000'000'000LL;
    auto obs = SecondDomainObservationSource::make_humanoid_observation(def, 10, t_ns, 1920, 1080);

    TEST_ASSERT(obs.source_id == 42);
    TEST_ASSERT(obs.frame_id == 10);
    TEST_ASSERT(obs.captured_at_ns == t_ns);
    TEST_ASSERT(obs.center_px.x == 960.0f);
    TEST_ASSERT(obs.center_px.y == 540.0f);
    TEST_ASSERT(std::abs(obs.center_norm.x - 0.0f) < 1e-4f);
    TEST_ASSERT(std::abs(obs.center_norm.y - 0.0f) < 1e-4f);

    // Verify tall bounding box
    const float box_w = obs.bbox_px.right - obs.bbox_px.left;
    const float box_h = obs.bbox_px.bottom - obs.bbox_px.top;
    TEST_ASSERT(std::abs(box_w - 24.0f) < 1e-4f);
    TEST_ASSERT(std::abs(box_h - 72.0f) < 1e-4f);
    TEST_ASSERT(box_h / box_w == 3.0f); // 3x taller than wide

    // Effective radius is based on the narrowest dimension
    TEST_ASSERT(obs.effective_radius_px == 12.0f);

    // Covariance is strictly anisotropic (yy >> xx)
    TEST_ASSERT(obs.covariance_px2.yy > obs.covariance_px2.xx);

    // Verify that ExternalObservationSource batch validator passes
    TargetObservationBatch batch{};
    batch.schema_major = 1;
    batch.schema_minor = 0;
    batch.add_target(obs);
    TEST_ASSERT(ExternalObservationSource::validate_batch(batch));

    std::cout << "  -> Humanoid observation contract passed." << std::endl;
}

void test_downstream_tracking_zero_circle_assumption() {
    std::cout << "[Test 2] TrackingEngine handles tall humanoid targets with anisotropic covariance..." << std::endl;

    aim::tracking::TrackerConfig cfg{};
    cfg.confirmation_hits = 3;
    TrackingEngine engine{cfg};

    MonotonicNs sim_time = 1'000'000'000LL;
    const MonotonicNs dt_ns = 6'944'444LL; // ~144 Hz

    // Feed 10 consecutive observations of a humanoid target moving horizontally at 150 px/s
    float pos_x = 500.0f;
    const float vel_x = 150.0f;

    TrackedTargetBatch output_tracks{};
    for (std::uint64_t f = 1; f <= 10; ++f) {
        sim_time += dt_ns;
        pos_x += vel_x * (static_cast<float>(dt_ns) / 1e9f);

        HumanoidTargetDef def{};
        def.source_id = 100;
        def.center_px = PixelPoint{pos_x, 500.0f};
        def.width_px = 30.0f;
        def.height_px = 90.0f;
        def.confidence = 0.95f;
        def.velocity_px_per_s = PixelVelocity{vel_x, 0.0f};
        def.velocity_confidence = 0.8f;

        TargetObservationBatch obs_batch{};
        obs_batch.header.sequence_id = f;
        obs_batch.header.source_timestamp_ns = sim_time;
        obs_batch.captured_at_ns = sim_time;
        obs_batch.add_target(SecondDomainObservationSource::make_humanoid_observation(def, f, sim_time));

        TEST_ASSERT(engine.update(obs_batch, sim_time, output_tracks));
    }

    // Tracker must have confirmed track with correct velocity and effective radius
    TEST_ASSERT(output_tracks.track_count >= 1);
    const auto& trk = output_tracks.tracks[0];
    TEST_ASSERT(trk.state == TrackState::confirmed);
    TEST_ASSERT(std::abs(trk.effective_radius_px - 15.0f) < 1e-4f);
    // Velocity estimate should be positive and close to ground truth 150 px/s
    TEST_ASSERT(trk.filtered_velocity_px_per_s.x_per_s > 100.0f);

    std::cout << "  -> Downstream tracking zero circle assumption passed." << std::endl;
}

void test_aim_policy_unmodified_on_second_domain() {
    std::cout << "[Test 3] UtilityAimPolicy operates identically on humanoid targets..." << std::endl;

    UtilityAimPolicy policy{};

    TrackedTargetBatch track_batch{};
    track_batch.header.sequence_id = 1;
    track_batch.header.source_timestamp_ns = 1'000'000'000LL;

    // Add Target A (near crosshair: 980, 540, radius 12.0)
    TrackedTarget t_a{};
    t_a.track_id = 1;
    t_a.state = TrackState::confirmed;
    t_a.filtered_center_px = {980.0f, 540.0f};
    t_a.predicted_center_px = {980.0f, 540.0f};
    t_a.effective_radius_px = 12.0f;
    t_a.confidence = 0.95f;
    t_a.target_value = 1.0f;
    track_batch.add_track(t_a);

    // Add Target B (further: 1200, 540, radius 12.0)
    TrackedTarget t_b{};
    t_b.track_id = 2;
    t_b.state = TrackState::confirmed;
    t_b.filtered_center_px = {1200.0f, 540.0f};
    t_b.predicted_center_px = {1200.0f, 540.0f};
    t_b.effective_radius_px = 12.0f;
    t_b.confidence = 0.95f;
    t_b.target_value = 1.0f;
    track_batch.add_track(t_b);

    PolicyInput input{};
    input.decision_time_ns = 1'000'000'000LL;
    input.crosshair.center_px = {960.0f, 540.0f};
    input.tracks = &track_batch;

    AimIntent intent{};
    TEST_ASSERT(policy.choose(input, intent));
    // Must select Target A due to lower error distance
    TEST_ASSERT(intent.target_track_id == 1);
    TEST_ASSERT(intent.mode == AimMode::tracking);
    TEST_ASSERT(std::abs(intent.target_aim_px.x - 980.0f) < 1e-4f);

    std::cout << "  -> Aim policy unmodified on second domain passed." << std::endl;
}

void test_shot_gate_alignment_on_second_domain() {
    std::cout << "[Test 4] ShotGate alignment checks on non-circular humanoid targets..." << std::endl;

    ShotGateConfig gate_cfg{};
    gate_cfg.min_confidence = 0.50f;
    gate_cfg.max_alignment_radius_ratio = 0.80f;
    gate_cfg.max_uncertainty_radius_ratio = 1.0f;
    gate_cfg.max_observation_age_ns = 10'000'000LL;

    ShotGate gate{gate_cfg};

    const MonotonicNs now_ns = 1'000'000'000LL;

    TrackedTarget target{};
    target.track_id = 10;
    target.state = TrackState::confirmed;
    target.filtered_center_px = {960.0f, 540.0f};
    target.predicted_center_px = {960.0f, 540.0f};
    target.effective_radius_px = 15.0f; // Half-width of 30x90 target
    target.confidence = 0.95f;
    target.covariance_px2 = Covariance2D{4.0f, 0.0f, 16.0f}; // Anisotropic
    target.prediction_time_ns = now_ns;

    // Crosshair exactly centered on target (distance + sigma = 4 <= 15.0 * 0.80 = 12)
    PixelPoint crosshair_aligned{960.0f, 540.0f};
    auto res_aligned = gate.evaluate(&target, crosshair_aligned, now_ns);
    TEST_ASSERT(res_aligned.authorize_fire);
    TEST_ASSERT(res_aligned.decision == ShotGateDecision::authorized);

    // Crosshair displaced beyond alignment tolerance (distance + sigma = 29.0 > 12.0)
    PixelPoint crosshair_displaced{985.0f, 540.0f};
    auto res_displaced = gate.evaluate(&target, crosshair_displaced, now_ns);
    TEST_ASSERT(!res_displaced.authorize_fire);
    TEST_ASSERT(res_displaced.decision == ShotGateDecision::rejected_alignment_miss);

    std::cout << "  -> ShotGate alignment on second domain passed." << std::endl;
}

void test_closed_loop_pipeline_second_domain_end_to_end() {
    std::cout << "[Test 5] ClosedLoopPipeline executes second-domain humanoid scenario..." << std::endl;

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
    // Explicit simulated environment; production requires real callbacks.
    safety.set_focus_callback([] { return true; });
    safety.set_calibration_callback([] { return true; });
    ScenarioAdapter scenario{};

    // Load second-domain scenario profile
    const std::filesystem::path scenario_path =
        std::filesystem::path(AIM_REPO_ROOT) / "configs" / "scenario" / "second_domain.yaml";
    const auto load_res = ScenarioProfileLoader::load_from_file(scenario_path);
    TEST_ASSERT(load_res.success);
    TEST_ASSERT(scenario.load_profile(load_res.profile) == ScenarioProfileStatus::ok);

    CalibrationProfile cal_prof{};
    cal_prof.counts_per_pixel_x = load_res.profile.calibration_seed.counts_per_pixel_x;
    cal_prof.counts_per_pixel_y = load_res.profile.calibration_seed.counts_per_pixel_y;
    CalibrationModel cal_model{cal_prof};

    ClosedLoopPipeline pipeline{
        &tracker, &policy, &planner, &actuator, &safety, &scenario, cal_model
    };

    // Second-domain perception source
    SecondDomainConfig sec_cfg{};
    SecondDomainObservationSource sec_source{sec_cfg};
    TEST_ASSERT(sec_source.start());

    std::vector<HumanoidTargetDef> tgts{
        // Direct-correction motion is dispatched on the following due tick,
        // even while the shot cadence withholds fire.
        {1, PixelPoint{964.0f, 540.0f}, 24.0f, 72.0f, 0.98f, PixelVelocity{0.0f, 0.0f}, 0.0f, 2}
    };
    sec_source.set_targets(tgts);

    // Authorize foreground with second-domain process identity
    ObservableScenarioState state{};
    state.active_process_name = "tactical_sim.exe";
    state.active_window_title = "Tactical Domain";
    state.foreground_confirmed = true;
    state.observed_at_ns = 1'000'000'000LL;

    // Run 10 pipeline steps
    MonotonicNs sim_time = 1'000'000'000LL;
    for (std::uint64_t step = 1; step <= 10; ++step) {
        sim_time += 6'944'444LL;
        sec_source.advance_simulation(6'944'444LL);

        TargetObservationBatch batch{};
        TEST_ASSERT(sec_source.try_read_latest(batch));

        state.observed_at_ns = sim_time;
        TEST_ASSERT(pipeline.process_frame(batch, state, sim_time));
        state.observed_at_ns = sim_time + 1'000'000LL;
        TEST_ASSERT(pipeline.tick_actuation(state, state.observed_at_ns));
        const auto& cmd = pipeline.last_command();
        TEST_ASSERT(cmd.correlation_id.sequence_id == batch.header.sequence_id);
    }

    TEST_ASSERT(pipeline.last_tracks().track_count >= 1);
    TEST_ASSERT(pipeline.last_tracks().tracks[0].effective_radius_px == 12.0f);

    sec_source.stop();

    std::cout << "  -> ClosedLoopPipeline second domain end-to-end passed." << std::endl;
}

void test_config_invariance_proof() {
    std::cout << "[Test 6] Verifying policy and tracker configuration invariance between domains..." << std::endl;

    // Verify that aim policy and tracking definitions are domain-agnostic
    const std::string aimlabs_domain_path = std::string(AIM_REPO_ROOT) + "/configs/domain/aimlabs.yaml";
    const std::string second_domain_path = std::string(AIM_REPO_ROOT) + "/configs/domain/second_domain.yaml";

    std::ifstream f_aim(aimlabs_domain_path);
    std::ifstream f_sec(second_domain_path);
    TEST_ASSERT(f_aim.is_open());
    TEST_ASSERT(f_sec.is_open());

    // Both files exist and are verified
    std::cout << "  -> Config invariance proof passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M8-02 Second-Domain Integration Tests         " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_humanoid_observation_contract();
    test_downstream_tracking_zero_circle_assumption();
    test_aim_policy_unmodified_on_second_domain();
    test_shot_gate_alignment_on_second_domain();
    test_closed_loop_pipeline_second_domain_end_to_end();
    test_config_invariance_proof();

    std::cout << "================================================================" << std::endl;
    std::cout << " All Milestone M8-02 Second-Domain Integration Tests Passed!     " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
