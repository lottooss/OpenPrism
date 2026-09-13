// tests/cpp/test_portability_conformance.cpp
// Milestone M8-03: Cross-Domain Portability Conformance and No-Policy-Retraining Benchmark

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
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

struct LatencyStats {
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double max_us{0.0};
    double mean_us{0.0};
};

LatencyStats compute_latency_stats(std::vector<double>& samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    LatencyStats stats{};
    stats.p50_us = samples[static_cast<std::size_t>(n * 0.50)];
    stats.p95_us = samples[static_cast<std::size_t>(n * 0.95)];
    stats.p99_us = samples[static_cast<std::size_t>(n * 0.99)];
    stats.max_us = samples.back();
    stats.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(n);
    return stats;
}

namespace {

void test_coordinate_and_covariance_contracts() {
    std::cout << "[Test 1] Verifying coordinate, confidence, and covariance contracts across domains..." << std::endl;

    // 1. Aimlabs-style circular observation
    TargetObservation aim_obs{};
    aim_obs.source_id = 1;
    aim_obs.frame_id = 100;
    aim_obs.captured_at_ns = 1'000'000'000LL;
    aim_obs.center_px = PixelPoint{960.0f, 540.0f};
    aim_obs.center_norm = NormalizedPoint{0.0f, 0.0f};
    aim_obs.bbox_px = BoundingBox{945.0f, 525.0f, 975.0f, 555.0f};
    aim_obs.effective_radius_px = 15.0f;
    aim_obs.confidence = 0.98f;
    aim_obs.covariance_px2 = Covariance2D{4.0f, 0.0f, 4.0f}; // Isotropic
    aim_obs.visibility = Visibility::visible;
    aim_obs.target_value = 1.0f;
    aim_obs.semantic_id = 1;

    // 2. Second-domain tall humanoid observation
    HumanoidTargetDef h_def{};
    h_def.source_id = 2;
    h_def.center_px = PixelPoint{1000.0f, 600.0f};
    h_def.width_px = 24.0f;
    h_def.height_px = 72.0f;
    h_def.confidence = 0.95f;
    h_def.velocity_px_per_s = PixelVelocity{50.0f, 0.0f};
    h_def.velocity_confidence = 0.8f;
    h_def.semantic_id = 2;

    auto sec_obs = SecondDomainObservationSource::make_humanoid_observation(h_def, 100, 1'000'000'000LL);

    // Verify coordinate space bounds: norm must be within [-1, 1]
    TEST_ASSERT(std::abs(aim_obs.center_norm.x) <= 1.0f && std::abs(aim_obs.center_norm.y) <= 1.0f);
    TEST_ASSERT(std::abs(sec_obs.center_norm.x) <= 1.0f && std::abs(sec_obs.center_norm.y) <= 1.0f);

    // Verify covariance: Second-domain must be anisotropic with positive variance
    TEST_ASSERT(sec_obs.covariance_px2.xx > 0.0f);
    TEST_ASSERT(sec_obs.covariance_px2.yy > 0.0f);
    TEST_ASSERT(sec_obs.covariance_px2.yy > sec_obs.covariance_px2.xx);

    // Verify effective radius: Narrow-axis dimension governs engagement
    TEST_ASSERT(sec_obs.effective_radius_px == 12.0f);

    TargetObservationBatch batch{};
    batch.schema_major = 1;
    batch.schema_minor = 0;
    batch.add_target(aim_obs);
    batch.add_target(sec_obs);

    TEST_ASSERT(ExternalObservationSource::validate_batch(batch));
    std::cout << "  -> Coordinate and covariance contracts passed." << std::endl;
}

void test_zero_policy_retraining_conformance() {
    std::cout << "[Test 2] Zero policy retraining conformance and multi-domain pipeline execution..." << std::endl;

    // Setup identical downstream stack
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

    // Domain 1: Aimlabs
    ScenarioAdapter aim_adapter{};
    const std::filesystem::path aim_path =
        std::filesystem::path(AIM_REPO_ROOT) / "configs" / "scenario" / "aimlabs.yaml";
    auto aim_res = ScenarioProfileLoader::load_from_file(aim_path);
    TEST_ASSERT(aim_res.success);
    TEST_ASSERT(aim_adapter.load_profile(aim_res.profile) == ScenarioProfileStatus::ok);

    CalibrationProfile aim_cal_prof{};
    aim_cal_prof.counts_per_pixel_x = aim_res.profile.calibration_seed.counts_per_pixel_x;
    aim_cal_prof.counts_per_pixel_y = aim_res.profile.calibration_seed.counts_per_pixel_y;
    CalibrationModel aim_cal_model{aim_cal_prof};

    ClosedLoopPipeline aim_pipeline{
        &tracker, &policy, &planner, &actuator, &safety, &aim_adapter, aim_cal_model
    };

    // Domain 2: Second Domain (Tactical Humanoid)
    ScenarioAdapter sec_adapter{};
    const std::filesystem::path sec_path =
        std::filesystem::path(AIM_REPO_ROOT) / "configs" / "scenario" / "second_domain.yaml";
    auto sec_res = ScenarioProfileLoader::load_from_file(sec_path);
    TEST_ASSERT(sec_res.success);
    TEST_ASSERT(sec_adapter.load_profile(sec_res.profile) == ScenarioProfileStatus::ok);

    CalibrationProfile sec_cal_prof{};
    sec_cal_prof.counts_per_pixel_x = sec_res.profile.calibration_seed.counts_per_pixel_x;
    sec_cal_prof.counts_per_pixel_y = sec_res.profile.calibration_seed.counts_per_pixel_y;
    CalibrationModel sec_cal_model{sec_cal_prof};

    SafetySupervisor sec_safety{};
    sec_safety.set_focus_callback([] { return true; });
    sec_safety.set_calibration_callback([] { return true; });
    ClosedLoopPipeline sec_pipeline{
        &tracker, &policy, &planner, &actuator, &sec_safety, &sec_adapter, sec_cal_model
    };

    // Execute 1,000 iterations on Aimlabs
    ObservableScenarioState aim_state{};
    aim_state.active_process_name = "Aimlab_tb.exe";
    aim_state.active_window_title = "Aimlabs";
    aim_state.foreground_confirmed = true;

    std::vector<double> aim_latencies_us;
    aim_latencies_us.reserve(1000);

    MonotonicNs t_ns = 1'000'000'000LL;
    for (std::uint64_t i = 1; i <= 1000; ++i) {
        t_ns += 6'944'444LL;
        TargetObservationBatch batch{};
        batch.header.sequence_id = i;
        batch.header.source_timestamp_ns = t_ns;
        batch.captured_at_ns = t_ns;

        TargetObservation obs{};
        obs.source_id = 1;
        obs.frame_id = i;
        obs.captured_at_ns = t_ns;
        obs.center_px = PixelPoint{965.0f, 542.0f};
        obs.effective_radius_px = 15.0f;
        obs.confidence = 0.95f;
        obs.covariance_px2 = Covariance2D{4.0f, 0.0f, 4.0f};
        batch.add_target(obs);

        aim_state.observed_at_ns = t_ns;

        const auto start = std::chrono::steady_clock::now();
        TEST_ASSERT(aim_pipeline.process_frame(batch, aim_state, t_ns + 500'000LL));
        const auto end = std::chrono::steady_clock::now();
        aim_latencies_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        TEST_ASSERT(aim_pipeline.tick_actuation(aim_state, t_ns + 1'500'000LL));
    }

    // Execute 1,000 iterations on Second Domain with tall humanoid targets
    ObservableScenarioState sec_state{};
    sec_state.active_process_name = "tactical_sim.exe";
    sec_state.active_window_title = "Tactical Domain";
    sec_state.foreground_confirmed = true;

    SecondDomainConfig sec_cfg{};
    SecondDomainObservationSource sec_source{sec_cfg};
    TEST_ASSERT(sec_source.start());

    std::vector<HumanoidTargetDef> tgts{
        {1, PixelPoint{965.0f, 542.0f}, 24.0f, 72.0f, 0.95f, PixelVelocity{0.0f, 0.0f}, 0.0f, 2}
    };
    sec_source.set_targets(tgts);

    std::vector<double> sec_latencies_us;
    sec_latencies_us.reserve(1000);

    for (std::uint64_t i = 1001; i <= 2000; ++i) {
        t_ns += 6'944'444LL;
        sec_source.advance_simulation(6'944'444LL);

        TargetObservationBatch batch{};
        TEST_ASSERT(sec_source.try_read_latest(batch));
        batch.header.sequence_id = i;
        batch.header.source_timestamp_ns = t_ns;
        batch.captured_at_ns = t_ns;

        sec_state.observed_at_ns = t_ns;

        const auto start = std::chrono::steady_clock::now();
        TEST_ASSERT(sec_pipeline.process_frame(batch, sec_state, t_ns + 500'000LL));
        const auto end = std::chrono::steady_clock::now();
        sec_latencies_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        TEST_ASSERT(sec_pipeline.tick_actuation(sec_state, t_ns + 1'500'000LL));
    }

    sec_source.stop();

    auto aim_stats = compute_latency_stats(aim_latencies_us);
    auto sec_stats = compute_latency_stats(sec_latencies_us);

    std::cout << "  -> Aimlabs Benchmark (1,000 frames):" << std::endl;
    std::cout << "     p50: " << aim_stats.p50_us << " us, p95: " << aim_stats.p95_us
              << " us, p99: " << aim_stats.p99_us << " us, max: " << aim_stats.max_us << " us" << std::endl;

    std::cout << "  -> Second Domain Benchmark (1,000 frames):" << std::endl;
    std::cout << "     p50: " << sec_stats.p50_us << " us, p95: " << sec_stats.p95_us
              << " us, p99: " << sec_stats.p99_us << " us, max: " << sec_stats.max_us << " us" << std::endl;

    // Both must be orders of magnitude below the 6.0 ms (6000 us) budget
    TEST_ASSERT(aim_stats.p99_us < 100.0); // Typically < 5 us
    TEST_ASSERT(sec_stats.p99_us < 100.0);

    // Verify ShotGate authorized shots on both domains
    TEST_ASSERT(aim_pipeline.stats().total_shots_authorized > 0);
    TEST_ASSERT(sec_pipeline.stats().total_shots_authorized > 0);

    std::cout << "  -> Zero policy retraining conformance passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M8-03 Portability Conformance Suite          " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_coordinate_and_covariance_contracts();
    test_zero_policy_retraining_conformance();

    std::cout << "================================================================" << std::endl;
    std::cout << " All Milestone M8-03 Portability Conformance Tests Passed!      " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
