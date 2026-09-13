// tests/cpp/test_end_to_end_benchmark.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

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

struct LatencyDistribution {
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double max_us{0.0};
    double mean_us{0.0};
};

LatencyDistribution compute_percentiles(std::vector<double>& samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    LatencyDistribution dist{};
    dist.p50_us = samples[static_cast<std::size_t>(n * 0.50)];
    dist.p95_us = samples[static_cast<std::size_t>(n * 0.95)];
    dist.p99_us = samples[static_cast<std::size_t>(n * 0.99)];
    dist.max_us = samples.back();
    dist.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(n);
    return dist;
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M7-03 End-to-End Latency & Score Benchmark   " << std::endl;
    std::cout << "================================================================" << std::endl;

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
    TEST_ASSERT(scenario.load_profile(make_generic_profile()) == ScenarioProfileStatus::ok);

    CalibrationProfile cal_prof{};
    cal_prof.counts_per_pixel_x = 1.0f;
    cal_prof.counts_per_pixel_y = 1.0f;
    CalibrationModel cal_model{cal_prof};

    ClosedLoopPipeline pipeline{
        &tracker, &policy, &planner, &actuator, &safety, &scenario, cal_model
    };

    const std::size_t kBenchmarkIterations = 5000;
    std::vector<double> tracking_latencies;
    std::vector<double> policy_latencies;
    std::vector<double> planner_latencies;
    std::vector<double> total_internal_latencies;

    tracking_latencies.reserve(kBenchmarkIterations);
    policy_latencies.reserve(kBenchmarkIterations);
    planner_latencies.reserve(kBenchmarkIterations);
    total_internal_latencies.reserve(kBenchmarkIterations);

    MonotonicNs sim_time_ns = 1'000'000'000LL;
    ObservableScenarioState state{};
    state.foreground_confirmed = true;

    // Simulate 3 dynamic targets moving across the screen (Aimlabs grid + tracking)
    struct SimTarget {
        TrackId id;
        float x;
        float y;
        float vx;
        float vy;
        float radius;
    };

    std::array<SimTarget, 3> targets{{
        {101, 950.0f, 530.0f, 15.0f, -10.0f, 18.0f},
        {102, 1100.0f, 400.0f, -20.0f, 15.0f, 15.0f},
        {103, 800.0f, 650.0f, 10.0f, 25.0f, 20.0f}
    }};

    std::uint64_t hits = 0;
    std::uint64_t attempts = 0;

    for (std::size_t i = 0; i < kBenchmarkIterations; ++i) {
        sim_time_ns += 6'944'444LL; // 144 Hz frame interval (~6.94 ms)

        // Update target positions with bounce
        for (auto& t : targets) {
            t.x += t.vx * 0.00694f;
            t.y += t.vy * 0.00694f;
            if (t.x < 300.0f || t.x > 1620.0f) t.vx = -t.vx;
            if (t.y < 200.0f || t.y > 880.0f) t.vy = -t.vy;
        }

        bus::TargetObservationBatch batch{};
        batch.header.sequence_id = i + 1;
        batch.header.source_timestamp_ns = sim_time_ns;
        batch.captured_at_ns = sim_time_ns;

        for (const auto& t : targets) {
            bus::TargetObservation obs{};
            obs.source_id = t.id;
            obs.center_px = PixelPoint{t.x, t.y};
            obs.effective_radius_px = t.radius;
            obs.confidence = 0.98f;
            obs.covariance_px2 = Covariance2D{2.0f, 0.0f, 2.0f}; // sigma ~= 1.4 px
            batch.add_target(obs);
        }

        // Execute timed frame through pipeline
        const auto tick_start = std::chrono::steady_clock::now();
        const bool success = pipeline.process_frame(batch, state, sim_time_ns + 500'000LL); // 0.5 ms capture delay
        const auto tick_end = std::chrono::steady_clock::now();

        TEST_ASSERT(success);

        const auto loop_dur_us = std::chrono::duration<double, std::micro>(tick_end - tick_start).count();
        total_internal_latencies.push_back(loop_dur_us);

        const auto stats = pipeline.stats();
        tracking_latencies.push_back(static_cast<double>(stats.last_tracking_latency_ns) / 1000.0);
        policy_latencies.push_back(static_cast<double>(stats.last_policy_latency_ns) / 1000.0);
        planner_latencies.push_back(static_cast<double>(stats.last_planner_latency_ns) / 1000.0);

        if (pipeline.last_shot_result().authorize_fire) {
            attempts++;
            // Check if crosshair is actually inside target
            const auto& active_target = targets[0]; // primary target
            const float dx = active_target.x - 960.0f;
            const float dy = active_target.y - 540.0f;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dist <= active_target.radius * 1.1f) {
                hits++;
            }
        }
    }

    const auto total_dist = compute_percentiles(total_internal_latencies);
    const auto track_dist = compute_percentiles(tracking_latencies);
    const auto policy_dist = compute_percentiles(policy_latencies);
    const auto plan_dist = compute_percentiles(planner_latencies);

    std::cout << "  Benchmark Execution Summary (" << kBenchmarkIterations << " frames):" << std::endl;
    std::cout << "    Frames Processed:        " << pipeline.stats().total_frames_processed << std::endl;
    std::cout << "    Commands Dispatched:     " << pipeline.stats().total_commands_dispatched << std::endl;
    std::cout << "    Shots Authorized:        " << pipeline.stats().total_shots_authorized << std::endl;
    std::cout << "    Stale Frames Dropped:    " << pipeline.stats().total_stale_dropped << std::endl;
    std::cout << "    Safety Rejections:       " << pipeline.stats().total_safety_rejections << std::endl;
    std::cout << std::endl;

    std::cout << "  Internal Stage Latencies (p50 / p95 / p99):" << std::endl;
    std::cout << "    Tracking & Prediction:   " << track_dist.p50_us << " us / " << track_dist.p95_us << " us / " << track_dist.p99_us << " us" << std::endl;
    std::cout << "    Aim Policy:              " << policy_dist.p50_us << " us / " << policy_dist.p95_us << " us / " << policy_dist.p99_us << " us" << std::endl;
    std::cout << "    Trajectory Planner:      " << plan_dist.p50_us << " us / " << plan_dist.p95_us << " us / " << plan_dist.p99_us << " us" << std::endl;
    std::cout << "    Total Internal Loop:     " << total_dist.p50_us << " us / " << total_dist.p95_us << " us / " << total_dist.p99_us << " us (" << (total_dist.p99_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    Max Internal Latency:    " << total_dist.max_us << " us (" << (total_dist.max_us / 1000.0) << " ms)" << std::endl;

    // Gate Criteria verification:
    // Internal loop p99 <= 6.0 ms (6000 us)
    TEST_ASSERT(total_dist.p99_us <= 6000.0);
    // Hard cutoff < 10.0 ms (10000 us)
    TEST_ASSERT(total_dist.max_us <= 10000.0);
    // Stale dropped should be 0 in clean run
    TEST_ASSERT(pipeline.stats().total_stale_dropped == 0);
    // Safety rejections should be 0 in nominal run
    TEST_ASSERT(pipeline.stats().total_safety_rejections == 0);

    std::cout << "================================================================" << std::endl;
    std::cout << " All Milestone M7-03 Benchmark Acceptance Passed!               " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
