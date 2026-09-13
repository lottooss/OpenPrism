// tests/cpp/test_policy_trajectory_benchmark.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/policy/utility_policy.hpp"
#include "aim/trajectory/trajectory_planner.hpp"

using namespace aim;
using namespace aim::policy;
using namespace aim::trajectory;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

struct PolicyBenchmarkMetrics {
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double max_us{0.0};
    double mean_us{0.0};
    std::uint32_t utility_switches{0};
    std::uint32_t nearest_switches{0};
    std::uint32_t total_shots_authorized{0};
};

PolicyBenchmarkMetrics run_policy_trajectory_benchmark() {
    UtilityAimPolicy utility_policy{};
    NearestTargetPolicy nearest_policy{};
    HybridTrajectoryPlanner planner{};
    PlannerLimits limits{};

    ControlState crosshair_state{};
    crosshair_state.current_crosshair_px = PixelPoint{960.0f, 540.0f};

    const std::size_t kTotalFrames = 1000;
    std::vector<double> latencies_us;
    latencies_us.reserve(kTotalFrames);

    std::uint32_t utility_switch_count = 0;
    std::uint32_t nearest_switch_count = 0;
    std::uint32_t shots_authorized = 0;

    TrackId last_utility_target = 0;
    TrackId last_nearest_target = 0;

    const float dt = 1.0f / 144.0f; // 144 Hz frame rate
    const MonotonicNs frame_dt_ns = static_cast<MonotonicNs>(dt * 1e9f);
    MonotonicNs sim_time_ns = 1'000'000'000LL;

    // Simulation: 3 targets maneuvering across the screen
    // Target 1: Sweeps right (800 -> 1120)
    // Target 2: Sweeps left (1120 -> 800) - crosses Target 1 around frame 500!
    // Target 3: Circling around (960, 400)
    float t1_x = 800.0f, t1_y = 540.0f, t1_vx = 45.0f;
    float t2_x = 1120.0f, t2_y = 545.0f, t2_vx = -45.0f;
    float t3_angle = 0.0f;

    for (std::size_t frame = 0; frame < kTotalFrames; ++frame) {
        sim_time_ns += frame_dt_ns;

        // Kinematic updates
        t1_x += t1_vx * dt;
        t2_x += t2_vx * dt;
        t3_angle += 2.0f * dt;
        const float t3_x = 960.0f + 150.0f * std::cos(t3_angle);
        const float t3_y = 400.0f + 80.0f * std::sin(t3_angle);

        bus::TrackedTargetBatch batch{};
        batch.header.sequence_id = frame;
        batch.timestamp_ns = sim_time_ns;

        // Target 1
        bus::TrackedTarget tgt1{};
        tgt1.track_id = 1;
        tgt1.state = bus::TrackState::confirmed;
        tgt1.filtered_center_px = PixelPoint{t1_x, t1_y};
        tgt1.predicted_center_px = PixelPoint{t1_x + t1_vx * 0.012f, t1_y};
        tgt1.confidence = 0.95f;
        tgt1.target_value = 1.0f;
        tgt1.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
        batch.add_track(tgt1);

        // Target 2
        bus::TrackedTarget tgt2{};
        tgt2.track_id = 2;
        tgt2.state = bus::TrackState::confirmed;
        tgt2.filtered_center_px = PixelPoint{t2_x, t2_y};
        tgt2.predicted_center_px = PixelPoint{t2_x + t2_vx * 0.012f, t2_y};
        tgt2.confidence = 0.92f;
        tgt2.target_value = 1.0f;
        tgt2.covariance_px2 = Covariance2D{1.2f, 0.0f, 1.2f};
        batch.add_track(tgt2);

        // Target 3
        bus::TrackedTarget tgt3{};
        tgt3.track_id = 3;
        tgt3.state = bus::TrackState::confirmed;
        tgt3.filtered_center_px = PixelPoint{t3_x, t3_y};
        tgt3.predicted_center_px = PixelPoint{t3_x, t3_y};
        tgt3.confidence = 0.90f;
        tgt3.target_value = 1.2f; // slightly higher reward
        tgt3.covariance_px2 = Covariance2D{1.5f, 0.0f, 1.5f};
        batch.add_track(tgt3);

        PolicyInput input{};
        input.correlation_id.sequence_id = frame;
        input.decision_time_ns = sim_time_ns;
        input.crosshair = CrosshairState{crosshair_state.current_crosshair_px, NormalizedPoint{0.0f, 0.0f}, false};
        input.tracks = &batch;

        // Run Nearest Baseline for comparative switch tracking
        bus::AimIntent nearest_intent{};
        if (nearest_policy.choose(input, nearest_intent)) {
            if (last_nearest_target != 0 && nearest_intent.target_track_id != last_nearest_target) {
                nearest_switch_count++;
            }
            last_nearest_target = nearest_intent.target_track_id;
        }

        // Benchmark Utility Policy + Trajectory Planner hot path
        const auto t_start = std::chrono::steady_clock::now();

        bus::AimIntent utility_intent{};
        const bool policy_ok = utility_policy.choose(input, utility_intent);

        TrajectoryPlan plan{};
        if (policy_ok) {
            planner.plan(utility_intent, crosshair_state, limits, plan);
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        latencies_us.push_back(duration_us);

        if (policy_ok) {
            if (last_utility_target != 0 && utility_intent.target_track_id != last_utility_target) {
                utility_switch_count++;
            }
            last_utility_target = utility_intent.target_track_id;

            if (utility_intent.authorize_fire) {
                shots_authorized++;
            }

            // Simulate closed-loop crosshair motion towards target
            if (plan.point_count > 0) {
                // Execute first microstep displacement
                crosshair_state.current_crosshair_px.x += static_cast<float>(plan.points[0].step_delta_x_counts);
                crosshair_state.current_crosshair_px.y += static_cast<float>(plan.points[0].step_delta_y_counts);
            }
        }
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    const std::size_t n = latencies_us.size();

    PolicyBenchmarkMetrics metrics{};
    metrics.p50_us = latencies_us[(n * 50) / 100];
    metrics.p95_us = latencies_us[(n * 95) / 100];
    metrics.p99_us = latencies_us[(n * 99) / 100];
    metrics.max_us = latencies_us.back();
    metrics.mean_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / static_cast<double>(n);
    metrics.utility_switches = utility_switch_count;
    metrics.nearest_switches = nearest_switch_count;
    metrics.total_shots_authorized = shots_authorized;

    return metrics;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M5 Policy & Trajectory Benchmark             " << std::endl;
    std::cout << "================================================================" << std::endl;

    const PolicyBenchmarkMetrics m = run_policy_trajectory_benchmark();

    std::cout << "  Execution Latency Benchmark (1000 frames, Policy + Planner):" << std::endl;
    std::cout << "    p50:  " << m.p50_us << " us (" << (m.p50_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    p95:  " << m.p95_us << " us (" << (m.p95_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    p99:  " << m.p99_us << " us (" << (m.p99_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    Max:  " << m.max_us << " us (" << (m.max_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    Mean: " << m.mean_us << " us (" << (m.mean_us / 1000.0) << " ms)" << std::endl;
    std::cout << std::endl;

    std::cout << "  Policy Quality & Stability Metrics:" << std::endl;
    std::cout << "    Utility Policy Target Switches: " << m.utility_switches << std::endl;
    std::cout << "    Nearest Baseline Target Switches: " << m.nearest_switches << std::endl;
    std::cout << "    Shots Authorized: " << m.total_shots_authorized << std::endl;

    // Acceptance criteria assertions
    TEST_ASSERT(m.p99_us <= 300.0); // p99 <= 0.30 ms
    TEST_ASSERT(m.utility_switches <= m.nearest_switches); // Hysteresis prevents thrashing
    TEST_ASSERT(m.total_shots_authorized > 100); // Closed-loop successfully acquires targets

    std::cout << "================================================================" << std::endl;
    std::cout << " All Milestone M5 Policy & Trajectory Acceptance Passed!        " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
