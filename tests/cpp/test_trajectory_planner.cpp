// tests/cpp/test_trajectory_planner.cpp
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/types.hpp"
#include "aim/trajectory/trajectory_planner.hpp"

using namespace aim;
using namespace aim::trajectory;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

void test_direct_feed_forward_small_error() {
    std::cout << "[Test 1] Direct feed-forward small error planning..." << std::endl;

    HybridTrajectoryPlanner planner{};
    PlannerLimits limits{};
    limits.small_error_threshold_px = 6.0f;

    ControlState state{};
    state.current_crosshair_px = PixelPoint{960.0f, 540.0f};

    bus::AimIntent intent{};
    intent.target_track_id = 1;
    intent.target_aim_px = PixelPoint{964.0f, 542.0f}; // 4px X, 2px Y (dist ~ 4.47px <= 6px)
    intent.command_deadline_ns = 1'000'000'000LL;

    TrajectoryPlan plan{};
    TEST_ASSERT(planner.plan(intent, state, limits, plan));
    TEST_ASSERT(plan.point_count == 1);
    TEST_ASSERT(plan.points[0].step_delta_x_counts == 4);
    TEST_ASSERT(plan.points[0].step_delta_y_counts == 2);
    TEST_ASSERT(std::abs(plan.points[0].position_px.x - 964.0f) < 1e-3f);
    TEST_ASSERT(std::abs(plan.points[0].position_px.y - 542.0f) < 1e-3f);

    std::cout << "  -> Direct feed-forward passed." << std::endl;
}

void test_minimum_jerk_large_motion_and_bounds() {
    std::cout << "[Test 2] Minimum-jerk large motion and kinematic bounds..." << std::endl;

    HybridTrajectoryPlanner planner{};
    PlannerLimits limits{};

    ControlState state{};
    state.current_crosshair_px = PixelPoint{500.0f, 500.0f};

    // Subtest A: 50px complete flick (T ~ 24ms <= 32ms capacity)
    bus::AimIntent intent_50px{};
    intent_50px.target_track_id = 1;
    intent_50px.target_aim_px = PixelPoint{550.0f, 500.0f};
    intent_50px.command_deadline_ns = 1'000'000'000LL;

    TrajectoryPlan plan_50px{};
    TEST_ASSERT(planner.plan(intent_50px, state, limits, plan_50px));
    TEST_ASSERT(plan_50px.point_count >= 15);
    TEST_ASSERT(plan_50px.point_count <= kMaxTrajectoryPoints);

    std::int32_t total_counts_x = 0;
    std::int32_t total_counts_y = 0;
    for (std::uint32_t i = 0; i < plan_50px.point_count; ++i) {
        total_counts_x += plan_50px.points[i].step_delta_x_counts;
        total_counts_y += plan_50px.points[i].step_delta_y_counts;

        const float vx = plan_50px.points[i].velocity_px_s.x_per_s;
        const float vy = plan_50px.points[i].velocity_px_s.y_per_s;
        const float v = std::hypot(vx, vy);
        TEST_ASSERT(v <= limits.max_velocity_px_s * 1.05f);
    }
    TEST_ASSERT(total_counts_x == 50);
    TEST_ASSERT(total_counts_y == 0);
    TEST_ASSERT(std::abs(plan_50px.points[plan_50px.point_count - 1].position_px.x - 550.0f) < 1e-2f);

    // Subtest B: 200px long flick (T ~ 75ms > 32ms rolling buffer)
    bus::AimIntent intent_200px{};
    intent_200px.target_track_id = 1;
    intent_200px.target_aim_px = PixelPoint{700.0f, 500.0f};
    intent_200px.command_deadline_ns = 1'000'000'000LL;

    TrajectoryPlan plan_200px{};
    TEST_ASSERT(planner.plan(intent_200px, state, limits, plan_200px));
    TEST_ASSERT(plan_200px.point_count == kMaxTrajectoryPoints); // Plans max 32 microsteps

    for (std::uint32_t i = 0; i < plan_200px.point_count; ++i) {
        const float vx = plan_200px.points[i].velocity_px_s.x_per_s;
        const float vy = plan_200px.points[i].velocity_px_s.y_per_s;
        const float v = std::hypot(vx, vy);
        TEST_ASSERT(v <= limits.max_velocity_px_s * 1.05f); // Strictly bounded
    }

    std::cout << "  -> Minimum-jerk motion and bounds passed." << std::endl;
}

void test_plan_replacement_and_cancellation() {
    std::cout << "[Test 3] Plan replacement, sequence IDs, and cancellation..." << std::endl;

    HybridTrajectoryPlanner planner{};
    PlannerLimits limits{};

    ControlState state{};
    state.current_crosshair_px = PixelPoint{100.0f, 100.0f};

    bus::AimIntent intent1{};
    intent1.target_track_id = 1;
    intent1.target_aim_px = PixelPoint{200.0f, 100.0f};

    TrajectoryPlan plan1{};
    TEST_ASSERT(planner.plan(intent1, state, limits, plan1));
    TEST_ASSERT(plan1.plan_id == 1);

    // Plan replacement
    bus::AimIntent intent2{};
    intent2.target_track_id = 1;
    intent2.target_aim_px = PixelPoint{300.0f, 100.0f};

    TrajectoryPlan plan2{};
    TEST_ASSERT(planner.plan(intent2, state, limits, plan2));
    TEST_ASSERT(plan2.plan_id == 2);

    // Cancellation
    planner.cancel();
    planner.reset();
    TEST_ASSERT(planner.plan(intent1, state, limits, plan1));
    TEST_ASSERT(plan1.plan_id == 1); // Reset back to 1

    std::cout << "  -> Plan replacement and cancellation passed." << std::endl;
}

void test_zero_displacement() {
    std::cout << "[Test 4] Zero displacement edge case..." << std::endl;

    HybridTrajectoryPlanner planner{};
    PlannerLimits limits{};

    ControlState state{};
    state.current_crosshair_px = PixelPoint{960.0f, 540.0f};

    bus::AimIntent intent{};
    intent.target_track_id = 1;
    intent.target_aim_px = PixelPoint{960.0f, 540.0f}; // exactly at target

    TrajectoryPlan plan{};
    TEST_ASSERT(planner.plan(intent, state, limits, plan));
    TEST_ASSERT(plan.point_count == 0); // 0 points needed

    std::cout << "  -> Zero displacement passed." << std::endl;
}

void test_refreshed_profile_kinematic_bounds() {
    for (const float distance : {7.0f, 15.0f, 50.0f, 100.0f, 200.0f, 300.0f}) {
        HybridTrajectoryPlanner planner;
        ControlState state;
        state.current_crosshair_px = {0.0f, 0.0f};
        PlannerLimits limits;
        bus::AimIntent intent{};
        intent.target_track_id = 1;
        intent.target_aim_px = {distance * 0.8f, distance * 0.6f};
        PixelVelocity previous_velocity{};
        PixelVelocity previous_acceleration{};
        for (unsigned tick = 0; tick < 160; ++tick) {
            intent.command_deadline_ns = 1'010'000'000LL + static_cast<MonotonicNs>(tick) * 1'000'000LL;
            TrajectoryPlan plan;
            TEST_ASSERT(planner.plan(intent, state, limits, plan));
            if (plan.point_count == 0) break;
            const auto& point = plan.points[0];
            const PixelVelocity velocity = point.velocity_px_s;
            const PixelVelocity acceleration{
                (velocity.x_per_s - previous_velocity.x_per_s) * 1000.0f,
                (velocity.y_per_s - previous_velocity.y_per_s) * 1000.0f};
            const PixelVelocity jerk{
                (acceleration.x_per_s - previous_acceleration.x_per_s) * 1000.0f,
                (acceleration.y_per_s - previous_acceleration.y_per_s) * 1000.0f};
            TEST_ASSERT(std::hypot(velocity.x_per_s, velocity.y_per_s) <= limits.max_velocity_px_s);
            TEST_ASSERT(std::hypot(acceleration.x_per_s, acceleration.y_per_s) <= limits.max_accel_px_s2);
            TEST_ASSERT(std::hypot(jerk.x_per_s, jerk.y_per_s) <= limits.max_jerk_px_s3);
            previous_velocity = velocity;
            previous_acceleration = acceleration;
            state.current_crosshair_px = point.position_px;
        }
        TEST_ASSERT(std::hypot(intent.target_aim_px.x - state.current_crosshair_px.x,
                               intent.target_aim_px.y - state.current_crosshair_px.y) < 0.1f);
    }
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M5-02 Trajectory Planner Unit Tests          " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_direct_feed_forward_small_error();
    test_minimum_jerk_large_motion_and_bounds();
    test_plan_replacement_and_cancellation();
    test_zero_displacement();
    test_refreshed_profile_kinematic_bounds();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M5-02 Trajectory Planner Tests Passed Successfully!        " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
