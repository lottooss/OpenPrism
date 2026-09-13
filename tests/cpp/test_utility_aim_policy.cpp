// tests/cpp/test_utility_aim_policy.cpp
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/types.hpp"
#include "aim/policy/utility_policy.hpp"

using namespace aim;
using namespace aim::policy;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

bus::TrackedTarget make_target(
    TrackId id,
    float x,
    float y,
    float vx = 0.0f,
    float vy = 0.0f,
    float value = 1.0f,
    float conf = 0.95f,
    bus::TrackState state = bus::TrackState::confirmed) {
    bus::TrackedTarget t{};
    t.track_id = id;
    t.state = state;
    t.filtered_center_px = PixelPoint{x, y};
    t.filtered_velocity_px_per_s = PixelVelocity{vx, vy};
    t.predicted_center_px = PixelPoint{x + vx * 0.010f, y + vy * 0.010f}; // 10ms lead
    t.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
    t.confidence = conf;
    t.target_value = value;
    t.semantic_id = 1;
    return t;
}

void test_utility_target_selection() {
    std::cout << "[Test 1] Utility target selection ranking..." << std::endl;

    UtilityAimPolicy policy{};
    PolicyInput input{};
    input.crosshair.center_px = PixelPoint{960.0f, 540.0f};
    input.decision_time_ns = 1'000'000'000LL;

    bus::TrackedTargetBatch batch{};
    // Target 1: Close (980, 540) -> 20px, value 1.0
    batch.add_track(make_target(1, 980.0f, 540.0f, 0.0f, 0.0f, 1.0f, 0.95f));
    // Target 2: Farther (1200, 540) -> 240px, value 1.0
    batch.add_track(make_target(2, 1200.0f, 540.0f, 0.0f, 0.0f, 1.0f, 0.95f));

    input.tracks = &batch;

    bus::AimIntent intent{};
    TEST_ASSERT(policy.choose(input, intent));
    TEST_ASSERT(intent.target_track_id == 1);
    TEST_ASSERT(std::abs(intent.target_aim_px.x - 980.0f) < 1.0f);
    TEST_ASSERT(!intent.authorize_fire); // 20px > 8px fire threshold

    std::cout << "  -> Utility ranking passed." << std::endl;
}

void test_switch_hysteresis_anti_thrashing() {
    std::cout << "[Test 2] Switch hysteresis and anti-thrashing..." << std::endl;

    UtilityAimPolicy policy{};
    PolicyInput input{};
    input.crosshair.center_px = PixelPoint{960.0f, 540.0f};
    input.decision_time_ns = 1'000'000'000LL;

    bus::TrackedTargetBatch batch{};
    // Target 1: at 930 (30px left)
    batch.add_track(make_target(1, 930.0f, 540.0f, 0.0f, 0.0f, 1.0f, 0.95f));
    // Target 2: at 995 (35px right)
    batch.add_track(make_target(2, 995.0f, 540.0f, 0.0f, 0.0f, 1.0f, 0.95f));

    input.tracks = &batch;

    // Step 1: Engage Target 1 (since it's closer: 30px vs 35px)
    bus::AimIntent intent1{};
    TEST_ASSERT(policy.choose(input, intent1));
    TEST_ASSERT(intent1.target_track_id == 1);
    TEST_ASSERT(policy.current_target_id() == 1);

    // Step 2: Now crosshair moves slightly to 965.
    // Distance to Target 1 is 35px. Distance to Target 2 is 30px.
    // Without hysteresis, greedy policy would immediately thrash and switch to Target 2!
    // With hysteresis bonus = 1.5, Target 1 has higher total utility.
    input.crosshair.center_px = PixelPoint{965.0f, 540.0f};
    bus::AimIntent intent2{};
    TEST_ASSERT(policy.choose(input, intent2));
    TEST_ASSERT(intent2.target_track_id == 1); // Remains locked on Target 1 due to hysteresis!

    std::cout << "  -> Switch hysteresis anti-thrashing passed." << std::endl;
}

void test_fire_authorization() {
    std::cout << "[Test 3] Fire authorization gating..." << std::endl;

    UtilityAimPolicy policy{};
    PolicyInput input{};
    input.crosshair.center_px = PixelPoint{960.0f, 540.0f};
    input.decision_time_ns = 1'000'000'000LL;

    bus::TrackedTargetBatch batch{};
    // Target directly on crosshair (3px error <= 8px threshold)
    batch.add_track(make_target(1, 963.0f, 540.0f, 0.0f, 0.0f, 1.0f, 0.95f, bus::TrackState::confirmed));
    input.tracks = &batch;

    bus::AimIntent intent{};
    TEST_ASSERT(policy.choose(input, intent));
    TEST_ASSERT(intent.target_track_id == 1);
    TEST_ASSERT(intent.authorize_fire); // Authorized!

    // If target is unconfirmed (tentative), fire is not authorized
    batch.clear();
    batch.add_track(make_target(2, 963.0f, 540.0f, 0.0f, 0.0f, 1.0f, 0.95f, bus::TrackState::tentative));
    bus::AimIntent intent_tentative{};
    TEST_ASSERT(policy.choose(input, intent_tentative));
    TEST_ASSERT(!intent_tentative.authorize_fire); // Not authorized for tentative

    std::cout << "  -> Fire authorization passed." << std::endl;
}

void test_lead_offset_computation() {
    std::cout << "[Test 4] Predictive lead offset computation..." << std::endl;

    UtilityAimPolicy policy{};
    PolicyInput input{};
    input.crosshair.center_px = PixelPoint{960.0f, 540.0f};
    input.decision_time_ns = 1'000'000'000LL;

    bus::TrackedTargetBatch batch{};
    // Target moving at 500 px/s in X
    batch.add_track(make_target(1, 800.0f, 540.0f, 500.0f, 0.0f, 1.0f, 0.95f));
    input.tracks = &batch;

    bus::AimIntent intent{};
    TEST_ASSERT(policy.choose(input, intent));
    // 500 px/s * 0.010s = 5.0 px lead
    TEST_ASSERT(std::abs(intent.lead_offset_px.x - 5.0f) < 1e-3f);
    TEST_ASSERT(std::abs(intent.lead_offset_px.y - 0.0f) < 1e-3f);

    std::cout << "  -> Predictive lead offset passed." << std::endl;
}

void test_baseline_policies() {
    std::cout << "[Test 5] Baseline policies (Nearest & Spatial Sweep)..." << std::endl;

    NearestTargetPolicy nearest_policy{};
    SpatialSweepPolicy sweep_policy{};

    PolicyInput input{};
    input.crosshair.center_px = PixelPoint{500.0f, 500.0f};

    bus::TrackedTargetBatch batch{};
    batch.add_track(make_target(1, 100.0f, 500.0f)); // Leftmost (400px dist)
    batch.add_track(make_target(2, 480.0f, 500.0f)); // Nearest (20px dist)
    input.tracks = &batch;

    bus::AimIntent nearest_intent{};
    TEST_ASSERT(nearest_policy.choose(input, nearest_intent));
    TEST_ASSERT(nearest_intent.target_track_id == 2); // Nearest picked 2

    bus::AimIntent sweep_intent{};
    TEST_ASSERT(sweep_policy.choose(input, sweep_intent));
    TEST_ASSERT(sweep_intent.target_track_id == 1); // Sweep picked 1 (leftmost)

    std::cout << "  -> Baseline policies passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M5-01 Aim Policy Unit Tests                  " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_utility_target_selection();
    test_switch_hysteresis_anti_thrashing();
    test_fire_authorization();
    test_lead_offset_computation();
    test_baseline_policies();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M5-01 Aim Policy Tests Passed Successfully!                " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
