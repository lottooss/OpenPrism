// tests/cpp/test_track_lifecycle.cpp
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/types.hpp"
#include "aim/tracking/track_lifecycle.hpp"

using namespace aim;
using namespace aim::tracking;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

bus::TargetObservation make_obs(float x, float y, float r = 15.0f, float conf = 0.95f, std::uint32_t sem_id = 1) {
    bus::TargetObservation obs{};
    obs.center_px = PixelPoint{x, y};
    obs.effective_radius_px = r;
    obs.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
    obs.confidence = conf;
    obs.semantic_id = sem_id;
    return obs;
}

void test_immediate_high_confidence_confirmation() {
    std::cout << "[Test 1] Immediate high confidence track confirmation..." << std::endl;

    MultiTargetTracker tracker{};
    bus::TargetObservationBatch batch{};
    batch.add_target(make_obs(100.0f, 100.0f, 15.0f, 0.90f));

    bus::TrackedTargetBatch out{};
    tracker.process(batch, 1000000000LL, out);

    TEST_ASSERT(out.track_count == 1);
    TEST_ASSERT(out.tracks[0].state == TrackState::confirmed);
    TEST_ASSERT(out.tracks[0].total_visible_frames == 1);
    TEST_ASSERT(out.tracks[0].total_missed_frames == 0);
    TEST_ASSERT(tracker.active_track_count() == 1);
    std::cout << "  -> Immediate confirmation passed." << std::endl;
}

void test_tentative_two_frame_confirmation() {
    std::cout << "[Test 2] Tentative two-frame confirmation..." << std::endl;

    MultiTargetTracker tracker{};

    // Frame 1: Low confidence observation (0.60 < 0.85) -> tentative
    bus::TargetObservationBatch batch1{};
    batch1.add_target(make_obs(100.0f, 100.0f, 15.0f, 0.60f));

    bus::TrackedTargetBatch out1{};
    tracker.process(batch1, 1000000000LL, out1);

    TEST_ASSERT(out1.track_count == 1);
    TEST_ASSERT(out1.tracks[0].state == TrackState::tentative);
    const TrackId id = out1.tracks[0].track_id;

    // Frame 2: Second matching observation -> confirmed
    bus::TargetObservationBatch batch2{};
    batch2.add_target(make_obs(102.0f, 100.0f, 15.0f, 0.65f));

    bus::TrackedTargetBatch out2{};
    tracker.process(batch2, 1006944444LL, out2); // ~144Hz dt

    TEST_ASSERT(out2.track_count == 1);
    TEST_ASSERT(out2.tracks[0].track_id == id);
    TEST_ASSERT(out2.tracks[0].state == TrackState::confirmed);
    TEST_ASSERT(out2.tracks[0].total_visible_frames == 2);
    std::cout << "  -> Tentative confirmation passed." << std::endl;
}

void test_tentative_ghost_pruning() {
    std::cout << "[Test 3] Tentative ghost target pruning on single miss..." << std::endl;

    MultiTargetTracker tracker{};

    // Frame 1: Low confidence tentative observation
    bus::TargetObservationBatch batch1{};
    batch1.add_target(make_obs(100.0f, 100.0f, 15.0f, 0.50f));

    bus::TrackedTargetBatch out1{};
    tracker.process(batch1, 1000000000LL, out1);
    TEST_ASSERT(out1.track_count == 1);
    TEST_ASSERT(out1.tracks[0].state == TrackState::tentative);

    // Frame 2: Empty observation (miss) -> pruned immediately
    bus::TargetObservationBatch batch2{};
    bus::TrackedTargetBatch out2{};
    tracker.process(batch2, 1006944444LL, out2);

    TEST_ASSERT(out2.track_count == 0);
    TEST_ASSERT(tracker.active_track_count() == 0);
    std::cout << "  -> Ghost pruning passed." << std::endl;
}

void test_occlusion_retention_and_recovery() {
    std::cout << "[Test 4] Three-frame occlusion retention and recovery..." << std::endl;

    MultiTargetTracker tracker{};

    // Frame 1: Confirmed target
    bus::TargetObservationBatch batch1{};
    batch1.add_target(make_obs(100.0f, 100.0f, 15.0f, 0.95f));
    bus::TrackedTargetBatch out1{};
    tracker.process(batch1, 1000000000LL, out1);
    TEST_ASSERT(out1.track_count == 1);
    const TrackId id = out1.tracks[0].track_id;

    // Frames 2, 3, 4: Occluded (empty observations)
    bus::TargetObservationBatch empty_batch{};
    bus::TrackedTargetBatch out_occ{};

    // Frame 2 (Miss 1) -> Occluded, confidence decays
    tracker.process(empty_batch, 1006944444LL, out_occ);
    TEST_ASSERT(out_occ.track_count == 1);
    TEST_ASSERT(out_occ.tracks[0].track_id == id);
    TEST_ASSERT(out_occ.tracks[0].state == TrackState::occluded);
    TEST_ASSERT(out_occ.tracks[0].total_missed_frames == 1);
    TEST_ASSERT(out_occ.tracks[0].confidence < 0.95f);

    // Frame 3 (Miss 2) -> Still retained
    tracker.process(empty_batch, 1013888888LL, out_occ);
    TEST_ASSERT(out_occ.track_count == 1);
    TEST_ASSERT(out_occ.tracks[0].state == TrackState::occluded);
    TEST_ASSERT(out_occ.tracks[0].total_missed_frames == 2);

    // Frame 4 (Miss 3) -> Still retained (at limit)
    tracker.process(empty_batch, 1020833333LL, out_occ);
    TEST_ASSERT(out_occ.track_count == 1);
    TEST_ASSERT(out_occ.tracks[0].state == TrackState::occluded);
    TEST_ASSERT(out_occ.tracks[0].total_missed_frames == 3);

    // Frame 5: Target reappears -> Recovered to Confirmed with SAME Track ID!
    bus::TargetObservationBatch batch5{};
    batch5.add_target(make_obs(102.0f, 101.0f, 15.0f, 0.90f));
    bus::TrackedTargetBatch out5{};
    tracker.process(batch5, 1027777777LL, out5);

    TEST_ASSERT(out5.track_count == 1);
    TEST_ASSERT(out5.tracks[0].track_id == id);
    TEST_ASSERT(out5.tracks[0].state == TrackState::confirmed);
    TEST_ASSERT(out5.tracks[0].total_visible_frames == 2);
    TEST_ASSERT(out5.tracks[0].total_missed_frames == 3);
    std::cout << "  -> Occlusion retention and recovery passed." << std::endl;
}

void test_occlusion_retirement_on_excess_misses() {
    std::cout << "[Test 5] Occlusion retirement after exceeding max missed frames..." << std::endl;

    MultiTargetTracker tracker{};

    // Frame 1: Confirmed target
    bus::TargetObservationBatch batch1{};
    batch1.add_target(make_obs(100.0f, 100.0f, 15.0f, 0.95f));
    bus::TrackedTargetBatch out{};
    tracker.process(batch1, 1000000000LL, out);
    TEST_ASSERT(out.track_count == 1);

    // Miss 1, 2, 3 frames
    bus::TargetObservationBatch empty_batch{};
    for (std::int64_t i = 1; i <= 3; ++i) {
        tracker.process(empty_batch, 1000000000LL + i * 6944444LL, out);
        TEST_ASSERT(out.track_count == 1);
    }

    // Miss 4th frame (exceeds max_missed_frames = 3) -> Retired!
    tracker.process(empty_batch, 1000000000LL + 4 * 6944444LL, out);
    TEST_ASSERT(out.track_count == 0);
    TEST_ASSERT(tracker.active_track_count() == 0);
    std::cout << "  -> Occlusion retirement passed." << std::endl;
}

void test_kinematic_tracking_and_direction_change() {
    std::cout << "[Test 6] Kinematic velocity estimation and direction reversal..." << std::endl;

    MultiTargetTracker tracker{};
    const float dt = 1.0f / 144.0f;
    const std::int64_t dt_ns = static_cast<std::int64_t>(dt * 1e9f);

    float cx = 100.0f;
    float cy = 200.0f;
    float vx = 300.0f; // Moving right at 300 px/s

    std::int64_t t_ns = 1000000000LL;
    bus::TrackedTargetBatch out{};

    // 1. Move right for 60 frames (~400ms)
    for (int f = 0; f < 60; ++f) {
        cx += vx * dt;
        t_ns += dt_ns;
        bus::TargetObservationBatch batch{};
        batch.add_target(make_obs(cx, cy, 15.0f, 0.95f));
        tracker.process(batch, t_ns, out);
    }

    TEST_ASSERT(out.track_count == 1);
    TEST_ASSERT(std::abs(out.tracks[0].filtered_center_px.x - cx) < 2.0f);
    TEST_ASSERT(std::abs(out.tracks[0].filtered_velocity_px_per_s.x_per_s - 300.0f) < 25.0f);

    // 2. Reverse direction: move left at -300 px/s for 60 frames
    vx = -300.0f;
    for (int f = 0; f < 60; ++f) {
        cx += vx * dt;
        t_ns += dt_ns;
        bus::TargetObservationBatch batch{};
        batch.add_target(make_obs(cx, cy, 15.0f, 0.95f));
        tracker.process(batch, t_ns, out);
    }

    TEST_ASSERT(out.track_count == 1);
    TEST_ASSERT(std::abs(out.tracks[0].filtered_center_px.x - cx) < 2.0f);
    TEST_ASSERT(std::abs(out.tracks[0].filtered_velocity_px_per_s.x_per_s - (-300.0f)) < 30.0f);
    std::cout << "  -> Kinematics & reversal passed." << std::endl;
}

void test_multi_target_crossing_identity_preservation() {
    std::cout << "[Test 7] Multi-target crossing and identity preservation..." << std::endl;

    MultiTargetTracker tracker{};
    const float dt = 1.0f / 144.0f;
    const std::int64_t dt_ns = static_cast<std::int64_t>(dt * 1e9f);

    // Target A starts at x=100, moves right (+200 px/s)
    // Target B starts at x=500, moves left (-200 px/s)
    // Target C is stationary at (300, 400)
    float ax = 100.0f, ay = 200.0f;
    float bx = 500.0f, by = 200.0f;
    float cx = 300.0f, cy = 400.0f;

    std::int64_t t_ns = 1000000000LL;
    bus::TrackedTargetBatch out{};

    // Warm up tracks
    bus::TargetObservationBatch init_batch{};
    init_batch.add_target(make_obs(ax, ay));
    init_batch.add_target(make_obs(bx, by));
    init_batch.add_target(make_obs(cx, cy));
    tracker.process(init_batch, t_ns, out);

    TEST_ASSERT(out.track_count == 3);
    const TrackId id_a = out.tracks[0].track_id;
    const TrackId id_b = out.tracks[1].track_id;
    const TrackId id_c = out.tracks[2].track_id;

    // Simulate crossing over 60 frames (crossing at x=300 around frame 35)
    for (int f = 0; f < 60; ++f) {
        ax += 200.0f * dt;
        bx -= 200.0f * dt;
        t_ns += dt_ns;

        bus::TargetObservationBatch batch{};
        batch.add_target(make_obs(ax, ay));
        batch.add_target(make_obs(bx, by));
        batch.add_target(make_obs(cx, cy));
        tracker.process(batch, t_ns, out);
    }

    TEST_ASSERT(out.track_count == 3);

    // Verify all 3 tracks are alive and maintain correct identities
    bool found_a = false, found_b = false, found_c = false;
    for (std::uint32_t i = 0; i < out.track_count; ++i) {
        if (out.tracks[i].track_id == id_a) {
            TEST_ASSERT(std::abs(out.tracks[i].filtered_center_px.x - ax) < 3.0f);
            found_a = true;
        } else if (out.tracks[i].track_id == id_b) {
            TEST_ASSERT(std::abs(out.tracks[i].filtered_center_px.x - bx) < 3.0f);
            found_b = true;
        } else if (out.tracks[i].track_id == id_c) {
            TEST_ASSERT(std::abs(out.tracks[i].filtered_center_px.x - cx) < 3.0f);
            found_c = true;
        }
    }
    TEST_ASSERT(found_a && found_b && found_c);
    std::cout << "  -> Multi-target crossing passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " Running OpenPrism M4-03 Track Lifecycle Test Suite " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_immediate_high_confidence_confirmation();
    test_tentative_two_frame_confirmation();
    test_tentative_ghost_pruning();
    test_occlusion_retention_and_recovery();
    test_occlusion_retirement_on_excess_misses();
    test_kinematic_tracking_and_direction_change();
    test_multi_target_crossing_identity_preservation();

    std::cout << "====================================================" << std::endl;
    std::cout << " All M4-03 Track Lifecycle Tests Passed!           " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
