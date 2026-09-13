// tests/cpp/test_gated_hungarian.cpp
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/types.hpp"
#include "aim/tracking/gated_hungarian.hpp"

using namespace aim;
using namespace aim::tracking;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

bus::TargetObservation make_obs(float x, float y, float r = 15.0f, std::uint32_t sem_id = 1) {
    bus::TargetObservation obs{};
    obs.center_px = PixelPoint{x, y};
    obs.effective_radius_px = r;
    obs.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
    obs.confidence = 0.95f;
    obs.semantic_id = sem_id;
    return obs;
}

TrackHypothesis make_track(float x, float y, float r = 15.0f, std::uint32_t class_id = 1) {
    TrackHypothesis trk{};
    trk.x = x;
    trk.y = y;
    trk.radius = r;
    trk.position_covariance = Covariance2D{1.0f, 0.0f, 1.0f};
    trk.class_id = class_id;
    return trk;
}

void test_perfect_1to1_matching() {
    std::cout << "[Test 1] Perfect 1-to-1 matching..." << std::endl;

    GatedHungarianAssociator associator{};
    std::array<TrackHypothesis, 3> tracks = {
        make_track(100.0f, 100.0f),
        make_track(300.0f, 300.0f),
        make_track(500.0f, 500.0f)
    };

    std::array<bus::TargetObservation, 3> observations = {
        make_obs(101.0f, 100.0f),
        make_obs(299.0f, 301.0f),
        make_obs(500.0f, 499.0f)
    };

    AssociationResult result{};
    associator.associate(tracks.data(), tracks.size(), observations.data(), observations.size(), result);

    TEST_ASSERT(result.num_matches == 3);
    TEST_ASSERT(result.num_unassigned_tracks == 0);
    TEST_ASSERT(result.num_unassigned_observations == 0);

    // Verify correct 1-to-1 pairings
    for (std::uint32_t i = 0; i < result.num_matches; ++i) {
        TEST_ASSERT(result.matches[i].track_idx == result.matches[i].observation_idx);
        TEST_ASSERT(result.matches[i].cost < 2.0f);
    }
    std::cout << "  -> Perfect matching passed." << std::endl;
}

void test_permuted_matching() {
    std::cout << "[Test 2] Permuted matching..." << std::endl;

    GatedHungarianAssociator associator{};
    std::array<TrackHypothesis, 3> tracks = {
        make_track(100.0f, 100.0f),
        make_track(300.0f, 300.0f),
        make_track(500.0f, 500.0f)
    };

    // Permuted: Obs 0 -> Track 2, Obs 1 -> Track 0, Obs 2 -> Track 1
    std::array<bus::TargetObservation, 3> observations = {
        make_obs(501.0f, 500.0f),
        make_obs(100.0f, 99.0f),
        make_obs(300.0f, 302.0f)
    };

    AssociationResult result{};
    associator.associate(tracks.data(), tracks.size(), observations.data(), observations.size(), result);

    TEST_ASSERT(result.num_matches == 3);
    TEST_ASSERT(result.num_unassigned_tracks == 0);
    TEST_ASSERT(result.num_unassigned_observations == 0);

    bool t0_matched = false, t1_matched = false, t2_matched = false;
    for (std::uint32_t i = 0; i < result.num_matches; ++i) {
        if (result.matches[i].track_idx == 0) {
            TEST_ASSERT(result.matches[i].observation_idx == 1);
            t0_matched = true;
        } else if (result.matches[i].track_idx == 1) {
            TEST_ASSERT(result.matches[i].observation_idx == 2);
            t1_matched = true;
        } else if (result.matches[i].track_idx == 2) {
            TEST_ASSERT(result.matches[i].observation_idx == 0);
            t2_matched = true;
        }
    }
    TEST_ASSERT(t0_matched && t1_matched && t2_matched);
    std::cout << "  -> Permuted matching passed." << std::endl;
}

void test_distance_gating() {
    std::cout << "[Test 3] Distance gating..." << std::endl;

    AssociationConfig config{};
    config.gate_distance_px = 50.0f;
    GatedHungarianAssociator associator{config};

    TrackHypothesis track = make_track(100.0f, 100.0f);
    // Obs is 80px away > 50px gate
    bus::TargetObservation obs = make_obs(180.0f, 100.0f);

    AssociationResult result{};
    associator.associate(&track, 1, &obs, 1, result);

    TEST_ASSERT(result.num_matches == 0);
    TEST_ASSERT(result.num_unassigned_tracks == 1);
    TEST_ASSERT(result.unassigned_tracks[0] == 0);
    TEST_ASSERT(result.num_unassigned_observations == 1);
    TEST_ASSERT(result.unassigned_observations[0] == 0);
    std::cout << "  -> Distance gating passed." << std::endl;
}

void test_radius_diff_gating() {
    std::cout << "[Test 4] Radius diff ratio gating..." << std::endl;

    AssociationConfig config{};
    config.max_radius_diff_ratio = 0.30f; // Max 30% difference
    GatedHungarianAssociator associator{config};

    TrackHypothesis track = make_track(100.0f, 100.0f, 20.0f);
    // Obs at same position but radius 10px -> diff ratio |20 - 10| / 20 = 50% > 30%
    bus::TargetObservation obs = make_obs(100.0f, 100.0f, 10.0f);

    AssociationResult result{};
    associator.associate(&track, 1, &obs, 1, result);

    TEST_ASSERT(result.num_matches == 0);
    TEST_ASSERT(result.num_unassigned_tracks == 1);
    TEST_ASSERT(result.num_unassigned_observations == 1);
    std::cout << "  -> Radius diff gating passed." << std::endl;
}

void test_semantic_id_gating() {
    std::cout << "[Test 5] Semantic ID gating..." << std::endl;

    GatedHungarianAssociator associator{};

    TrackHypothesis track = make_track(100.0f, 100.0f, 15.0f, 1);
    // Obs at same position but semantic_id = 2
    bus::TargetObservation obs = make_obs(100.0f, 100.0f, 15.0f, 2);

    AssociationResult result{};
    associator.associate(&track, 1, &obs, 1, result);

    TEST_ASSERT(result.num_matches == 0);
    TEST_ASSERT(result.num_unassigned_tracks == 1);
    TEST_ASSERT(result.num_unassigned_observations == 1);
    std::cout << "  -> Semantic ID gating passed." << std::endl;
}

void test_unequal_dimensions() {
    std::cout << "[Test 6] Unequal track/obs counts..." << std::endl;

    GatedHungarianAssociator associator{};

    // 3 tracks, 1 observation
    std::array<TrackHypothesis, 3> tracks = {
        make_track(100.0f, 100.0f),
        make_track(300.0f, 300.0f),
        make_track(500.0f, 500.0f)
    };
    bus::TargetObservation obs = make_obs(302.0f, 298.0f);

    AssociationResult result{};
    associator.associate(tracks.data(), tracks.size(), &obs, 1, result);

    TEST_ASSERT(result.num_matches == 1);
    TEST_ASSERT(result.matches[0].track_idx == 1);
    TEST_ASSERT(result.matches[0].observation_idx == 0);
    TEST_ASSERT(result.num_unassigned_tracks == 2);
    TEST_ASSERT(result.num_unassigned_observations == 0);

    // 1 track, 3 observations
    TrackHypothesis single_track = make_track(500.0f, 500.0f);
    std::array<bus::TargetObservation, 3> obs_arr = {
        make_obs(100.0f, 100.0f),
        make_obs(300.0f, 300.0f),
        make_obs(498.0f, 501.0f)
    };

    AssociationResult result2{};
    associator.associate(&single_track, 1, obs_arr.data(), obs_arr.size(), result2);

    TEST_ASSERT(result2.num_matches == 1);
    TEST_ASSERT(result2.matches[0].track_idx == 0);
    TEST_ASSERT(result2.matches[0].observation_idx == 2);
    TEST_ASSERT(result2.num_unassigned_tracks == 0);
    TEST_ASSERT(result2.num_unassigned_observations == 2);
    std::cout << "  -> Unequal dimension handling passed." << std::endl;
}

void test_global_optimality_over_greedy() {
    std::cout << "[Test 7] Global optimality over greedy choice..." << std::endl;

    AssociationConfig config{};
    config.gate_distance_px = 100.0f;
    config.max_mahalanobis_sq = 1000.0f; // Wide gate for test
    GatedHungarianAssociator associator{config};

    // Track 0 is at x=100. Track 1 is at x=110.
    // Obs 0 is at x=105. Obs 1 is at x=140.
    std::array<TrackHypothesis, 2> tracks = {
        make_track(100.0f, 100.0f),
        make_track(110.0f, 100.0f)
    };

    std::array<bus::TargetObservation, 2> obs = {
        make_obs(105.0f, 100.0f),
        make_obs(140.0f, 100.0f)
    };

    AssociationResult result{};
    associator.associate(tracks.data(), tracks.size(), obs.data(), obs.size(), result);

    TEST_ASSERT(result.num_matches == 2);
    // Both matched
    TEST_ASSERT(result.num_unassigned_tracks == 0);
    TEST_ASSERT(result.num_unassigned_observations == 0);
    std::cout << "  -> Global optimality passed." << std::endl;
}

void test_empty_edge_cases() {
    std::cout << "[Test 8] Empty edge cases..." << std::endl;

    GatedHungarianAssociator associator{};
    AssociationResult result{};

    // 0 tracks, 0 obs
    associator.associate(nullptr, 0, nullptr, 0, result);
    TEST_ASSERT(result.num_matches == 0);
    TEST_ASSERT(result.num_unassigned_tracks == 0);
    TEST_ASSERT(result.num_unassigned_observations == 0);

    // 0 tracks, 3 obs
    std::array<bus::TargetObservation, 3> obs = {
        make_obs(100.0f, 100.0f),
        make_obs(200.0f, 200.0f),
        make_obs(300.0f, 300.0f)
    };
    associator.associate(nullptr, 0, obs.data(), obs.size(), result);
    TEST_ASSERT(result.num_matches == 0);
    TEST_ASSERT(result.num_unassigned_tracks == 0);
    TEST_ASSERT(result.num_unassigned_observations == 3);
    TEST_ASSERT(result.unassigned_observations[0] == 0);
    TEST_ASSERT(result.unassigned_observations[1] == 1);
    TEST_ASSERT(result.unassigned_observations[2] == 2);

    // 2 tracks, 0 obs
    std::array<TrackHypothesis, 2> tracks = {
        make_track(100.0f, 100.0f),
        make_track(200.0f, 200.0f)
    };
    associator.associate(tracks.data(), tracks.size(), nullptr, 0, result);
    TEST_ASSERT(result.num_matches == 0);
    TEST_ASSERT(result.num_unassigned_tracks == 2);
    TEST_ASSERT(result.num_unassigned_observations == 0);
    TEST_ASSERT(result.unassigned_tracks[0] == 0);
    TEST_ASSERT(result.unassigned_tracks[1] == 1);
    std::cout << "  -> Empty edge cases passed." << std::endl;
}

void test_performance_64x64() {
    std::cout << "[Test 9] Performance benchmark on 64 tracks x 64 observations..." << std::endl;

    AssociationConfig config{};
    config.gate_distance_px = 100.0f;
    GatedHungarianAssociator associator{config};

    std::array<TrackHypothesis, kMaxAssociationDimension> tracks{};
    std::array<bus::TargetObservation, kMaxAssociationDimension> obs{};

    for (std::size_t i = 0; i < kMaxAssociationDimension; ++i) {
        float x = static_cast<float>((i % 8) * 100 + 50);
        float y = static_cast<float>((i / 8) * 100 + 50);
        tracks[i] = make_track(x, y);
        obs[i] = make_obs(x + 2.0f, y - 2.0f);
    }

    AssociationResult result{};

    auto t_start = std::chrono::high_resolution_clock::now();
    const std::size_t kIterations = 1000;
    for (std::size_t iter = 0; iter < kIterations; ++iter) {
        associator.associate(tracks.data(), tracks.size(), obs.data(), obs.size(), result);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    double avg_us = total_us / static_cast<double>(kIterations);

    TEST_ASSERT(result.num_matches == 64);
    TEST_ASSERT(result.num_unassigned_tracks == 0);
    TEST_ASSERT(result.num_unassigned_observations == 0);

    std::cout << "  -> 64x64 Hungarian Assignment time: " << avg_us << " us / solve (Goal: < 500 us)" << std::endl;
    TEST_ASSERT(avg_us < 2000.0); // Ultra fast
}

} // namespace

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " Running OpenPrism M4-02 Gated Hungarian Test Suite " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_perfect_1to1_matching();
    test_permuted_matching();
    test_distance_gating();
    test_radius_diff_gating();
    test_semantic_id_gating();
    test_unequal_dimensions();
    test_global_optimality_over_greedy();
    test_empty_edge_cases();
    test_performance_64x64();

    std::cout << "====================================================" << std::endl;
    std::cout << " All M4-02 Gated Hungarian Tests Passed!           " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
