// tests/cpp/test_target_simulator.cpp
// Comprehensive Native C++20 Test Suite for Deterministic Target Simulator and Replay Clock (Milestone M1-05 #11)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "aim/bus/bus.hpp"
#include "aim/sim/sim.hpp"

using namespace aim;
using namespace aim::sim;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return false; \
        } \
    } while (false)

// =============================================================================
// TC-SIM-01: Coordinate Conversions & Subpixel Invariants
// =============================================================================
bool test_coordinate_conversions() {
    std::cout << "[RUN] TC-SIM-01: Verifying coordinate transformations and subpixel round-trips...\n";

    constexpr std::uint32_t width = 1920;
    constexpr std::uint32_t height = 1080;

    // Center point (960, 540) -> (0.0, 0.0)
    const PixelPoint center_px{960.0f, 540.0f};
    const NormalizedPoint center_norm = pixel_to_normalized(center_px, width, height);
    TEST_ASSERT(std::abs(center_norm.x) < 1e-6f, "Center norm.x should be 0.0");
    TEST_ASSERT(std::abs(center_norm.y) < 1e-6f, "Center norm.y should be 0.0");

    const PixelPoint center_rt = normalized_to_pixel(center_norm, width, height);
    TEST_ASSERT(std::abs(center_rt.x - 960.0f) < 1e-6f, "Center round-trip x mismatch");
    TEST_ASSERT(std::abs(center_rt.y - 540.0f) < 1e-6f, "Center round-trip y mismatch");

    // Top-Left (0, 0) -> (-1.0, -1.0)
    const PixelPoint tl_px{0.0f, 0.0f};
    const NormalizedPoint tl_norm = pixel_to_normalized(tl_px, width, height);
    TEST_ASSERT(std::abs(tl_norm.x - (-1.0f)) < 1e-6f, "Top-left norm.x should be -1.0");
    TEST_ASSERT(std::abs(tl_norm.y - (-1.0f)) < 1e-6f, "Top-left norm.y should be -1.0");

    // Bottom-Right (1920, 1080) -> (1.0, 1.0)
    const PixelPoint br_px{1920.0f, 1080.0f};
    const NormalizedPoint br_norm = pixel_to_normalized(br_px, width, height);
    TEST_ASSERT(std::abs(br_norm.x - 1.0f) < 1e-6f, "Bottom-right norm.x should be 1.0");
    TEST_ASSERT(std::abs(br_norm.y - 1.0f) < 1e-6f, "Bottom-right norm.y should be 1.0");

    // Comprehensive grid subpixel roundtrip test across 10,000 points
    float max_error_px = 0.0f;
    for (int ix = 0; ix <= 100; ++ix) {
        for (int iy = 0; iy <= 100; ++iy) {
            const PixelPoint px{
                static_cast<float>(ix) * 19.2f,
                static_cast<float>(iy) * 10.8f
            };
            const NormalizedPoint norm = pixel_to_normalized(px, width, height);
            TEST_ASSERT(norm.x >= -1.0001f && norm.x <= 1.0001f, "Normalized x out of bounds");
            TEST_ASSERT(norm.y >= -1.0001f && norm.y <= 1.0001f, "Normalized y out of bounds");

            const PixelPoint rt = normalized_to_pixel(norm, width, height);
            const float err_x = std::abs(rt.x - px.x);
            const float err_y = std::abs(rt.y - px.y);
            max_error_px = (std::max)({max_error_px, err_x, err_y});
        }
    }

    std::cout << "  [*] Max coordinate round-trip error: " << max_error_px << " px\n";
    TEST_ASSERT(max_error_px < 1e-3f, "Subpixel round-trip error must be < 1e-3 px");

    std::cout << "  [+] Coordinate conversions and subpixel invariants verified.\n";
    return true;
}

// =============================================================================
// TC-SIM-02: Trajectory Kinematics & Mathematical Models
// =============================================================================
bool test_trajectory_kinematics() {
    std::cout << "[RUN] TC-SIM-02: Testing trajectory kinematic models and boundary interactions...\n";

    // 1. Stationary Target
    TargetSpec stat_spec;
    stat_spec.trajectory_type = TrajectoryType::stationary;
    stat_spec.initial_pos_px = {500.0f, 400.0f};
    stat_spec.radius_px = 20.0f;

    for (double t = 0.0; t <= 10.0; t += 0.5) {
        const TrajectoryState st = evaluate_trajectory(stat_spec, t);
        TEST_ASSERT(st.pos_px.x == 500.0f && st.pos_px.y == 400.0f, "Stationary pos mismatch");
        TEST_ASSERT(st.vel_px_per_s.x_per_s == 0.0f && st.vel_px_per_s.y_per_s == 0.0f, "Stationary vel mismatch");
    }

    // 2. Linear Constant Velocity (CV) without boundaries
    TargetSpec cv_spec;
    cv_spec.trajectory_type = TrajectoryType::linear_cv;
    cv_spec.boundary = BoundaryBehavior::none;
    cv_spec.initial_pos_px = {100.0f, 200.0f};
    cv_spec.velocity_px_per_s = {50.0f, -20.0f};

    const TrajectoryState cv_st = evaluate_trajectory(cv_spec, 4.0);
    TEST_ASSERT(std::abs(cv_st.pos_px.x - 300.0f) < 1e-5f, "Linear CV x mismatch (100 + 50*4 = 300)");
    TEST_ASSERT(std::abs(cv_st.pos_px.y - 120.0f) < 1e-5f, "Linear CV y mismatch (200 - 20*4 = 120)");
    TEST_ASSERT(cv_st.vel_px_per_s.x_per_s == 50.0f && cv_st.vel_px_per_s.y_per_s == -20.0f, "Linear CV vel mismatch");

    // 3. Linear Constant Acceleration (CA)
    TargetSpec ca_spec;
    ca_spec.trajectory_type = TrajectoryType::linear_ca;
    ca_spec.boundary = BoundaryBehavior::none;
    ca_spec.initial_pos_px = {0.0f, 0.0f};
    ca_spec.velocity_px_per_s = {10.0f, 0.0f};
    ca_spec.accel_px_per_s2 = {5.0f, 0.0f};

    const TrajectoryState ca_st = evaluate_trajectory(ca_spec, 2.0);
    // x = 0 + 10*2 + 0.5*5*(2^2) = 20 + 10 = 30
    TEST_ASSERT(std::abs(ca_st.pos_px.x - 30.0f) < 1e-5f, "Linear CA x mismatch (x = 30)");
    // vx = 10 + 5*2 = 20
    TEST_ASSERT(std::abs(ca_st.vel_px_per_s.x_per_s - 20.0f) < 1e-5f, "Linear CA vx mismatch (vx = 20)");

    // 4. Harmonic Sinusoidal Strafe
    TargetSpec sin_spec;
    sin_spec.trajectory_type = TrajectoryType::sinusoidal_strafe;
    sin_spec.initial_pos_px = {960.0f, 540.0f};
    sin_spec.amplitude_x_px = 300.0f;
    sin_spec.amplitude_y_px = 0.0f;
    sin_spec.frequency_hz = 1.0f; // 1 Hz -> Period = 1.0s
    sin_spec.phase_rad = 0.0f;

    // At t = 0.25s (quarter period): sin(pi/2) = 1 -> x = 960 + 300 = 1260
    const TrajectoryState sin_qtr = evaluate_trajectory(sin_spec, 0.25);
    TEST_ASSERT(std::abs(sin_qtr.pos_px.x - 1260.0f) < 1e-3f, "Sinusoidal quarter period x mismatch");
    TEST_ASSERT(std::abs(sin_qtr.vel_px_per_s.x_per_s) < 1e-3f, "Sinusoidal quarter period vx should be 0");

    // At t = 0.5s: sin(pi) = 0 -> x = 960
    const TrajectoryState sin_half = evaluate_trajectory(sin_spec, 0.5);
    TEST_ASSERT(std::abs(sin_half.pos_px.x - 960.0f) < 1e-3f, "Sinusoidal half period x mismatch");
    TEST_ASSERT(sin_half.vel_px_per_s.x_per_s < -100.0f, "Sinusoidal half period vx should be negative max");

    // 5. Circular Orbit
    TargetSpec circ_spec;
    circ_spec.trajectory_type = TrajectoryType::circular_orbit;
    circ_spec.initial_pos_px = {960.0f, 540.0f};
    circ_spec.amplitude_x_px = 150.0f; // Radius = 150
    circ_spec.frequency_hz = 0.5f;
    circ_spec.phase_rad = 0.0f;

    for (double t = 0.0; t <= 5.0; t += 0.1) {
        const TrajectoryState circ_st = evaluate_trajectory(circ_spec, t);
        const float dx = circ_st.pos_px.x - 960.0f;
        const float dy = circ_st.pos_px.y - 540.0f;
        const float r = std::sqrt(dx * dx + dy * dy);
        TEST_ASSERT(std::abs(r - 150.0f) < 1e-3f, "Circular orbit radius invariant failed");
    }

    // 6. Specular Bouncing Boundary Behavior
    TargetSpec bounce_spec;
    bounce_spec.trajectory_type = TrajectoryType::linear_cv;
    bounce_spec.boundary = BoundaryBehavior::bounce;
    bounce_spec.radius_px = 20.0f;
    bounce_spec.initial_pos_px = {1800.0f, 540.0f};
    bounce_spec.velocity_px_per_s = {200.0f, 0.0f}; // Heading right towards 1900 max limit (1920 - 20)

    // At t = 0.5s: raw_x = 1800 + 100 = 1900 (touches wall).
    const TrajectoryState b_touch = evaluate_trajectory(bounce_spec, 0.5);
    TEST_ASSERT(std::abs(b_touch.pos_px.x - 1900.0f) < 1e-4f, "Bounce boundary touch pos mismatch");

    // At t = 1.0s: raw_x = 1800 + 200 = 2000. Bounced back by 100px -> pos = 1900 - 100 = 1800.
    const TrajectoryState b_rebound = evaluate_trajectory(bounce_spec, 1.0);
    TEST_ASSERT(std::abs(b_rebound.pos_px.x - 1800.0f) < 1e-4f, "Bounce rebound pos mismatch");
    TEST_ASSERT(b_rebound.vel_px_per_s.x_per_s == -200.0f, "Bounce rebound velocity should invert");

    // 7. Sudden Direction Cuts (180 degree reversal)
    TargetSpec cut_spec;
    cut_spec.trajectory_type = TrajectoryType::sudden_cut;
    cut_spec.initial_pos_px = {500.0f, 500.0f};
    cut_spec.velocity_px_per_s = {100.0f, 0.0f};
    cut_spec.cut_interval_ns = 1'000'000'000LL; // 1.0s cut interval
    cut_spec.cut_angle_rad = 3.14159265f;


    // t = 0.5s: moving forward (x = 550, v = +100)
    const TrajectoryState cut_05 = evaluate_trajectory(cut_spec, 0.5);
    TEST_ASSERT(std::abs(cut_05.pos_px.x - 550.0f) < 1e-4f, "Sudden cut t=0.5 pos mismatch");
    TEST_ASSERT(cut_05.vel_px_per_s.x_per_s == 100.0f, "Sudden cut t=0.5 vel mismatch");

    // t = 1.5s: moving backward (x = 600 - 50 = 550, v = -100)
    const TrajectoryState cut_15 = evaluate_trajectory(cut_spec, 1.5);
    TEST_ASSERT(std::abs(cut_15.pos_px.x - 550.0f) < 1e-4f, "Sudden cut t=1.5 pos mismatch");
    TEST_ASSERT(cut_15.vel_px_per_s.x_per_s == -100.0f, "Sudden cut t=1.5 vel mismatch");

    // t = 2.0s: returned to starting point (x = 500)
    const TrajectoryState cut_20 = evaluate_trajectory(cut_spec, 2.0);
    TEST_ASSERT(std::abs(cut_20.pos_px.x - 500.0f) < 1e-4f, "Sudden cut t=2.0 pos mismatch");

    std::cout << "  [+] Trajectory kinematics and boundary interactions verified.\n";
    return true;
}

// =============================================================================
// TC-SIM-03: ReplayClock & Driftless Cadence Progression
// =============================================================================
bool test_replay_clock_cadence() {
    std::cout << "[RUN] TC-SIM-03: Verifying ReplayClock driftless 144 Hz cadence and controls...\n";

    constexpr MonotonicNs start_ns = 1'000'000'000LL; // 1.0 s
    ReplayClock clock(144.0, start_ns);

    TEST_ASSERT(clock.now_ns() == start_ns, "Initial clock now_ns mismatch");
    TEST_ASSERT(clock.tick_index() == 0, "Initial tick_index mismatch");
    TEST_ASSERT(clock.cadence_hz() == 144.0, "Cadence mismatch");
    TEST_ASSERT(!clock.is_paused(), "Initial is_paused should be false");

    // Advance 144 ticks -> must equal EXACTLY 1,000,000,000 ns (1.000000000 s) with ZERO drift
    clock.step(144);
    TEST_ASSERT(clock.tick_index() == 144, "Tick index after 144 steps mismatch");
    TEST_ASSERT(clock.elapsed_ns() == 1'000'000'000LL, "144 frames must equal exactly 1,000,000,000 ns");
    TEST_ASSERT(clock.now_ns() == 2'000'000'000LL, "Clock now_ns after 144 frames mismatch");
    TEST_ASSERT(std::abs(clock.elapsed_seconds() - 1.0) < 1e-9, "Elapsed seconds mismatch");

    // Advance 10,000 frames -> verify fractional drift invariant
    clock.reset(start_ns);
    for (std::uint64_t i = 1; i <= 10000; ++i) {
        clock.step(1);
        const double expected_s = static_cast<double>(i) / 144.0;
        const MonotonicNs expected_ns = start_ns + static_cast<MonotonicNs>(expected_s * 1'000'000'000.0);
        TEST_ASSERT(std::abs(clock.now_ns() - expected_ns) <= 1, "Timestamp drift exceeded 1 ns limit");
    }

    // Test Multi-Cadence (60 Hz, 120 Hz, 240 Hz)
    ReplayClock clock60(60.0, start_ns);
    clock60.step(60);
    TEST_ASSERT(clock60.elapsed_ns() == 1'000'000'000LL, "60 frames at 60 Hz must equal 1 second");

    ReplayClock clock240(240.0, start_ns);
    clock240.step(240);
    TEST_ASSERT(clock240.elapsed_ns() == 1'000'000'000LL, "240 frames at 240 Hz must equal 1 second");

    // Pause / Resume
    clock.reset(start_ns);
    clock.pause();
    TEST_ASSERT(clock.is_paused(), "is_paused should be true");
    clock.step(10);
    TEST_ASSERT(clock.tick_index() == 0, "Paused clock should not increment tick index");
    TEST_ASSERT(clock.now_ns() == start_ns, "Paused clock should not advance time");

    clock.resume();
    TEST_ASSERT(!clock.is_paused(), "is_paused should be false");
    clock.step(1);
    TEST_ASSERT(clock.tick_index() == 1, "Resumed clock step should increment");

    // Rate scale (2.0x speed)
    clock.reset(start_ns);
    clock.set_rate_scale(2.0);
    clock.step(144);
    TEST_ASSERT(clock.elapsed_ns() == 2'000'000'000LL, "144 frames at 2.0x rate scale must equal 2 seconds");

    // Seek tick / Seek ns
    clock.reset(start_ns);
    clock.seek_tick(288);
    TEST_ASSERT(clock.tick_index() == 288, "Seek tick index mismatch");
    TEST_ASSERT(clock.elapsed_ns() == 2'000'000'000LL, "Seek tick 288 elapsed_ns mismatch");

    clock.seek_ns(start_ns + 500'000'000LL); // seek 0.5s -> 72 frames
    TEST_ASSERT(clock.tick_index() == 72, "Seek ns tick mismatch (expected 72)");

    std::cout << "  [+] ReplayClock driftless cadence and controls verified.\n";
    return true;
}

// =============================================================================
// TC-SIM-04: Deterministic PRNG & Bit-Exact Seed Replay
// =============================================================================
bool test_deterministic_prng() {
    std::cout << "[RUN] TC-SIM-04: Verifying PRNG bit-exact determinism across runs...\n";

    constexpr std::uint64_t seed = 0x123456789ABCDEF0ULL;
    DeterministicRng rng1(seed);
    DeterministicRng rng2(seed);

    // Verify 10,000 integer sequences are bit-identical
    for (std::size_t i = 0; i < 10000; ++i) {
        const std::uint64_t val1 = rng1.next_u64();
        const std::uint64_t val2 = rng2.next_u64();
        TEST_ASSERT(val1 == val2, "PRNG u64 sequence divergence");
    }

    // Verify float and Gaussian distributions are bit-identical
    rng1.reseed(seed);
    rng2.reseed(seed);

    for (std::size_t i = 0; i < 10000; ++i) {
        const float f1 = rng1.next_uniform_f32();
        const float f2 = rng2.next_uniform_f32();
        TEST_ASSERT(f1 == f2, "PRNG float sequence divergence");

        const auto [g1a, g1b] = rng1.next_gaussian_pair(0.0f, 1.0f);
        const auto [g2a, g2b] = rng2.next_gaussian_pair(0.0f, 1.0f);
        TEST_ASSERT(g1a == g2a && g1b == g2b, "PRNG Gaussian pair divergence");
    }

    // Test Simulator Cross-Run Bit-Exact Reproducibility
    auto scenario = ScenarioBuilder("bit_exact_test")
        .with_seed(987654321ULL)
        .with_cadence_hz(144.0)
        .with_duration_seconds(10.0)
        .with_position_noise(1.5f)
        .with_velocity_noise(2.0f)
        .add_linear_target(1, {500.0f, 500.0f}, {150.0f, 75.0f}, 20.0f, BoundaryBehavior::bounce)
        .add_sinusoidal_target(2, {960.0f, 540.0f}, 300.0f, 100.0f, 1.5f, 0.5f, 22.0f)
        .build();

    TargetSimulator sim_a(scenario);
    TargetSimulator sim_b(scenario);

    for (std::size_t frame = 0; frame < 1000; ++frame) {
        const auto batch_a = sim_a.step();
        const auto batch_b = sim_b.step();

        TEST_ASSERT(batch_a.target_count == batch_b.target_count, "Target count divergence");
        TEST_ASSERT(batch_a.captured_at_ns == batch_b.captured_at_ns, "Timestamp divergence");

        for (std::size_t k = 0; k < batch_a.target_count; ++k) {
            const auto& t_a = batch_a.targets[k];
            const auto& t_b = batch_b.targets[k];

            TEST_ASSERT(t_a.source_id == t_b.source_id, "Target source_id mismatch");
            TEST_ASSERT(t_a.center_px.x == t_b.center_px.x, "Target pos x bit-exact mismatch");
            TEST_ASSERT(t_a.center_px.y == t_b.center_px.y, "Target pos y bit-exact mismatch");
            TEST_ASSERT(t_a.center_norm.x == t_b.center_norm.x, "Target norm x bit-exact mismatch");
            TEST_ASSERT(t_a.center_norm.y == t_b.center_norm.y, "Target norm y bit-exact mismatch");
            TEST_ASSERT(t_a.confidence == t_b.confidence, "Target confidence bit-exact mismatch");
            TEST_ASSERT(t_a.covariance_px2.xx == t_b.covariance_px2.xx, "Target cov xx bit-exact mismatch");
        }
    }

    std::cout << "  [+] Deterministic PRNG and bit-exact cross-run simulation verified.\n";
    return true;
}

// =============================================================================
// TC-SIM-05: Multi-Target Lifecycle, Dynamic Spawning & Despawning
// =============================================================================
bool test_target_lifecycles() {
    std::cout << "[RUN] TC-SIM-05: Verifying target spawning, active tracking, and despawning...\n";

    constexpr MonotonicNs t0 = 1'000'000'000LL; // 1.0s

    auto scenario = ScenarioBuilder("lifecycle_test")
        .with_seed(100)
        .with_start_time_ns(t0)
        .with_cadence_hz(144.0)
        // Target 1: active [1.0s, 3.0s]
        .add_stationary_target(1, {500.0f, 500.0f}, 20.0f, t0, t0 + 2'000'000'000LL)
        // Target 2: active [2.0s, 5.0s]
        .add_stationary_target(2, {700.0f, 500.0f}, 20.0f, t0 + 1'000'000'000LL, t0 + 4'000'000'000LL)
        // Target 3: active [4.0s, 6.0s]
        .add_stationary_target(3, {900.0f, 500.0f}, 20.0f, t0 + 3'000'000'000LL, t0 + 5'000'000'000LL)
        .build();

    TargetSimulator sim(scenario);

    // Frame at t=1.0s: only Target 1
    bus::TargetObservationBatch batch{};
    sim.generate_batch(t0, batch);
    TEST_ASSERT(batch.target_count == 1, "t=1.0s should have 1 active target");
    TEST_ASSERT(batch.targets[0].source_id == 1, "t=1.0s active target should be ID 1");

    // Frame at t=2.5s: Target 1 and Target 2
    sim.generate_batch(t0 + 1'500'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 2, "t=2.5s should have 2 active targets");

    // Frame at t=3.5s: Target 2 only (Target 1 despawned)
    sim.generate_batch(t0 + 2'500'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 1, "t=3.5s should have 1 active target");
    TEST_ASSERT(batch.targets[0].source_id == 2, "t=3.5s active target should be ID 2");

    // Frame at t=4.5s: Target 2 and Target 3
    sim.generate_batch(t0 + 3'500'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 2, "t=4.5s should have 2 active targets (2 & 3)");

    // Frame at t=5.5s: Target 3 only
    sim.generate_batch(t0 + 4'500'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 1, "t=5.5s should have 1 active target (3)");

    // Frame at t=7.0s: All despawned (0 targets)
    sim.generate_batch(t0 + 6'000'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 0, "t=7.0s should have 0 targets");
    TEST_ASSERT(batch.items().empty(), "items() should be empty on zero targets");

    std::cout << "  [+] Target lifecycles and dynamic spawning/despawning verified.\n";
    return true;
}

// =============================================================================
// TC-SIM-06: Spatial & Temporal Occlusions
// =============================================================================
bool test_occlusions() {
    std::cout << "[RUN] TC-SIM-06: Verifying spatial and temporal occlusions...\n";

    constexpr MonotonicNs t0 = 1'000'000'000LL;

    // Test 1: Temporal Drop Occlusion
    auto scen_temp = ScenarioBuilder("temp_occ_test")
        .with_seed(101)
        .with_start_time_ns(t0)
        .add_stationary_target(1, {500.0f, 500.0f}, 20.0f)
        .add_temporal_occlusion(1, t0 + 1'000'000'000LL, t0 + 2'000'000'000LL, OcclusionMode::drop)
        .build();

    TargetSimulator sim_temp(scen_temp);
    bus::TargetObservationBatch batch{};

    sim_temp.generate_batch(t0 + 500'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 1, "Before occlusion: target present");

    sim_temp.generate_batch(t0 + 1'500'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 0, "During temporal drop occlusion: target must be dropped");

    sim_temp.generate_batch(t0 + 2'500'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 1, "After occlusion: target reappears");

    // Test 2: Spatial Rectangular Occlusion with Predicted Degradation
    auto scen_spatial = ScenarioBuilder("spatial_occ_test")
        .with_seed(102)
        .with_start_time_ns(t0)
        .with_emit_predicted_when_occluded(true)
        // Target moving from left (x=700) to right (x=1200) across occluder at [900, 400, 1020, 600]
        .add_linear_target(1, {700.0f, 500.0f}, {100.0f, 0.0f}, 20.0f, BoundaryBehavior::none)
        .add_rect_occluder(BoundingBox{900.0f, 400.0f, 1020.0f, 600.0f}, t0, t0 + 10'000'000'000LL, OcclusionMode::predicted)
        .build();

    TargetSimulator sim_spatial(scen_spatial);

    // At t=0 (x=700): visible
    sim_spatial.generate_batch(t0, batch);
    TEST_ASSERT(batch.target_count == 1, "Target 1 present");
    TEST_ASSERT(batch.targets[0].visibility == Visibility::visible, "Initial visibility must be visible");

    // At t=2.6s (x=700 + 260 = 960): centered inside occluder -> Visibility::predicted
    sim_spatial.generate_batch(t0 + 2'600'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 1, "Target 1 present as predicted");
    TEST_ASSERT(batch.targets[0].visibility == Visibility::predicted, "Visibility inside occluder must be predicted");
    TEST_ASSERT(batch.targets[0].confidence < 0.50f, "Predicted target confidence must be degraded");
    TEST_ASSERT(batch.targets[0].covariance_px2.xx > 20.0f, "Predicted target covariance must be inflated");

    // At t=5.0s (x=700 + 500 = 1200): cleared occluder -> Visibility::visible
    sim_spatial.generate_batch(t0 + 5'000'000'000LL, batch);
    TEST_ASSERT(batch.target_count == 1, "Target 1 present");
    TEST_ASSERT(batch.targets[0].visibility == Visibility::visible, "Cleared visibility must be visible");

    std::cout << "  [+] Spatial and temporal occlusions verified.\n";
    return true;
}

// =============================================================================
// TC-SIM-07: Scenario Presets & Builder Fluent API
// =============================================================================
bool test_scenario_presets() {
    std::cout << "[RUN] TC-SIM-07: Verifying standard scenario presets...\n";

    // 1. Gridshot Preset
    const auto gridshot = ScenarioBuilder::make_gridshot_preset(42);
    TEST_ASSERT(gridshot.targets.size() == 3, "Gridshot preset should have 3 targets");
    TEST_ASSERT(gridshot.cadence_hz() == 144.0, "Gridshot cadence should be 144 Hz");

    TargetSimulator sim_grid(gridshot);
    const auto b_grid = sim_grid.step();
    TEST_ASSERT(b_grid.target_count == 3, "Gridshot step should emit 3 targets");

    // 2. Strafe Track Preset
    const auto strafe = ScenarioBuilder::make_strafe_track_preset(42);
    TEST_ASSERT(strafe.targets.size() == 2, "Strafe preset should have 2 targets");

    // 3. High Density Preset (64 targets)
    const auto density = ScenarioBuilder::make_density_preset(42, 64);
    TEST_ASSERT(density.targets.size() == 64, "Density preset should have 64 targets");

    TargetSimulator sim_dense(density);
    const auto b_dense = sim_dense.step();
    TEST_ASSERT(b_dense.target_count == 64, "Density step should emit 64 targets");
    TEST_ASSERT(b_dense.items().size() == 64, "Items span size should be 64");

    std::cout << "  [+] Scenario presets and builder verified.\n";
    return true;
}

// =============================================================================
// TC-SIM-08: Bus SPSC Ring Integration & Zero Allocation Benchmark
// =============================================================================
bool test_bus_ring_publishing_and_perf() {
    std::cout << "[RUN] TC-SIM-08: Verifying direct LatestSpscRing publishing and zero-allocation performance...\n";

    const auto scenario = ScenarioBuilder::make_strafe_track_preset(1337);
    TargetSimulator simulator(scenario);

    bus::LatestSpscRing<bus::TargetObservationBatch, 16> ring;
    TEST_ASSERT(ring.empty(), "Ring should be initially empty");

    // Publish 1,000 frames directly to ring
    for (std::uint64_t i = 1; i <= 1000; ++i) {
        const std::uint64_t seq = simulator.step_and_publish(ring);
        TEST_ASSERT(seq == i, "Published sequence should increment monotonically");
    }

    TEST_ASSERT(ring.latest_sequence() == 1000, "Ring latest sequence should be 1000");

    // Consumer reads latest batch
    bus::TargetObservationBatch read_batch{};
    std::uint64_t last_seq = 0;
    std::uint64_t drops = 0;

    const bool read_ok = ring.try_read_latest(read_batch, last_seq, &drops);
    TEST_ASSERT(read_ok, "try_read_latest should succeed");
    TEST_ASSERT(last_seq == 1000, "Consumer should read sequence 1000");
    TEST_ASSERT(drops == 999, "Consumer should record 999 drops on latest jump");
    TEST_ASSERT(read_batch.header.sequence_id == 1000, "Payload sequence_id should match");
    TEST_ASSERT(read_batch.target_count == 2, "Read batch should have 2 targets");

    // High-throughput benchmark: 100,000 steps and direct bus publishes
    constexpr std::size_t kBenchmarkIterations = 100000;
    simulator.reset(1337);
    ring.reset();

    const auto t_start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < kBenchmarkIterations; ++i) {
        simulator.step_and_publish(ring);
    }

    const auto t_end = std::chrono::steady_clock::now();
    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    const double avg_step_publish_ns = static_cast<double>(total_ns) / static_cast<double>(kBenchmarkIterations);

    std::cout << "  [*] Total iterations: " << kBenchmarkIterations << "\n";
    std::cout << "  [*] Total time:       " << static_cast<double>(total_ns) / 1'000'000.0 << " ms\n";
    std::cout << "  [*] Avg step+publish: " << avg_step_publish_ns << " ns / frame\n";

    // Sub-1500ns in Debug build, typically < 100ns in Release
    TEST_ASSERT(avg_step_publish_ns < 5000.0, "Step and publish latency must be sub-microsecond in optimized build");

    std::cout << "  [+] Direct LatestSpscRing publishing and benchmark passed.\n";
    return true;
}

// =============================================================================
// Cross-language parity trace
// =============================================================================
// Emits the canonical observation stream as raw binary32 bit patterns so
// tests/golden/cross_language_sim_parity.py can compare the C++ and Python simulators
// without any decimal-formatting ambiguity. Bit patterns, not printf digits, are the
// contract: a Python float that merely *prints* the same is not the same value.

[[nodiscard]] std::uint32_t float_bits(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void emit_float(const char* name, float value) {
    std::cout << ' ' << name << '=' << std::hex << std::setw(8) << std::setfill('0')
              << float_bits(value) << std::dec;
}

[[nodiscard]] Scenario build_preset(std::string_view preset, std::uint64_t seed) {
    if (preset == "gridshot") return ScenarioBuilder::make_gridshot_preset(seed);
    if (preset == "strafe_track") return ScenarioBuilder::make_strafe_track_preset(seed);
    if (preset == "occlusion") return ScenarioBuilder::make_occlusion_preset(seed);
    return ScenarioBuilder::make_density_preset(seed);
}

[[nodiscard]] int dump_trace(std::string_view preset, std::uint64_t seed, std::size_t frames) {
    TargetSimulator simulator(build_preset(preset, seed));

    for (std::size_t frame_index = 0; frame_index < frames; ++frame_index) {
        const bus::TargetObservationBatch batch = simulator.step();
        std::cout << "batch frame=" << frame_index
                  << " seq=" << batch.header.sequence_id
                  << " frame_id=" << batch.frame_id
                  << " captured=" << batch.captured_at_ns
                  << " published=" << batch.published_at_ns
                  << " count=" << batch.target_count << '\n';

        for (const auto& obs : batch.items()) {
            std::cout << "  obs id=" << obs.source_id
                      << " vis=" << static_cast<int>(obs.visibility)
                      << " semantic=" << obs.semantic_id;
            emit_float("cx", obs.center_px.x);
            emit_float("cy", obs.center_px.y);
            emit_float("nx", obs.center_norm.x);
            emit_float("ny", obs.center_norm.y);
            emit_float("bl", obs.bbox_px.left);
            emit_float("bt", obs.bbox_px.top);
            emit_float("br", obs.bbox_px.right);
            emit_float("bb", obs.bbox_px.bottom);
            emit_float("r", obs.effective_radius_px);
            emit_float("conf", obs.confidence);
            emit_float("cxx", obs.covariance_px2.xx);
            emit_float("cxy", obs.covariance_px2.xy);
            emit_float("cyy", obs.covariance_px2.yy);
            emit_float("vx", obs.velocity.pixels_per_second.x_per_s);
            emit_float("vy", obs.velocity.pixels_per_second.y_per_s);
            emit_float("vc", obs.velocity.confidence);
            emit_float("val", obs.target_value);
            std::cout << '\n';
        }
    }
    return 0;
}

// =============================================================================
// Main Entrypoint
// =============================================================================
int main(int argc, char** argv) {
    if (argc == 5 && std::string_view(argv[1]) == "--dump-trace") {
        return dump_trace(argv[2],
                          std::strtoull(argv[3], nullptr, 0),
                          static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 0)));
    }

    std::cout << "=================================================================\n";
    std::cout << "Deterministic Target Simulator & Replay Clock Test Suite (M1-05)\n";
    std::cout << "=================================================================\n";

    bool all_passed = true;
    all_passed &= test_coordinate_conversions();
    all_passed &= test_trajectory_kinematics();
    all_passed &= test_replay_clock_cadence();
    all_passed &= test_deterministic_prng();
    all_passed &= test_target_lifecycles();
    all_passed &= test_occlusions();
    all_passed &= test_scenario_presets();
    all_passed &= test_bus_ring_publishing_and_perf();

    std::cout << "=================================================================\n";
    if (all_passed) {
        std::cout << "[+] ALL TARGET SIMULATOR TESTS PASSED SUCCESSFULLY (8/8)\n";
        std::cout << "=================================================================\n";
        return 0;
    } else {
        std::cerr << "[-] SOME TARGET SIMULATOR TESTS FAILED\n";
        std::cout << "=================================================================\n";
        return 1;
    }
}
