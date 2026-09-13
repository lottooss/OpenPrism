// tests/cpp/test_latency_estimator_and_extrapolation.cpp
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/types.hpp"
#include "aim/tracking/latency_estimator.hpp"
#include "aim/tracking/prediction_extrapolator.hpp"
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

bus::TargetObservation make_obs(float x, float y, float r = 15.0f, float conf = 0.95f) {
    bus::TargetObservation obs{};
    obs.center_px = PixelPoint{x, y};
    obs.effective_radius_px = r;
    obs.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
    obs.confidence = conf;
    obs.semantic_id = 1;
    return obs;
}

void test_latency_outlier_rejection_and_ewma() {
    std::cout << "[Test 1] Latency outlier rejection and EWMA filtering..." << std::endl;

    LatencyEstimatorConfig config{};
    config.min_latency_ns = 2'000'000LL;  // 2 ms
    config.max_latency_ns = 20'000'000LL; // 20 ms
    config.ewma_alpha = 0.20f;
    config.warmup_samples = 4;
    CommandEffectLatencyEstimator estimator{config};

    MonotonicNs t = 1'000'000'000LL;

    // Feed normal 5ms samples
    for (int i = 0; i < 10; ++i) {
        t += 6'944'444LL;
        estimator.update(5'000'000LL, t);
    }

    TEST_ASSERT(estimator.estimate().is_stable);
    TEST_ASSERT(std::abs(estimator.estimate().estimated_latency_ns - 5'000'000LL) < 200'000LL);

    // Feed a massive 100ms outlier spike -> should be clamped to 20ms max
    t += 6'944'444LL;
    estimator.update(100'000'000LL, t);

    // After 1 spike with alpha=0.2, estimate is 5ms + 0.2*(20ms - 5ms) = 8ms
    TEST_ASSERT(estimator.estimate().estimated_latency_ns <= 8'500'000LL);

    // Feed a negative outlier (-50ms) -> clamped to 2ms min
    t += 6'944'444LL;
    estimator.update(-50'000'000LL, t);
    TEST_ASSERT(estimator.estimate().estimated_latency_ns >= 5'000'000LL);

    std::cout << "  -> Outlier rejection passed." << std::endl;
}

void test_stability_gate_and_fail_closed() {
    std::cout << "[Test 2] Stability gating and fail-closed behavior..." << std::endl;

    LatencyEstimatorConfig config{};
    config.warmup_samples = 5;
    config.max_allowed_std_dev_ns = 1'000'000.0f; // 1 ms max std dev
    CommandEffectLatencyEstimator estimator{config};

    MonotonicNs t = 1'000'000'000LL;

    // Sample 1: not stable yet (needs 5 samples)
    estimator.update(5'000'000LL, t);
    TEST_ASSERT(!estimator.estimate().is_stable);

    // High jitter samples (std dev > 1ms)
    estimator.update(2'000'000LL, t += 7'000'000LL);
    estimator.update(15'000'000LL, t += 7'000'000LL);
    estimator.update(2'000'000LL, t += 7'000'000LL);
    estimator.update(18'000'000LL, t += 7'000'000LL);
    estimator.update(2'000'000LL, t += 7'000'000LL);

    // Even with >= 5 samples, high jitter prevents is_stable from becoming true
    TEST_ASSERT(estimator.estimate().sample_count >= 5);
    TEST_ASSERT(!estimator.estimate().is_stable);

    std::cout << "  -> Stability gating passed." << std::endl;
}

void test_forward_extrapolation_beats_current_frame() {
    std::cout << "[Test 3] Forward extrapolation beats current-frame position..." << std::endl;

    MultiTargetTracker tracker{};
    const float dt = 1.0f / 144.0f;
    const std::int64_t dt_ns = static_cast<std::int64_t>(dt * 1e9f);
    const float vx = 400.0f; // Fast moving target at 400 px/s
    const float vy = 100.0f;
    const float latency_horizon_s = 0.015f; // 15 ms command-effect delay
    const std::int64_t latency_ns = static_cast<std::int64_t>(latency_horizon_s * 1e9f);

    float cx = 100.0f;
    float cy = 200.0f;
    std::int64_t t_ns = 1'000'000'000LL;
    bus::TrackedTargetBatch out{};

    // Warm up tracker and latency estimator
    for (int f = 0; f < 60; ++f) {
        cx += vx * dt;
        cy += vy * dt;
        t_ns += dt_ns;

        tracker.record_latency_sample(latency_ns, t_ns);

        bus::TargetObservationBatch batch{};
        batch.add_target(make_obs(cx, cy));
        tracker.process(batch, t_ns, t_ns + latency_ns, out);
    }

    TEST_ASSERT(out.track_count == 1);
    const auto& trk = out.tracks[0];

    // Ground truth position at effect time (t_ns + 15ms)
    const float true_effect_x = cx + vx * latency_horizon_s;
    const float true_effect_y = cy + vy * latency_horizon_s;

    // Error if we did NOT extrapolate (current frame error)
    const float raw_err_x = std::abs(trk.filtered_center_px.x - true_effect_x);
    const float raw_err_y = std::abs(trk.filtered_center_px.y - true_effect_y);
    const float raw_err = std::hypot(raw_err_x, raw_err_y);

    // Error with our forward extrapolation
    const float pred_err_x = std::abs(trk.predicted_center_px.x - true_effect_x);
    const float pred_err_y = std::abs(trk.predicted_center_px.y - true_effect_y);
    const float pred_err = std::hypot(pred_err_x, pred_err_y);

    std::cout << "  -> Current-frame error: " << raw_err << " px, Extrapolated error: " << pred_err << " px" << std::endl;

    // Raw error is approx 6.2 px (400 * 0.015 = 6.0 in X, 1.5 in Y -> sqrt(36 + 2.25) = 6.18)
    TEST_ASSERT(raw_err > 5.5f);
    // Extrapolated prediction error is < 0.6 px (10x better!)
    TEST_ASSERT(pred_err < 0.6f);
    TEST_ASSERT(pred_err < raw_err * 0.15f);

    std::cout << "  -> Prediction accuracy passed (>10x improvement over baseline)." << std::endl;
}

void test_acceleration_forward_extrapolation() {
    std::cout << "[Test 4] Acceleration forward kinematic extrapolation..." << std::endl;

    ExtrapolatorConfig config{};
    config.extrapolate_acceleration = true;
    TargetExtrapolator extrapolator{config};

    MotionState state{};
    state.x = 200.0f;
    state.y = 300.0f;
    state.vx = 200.0f;
    state.vy = 0.0f;
    state.ax = 500.0f; // Accelerating in X at 500 px/s^2
    state.ay = 0.0f;

    Covariance2D cov{1.0f, 0.0f, 1.0f};

    const MonotonicNs current_t = 1'000'000'000LL;
    const MonotonicNs effect_t = 1'020'000'000LL; // 20 ms horizon

    const auto res = extrapolator.extrapolate(state, cov, current_t, effect_t, true);

    // Expected X: 200 + 200 * 0.02 + 0.5 * 500 * (0.02)^2 = 200 + 4.0 + 0.10 = 204.10 px
    const float expected_x = 200.0f + 200.0f * 0.02f + 0.5f * 500.0f * 0.0004f;
    TEST_ASSERT(std::abs(res.predicted_center_px.x - expected_x) < 1e-4f);
    TEST_ASSERT(res.predicted_covariance_px2.xx > cov.xx); // Covariance expanded with horizon

    std::cout << "  -> Acceleration extrapolation passed." << std::endl;
}

void test_horizon_bounding_and_clamping() {
    std::cout << "[Test 5] Maximum extrapolation horizon bounding..." << std::endl;

    ExtrapolatorConfig config{};
    config.max_horizon_s = 0.040f; // 40 ms max limit
    TargetExtrapolator extrapolator{config};

    MotionState state{};
    state.x = 100.0f;
    state.vx = 1000.0f;

    Covariance2D cov{1.0f, 0.0f, 1.0f};

    const MonotonicNs current_t = 1'000'000'000LL;
    const MonotonicNs far_future_t = 2'000'000'000LL; // 1.0 second in future (excessive!)

    const auto res = extrapolator.extrapolate(state, cov, current_t, far_future_t, true);

    // Clamped to 40ms -> X increases by at most 1000 * 0.040 = 40px (not 1000px!)
    TEST_ASSERT(std::abs(res.predicted_center_px.x - 140.0f) < 1e-3f);
    TEST_ASSERT(res.horizon_s == 0.040f);

    std::cout << "  -> Horizon bounding passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M4-04 Latency & Prediction Extrapolation Tests" << std::endl;
    std::cout << "================================================================" << std::endl;

    test_latency_outlier_rejection_and_ewma();
    test_stability_gate_and_fail_closed();
    test_forward_extrapolation_beats_current_frame();
    test_acceleration_forward_extrapolation();
    test_horizon_bounding_and_clamping();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M4-04 Latency & Prediction Tests Passed!                   " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
