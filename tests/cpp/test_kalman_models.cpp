// tests/cpp/test_kalman_models.cpp
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "aim/tracking/kalman_models.hpp"

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

using namespace aim;
using namespace aim::tracking;

void test_stationary_target_convergence() {
    AdaptiveKalmanFilter kf;
    Covariance2D init_cov{10.0f, 0.0f, 10.0f};
    kf.initialize(960.0f, 540.0f, init_cov);

    // Simulate 20 frames at 144Hz of a stationary target with small sensor noise
    const float dt = 1.0f / 144.0f;
    for (int i = 0; i < 20; ++i) {
        kf.predict(dt);
        Covariance2D meas_cov{2.0f, 0.0f, 2.0f};
        float d_m_sq = kf.update(960.0f, 540.0f, meas_cov);
        TEST_ASSERT(d_m_sq >= 0.0f);
    }

    auto s = kf.state();
    TEST_ASSERT(std::abs(s.x - 960.0f) < 0.1f);
    TEST_ASSERT(std::abs(s.y - 540.0f) < 0.1f);
    TEST_ASSERT(std::abs(s.vx) < 1.0f);
    TEST_ASSERT(std::abs(s.vy) < 1.0f);

    auto pos_cov = kf.position_covariance();
    TEST_ASSERT(pos_cov.xx < init_cov.xx); // Uncertainty reduced
    TEST_ASSERT(pos_cov.yy < init_cov.yy);

    std::cout << "[PASS] test_stationary_target_convergence\n";
}

void test_constant_velocity_tracking_and_extrapolation() {
    AdaptiveKalmanFilter kf;
    Covariance2D init_cov{5.0f, 0.0f, 5.0f};
    kf.initialize(100.0f, 200.0f, init_cov);

    const float vx_true = 300.0f; // 300 px/s
    const float vy_true = -150.0f; // -150 px/s
    const float dt = 1.0f / 144.0f;

    float current_x = 100.0f;
    float current_y = 200.0f;

    for (int i = 0; i < 50; ++i) {
        current_x += vx_true * dt;
        current_y += vy_true * dt;

        kf.predict(dt);
        Covariance2D meas_cov{1.0f, 0.0f, 1.0f};
        kf.update(current_x, current_y, meas_cov);
    }

    auto s = kf.state();
    TEST_ASSERT(std::abs(s.x - current_x) < 0.5f);
    TEST_ASSERT(std::abs(s.y - current_y) < 0.5f);
    TEST_ASSERT(std::abs(s.vx - vx_true) < 10.0f);
    TEST_ASSERT(std::abs(s.vy - vy_true) < 10.0f);

    // Test forward extrapolation 20ms into the future
    const float horizon = 0.020f;
    auto extrapolated = kf.extrapolate(horizon);
    float expected_extrap_x = current_x + vx_true * horizon;
    float expected_extrap_y = current_y + vy_true * horizon;

    TEST_ASSERT(std::abs(extrapolated.x - expected_extrap_x) < 1.0f);
    TEST_ASSERT(std::abs(extrapolated.y - expected_extrap_y) < 1.0f);

    std::cout << "[PASS] test_constant_velocity_tracking_and_extrapolation\n";
}

void test_maneuver_reversal_and_covariance_boundedness() {
    AdaptiveKalmanFilter kf;
    kf.initialize(500.0f, 500.0f, {5.0f, 0.0f, 5.0f});

    const float dt = 1.0f / 144.0f;
    float x = 500.0f;
    float vx = 400.0f;

    // Moving right
    for (int i = 0; i < 30; ++i) {
        x += vx * dt;
        kf.predict(dt);
        kf.update(x, 500.0f, {1.0f, 0.0f, 1.0f});
    }

    // Abrupt reversal: moving left
    vx = -400.0f;
    for (int i = 0; i < 30; ++i) {
        x += vx * dt;
        kf.predict(dt);
        kf.update(x, 500.0f, {1.0f, 0.0f, 1.0f});
    }

    auto pos_cov = kf.position_covariance();
    TEST_ASSERT(pos_cov.xx > 0.0f && pos_cov.xx < 1000.0f);
    TEST_ASSERT(pos_cov.yy > 0.0f && pos_cov.yy < 1000.0f);
    TEST_ASSERT(!std::isnan(pos_cov.xx) && !std::isnan(pos_cov.yy));

    std::cout << "[PASS] test_maneuver_reversal_and_covariance_boundedness\n";
}

int main() {
    std::cout << "Running Adaptive Kalman Models C++ Unit Tests...\n";
    test_stationary_target_convergence();
    test_constant_velocity_tracking_and_extrapolation();
    test_maneuver_reversal_and_covariance_boundedness();
    std::cout << "All Adaptive Kalman Models C++ Unit Tests Passed Successfully!\n";
    return 0;
}
