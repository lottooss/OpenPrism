// include/aim/tracking/kalman_models.hpp
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim::tracking {

enum class MotionModelType : std::uint8_t {
    Stationary = 0,
    ConstantVelocity = 1,
    ConstantAcceleration = 2
};

struct MotionState {
    float x{0.0f};
    float y{0.0f};
    float vx{0.0f};
    float vy{0.0f};
    float ax{0.0f};
    float ay{0.0f};
};

struct ModelProbabilities {
    float prob_stationary{0.3333f};
    float prob_cv{0.3333f};
    float prob_ca{0.3334f};
};

struct KalmanFilterConfig {
    float q_pos{10.0f};          // Position process noise variance
    float q_vel{2000.0f};        // Velocity process noise variance
    float q_acc{10000.0f};       // Acceleration process noise variance
    float r_default{2.0f};       // Default measurement noise variance (px^2)
    float max_covariance{1e6f};
    float min_covariance{1e-4f};
};

/// @brief Numerically stable 6D (pos, vel, acc) Adaptive Kalman Filter with Joseph-form covariance updates.
/// Allocation-free fixed storage.
class AdaptiveKalmanFilter {
public:
    explicit AdaptiveKalmanFilter(KalmanFilterConfig config = {});

    /// @brief Initialize track state with initial position measurement.
    void initialize(float x, float y, const Covariance2D& initial_cov) noexcept;

    /// @brief Predict state forward by delta_time_s.
    void predict(float dt_s) noexcept;

    /// @brief Update state with 2D position measurement and measurement noise covariance R.
    /// Returns Mahalanobis distance squared (innovation d_M^2).
    float update(float z_x, float z_y, const Covariance2D& meas_cov) noexcept;

    /// @brief Extrapolate state forward by prediction_horizon_s without mutating current filter state.
    [[nodiscard]] MotionState extrapolate(float horizon_s) const noexcept;

    [[nodiscard]] MotionState state() const noexcept;
    [[nodiscard]] Covariance2D position_covariance() const noexcept;
    [[nodiscard]] Covariance2D velocity_covariance() const noexcept;
    [[nodiscard]] ModelProbabilities model_probabilities() const noexcept { return probabilities_; }
    [[nodiscard]] float last_mahalanobis_sq() const noexcept { return last_mahalanobis_sq_; }

    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }
    void reset() noexcept;

private:
    void enforce_covariance_stability() noexcept;
    void update_model_probabilities(float residual_sq_pos, float residual_sq_vel, float residual_sq_acc) noexcept;

    KalmanFilterConfig config_;
    bool is_initialized_{false};
    std::uint32_t update_count_{0};
    float last_dt_s_{1.0f / 144.0f};

    // State vector: [x, y, vx, vy, ax, ay]^T
    std::array<float, 6> x_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Covariance matrix 6x6 (Row-major)
    std::array<float, 36> p_{};

    ModelProbabilities probabilities_{};
    float last_mahalanobis_sq_{0.0f};
};

} // namespace aim::tracking
