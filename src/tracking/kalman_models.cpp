// src/tracking/kalman_models.cpp
#include "aim/tracking/kalman_models.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace aim::tracking {

AdaptiveKalmanFilter::AdaptiveKalmanFilter(KalmanFilterConfig config)
    : config_(config) {
    reset();
}

void AdaptiveKalmanFilter::reset() noexcept {
    is_initialized_ = false;
    update_count_ = 0;
    last_dt_s_ = 1.0f / 144.0f;
    x_.fill(0.0f);
    p_.fill(0.0f);
    probabilities_ = ModelProbabilities{};
    last_mahalanobis_sq_ = 0.0f;
}

void AdaptiveKalmanFilter::initialize(float x, float y, const Covariance2D& initial_cov) noexcept {
    reset();
    x_[0] = x;
    x_[1] = y;
    x_[2] = 0.0f; // vx
    x_[3] = 0.0f; // vy
    x_[4] = 0.0f; // ax
    x_[5] = 0.0f; // ay

    // Initial covariance
    p_.fill(0.0f);
    p_[0 * 6 + 0] = std::max(config_.min_covariance, initial_cov.xx);
    p_[0 * 6 + 1] = initial_cov.xy;
    p_[1 * 6 + 0] = initial_cov.xy;
    p_[1 * 6 + 1] = std::max(config_.min_covariance, initial_cov.yy);

    // Initial velocity & acceleration uncertainty
    p_[2 * 6 + 2] = 100000.0f; // vx variance (~316 px/s std dev)
    p_[3 * 6 + 3] = 100000.0f; // vy variance
    p_[4 * 6 + 4] = 500000.0f; // ax variance
    p_[5 * 6 + 5] = 500000.0f; // ay variance

    is_initialized_ = true;
    update_count_ = 0;
}

void AdaptiveKalmanFilter::predict(float dt_s) noexcept {
    if (!is_initialized_ || dt_s <= 0.0f) {
        return;
    }

    last_dt_s_ = dt_s;
    const float dt = dt_s;
    const float dt2 = 0.5f * dt * dt;

    // 1. State prediction: x = F * x
    const float px = x_[0] + dt * x_[2] + dt2 * x_[4];
    const float py = x_[1] + dt * x_[3] + dt2 * x_[5];
    const float vx = x_[2] + dt * x_[4];
    const float vy = x_[3] + dt * x_[5];
    const float ax = x_[4];
    const float ay = x_[5];

    x_[0] = px;
    x_[1] = py;
    x_[2] = vx;
    x_[3] = vy;
    x_[4] = ax;
    x_[5] = ay;

    // 2. Covariance prediction: P = F * P * F^T + Q
    // We compute F * P
    std::array<float, 36> fp{};
    for (std::size_t col = 0; col < 6; ++col) {
        fp[0 * 6 + col] = p_[0 * 6 + col] + dt * p_[2 * 6 + col] + dt2 * p_[4 * 6 + col];
        fp[1 * 6 + col] = p_[1 * 6 + col] + dt * p_[3 * 6 + col] + dt2 * p_[5 * 6 + col];
        fp[2 * 6 + col] = p_[2 * 6 + col] + dt * p_[4 * 6 + col];
        fp[3 * 6 + col] = p_[3 * 6 + col] + dt * p_[5 * 6 + col];
        fp[4 * 6 + col] = p_[4 * 6 + col];
        fp[5 * 6 + col] = p_[5 * 6 + col];
    }

    // P_new = (FP) * F^T
    for (std::size_t row = 0; row < 6; ++row) {
        p_[row * 6 + 0] = fp[row * 6 + 0] + dt * fp[row * 6 + 2] + dt2 * fp[row * 6 + 4];
        p_[row * 6 + 1] = fp[row * 6 + 1] + dt * fp[row * 6 + 3] + dt2 * fp[row * 6 + 5];
        p_[row * 6 + 2] = fp[row * 6 + 2] + dt * fp[row * 6 + 4];
        p_[row * 6 + 3] = fp[row * 6 + 3] + dt * fp[row * 6 + 5];
        p_[row * 6 + 4] = fp[row * 6 + 4];
        p_[row * 6 + 5] = fp[row * 6 + 5];
    }

    // Add Process Noise Q(dt)
    const float qp = config_.q_pos * dt;
    const float qv = config_.q_vel * dt;
    const float qa = config_.q_acc * dt;

    p_[0 * 6 + 0] += qp;
    p_[1 * 6 + 1] += qp;
    p_[2 * 6 + 2] += qv;
    p_[3 * 6 + 3] += qv;
    p_[4 * 6 + 4] += qa;
    p_[5 * 6 + 5] += qa;

    enforce_covariance_stability();
}

float AdaptiveKalmanFilter::update(float z_x, float z_y, const Covariance2D& meas_cov) noexcept {
    if (!is_initialized_) {
        initialize(z_x, z_y, meas_cov);
        return 0.0f;
    }

    if (update_count_ == 0 && last_dt_s_ > 0.0f) {
        x_[2] = (z_x - x_[0]) / last_dt_s_;
        x_[3] = (z_y - x_[1]) / last_dt_s_;
        x_[0] = z_x;
        x_[1] = z_y;
        update_count_ = 1;
        enforce_covariance_stability();
        return 0.0f;
    }
    update_count_++;

    // Residual innovation: y = z - H * x
    const float y0 = z_x - x_[0];
    const float y1 = z_y - x_[1];

    // Measurement covariance R
    const float r00 = std::max(config_.min_covariance, meas_cov.xx > 0.0f ? meas_cov.xx : config_.r_default);
    const float r01 = meas_cov.xy;
    const float r10 = meas_cov.xy;
    const float r11 = std::max(config_.min_covariance, meas_cov.yy > 0.0f ? meas_cov.yy : config_.r_default);

    // Innovation covariance S = H * P * H^T + R (2x2)
    const float s00 = p_[0 * 6 + 0] + r00;
    const float s01 = p_[0 * 6 + 1] + r01;
    const float s10 = p_[1 * 6 + 0] + r10;
    const float s11 = p_[1 * 6 + 1] + r11;

    // Invert 2x2 matrix S
    const float det = s00 * s11 - s01 * s10;
    if (std::abs(det) < 1e-9f) {
        return 0.0f;
    }
    const float inv_det = 1.0f / det;
    const float inv_s00 =  s11 * inv_det;
    const float inv_s01 = -s01 * inv_det;
    const float inv_s10 = -s10 * inv_det;
    const float inv_s11 =  s00 * inv_det;

    // Mahalanobis distance squared: d_M^2 = y^T * S^-1 * y
    const float d_m_sq = y0 * (inv_s00 * y0 + inv_s01 * y1) + y1 * (inv_s10 * y0 + inv_s11 * y1);
    last_mahalanobis_sq_ = std::max(0.0f, d_m_sq);

    // Kalman Gain K = P * H^T * S^-1 (6x2)
    // H^T has 1 at (0,0) and (1,1)
    std::array<float, 12> k{}; // 6x2
    for (std::size_t row = 0; row < 6; ++row) {
        const float p_h0 = p_[row * 6 + 0];
        const float p_h1 = p_[row * 6 + 1];
        k[row * 2 + 0] = p_h0 * inv_s00 + p_h1 * inv_s10;
        k[row * 2 + 1] = p_h0 * inv_s01 + p_h1 * inv_s11;
    }

    // State update: x = x + K * y
    for (std::size_t row = 0; row < 6; ++row) {
        x_[row] += k[row * 2 + 0] * y0 + k[row * 2 + 1] * y1;
    }

    // Joseph-form stabilized covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
    // I_KH = I - K*H (6x6)
    std::array<float, 36> i_kh{};
    for (std::size_t r = 0; r < 6; ++r) {
        for (std::size_t c = 0; c < 6; ++c) {
            float val = (r == c) ? 1.0f : 0.0f;
            if (c == 0) val -= k[r * 2 + 0];
            if (c == 1) val -= k[r * 2 + 1];
            i_kh[r * 6 + c] = val;
        }
    }

    // temp = (I - K*H) * P
    std::array<float, 36> temp{};
    for (std::size_t r = 0; r < 6; ++r) {
        for (std::size_t c = 0; c < 6; ++c) {
            float sum = 0.0f;
            for (std::size_t k_idx = 0; k_idx < 6; ++k_idx) {
                sum += i_kh[r * 6 + k_idx] * p_[k_idx * 6 + c];
            }
            temp[r * 6 + c] = sum;
        }
    }

    // P = temp * (I - K*H)^T + K * R * K^T
    for (std::size_t r = 0; r < 6; ++r) {
        for (std::size_t c = 0; c < 6; ++c) {
            float sum = 0.0f;
            for (std::size_t k_idx = 0; k_idx < 6; ++k_idx) {
                sum += temp[r * 6 + k_idx] * i_kh[c * 6 + k_idx];
            }

            // Add K * R * K^T term
            const float kr0 = k[r * 2 + 0] * r00 + k[r * 2 + 1] * r10;
            const float kr1 = k[r * 2 + 0] * r01 + k[r * 2 + 1] * r11;
            sum += kr0 * k[c * 2 + 0] + kr1 * k[c * 2 + 1];

            p_[r * 6 + c] = sum;
        }
    }

    enforce_covariance_stability();
    update_model_probabilities(y0 * y0 + y1 * y1, x_[2] * x_[2] + x_[3] * x_[3], x_[4] * x_[4] + x_[5] * x_[5]);

    return last_mahalanobis_sq_;
}

void AdaptiveKalmanFilter::enforce_covariance_stability() noexcept {
    // Symmetrize and clamp diagonal variance
    for (std::size_t r = 0; r < 6; ++r) {
        for (std::size_t c = r; c < 6; ++c) {
            float avg = 0.5f * (p_[r * 6 + c] + p_[c * 6 + r]);
            p_[r * 6 + c] = avg;
            p_[c * 6 + r] = avg;
        }
    }
    for (std::size_t r = 0; r < 6; ++r) {
        p_[r * 6 + r] = std::clamp(p_[r * 6 + r], config_.min_covariance, config_.max_covariance);
    }
}

void AdaptiveKalmanFilter::update_model_probabilities(
    [[maybe_unused]] float residual_sq_pos, float speed_sq, float acc_sq) noexcept {
    // Adaptive model weighting: stationary when speed is low, CA when acceleration is high
    const float speed = std::sqrt(speed_sq);
    const float acc = std::sqrt(acc_sq);

    float l_stat = std::exp(-0.5f * (speed / 10.0f));
    float l_ca = 1.0f - std::exp(-0.5f * (acc / 100.0f));
    float l_cv = 1.0f;

    float total = l_stat + l_cv + l_ca + 1e-6f;
    probabilities_.prob_stationary = l_stat / total;
    probabilities_.prob_cv = l_cv / total;
    probabilities_.prob_ca = l_ca / total;
}

MotionState AdaptiveKalmanFilter::extrapolate(float horizon_s) const noexcept {
    if (!is_initialized_ || horizon_s <= 0.0f) {
        return state();
    }
    const float dt = horizon_s;
    const float dt2 = 0.5f * dt * dt;

    MotionState s{};
    s.x = x_[0] + dt * x_[2] + dt2 * x_[4];
    s.y = x_[1] + dt * x_[3] + dt2 * x_[5];
    s.vx = x_[2] + dt * x_[4];
    s.vy = x_[3] + dt * x_[5];
    s.ax = x_[4];
    s.ay = x_[5];
    return s;
}

MotionState AdaptiveKalmanFilter::state() const noexcept {
    MotionState s{};
    s.x = x_[0];
    s.y = x_[1];
    s.vx = x_[2];
    s.vy = x_[3];
    s.ax = x_[4];
    s.ay = x_[5];
    return s;
}

Covariance2D AdaptiveKalmanFilter::position_covariance() const noexcept {
    return Covariance2D{p_[0 * 6 + 0], p_[0 * 6 + 1], p_[1 * 6 + 1]};
}

Covariance2D AdaptiveKalmanFilter::velocity_covariance() const noexcept {
    return Covariance2D{p_[2 * 6 + 2], p_[2 * 6 + 3], p_[3 * 6 + 3]};
}

} // namespace aim::tracking
