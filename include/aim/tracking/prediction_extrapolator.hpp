// include/aim/tracking/prediction_extrapolator.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/tracking/kalman_models.hpp"
#include "aim/tracking/latency_estimator.hpp"

namespace aim::tracking {

struct ExtrapolatorConfig {
    float max_horizon_s{0.050f};         // 50 ms maximum extrapolation horizon
    float position_process_noise{15.0f}; // Covariance expansion rate (px^2 / s)
    bool extrapolate_acceleration{true}; // Include second-order acceleration term
};

struct ExtrapolatedTargetState {
    PixelPoint predicted_center_px{};
    Covariance2D predicted_covariance_px2{};
    MonotonicNs target_time_ns{0};
    float horizon_s{0.0f};
    bool is_valid{false};
};

class TargetExtrapolator {
public:
    explicit TargetExtrapolator(ExtrapolatorConfig config = {})
        : config_(config) {}

    /// @brief Extrapolate a motion state to a target timestamp.
    [[nodiscard]] ExtrapolatedTargetState extrapolate(
        const MotionState& state,
        const Covariance2D& base_cov,
        MonotonicNs current_time_ns,
        MonotonicNs target_time_ns,
        bool is_stable_estimate = true) const noexcept {
        ExtrapolatedTargetState result{};
        result.target_time_ns = target_time_ns;

        if (!is_stable_estimate || target_time_ns <= current_time_ns) {
            // Fail closed: return current measured state with no forward projection
            result.predicted_center_px = PixelPoint{state.x, state.y};
            result.predicted_covariance_px2 = base_cov;
            result.horizon_s = 0.0f;
            result.is_valid = is_stable_estimate;
            return result;
        }

        const MonotonicNs diff_ns = target_time_ns - current_time_ns;
        const float horizon_s = std::clamp(
            static_cast<float>(diff_ns) * 1e-9f,
            0.0f,
            config_.max_horizon_s
        );

        result.horizon_s = horizon_s;

        // Kinematic forward projection
        float pred_x = state.x + state.vx * horizon_s;
        float pred_y = state.y + state.vy * horizon_s;

        if (config_.extrapolate_acceleration) {
            const float half_dt2 = 0.5f * horizon_s * horizon_s;
            pred_x += state.ax * half_dt2;
            pred_y += state.ay * half_dt2;
        }

        result.predicted_center_px = PixelPoint{pred_x, pred_y};

        // Covariance expansion
        const float cov_growth = config_.position_process_noise * horizon_s;
        result.predicted_covariance_px2 = Covariance2D{
            base_cov.xx + cov_growth,
            base_cov.xy,
            base_cov.yy + cov_growth
        };

        result.is_valid = true;
        return result;
    }

    [[nodiscard]] const ExtrapolatorConfig& config() const noexcept { return config_; }

private:
    ExtrapolatorConfig config_;
};

} // namespace aim::tracking
