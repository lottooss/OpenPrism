// src/calibration/control_calibration.cpp
#include "aim/calibration/control_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aim::calibration {
namespace {

bool valid_configuration(const OnlineCalibrationConfig& config) noexcept {
    return std::isfinite(config.initial_cpi) && config.initial_cpi > 10.0f &&
           std::isfinite(config.initial_fov_deg) && config.initial_fov_deg >= 30.0f &&
           config.initial_fov_deg <= 150.0f &&
           std::isfinite(config.initial_counts_per_pixel) && config.initial_counts_per_pixel >= 0.001f &&
           config.initial_counts_per_pixel <= 100.0f &&
           std::isfinite(config.max_drift_fraction) && config.max_drift_fraction >= 0.0f &&
           config.max_drift_fraction < 1.0f &&
           std::isfinite(config.max_step_delta) && config.max_step_delta >= 0.0f &&
           std::isfinite(config.min_magnitude_px) && config.min_magnitude_px > 0.0f &&
           std::isfinite(config.outlier_ratio_threshold) && config.outlier_ratio_threshold >= 0.0f &&
           config.outlier_ratio_threshold < 1.0f &&
           std::isfinite(config.learning_rate) && config.learning_rate >= 0.0f && config.learning_rate <= 1.0f &&
           static_cast<double>(config.initial_cpi) * (1.0 + config.max_drift_fraction) <=
               static_cast<double>(std::numeric_limits<float>::max());
}

} // namespace

ControlCalibration::ControlCalibration(const OnlineCalibrationConfig& config) noexcept
    : config_(config) {
    (void)initialize(config.initial_cpi, config.initial_fov_deg);
}

bool ControlCalibration::initialize(float initial_cpi, float fov_deg) noexcept {
    config_.initial_cpi = initial_cpi;
    config_.initial_fov_deg = fov_deg;
    if (!valid_configuration(config_)) {
        initialized_ = false;
        baseline_counts_per_pixel_ = 0.0f;
        baseline_cpi_ = 0.0f;
        reset();
        return false;
    }
    baseline_cpi_ = initial_cpi;
    current_cpi_ = initial_cpi;
    baseline_counts_per_pixel_ = config_.initial_counts_per_pixel;
    current_counts_per_pixel_ = config_.initial_counts_per_pixel;
    successful_updates_ = 0;
    rejected_outliers_ = 0;
    initialized_ = true;
    return true;
}

bool ControlCalibration::update_feedback(PixelPoint expected_delta, PixelPoint observed_delta) noexcept {
    if (!initialized_) {
        return false;
    }

    if (!std::isfinite(expected_delta.x) || !std::isfinite(expected_delta.y) ||
        !std::isfinite(observed_delta.x) || !std::isfinite(observed_delta.y)) {
        ++rejected_outliers_;
        return false;
    }
    // Widen before multiplication so finite float feedback cannot overflow the
    // magnitude or dot product and poison an otherwise bounded estimate.
    const double exp_mag = std::hypot(static_cast<double>(expected_delta.x), static_cast<double>(expected_delta.y));
    const double obs_mag = std::hypot(static_cast<double>(observed_delta.x), static_cast<double>(observed_delta.y));

    // Gate 1: Check minimum magnitude for reliable attribution
    if (exp_mag < config_.min_magnitude_px || obs_mag < 0.5f) {
        ++rejected_outliers_;
        return false;
    }

    // Gate 2: Directional sign agreement (dot product must be positive)
    const double dot = static_cast<double>(expected_delta.x) * observed_delta.x +
                       static_cast<double>(expected_delta.y) * observed_delta.y;
    if (dot <= 0.0f) {
        ++rejected_outliers_;
        return false;
    }

    // Gate 3: Relative error outlier rejection
    const double rel_diff = std::abs(obs_mag - exp_mag) / exp_mag;
    if (rel_diff > config_.outlier_ratio_threshold) {
        ++rejected_outliers_;
        return false;
    }

    // Target counts-per-pixel ratio from observation
    // If observed < expected, we moved less than expected -> true counts_per_pixel is higher
    const double ratio = exp_mag / obs_mag;
    const double target_cpp = baseline_counts_per_pixel_ * ratio;

    // Bounded step adjustment
    const double delta = (target_cpp - current_counts_per_pixel_) * config_.learning_rate;
    const double clamped_delta = std::clamp(delta, -static_cast<double>(config_.max_step_delta),
                                          static_cast<double>(config_.max_step_delta));
    double new_cpp = current_counts_per_pixel_ + clamped_delta;

    // Bounded overall drift from initial baseline
    const double min_allowed = baseline_counts_per_pixel_ * (1.0 - config_.max_drift_fraction);
    const double max_allowed = baseline_counts_per_pixel_ * (1.0 + config_.max_drift_fraction);
    new_cpp = std::clamp(new_cpp, min_allowed, max_allowed);

    const double new_cpi = baseline_cpi_ * (new_cpp / baseline_counts_per_pixel_);
    if (!std::isfinite(new_cpp) || !std::isfinite(new_cpi) || new_cpp <= 0.0 || new_cpi <= 0.0 ||
        new_cpp > std::numeric_limits<float>::max() || new_cpi > std::numeric_limits<float>::max()) {
        ++rejected_outliers_;
        return false;
    }
    current_counts_per_pixel_ = static_cast<float>(new_cpp);
    current_cpi_ = static_cast<float>(new_cpi);
    ++successful_updates_;
    return true;
}

float ControlCalibration::current_cpi() const noexcept {
    return current_cpi_;
}

float ControlCalibration::counts_per_pixel() const noexcept {
    return current_counts_per_pixel_;
}

void ControlCalibration::reset() noexcept {
    current_counts_per_pixel_ = baseline_counts_per_pixel_;
    current_cpi_ = baseline_cpi_;
    successful_updates_ = 0;
    rejected_outliers_ = 0;
}

void ControlCalibration::rollback_to_baseline() noexcept {
    reset();
}

} // namespace aim::calibration
