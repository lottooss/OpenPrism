// src/calibration/calibration_model.cpp
#include "aim/calibration/calibration_model.hpp"

#include <limits>

namespace aim::calibration {
namespace {

bool finite_in_range(float value, float minimum, float maximum) noexcept {
    return std::isfinite(value) && value >= minimum && value <= maximum;
}

bool within_frame(double dx, double dy, const CalibrationProfile& profile) noexcept {
    return std::isfinite(dx) && std::isfinite(dy) &&
           std::abs(dx) <= static_cast<double>(profile.resolution_width) &&
           std::abs(dy) <= static_cast<double>(profile.resolution_height);
}

bool round_counts(double value, std::int32_t& output) noexcept {
    const double rounded = std::round(value);
    if (!std::isfinite(rounded) ||
        rounded < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        return false;
    }
    output = static_cast<std::int32_t>(rounded);
    return true;
}

} // namespace

CalibrationModel::CalibrationModel(const CalibrationProfile& profile) noexcept
    : profile_(profile) {}

bool CalibrationModel::set_profile(const CalibrationProfile& profile) noexcept {
    profile_ = profile;
    return is_valid();
}

bool CalibrationModel::is_valid() const noexcept {
    // Numeric context bounds match the versioned calibration-profile manifest.
    return profile_.resolution_width >= 320 && profile_.resolution_width <= 7680 &&
           profile_.resolution_height >= 240 && profile_.resolution_height <= 4320 &&
           finite_in_range(profile_.counts_per_pixel_x, 0.001f, 100.0f) &&
           finite_in_range(profile_.counts_per_pixel_y, 0.001f, 100.0f) &&
           finite_in_range(profile_.fov_horizontal_deg, 30.0f, 150.0f) &&
           finite_in_range(profile_.in_game_sensitivity, 0.001f, 100.0f) &&
           finite_in_range(profile_.deadband_counts, 0.0f, 10.0f) &&
           finite_in_range(profile_.nonlinearity_alpha, -1.0f, 1.0f) &&
           finite_in_range(profile_.cross_coupling_xy, -0.5f, 0.5f) &&
           finite_in_range(profile_.rmse_pixels, 0.0f, 50.0f);
}

MouseCounts CalibrationModel::pixels_to_counts(float dx_px, float dy_px) const noexcept {
    MouseCounts result{};
    (void)try_pixels_to_counts(dx_px, dy_px, result);
    return result;
}

bool CalibrationModel::try_pixels_to_counts(float dx_px, float dy_px, MouseCounts& output) const noexcept {
    output = {};
    if (!is_valid() || !within_frame(dx_px, dy_px, profile_)) {
        return false;
    }

    const double dx = dx_px;
    const double dy = dy_px;
    const double coupling = profile_.cross_coupling_xy;
    const double dist = std::hypot(dx, dy);
    const double diag = std::hypot(static_cast<double>(profile_.resolution_width),
                                   static_cast<double>(profile_.resolution_height));
    const double radial_term = static_cast<double>(profile_.nonlinearity_alpha) * dist / diag;
    // r -> r * (1 + alpha*r/diag) must remain strictly increasing. Beyond
    // this turning point a negative alpha produces an ambiguous inverse.
    if (!std::isfinite(radial_term) || 1.0 + 2.0 * radial_term <= 0.0) {
        return false;
    }
    const double scale = 1.0 + radial_term;
    double raw_x = (dx - coupling * dy) * profile_.counts_per_pixel_x * scale;
    double raw_y = (dy - coupling * dx) * profile_.counts_per_pixel_y * scale;
    if (std::abs(raw_x) > 0.1) {
        raw_x += std::copysign(static_cast<double>(profile_.deadband_counts), raw_x);
    }
    if (std::abs(raw_y) > 0.1) {
        raw_y += std::copysign(static_cast<double>(profile_.deadband_counts), raw_y);
    }

    MouseCounts converted{};
    if (!round_counts(raw_x, converted.counts_x) || !round_counts(raw_y, converted.counts_y)) {
        return false;
    }
    output = converted;
    return true;
}

PixelDisplacement CalibrationModel::counts_to_pixels(std::int32_t counts_x,
                                                     std::int32_t counts_y) const noexcept {
    PixelDisplacement result{};
    (void)try_counts_to_pixels(counts_x, counts_y, result);
    return result;
}

bool CalibrationModel::try_counts_to_pixels(std::int32_t counts_x, std::int32_t counts_y,
                                            PixelDisplacement& output) const noexcept {
    output = {};
    if (!is_valid()) {
        return false;
    }

    const auto remove_deadband = [this](std::int32_t counts) noexcept {
        const double raw = static_cast<double>(counts);
        return std::abs(raw) > profile_.deadband_counts
            ? raw - std::copysign(static_cast<double>(profile_.deadband_counts), raw)
            : 0.0;
    };
    const double raw_x = remove_deadband(counts_x) / profile_.counts_per_pixel_x;
    const double raw_y = remove_deadband(counts_y) / profile_.counts_per_pixel_y;
    const double coupling = profile_.cross_coupling_xy;
    // Invert the cross-axis matrix exactly, including its determinant.
    const double determinant = 1.0 - coupling * coupling;
    const double scaled_x = (raw_x + coupling * raw_y) / determinant;
    const double scaled_y = (raw_y + coupling * raw_x) / determinant;
    const double scaled_radius = std::hypot(scaled_x, scaled_y);
    const double diag = std::hypot(static_cast<double>(profile_.resolution_width),
                                   static_cast<double>(profile_.resolution_height));
    const double discriminant = 1.0 + 4.0 * profile_.nonlinearity_alpha * scaled_radius / diag;
    if (!std::isfinite(discriminant) || discriminant <= 0.0) {
        return false;
    }
    // Stable quadratic inversion, continuous at alpha = 0 and radius = 0.
    const double scale = 0.5 * (1.0 + std::sqrt(discriminant));
    const double dx = scaled_x / scale;
    const double dy = scaled_y / scale;
    if (!within_frame(dx, dy, profile_)) {
        return false;
    }
    output = PixelDisplacement{static_cast<float>(dx), static_cast<float>(dy)};
    return true;
}

} // namespace aim::calibration
