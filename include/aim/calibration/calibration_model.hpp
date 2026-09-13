// include/aim/calibration/calibration_model.hpp
#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#include "aim/core/types.hpp"

namespace aim::calibration {

struct CalibrationProfile {
    std::string profile_id{"default_1080p"};
    std::uint32_t resolution_width{1920};
    std::uint32_t resolution_height{1080};
    float fov_horizontal_deg{103.0f};
    float in_game_sensitivity{1.0f};
    float counts_per_pixel_x{1.0f};
    float counts_per_pixel_y{1.0f};
    float deadband_counts{0.0f};
    float nonlinearity_alpha{0.0f};
    float cross_coupling_xy{0.0f};
    float rmse_pixels{0.0f};
};

struct MouseCounts {
    std::int32_t counts_x{0};
    std::int32_t counts_y{0};
};

struct PixelDisplacement {
    float dx_px{0.0f};
    float dy_px{0.0f};
};

/// @brief Visual counts-to-pixels and pixels-to-counts calibration model (Milestone M6-03).
class CalibrationModel {
public:
    explicit CalibrationModel(const CalibrationProfile& profile = {}) noexcept;

    bool set_profile(const CalibrationProfile& profile) noexcept;
    [[nodiscard]] const CalibrationProfile& profile() const noexcept { return profile_; }

    /// @brief Convert desired pixel delta to hardware mouse counts
    [[nodiscard]] MouseCounts pixels_to_counts(float dx_px, float dy_px) const noexcept;

    /// @brief Convert dispatched mouse counts to expected visual pixel displacement
    [[nodiscard]] PixelDisplacement counts_to_pixels(std::int32_t counts_x, std::int32_t counts_y) const noexcept;

    /// Checked conversions for dispatch callers. Failure zeroes output and must
    /// cancel actuation. Inputs must be finite, within the profile's full-frame
    /// displacement range, and on the invertible branch of the radial model.
    [[nodiscard]] bool try_pixels_to_counts(float dx_px, float dy_px, MouseCounts& output) const noexcept;
    [[nodiscard]] bool try_counts_to_pixels(std::int32_t counts_x, std::int32_t counts_y,
                                           PixelDisplacement& output) const noexcept;

    /// @brief Verify calibration validity (fails closed if gains are uncalibrated or out of bounds)
    [[nodiscard]] bool is_valid() const noexcept;

private:
    CalibrationProfile profile_{};
};

} // namespace aim::calibration
