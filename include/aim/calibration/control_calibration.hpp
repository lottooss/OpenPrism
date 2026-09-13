// include/aim/calibration/control_calibration.hpp
#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>

#include "aim/interfaces/control_calibration.hpp"

namespace aim::calibration {

struct OnlineCalibrationConfig {
    float initial_cpi{800.0f};
    float initial_fov_deg{103.0f};
    float initial_counts_per_pixel{1.25f};
    float max_drift_fraction{0.25f};       // Maximum +/- 25% drift from baseline
    float max_step_delta{0.015f};          // Strict bounded step change per update
    float min_magnitude_px{3.0f};          // Minimum motion required for attributable update
    float outlier_ratio_threshold{0.35f};  // Discard feedback deviating >35% from expected
    float learning_rate{0.02f};            // Slow, conservative exponential smoothing
};

/// @brief Bounded online visual-response calibration (Milestone M6-04).
/// Updates counts-per-pixel only when attributable, high-confidence feedback is observed.
class ControlCalibration final : public IControlCalibration {
public:
    explicit ControlCalibration(const OnlineCalibrationConfig& config = {}) noexcept;

    bool initialize(float initial_cpi, float fov_deg) noexcept override;
    bool update_feedback(PixelPoint expected_delta, PixelPoint observed_delta) noexcept override;

    [[nodiscard]] float current_cpi() const noexcept override;
    [[nodiscard]] float counts_per_pixel() const noexcept override;
    void reset() noexcept override;

    /// @brief Rollback active parameter to initial baseline
    void rollback_to_baseline() noexcept;

    [[nodiscard]] std::uint64_t successful_updates() const noexcept { return successful_updates_; }
    [[nodiscard]] std::uint64_t rejected_outliers() const noexcept { return rejected_outliers_; }

private:
    OnlineCalibrationConfig config_{};
    float baseline_counts_per_pixel_{1.25f};
    float baseline_cpi_{800.0f};
    float current_counts_per_pixel_{1.25f};
    float current_cpi_{800.0f};
    std::uint64_t successful_updates_{0};
    std::uint64_t rejected_outliers_{0};
    bool initialized_{false};
};

} // namespace aim::calibration
