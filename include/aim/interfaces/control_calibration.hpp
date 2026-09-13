// include/aim/interfaces/control_calibration.hpp
#pragma once

#include "aim/core/types.hpp"

namespace aim {

/// @brief Abstraction for online closed-loop mouse calibration (CPI, counts per pixel).
class IControlCalibration {
public:
    virtual ~IControlCalibration() = default;
    virtual bool initialize(float initial_cpi, float fov_deg) noexcept = 0;
    virtual bool update_feedback(PixelPoint expected_delta, PixelPoint observed_delta) noexcept = 0;
    [[nodiscard]] virtual float current_cpi() const noexcept = 0;
    [[nodiscard]] virtual float counts_per_pixel() const noexcept = 0;
    virtual void reset() noexcept = 0;
};

} // namespace aim
