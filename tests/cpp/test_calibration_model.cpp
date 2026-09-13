// tests/cpp/test_calibration_model.cpp
#include <cmath>
#include <iostream>
#include <limits>

#include "aim/calibration/calibration_model.hpp"

using namespace aim::calibration;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

void assert_invalid_profile(const CalibrationProfile& profile) {
    CalibrationModel model{profile};
    TEST_ASSERT(!model.is_valid());
    MouseCounts counts{99, -99};
    PixelDisplacement pixels{99.0f, -99.0f};
    TEST_ASSERT(!model.try_pixels_to_counts(10.0f, 20.0f, counts));
    TEST_ASSERT(counts.counts_x == 0 && counts.counts_y == 0);
    TEST_ASSERT(!model.try_counts_to_pixels(10, 20, pixels));
    TEST_ASSERT(pixels.dx_px == 0.0f && pixels.dy_px == 0.0f);
}

void test_every_numeric_profile_field() {
    struct FieldBounds { float CalibrationProfile::*field; float minimum; float maximum; };
    const FieldBounds fields[] = {
        {&CalibrationProfile::counts_per_pixel_x, 0.001f, 100.0f},
        {&CalibrationProfile::counts_per_pixel_y, 0.001f, 100.0f},
        {&CalibrationProfile::fov_horizontal_deg, 30.0f, 150.0f},
        {&CalibrationProfile::in_game_sensitivity, 0.001f, 100.0f},
        {&CalibrationProfile::deadband_counts, 0.0f, 10.0f},
        {&CalibrationProfile::nonlinearity_alpha, -1.0f, 1.0f},
        {&CalibrationProfile::cross_coupling_xy, -0.5f, 0.5f},
        {&CalibrationProfile::rmse_pixels, 0.0f, 50.0f}
    };
    for (const auto& field : fields) {
        for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    field.minimum - 1.0f, field.maximum + 1.0f}) {
            CalibrationProfile profile{};
            profile.*field.field = invalid;
            assert_invalid_profile(profile);
        }
        for (const float valid : {field.minimum, field.maximum}) {
            CalibrationProfile profile{};
            profile.*field.field = valid;
            TEST_ASSERT(CalibrationModel{profile}.is_valid());
        }
    }
    for (const auto invalid_width : {0u, 319u, 7681u, std::numeric_limits<std::uint32_t>::max()}) {
        CalibrationProfile profile{};
        profile.resolution_width = invalid_width;
        assert_invalid_profile(profile);
    }
    for (const auto invalid_height : {0u, 239u, 4321u, std::numeric_limits<std::uint32_t>::max()}) {
        CalibrationProfile profile{};
        profile.resolution_height = invalid_height;
        assert_invalid_profile(profile);
    }
}

void test_checked_conversion_failure_clears_both_axes() {
    CalibrationModel model{};
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::max(),
                                -std::numeric_limits<float>::max(), 8000.0f, -8000.0f}) {
        for (int axis = 0; axis < 2; ++axis) {
            MouseCounts counts{10, 10};
            const float dx = axis == 0 ? invalid : 20.0f;
            const float dy = axis == 1 ? invalid : 20.0f;
            TEST_ASSERT(!model.try_pixels_to_counts(dx, dy, counts));
            TEST_ASSERT(counts.counts_x == 0 && counts.counts_y == 0);
            const auto legacy = model.pixels_to_counts(dx, dy);
            TEST_ASSERT(legacy.counts_x == 0 && legacy.counts_y == 0);
        }
    }
    for (const auto extreme : {std::numeric_limits<std::int32_t>::min(),
                               std::numeric_limits<std::int32_t>::max()}) {
        PixelDisplacement pixels{10.0f, 10.0f};
        TEST_ASSERT(!model.try_counts_to_pixels(extreme, 20, pixels));
        TEST_ASSERT(pixels.dx_px == 0.0f && pixels.dy_px == 0.0f);
        TEST_ASSERT(!model.try_counts_to_pixels(20, extreme, pixels));
    }
    MouseCounts counts{10, 10};
    TEST_ASSERT(model.try_pixels_to_counts(0.0f, 0.0f, counts));
    TEST_ASSERT(counts.counts_x == 0 && counts.counts_y == 0);
}

void test_coupled_nonlinear_inverse_roundtrip() {
    // High cross-coupling exposes the omitted determinant; signed radial gains
    // expose the prior inverse's complete omission of nonlinearity.
    for (const float coupling : {-0.5f, 0.0f, 0.5f}) {
        for (const float alpha : {-0.5f, 0.0f, 0.8f}) {
            CalibrationProfile profile{};
            profile.counts_per_pixel_x = 8.0f;
            profile.counts_per_pixel_y = 12.0f;
            profile.cross_coupling_xy = coupling;
            profile.nonlinearity_alpha = alpha;
            profile.deadband_counts = 2.0f;
            CalibrationModel model{profile};
            for (const float x : {-800.0f, -100.0f, 0.0f, 100.0f, 800.0f}) {
                for (const float y : {-400.0f, 0.0f, 400.0f}) {
                    MouseCounts counts{};
                    PixelDisplacement pixels{};
                    TEST_ASSERT(model.try_pixels_to_counts(x, y, counts));
                    TEST_ASSERT(model.try_counts_to_pixels(counts.counts_x, counts.counts_y, pixels));
                    TEST_ASSERT(std::abs(pixels.dx_px - x) < 0.3f);
                    TEST_ASSERT(std::abs(pixels.dy_px - y) < 0.3f);
                }
            }
        }
    }
    CalibrationProfile profile{};
    profile.nonlinearity_alpha = -1.0f;
    CalibrationModel model{profile};
    MouseCounts counts{10, 10};
    TEST_ASSERT(!model.try_pixels_to_counts(1900.0f, 1000.0f, counts));
    TEST_ASSERT(counts.counts_x == 0 && counts.counts_y == 0);
    PixelDisplacement pixels{10.0f, 10.0f};
    TEST_ASSERT(!model.try_counts_to_pixels(1000, 1000, pixels));
    TEST_ASSERT(pixels.dx_px == 0.0f && pixels.dy_px == 0.0f);
}

void test_ideal_roundtrip_mapping() {
    std::cout << "[Test 1] Ideal 1:1 round-trip conversion..." << std::endl;

    CalibrationProfile profile{};
    profile.counts_per_pixel_x = 1.25f;
    profile.counts_per_pixel_y = 1.25f;

    CalibrationModel model{profile};
    TEST_ASSERT(model.is_valid());

    const auto counts = model.pixels_to_counts(100.0f, -50.0f);
    TEST_ASSERT(counts.counts_x == 125);
    TEST_ASSERT(counts.counts_y == -62 || counts.counts_y == -63);

    const auto pixels = model.counts_to_pixels(125, -63);
    TEST_ASSERT(std::abs(pixels.dx_px - 100.0f) < 1.0f);
    TEST_ASSERT(std::abs(pixels.dy_px - (-50.4f)) < 1.0f);

    std::cout << "  -> Ideal round-trip mapping passed." << std::endl;
}

void test_deadband_compensation() {
    std::cout << "[Test 2] Deadband friction compensation..." << std::endl;

    CalibrationProfile profile{};
    profile.counts_per_pixel_x = 1.0f;
    profile.counts_per_pixel_y = 1.0f;
    profile.deadband_counts = 2.0f; // +2 count threshold

    CalibrationModel model{profile};
    TEST_ASSERT(model.is_valid());

    const auto counts_zero = model.pixels_to_counts(0.0f, 0.0f);
    TEST_ASSERT(counts_zero.counts_x == 0);
    TEST_ASSERT(counts_zero.counts_y == 0);

    const auto counts_small = model.pixels_to_counts(1.0f, -1.0f);
    TEST_ASSERT(counts_small.counts_x == 3); // 1 + 2 = 3
    TEST_ASSERT(counts_small.counts_y == -3); // -1 - 2 = -3

    std::cout << "  -> Deadband compensation passed." << std::endl;
}

void test_validation_bounds_failsafe() {
    std::cout << "[Test 3] Invalid calibration bounds fail-closed..." << std::endl;

    CalibrationProfile invalid_profile{};
    invalid_profile.counts_per_pixel_x = -1.0f; // Negative gain illegal!

    CalibrationModel model{invalid_profile};
    TEST_ASSERT(!model.is_valid());

    // When invalid, all outputs fail safe to 0
    const auto counts = model.pixels_to_counts(100.0f, 100.0f);
    TEST_ASSERT(counts.counts_x == 0);
    TEST_ASSERT(counts.counts_y == 0);

    std::cout << "  -> Validation bounds fail-closed passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M6-03 Calibration Model Unit Tests           " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_ideal_roundtrip_mapping();
    test_deadband_compensation();
    test_validation_bounds_failsafe();
    test_every_numeric_profile_field();
    test_checked_conversion_failure_clears_both_axes();
    test_coupled_nonlinear_inverse_roundtrip();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M6-03 Calibration Model Tests Passed Successfully!         " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
