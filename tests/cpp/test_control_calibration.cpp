// tests/cpp/test_control_calibration.cpp
#include <cmath>
#include <iostream>
#include <limits>

#include "aim/calibration/control_calibration.hpp"

using namespace aim;
using namespace aim::calibration;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

void test_nonfinite_feedback_preserves_valid_estimate() {
    ControlCalibration calibration{};
    const float initial = calibration.counts_per_pixel();
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()}) {
        for (int field = 0; field < 4; ++field) {
            PixelPoint expected{100.0f, 10.0f};
            PixelPoint observed{100.0f, 10.0f};
            if (field == 0) expected.x = invalid;
            if (field == 1) expected.y = invalid;
            if (field == 2) observed.x = invalid;
            if (field == 3) observed.y = invalid;
            TEST_ASSERT(!calibration.update_feedback(expected, observed));
            TEST_ASSERT(calibration.counts_per_pixel() == initial);
            TEST_ASSERT(std::isfinite(calibration.current_cpi()));
        }
    }
    TEST_ASSERT(calibration.successful_updates() == 0);
    TEST_ASSERT(calibration.rejected_outliers() == 12);
    TEST_ASSERT(calibration.update_feedback({100.0f, 0.0f}, {95.0f, 0.0f}));
    const float huge = std::numeric_limits<float>::max();
    TEST_ASSERT(calibration.update_feedback({huge, huge}, {huge, huge}));
    TEST_ASSERT(std::isfinite(calibration.counts_per_pixel()));
}

void test_invalid_configuration_and_reinitialization_fail_closed() {
    float OnlineCalibrationConfig::* const fields[] = {
        &OnlineCalibrationConfig::initial_cpi, &OnlineCalibrationConfig::initial_fov_deg,
        &OnlineCalibrationConfig::initial_counts_per_pixel, &OnlineCalibrationConfig::max_drift_fraction,
        &OnlineCalibrationConfig::max_step_delta, &OnlineCalibrationConfig::min_magnitude_px,
        &OnlineCalibrationConfig::outlier_ratio_threshold, &OnlineCalibrationConfig::learning_rate
    };
    for (const auto field : fields) {
        for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(), -1.0f}) {
            OnlineCalibrationConfig config{};
            config.*field = invalid;
            ControlCalibration calibration{config};
            TEST_ASSERT(!calibration.update_feedback({100.0f, 0.0f}, {95.0f, 0.0f}));
            TEST_ASSERT(calibration.counts_per_pixel() == 0.0f);
            TEST_ASSERT(calibration.current_cpi() == 0.0f);
        }
    }
    ControlCalibration calibration{};
    TEST_ASSERT(!calibration.initialize(std::numeric_limits<float>::quiet_NaN(), 103.0f));
    TEST_ASSERT(calibration.counts_per_pixel() == 0.0f);
    TEST_ASSERT(!calibration.update_feedback({100.0f, 0.0f}, {95.0f, 0.0f}));
    TEST_ASSERT(calibration.initialize(800.0f, 103.0f));
    TEST_ASSERT(calibration.update_feedback({100.0f, 0.0f}, {95.0f, 0.0f}));
}

void test_nominal_convergence() {
    std::cout << "[Test 1] Nominal adaptation convergence..." << std::endl;

    OnlineCalibrationConfig config{};
    config.initial_counts_per_pixel = 1.0f;
    config.initial_cpi = 800.0f;
    config.learning_rate = 0.1f;

    ControlCalibration calib{config};

    // Simulate true system having counts_per_pixel = 1.10 (moved less pixels than expected by 10%)
    // Expected: 100 px, Observed: 91 px -> ratio = 100/91 ~= 1.0989
    for (int i = 0; i < 30; ++i) {
        PixelPoint exp{100.0f, 0.0f};
        PixelPoint obs{91.0f, 0.0f};
        TEST_ASSERT(calib.update_feedback(exp, obs));
    }

    TEST_ASSERT(calib.successful_updates() == 30);
    TEST_ASSERT(std::abs(calib.counts_per_pixel() - 1.10f) < 0.05f);
    std::cout << "  -> Adapted cpp = " << calib.counts_per_pixel() << " (converged near 1.10)." << std::endl;
}

void test_drift_bounding() {
    std::cout << "[Test 2] Strict drift bounding (+/- 25%)..." << std::endl;

    OnlineCalibrationConfig config{};
    config.initial_counts_per_pixel = 1.0f;
    config.max_drift_fraction = 0.20f; // Max 1.20, Min 0.80
    config.learning_rate = 0.5f;

    ControlCalibration calib{config};

    // Push extreme updates to force drift
    for (int i = 0; i < 50; ++i) {
        PixelPoint exp{100.0f, 0.0f};
        PixelPoint obs{80.0f, 0.0f}; // 25% error
        calib.update_feedback(exp, obs);
    }

    // Must be clamped at exactly 1.20
    TEST_ASSERT(calib.counts_per_pixel() <= 1.2001f);
    std::cout << "  -> Max drift clamped at: " << calib.counts_per_pixel() << std::endl;
}

void test_outlier_rejection() {
    std::cout << "[Test 3] Outlier rejection and rollback..." << std::endl;

    OnlineCalibrationConfig config{};
    config.initial_counts_per_pixel = 1.0f;
    config.outlier_ratio_threshold = 0.30f;

    ControlCalibration calib{config};

    // Extreme divergent observation (50% error)
    PixelPoint exp{100.0f, 0.0f};
    PixelPoint obs{50.0f, 0.0f};
    TEST_ASSERT(!calib.update_feedback(exp, obs)); // Rejected
    TEST_ASSERT(calib.rejected_outliers() == 1);
    TEST_ASSERT(calib.counts_per_pixel() == 1.0f); // Unchanged

    // Negative direction observation
    PixelPoint obs_neg{-100.0f, 0.0f};
    TEST_ASSERT(!calib.update_feedback(exp, obs_neg)); // Rejected
    TEST_ASSERT(calib.rejected_outliers() == 2);

    // Rollback test
    calib.rollback_to_baseline();
    TEST_ASSERT(calib.counts_per_pixel() == 1.0f);
    TEST_ASSERT(calib.successful_updates() == 0);

    std::cout << "  -> Outlier rejection and rollback passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M6-04 Online Calibration Unit Tests          " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_nominal_convergence();
    test_drift_bounding();
    test_outlier_rejection();
    test_nonfinite_feedback_preserves_valid_estimate();
    test_invalid_configuration_and_reinitialization_fail_closed();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M6-04 Online Calibration Tests Passed Successfully!        " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
