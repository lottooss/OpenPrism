#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "aim/runtime/visual_calibration.hpp"

using namespace aim;
using namespace aim::runtime;

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "Check failed at line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace {
CalibrationObservation observation(SequenceId frame, MonotonicNs time, PixelPoint center) {
    return {frame, time, 7, center, 0.98f, 0.2f, true, true};
}

VisualCalibrationSample measured(SequenceId pulse, std::int32_t x, std::int32_t y,
                                 bool held_out, bool noise = false) {
    constexpr double cpp_x = 1.7, cpp_y = 2.4, coupling = 0.18, deadband = 1.5;
    const auto effective = [](std::int32_t count) {
        return std::copysign((std::max)(0.0, std::abs(static_cast<double>(count)) - deadband), static_cast<double>(count));
    };
    const double ex = effective(x) / cpp_x, ey = effective(y) / cpp_y;
    const double dx = (ex + coupling * ey) / (1.0 - coupling * coupling);
    const double dy = (ey + coupling * ex) / (1.0 - coupling * coupling);
    const float nx = noise ? static_cast<float>(std::sin(static_cast<double>(pulse) * 1.3) * 0.10) : 0.0f;
    const float ny = noise ? static_cast<float>(std::cos(static_cast<double>(pulse) * 0.7) * 0.10) : 0.0f;
    const MonotonicNs base = 1'000'000'000 + static_cast<MonotonicNs>(pulse) * 500'000'000;
    const SequenceId frame = pulse * 10;
    const PixelPoint before{960.0f, 540.0f};
    const PixelPoint after{before.x - static_cast<float>(dx) + nx, before.y - static_cast<float>(dy) + ny};
    VisualCalibrationSample sample;
    sample.pulse_sequence = pulse;
    sample.dispatched_at_ns = base + 21'000'000;
    sample.counts_x = x; sample.counts_y = y;
    sample.dispatch_accepted = true; sample.held_out = held_out;
    sample.baseline = observation(frame + 1, base, before);
    sample.before = observation(frame + 2, base + 20'000'000, before);
    sample.last_unchanged = observation(frame + 3, base + 26'000'000, before);
    // Frame gaps are intentional: dropped observations widen timing brackets
    // without fabricating a response timestamp or invalidating a stable pulse.
    sample.first_response = observation(frame + 5, base + 33'000'000, after);
    sample.after = sample.first_response;
    sample.settled = observation(frame + 8, base + 53'000'000, after);
    return sample;
}

void shift_response(VisualCalibrationSample& sample, float dx, float dy) {
    for (auto* observation_ptr : {&sample.first_response, &sample.after, &sample.settled}) {
        observation_ptr->center_px.x += dx;
        observation_ptr->center_px.y += dy;
    }
}

void fill(VisualCalibrationFitter& fitter, bool noise = false, bool train_outlier = false,
          bool held_out_outlier = false, bool single_axis = false) {
    SequenceId sequence = 1;
    for (const int magnitude : {12, 20, 36, 52}) {
        for (int direction = 0; direction < 4; ++direction) {
            int x = direction < 2 ? magnitude * (direction == 0 ? 1 : -1) : 0;
            int y = direction >= 2 ? magnitude * (direction == 2 ? 1 : -1) : 0;
            if (single_axis) { x += y; y = 0; }
            auto sample = measured(sequence++, x, y, false, noise);
            if (train_outlier && sample.pulse_sequence == 1) shift_response(sample, -20.0f, 0.0f);
            CHECK(fitter.add_sample(sample) == VisualCalibrationStatus::ok);
        }
    }
    const int held_out[8][2]{{16, 23}, {28, -17}, {-33, 21}, {-18, -31},
                            {43, 29}, {-45, 32}, {19, -41}, {-38, -27}};
    for (const auto& pulse : held_out) {
        auto sample = measured(sequence++, pulse[0], pulse[1], true, noise);
        if (held_out_outlier) shift_response(sample, 10.0f, 0.0f);
        CHECK(fitter.add_sample(sample) == VisualCalibrationStatus::ok);
    }
}

void test_fit_known_mapping_and_timing() {
    for (const bool noisy : {false, true}) {
        VisualCalibrationFitter fitter;
        fill(fitter, noisy);
        const auto result = fitter.fit({});
        CHECK(result.succeeded());
        CHECK(result.training_samples == 16 && result.held_out_samples == 8);
        CHECK(std::abs(result.profile.counts_per_pixel_x - 1.7f) < 0.025f);
        CHECK(std::abs(result.profile.counts_per_pixel_y - 2.4f) < 0.025f);
        CHECK(std::abs(result.profile.cross_coupling_xy - 0.18f) < 0.01f);
        CHECK(std::abs(result.profile.deadband_counts - 1.5f) < 0.25f);
        CHECK(result.profile.nonlinearity_alpha == 0.0f);
        CHECK(result.profile.rmse_pixels < 0.2f);
        CHECK(result.held_out_max_error_px < 0.3f);
        CHECK(result.effect_lower_p50_ns == 5'000'000);
        CHECK(result.effect_upper_p50_ns == 12'000'000);
        CHECK(result.effect_upper_p95_ns == 12'000'000);
        CHECK(result.largest_effect_bracket_ns == 7'000'000);
        CHECK(result.measured_max_pulse_counts == 52);
    }
}

void test_training_outlier_rejection_and_held_out_independence() {
    VisualCalibrationFitter fitter;
    fill(fitter, true, true);
    const auto fit = fitter.fit({});
    CHECK(fit.succeeded());
    CHECK(fit.rejected_training_outliers == 1);
    CHECK(fit.profile.rmse_pixels < 0.3f);

    VisualCalibrationFitter bad_validation;
    fill(bad_validation, false, false, true);
    const auto rejected = bad_validation.fit({});
    CHECK(rejected.status == VisualCalibrationStatus::held_out_error);
    CHECK(!calibration::CalibrationModel(rejected.profile).is_valid());
}

void test_sample_validation_and_wrong_world_sign() {
    for (int fault = 0; fault < 12; ++fault) {
        VisualCalibrationFitter fitter;
        auto sample = measured(1, 20, 0, false);
        VisualCalibrationStatus expected = VisualCalibrationStatus::invalid_observation;
        if (fault == 0) { sample.dispatch_accepted = false; expected = VisualCalibrationStatus::rejected_dispatch; }
        if (fault == 1) { sample.after.foreground_authorized = false; expected = VisualCalibrationStatus::lost_focus; }
        if (fault == 2) { sample.after.association_unambiguous = false; expected = VisualCalibrationStatus::ambiguous_association; }
        if (fault == 3) { sample.after.target_id = 8; expected = VisualCalibrationStatus::ambiguous_association; }
        if (fault == 4) sample.after.center_px.x = std::numeric_limits<float>::quiet_NaN();
        if (fault == 5) sample.after.confidence = 0.1f;
        if (fault == 6) sample.after.uncertainty_px = std::numeric_limits<float>::infinity();
        if (fault == 7) { sample.counts_x = std::numeric_limits<std::int32_t>::min(); expected = VisualCalibrationStatus::invalid_pulse; }
        if (fault == 8) { sample.dispatched_at_ns = sample.before.captured_at_ns + 11'000'000; expected = VisualCalibrationStatus::invalid_timing; }
        if (fault == 9) { sample.baseline.center_px.x += 4.0f; expected = VisualCalibrationStatus::moving_target; }
        if (fault == 10) { sample.settled.center_px.y += 4.0f; expected = VisualCalibrationStatus::moving_target; }
        if (fault == 11) {
            for (auto* observation_ptr : {&sample.first_response, &sample.after, &sample.settled})
                observation_ptr->center_px.x = 2.0f * sample.before.center_px.x - observation_ptr->center_px.x;
            expected = VisualCalibrationStatus::wrong_direction;
        }
        CHECK(fitter.add_sample(sample) == expected);
        CHECK(fitter.size() == 0 && fitter.rejected_samples() == 1);
    }
}

void test_unidentifiable_and_capacity() {
    VisualCalibrationFitter only_x;
    fill(only_x, false, false, false, true);
    CHECK(only_x.fit({}).status == VisualCalibrationStatus::unidentifiable);
    VisualCalibrationFitter empty;
    CHECK(empty.fit({}).status == VisualCalibrationStatus::insufficient_samples);
    for (SequenceId i = 1; i <= VisualCalibrationFitter::kCapacity; ++i)
        CHECK(empty.add_sample(measured(i, 20, 0, false)) == VisualCalibrationStatus::ok);
    CHECK(empty.add_sample(measured(129, 20, 0, false)) == VisualCalibrationStatus::capacity_exceeded);
    CHECK(empty.size() == VisualCalibrationFitter::kCapacity);
    empty.reset();
    CHECK(empty.size() == 0 && empty.rejected_samples() == 0);
    auto sample = measured(1, 20, 0, false);
    CHECK(empty.add_sample(sample) == VisualCalibrationStatus::ok);
    CHECK(empty.add_sample(sample) == VisualCalibrationStatus::invalid_timing);

    VisualCalibrationConfig config;
    config.maximum_sample_error_px = 5.1f;
    VisualCalibrationFitter invalid(config);
    CHECK(invalid.add_sample(sample) == VisualCalibrationStatus::invalid_configuration);
    CHECK(invalid.fit({}).status == VisualCalibrationStatus::invalid_configuration);
}
} // namespace

int main() {
    test_fit_known_mapping_and_timing();
    test_training_outlier_rejection_and_held_out_independence();
    test_sample_validation_and_wrong_world_sign();
    test_unidentifiable_and_capacity();
    std::cout << "Visual calibration fitting tests passed; all measurements are deterministic fixtures.\n";
}
