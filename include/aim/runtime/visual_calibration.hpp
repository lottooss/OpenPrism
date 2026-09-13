#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "aim/calibration/calibration_model.hpp"
#include "aim/core/types.hpp"

namespace aim::runtime {

struct CalibrationObservation {
    SequenceId frame_id{0};
    MonotonicNs captured_at_ns{0};
    TrackId target_id{0};
    PixelPoint center_px{};
    float confidence{0.0f};
    float uncertainty_px{0.0f};
    bool foreground_authorized{false};
    bool association_unambiguous{false};
};

struct VisualCalibrationSample {
    SequenceId pulse_sequence{0};
    MonotonicNs dispatched_at_ns{0};
    std::int32_t counts_x{0};
    std::int32_t counts_y{0};
    bool dispatch_accepted{false};
    bool held_out{false};
    CalibrationObservation baseline{}; // stationary frame preceding before
    CalibrationObservation before{};   // fresh frame used to authorize pulse
    CalibrationObservation last_unchanged{}; // may equal before
    CalibrationObservation first_response{};
    CalibrationObservation after{};    // settled response; may equal first_response
    CalibrationObservation settled{};  // independent later stationary confirmation
};

struct VisualCalibrationConfig {
    std::uint32_t width{1920};
    std::uint32_t height{1080};
    std::int32_t max_pulse_counts{64};
    std::size_t minimum_training_samples{12};
    std::size_t minimum_held_out_samples{8};
    float minimum_confidence{0.8f};
    float maximum_uncertainty_px{3.0f}; // native decoder's conservative 1-model-pixel uncertainty
    float stationary_tolerance_px{1.0f};
    float minimum_response_px{2.0f};
    float maximum_response_px{200.0f};
    float maximum_sample_error_px{5.0f};
    MonotonicNs minimum_stability_ns{10'000'000};
    MonotonicNs maximum_effect_delay_ns{100'000'000};
    MonotonicNs maximum_sample_duration_ns{300'000'000};
};

enum class VisualCalibrationStatus : std::uint8_t {
    ok, invalid_configuration, capacity_exceeded, rejected_dispatch,
    invalid_observation, lost_focus, ambiguous_association, invalid_timing,
    invalid_pulse, moving_target, insufficient_response, wrong_direction,
    insufficient_samples, unidentifiable, inconsistent_response, held_out_error
};

struct VisualCalibrationFit {
    VisualCalibrationStatus status{VisualCalibrationStatus::insufficient_samples};
    calibration::CalibrationProfile profile{.counts_per_pixel_x = 0.0f,
                                             .counts_per_pixel_y = 0.0f};
    std::size_t training_samples{0};
    std::size_t held_out_samples{0};
    std::size_t rejected_training_outliers{0};
    float training_rmse_px{0.0f};
    float held_out_max_error_px{0.0f};
    std::int32_t measured_max_pulse_counts{0};
    // Capture brackets, not photon-to-photon point estimates. No timestamp is
    // synthesized between frames. The upper bound is first visible response.
    MonotonicNs effect_lower_p50_ns{0};
    MonotonicNs effect_upper_p50_ns{0};
    MonotonicNs effect_upper_p95_ns{0};
    MonotonicNs largest_effect_bracket_ns{0};
    [[nodiscard]] bool succeeded() const noexcept { return status == VisualCalibrationStatus::ok; }
};

// Pure measurement/fitting component: never dispatches input or supplies guessed
// calibration. add_sample uses fixed storage. fit is an explicit cold operation.
// Host owns target association and must abort collection on focus/estop failure.
class VisualCalibrationFitter {
public:
    static constexpr std::size_t kCapacity = 128;
    explicit VisualCalibrationFitter(const VisualCalibrationConfig& config = {}) noexcept;
    [[nodiscard]] VisualCalibrationStatus add_sample(const VisualCalibrationSample& sample) noexcept;
    [[nodiscard]] VisualCalibrationFit fit(const calibration::CalibrationProfile& context) const;
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t rejected_samples() const noexcept { return rejected_; }
    [[nodiscard]] bool configuration_valid() const noexcept { return config_valid_; }
    void reset() noexcept;

private:
    [[nodiscard]] VisualCalibrationStatus validate(const VisualCalibrationSample& sample) const noexcept;
    VisualCalibrationConfig config_{};
    bool config_valid_{false};
    std::array<VisualCalibrationSample, kCapacity> samples_{};
    std::size_t count_{0};
    std::size_t rejected_{0};
};

} // namespace aim::runtime
