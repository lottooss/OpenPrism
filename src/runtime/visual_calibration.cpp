#include "aim/runtime/visual_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aim::runtime {
namespace {
constexpr MonotonicNs kSourceAgeNs = 10'000'000;

bool between(float value, float low, float high) noexcept {
    return std::isfinite(value) && value >= low && value <= high;
}

double distance(PixelPoint a, PixelPoint b) noexcept {
    return std::hypot(static_cast<double>(a.x) - b.x, static_cast<double>(a.y) - b.y);
}

bool ordered(const CalibrationObservation& a, const CalibrationObservation& b,
             bool same_allowed = false) noexcept {
    if (a.frame_id < b.frame_id && a.captured_at_ns < b.captured_at_ns) return true;
    return same_allowed && a.frame_id == b.frame_id && a.captured_at_ns == b.captured_at_ns &&
           a.center_px.x == b.center_px.x && a.center_px.y == b.center_px.y;
}

double effective_counts(std::int32_t counts, double deadband) noexcept {
    const double value = counts;
    return std::copysign((std::max)(0.0, std::abs(value) - deadband), value);
}

double median(std::array<double, VisualCalibrationFitter::kCapacity>& values, std::size_t count) {
    std::sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
    return count % 2 != 0 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2.0;
}

PixelPoint camera_response(const VisualCalibrationSample& sample) noexcept {
    // In a fixed-crosshair camera, positive mouse counts move a stationary world
    // target in the opposite screen direction. Never fit the reversed sign.
    return {sample.before.center_px.x - sample.after.center_px.x,
            sample.before.center_px.y - sample.after.center_px.y};
}

bool coverage(const std::array<VisualCalibrationSample, VisualCalibrationFitter::kCapacity>& samples,
              std::size_t count, const std::array<bool, VisualCalibrationFitter::kCapacity>& included) {
    bool positive[2]{}, negative[2]{};
    std::int32_t smallest[2]{std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max()};
    std::int32_t largest[2]{};
    for (std::size_t i = 0; i < count; ++i) {
        if (!included[i] || samples[i].held_out) continue;
        const auto& sample = samples[i];
        if (sample.counts_x != 0 && sample.counts_y != 0) continue;
        const int axis = sample.counts_x != 0 ? 0 : 1;
        const auto value = axis == 0 ? sample.counts_x : sample.counts_y;
        positive[axis] = positive[axis] || value > 0;
        negative[axis] = negative[axis] || value < 0;
        const auto magnitude = static_cast<std::int32_t>(std::abs(static_cast<std::int64_t>(value)));
        smallest[axis] = (std::min)(smallest[axis], magnitude);
        largest[axis] = (std::max)(largest[axis], magnitude);
    }
    return positive[0] && negative[0] && positive[1] && negative[1] &&
           largest[0] - smallest[0] >= 2 && largest[1] - smallest[1] >= 2;
}

double prediction_error(const VisualCalibrationSample& sample,
                        const calibration::CalibrationModel& model) noexcept {
    calibration::PixelDisplacement predicted{};
    if (!model.try_counts_to_pixels(sample.counts_x, sample.counts_y, predicted))
        return std::numeric_limits<double>::infinity();
    const auto actual = camera_response(sample);
    return std::hypot(static_cast<double>(actual.x) - predicted.dx_px,
                      static_cast<double>(actual.y) - predicted.dy_px);
}
} // namespace

VisualCalibrationFitter::VisualCalibrationFitter(const VisualCalibrationConfig& config) noexcept
    : config_(config) {
    config_valid_ = config.width >= 320 && config.width <= 7680 && config.height >= 240 && config.height <= 4320 &&
        config.max_pulse_counts > 0 && config.max_pulse_counts <= 250 &&
        config.minimum_training_samples >= 8 && config.minimum_training_samples <= kCapacity &&
        config.minimum_held_out_samples >= 4 && config.minimum_held_out_samples <= kCapacity &&
        config.minimum_training_samples + config.minimum_held_out_samples <= kCapacity &&
        between(config.minimum_confidence, 0.5f, 1.0f) &&
        between(config.maximum_uncertainty_px, 0.0f, 5.0f) &&
        between(config.stationary_tolerance_px, 0.0f, 2.0f) &&
        between(config.minimum_response_px, 1.0f, 10.0f) &&
        between(config.maximum_response_px, config.minimum_response_px, 500.0f) &&
        between(config.maximum_sample_error_px, 0.01f, 5.0f) &&
        config.minimum_stability_ns >= 1'000'000 && config.minimum_stability_ns <= 100'000'000 &&
        config.maximum_effect_delay_ns > 0 && config.maximum_effect_delay_ns <= 250'000'000 &&
        config.maximum_sample_duration_ns >= config.maximum_effect_delay_ns &&
        config.maximum_sample_duration_ns <= 1'000'000'000;
}

VisualCalibrationStatus VisualCalibrationFitter::validate(const VisualCalibrationSample& sample) const noexcept {
    if (!config_valid_) return VisualCalibrationStatus::invalid_configuration;
    if (!sample.dispatch_accepted) return VisualCalibrationStatus::rejected_dispatch;
    const auto limit = static_cast<std::int64_t>(config_.max_pulse_counts);
    if (sample.pulse_sequence == 0 || (sample.counts_x == 0 && sample.counts_y == 0) ||
        std::abs(static_cast<std::int64_t>(sample.counts_x)) > limit ||
        std::abs(static_cast<std::int64_t>(sample.counts_y)) > limit)
        return VisualCalibrationStatus::invalid_pulse;
    const CalibrationObservation* observations[]{&sample.baseline, &sample.before, &sample.last_unchanged,
        &sample.first_response, &sample.after, &sample.settled};
    for (const auto* observation : observations) {
        if (!observation->foreground_authorized) return VisualCalibrationStatus::lost_focus;
        if (!observation->association_unambiguous || observation->target_id == 0 ||
            observation->target_id != sample.before.target_id)
            return VisualCalibrationStatus::ambiguous_association;
        if (observation->frame_id == 0 || observation->captured_at_ns <= 0 ||
            !between(observation->confidence, config_.minimum_confidence, 1.0f) ||
            !between(observation->uncertainty_px, 0.0f, config_.maximum_uncertainty_px) ||
            !between(observation->center_px.x, 0.0f, static_cast<float>(config_.width - 1)) ||
            !between(observation->center_px.y, 0.0f, static_cast<float>(config_.height - 1)))
            return VisualCalibrationStatus::invalid_observation;
    }
    if (!ordered(sample.baseline, sample.before) || !ordered(sample.before, sample.last_unchanged, true) ||
        !ordered(sample.last_unchanged, sample.first_response) || !ordered(sample.first_response, sample.after, true) ||
        !ordered(sample.after, sample.settled) || sample.dispatched_at_ns < sample.before.captured_at_ns ||
        sample.dispatched_at_ns >= sample.first_response.captured_at_ns ||
        sample.dispatched_at_ns - sample.before.captured_at_ns > kSourceAgeNs ||
        sample.before.captured_at_ns - sample.baseline.captured_at_ns < config_.minimum_stability_ns ||
        sample.settled.captured_at_ns - sample.after.captured_at_ns < config_.minimum_stability_ns ||
        sample.first_response.captured_at_ns - sample.dispatched_at_ns > config_.maximum_effect_delay_ns ||
        sample.settled.captured_at_ns - sample.baseline.captured_at_ns > config_.maximum_sample_duration_ns)
        return VisualCalibrationStatus::invalid_timing;
    if (count_ != 0 && (sample.pulse_sequence <= samples_[count_ - 1].pulse_sequence ||
        sample.before.captured_at_ns <= samples_[count_ - 1].settled.captured_at_ns))
        return VisualCalibrationStatus::invalid_timing;
    if (distance(sample.baseline.center_px, sample.before.center_px) > config_.stationary_tolerance_px ||
        distance(sample.last_unchanged.center_px, sample.before.center_px) > config_.stationary_tolerance_px ||
        distance(sample.after.center_px, sample.settled.center_px) > config_.stationary_tolerance_px)
        return VisualCalibrationStatus::moving_target;
    const auto response = camera_response(sample);
    const double magnitude = std::hypot(response.x, response.y);
    if (magnitude < config_.minimum_response_px || magnitude > config_.maximum_response_px ||
        distance(sample.before.center_px, sample.first_response.center_px) < config_.minimum_response_px)
        return VisualCalibrationStatus::insufficient_response;
    if ((sample.counts_y == 0 && static_cast<double>(sample.counts_x) * response.x <= 0.0) ||
        (sample.counts_x == 0 && static_cast<double>(sample.counts_y) * response.y <= 0.0))
        return VisualCalibrationStatus::wrong_direction;
    return VisualCalibrationStatus::ok;
}

VisualCalibrationStatus VisualCalibrationFitter::add_sample(const VisualCalibrationSample& sample) noexcept {
    auto status = validate(sample);
    if (status == VisualCalibrationStatus::ok && count_ == kCapacity)
        status = VisualCalibrationStatus::capacity_exceeded;
    if (status != VisualCalibrationStatus::ok) { ++rejected_; return status; }
    samples_[count_++] = sample;
    return VisualCalibrationStatus::ok;
}

void VisualCalibrationFitter::reset() noexcept { count_ = 0; rejected_ = 0; }

VisualCalibrationFit VisualCalibrationFitter::fit(const calibration::CalibrationProfile& context) const {
    VisualCalibrationFit result;
    if (!config_valid_ || context.resolution_width != config_.width || context.resolution_height != config_.height ||
        !between(context.fov_horizontal_deg, 30.0f, 150.0f) || !between(context.in_game_sensitivity, 0.001f, 100.0f)) {
        result.status = VisualCalibrationStatus::invalid_configuration;
        return result;
    }
    std::array<bool, kCapacity> training{};
    std::size_t training_count = 0, held_out_count = 0;
    for (std::size_t i = 0; i < count_; ++i) {
        if (samples_[i].held_out) ++held_out_count;
        else { training[i] = true; ++training_count; }
    }
    if (training_count < config_.minimum_training_samples || held_out_count < config_.minimum_held_out_samples)
        return result;
    if (!coverage(samples_, count_, training)) { result.status = VisualCalibrationStatus::unidentifiable; return result; }

    double best_score = std::numeric_limits<double>::infinity();
    double best_sse = 0.0;
    std::size_t best_inliers = 0;
    calibration::CalibrationProfile best_profile = context;
    std::array<bool, kCapacity> best_included{};
    // Fit only training pulses. The explicit local model is linear with shared
    // deadband/coupling; alpha=0 is a model choice, not a fabricated measurement.
    // Distinct signed magnitudes identify deadband independently of gain.
    for (int deadband_step = 0; deadband_step <= 100; ++deadband_step) {
        const double deadband = static_cast<double>(deadband_step) / 10.0;
        std::array<double, kCapacity> xx_rates{}, yx_rates{}, xy_rates{}, yy_rates{};
        std::size_t nx = 0, ny = 0;
        for (std::size_t i = 0; i < count_; ++i) {
            if (!training[i]) continue;
            const auto& sample = samples_[i];
            const auto response = camera_response(sample);
            const double x = effective_counts(sample.counts_x, deadband);
            const double y = effective_counts(sample.counts_y, deadband);
            if (sample.counts_y == 0 && x != 0.0) { xx_rates[nx] = response.x / x; yx_rates[nx++] = response.y / x; }
            if (sample.counts_x == 0 && y != 0.0) { xy_rates[ny] = response.x / y; yy_rates[ny++] = response.y / y; }
        }
        if (nx < 2 || ny < 2) continue;
        const double ax = median(xx_rates, nx), ay = median(yx_rates, nx);
        const double bx = median(xy_rates, ny), by = median(yy_rates, ny);
        std::array<bool, kCapacity> included{};
        std::size_t inliers = 0;
        double xx = 0.0, xy = 0.0, yy = 0.0, px_x = 0.0, px_y = 0.0, py_x = 0.0, py_y = 0.0;
        for (std::size_t i = 0; i < count_; ++i) {
            if (!training[i]) continue;
            const auto& sample = samples_[i];
            const auto response = camera_response(sample);
            const double x = effective_counts(sample.counts_x, deadband), y = effective_counts(sample.counts_y, deadband);
            if (std::hypot(response.x - ax * x - bx * y, response.y - ay * x - by * y) > config_.maximum_sample_error_px)
                continue;
            included[i] = true; ++inliers;
            xx += x * x; xy += x * y; yy += y * y;
            px_x += response.x * x; px_y += response.x * y;
            py_x += response.y * x; py_y += response.y * y;
        }
        if (inliers < config_.minimum_training_samples || (training_count - inliers) * 5 > training_count ||
            !coverage(samples_, count_, included)) continue;
        const double determinant = xx * yy - xy * xy;
        if (!(determinant > 1.0e-8 * xx * yy)) continue;
        const double bxx = (px_x * yy - px_y * xy) / determinant;
        const double bxy = (px_y * xx - px_x * xy) / determinant;
        const double byx = (py_x * yy - py_y * xy) / determinant;
        const double byy = (py_y * xx - py_x * xy) / determinant;
        if (!(bxx > 0.0 && byy > 0.0)) continue;
        const double coupling_x = byx / bxx, coupling_y = bxy / byy;
        if (!std::isfinite(coupling_x) || !std::isfinite(coupling_y) || std::abs(coupling_x - coupling_y) > 0.10) continue;
        const double coupling = (coupling_x + coupling_y) / 2.0;
        if (std::abs(coupling) > 0.5) continue;
        const double gain_x = 1.0 / (bxx * (1.0 - coupling * coupling));
        const double gain_y = 1.0 / (byy * (1.0 - coupling * coupling));
        if (!std::isfinite(gain_x) || !std::isfinite(gain_y) || gain_x < 0.001 || gain_x > 100.0 ||
            gain_y < 0.001 || gain_y > 100.0) continue;
        auto profile = context;
        profile.counts_per_pixel_x = static_cast<float>(gain_x);
        profile.counts_per_pixel_y = static_cast<float>(gain_y);
        profile.cross_coupling_xy = static_cast<float>(coupling);
        profile.deadband_counts = static_cast<float>(deadband);
        profile.nonlinearity_alpha = 0.0f;
        profile.rmse_pixels = 0.0f;
        const calibration::CalibrationModel model(profile);
        if (!model.is_valid()) continue;
        double sse = 0.0;
        bool consistent = true;
        for (std::size_t i = 0; i < count_; ++i) {
            if (!included[i]) continue;
            const double error = prediction_error(samples_[i], model);
            if (error > config_.maximum_sample_error_px) { consistent = false; break; }
            sse += error * error;
        }
        if (!consistent) continue;
        const double penalty = static_cast<double>(training_count - inliers) *
            config_.maximum_sample_error_px * config_.maximum_sample_error_px;
        const double score = (sse + penalty) / static_cast<double>(training_count);
        if (score < best_score) {
            best_score = score; best_sse = sse; best_inliers = inliers;
            best_profile = profile; best_included = included;
        }
    }
    if (!std::isfinite(best_score)) { result.status = VisualCalibrationStatus::inconsistent_response; return result; }
    const calibration::CalibrationModel model(best_profile);
    double held_out_sse = 0.0, held_out_max = 0.0;
    bool positive[2]{}, negative[2]{};
    std::array<MonotonicNs, kCapacity> lower{}, upper{};
    std::size_t timing_count = 0;
    for (std::size_t i = 0; i < count_; ++i) {
        const auto& sample = samples_[i];
        if (sample.held_out) {
            const double error = prediction_error(sample, model);
            if (!std::isfinite(error) || error > config_.maximum_sample_error_px) {
                result.status = VisualCalibrationStatus::held_out_error; return result;
            }
            held_out_sse += error * error;
            held_out_max = (std::max)(held_out_max, error);
            positive[0] = positive[0] || sample.counts_x > 0; negative[0] = negative[0] || sample.counts_x < 0;
            positive[1] = positive[1] || sample.counts_y > 0; negative[1] = negative[1] || sample.counts_y < 0;
        } else if (!best_included[i]) continue;
        const auto lower_ns = (std::max)(sample.dispatched_at_ns, sample.last_unchanged.captured_at_ns) - sample.dispatched_at_ns;
        const auto upper_ns = sample.first_response.captured_at_ns - sample.dispatched_at_ns;
        lower[timing_count] = lower_ns; upper[timing_count++] = upper_ns;
        result.largest_effect_bracket_ns = (std::max)(result.largest_effect_bracket_ns, upper_ns - lower_ns);
        const auto magnitude = (std::max)(std::abs(static_cast<std::int64_t>(sample.counts_x)),
                                         std::abs(static_cast<std::int64_t>(sample.counts_y)));
        result.measured_max_pulse_counts = (std::max)(result.measured_max_pulse_counts, static_cast<std::int32_t>(magnitude));
    }
    if (!positive[0] || !negative[0] || !positive[1] || !negative[1]) {
        result.status = VisualCalibrationStatus::unidentifiable; return result;
    }
    std::sort(lower.begin(), lower.begin() + static_cast<std::ptrdiff_t>(timing_count));
    std::sort(upper.begin(), upper.begin() + static_cast<std::ptrdiff_t>(timing_count));
    best_profile.rmse_pixels = static_cast<float>(std::sqrt(held_out_sse / static_cast<double>(held_out_count)));
    result.profile = best_profile;
    result.training_samples = best_inliers;
    result.held_out_samples = held_out_count;
    result.rejected_training_outliers = training_count - best_inliers;
    result.training_rmse_px = static_cast<float>(std::sqrt(best_sse / static_cast<double>(best_inliers)));
    result.held_out_max_error_px = static_cast<float>(held_out_max);
    result.effect_lower_p50_ns = lower[timing_count / 2];
    result.effect_upper_p50_ns = upper[timing_count / 2];
    result.effect_upper_p95_ns = upper[timing_count * 95 / 100];
    result.status = VisualCalibrationStatus::ok;
    return result;
}

} // namespace aim::runtime
