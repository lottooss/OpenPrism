// include/aim/tracking/latency_estimator.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim::tracking {

struct LatencyEstimatorConfig {
    MonotonicNs min_latency_ns{1'500'000LL};     // 1.5 ms minimum plausible latency
    MonotonicNs max_latency_ns{25'000'000LL};    // 25.0 ms maximum allowed cutoff
    MonotonicNs default_latency_ns{6'000'000LL}; // 6.0 ms nominal latency
    float ewma_alpha{0.15f};                     // Smoothing factor for mean
    float ewma_var_alpha{0.10f};                 // Smoothing factor for variance
    float max_allowed_std_dev_ns{3'000'000.0f};  // 3.0 ms max standard deviation for stability
    std::uint32_t warmup_samples{5};             // Number of samples required before declaring stable
    MonotonicNs max_sample_age_ns{100'000'000LL};// 100 ms max age before decaying to stale
};

struct LatencyEstimate {
    MonotonicNs estimated_latency_ns{6'000'000LL};
    float standard_deviation_ns{0.0f};
    bool is_stable{false};
    std::uint64_t sample_count{0};
    MonotonicNs last_sample_time_ns{0};
};

/// @brief Fuses pipeline stage durations into an adaptive, outlier-rejected estimate of command-effect latency.
class CommandEffectLatencyEstimator {
public:
    explicit CommandEffectLatencyEstimator(LatencyEstimatorConfig config = {})
        : config_(config) {
        reset();
    }

    void reset() noexcept {
        estimate_.estimated_latency_ns = config_.default_latency_ns;
        estimate_.standard_deviation_ns = 0.0f;
        estimate_.is_stable = false;
        estimate_.sample_count = 0;
        estimate_.last_sample_time_ns = 0;
        variance_ns2_ = 0.0f;
    }

    /// @brief Ingest a new measured pipeline latency sample.
    void update(MonotonicNs measured_latency_ns, MonotonicNs current_time_ns) noexcept {
        estimate_.last_sample_time_ns = current_time_ns;

        // 1. Strict outlier clamping / rejection
        const MonotonicNs clamped = std::clamp(measured_latency_ns, config_.min_latency_ns, config_.max_latency_ns);
        const float sample_f = static_cast<float>(clamped);

        if (estimate_.sample_count == 0) {
            estimate_.estimated_latency_ns = clamped;
            variance_ns2_ = 0.0f;
            estimate_.standard_deviation_ns = 0.0f;
            estimate_.sample_count = 1;
            estimate_.is_stable = (config_.warmup_samples <= 1);
            return;
        }

        // 2. Exponentially Weighted Moving Average & Variance
        const float prev_mean_f = static_cast<float>(estimate_.estimated_latency_ns);
        const float diff = sample_f - prev_mean_f;

        const float new_mean_f = prev_mean_f + config_.ewma_alpha * diff;
        estimate_.estimated_latency_ns = static_cast<MonotonicNs>(new_mean_f);

        variance_ns2_ = (1.0f - config_.ewma_var_alpha) * variance_ns2_ + config_.ewma_var_alpha * (diff * diff);
        estimate_.standard_deviation_ns = std::sqrt((std::max)(0.0f, variance_ns2_));

        estimate_.sample_count++;

        // 3. Stability check
        estimate_.is_stable = (estimate_.sample_count >= config_.warmup_samples) &&
                              (estimate_.standard_deviation_ns <= config_.max_allowed_std_dev_ns);
    }

    /// @brief Predict the absolute command-effect timestamp given a capture timestamp.
    [[nodiscard]] MonotonicNs predict_effect_time(MonotonicNs capture_time_ns, MonotonicNs current_time_ns) const noexcept {
        // If estimate is stale (> max_sample_age_ns without update), fail-closed to nominal
        if (estimate_.last_sample_time_ns > 0 &&
            (current_time_ns - estimate_.last_sample_time_ns) > config_.max_sample_age_ns) {
            return capture_time_ns + config_.default_latency_ns;
        }
        return capture_time_ns + estimate_.estimated_latency_ns;
    }

    [[nodiscard]] const LatencyEstimate& estimate() const noexcept { return estimate_; }
    [[nodiscard]] const LatencyEstimatorConfig& config() const noexcept { return config_; }

private:
    LatencyEstimatorConfig config_;
    LatencyEstimate estimate_{};
    float variance_ns2_{0.0f};
};

} // namespace aim::tracking
