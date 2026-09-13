// include/aim/tracking/track_lifecycle.hpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/tracking/gated_hungarian.hpp"
#include "aim/tracking/kalman_models.hpp"
#include "aim/tracking/latency_estimator.hpp"
#include "aim/tracking/prediction_extrapolator.hpp"

namespace aim::tracking {

using bus::TrackState;

struct TrackerConfig {
    float immediate_confirm_confidence{0.85f};
    std::uint32_t confirmation_hits{2};
    std::uint32_t max_missed_frames{3};
    float confidence_decay_rate{0.80f};
    float max_engagement_uncertainty_px{30.0f};
    AssociationConfig association_config{};
    KalmanFilterConfig kalman_config{};
    LatencyEstimatorConfig latency_config{};
    ExtrapolatorConfig extrapolator_config{};
};

struct TrackEntry {
    TrackId track_id{0};
    TrackState state{TrackState::tentative};
    std::uint32_t total_visible_frames{0};
    std::uint32_t total_missed_frames{0};
    std::uint32_t consecutive_hits{0};
    std::uint32_t consecutive_misses{0};
    float effective_radius_px{15.0f};
    float confidence{1.0f};
    float target_value{1.0f};
    std::uint32_t semantic_id{0};
    MonotonicNs last_seen_ns{0};
    AdaptiveKalmanFilter filter{};
    bool is_active{false};
};

/// @brief Production Multi-Target Tracker managing the lifecycle, kinematics, and forward extrapolation of all targets.
/// Allocation-free bounded state machine.
class MultiTargetTracker {
public:
    explicit MultiTargetTracker(TrackerConfig config = {});

    void reset() noexcept;

    /// @brief Record feedback of end-to-end command dispatch latency to update the adaptive predictor.
    void record_latency_sample(MonotonicNs measured_latency_ns, MonotonicNs current_time_ns) noexcept;

    /// @brief Ingest observations and update all active tracks, extrapolating to predicted effect time.
    void process(
        const bus::TargetObservationBatch& observations,
        MonotonicNs current_timestamp_ns,
        bus::TrackedTargetBatch& out_tracks) noexcept;

    /// @brief Ingest observations and update all active tracks, extrapolating to an explicit target timestamp.
    void process(
        const bus::TargetObservationBatch& observations,
        MonotonicNs current_timestamp_ns,
        MonotonicNs prediction_target_time_ns,
        bus::TrackedTargetBatch& out_tracks) noexcept;

    [[nodiscard]] std::size_t active_track_count() const noexcept;

    [[nodiscard]] const TrackerConfig& config() const noexcept { return config_; }

    [[nodiscard]] const CommandEffectLatencyEstimator& latency_estimator() const noexcept {
        return latency_estimator_;
    }

    [[nodiscard]] const TargetExtrapolator& extrapolator() const noexcept {
        return extrapolator_;
    }

    [[nodiscard]] const std::array<TrackEntry, bus::kMaxTrackedTargets>& tracks() const noexcept {
        return tracks_;
    }

private:
    TrackerConfig config_;
    TrackId next_track_id_{1};
    std::array<TrackEntry, bus::kMaxTrackedTargets> tracks_{};
    GatedHungarianAssociator associator_;
    CommandEffectLatencyEstimator latency_estimator_;
    TargetExtrapolator extrapolator_;
    MonotonicNs last_process_time_ns_{0};

    // Pre-allocated internal buffers for association (zero hot-path allocations)
    std::array<TrackHypothesis, bus::kMaxTrackedTargets> hypotheses_{};
    std::array<std::size_t, bus::kMaxTrackedTargets> active_track_indices_{};
    AssociationResult association_result_{};
};

} // namespace aim::tracking
