// include/aim/tracking/tracking_engine.hpp
#pragma once

#include "aim/interfaces/tracking_engine.hpp"
#include "aim/tracking/track_lifecycle.hpp"

namespace aim::tracking {

/// @brief Production implementation of ITrackingEngine wrapping MultiTargetTracker.
class TrackingEngine final : public ITrackingEngine {
public:
    TrackingEngine() = default;

    explicit TrackingEngine(const aim::tracking::TrackerConfig& config)
        : tracker_(config), initialized_(true) {}

    bool initialize(const aim::TrackerConfig& config) noexcept override {
        aim::tracking::TrackerConfig trk_cfg{};
        trk_cfg.association_config.gate_distance_px = config.gate_distance_px;
        trk_cfg.immediate_confirm_confidence = config.min_init_confidence;
        trk_cfg.max_missed_frames = config.max_coasting_frames;
        trk_cfg.kalman_config.q_pos = config.position_process_noise;
        trk_cfg.kalman_config.q_vel = config.velocity_process_noise;
        trk_cfg.kalman_config.r_default = config.measurement_noise;
        tracker_ = MultiTargetTracker{trk_cfg};
        initialized_ = true;
        health_ = {};
        return true;
    }

    bool initialize(const aim::tracking::TrackerConfig& config) noexcept {
        tracker_ = MultiTargetTracker{config};
        initialized_ = true;
        health_ = {};
        return true;
    }

    void reset() noexcept override {
        tracker_.reset();
        health_ = {};
    }

    bool update(const bus::TargetObservationBatch& observations,
                MonotonicNs prediction_target_time_ns,
                bus::TrackedTargetBatch& out_tracks) noexcept override {
        if (!initialized_) {
            return false;
        }
        const MonotonicNs current_ns = (observations.captured_at_ns > 0)
            ? observations.captured_at_ns
            : prediction_target_time_ns;
        tracker_.process(observations, current_ns, prediction_target_time_ns, out_tracks);
        health_.active_tracks = static_cast<std::uint32_t>(tracker_.active_track_count());
        health_.total_associations += out_tracks.track_count;
        return true;
    }

    [[nodiscard]] TrackerHealth health() const noexcept override {
        return health_;
    }

    [[nodiscard]] MultiTargetTracker& inner() noexcept { return tracker_; }
    [[nodiscard]] const MultiTargetTracker& inner() const noexcept { return tracker_; }

private:
    MultiTargetTracker tracker_{};
    TrackerHealth health_{};
    bool initialized_{true};
};

} // namespace aim::tracking
