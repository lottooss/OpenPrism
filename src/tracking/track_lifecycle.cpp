// src/tracking/track_lifecycle.cpp
#include "aim/tracking/track_lifecycle.hpp"

#include <algorithm>
#include <cmath>

namespace aim::tracking {

MultiTargetTracker::MultiTargetTracker(TrackerConfig config)
    : config_(config),
      associator_(config.association_config),
      latency_estimator_(config.latency_config),
      extrapolator_(config.extrapolator_config) {
    reset();
}

void MultiTargetTracker::reset() noexcept {
    next_track_id_ = 1;
    last_process_time_ns_ = 0;
    latency_estimator_.reset();
    for (auto& track : tracks_) {
        track.is_active = false;
        track.filter.reset();
    }
}

void MultiTargetTracker::record_latency_sample(MonotonicNs measured_latency_ns, MonotonicNs current_time_ns) noexcept {
    latency_estimator_.update(measured_latency_ns, current_time_ns);
}

std::size_t MultiTargetTracker::active_track_count() const noexcept {
    std::size_t count = 0;
    for (const auto& track : tracks_) {
        if (track.is_active) {
            ++count;
        }
    }
    return count;
}

void MultiTargetTracker::process(
    const bus::TargetObservationBatch& observations,
    MonotonicNs current_timestamp_ns,
    bus::TrackedTargetBatch& out_tracks) noexcept {
    const MonotonicNs effect_time_ns = latency_estimator_.predict_effect_time(
        observations.captured_at_ns > 0 ? observations.captured_at_ns : current_timestamp_ns,
        current_timestamp_ns
    );
    process(observations, current_timestamp_ns, effect_time_ns, out_tracks);
}

void MultiTargetTracker::process(
    const bus::TargetObservationBatch& observations,
    MonotonicNs current_timestamp_ns,
    MonotonicNs prediction_target_time_ns,
    bus::TrackedTargetBatch& out_tracks) noexcept {
    out_tracks.clear();
    out_tracks.header = observations.header;
    out_tracks.frame_id = observations.frame_id;
    out_tracks.timestamp_ns = current_timestamp_ns;

    // 1. Calculate time delta
    float dt_s = (1.0f / 144.0f);
    if (last_process_time_ns_ > 0 && current_timestamp_ns > last_process_time_ns_) {
        const auto diff_ns = current_timestamp_ns - last_process_time_ns_;
        const float calc_dt = static_cast<float>(diff_ns) * 1e-9f;
        if (calc_dt > 0.0f && calc_dt < 0.5f) {
            dt_s = calc_dt;
        }
    }
    last_process_time_ns_ = current_timestamp_ns;

    // 2. Extrapolate active tracks with Kalman prediction
    std::size_t num_active = 0;
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        if (tracks_[i].is_active) {
            tracks_[i].filter.predict(dt_s);

            const MotionState s = tracks_[i].filter.state();
            hypotheses_[num_active] = TrackHypothesis{
                s.x,
                s.y,
                tracks_[i].effective_radius_px,
                tracks_[i].filter.position_covariance(),
                tracks_[i].semantic_id
            };
            active_track_indices_[num_active] = i;
            ++num_active;
        }
    }

    // 3. Multi-target Hungarian association
    associator_.associate(
        hypotheses_.data(),
        num_active,
        observations.targets.data(),
        observations.target_count,
        association_result_
    );

    // Track which active track indices were matched
    std::array<bool, bus::kMaxTrackedTargets> active_matched{};

    // 4. Update matched tracks
    for (std::uint32_t m = 0; m < association_result_.num_matches; ++m) {
        const auto& match = association_result_.matches[m];
        const std::size_t active_idx = match.track_idx;
        const std::size_t track_idx = active_track_indices_[active_idx];
        const std::size_t obs_idx = match.observation_idx;
        const auto& obs = observations.targets[obs_idx];

        TrackEntry& trk = tracks_[track_idx];
        trk.filter.update(obs.center_px.x, obs.center_px.y, obs.covariance_px2);
        trk.total_visible_frames++;
        trk.consecutive_hits++;
        trk.consecutive_misses = 0;
        trk.effective_radius_px = obs.effective_radius_px;
        trk.confidence = obs.confidence;
        trk.semantic_id = obs.semantic_id;
        trk.last_seen_ns = current_timestamp_ns;

        // Lifecycle transition: tentative -> confirmed on consecutive hits or immediate threshold
        if (trk.state == TrackState::tentative) {
            if (trk.consecutive_hits >= config_.confirmation_hits ||
                obs.confidence >= config_.immediate_confirm_confidence) {
                trk.state = TrackState::confirmed;
            }
        } else if (trk.state == TrackState::occluded || trk.state == TrackState::coasting) {
            trk.state = TrackState::confirmed;
        }

        active_matched[active_idx] = true;
    }

    // 5. Handle unassigned (missed) tracks
    for (std::uint32_t u = 0; u < association_result_.num_unassigned_tracks; ++u) {
        const std::size_t active_idx = association_result_.unassigned_tracks[u];
        const std::size_t track_idx = active_track_indices_[active_idx];
        TrackEntry& trk = tracks_[track_idx];

        trk.total_missed_frames++;
        trk.consecutive_misses++;
        trk.consecutive_hits = 0;
        trk.confidence *= config_.confidence_decay_rate;

        if (trk.state == TrackState::tentative) {
            // Tentative track missing an observation is pruned immediately (reject noise/ghosts)
            trk.state = TrackState::lost;
            trk.is_active = false;
        } else if (trk.state == TrackState::confirmed) {
            // Confirmed track enters occlusion retention
            trk.state = TrackState::occluded;
        } else if (trk.state == TrackState::occluded || trk.state == TrackState::coasting) {
            if (trk.consecutive_misses > config_.max_missed_frames) {
                trk.state = TrackState::lost;
                trk.is_active = false;
            }
        }
    }

    // 6. Handle unassigned observations (Initialize new tracks)
    for (std::uint32_t u = 0; u < association_result_.num_unassigned_observations; ++u) {
        const std::size_t obs_idx = association_result_.unassigned_observations[u];
        const auto& obs = observations.targets[obs_idx];

        // Find free slot
        for (std::size_t i = 0; i < tracks_.size(); ++i) {
            if (!tracks_[i].is_active) {
                TrackEntry& trk = tracks_[i];
                trk.track_id = next_track_id_++;
                trk.is_active = true;
                trk.total_visible_frames = 1;
                trk.total_missed_frames = 0;
                trk.consecutive_hits = 1;
                trk.consecutive_misses = 0;
                trk.effective_radius_px = obs.effective_radius_px;
                trk.confidence = obs.confidence;
                trk.target_value = 1.0f;
                trk.semantic_id = obs.semantic_id;
                trk.last_seen_ns = current_timestamp_ns;
                trk.filter.initialize(obs.center_px.x, obs.center_px.y, obs.covariance_px2);

                if (obs.confidence >= config_.immediate_confirm_confidence) {
                    trk.state = TrackState::confirmed;
                } else {
                    trk.state = TrackState::tentative;
                }
                break;
            }
        }
    }

    // 7. Output published TrackedTargetBatch with Forward Prediction
    const bool is_stable = latency_estimator_.estimate().is_stable || (prediction_target_time_ns > current_timestamp_ns);

    for (const auto& trk : tracks_) {
        if (!trk.is_active) {
            continue;
        }

        bus::TrackedTarget target{};
        target.track_id = trk.track_id;
        target.state = trk.state;
        target.total_visible_frames = trk.total_visible_frames;
        target.total_missed_frames = trk.total_missed_frames;

        const MotionState s = trk.filter.state();
        target.filtered_center_px = PixelPoint{s.x, s.y};
        target.filtered_velocity_px_per_s = PixelVelocity{s.vx, s.vy};
        target.filtered_acceleration_px_per_s2 = PixelAcceleration{s.ax, s.ay};
        target.covariance_px2 = trk.filter.position_covariance();

        // Forward extrapolation
        const auto extrap = extrapolator_.extrapolate(
            s,
            target.covariance_px2,
            current_timestamp_ns,
            prediction_target_time_ns,
            is_stable
        );

        target.predicted_center_px = extrap.predicted_center_px;
        target.prediction_time_ns = extrap.target_time_ns;
        target.covariance_px2 = extrap.predicted_covariance_px2;

        target.effective_radius_px = trk.effective_radius_px;
        target.confidence = trk.confidence;
        target.target_value = trk.target_value;
        target.semantic_id = trk.semantic_id;

        out_tracks.add_track(target);
    }
}

} // namespace aim::tracking
