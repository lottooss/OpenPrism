// src/policy/utility_policy.cpp
#include "aim/policy/utility_policy.hpp"

#include <cmath>
#include <limits>

namespace aim::policy {

namespace {

NormalizedPoint to_normalized_coords(const PixelPoint& pt, float screen_w, float screen_h) noexcept {
    const float half_w = screen_w * 0.5f;
    const float half_h = screen_h * 0.5f;
    const float norm_x = (pt.x - half_w) / half_w;
    const float norm_y = (pt.y - half_h) / half_h;
    return NormalizedPoint{
        std::clamp(norm_x, -1.0f, 1.0f),
        std::clamp(norm_y, -1.0f, 1.0f)
    };
}

} // namespace

bool UtilityAimPolicy::choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept {
    out_intent = bus::AimIntent{};
    out_intent.header = input.correlation_id;
    last_decision_time_ns_ = input.decision_time_ns;

    if (input.tracks == nullptr || input.tracks->track_count == 0) {
        current_target_id_ = 0;
        return false;
    }

    const auto& batch = *input.tracks;
    const PixelPoint crosshair = input.crosshair.center_px;

    float best_utility = -std::numeric_limits<float>::infinity();
    const bus::TrackedTarget* best_target = nullptr;

    for (std::uint32_t i = 0; i < batch.track_count; ++i) {
        const auto& tgt = batch.tracks[i];

        // 1. Skip lost or non-engageable tracks
        if (tgt.state == bus::TrackState::lost) {
            continue;
        }

        // 2. Minimum confidence gate
        if (tgt.confidence < weights_.min_engagement_confidence) {
            continue;
        }

        // 3. Compute distance to predicted position
        const float dx = tgt.predicted_center_px.x - crosshair.x;
        const float dy = tgt.predicted_center_px.y - crosshair.y;
        const float dist_px = std::hypot(dx, dy);

        // 4. Uncertainty penalty (trace of covariance)
        const float uncertainty = std::sqrt(std::max(0.0f, tgt.covariance_px2.xx + tgt.covariance_px2.yy));

        // 5. Confirmation bonus / occlusion penalty
        float state_multiplier = 1.0f;
        if (tgt.state == bus::TrackState::confirmed) {
            state_multiplier = 1.2f;
        } else if (tgt.state == bus::TrackState::occluded) {
            state_multiplier = 0.5f;
        }

        // 6. Utility formulation: U = W_v * V + W_c * C - W_d * D - W_u * U_cov + Hysteresis
        float utility = weights_.weight_value * tgt.target_value * state_multiplier +
                        weights_.weight_confidence * tgt.confidence -
                        weights_.weight_distance * dist_px -
                        weights_.weight_uncertainty * uncertainty;

        // Switch hysteresis bonus for currently locked target
        if (current_target_id_ != 0 && tgt.track_id == current_target_id_) {
            utility += weights_.switch_hysteresis_bonus;
        }

        if (utility > best_utility) {
            best_utility = utility;
            best_target = &tgt;
        }
    }

    if (best_target == nullptr) {
        current_target_id_ = 0;
        return false;
    }

    // Populate AimIntent
    current_target_id_ = best_target->track_id;
    out_intent.target_track_id = best_target->track_id;

    const float err_x = best_target->predicted_center_px.x - crosshair.x;
    const float err_y = best_target->predicted_center_px.y - crosshair.y;
    out_intent.error_distance_px = std::hypot(err_x, err_y);

    if (out_intent.error_distance_px > 40.0f) {
        out_intent.mode = bus::AimMode::flick;
    } else if (out_intent.error_distance_px <= weights_.fire_authorization_threshold_px) {
        out_intent.mode = bus::AimMode::micro_correction;
    } else {
        out_intent.mode = bus::AimMode::tracking;
    }

    out_intent.target_aim_px = best_target->predicted_center_px;
    out_intent.target_aim_norm = to_normalized_coords(best_target->predicted_center_px, weights_.screen_width_px, weights_.screen_height_px);

    // Lead offset = predicted - filtered
    out_intent.lead_offset_px = PixelPoint{
        best_target->predicted_center_px.x - best_target->filtered_center_px.x,
        best_target->predicted_center_px.y - best_target->filtered_center_px.y
    };

    out_intent.confidence = best_target->confidence;
    out_intent.utility_score = best_utility;

    // Fire authorization: on target within threshold, confirmed track, high confidence
    out_intent.authorize_fire = (out_intent.error_distance_px <= weights_.fire_authorization_threshold_px) &&
                                (best_target->state == bus::TrackState::confirmed) &&
                                (best_target->confidence >= 0.70f);

    out_intent.command_deadline_ns = input.decision_time_ns + 10'000'000LL; // 10 ms deadline
    return true;
}

bool NearestTargetPolicy::choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept {
    out_intent = bus::AimIntent{};
    out_intent.header = input.correlation_id;

    if (input.tracks == nullptr || input.tracks->track_count == 0) {
        return false;
    }

    const auto& batch = *input.tracks;
    const PixelPoint crosshair = input.crosshair.center_px;

    float min_dist_sq = std::numeric_limits<float>::infinity();
    const bus::TrackedTarget* nearest_target = nullptr;

    for (std::uint32_t i = 0; i < batch.track_count; ++i) {
        const auto& tgt = batch.tracks[i];
        if (tgt.state == bus::TrackState::lost) continue;

        const float dx = tgt.predicted_center_px.x - crosshair.x;
        const float dy = tgt.predicted_center_px.y - crosshair.y;
        const float dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_target = &tgt;
        }
    }

    if (nearest_target == nullptr) {
        return false;
    }

    out_intent.target_track_id = nearest_target->track_id;
    out_intent.mode = bus::AimMode::tracking;
    out_intent.target_aim_px = nearest_target->predicted_center_px;
    out_intent.target_aim_norm = to_normalized_coords(nearest_target->predicted_center_px, 1920.0f, 1080.0f);
    out_intent.error_distance_px = std::sqrt(min_dist_sq);
    out_intent.confidence = nearest_target->confidence;
    out_intent.utility_score = 1.0f;
    out_intent.authorize_fire = (out_intent.error_distance_px <= 8.0f);
    return true;
}

bool SpatialSweepPolicy::choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept {
    out_intent = bus::AimIntent{};
    out_intent.header = input.correlation_id;

    if (input.tracks == nullptr || input.tracks->track_count == 0) {
        return false;
    }

    const auto& batch = *input.tracks;
    const PixelPoint crosshair = input.crosshair.center_px;

    float leftmost_x = std::numeric_limits<float>::infinity();
    const bus::TrackedTarget* leftmost_target = nullptr;

    for (std::uint32_t i = 0; i < batch.track_count; ++i) {
        const auto& tgt = batch.tracks[i];
        if (tgt.state == bus::TrackState::lost) continue;

        if (tgt.predicted_center_px.x < leftmost_x) {
            leftmost_x = tgt.predicted_center_px.x;
            leftmost_target = &tgt;
        }
    }

    if (leftmost_target == nullptr) {
        return false;
    }

    out_intent.target_track_id = leftmost_target->track_id;
    out_intent.mode = bus::AimMode::tracking;
    out_intent.target_aim_px = leftmost_target->predicted_center_px;
    out_intent.target_aim_norm = to_normalized_coords(leftmost_target->predicted_center_px, 1920.0f, 1080.0f);
    const float dx = leftmost_target->predicted_center_px.x - crosshair.x;
    const float dy = leftmost_target->predicted_center_px.y - crosshair.y;
    out_intent.error_distance_px = std::hypot(dx, dy);
    out_intent.confidence = leftmost_target->confidence;
    out_intent.utility_score = 1.0f;
    out_intent.authorize_fire = (out_intent.error_distance_px <= 8.0f);
    return true;
}

} // namespace aim::policy
