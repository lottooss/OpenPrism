#include "aim/policy/shot_gate.hpp"

namespace aim::policy {

ShotGateResult
ShotGate::evaluate(const bus::TrackedTarget *target, PixelPoint crosshair,
                   MonotonicNs now_ns,
                   const ISafetySupervisor *supervisor) noexcept {
  ShotGateResult result{};
  const auto reject = [&result](ShotGateDecision decision) {
    result.decision = decision;
    return result;
  };
  if (!target)
    return reject(ShotGateDecision::rejected_no_target);
  if (supervisor && supervisor->is_latched()) {
    return reject(ShotGateDecision::rejected_safety_latched);
  }

  const auto &cov = target->covariance_px2;
  const double determinant = static_cast<double>(cov.xx) * cov.yy -
                             static_cast<double>(cov.xy) * cov.xy;
  if (target->track_id == 0 || now_ns <= 0 ||
      !std::isfinite(target->confidence) || target->confidence < 0.0f ||
      target->confidence > 1.0f ||
      !std::isfinite(target->effective_radius_px) ||
      target->effective_radius_px <= 0.0f ||
      !std::isfinite(target->predicted_center_px.x) ||
      !std::isfinite(target->predicted_center_px.y) ||
      !std::isfinite(crosshair.x) || !std::isfinite(crosshair.y) ||
      !std::isfinite(cov.xx) || !std::isfinite(cov.xy) ||
      !std::isfinite(cov.yy) || cov.xx < 0.0f || cov.yy < 0.0f ||
      determinant < 0.0 || !std::isfinite(config_.min_confidence) ||
      config_.min_confidence < 0.0f || config_.min_confidence > 1.0f ||
      !std::isfinite(config_.max_alignment_radius_ratio) ||
      config_.max_alignment_radius_ratio <= 0.0f ||
      config_.max_alignment_radius_ratio > 1.0f ||
      !std::isfinite(config_.max_uncertainty_radius_ratio) ||
      config_.max_uncertainty_radius_ratio < 0.0f ||
      config_.max_uncertainty_radius_ratio > 1.0f ||
      config_.max_observation_age_ns <= 0 ||
      config_.max_observation_age_ns > 10'000'000LL ||
      config_.max_prediction_lead_ns <= 0 ||
      config_.max_prediction_lead_ns > 100'000'000LL ||
      config_.min_shot_interval_ns <= 0 ||
      config_.same_target_retry_ns < config_.min_shot_interval_ns ||
      config_.fire_button < MouseButton::left ||
      config_.fire_button > MouseButton::extra2) {
    return reject(ShotGateDecision::rejected_invalid_input);
  }

  // The host also gates capture age: prediction time does not establish
  // freshness.
  if (target->prediction_time_ns <= 0 ||
      (target->prediction_time_ns <= now_ns &&
       now_ns - target->prediction_time_ns > config_.max_observation_age_ns) ||
      (target->prediction_time_ns > now_ns &&
       target->prediction_time_ns - now_ns > config_.max_prediction_lead_ns)) {
    return reject(ShotGateDecision::rejected_deadline_expired);
  }
  if (target->confidence < config_.min_confidence) {
    return reject(ShotGateDecision::rejected_low_confidence);
  }
  result.effective_radius_px = target->effective_radius_px;
  const float sigma = compute_uncertainty_sigma(cov);
  result.uncertainty_sigma_px = sigma;
  if (!std::isfinite(sigma) ||
      sigma >
          target->effective_radius_px * config_.max_uncertainty_radius_ratio) {
    return reject(ShotGateDecision::rejected_uncertainty_exceeded);
  }
  const float distance =
      std::hypot(target->predicted_center_px.x - crosshair.x,
                 target->predicted_center_px.y - crosshair.y);
  result.alignment_distance_px = distance;
  // Contain the uncertainty disk as well as the aim point. Independent radius
  // tests previously authorized uncertain shots beyond the target edge.
  if (!std::isfinite(distance) ||
      distance + sigma >
          target->effective_radius_px * config_.max_alignment_radius_ratio) {
    return reject(ShotGateDecision::rejected_alignment_miss);
  }
  // total_missed_frames is lifetime history, not current visibility. A matched
  // observation restores confirmed state after a short occlusion.
  if (target->state != bus::TrackState::confirmed ||
      target->total_visible_frames == 0) {
    return reject(ShotGateDecision::rejected_no_fresh_observation);
  }
  if (has_shot_ && (now_ns < last_shot_ns_ ||
                    now_ns - last_shot_ns_ < config_.min_shot_interval_ns)) {
    return reject(ShotGateDecision::rejected_cooldown);
  }
  for (const auto &receipt : receipts_) {
    if (receipt.track_id != target->track_id)
      continue;
    if (target->total_visible_frames <= receipt.visible_frames) {
      return reject(ShotGateDecision::rejected_no_fresh_observation);
    }
    if (now_ns < receipt.dispatched_at_ns ||
        now_ns - receipt.dispatched_at_ns < config_.same_target_retry_ns) {
      return reject(ShotGateDecision::rejected_cooldown);
    }
  }
  result.decision = ShotGateDecision::authorized;
  result.authorize_fire = true;
  result.button_transition = {config_.fire_button, ButtonAction::click};
  return result;
}

void ShotGate::record_dispatched_shot(const bus::TrackedTarget &target,
                                      MonotonicNs now_ns) noexcept {
  auto *slot = &receipts_[0];
  for (auto &receipt : receipts_) {
    if (receipt.track_id == target.track_id) {
      slot = &receipt;
      break;
    }
    if (receipt.dispatched_at_ns < slot->dispatched_at_ns)
      slot = &receipt;
  }
  *slot = {target.track_id, now_ns, target.total_visible_frames};
  last_shot_ns_ = now_ns;
  has_shot_ = true;
}

} // namespace aim::policy
