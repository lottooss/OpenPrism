// include/aim/policy/shot_gate.hpp
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/actuator.hpp"
#include "aim/core/safety.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/interfaces/safety_supervisor.hpp"

namespace aim::policy {

enum class ShotGateDecision : std::uint8_t {
  authorized = 0,
  rejected_uncertainty_exceeded = 1,
  rejected_low_confidence = 2,
  rejected_alignment_miss = 3,
  rejected_deadline_expired = 4,
  rejected_safety_latched = 5,
  rejected_no_target = 6,
  rejected_invalid_input = 7,
  rejected_cooldown = 8,
  rejected_no_fresh_observation = 9,
  rejected_policy_veto = 10
};

[[nodiscard]] constexpr std::string_view
shot_gate_decision_to_string(ShotGateDecision decision) noexcept {
  switch (decision) {
  case ShotGateDecision::authorized:
    return "authorized";
  case ShotGateDecision::rejected_uncertainty_exceeded:
    return "rejected_uncertainty_exceeded";
  case ShotGateDecision::rejected_low_confidence:
    return "rejected_low_confidence";
  case ShotGateDecision::rejected_alignment_miss:
    return "rejected_alignment_miss";
  case ShotGateDecision::rejected_deadline_expired:
    return "rejected_deadline_expired";
  case ShotGateDecision::rejected_safety_latched:
    return "rejected_safety_latched";
  case ShotGateDecision::rejected_no_target:
    return "rejected_no_target";
  case ShotGateDecision::rejected_invalid_input:
    return "rejected_invalid_input";
  case ShotGateDecision::rejected_cooldown:
    return "rejected_cooldown";
  case ShotGateDecision::rejected_no_fresh_observation:
    return "rejected_no_fresh_observation";
  case ShotGateDecision::rejected_policy_veto:
    return "rejected_policy_veto";
  }
  return "unknown";
}

struct ShotGateConfig {
  float min_confidence{0.80f}; // Minimum target detection confidence
  float max_uncertainty_radius_ratio{
      1.0f}; // Uncertainty ellipse semi-major axis <= effective_radius * ratio
  float max_alignment_radius_ratio{
      0.80f}; // Alignment plus uncertainty must fit inside this radius
  MonotonicNs max_observation_age_ns{
      10'000'000LL}; // 10 ms hard deadline cutoff
  MouseButton fire_button{MouseButton::left};
  MonotonicNs min_shot_interval_ns{150'000'000LL};
  MonotonicNs same_target_retry_ns{350'000'000LL};
  // Measured command-effect prediction horizon, bounded to 100 ms. This
  // never extends the separate hard 10 ms source-observation age cutoff.
  MonotonicNs max_prediction_lead_ns{10'000'000LL};
};

struct ShotGateResult {
  ShotGateDecision decision{ShotGateDecision::rejected_no_target};
  bool authorize_fire{false};
  float uncertainty_sigma_px{0.0f};
  float alignment_distance_px{0.0f};
  float effective_radius_px{0.0f};
  ButtonTransition button_transition{};
};

/// @brief Uncertainty-aware engagement and shot gating (Milestone M7-02).
/// Gates trigger decision based on covariance bounds, target radius,
/// confidence, crosshair alignment, timing cutoff, and centralized safety
/// state.
class ShotGate {
public:
  explicit ShotGate(const ShotGateConfig &config = {}) noexcept
      : config_(config) {}

  [[nodiscard]] ShotGateResult
  evaluate(const bus::TrackedTarget *target, PixelPoint current_crosshair_px,
           MonotonicNs now_ns,
           const ISafetySupervisor *supervisor = nullptr) noexcept;

  /// Commit only after the actuator accepts a shot. Single-thread owned;
  /// evaluating a candidate must not consume cadence on failed dispatch.
  void record_dispatched_shot(const bus::TrackedTarget &target,
                              MonotonicNs now_ns) noexcept;

  [[nodiscard]] static float
  compute_uncertainty_sigma(const Covariance2D &cov) noexcept {
    const float tr = cov.xx + cov.yy;
    const float diff = (cov.xx - cov.yy) * 0.5f;
    const float disc =
        std::sqrt((std::max)(0.0f, diff * diff + cov.xy * cov.xy));
    const float lambda_max = tr * 0.5f + disc;
    return std::sqrt((std::max)(0.0f, lambda_max));
  }

  void reset() noexcept {
    receipts_ = {};
    last_shot_ns_ = 0;
    has_shot_ = false;
  }

private:
  ShotGateConfig config_{};
  struct ShotReceipt {
    TrackId track_id{0};
    MonotonicNs dispatched_at_ns{0};
    std::uint32_t visible_frames{0};
  };
  std::array<ShotReceipt, bus::kMaxTrackedTargets> receipts_{};
  MonotonicNs last_shot_ns_{0};
  bool has_shot_{false};
};

} // namespace aim::policy
