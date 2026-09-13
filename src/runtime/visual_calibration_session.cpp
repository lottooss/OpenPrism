#include "aim/runtime/visual_calibration_session.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aim::runtime {
namespace {
constexpr MonotonicNs kFreshnessNs = 10'000'000;
constexpr double kMaximumSceneTranslationPx = 200.0;
constexpr double kMaximumSceneResidualPx = 3.0;
constexpr double kGeometryEpsilonPx = 1.0e-4;
constexpr std::size_t kMaximumSceneSearchWork = 4096;
// Motion blur can briefly lower a real target below the measurement floor,
// while the Aim Lab detector's unrelated weapon false positive is around
// 0.27-0.34.  This floor is used only for complete-scene geometry matching;
// samples and pulse decisions continue to require the configured .8 floor.
constexpr float kGeometryConfidenceFloor = 0.5f;

double separation(PixelPoint a, PixelPoint b) noexcept {
  return std::hypot(static_cast<double>(a.x) - b.x,
                    static_cast<double>(a.y) - b.y);
}
bool finite_between(float value, float low, float high) noexcept {
  return std::isfinite(value) && value >= low && value <= high;
}
double largest_variance(const Covariance2D &covariance) noexcept {
  const double difference = static_cast<double>(covariance.xx) - covariance.yy;
  return 0.5 * (static_cast<double>(covariance.xx) + covariance.yy +
                std::sqrt(difference * difference +
                          4.0 * covariance.xy * covariance.xy));
}
} // namespace

VisualCalibrationSession::VisualCalibrationSession(
    const VisualCalibrationSessionConfig &config) noexcept
    : config_(config), fitter_(config.fitting) {}

bool VisualCalibrationSession::start(MonotonicNs now_ns) noexcept {
  // Fixed pulse schedule identifies shared deadband and both signed axes. Its
  // largest pulse is 64 counts regardless of any wider downstream limits.
  if (!fitter_.configuration_valid() || now_ns <= 0 ||
      config_.total_deadline_ns <= 0 ||
      config_.total_deadline_ns > 30'000'000'000 ||
      config_.fitting.max_pulse_counts < 32 ||
      (config_.horizontal_fov_deg != 0.0f &&
       !finite_between(config_.horizontal_fov_deg, 30.0f, 150.0f)) ||
      !finite_between(config_.crosshair_px.x, 0.0f,
                      static_cast<float>(config_.fitting.width - 1)) ||
      !finite_between(config_.crosshair_px.y, 0.0f,
                      static_cast<float>(config_.fitting.height - 1)) ||
      !finite_between(config_.selection_radius_px, 1.0f, 1000.0f) ||
      !finite_between(config_.association_radius_px, 1.0f, 200.0f) ||
      !finite_between(config_.maximum_radius_change_fraction, 0.0f, 0.25f)) {
    (void)fail(VisualCalibrationStatus::invalid_configuration);
    return false;
  }
  fitter_.reset();
  scene_focal_px_ =
      config_.horizontal_fov_deg == 0.0f
          ? 0.0
          : 0.5 * config_.fitting.width /
                std::tan(static_cast<double>(config_.horizontal_fov_deg) *
                         3.14159265358979323846 / 360.0);
  started_at_ns_ = last_now_ns_ = now_ns;
  last_frame_ns_ = 0;
  last_frame_id_ = 0;
  association_id_ = 0;
  pulse_index_ = 0;
  sample_ = {};
  scene_count_ = 0;
  scene_match_confident_ = false;
  state_ = CalibrationSessionState::seeking_target;
  status_ = VisualCalibrationStatus::ok;
  return true;
}

CalibrationSessionUpdate
VisualCalibrationSession::fail(VisualCalibrationStatus status) noexcept {
  state_ = CalibrationSessionState::failed;
  status_ = status;
  return update();
}

CalibrationSessionUpdate
VisualCalibrationSession::poll(MonotonicNs now_ns, bool authorized) noexcept {
  if (state_ == CalibrationSessionState::idle || complete() || failed())
    return update();
  if (!authorized)
    return fail(VisualCalibrationStatus::lost_focus);
  if (now_ns < last_now_ns_ || now_ns < started_at_ns_ ||
      now_ns - started_at_ns_ >= config_.total_deadline_ns)
    return fail(VisualCalibrationStatus::invalid_timing);
  last_now_ns_ = now_ns;
  if (state_ == CalibrationSessionState::awaiting_dispatch &&
      now_ns - sample_.before.captured_at_ns > kFreshnessNs)
    return fail(VisualCalibrationStatus::invalid_timing);
  if (state_ == CalibrationSessionState::awaiting_response &&
      now_ns - sample_.dispatched_at_ns >
          config_.fitting.maximum_effect_delay_ns)
    return fail(VisualCalibrationStatus::insufficient_response);
  if (state_ == CalibrationSessionState::settling &&
      now_ns - sample_.baseline.captured_at_ns >
          config_.fitting.maximum_sample_duration_ns)
    return fail(VisualCalibrationStatus::moving_target);
  return update();
}

bool VisualCalibrationSession::eligible(
    const bus::TargetObservation &observation,
    float minimum_confidence) const noexcept {
  const auto &fit = config_.fitting;
  const auto &covariance = observation.covariance_px2;
  if (observation.visibility != Visibility::visible ||
      !finite_between(observation.confidence, minimum_confidence, 1.0f) ||
      !finite_between(observation.center_px.x, 0.0f,
                      static_cast<float>(fit.width - 1)) ||
      !finite_between(observation.center_px.y, 0.0f,
                      static_cast<float>(fit.height - 1)) ||
      !finite_between(observation.effective_radius_px, 2.0f, 100.0f) ||
      !finite_between(covariance.xx, 0.0f,
                      fit.maximum_uncertainty_px *
                          fit.maximum_uncertainty_px) ||
      !finite_between(covariance.yy, 0.0f,
                      fit.maximum_uncertainty_px *
                          fit.maximum_uncertainty_px) ||
      !std::isfinite(covariance.xy) ||
      static_cast<double>(covariance.xy) * covariance.xy >
          static_cast<double>(covariance.xx) * covariance.yy ||
      largest_variance(covariance) >
          static_cast<double>(fit.maximum_uncertainty_px) *
              fit.maximum_uncertainty_px)
    return false;
  const auto &box = observation.bbox_px;
  if (!std::isfinite(box.left) || !std::isfinite(box.top) ||
      !std::isfinite(box.right) || !std::isfinite(box.bottom))
    return false;
  const double width = static_cast<double>(box.right) - box.left;
  const double height = static_cast<double>(box.bottom) - box.top;
  return width > 2.0 && height > 2.0 && width / height >= 0.70 &&
         width / height <= 1.43 && observation.center_px.x >= box.left &&
         observation.center_px.x <= box.right &&
         observation.center_px.y >= box.top &&
         observation.center_px.y <= box.bottom;
}

bool VisualCalibrationSession::eligible(
    const bus::TargetObservation &observation) const noexcept {
  return eligible(observation, config_.fitting.minimum_confidence);
}

bool VisualCalibrationSession::geometry_eligible(
    const bus::TargetObservation &observation) const noexcept {
  return eligible(observation, kGeometryConfidenceFloor);
}

const bus::TargetObservation *VisualCalibrationSession::select(
    const bus::TargetObservationBatch &batch) const noexcept {
  const bus::TargetObservation *nearest = nullptr;
  double nearest_distance = std::numeric_limits<double>::infinity();
  bool tied = false;
  for (const auto &candidate : batch.items()) {
    if (!eligible(candidate))
      continue;
    const double distance =
        separation(candidate.center_px, config_.crosshair_px);
    if (distance >
        static_cast<double>(config_.selection_radius_px) + kGeometryEpsilonPx)
      continue;
    if (distance + kGeometryEpsilonPx < nearest_distance) {
      nearest = &candidate;
      nearest_distance = distance;
      tied = false;
    } else if (std::abs(distance - nearest_distance) <= kGeometryEpsilonPx) {
      tied = true;
    }
  }
  if (nearest == nullptr || tied)
    return nullptr;

  // Preserve the one-target path only when the nearest candidate is
  // genuinely isolated. A farther isolated target must not win over a
  // nearer clustered target whose identity requires scene matching.
  for (const auto &other : batch.items()) {
    if (&other != nearest && eligible(other) &&
        separation(nearest->center_px, other.center_px) <=
            2.0 * static_cast<double>(config_.association_radius_px))
      return nullptr;
  }
  return nearest;
}

const bus::TargetObservation *VisualCalibrationSession::associate(
    const bus::TargetObservationBatch &batch) const noexcept {
  const bus::TargetObservation *match = nullptr;
  for (const auto &candidate : batch.items()) {
    if (!eligible(candidate) || candidate.semantic_id != semantic_id_ ||
        separation(candidate.center_px, association_center_) >
            config_.association_radius_px ||
        std::abs(candidate.effective_radius_px - association_radius_) >
            association_radius_ * config_.maximum_radius_change_fraction)
      continue;
    if (match != nullptr)
      return nullptr; // closest is insufficient when identity is
                      // ambiguous
    match = &candidate;
  }
  return match;
}

const bus::TargetObservation *VisualCalibrationSession::select_scene(
    const bus::TargetObservationBatch &batch) noexcept {
  scene_count_ = 0;
  double closest = std::numeric_limits<double>::infinity();
  bool selected = false;
  bool tied = false;
  for (const auto &candidate : batch.items()) {
    if (!eligible(candidate))
      continue;
    if (scene_count_ == kMaxSceneAnchors) {
      scene_count_ = 0;
      return nullptr;
    }
    const double distance =
        separation(candidate.center_px, config_.crosshair_px);
    if (distance + kGeometryEpsilonPx < closest) {
      closest = distance;
      selected_scene_index_ = scene_count_;
      selected = true;
      tied = false;
    } else if (std::abs(distance - closest) <= kGeometryEpsilonPx) {
      tied = true;
    }
    reference_scene_[scene_count_++] = {candidate.center_px,
                                        candidate.effective_radius_px,
                                        candidate.semantic_id};
  }
  if (scene_count_ < 3 || !selected || tied ||
      closest > static_cast<double>(config_.selection_radius_px) +
                    kGeometryEpsilonPx) {
    scene_count_ = 0;
    return nullptr;
  }
  // Even the initial snapshot must have distinguishable anchors. Later
  // stable frames independently validate this pattern before offering input.
  const auto *target = associate_scene(batch, false);
  if (!target)
    scene_count_ = 0;
  return target;
}

const bus::TargetObservation *VisualCalibrationSession::associate_scene(
    const bus::TargetObservationBatch &batch,
    bool allow_low_confidence) noexcept {
  std::array<const bus::TargetObservation *, bus::kMaxObservations>
      candidates{};
  std::size_t count = 0;
  std::size_t confident_count = 0;
  for (const auto &candidate : batch.items()) {
    if (eligible(candidate))
      ++confident_count;
    if (!(allow_low_confidence ? geometry_eligible(candidate)
                               : eligible(candidate)))
      continue;
    if (count == candidates.size())
      return nullptr;
    candidates[count++] = &candidate;
  }
  // No partial matches, dropped anchors or extra eligible detections.
  if (count < scene_count_ || scene_count_ < 3 ||
      (!allow_low_confidence && count != scene_count_) ||
      (allow_low_confidence && confident_count > scene_count_))
    return nullptr;
  const auto compatible = [this, &candidates](std::size_t i, std::size_t j) {
    const auto &reference = reference_scene_[i];
    return candidates[j]->semantic_id == reference.semantic_id &&
           std::abs(candidates[j]->effective_radius_px - reference.radius) <=
               reference.radius * config_.maximum_radius_change_fraction;
  };
  const bool camera = scene_focal_px_ > 0.0 &&
                      (state_ == CalibrationSessionState::awaiting_response ||
                       state_ == CalibrationSessionState::settling);
  const bool horizontal = sample_.counts_x != 0;
  const auto displacement = [&](std::size_t i, std::size_t j) -> PixelPoint {
    const auto a = reference_scene_[i].center;
    const auto b = candidates[j]->center_px;
    if (!camera)
      return {b.x - a.x, b.y - a.y};
    const double origin =
        horizontal ? config_.crosshair_px.x : config_.crosshair_px.y;
    const double u = ((horizontal ? a.x : a.y) - origin) / scene_focal_px_;
    const double v = ((horizontal ? b.x : b.y) - origin) / scene_focal_px_;
    const double denominator = 1.0 + u * v;
    const float shift =
        denominator > 0.01
            ? static_cast<float>(scene_focal_px_ * (v - u) / denominator)
            : std::numeric_limits<float>::infinity();
    return horizontal ? PixelPoint{shift, 0.0f} : PixelPoint{0.0f, shift};
  };
  const auto project = [&](std::size_t i, PixelPoint shift) -> PixelPoint {
    const auto p = reference_scene_[i].center;
    if (!camera)
      return {p.x + shift.x, p.y + shift.y};
    const double u = (p.x - config_.crosshair_px.x) / scene_focal_px_;
    const double v = (p.y - config_.crosshair_px.y) / scene_focal_px_;
    const double tangent = -(horizontal ? shift.x : shift.y) / scene_focal_px_;
    const double denominator = 1.0 + (horizontal ? u : v) * tangent;
    if (!std::isfinite(denominator) || denominator <= 0.01)
      return {std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity()};
    const double scale = std::sqrt(1.0 + tangent * tangent);
    return {static_cast<float>(config_.crosshair_px.x +
                               scene_focal_px_ *
                                   (horizontal ? u - tangent : u * scale) /
                                   denominator),
            static_cast<float>(config_.crosshair_px.y +
                               scene_focal_px_ *
                                   (horizontal ? v * scale : v - tangent) /
                                   denominator)};
  };
  std::array<std::size_t, kMaxSceneAnchors> accepted_mapping{};
  bool accepted_confident = false;
  bool found = false;
  bool ambiguous = false;
  bool search_exhausted = false;
  std::size_t search_work = 0;
  // Enumerate every bounded displacement hypothesis and every bijection that
  // could support it. The reference scene is capped at eight anchors; extra
  // low-confidence detections may be ignored only when they cannot compete.
  for (std::size_t seed_i = 0; seed_i < scene_count_; ++seed_i) {
    for (std::size_t seed_j = 0; seed_j < count; ++seed_j) {
      if (!compatible(seed_i, seed_j))
        continue;
      const PixelPoint seed = displacement(seed_i, seed_j);
      if (separation(seed, {}) >
          kMaximumSceneTranslationPx + kGeometryEpsilonPx)
        continue;
      std::array<std::size_t, kMaxSceneAnchors> mapping{};
      std::array<bool, bus::kMaxObservations> used{};
      mapping.fill(0);
      used.fill(false);

      // With a seed taken from one true pair, every other true pair is
      // at most twice the final residual away from that seed. Enumerate
      // that bounded neighbourhood, then refine from the complete map.
      const double seed_gate =
          2.0 * kMaximumSceneResidualPx + kGeometryEpsilonPx;
      auto enumerate = [&](auto &&self, std::size_t reference_index, double dx,
                           double dy) noexcept -> void {
        if (ambiguous || search_exhausted)
          return;
        if (++search_work > kMaximumSceneSearchWork) {
          search_exhausted = true;
          return;
        }
        if (reference_index == scene_count_) {
          if (scene_count_ == 0)
            return;
          const PixelPoint translation{
              static_cast<float>(dx / static_cast<double>(scene_count_)),
              static_cast<float>(dy / static_cast<double>(scene_count_))};
          if (separation(translation, {}) >
              kMaximumSceneTranslationPx + kGeometryEpsilonPx)
            return;
          for (std::size_t i = 0; i < scene_count_; ++i) {
            const PixelPoint predicted = project(i, translation);
            if (separation(predicted, candidates[mapping[i]]->center_px) >
                kMaximumSceneResidualPx + kGeometryEpsilonPx)
              return;
          }
          // A complete reference map may ignore low-confidence
          // clutter, but every unmatched geometry-eligible
          // detection at the measurement floor is an additional
          // target and must fail closed. A low-confidence candidate
          // is also rejected when it could compete for a reference
          // anchor at this translation; this prevents a strict
          // extra from replacing a temporarily blurred real anchor
          // based on candidate order.
          for (std::size_t candidate_index = 0; candidate_index < count;
               ++candidate_index) {
            if (used[candidate_index])
              continue;
            if (eligible(*candidates[candidate_index]))
              return;
            for (std::size_t i = 0; i < scene_count_; ++i) {
              if (!compatible(i, candidate_index))
                continue;
              const PixelPoint predicted = project(i, translation);
              if (separation(predicted,
                             candidates[candidate_index]->center_px) <=
                  kMaximumSceneResidualPx + kGeometryEpsilonPx)
                return;
            }
          }
          if (!found) {
            found = true;
            accepted_mapping = mapping;
            accepted_confident = true;
            for (std::size_t i = 0; i < scene_count_; ++i) {
              if (!eligible(*candidates[mapping[i]])) {
                accepted_confident = false;
                break;
              }
            }
          } else if (mapping != accepted_mapping) {
            ambiguous = true;
          }
          return;
        }

        const PixelPoint predicted = project(reference_index, seed);
        for (std::size_t candidate_index = 0; candidate_index < count;
             ++candidate_index) {
          if (used[candidate_index] ||
              !compatible(reference_index, candidate_index) ||
              separation(predicted, candidates[candidate_index]->center_px) >
                  seed_gate)
            continue;
          mapping[reference_index] = candidate_index;
          used[candidate_index] = true;
          const auto step = displacement(reference_index, candidate_index);
          self(self, reference_index + 1, dx + step.x, dy + step.y);
          used[candidate_index] = false;
          if (ambiguous)
            return;
        }
      };
      enumerate(enumerate, 0, 0.0, 0.0);
      if (ambiguous || search_exhausted)
        return nullptr;
    }
  }
  if (!found)
    return nullptr;
  scene_match_confident_ = accepted_confident;
  for (std::size_t i = 0; i < scene_count_; ++i) {
    const auto &match = *candidates[accepted_mapping[i]];
    matched_scene_[i] = {match.center_px, match.effective_radius_px,
                         match.semantic_id};
  }
  return candidates[accepted_mapping[selected_scene_index_]];
}

bool VisualCalibrationSession::scene_stationary(
    const Scene &reference) const noexcept {
  for (std::size_t i = 0; i < scene_count_; ++i) {
    if (separation(reference[i].center, matched_scene_[i].center) >
        config_.fitting.stationary_tolerance_px)
      return false;
  }
  return true;
}

CalibrationObservation VisualCalibrationSession::measured(
    const bus::TargetObservation &observation,
    const bus::TargetObservationBatch &batch) const noexcept {
  return {batch.frame_id,
          batch.captured_at_ns,
          association_id_,
          observation.center_px,
          observation.confidence,
          static_cast<float>(
              std::sqrt(largest_variance(observation.covariance_px2))),
          true,
          true};
}

CalibrationSessionUpdate
VisualCalibrationSession::observe(const bus::TargetObservationBatch &batch,
                                  MonotonicNs now_ns,
                                  bool authorized) noexcept {
  (void)poll(now_ns, authorized);
  if (state_ == CalibrationSessionState::idle || complete() || failed())
    return update();
  if (batch.target_count > bus::kMaxObservations ||
      batch.source_width != config_.fitting.width ||
      batch.source_height != config_.fitting.height || batch.frame_id == 0 ||
      batch.captured_at_ns <= 0 || batch.header.sequence_id != batch.frame_id ||
      batch.header.source_timestamp_ns != batch.captured_at_ns ||
      batch.captured_at_ns > now_ns ||
      now_ns - batch.captured_at_ns > kFreshnessNs)
    return fail(VisualCalibrationStatus::invalid_observation);
  if (batch.frame_id == last_frame_id_ &&
      batch.captured_at_ns == last_frame_ns_)
    return update();
  if (batch.frame_id <= last_frame_id_ ||
      batch.captured_at_ns <= last_frame_ns_)
    return fail(VisualCalibrationStatus::invalid_timing);
  for (const auto &target : batch.items()) {
    if (target.frame_id != batch.frame_id ||
        target.captured_at_ns != batch.captured_at_ns)
      return fail(VisualCalibrationStatus::invalid_observation);
  }
  last_frame_id_ = batch.frame_id;
  last_frame_ns_ = batch.captured_at_ns;
  if (state_ == CalibrationSessionState::awaiting_dispatch)
    return update();
  if (state_ == CalibrationSessionState::seeking_target) {
    scene_count_ = 0;
    scene_match_confident_ = false;
    const auto *target = select(batch);
    if (!target)
      target = select_scene(batch);
    if (!target) {
      status_ = VisualCalibrationStatus::ambiguous_association;
      return update();
    }
    ++association_id_;
    association_center_ = target->center_px;
    association_radius_ = target->effective_radius_px;
    semantic_id_ = target->semantic_id;
    sample_ = {};
    sample_.baseline = measured(*target, batch);
    state_ = CalibrationSessionState::establishing_baseline;
    status_ = VisualCalibrationStatus::ok;
    return update();
  }
  const auto *target =
      scene_count_ == 0 ? associate(batch) : associate_scene(batch, true);
  if (!target) {
    if (state_ == CalibrationSessionState::establishing_baseline) {
      state_ = CalibrationSessionState::seeking_target;
      status_ = VisualCalibrationStatus::ambiguous_association;
      return update();
    }
    return fail(VisualCalibrationStatus::ambiguous_association);
  }
  // A complete geometric scene can remain attributable while one or more
  // detections are temporarily below the measurement confidence floor (for
  // example, motion blur immediately after a physical pulse). Keep the
  // in-flight state and wait for a strict frame; never record or dispatch on
  // the low-confidence frame.
  if (scene_count_ != 0 && !scene_match_confident_) {
    if (state_ == CalibrationSessionState::establishing_baseline &&
        !scene_stationary(reference_scene_))
      status_ = VisualCalibrationStatus::moving_target;
    return update();
  }
  const auto observation = measured(*target, batch);
  association_center_ = target->center_px;
  if (state_ == CalibrationSessionState::establishing_baseline) {
    if (separation(sample_.baseline.center_px, observation.center_px) >
            config_.fitting.stationary_tolerance_px ||
        !scene_stationary(reference_scene_)) {
      reference_scene_ = matched_scene_;
      sample_.baseline = observation;
      status_ = VisualCalibrationStatus::moving_target;
      return update();
    }
    if (observation.captured_at_ns - sample_.baseline.captured_at_ns <
        config_.fitting.minimum_stability_ns)
      return update();
    sample_.before = sample_.last_unchanged = observation;
    reference_scene_ = matched_scene_;
    sample_.pulse_sequence = static_cast<SequenceId>(pulse_index_) + 1;
    sample_.held_out = pulse_index_ >= 16;
    const int scale = config_.fitting.max_pulse_counts >= 64 ? 16 : 8;
    const int magnitude =
        pulse_index_ < 16
            ? scale * (static_cast<int>(pulse_index_ / 4) + 1)
            : 3 * scale / 2 + scale * static_cast<int>((pulse_index_ - 16) / 4);
    const auto direction = pulse_index_ % 4;
    sample_.counts_x =
        direction < 2 ? (direction == 0 ? magnitude : -magnitude) : 0;
    sample_.counts_y =
        direction >= 2 ? (direction == 2 ? magnitude : -magnitude) : 0;
    state_ = CalibrationSessionState::awaiting_dispatch;
    status_ = VisualCalibrationStatus::ok;
    auto result = update();
    result.pulse = VisualCalibrationPulse{
        sample_.pulse_sequence, sample_.counts_x, sample_.counts_y,
        sample_.held_out, sample_.before.captured_at_ns};
    return result;
  }
  if (state_ == CalibrationSessionState::awaiting_response) {
    const double response =
        separation(sample_.before.center_px, observation.center_px);
    if (response <= config_.fitting.stationary_tolerance_px &&
        scene_stationary(reference_scene_))
      sample_.last_unchanged = observation;
    if (observation.captured_at_ns <= sample_.dispatched_at_ns ||
        response < config_.fitting.minimum_response_px)
      return update();
    sample_.first_response = sample_.after = observation;
    settling_scene_ = matched_scene_;
    state_ = CalibrationSessionState::settling;
    return update();
  }
  if (state_ == CalibrationSessionState::settling) {
    if (separation(sample_.after.center_px, observation.center_px) >
            config_.fitting.stationary_tolerance_px ||
        !scene_stationary(settling_scene_)) {
      settling_scene_ = matched_scene_;
      sample_.after = observation;
      return update();
    }
    if (observation.captured_at_ns - sample_.after.captured_at_ns <
        config_.fitting.minimum_stability_ns)
      return update();
    sample_.settled = observation;
    const auto status = fitter_.add_sample(sample_);
    if (status != VisualCalibrationStatus::ok)
      return fail(status);
    ++pulse_index_;
    state_ = pulse_index_ == kPulseCount
                 ? CalibrationSessionState::complete
                 : CalibrationSessionState::seeking_target;
    status_ = VisualCalibrationStatus::ok;
  }
  return update();
}

bool VisualCalibrationSession::confirm_dispatch(SequenceId sequence,
                                                MonotonicNs dispatched_at_ns,
                                                bool accepted) noexcept {
  if (state_ != CalibrationSessionState::awaiting_dispatch ||
      sequence != sample_.pulse_sequence) {
    (void)fail(VisualCalibrationStatus::invalid_pulse);
    return false;
  }
  if (!accepted) {
    (void)fail(VisualCalibrationStatus::rejected_dispatch);
    return false;
  }
  if (dispatched_at_ns < last_now_ns_ ||
      dispatched_at_ns < sample_.before.captured_at_ns ||
      dispatched_at_ns - sample_.before.captured_at_ns > kFreshnessNs ||
      dispatched_at_ns - started_at_ns_ >= config_.total_deadline_ns) {
    (void)fail(VisualCalibrationStatus::invalid_timing);
    return false;
  }
  sample_.dispatch_accepted = true;
  sample_.dispatched_at_ns = dispatched_at_ns;
  last_now_ns_ = dispatched_at_ns;
  state_ = CalibrationSessionState::awaiting_response;
  return true;
}

VisualCalibrationFit VisualCalibrationSession::fit(
    const calibration::CalibrationProfile &context) const {
  if (!complete()) {
    VisualCalibrationFit result;
    result.status =
        failed() ? status_ : VisualCalibrationStatus::insufficient_samples;
    return result;
  }
  return fitter_.fit(context);
}

} // namespace aim::runtime
