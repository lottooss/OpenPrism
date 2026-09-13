#pragma once

#include <array>
#include <optional>

#include "aim/bus/bus_traits.hpp"
#include "aim/runtime/visual_calibration.hpp"

namespace aim::runtime {

enum class CalibrationSessionState : std::uint8_t {
  idle,
  seeking_target,
  establishing_baseline,
  awaiting_dispatch,
  awaiting_response,
  settling,
  complete,
  failed
};

struct VisualCalibrationPulse {
  SequenceId sequence{0};
  std::int32_t counts_x{0};
  std::int32_t counts_y{0};
  bool held_out{false};
  MonotonicNs source_timestamp_ns{0};
};

struct CalibrationSessionUpdate {
  CalibrationSessionState state{CalibrationSessionState::idle};
  VisualCalibrationStatus status{VisualCalibrationStatus::ok};
  std::optional<VisualCalibrationPulse> pulse{};
};

struct VisualCalibrationSessionConfig {
  VisualCalibrationConfig fitting{};
  PixelPoint crosshair_px{960.0f, 540.0f};
  // Optional pinhole-camera geometry for axis-only calibration pulses.
  // Zero retains the generic screen-translation model.
  float horizontal_fov_deg{0.0f};
  float selection_radius_px{600.0f};
  float association_radius_px{100.0f};
  float maximum_radius_change_fraction{0.20f};
  MonotonicNs total_deadline_ns{30'000'000'000};
};

// Acquisition state only. The host authorizes and dispatches each offered pulse
// once, then calls confirm_dispatch with the actual accepted receipt/time. No
// input is dispatched here. Loss/ambiguity during an in-flight pulse fails the
// session; pre-pulse ambiguities retry without generating any input.
// Clustered targets require 3-8 persistent anchors and a unique complete scene
// translation; individual detection IDs are not assumed to persist.
class VisualCalibrationSession {
public:
  static constexpr std::size_t kPulseCount = 24;
  explicit VisualCalibrationSession(
      const VisualCalibrationSessionConfig &config = {}) noexcept;
  [[nodiscard]] bool start(MonotonicNs now_ns) noexcept;
  [[nodiscard]] CalibrationSessionUpdate
  observe(const bus::TargetObservationBatch &batch, MonotonicNs now_ns,
          bool authorized) noexcept;
  [[nodiscard]] bool confirm_dispatch(SequenceId sequence,
                                      MonotonicNs dispatched_at_ns,
                                      bool accepted) noexcept;
  // Call even when no observation is available, so capture loss cannot evade
  // either the per-pulse timeout or the whole-session deadline.
  [[nodiscard]] CalibrationSessionUpdate poll(MonotonicNs now_ns,
                                              bool authorized) noexcept;
  [[nodiscard]] VisualCalibrationFit
  fit(const calibration::CalibrationProfile &context) const;
  [[nodiscard]] bool complete() const noexcept {
    return state_ == CalibrationSessionState::complete;
  }
  [[nodiscard]] bool failed() const noexcept {
    return state_ == CalibrationSessionState::failed;
  }
  [[nodiscard]] std::size_t samples() const noexcept { return fitter_.size(); }

private:
  [[nodiscard]] CalibrationSessionUpdate update() const noexcept {
    return {state_, status_, {}};
  }
  [[nodiscard]] CalibrationSessionUpdate
  fail(VisualCalibrationStatus status) noexcept;
  [[nodiscard]] bool eligible(const bus::TargetObservation &observation,
                              float minimum_confidence) const noexcept;
  [[nodiscard]] bool
  eligible(const bus::TargetObservation &observation) const noexcept;
  [[nodiscard]] bool
  geometry_eligible(const bus::TargetObservation &observation) const noexcept;
  [[nodiscard]] const bus::TargetObservation *
  select(const bus::TargetObservationBatch &batch) const noexcept;
  [[nodiscard]] const bus::TargetObservation *
  associate(const bus::TargetObservationBatch &batch) const noexcept;
  struct SceneAnchor {
    PixelPoint center{};
    float radius{0.0f};
    std::uint32_t semantic_id{0};
  };
  static constexpr std::size_t kMaxSceneAnchors = 8;
  using Scene = std::array<SceneAnchor, kMaxSceneAnchors>;
  [[nodiscard]] const bus::TargetObservation *
  select_scene(const bus::TargetObservationBatch &batch) noexcept;
  [[nodiscard]] const bus::TargetObservation *
  associate_scene(const bus::TargetObservationBatch &batch,
                  bool allow_low_confidence) noexcept;
  [[nodiscard]] bool scene_stationary(const Scene &reference) const noexcept;
  [[nodiscard]] CalibrationObservation
  measured(const bus::TargetObservation &observation,
           const bus::TargetObservationBatch &batch) const noexcept;
  VisualCalibrationSessionConfig config_{};
  double scene_focal_px_{0.0};
  VisualCalibrationFitter fitter_;
  CalibrationSessionState state_{CalibrationSessionState::idle};
  VisualCalibrationStatus status_{VisualCalibrationStatus::ok};
  MonotonicNs started_at_ns_{0};
  MonotonicNs last_now_ns_{0};
  MonotonicNs last_frame_ns_{0};
  SequenceId last_frame_id_{0};
  TrackId association_id_{0};
  std::size_t pulse_index_{0};
  VisualCalibrationSample sample_{};
  PixelPoint association_center_{};
  float association_radius_{0.0f};
  std::uint32_t semantic_id_{0};
  // Reference order defines association for one pulse. Freeze at its actual
  // pre-dispatch observation; never accumulate frame-to-frame identity drift.
  Scene reference_scene_{};
  Scene matched_scene_{};
  Scene settling_scene_{};
  std::size_t scene_count_{0};
  std::size_t selected_scene_index_{0};
  bool scene_match_confident_{false};
};

} // namespace aim::runtime
