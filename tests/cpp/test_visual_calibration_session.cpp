#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>

#include "aim/runtime/visual_calibration_session.hpp"

using namespace aim;
using namespace aim::runtime;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Check failed at line " << __LINE__                         \
                << ": " #condition "\n";                                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

namespace {
bus::TargetObservationBatch frame(SequenceId id, MonotonicNs time,
                                  PixelPoint center = {960.0f, 540.0f}) {
  bus::TargetObservationBatch batch;
  batch.frame_id = id;
  batch.captured_at_ns = time;
  batch.header = {id, time, 1, 0};
  bus::TargetObservation target;
  target.frame_id = id;
  target.captured_at_ns = time;
  target.center_px = center;
  target.effective_radius_px = 20.0f;
  target.confidence = 0.99f;
  target.covariance_px2 = {9.0f, 0.0f, 9.0f};
  target.bbox_px = {center.x - 20.0f, center.y - 20.0f, center.x + 20.0f,
                    center.y + 20.0f};
  CHECK(batch.add_target(target));
  return batch;
}

template <std::size_t N>
bus::TargetObservationBatch
scene_frame(SequenceId id, MonotonicNs time,
            const std::array<PixelPoint, N> &centers) {
  bus::TargetObservationBatch batch;
  batch.frame_id = id;
  batch.captured_at_ns = time;
  batch.header = {id, time, 1, 0};
  for (const auto &center : centers) {
    bus::TargetObservation target;
    target.frame_id = id;
    target.captured_at_ns = time;
    target.center_px = center;
    target.effective_radius_px = 20.0f;
    target.confidence = 0.99f;
    target.covariance_px2 = {9.0f, 0.0f, 9.0f};
    target.bbox_px = {center.x - 20.0f, center.y - 20.0f, center.x + 20.0f,
                      center.y + 20.0f};
    CHECK(batch.add_target(target));
  }
  return batch;
}

void apply_fixture_pulse(PixelPoint &target,
                         const VisualCalibrationPulse &pulse) {
  // Independent simulated camera, not OS input. Right/down input moves a
  // stationary world target left/up in captured coordinates.
  constexpr double coupling = 0.18;
  const auto effective = [](std::int32_t count) {
    return std::copysign(
        (std::max)(0.0, std::abs(static_cast<double>(count)) - 1.5),
        static_cast<double>(count));
  };
  const double x = effective(pulse.counts_x) / 1.7;
  const double y = effective(pulse.counts_y) / 2.4;
  target.x -=
      static_cast<float>((x + coupling * y) / (1.0 - coupling * coupling));
  target.y -=
      static_cast<float>((y + coupling * x) / (1.0 - coupling * coupling));
}

VisualCalibrationPulse first_pulse(VisualCalibrationSession &session) {
  CHECK(session.start(1'000'000'000));
  CHECK(!session.observe(frame(1, 1'000'000'000), 1'000'000'000, true).pulse);
  auto update = session.observe(frame(2, 1'012'000'000), 1'012'000'000, true);
  CHECK(update.pulse.has_value());
  return *update.pulse;
}

void test_complete_session_and_dropped_frames(int pulse_limit = 64) {
  for (const bool drop_frames : {false, true}) {
    VisualCalibrationSessionConfig config;
    config.fitting.max_pulse_counts = pulse_limit;
    VisualCalibrationSession session(config);
    MonotonicNs now = 1'000'000'000;
    CHECK(session.start(now));
    PixelPoint target{960.0f, 540.0f};
    std::optional<VisualCalibrationPulse> pending;
    MonotonicNs effect_at = 0;
    std::size_t pulses = 0;
    std::int32_t x_sum = 0, y_sum = 0;
    for (SequenceId id = 1;
         id < 2000 && !session.complete() && !session.failed();
         ++id, now += 4'000'000) {
      if (pending && now >= effect_at) {
        apply_fixture_pulse(target, *pending);
        pending.reset();
      }
      (void)session.poll(now, true);
      if (drop_frames && id % 7 == 0)
        continue;
      auto update = session.observe(frame(id, now, target), now, true);
      if (!update.pulse)
        continue;
      CHECK(!pending);
      CHECK(update.pulse->sequence == pulses + 1);
      CHECK(update.pulse->held_out == (pulses >= 16));
      CHECK(std::abs(update.pulse->counts_x) <= pulse_limit &&
            std::abs(update.pulse->counts_y) <= pulse_limit);
      CHECK((update.pulse->counts_x == 0) != (update.pulse->counts_y == 0));
      x_sum += update.pulse->counts_x;
      y_sum += update.pulse->counts_y;
      if (pulses % 2 == 1)
        CHECK(x_sum == 0 && y_sum == 0);
      pending = update.pulse;
      CHECK(session.confirm_dispatch(pending->sequence, now + 1'000'000, true));
      effect_at = now + 13'000'000;
      ++pulses;
    }
    CHECK(session.complete());
    CHECK(pulses == 24 && session.samples() == 24);
    const auto result = session.fit({});
    CHECK(result.succeeded());
    CHECK(result.measured_max_pulse_counts == pulse_limit);
    CHECK(std::abs(result.profile.counts_per_pixel_x - 1.7f) < 0.01f);
    CHECK(std::abs(result.profile.counts_per_pixel_y - 2.4f) < 0.01f);
    CHECK(std::abs(result.profile.deadband_counts - 1.5f) < 0.11f);
    CHECK(result.profile.rmse_pixels < 0.05f);
    CHECK(result.effect_lower_p50_ns <= 12'000'000 &&
          result.effect_upper_p50_ns >= 12'000'000);
  }
}

void test_three_cluster_translation_with_noise_and_reordered_detections() {
  VisualCalibrationSession session;
  CHECK(session.start(1'000'000'000));

  // Aim Lab's Gridshot layout is a tight three-target scene. The nearest
  // target is intentionally not isolated at the 100 px association radius.
  const std::array<PixelPoint, 3> initial{
      {{860.2f, 540.1f}, {958.3f, 540.2f}, {1056.1f, 539.9f}}};
  auto first = scene_frame(1, 1'000'000'000, initial);
  CHECK(!session.observe(first, 1'000'000'000, true).pulse);

  const std::array<PixelPoint, 3> stable{
      {{860.4f, 540.2f}, {958.1f, 540.0f}, {1056.3f, 540.1f}}};
  auto second = scene_frame(2, 1'012'000'000, stable);
  std::reverse(second.targets.begin(),
               second.targets.begin() + second.target_count);
  auto update = session.observe(second, 1'012'000'000, true);
  CHECK(update.pulse.has_value());
  CHECK(session.confirm_dispatch(update.pulse->sequence, 1'013'000'000, true));

  // All three anchors undergo one common displacement, with independent
  // subpixel detector noise and a different candidate ordering.
  const std::array<PixelPoint, 3> moved{
      {{852.4f, 538.1f}, {950.0f, 538.0f}, {1048.3f, 538.1f}}};
  auto response = scene_frame(3, 1'028'000'000, moved);
  std::reverse(response.targets.begin(),
               response.targets.begin() + response.target_count);
  CHECK(!session.observe(response, 1'028'000'000, true).pulse);
  auto settled = scene_frame(4, 1'040'000'000, moved);
  std::reverse(settled.targets.begin(),
               settled.targets.begin() + settled.target_count);
  CHECK(!session.observe(settled, 1'040'000'000, true).pulse);
  CHECK(!session.failed());
  CHECK(session.samples() == 1);
}

template <std::size_t N>
VisualCalibrationPulse
first_scene_pulse(VisualCalibrationSession &session,
                  const std::array<PixelPoint, N> &centers) {
  CHECK(session.start(1'000'000'000));
  CHECK(
      !session
           .observe(scene_frame(1, 1'000'000'000, centers), 1'000'000'000, true)
           .pulse);
  const auto update = session.observe(scene_frame(2, 1'012'000'000, centers),
                                      1'012'000'000, true);
  CHECK(update.pulse.has_value());
  return *update.pulse;
}

void set_scene_confidence(bus::TargetObservationBatch &batch, std::size_t index,
                          float confidence) {
  CHECK(index < batch.target_count);
  batch.targets[index].confidence = confidence;
}

void set_scene_radius(bus::TargetObservationBatch &batch, std::size_t index,
                      float radius) {
  CHECK(index < batch.target_count);
  auto &target = batch.targets[index];
  target.effective_radius_px = radius;
  target.bbox_px = {target.center_px.x - radius, target.center_px.y - radius,
                    target.center_px.x + radius, target.center_px.y + radius};
}

void test_blurred_scene_frame_waits_then_recovers() {
  VisualCalibrationSession session;
  const std::array<PixelPoint, 3> base{
      {{860.0f, 540.0f}, {958.0f, 540.0f}, {1056.0f, 540.0f}}};
  const auto pulse = first_scene_pulse(session, base);
  CHECK(session.confirm_dispatch(pulse.sequence, 1'013'000'000, true));

  const std::array<PixelPoint, 3> moved{
      {{852.0f, 538.0f}, {950.0f, 538.0f}, {1048.0f, 538.0f}}};
  auto blurred = scene_frame(3, 1'028'000'000, moved);
  set_scene_confidence(blurred, 2, 0.75f);
  const auto blurred_update = session.observe(blurred, 1'028'000'000, true);
  CHECK(!blurred_update.pulse && !session.failed());
  CHECK(session.samples() == 0);

  // Once every matched anchor clears the strict measurement floor, the
  // response is measured normally; the blurred frame never becomes a sample.
  CHECK(!session
             .observe(scene_frame(4, 1'040'000'000, moved), 1'040'000'000, true)
             .pulse);
  CHECK(!session
             .observe(scene_frame(5, 1'052'000'000, moved), 1'052'000'000, true)
             .pulse);
  CHECK(!session.failed());
  CHECK(session.samples() == 1);
}

void test_blurred_scene_frame_times_out_without_extra_pulse() {
  VisualCalibrationSession session;
  const std::array<PixelPoint, 3> base{
      {{860.0f, 540.0f}, {958.0f, 540.0f}, {1056.0f, 540.0f}}};
  const auto pulse = first_scene_pulse(session, base);
  CHECK(session.confirm_dispatch(pulse.sequence, 1'013'000'000, true));

  const std::array<PixelPoint, 3> moved{
      {{852.0f, 538.0f}, {950.0f, 538.0f}, {1048.0f, 538.0f}}};
  auto blurred = scene_frame(3, 1'028'000'000, moved);
  set_scene_confidence(blurred, 2, 0.75f);
  CHECK(!session.observe(blurred, 1'028'000'000, true).pulse);
  CHECK(!session.failed());
  (void)session.poll(1'114'000'001, true);
  CHECK(session.failed());
  CHECK(session.samples() == 0);
}

void test_recorded_gridshot_blur_ignores_weapon_detection() {
  VisualCalibrationSession session;
  const std::array<PixelPoint, 3> reference{
      {{926.5419f, 271.6146f}, {828.6584f, 367.8752f}, {829.1058f, 467.7867f}}};
  const std::array<float, 3> reference_radii{{38.1044f, 38.3300f, 37.4626f}};
  CHECK(session.start(147'388'639'558'900));

  auto first = scene_frame(3, 147'388'639'558'900, reference);
  for (std::size_t i = 0; i < reference.size(); ++i)
    set_scene_radius(first, i, reference_radii[i]);
  CHECK(!session.observe(first, 147'388'639'558'900, true).pulse);

  auto stable = scene_frame(5, 147'388'679'067'500, reference);
  for (std::size_t i = 0; i < reference.size(); ++i)
    set_scene_radius(stable, i, reference_radii[i]);
  auto pulse = session.observe(stable, 147'388'679'067'500, true).pulse;
  CHECK(pulse.has_value());
  CHECK(session.confirm_dispatch(pulse->sequence, 147'388'680'067'500, true));

  // This is the recorded post-pulse frame: the three real targets share a
  // translation while the third target is below the strict .8 measurement
  // floor. The fourth detection is the unrelated low-confidence weapon
  // false positive seen in the same capture.
  const std::array<PixelPoint, 3> moved{
      {{910.7688f, 272.9244f}, {813.5209f, 369.3804f}, {813.7496f, 466.7286f}}};
  const std::array<float, 3> moved_radii{{38.2051f, 38.1948f, 37.5833f}};
  const std::array<PixelPoint, 4> recorded_centers{
      {moved[0], moved[1], moved[2], {1201.3105f, 706.2604f}}};
  auto blurred = scene_frame(6, 147'388'692'931'200, recorded_centers);
  for (std::size_t i = 0; i < moved_radii.size(); ++i)
    set_scene_radius(blurred, i, moved_radii[i]);
  set_scene_radius(blurred, 3, 47.0253f);
  set_scene_confidence(blurred, 0, 0.8657f);
  set_scene_confidence(blurred, 1, 0.8184f);
  set_scene_confidence(blurred, 2, 0.7588f);
  set_scene_confidence(blurred, 3, 0.2672f);
  const auto blurred_update =
      session.observe(blurred, 147'388'699'852'400, true);
  CHECK(!blurred_update.pulse && !session.failed());
  CHECK(session.samples() == 0);

  auto recovered = scene_frame(7, 147'388'706'000'000, moved);
  for (std::size_t i = 0; i < moved_radii.size(); ++i)
    set_scene_radius(recovered, i, moved_radii[i]);
  CHECK(!session.observe(recovered, 147'388'706'000'000, true).pulse);
  CHECK(session.samples() == 0); // first trusted response, not yet settled
  auto settled = scene_frame(8, 147'388'718'000'000, moved);
  for (std::size_t i = 0; i < moved_radii.size(); ++i)
    set_scene_radius(settled, i, moved_radii[i]);
  CHECK(!session.observe(settled, 147'388'718'000'000, true).pulse);
  CHECK(!session.failed());
  CHECK(session.samples() == 1);
}

void test_full_three_cluster_replay() {
  VisualCalibrationSession session;
  std::array<PixelPoint, 3> targets{
      {{860.0f, 540.0f}, {958.0f, 540.0f}, {1056.0f, 540.0f}}};
  CHECK(session.start(1'000'000'000));

  std::optional<VisualCalibrationPulse> pending;
  MonotonicNs now = 1'000'000'000;
  MonotonicNs effect_at = 0;
  std::size_t pulses = 0;
  for (SequenceId id = 1; id < 2000 && !session.complete() && !session.failed();
       ++id, now += 4'000'000) {
    if (pending && now >= effect_at) {
      for (auto &target : targets)
        apply_fixture_pulse(target, *pending);
      pending.reset();
    }
    (void)session.poll(now, true);
    auto observed = scene_frame(id, now, targets);
    // Detection order is not an identity contract. Exercise a stable but
    // changing order while preserving the same complete scene.
    if (id % 3 == 0)
      std::reverse(observed.targets.begin(),
                   observed.targets.begin() + observed.target_count);
    const auto update = session.observe(observed, now, true);
    if (!update.pulse)
      continue;
    CHECK(!pending);
    CHECK(update.pulse->sequence == pulses + 1);
    CHECK(update.pulse->held_out == (pulses >= 16));
    CHECK((update.pulse->counts_x == 0) != (update.pulse->counts_y == 0));
    pending = update.pulse;
    CHECK(session.confirm_dispatch(pending->sequence, now + 1'000'000, true));
    effect_at = now + 13'000'000;
    ++pulses;
  }
  CHECK(session.complete());
  CHECK(!session.failed());
  CHECK(pulses == 24 && session.samples() == 24);
  const auto result = session.fit({});
  CHECK(result.succeeded());
  CHECK(std::abs(result.profile.counts_per_pixel_x - 1.7f) < 0.01f);
  CHECK(std::abs(result.profile.counts_per_pixel_y - 2.4f) < 0.01f);
  CHECK(std::abs(result.profile.deadband_counts - 1.5f) < 0.11f);
  CHECK(result.held_out_max_error_px < 0.05f);
}

void test_camera_rotation_scene_replay() {
  for (bool nonrigid : {false, true}) {
    VisualCalibrationSessionConfig config;
    config.horizontal_fov_deg = 103.0f;
    VisualCalibrationSession session(config);
    std::array<PixelPoint, 3> targets{
        {{1023.0f, 273.0f}, {1023.0f, 372.0f}, {926.0f, 468.0f}}};
    CHECK(session.start(1'000'000'000));
    std::optional<VisualCalibrationPulse> pending;
    MonotonicNs effect_at = 0;
    std::size_t pulses = 0;
    const double focal =
        960.0 / std::tan(103.0 * 3.14159265358979323846 / 360.0);
    for (SequenceId id = 1;
         id < 2000 && !session.complete() && !session.failed(); ++id) {
      const MonotonicNs now =
          1'000'000'000 + static_cast<MonotonicNs>(id - 1) * 4'000'000;
      if (pending && now >= effect_at) {
        const bool horizontal = pending->counts_x != 0;
        const double angle =
            (horizontal ? pending->counts_x : pending->counts_y) * 0.07 *
            3.14159265358979323846 / 180.0;
        for (auto &p : targets) {
          const double x = (p.x - 960.0) / focal, y = (p.y - 540.0) / focal;
          const double z =
              std::cos(angle) + (horizontal ? x : y) * std::sin(angle);
          p = {
              static_cast<float>(
                  960.0 +
                  focal *
                      (horizontal ? x * std::cos(angle) - std::sin(angle) : x) /
                      z),
              static_cast<float>(
                  540.0 +
                  focal *
                      (horizontal ? y : y * std::cos(angle) - std::sin(angle)) /
                      z)};
        }
        if (nonrigid)
          targets[1].y += 12.0f;
        pending.reset();
      }
      const auto update =
          session.observe(scene_frame(id, now, targets), now, true);
      if (update.pulse) {
        CHECK(!pending);
        pending = update.pulse;
        CHECK(
            session.confirm_dispatch(pending->sequence, now + 1'000'000, true));
        effect_at = now + 13'000'000;
        ++pulses;
      }
    }
    if (nonrigid) {
      CHECK(session.failed() && session.samples() == 0 && pulses == 1);
    } else {
      CHECK(session.complete() && session.samples() == 24);
      const auto result = session.fit({});
      CHECK(result.succeeded() && result.profile.rmse_pixels < 1.0f);
    }
  }
}

void test_scene_ambiguity_and_inflight_rejections() {
  // Two nearly coincident anchors make both identity and swap maps fit the
  // residual bound. The session must stay in seeking state without offering
  // a pulse based on an arbitrary mapping.
  {
    VisualCalibrationSession session;
    const std::array<PixelPoint, 3> regular{
        {{900.0f, 540.0f}, {902.0f, 540.0f}, {1000.0f, 600.0f}}};
    CHECK(session.start(1'000'000'000));
    const auto update = session.observe(scene_frame(1, 1'000'000'000, regular),
                                        1'000'000'000, true);
    CHECK(!update.pulse && !session.failed());
    CHECK(update.state == CalibrationSessionState::seeking_target);
  }

  const std::array<PixelPoint, 3> base{
      {{860.0f, 540.0f}, {958.0f, 540.0f}, {1056.0f, 540.0f}}};
  for (int fault = 0; fault < 5; ++fault) {
    VisualCalibrationSession session;
    const auto pulse = first_scene_pulse(session, base);
    CHECK(session.confirm_dispatch(pulse.sequence, 1'013'000'000, true));

    if (fault == 0) {
      // Missing anchor.
      auto observed = scene_frame(3, 1'016'000'000, base);
      observed.target_count = 2;
      (void)session.observe(observed, 1'016'000'000, true);
    } else if (fault == 1) {
      // Additional eligible target.
      const std::array<PixelPoint, 4> with_new{{{860.0f, 540.0f},
                                                {958.0f, 540.0f},
                                                {1056.0f, 540.0f},
                                                {1154.0f, 540.0f}}};
      (void)session.observe(scene_frame(3, 1'016'000'000, with_new),
                            1'016'000'000, true);
    } else if (fault == 2) {
      // Radius drift beyond the 20% compatibility gate.
      auto observed = scene_frame(3, 1'016'000'000, base);
      observed.targets[1].effective_radius_px = 30.0f;
      observed.targets[1].bbox_px = {928.0f, 510.0f, 988.0f, 570.0f};
      (void)session.observe(observed, 1'016'000'000, true);
    } else if (fault == 3) {
      // Non-rigid motion cannot be explained by a common translation.
      auto observed = scene_frame(3, 1'016'000'000, base);
      observed.targets[1].center_px.x += 12.0f;
      observed.targets[1].bbox_px.left += 12.0f;
      observed.targets[1].bbox_px.right += 12.0f;
      (void)session.observe(observed, 1'016'000'000, true);
    } else {
      // An additional strict-confidence target must reject even when a
      // real anchor is temporarily below the measurement floor. A
      // count-only check would incorrectly treat the three strict
      // detections as the complete scene.
      const std::array<PixelPoint, 4> with_new{{{860.0f, 540.0f},
                                                {958.0f, 540.0f},
                                                {1056.0f, 540.0f},
                                                {1154.0f, 540.0f}}};
      auto observed = scene_frame(3, 1'016'000'000, with_new);
      set_scene_confidence(observed, 2, 0.75f);
      set_scene_confidence(observed, 3, 0.95f);
      (void)session.observe(observed, 1'016'000'000, true);
    }
    CHECK(session.failed());
    CHECK(session.samples() == 0);
  }
}

void test_ambiguity_before_pulse_retries_without_input() {
  VisualCalibrationSession session;
  CHECK(session.start(1'000'000'000));
  auto ambiguous = frame(1, 1'000'000'000);
  auto second = ambiguous.targets[0];
  second.center_px.x += 60.0f;
  second.bbox_px.left += 60.0f;
  second.bbox_px.right += 60.0f;
  CHECK(ambiguous.add_target(second));
  auto update = session.observe(ambiguous, 1'000'000'000, true);
  CHECK(!update.pulse && !session.failed());
  CHECK(update.state == CalibrationSessionState::seeking_target);
  CHECK(!session.observe(frame(2, 1'004'000'000), 1'004'000'000, true).pulse);
  CHECK(session.observe(frame(3, 1'016'000'000), 1'016'000'000, true)
            .pulse.has_value());
}

void test_failed_receipts_focus_loss_and_missing_response() {
  for (int fault = 0; fault < 5; ++fault) {
    VisualCalibrationSession session;
    const auto pulse = first_pulse(session);
    if (fault == 0)
      CHECK(!session.confirm_dispatch(pulse.sequence, 1'013'000'000, false));
    if (fault == 1)
      CHECK(!session.confirm_dispatch(pulse.sequence + 1, 1'013'000'000, true));
    if (fault == 2)
      CHECK(!session.confirm_dispatch(pulse.sequence, 1'023'000'000, true));
    if (fault >= 3) {
      CHECK(session.confirm_dispatch(pulse.sequence, 1'013'000'000, true));
      if (fault == 3)
        (void)session.poll(1'014'000'000, false);
      else
        (void)session.poll(1'114'000'000, true);
    }
    CHECK(session.failed() && session.samples() == 0);
    CHECK(!session.fit({}).succeeded());
  }
}

void test_in_flight_ambiguity_and_provenance_faults_fail() {
  for (int fault = 0; fault < 5; ++fault) {
    VisualCalibrationSession session;
    const auto pulse = first_pulse(session);
    CHECK(session.confirm_dispatch(pulse.sequence, 1'013'000'000, true));
    auto observed = frame(3, 1'016'000'000);
    if (fault == 0)
      CHECK(observed.add_target(observed.targets[0]));
    if (fault == 1)
      observed.target_count = 0;
    if (fault == 2)
      observed.header.source_timestamp_ns = 0;
    if (fault == 3)
      observed.targets[0].captured_at_ns -= 1;
    if (fault == 4)
      observed.targets[0].covariance_px2.xy =
          9.0f; // major sigma >3 despite diagonal sigma=3
    CHECK(!session.observe(observed, 1'016'000'000, true).pulse);
    CHECK(session.failed());
  }
}

void test_stationarity_duplicate_frames_and_hard_deadline() {
  VisualCalibrationSession session;
  CHECK(session.start(1'000'000'000));
  auto first = frame(1, 1'000'000'000);
  CHECK(!session.observe(first, 1'000'000'000, true).pulse);
  CHECK(!session.observe(first, 1'001'000'000, true)
             .pulse); // replay never advances stability
  CHECK(!session
             .observe(frame(2, 1'012'000'000, {964.0f, 540.0f}), 1'012'000'000,
                      true)
             .pulse);
  CHECK(!session
             .observe(frame(3, 1'024'000'000, {968.0f, 540.0f}), 1'024'000'000,
                      true)
             .pulse);
  CHECK(!session.failed() && session.samples() == 0);
  (void)session.poll(31'000'000'000, true);
  CHECK(session.failed());
  VisualCalibrationSessionConfig config;
  config.total_deadline_ns = 30'000'000'001;
  VisualCalibrationSession invalid(config);
  CHECK(!invalid.start(1'000'000'000));
}
} // namespace

int main() {
  test_complete_session_and_dropped_frames();
  test_complete_session_and_dropped_frames(32);
  test_three_cluster_translation_with_noise_and_reordered_detections();
  test_blurred_scene_frame_waits_then_recovers();
  test_blurred_scene_frame_times_out_without_extra_pulse();
  test_recorded_gridshot_blur_ignores_weapon_detection();
  test_full_three_cluster_replay();
  test_camera_rotation_scene_replay();
  test_scene_ambiguity_and_inflight_rejections();
  test_ambiguity_before_pulse_retries_without_input();
  test_failed_receipts_focus_loss_and_missing_response();
  test_in_flight_ambiguity_and_provenance_faults_fail();
  test_stationarity_duplicate_frames_and_hard_deadline();
  std::cout << "Visual calibration acquisition state tests passed; zero OS "
               "input.\n";
}
