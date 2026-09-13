// Observation-only diagnostic for the preserved experimental GDI/color
// detector. Production actuation requires the integrations listed in
// docs/DELIVERY.md.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "aim/core/actuator.hpp"
#include "aim/core/clock.hpp"
#include "aim/policy/shot_gate.hpp"
#include "aim/policy/utility_policy.hpp"
#include "aim/tracking/tracking_engine.hpp"

using namespace aim;

struct DetectedBlob {
  float cx{0.0f};
  float cy{0.0f};
  float radius{20.0f};
  std::uint32_t pixel_count{0};
};

// Scan a BGRA buffer to locate bright spheres (cyan, ruby, emerald, gold,
// purple, white) on neutral wall.
std::vector<DetectedBlob> detect_spheres(const std::uint8_t *bgra, int width,
                                         int height, int roi_left, int roi_top,
                                         int roi_right, int roi_bottom,
                                         int step, int &out_max_brightness) {

  out_max_brightness = 0;
  std::vector<DetectedBlob> blobs;
  if (!bgra || width <= 0 || height <= 0)
    return blobs;

  roi_left = std::clamp(roi_left, 0, width - 1);
  roi_top = std::clamp(roi_top, 0, height - 1);
  roi_right = std::clamp(roi_right, 0, width - 1);
  roi_bottom = std::clamp(roi_bottom, 0, height - 1);

  struct HitPoint {
    int x;
    int y;
  };
  std::vector<HitPoint> hits;
  hits.reserve(8192);

  int max_b = 0;

  for (int y = roi_top; y < roi_bottom; y += step) {
    const std::uint8_t *row = bgra + static_cast<std::size_t>(y) * width * 4;
    for (int x = roi_left; x < roi_right; x += step) {
      const std::uint8_t *px = row + static_cast<std::size_t>(x) * 4;
      const int b = px[0];
      const int g = px[1];
      const int r = px[2];

      const int max_c = std::max({r, g, b});
      const int min_c = std::min({r, g, b});
      if (max_c > max_b)
        max_b = max_c;

      const int sat = max_c - min_c;

      // Target sphere color signatures:
      // Matches cyan, ruby, emerald, lime, gold, orange, magenta, and custom
      // lavender/lilac spheres (G5). In Aimlabs, walls are dark (max_c <= 50)
      // or neutral gray (sat < 15). Spheres have max_c >= 120 and either
      // noticeable saturation (sat >= 18) or bright glowing luminance (max_c >=
      // 165, min_c >= 95).
      const bool is_target_color =
          (max_c >= 120 && (sat >= 18 || (max_c >= 165 && min_c >= 95)));

      if (is_target_color) {
        hits.push_back({x, y});
      }
    }
  }

  out_max_brightness = max_b;
  if (hits.empty())
    return blobs;

  // Cluster hits into connected spheres using running centroid distance
  const int max_merge_dist_sq =
      50 * 50; // max sphere diameter in Aimlabs is ~70px
  std::vector<bool> visited(hits.size(), false);

  for (std::size_t i = 0; i < hits.size(); ++i) {
    if (visited[i])
      continue;

    visited[i] = true;
    std::uint32_t cluster_hits = 1;
    double sum_x = hits[i].x;
    double sum_y = hits[i].y;
    int min_x = hits[i].x;
    int max_x = hits[i].x;
    int min_y = hits[i].y;
    int max_y = hits[i].y;

    for (std::size_t j = i + 1; j < hits.size(); ++j) {
      if (visited[j])
        continue;

      const double cur_cx = sum_x / cluster_hits;
      const double cur_cy = sum_y / cluster_hits;
      const double dx = hits[j].x - cur_cx;
      const double dy = hits[j].y - cur_cy;
      if (dx * dx + dy * dy < max_merge_dist_sq) {
        visited[j] = true;
        sum_x += hits[j].x;
        sum_y += hits[j].y;
        min_x = std::min(min_x, hits[j].x);
        max_x = std::max(max_x, hits[j].x);
        min_y = std::min(min_y, hits[j].y);
        max_y = std::max(max_y, hits[j].y);
        ++cluster_hits;
      }
    }

    const int w = max_x - min_x + step;
    const int h = max_y - min_y + step;
    const float aspect =
        static_cast<float>(w) / std::max(1.0f, static_cast<float>(h));

    // Filter: minimum hits (at step=3, a target sphere has >= 40 hits),
    // spherical aspect ratio (0.60 - 1.65), and bounded size (12px - 85px).
    // cluster_hits >= 25 cleanly rejects small text and UI icons.
    if (cluster_hits >= 25 && cluster_hits <= 1000 && w >= 12 && w <= 85 &&
        h >= 12 && h <= 85 && aspect >= 0.60f && aspect <= 1.65f) {

      const float cx = static_cast<float>(sum_x / cluster_hits);
      const float cy = static_cast<float>(sum_y / cluster_hits);

      // Ignore tiny cluster exactly at center (crosshair)
      const float dist_from_center =
          std::hypot(cx - width * 0.5f, cy - height * 0.5f);
      if (dist_from_center < 16.0f && cluster_hits < 30) {
        continue;
      }

      DetectedBlob blob{};
      blob.cx = cx;
      blob.cy = cy;
      blob.pixel_count = cluster_hits;
      blob.radius = std::clamp(
          std::sqrt(static_cast<float>(cluster_hits * step * step) / 3.14159f),
          10.0f, 45.0f);
      blobs.push_back(blob);
      if (blobs.size() >= 16)
        break;
    }
  }

  // Merge any overlapping blobs (if a large sphere was split into two nearby
  // clusters)
  for (std::size_t i = 0; i < blobs.size(); ++i) {
    for (std::size_t j = i + 1; j < blobs.size();) {
      const float dx = blobs[i].cx - blobs[j].cx;
      const float dy = blobs[i].cy - blobs[j].cy;
      const float dist = std::hypot(dx, dy);
      if (dist < (blobs[i].radius + blobs[j].radius) * 0.75f) {
        // Merge j into i
        const float total_px =
            static_cast<float>(blobs[i].pixel_count + blobs[j].pixel_count);
        blobs[i].cx = (blobs[i].cx * blobs[i].pixel_count +
                       blobs[j].cx * blobs[j].pixel_count) /
                      total_px;
        blobs[i].cy = (blobs[i].cy * blobs[i].pixel_count +
                       blobs[j].cy * blobs[j].pixel_count) /
                      total_px;
        blobs[i].pixel_count += blobs[j].pixel_count;
        blobs[i].radius = std::max(blobs[i].radius, blobs[j].radius);
        blobs.erase(blobs.begin() + j);
      } else {
        ++j;
      }
    }
  }

  return blobs;
}

namespace {
struct Capture {
  HDC screen{nullptr};
  HDC memory{nullptr};
  HBITMAP bitmap{nullptr};
  HGDIOBJ previous{nullptr};
  void *pixels{nullptr};
  ~Capture() {
    if (previous && memory)
      SelectObject(memory, previous);
    if (bitmap)
      DeleteObject(bitmap);
    if (memory)
      DeleteDC(memory);
    if (screen)
      ReleaseDC(nullptr, screen);
  }
  bool initialize() {
    screen = GetDC(nullptr);
    if (!screen)
      return false;
    memory = CreateCompatibleDC(screen);
    if (!memory)
      return false;
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = 1920;
    info.bmiHeader.biHeight = -1080;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    bitmap =
        CreateDIBSection(memory, &info, DIB_RGB_COLORS, &pixels, nullptr, 0);
    if (!bitmap || !pixels)
      return false;
    previous = SelectObject(memory, bitmap);
    return previous && previous != HGDI_ERROR;
  }
  bool read() {
    return BitBlt(memory, 0, 0, 1920, 1080, screen, 0, 0, SRCCOPY) &&
           GdiFlush();
  }
};

bool aimlabs_focused() {
  const HWND window = GetForegroundWindow();
  if (!window)
    return false;
  DWORD process_id = 0;
  GetWindowThreadProcessId(window, &process_id);
  const HANDLE process =
      OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
  if (!process)
    return false;
  char name[MAX_PATH]{};
  DWORD size = MAX_PATH;
  const bool obtained =
      QueryFullProcessImageNameA(process, 0, name, &size) != FALSE;
  CloseHandle(process);
  if (!obtained)
    return false;
  const std::string_view path{name, size};
  const auto leaf = path.substr(path.find_last_of("\\/") + 1);
  // Exact process basename, never title or a substring of a development path.
  return _stricmp(std::string(leaf).c_str(), "AimLab_tb.exe") == 0 ||
         _stricmp(std::string(leaf).c_str(), "AimLab.exe") == 0;
}

void help() {
  std::cout
      << "Aimlabs observation-only diagnostic; NO mouse or keyboard output.\n"
         "Usage: aim_live_runner [--test] [--duration=SECONDS]\n"
         "--test uses synthetic observations and NullActuator; default "
         "observes focused Aimlabs at 1920x1080.\n"
         "F12 pauses; F11 resumes; Escape exits. Default duration: 30 seconds "
         "(1..300).\n"
         "Simulated shots are command candidates, not measured hits. See "
         "docs/DELIVERY.md.\n";
}
} // namespace

int main(int argc, char **argv) {
  bool synthetic = false;
  int duration = 30;
  for (int i = 1; i < argc; ++i) {
    const std::string argument{argv[i]};
    if (argument == "--help") {
      help();
      return 0;
    }
    if (argument == "--test") {
      synthetic = true;
      continue;
    }
    if (argument.rfind("--duration=", 0) == 0) {
      try {
        std::size_t parsed = 0;
        const auto value = argument.substr(11);
        duration = std::stoi(value, &parsed);
        if (parsed == value.size() && duration >= 1 && duration <= 300)
          continue;
      } catch (...) {
      }
    }
    std::cerr << "Invalid option: " << argument
              << ". Physical actuation, arbitrary windows and auto-play are "
                 "unavailable.\n";
    return 2;
  }
  help();
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  Capture capture;
  if (!synthetic &&
      (GetSystemMetrics(SM_CXSCREEN) != 1920 ||
       GetSystemMetrics(SM_CYSCREEN) != 1080 || !capture.initialize())) {
    std::cerr << "Requires a capturable primary 1920x1080 desktop.\n";
    return 1;
  }
  tracking::TrackingEngine tracker;
  policy::UtilityAimPolicy policy;
  policy::ShotGate gate;
  NullActuator actuator;
  if (!tracker.initialize(aim::TrackerConfig{}) || !policy.initialize({}) ||
      !actuator.initialize({}) || !actuator.start())
    return 1;
  QpcClock clock;
  const MonotonicNs began = clock.now_ns();
  SequenceId frame = 0;
  SequenceId command_sequence = 0;
  std::uint64_t simulated_shots = 0;
  std::uint64_t stale_frames = 0;
  std::uint64_t rejected_dispatches = 0;
  std::array<std::uint64_t, 11> gate_counts{};
  bool paused = false;
  bool has_focus = false;
  MonotonicNs last_report = began;
  while (clock.now_ns() - began <
         static_cast<MonotonicNs>(duration) * 1'000'000'000LL) {
    if (GetAsyncKeyState(VK_ESCAPE) & 0x8000)
      break;
    if (GetAsyncKeyState(VK_F12) & 0x8000)
      paused = true;
    else if (GetAsyncKeyState(VK_F11) & 0x8000)
      paused = false;
    const bool focused = synthetic || aimlabs_focused();
    if (!focused || paused) {
      actuator.cancel_pending();
      // Preserve track IDs/shot receipts across a short focus interruption.
      has_focus = false;
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
      continue;
    }
    if (!has_focus) {
      std::cout << "Observing " << (synthetic ? "synthetic targets" : "Aimlabs")
                << "; no OS input.\n";
      has_focus = true;
    }
    const MonotonicNs captured_at = clock.now_ns();
    if (!synthetic &&
        (!capture.read() || GetSystemMetrics(SM_CXSCREEN) != 1920 ||
         GetSystemMetrics(SM_CYSCREEN) != 1080)) {
      actuator.cancel_pending();
      std::cerr << "Capture failed or display changed; diagnostic stopped.\n";
      return 1;
    }
    bus::TargetObservationBatch observations{};
    observations.header.sequence_id = ++frame;
    observations.header.source_timestamp_ns = captured_at;
    observations.header.pipeline_run_id = 1;
    observations.captured_at_ns = captured_at;
    if (synthetic) {
      bus::TargetObservation target{};
      target.source_id = 1;
      target.center_px = {960.0f, 540.0f};
      target.center_norm = {0.0f, 0.0f};
      target.effective_radius_px = 20.0f;
      target.confidence = 0.95f;
      target.covariance_px2 = {1.0f, 0.0f, 1.0f};
      observations.add_target(target);
    } else {
      int brightness = 0;
      const auto blobs =
          detect_spheres(static_cast<const std::uint8_t *>(capture.pixels),
                         1920, 1080, 0, 0, 1920, 1080, 3, brightness);
      std::uint64_t source_id = 0;
      for (const auto &blob : blobs) {
        bus::TargetObservation target{};
        target.source_id = ++source_id;
        target.center_px = {blob.cx, blob.cy};
        target.center_norm = {blob.cx / 960.0f - 1.0f, blob.cy / 540.0f - 1.0f};
        target.effective_radius_px = blob.radius;
        // Heuristic placeholders, explicitly NOT measured
        // confidence/covariance.
        target.confidence = 0.80f;
        target.covariance_px2 = {9.0f, 0.0f, 9.0f};
        observations.add_target(target);
      }
    }
    const MonotonicNs now = clock.now_ns();
    if (now - captured_at > 10'000'000LL) {
      ++stale_frames;
      actuator.cancel_pending();
      continue;
    }
    bus::TrackedTargetBatch tracks{};
    if (!tracker.update(observations, now + 2'000'000LL, tracks)) {
      actuator.cancel_pending();
      return 1;
    }
    PolicyInput input{};
    input.correlation_id = observations.header;
    input.decision_time_ns = now;
    input.tracks = &tracks;
    bus::AimIntent intent{};
    if (!policy.choose(input, intent))
      intent = {};
    const bus::TrackedTarget *selected = nullptr;
    for (const auto &track : tracks.items()) {
      if (track.track_id == intent.target_track_id) {
        selected = &track;
        break;
      }
    }
    auto shot = gate.evaluate(selected, input.crosshair.center_px, now);
    if (shot.authorize_fire && !intent.authorize_fire) {
      shot.authorize_fire = false;
      shot.decision = policy::ShotGateDecision::rejected_policy_veto;
    }
    ++gate_counts[static_cast<std::size_t>(shot.decision)];
    if (shot.authorize_fire && selected) {
      const auto dispatch_time = clock.now_ns();
      if (dispatch_time - captured_at > 10'000'000LL) {
        ++stale_frames;
        actuator.cancel_pending();
        continue;
      }
      ActuationCommand command{};
      command.sequence_id = ++command_sequence;
      command.correlation_id = observations.header;
      command.generated_at_ns = captured_at;
      command.desired_apply_time_ns = dispatch_time;
      command.button_transition = shot.button_transition;
      if (actuator.submit_latest(command) == SubmitResult::submitted) {
        gate.record_dispatched_shot(*selected, dispatch_time);
        ++simulated_shots;
      } else {
        ++rejected_dispatches;
        actuator.cancel_pending();
      }
    }
    if (now - last_report >= 1'000'000'000LL) {
      std::cout << "Frames=" << frame << " simulated shots=" << simulated_shots
                << " stale=" << stale_frames << '\n';
      last_report = now;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(7));
  }
  actuator.shutdown();
  std::cout << "Frames=" << frame << " simulated shots=" << simulated_shots
            << " stale=" << stale_frames
            << " rejected dispatches=" << rejected_dispatches << '\n';
  for (std::size_t i = 0; i < gate_counts.size(); ++i) {
    std::cout << policy::shot_gate_decision_to_string(
                     static_cast<policy::ShotGateDecision>(i))
              << '=' << gate_counts[i] << '\n';
  }
  return 0;
}
