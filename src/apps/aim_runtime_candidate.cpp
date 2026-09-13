// Native integration candidate. Hardware acceptance remains a measured test,
// not a property inferred from successful startup or an artifact hash.
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "aim/actuation/sendinput_actuator.hpp"
#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/core/clock.hpp"
#include "aim/perception/capture_to_tensor_pipeline.hpp"
#include "aim/perception/tensorrt_engine_runner.hpp"
#include "aim/pipeline/closed_loop_pipeline.hpp"
#include "aim/policy/utility_policy.hpp"
#include "aim/runtime/calibration_evidence.hpp"
#include "aim/runtime/calibration_profile_loader.hpp"
#include "aim/runtime/visual_calibration_session.hpp"
#include "aim/safety/safety_supervisor.hpp"
#include "aim/scenario/scenario_profile_loader.hpp"
#include "aim/tracking/tracking_engine.hpp"
#include "aim/trajectory/trajectory_planner.hpp"

namespace {
using namespace aim;

struct Options {
    std::filesystem::path assets;
    std::filesystem::path engine;
    std::filesystem::path calibration;
    std::filesystem::path scenario;
    std::filesystem::path report;
    std::string hash;
    int seconds{30};
    float fov{0.0f};
    float sensitivity{0.0f};
    bool actuate{false};
    bool preflight{false};
    bool calibrate{false};
    bool observe{false};
};

template <typename T> T number(std::string_view text) {
    T result{};
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Invalid numeric option");
    return result;
}

Options parse(int argc, char **argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--actuate")
            result.actuate = true;
        else if (arg == "--preflight")
            result.preflight = true;
        else if (arg == "--calibrate")
            result.calibrate = true;
        else if (arg == "--observe")
            result.observe = true;
        else if (arg.starts_with("--assets="))
            result.assets = arg.substr(9);
        else if (arg.starts_with("--engine="))
            result.engine = arg.substr(9);
        else if (arg.starts_with("--sha256="))
            result.hash = arg.substr(9);
        else if (arg.starts_with("--calibration="))
            result.calibration = arg.substr(14);
        else if (arg.starts_with("--scenario="))
            result.scenario = arg.substr(11);
        else if (arg.starts_with("--report="))
            result.report = arg.substr(9);
        else if (arg.starts_with("--duration="))
            result.seconds = number<int>(arg.substr(11));
        else if (arg.starts_with("--fov="))
            result.fov = number<float>(arg.substr(6));
        else if (arg.starts_with("--sensitivity="))
            result.sensitivity = number<float>(arg.substr(14));
        else
            throw std::runtime_error("Unknown option");
    }
    if (result.assets.empty() || result.engine.empty() || result.hash.empty() ||
        (!result.observe && result.calibration.empty()) ||
        result.scenario.empty() || result.seconds < 1 || result.seconds > 300 ||
        !std::isfinite(result.fov) || !std::isfinite(result.sensitivity) ||
        result.fov < 30.0f || result.fov > 150.0f ||
        result.sensitivity < 0.001f || result.sensitivity > 100.0f ||
        (result.observe && (result.calibrate || result.actuate)) ||
        (result.calibrate && result.actuate) ||
        (result.preflight && (result.calibrate || result.actuate)))
        throw std::runtime_error("Missing required assets/settings or invalid "
                                 "duration (1..300 seconds)");
    return result;
}

std::filesystem::path trusted_engine_path(const Options &options) {
    const auto root = std::filesystem::canonical(options.assets);
    const auto path = std::filesystem::canonical(root / options.engine);
    auto part = path.begin();
    for (auto required = root.begin(); required != root.end();
         ++required, ++part) {
        if (part == path.end() || *part != *required)
            throw std::runtime_error("Engine resolves outside the configured "
                                     "trusted artifact directory");
    }
    return path;
}

// Fixed buffers are reused per observation. These are ordinary OS window
// identity queries; no target process memory is accessed.
struct Foreground {
    std::array<char, 32768> process{};
    std::array<char, 512> title{};
    std::uintptr_t window_handle{0};
    std::uint32_t process_id{0};
    ObservableScenarioState sample(MonotonicNs now, float elapsed) noexcept {
        window_handle = 0;
        process_id = 0;
        ObservableScenarioState state{};
        state.observed_at_ns = now;
        state.elapsed_session_seconds = elapsed;
        const HWND window = GetForegroundWindow();
        if (!window || IsIconic(window))
            return state;
        RECT client{};
        POINT origin{};
        if (!GetClientRect(window, &client) ||
            !ClientToScreen(window, &origin) || origin.x != 0 ||
            origin.y != 0 || client.right != 1920 || client.bottom != 1080)
            return state;
        DWORD observed_process_id{};
        GetWindowThreadProcessId(window, &observed_process_id);
        const HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                          FALSE, observed_process_id);
        if (!handle)
            return state;
        DWORD length = static_cast<DWORD>(process.size());
        const BOOL queried =
            QueryFullProcessImageNameA(handle, 0, process.data(), &length);
        CloseHandle(handle);
        if (!queried || length == 0 || length >= process.size())
            return state;
        const std::string_view path{process.data(), length};
        state.active_process_name = path.substr(path.find_last_of("\\/") + 1U);
        const int title_length = GetWindowTextA(window, title.data(),
                                                static_cast<int>(title.size()));
        if (title_length > 0)
            state.active_window_title = {
                title.data(), static_cast<std::size_t>(title_length)};
        state.foreground_confirmed = GetForegroundWindow() == window;
        if (state.foreground_confirmed) {
            window_handle = reinterpret_cast<std::uintptr_t>(window);
            process_id = observed_process_id;
        }
        return state;
    }
};

// A queued hotkey retains short key taps which polling GetAsyncKeyState's
// current-down bit can miss during the cold operator wait.
class ArmHotkey {
  public:
    explicit ArmHotkey(bool enabled) : enabled_(enabled) {
        if (enabled_ && !RegisterHotKey(nullptr, kId, MOD_NOREPEAT, VK_F11))
            throw std::runtime_error(
                "F11 is already registered by another application");
        if (enabled_)
            static_cast<void>(GetAsyncKeyState(VK_F12));
    }
    ~ArmHotkey() {
        if (enabled_)
            UnregisterHotKey(nullptr, kId);
    }
    void poll() noexcept {
        MSG message{};
        for (int i = 0; i < 8 && PeekMessageW(&message, nullptr, WM_HOTKEY,
                                              WM_HOTKEY, PM_REMOVE);
             ++i) {
            if (message.wParam == kId)
                armed_ = true;
        }
        // Windows reserves F12 for debuggers, so it cannot be registered.
        // The current-down bit remains authoritative; the transition bit also
        // catches short taps between polls. Clear old transitions on setup.
        if (enabled_ && (GetAsyncKeyState(VK_F12) & 0x8001) != 0)
            stopped_ = true;
    }
    bool received() noexcept {
        poll();
        return armed_;
    }
    bool stopped() noexcept {
        poll();
        return stopped_;
    }

  private:
    static constexpr int kId = 0x4149;
    bool enabled_;
    bool armed_{false};
    bool stopped_{false};
};

struct ObservationTrace {
    std::array<bus::TargetObservationBatch, 128> frames{};
    std::array<MonotonicNs, 128> collected_at{};
    std::size_t count{0};
    struct Pulse {
        runtime::VisualCalibrationPulse pulse{};
        MonotonicNs dispatched_at_ns{0};
        bool accepted{false};
    };
    std::array<Pulse, runtime::VisualCalibrationSession::kPulseCount> pulses{};
    std::size_t pulse_count{0};
};

int run(const Options &options) {
    if (!options.report.empty() && std::filesystem::exists(options.report))
        throw std::runtime_error(
            "Use a new output filename for the runtime report");
    calibration::CalibrationProfile profile{};
    profile.counts_per_pixel_x = 0.0f;
    profile.counts_per_pixel_y = 0.0f;
    profile.profile_id = "aimlabs_visual_measured";
    profile.fov_horizontal_deg = options.fov;
    profile.in_game_sensitivity = options.sensitivity;
    if (!options.calibrate && !options.observe)
        profile = runtime::load_calibration_profile(options.calibration);
    if (!options.calibrate && !options.observe &&
        !runtime::calibration_matches_session(profile, options.fov,
                                              options.sensitivity))
        throw std::runtime_error(
            "Calibration must meet held-out RMSE <=5px and match 1920x1080 and "
            "supplied FOV/sensitivity");
    const auto scenario_result =
        scenario::ScenarioProfileLoader::load_from_file(options.scenario);
    if (!scenario_result.success ||
        !scenario_result.profile.foreground.require_match ||
        scenario_result.profile.foreground.process_name.empty())
        throw std::runtime_error(
            "A valid scenario with an exact foreground executable is required");
    calibration::CalibrationModel calibration(profile);
    scenario::ScenarioAdapter scenario;
    if (scenario.load_profile(scenario_result.profile) !=
        ScenarioProfileStatus::ok)
        throw std::runtime_error("Scenario initialization failed");
    ModelManifest manifest;
    manifest.model_path = trusted_engine_path(options).string();
    manifest.engine_sha256 = options.hash;
    perception::VerifiedEngineArtifact verified;
    if (perception::EngineArtifactVerifier::load_and_verify(
            manifest, {}, verified) != PerceptionStatus::ok)
        throw std::runtime_error(
            "Engine artifact missing or SHA-256 verification failed");
    verified.reset();
    runtime::CalibrationRuntimeEvidence calibration_evidence{};
    if (options.actuate)
        calibration_evidence = runtime::load_calibration_evidence(
            options.calibration, options.hash);
    if (options.calibrate && std::filesystem::exists(options.calibration))
        throw std::runtime_error(
            "Use a new output filename for measured calibration");
    if (!perception::TensorRtEngineRunner::is_backend_compiled())
        throw std::runtime_error(
            "TensorRT SDK backend unavailable; rebuild with AIM_TENSORRT_ROOT");
    if (GetSystemMetrics(SM_CMONITORS) != 1 ||
        GetSystemMetrics(SM_CXSCREEN) != 1920 ||
        GetSystemMetrics(SM_CYSCREEN) != 1080)
        throw std::runtime_error(
            "Candidate requires one active 1920x1080 display to guarantee "
            "capture/viewport identity");

    auto clock = std::make_shared<QpcClock>();
    auto cuda = std::make_shared<capture::RealCudaInteropBackend>();
    perception::CaptureToTensorPipeline capture(nullptr, nullptr, clock);
    perception::CaptureToTensorConfig capture_config;
    capture_config.capture_config.timeout_ms = 0;
    if (!capture.initialize(capture_config, cuda) || !capture.warmup(10))
        throw std::runtime_error(
            "DXGI/WGC GPU preprocessing initialization failed");
    auto *source =
        dynamic_cast<capture::UnifiedCaptureSource *>(capture.frame_source());
    int device = -1;
    if (!source ||
        !cuda->validate_adapter_match(source->active_adapter_luid(), device) ||
        device < 0)
        throw std::runtime_error("Capture and CUDA device identity mismatch");
    manifest.target_gpu_device_id = static_cast<std::uint32_t>(device);
    // Declaration order also protects exception unwinding: inference drains
    // before the producer lease releases its input storage.
    perception::PreprocessedTensorLease tensor;
    perception::TensorRtEngineRunner perception;
    if (perception.initialize(manifest) != PerceptionStatus::ok ||
        perception.warmup(10) != PerceptionStatus::ok)
        throw std::runtime_error(
            "Real TensorRT deserialization/binding/warmup failed");

    tracking::TrackingEngine tracker;
    policy::UtilityAimPolicy policy;
    trajectory::TrajectoryPlanner planner;
    NullActuator null_actuator;
    actuation::SendInputActuator physical_actuator;
    IActuator *actuator = &null_actuator;
    if (options.actuate || options.calibrate) {
        if (scenario_result.profile.foreground.window_title_substring.empty())
            throw std::runtime_error(
                "Physical actuator requires a nonempty scenario window title");
        actuator = &physical_actuator;
    } else if (!null_actuator.initialize({})) {
        throw std::runtime_error("Null actuator initialization failed");
    }
    if (!tracker.initialize(aim::TrackerConfig{}) || !policy.initialize({}) ||
        !capture.start())
        throw std::runtime_error("Runtime initialization failed");
    safety::SafetySupervisorConfig safety_config;
    if (options.actuate)
        safety_config.max_single_step_counts =
            calibration_evidence.maximum_step_counts;
    safety::SafetySupervisor safety(safety_config);
    Foreground foreground;
    safety.set_focus_callback([&] {
        return scenario.foreground_authorized(
            foreground.sample(clock->now_ns(), 0.0f));
    });
    safety.set_calibration_callback([&] { return calibration.is_valid(); });
    pipeline::ClosedLoopPipelineConfig pipeline_config;
    if (options.actuate) {
        // Leave 2 ms for focus verification and OS dispatch. The actuator still
        // independently enforces the architecture's hard 10 ms source cutoff.
        pipeline_config.max_internal_latency_ns = 8'000'000LL;
        pipeline_config.command_lead_time_ns =
            calibration_evidence.effect_upper_p50_ns;
        pipeline_config.shot_gate_config.max_prediction_lead_ns =
            std::max<MonotonicNs>(10'000'000LL,
                                  calibration_evidence.effect_upper_p50_ns);
        const float gain_bound =
            std::max(profile.counts_per_pixel_x, profile.counts_per_pixel_y) *
            (1.0f + std::abs(profile.cross_coupling_xy));
        const float bounded_pixels =
            static_cast<float>(calibration_evidence.maximum_step_counts) /
            (2.0f * gain_bound);
        pipeline_config.planner_limits.max_velocity_px_s =
            std::min(5000.0f, bounded_pixels * 1000.0f);
        pipeline_config.planner_limits.small_error_threshold_px =
            std::min(6.0f, bounded_pixels);
    }
    pipeline::ClosedLoopPipeline pipeline(&tracker, &policy, &planner, actuator,
                                          &safety, &scenario, calibration,
                                          pipeline_config);
    runtime::VisualCalibrationSessionConfig calibration_config;
    // Small local probes limit perspective distortion and measure a tighter
    // range. The saved range also caps all subsequent production commands.
    calibration_config.fitting.max_pulse_counts = 32;
    calibration_config.horizontal_fov_deg = options.fov;
    runtime::VisualCalibrationSession calibration_session(calibration_config);
    auto trace =
        options.report.empty() ? nullptr : std::make_unique<ObservationTrace>();
    ArmHotkey arm_hotkey(!options.preflight);
    if (!options.preflight)
        std::cout
            << (options.calibrate
                    ? "VISUAL CALIBRATION: small mouse pulses, no shooting"
                : options.actuate ? "PHYSICAL INPUT candidate"
                : options.observe ? "OBSERVATION ONLY"
                                  : "NullActuator candidate")
            << ": focus the configured application; press F11 to start. F12 "
               "or Escape stops.\n"
            << std::flush;
    const auto waiting_since = clock->now_ns();
    while (!options.preflight && !arm_hotkey.received()) {
        if (arm_hotkey.stopped() ||
            (GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0 ||
            (GetAsyncKeyState(VK_F12) & 0x8000) != 0 ||
            clock->now_ns() - waiting_since > 60'000'000'000LL)
            return 2;
        Sleep(10); // operator arm wait, before the runtime hot path
    }
    const auto armed_foreground = foreground.sample(clock->now_ns(), 0.0f);
    if (!options.preflight && !scenario.foreground_authorized(armed_foreground))
        throw std::runtime_error(
            "Configured foreground must be focused when arming");
    if (options.actuate || options.calibrate) {
        actuation::SendInputConfig config;
        // Pin the exact observed title after the scenario has authorized the
        // executable and viewport. This also preserves the platform's casing.
        config.target_window_title = armed_foreground.active_window_title;
        config.target_window_handle = foreground.window_handle;
        config.target_process_id = foreground.process_id;
        if (!physical_actuator.initialize(config))
            throw std::runtime_error("Physical actuator initialization failed");
    }
    if (!actuator->start())
        throw std::runtime_error("Actuator could not start");
    const auto started = clock->now_ns();
    if (options.calibrate && !calibration_session.start(started))
        throw std::runtime_error("Calibration session could not start");
    MonotonicNs next_tick = started;
    InferenceTicket ticket;
    bool inference_pending = false;
    std::uint64_t frames = 0;
    std::uint64_t detections = 0;
    float maximum_confidence = 0.0f;
    std::uint64_t calibration_pulses = 0;
    runtime::VisualCalibrationStatus calibration_status{};
    int exit_code = 0;
    std::string_view termination_reason{"duration_complete"};
    std::uint64_t rejected_frames = 0;
    while (clock->now_ns() - started <
           static_cast<MonotonicNs>(options.preflight ? 3 : options.seconds) *
               1'000'000'000LL) {
        const auto now = clock->now_ns();
        if (arm_hotkey.stopped() || (GetAsyncKeyState(VK_F12) & 0x8000) != 0 ||
            (GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0) {
            termination_reason = "operator_stop";
            break;
        }
        auto state =
            foreground.sample(now, static_cast<float>(now - started) / 1.0e9f);
        if (scenario.scenario_complete(state)) {
            termination_reason = "scenario_complete";
            break;
        }
        if ((!options.preflight && !scenario.foreground_authorized(state)) ||
            !capture.is_healthy() || !capture.is_actuation_permitted() ||
            safety.is_latched() || actuator->health().is_latched ||
            GetSystemMetrics(SM_CMONITORS) != 1 ||
            GetSystemMetrics(SM_CXSCREEN) != 1920 ||
            GetSystemMetrics(SM_CYSCREEN) != 1080) {
            exit_code = 2;
            termination_reason =
                !scenario.foreground_authorized(state) ? "lost_focus"
                : !capture.is_healthy()                ? "capture_unhealthy"
                : !capture.is_actuation_permitted()    ? "capture_not_permitted"
                : safety.is_latched()                  ? "supervisor_latched"
                : actuator->health().is_latched        ? "actuator_latched"
                                                       : "display_changed";
            break;
        }
        if (options.calibrate) {
            calibration_status = calibration_session.poll(now, true).status;
            if (calibration_session.failed()) {
                exit_code = 2;
                termination_reason = "calibration_failed";
                break;
            }
            if (calibration_session.complete()) {
                termination_reason = "calibration_complete";
                break;
            }
        }
        if (!options.calibrate && !options.observe && !options.preflight &&
            now >= next_tick) {
            static_cast<void>(pipeline.tick_actuation(state, clock->now_ns()));
            next_tick =
                now + 1'000'000LL; // no catch-up burst after a delayed tick
        }
        if (inference_pending) {
            bus::TargetObservationBatch observations;
            const auto collected = perception.try_collect(ticket, observations);
            if (collected == PollResult::error ||
                collected == PollResult::empty) {
                exit_code = 2;
                break;
            }
            if (collected == PollResult::ready) {
                inference_pending = false;
                tensor.reset();
                if (trace && trace->count < trace->frames.size()) {
                    trace->frames[trace->count] = observations;
                    trace->collected_at[trace->count] = clock->now_ns();
                    ++trace->count;
                }
                if ((options.preflight || options.calibrate) &&
                    clock->now_ns() - observations.captured_at_ns >
                        10'000'000LL) {
                    // The acquisition state consumes only fresh measurements.
                    // Dropping this result cannot offer another pulse, and its
                    // existing response/session deadlines continue to run.
                    ++rejected_frames;
                    continue;
                }
                scenario.enrich(observations, state);
                state.visible_target_count = observations.target_count;
                detections += observations.target_count;
                for (const auto &target : observations.items())
                    maximum_confidence =
                        std::max(maximum_confidence, target.confidence);
                if (options.calibrate) {
                    const auto update = calibration_session.observe(
                        observations, clock->now_ns(), true);
                    calibration_status = update.status;
                    if (update.pulse) {
                        const auto &pulse = *update.pulse;
                        const auto dispatch_time = clock->now_ns();
                        const bool axis_only =
                            (pulse.counts_x == 0) != (pulse.counts_y == 0);
                        const bool bounded =
                            pulse.counts_x >= -64 && pulse.counts_x <= 64 &&
                            pulse.counts_y >= -64 && pulse.counts_y <= 64;
                        const bool fresh =
                            pulse.source_timestamp_ns > 0 &&
                            dispatch_time >= pulse.source_timestamp_ns &&
                            dispatch_time - pulse.source_timestamp_ns <=
                                10'000'000LL;
                        // Explicit calibration protocol only: no production
                        // plan or guessed calibration can authorize these
                        // bounded, button-free pulses.
                        const bool authorized =
                            axis_only && bounded && fresh &&
                            !arm_hotkey.stopped() &&
                            (GetAsyncKeyState(VK_F12) & 0x8000) == 0 &&
                            (GetAsyncKeyState(VK_ESCAPE) & 0x8000) == 0 &&
                            scenario.foreground_authorized(
                                foreground.sample(dispatch_time, 0.0f));
                        ActuationCommand command;
                        command.sequence_id = pulse.sequence;
                        command.correlation_id = observations.header;
                        command.correlation_id.source_timestamp_ns =
                            pulse.source_timestamp_ns;
                        command.generated_at_ns = dispatch_time;
                        command.desired_apply_time_ns = dispatch_time;
                        command.delta_x_counts = pulse.counts_x;
                        command.delta_y_counts = pulse.counts_y;
                        const bool accepted =
                            authorized && actuator->submit_latest(command) ==
                                              SubmitResult::submitted;
                        if (accepted)
                            ++calibration_pulses;
                        if (trace && trace->pulse_count < trace->pulses.size())
                            trace->pulses[trace->pulse_count++] = {
                                pulse, dispatch_time, accepted};
                        // The pre-submit monotonic bound includes the OS call;
                        // this conservatively brackets visible effect rather
                        // than inventing an effect timestamp.
                        if (!calibration_session.confirm_dispatch(
                                pulse.sequence, dispatch_time, accepted)) {
                            exit_code = 2;
                            break;
                        }
                    }
                } else if (!options.observe && !options.preflight) {
                    const auto prior_stale =
                        pipeline.stats().total_stale_dropped;
                    if (!pipeline.process_frame(observations, state,
                                                clock->now_ns())) {
                        ++rejected_frames;
                        // Stale frames already cancel pending motion. Wait for
                        // new evidence; never count rejected work as success.
                        if (pipeline.stats().total_stale_dropped > prior_stale)
                            continue;
                        exit_code = 2;
                        termination_reason = "pipeline_failed";
                        break;
                    }
                }
                ++frames;
                if (options.preflight && frames >= 8)
                    break;
            }
        } else if (!tensor && capture.process_frame(tensor)) {
            // Retain the tensor lease until collection; producer work is polled
            // before enqueue, avoiding both a CPU wait and a cross-stream race.
        }
        if (tensor && !inference_pending) {
            const auto &descriptor = tensor.descriptor();
            const auto enqueue_time = clock->now_ns();
            if (enqueue_time < descriptor.captured_at_ns ||
                enqueue_time - descriptor.captured_at_ns > 10'000'000LL) {
                tensor.reset();
            } else if (!descriptor.native_ready_event) {
                exit_code = 2;
                break;
            } else {
                const auto ready =
                    cuda->query_event(descriptor.native_ready_event);
                if (ready == capture::CudaResult::error_not_ready)
                    continue;
                if (ready != capture::CudaResult::success) {
                    exit_code = 2;
                    break;
                }
                const PerceptionRequest request{
                    descriptor.frame_id, descriptor.correlation_id,
                    descriptor.captured_at_ns, descriptor.gpu_tensor_ptr,
                    descriptor.size_bytes};
                if (perception.enqueue(request, ticket) !=
                    PerceptionStatus::ok) {
                    exit_code = 2;
                    break;
                }
                inference_pending = true;
            }
        }
        YieldProcessor();
    }
    const auto physical_stats = physical_actuator.stats();
    const auto safety_reason = safety.last_reason();
    actuator->emergency_stop();
    perception
        .shutdown(); // completes outstanding ownership before tensor release
    tensor.reset();
    capture.stop();
    actuator->shutdown();
    const auto stats = pipeline.stats();
    if (trace)
        try {
            nlohmann::json records = nlohmann::json::array();
            nlohmann::json pulses = nlohmann::json::array();
            for (std::size_t i = 0; i < trace->pulse_count; ++i) {
                const auto &entry = trace->pulses[i];
                pulses.push_back(
                    {{"sequence", entry.pulse.sequence},
                     {"counts_x", entry.pulse.counts_x},
                     {"counts_y", entry.pulse.counts_y},
                     {"source_timestamp_ns", entry.pulse.source_timestamp_ns},
                     {"dispatched_at_ns", entry.dispatched_at_ns},
                     {"accepted", entry.accepted},
                     {"held_out", entry.pulse.held_out}});
            }
            for (std::size_t i = 0; i < trace->count; ++i) {
                const auto &frame = trace->frames[i];
                nlohmann::json targets = nlohmann::json::array();
                for (const auto &target : frame.items())
                    targets.push_back({{"x", target.center_px.x},
                                       {"y", target.center_px.y},
                                       {"radius", target.effective_radius_px},
                                       {"confidence", target.confidence}});
                records.push_back({{"frame_id", frame.frame_id},
                                   {"captured_at_ns", frame.captured_at_ns},
                                   {"collected_at_ns", trace->collected_at[i]},
                                   {"targets", targets}});
            }
            if (!options.report.parent_path().empty())
                std::filesystem::create_directories(
                    options.report.parent_path());
            std::ofstream output(options.report);
            output.exceptions(std::ios::failbit | std::ios::badbit);
            output << nlohmann::json{{"engine_sha256", options.hash},
                                     {"frames_processed", frames},
                                     {"accepted_commands",
                                      stats.total_commands_dispatched},
                                     {"accepted_shots",
                                      stats.total_shots_authorized},
                                     {"accepted_calibration_pulses",
                                      calibration_pulses},
                                     {"calibration_pulses", pulses},
                                     {"calibration_status",
                                      static_cast<int>(calibration_status)},
                                     {"calibration_samples",
                                      calibration_session.samples()},
                                     {"termination_reason", termination_reason},
                                     {"safety_reason",
                                      static_cast<int>(safety_reason)},
                                     {"actuator_stale_dropped",
                                      physical_stats.stale_dropped},
                                     {"actuator_foreground_rejected",
                                      physical_stats.foreground_rejected},
                                     {"actuator_dispatch_failures",
                                      physical_stats.dispatch_failures},
                                     {"rejected_frames", rejected_frames},
                                     {"shots_rejected",
                                      stats.total_shots_rejected},
                                     {"safety_rejections",
                                      stats.total_safety_rejections},
                                     {"source_stale_dropped",
                                      stats.total_stale_dropped},
                                     {"trace_capacity", 128},
                                     {"observations", records}}
                          .dump(2)
                   << '\n';
        } catch (const std::exception &error) {
            // Diagnostics cannot discard a successfully measured calibration.
            // Input and GPU work have already stopped before this cold I/O.
            std::cerr << "Runtime report could not be written: " << error.what()
                      << '\n';
        }
    std::cout << "Frames=" << frames << " detections=" << detections
              << " max confidence=" << maximum_confidence
              << " accepted commands=" << stats.total_commands_dispatched
              << " accepted shots=" << stats.total_shots_authorized
              << " calibration pulses=" << calibration_pulses
              << " stale=" << stats.total_stale_dropped << '\n';
    if (options.calibrate) {
        if (!calibration_session.complete() || exit_code != 0) {
            std::cerr << "Calibration incomplete: accepted samples="
                      << calibration_session.samples()
                      << " status=" << static_cast<int>(calibration_status)
                      << ". No calibration was saved.\n";
            return 2;
        }
        const auto fit = calibration_session.fit(profile);
        if (!fit.succeeded()) {
            std::cerr << "Calibration rejected: fit status="
                      << static_cast<int>(fit.status) << '\n';
            return 2;
        }
        SYSTEMTIME utc{};
        GetSystemTime(&utc); // Measurement label only; never used by control.
        std::array<char, 32> date{};
        std::snprintf(
            date.data(), date.size(), "%04u-%02u-%02uT%02u:%02u:%02uZ",
            static_cast<unsigned>(utc.wYear), static_cast<unsigned>(utc.wMonth),
            static_cast<unsigned>(utc.wDay), static_cast<unsigned>(utc.wHour),
            static_cast<unsigned>(utc.wMinute),
            static_cast<unsigned>(utc.wSecond));
        runtime::save_calibration_measurement(options.calibration, fit,
                                              options.hash, date.data());
        std::cout << "Measured calibration saved; held-out RMSE="
                  << fit.profile.rmse_pixels
                  << "px, counts/pixel=" << fit.profile.counts_per_pixel_x
                  << ',' << fit.profile.counts_per_pixel_y << '\n';
    }
    if (options.preflight) {
        if (frames < 8 || exit_code != 0) {
            std::cerr << "Preflight requires 8 fresh real capture/inference "
                         "frames within 3 seconds.\n";
            return 2;
        }
        std::cout << "Real GPU capture and TensorRT inference preflight "
                     "passed; no input dispatched.\n";
    }
    return frames == 0 ? 2 : exit_code;
}
} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "aim_runtime_candidate --assets=DIR --engine=FILE "
                     "--sha256=HEX --calibration=JSON "
                     "--scenario=YAML --fov=DEGREES --sensitivity=VALUE "
                     "[--duration=1..300] "
                     "[--report=JSON] "
                     "[--preflight] [--actuate | --calibrate | --observe]\n"
                     "Defaults to NullActuator. Requires trusted real engine "
                     "and measured calibration.\n";
        return 0;
    }
    try {
        if (!SetProcessDpiAwarenessContext(
                DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) &&
            !AreDpiAwarenessContextsEqual(
                GetThreadDpiAwarenessContext(),
                DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
            throw std::runtime_error(
                "Physical-pixel DPI awareness could not be established");
        return run(parse(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "Preflight/runtime failed: " << error.what() << '\n';
        return 2;
    }
}
