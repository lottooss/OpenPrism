// include/aim/config/types.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aim::config {

struct RuntimeConfig {
    double internal_deadline_ms{10.0};
    double target_p99_ms{6.0};
    std::uint32_t warmup_iterations{200};
    bool allocation_audit{true};
    std::string thread_priority{"high"};
    std::uint32_t cpu_affinity_mask{0};

    [[nodiscard]] bool operator==(const RuntimeConfig&) const = default;
};

struct CaptureConfig {
    std::string backend{"dxgi"};
    std::string fallback{"wgc"};
    std::uint32_t source_width{1920};
    std::uint32_t source_height{1080};
    std::string pixel_format{"bgra8_sdr"};
    std::uint32_t buffers{2};
    bool latest_only{true};
    double stale_after_ms{12.0};
    std::uint32_t display_index{0};
    bool allow_cross_adapter_copy{false};

    [[nodiscard]] bool operator==(const CaptureConfig&) const = default;
};

struct PerceptionConfig {
    std::string plugin{"yolo11n_aimlabs"};
    std::string precision{"fp16"};
    std::uint32_t batch{1};
    std::uint32_t input_width{640};
    std::uint32_t input_height{384};
    std::uint32_t max_targets{64};
    bool cuda_graph{true};
    double confidence_floor{0.20};
    double nms_iou_threshold{0.45};
    std::string model_path{""};
    std::string model_sha256{""};

    [[nodiscard]] bool operator==(const PerceptionConfig&) const = default;
};

struct TrackingConfig {
    std::vector<std::string> models{"stationary", "constant_velocity", "constant_acceleration"};
    std::string association{"hungarian"};
    std::uint32_t max_missed_frames{3};
    double immediate_confidence{0.85};
    double require_second_observation_below{0.85};
    bool engage_if_uncertainty_within_radius{true};
    double process_noise_scale{1.0};
    double measurement_noise_scale{1.0};
    double gating_threshold_chi2{9.21};

    [[nodiscard]] bool operator==(const TrackingConfig&) const = default;
};

struct PolicyConfig {
    std::string plugin{"deterministic_utility"};
    std::string objective{"raw_score"};
    std::uint32_t horizon_targets{3};
    double switch_hysteresis{0.08};
    double target_lead_time_ms{0.0};

    [[nodiscard]] bool operator==(const PolicyConfig&) const = default;
};

struct TrajectoryConfig {
    std::string small_error_mode{"direct_feedforward"};
    std::string large_error_mode{"jerk_limited"};
    std::string terminal_mode{"critically_damped_pd"};
    std::string optional_profile{"minimum_jerk"};
    double small_error_threshold_px{15.0};
    double max_velocity_counts_per_s{50000.0};
    double max_acceleration_counts_per_s2{500000.0};
    double max_jerk_counts_per_s3{10000000.0};
    double pd_kp{1.0};
    double pd_kd{0.1};

    [[nodiscard]] bool operator==(const TrajectoryConfig&) const = default;
};

struct ActuatorConfig {
    std::string backend{"sendinput"};
    std::uint32_t scheduler_hz{1000};
    bool relative_counts{true};
    bool cancel_superseded{true};
    double counts_per_pixel_x{1.0};
    double counts_per_pixel_y{1.0};
    std::string hid_com_port{""};
    std::uint32_t hid_baud_rate{115200};

    [[nodiscard]] bool operator==(const ActuatorConfig&) const = default;
};

struct SafetyConfig {
    bool require_foreground_match{true};
    std::string target_process_name{"Aimlab_tb.exe"};
    std::string target_window_title{"Aimlabs"};
    bool require_emergency_stop{true};
    std::string emergency_stop_key{"F12"};
    bool fail_closed{true};
    std::uint32_t max_delta_counts_per_dispatch{500};
    double max_active_engagement_seconds{60.0};

    [[nodiscard]] bool operator==(const SafetyConfig&) const = default;
};

struct ResolvedConfig {
    std::uint32_t schema_version{1};
    RuntimeConfig runtime{};
    CaptureConfig capture{};
    PerceptionConfig perception{};
    TrackingConfig tracking{};
    PolicyConfig policy{};
    TrajectoryConfig trajectory{};
    ActuatorConfig actuator{};
    SafetyConfig safety{};

    [[nodiscard]] bool operator==(const ResolvedConfig&) const = default;
};

} // namespace aim::config
