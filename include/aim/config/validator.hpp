// include/aim/config/validator.hpp
#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "aim/config/types.hpp"

namespace aim::config {

enum class HotReloadStatus {
    Allowed = 0,
    RequiresQuiescentRestart = 1,
    InvalidValue = 2
};

struct HotReloadCheckResult {
    HotReloadStatus status{HotReloadStatus::Allowed};
    std::vector<std::string> modified_allowed_fields{};
    std::vector<std::string> conflicting_structural_fields{};
    std::vector<std::string> validation_errors{};

    [[nodiscard]] bool is_allowed() const noexcept {
        return status == HotReloadStatus::Allowed;
    }
};

class Validator {
public:
    static bool validate_and_clamp(
        ResolvedConfig& cfg,
        std::vector<std::string>& warnings,
        std::vector<std::string>& errors)
    {
        bool is_valid = true;

        // 1. Schema version check
        if (cfg.schema_version != 1) {
            errors.push_back("Unsupported schema_version: " + std::to_string(cfg.schema_version) + " (must be 1)");
            is_valid = false;
        }

        // 2. Fatal structural checks
        if (cfg.runtime.internal_deadline_ms <= 0.0) {
            errors.push_back("runtime.internal_deadline_ms must be positive, got: " + std::to_string(cfg.runtime.internal_deadline_ms));
            is_valid = false;
        }
        if (cfg.runtime.target_p99_ms <= 0.0) {
            errors.push_back("runtime.target_p99_ms must be positive, got: " + std::to_string(cfg.runtime.target_p99_ms));
            is_valid = false;
        }
        if (cfg.capture.source_width == 0 || cfg.capture.source_height == 0) {
            errors.push_back("capture source dimensions must be non-zero");
            is_valid = false;
        }
        if (cfg.perception.input_width == 0 || cfg.perception.input_height == 0) {
            errors.push_back("perception input dimensions must be non-zero");
            is_valid = false;
        }
        if (cfg.perception.batch != 1) {
            errors.push_back("perception.batch must be 1 (strictly batch-1 hot path)");
            is_valid = false;
        }
        if (cfg.actuator.scheduler_hz == 0) {
            errors.push_back("actuator.scheduler_hz must be greater than 0");
            is_valid = false;
        }

        // 3. Enum validation
        static const std::unordered_set<std::string> kValidCaptureBackends{"dxgi", "wgc", "replay", "null"};
        if (!kValidCaptureBackends.contains(cfg.capture.backend)) {
            errors.push_back("Invalid capture.backend: '" + cfg.capture.backend + "'");
            is_valid = false;
        }

        static const std::unordered_set<std::string> kValidCaptureFallbacks{"wgc", "none", "null"};
        if (!kValidCaptureFallbacks.contains(cfg.capture.fallback)) {
            errors.push_back("Invalid capture.fallback: '" + cfg.capture.fallback + "'");
            is_valid = false;
        }

        static const std::unordered_set<std::string> kValidPixelFormats{"bgra8_sdr", "rgba8_sdr", "nv12"};
        if (!kValidPixelFormats.contains(cfg.capture.pixel_format)) {
            errors.push_back("Invalid capture.pixel_format: '" + cfg.capture.pixel_format + "'");
            is_valid = false;
        }

        static const std::unordered_set<std::string> kValidPrecisions{"fp16", "fp32", "int8"};
        if (!kValidPrecisions.contains(cfg.perception.precision)) {
            errors.push_back("Invalid perception.precision: '" + cfg.perception.precision + "'");
            is_valid = false;
        }

        static const std::unordered_set<std::string> kValidObjectives{"raw_score", "min_switch_cost", "nearest_target", "highest_confidence"};
        if (!kValidObjectives.contains(cfg.policy.objective)) {
            errors.push_back("Invalid policy.objective: '" + cfg.policy.objective + "'");
            is_valid = false;
        }

        static const std::unordered_set<std::string> kValidActuatorBackends{"sendinput", "usb_hid", "null"};
        if (!kValidActuatorBackends.contains(cfg.actuator.backend)) {
            errors.push_back("Invalid actuator.backend: '" + cfg.actuator.backend + "'");
            is_valid = false;
        }

        if (!is_valid) {
            return false;
        }

        // 4. Safe continuous parameter clamping with diagnostic warnings
        auto clamp_val = [&warnings](auto& val, auto min_v, auto max_v, const char* name) {
            if (val < min_v) {
                std::ostringstream oss;
                oss << "Clamped " << name << " from " << val << " to minimum " << min_v;
                warnings.push_back(oss.str());
                val = min_v;
            } else if (val > max_v) {
                std::ostringstream oss;
                oss << "Clamped " << name << " from " << val << " to maximum " << max_v;
                warnings.push_back(oss.str());
                val = max_v;
            }
        };

        clamp_val(cfg.perception.confidence_floor, 0.0, 1.0, "perception.confidence_floor");
        clamp_val(cfg.perception.nms_iou_threshold, 0.0, 1.0, "perception.nms_iou_threshold");
        clamp_val(cfg.tracking.immediate_confidence, 0.0, 1.0, "tracking.immediate_confidence");
        clamp_val(cfg.tracking.require_second_observation_below, 0.0, 1.0, "tracking.require_second_observation_below");
        clamp_val(cfg.policy.switch_hysteresis, 0.0, 10.0, "policy.switch_hysteresis");
        clamp_val(cfg.runtime.warmup_iterations, 0u, 10000u, "runtime.warmup_iterations");
        clamp_val(cfg.trajectory.small_error_threshold_px, 0.0, 500.0, "trajectory.small_error_threshold_px");
        clamp_val(cfg.trajectory.max_velocity_counts_per_s, 100.0, 1000000.0, "trajectory.max_velocity_counts_per_s");
        clamp_val(cfg.trajectory.max_acceleration_counts_per_s2, 1000.0, 100000000.0, "trajectory.max_acceleration_counts_per_s2");
        clamp_val(cfg.trajectory.max_jerk_counts_per_s3, 10000.0, 1000000000.0, "trajectory.max_jerk_counts_per_s3");
        clamp_val(cfg.actuator.scheduler_hz, 100u, 8000u, "actuator.scheduler_hz");

        return true;
    }

    static HotReloadCheckResult check_hot_reload(
        const ResolvedConfig& current,
        const ResolvedConfig& updated)
    {
        HotReloadCheckResult result{};

        // Validate updated configuration first
        ResolvedConfig temp = updated;
        std::vector<std::string> warnings;
        if (!validate_and_clamp(temp, warnings, result.validation_errors)) {
            result.status = HotReloadStatus::InvalidValue;
            return result;
        }

        // Check runtime domain
        if (current.runtime.internal_deadline_ms != updated.runtime.internal_deadline_ms) {
            result.modified_allowed_fields.push_back("runtime.internal_deadline_ms");
        }
        if (current.runtime.target_p99_ms != updated.runtime.target_p99_ms) {
            result.modified_allowed_fields.push_back("runtime.target_p99_ms");
        }
        if (current.runtime.warmup_iterations != updated.runtime.warmup_iterations ||
            current.runtime.allocation_audit != updated.runtime.allocation_audit ||
            current.runtime.thread_priority != updated.runtime.thread_priority ||
            current.runtime.cpu_affinity_mask != updated.runtime.cpu_affinity_mask)
        {
            result.conflicting_structural_fields.push_back("runtime.structural_settings");
        }

        // Check capture domain (all structural)
        if (current.capture != updated.capture) {
            result.conflicting_structural_fields.push_back("capture");
        }

        // Check perception domain
        if (current.perception.confidence_floor != updated.perception.confidence_floor) {
            result.modified_allowed_fields.push_back("perception.confidence_floor");
        }
        if (current.perception.nms_iou_threshold != updated.perception.nms_iou_threshold) {
            result.modified_allowed_fields.push_back("perception.nms_iou_threshold");
        }
        if (current.perception.plugin != updated.perception.plugin ||
            current.perception.precision != updated.perception.precision ||
            current.perception.input_width != updated.perception.input_width ||
            current.perception.input_height != updated.perception.input_height ||
            current.perception.max_targets != updated.perception.max_targets ||
            current.perception.cuda_graph != updated.perception.cuda_graph ||
            current.perception.model_path != updated.perception.model_path)
        {
            result.conflicting_structural_fields.push_back("perception.structural_settings");
        }

        // Check tracking domain
        if (current.tracking.max_missed_frames != updated.tracking.max_missed_frames) {
            result.modified_allowed_fields.push_back("tracking.max_missed_frames");
        }
        if (current.tracking.immediate_confidence != updated.tracking.immediate_confidence) {
            result.modified_allowed_fields.push_back("tracking.immediate_confidence");
        }
        if (current.tracking.require_second_observation_below != updated.tracking.require_second_observation_below) {
            result.modified_allowed_fields.push_back("tracking.require_second_observation_below");
        }
        if (current.tracking.engage_if_uncertainty_within_radius != updated.tracking.engage_if_uncertainty_within_radius) {
            result.modified_allowed_fields.push_back("tracking.engage_if_uncertainty_within_radius");
        }
        if (current.tracking.process_noise_scale != updated.tracking.process_noise_scale) {
            result.modified_allowed_fields.push_back("tracking.process_noise_scale");
        }
        if (current.tracking.measurement_noise_scale != updated.tracking.measurement_noise_scale) {
            result.modified_allowed_fields.push_back("tracking.measurement_noise_scale");
        }
        if (current.tracking.gating_threshold_chi2 != updated.tracking.gating_threshold_chi2) {
            result.modified_allowed_fields.push_back("tracking.gating_threshold_chi2");
        }
        if (current.tracking.models != updated.tracking.models ||
            current.tracking.association != updated.tracking.association)
        {
            result.conflicting_structural_fields.push_back("tracking.structural_settings");
        }

        // Check policy domain
        if (current.policy.switch_hysteresis != updated.policy.switch_hysteresis) {
            result.modified_allowed_fields.push_back("policy.switch_hysteresis");
        }
        if (current.policy.horizon_targets != updated.policy.horizon_targets) {
            result.modified_allowed_fields.push_back("policy.horizon_targets");
        }
        if (current.policy.target_lead_time_ms != updated.policy.target_lead_time_ms) {
            result.modified_allowed_fields.push_back("policy.target_lead_time_ms");
        }
        if (current.policy.plugin != updated.policy.plugin ||
            current.policy.objective != updated.policy.objective)
        {
            result.conflicting_structural_fields.push_back("policy.structural_settings");
        }

        // Check trajectory domain (structural changes require restart)
        if (current.trajectory != updated.trajectory) {
            result.conflicting_structural_fields.push_back("trajectory");
        }

        // Check actuator domain (structural changes require restart)
        if (current.actuator != updated.actuator) {
            result.conflicting_structural_fields.push_back("actuator");
        }

        // Check safety domain
        if (current.safety.fail_closed != updated.safety.fail_closed) {
            result.modified_allowed_fields.push_back("safety.fail_closed");
        }
        if (current.safety.max_delta_counts_per_dispatch != updated.safety.max_delta_counts_per_dispatch) {
            result.modified_allowed_fields.push_back("safety.max_delta_counts_per_dispatch");
        }
        if (current.safety.max_active_engagement_seconds != updated.safety.max_active_engagement_seconds) {
            result.modified_allowed_fields.push_back("safety.max_active_engagement_seconds");
        }
        if (current.safety.require_foreground_match != updated.safety.require_foreground_match ||
            current.safety.target_process_name != updated.safety.target_process_name ||
            current.safety.target_window_title != updated.safety.target_window_title ||
            current.safety.require_emergency_stop != updated.safety.require_emergency_stop ||
            current.safety.emergency_stop_key != updated.safety.emergency_stop_key)
        {
            result.conflicting_structural_fields.push_back("safety.structural_settings");
        }

        if (!result.conflicting_structural_fields.empty()) {
            result.status = HotReloadStatus::RequiresQuiescentRestart;
        } else {
            result.status = HotReloadStatus::Allowed;
        }

        return result;
    }
};

} // namespace aim::config
