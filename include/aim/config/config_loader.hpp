// include/aim/config/config_loader.hpp
#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "aim/config/sha256.hpp"
#include "aim/config/types.hpp"
#include "aim/config/validator.hpp"

namespace aim::config {

struct LoadResult {
    bool success{false};
    ResolvedConfig config{};
    nlohmann::json raw_json{};
    std::string sha256_hash{};
    std::vector<std::string> source_layers{};
    std::vector<std::string> errors{};
    std::vector<std::string> warnings{};
};

class ConfigLoader {
public:
    /// @brief Deep merge JSON source into target (dict recursive, list replacement, scalar overwrite)
    static void merge_json(nlohmann::json& target, const nlohmann::json& source) {
        if (!source.is_object()) {
            target = source;
            return;
        }

        if (!target.is_object()) {
            target = nlohmann::json::object();
        }

        for (auto it = source.begin(); it != source.end(); ++it) {
            const std::string& key = it.key();
            const auto& val = it.value();

            if (val.is_object() && target.contains(key) && target[key].is_object()) {
                merge_json(target[key], val);
            } else {
                target[key] = val; // Atomic replacement for arrays and scalars
            }
        }
    }

    /// @brief Convert YAML::Node to nlohmann::json
    static nlohmann::json yaml_to_json(const YAML::Node& node) {
        if (!node.IsDefined() || node.IsNull()) {
            return nullptr;
        }
        if (node.IsScalar()) {
            const std::string& scalar = node.Scalar();

            // Boolean check
            if (scalar == "true" || scalar == "True" || scalar == "TRUE") {
                return true;
            }
            if (scalar == "false" || scalar == "False" || scalar == "FALSE") {
                return false;
            }

            // Integer check
            try {
                std::size_t idx = 0;
                const long long int_val = std::stoll(scalar, &idx, 0);
                if (idx == scalar.size()) {
                    if (int_val >= 0 && static_cast<unsigned long long>(int_val) <= UINT32_MAX) {
                        return static_cast<std::uint32_t>(int_val);
                    }
                    return int_val;
                }
            } catch (...) {}

            // Double check
            try {
                std::size_t idx = 0;
                const double dbl_val = std::stod(scalar, &idx);
                if (idx == scalar.size()) {
                    return dbl_val;
                }
            } catch (...) {}

            return scalar;
        }
        if (node.IsSequence()) {
            auto arr = nlohmann::json::array();
            for (const auto& elem : node) {
                arr.push_back(yaml_to_json(elem));
            }
            return arr;
        }
        if (node.IsMap()) {
            auto obj = nlohmann::json::object();
            for (const auto& pair : node) {
                obj[pair.first.as<std::string>()] = yaml_to_json(pair.second);
            }
            return obj;
        }
        return nullptr;
    }

    /// @brief Validate JSON against strict schema definitions (Draft 2020-12 unknown key rejection)
    static bool validate_json_schema(const nlohmann::json& doc, std::vector<std::string>& errors) {
        if (!doc.is_object()) {
            errors.push_back("Configuration root must be a JSON object");
            return false;
        }

        // Top-level required keys
        static const std::set<std::string> kAllowedRootKeys{
            "schema_version", "runtime", "capture", "perception",
            "tracking", "policy", "trajectory", "actuator", "safety"
        };

        for (auto it = doc.begin(); it != doc.end(); ++it) {
            if (!kAllowedRootKeys.contains(it.key())) {
                errors.push_back("Unknown root configuration property: '" + it.key() + "'");
            }
        }

        for (const auto& req : kAllowedRootKeys) {
            if (!doc.contains(req)) {
                errors.push_back("Missing required root property: '" + req + "'");
            }
        }

        if (!errors.empty()) {
            return false;
        }

        // Validate subdomains and reject unknown properties
        auto check_keys = [&errors](const nlohmann::json& sub, const std::string& domain_name, const std::set<std::string>& allowed, const std::set<std::string>& required) {
            if (!sub.is_object()) {
                errors.push_back("Domain '" + domain_name + "' must be an object");
                return;
            }
            for (auto it = sub.begin(); it != sub.end(); ++it) {
                if (!allowed.contains(it.key())) {
                    errors.push_back("Unknown property in '" + domain_name + "': '" + it.key() + "'");
                }
            }
            for (const auto& req : required) {
                if (!sub.contains(req)) {
                    errors.push_back("Missing required property in '" + domain_name + "': '" + req + "'");
                }
            }
        };

        check_keys(doc["runtime"], "runtime",
            {"internal_deadline_ms", "target_p99_ms", "warmup_iterations", "allocation_audit", "thread_priority", "cpu_affinity_mask"},
            {"internal_deadline_ms", "target_p99_ms", "warmup_iterations", "allocation_audit"}
        );

        check_keys(doc["capture"], "capture",
            {"backend", "fallback", "source_width", "source_height", "pixel_format", "buffers", "latest_only", "stale_after_ms", "display_index", "allow_cross_adapter_copy"},
            {"backend", "fallback", "source_width", "source_height", "pixel_format", "buffers", "latest_only", "stale_after_ms"}
        );

        check_keys(doc["perception"], "perception",
            {"plugin", "precision", "batch", "input_width", "input_height", "max_targets", "cuda_graph", "confidence_floor", "nms_iou_threshold", "model_path", "model_sha256"},
            {"plugin", "precision", "batch", "input_width", "input_height", "max_targets", "cuda_graph", "confidence_floor"}
        );

        check_keys(doc["tracking"], "tracking",
            {"models", "association", "max_missed_frames", "immediate_confidence", "require_second_observation_below", "engage_if_uncertainty_within_radius", "process_noise_scale", "measurement_noise_scale", "gating_threshold_chi2"},
            {"models", "association", "max_missed_frames", "immediate_confidence", "require_second_observation_below", "engage_if_uncertainty_within_radius"}
        );

        check_keys(doc["policy"], "policy",
            {"plugin", "objective", "horizon_targets", "switch_hysteresis", "target_lead_time_ms"},
            {"plugin", "objective", "horizon_targets", "switch_hysteresis"}
        );

        check_keys(doc["trajectory"], "trajectory",
            {"small_error_mode", "large_error_mode", "terminal_mode", "optional_profile", "small_error_threshold_px", "max_velocity_counts_per_s", "max_acceleration_counts_per_s2", "max_jerk_counts_per_s3", "pd_kp", "pd_kd"},
            {"small_error_mode", "large_error_mode", "terminal_mode", "optional_profile"}
        );

        check_keys(doc["actuator"], "actuator",
            {"backend", "scheduler_hz", "relative_counts", "cancel_superseded", "counts_per_pixel_x", "counts_per_pixel_y", "hid_com_port", "hid_baud_rate"},
            {"backend", "scheduler_hz", "relative_counts", "cancel_superseded"}
        );

        check_keys(doc["safety"], "safety",
            {"require_foreground_match", "target_process_name", "target_window_title", "require_emergency_stop", "emergency_stop_key", "fail_closed", "max_delta_counts_per_dispatch", "max_active_engagement_seconds"},
            {"require_foreground_match", "require_emergency_stop", "fail_closed"}
        );

        // Type safety checks
        if (doc.contains("schema_version") && !doc["schema_version"].is_number_integer()) {
            errors.push_back("schema_version must be an integer");
        }
        if (doc.contains("tracking") && doc["tracking"].contains("models") && !doc["tracking"]["models"].is_array()) {
            errors.push_back("tracking.models must be an array");
        }
        if (doc.contains("capture") && doc["capture"].contains("source_width") && !doc["capture"]["source_width"].is_number_integer()) {
            errors.push_back("capture.source_width must be an integer");
        }

        return errors.empty();
    }

    /// @brief Convert nlohmann::json to ResolvedConfig struct
    static bool from_json(const nlohmann::json& j, ResolvedConfig& cfg, std::vector<std::string>& errors) {
        try {
            cfg.schema_version = j.value("schema_version", 1u);

            // Runtime
            const auto& rt = j["runtime"];
            cfg.runtime.internal_deadline_ms = rt.value("internal_deadline_ms", 10.0);
            cfg.runtime.target_p99_ms = rt.value("target_p99_ms", 6.0);
            cfg.runtime.warmup_iterations = rt.value("warmup_iterations", 200u);
            cfg.runtime.allocation_audit = rt.value("allocation_audit", true);
            cfg.runtime.thread_priority = rt.value("thread_priority", "high");
            cfg.runtime.cpu_affinity_mask = rt.value("cpu_affinity_mask", 0u);

            // Capture
            const auto& cap = j["capture"];
            cfg.capture.backend = cap.value("backend", "dxgi");
            cfg.capture.fallback = cap.value("fallback", "wgc");
            cfg.capture.source_width = cap.value("source_width", 1920u);
            cfg.capture.source_height = cap.value("source_height", 1080u);
            cfg.capture.pixel_format = cap.value("pixel_format", "bgra8_sdr");
            cfg.capture.buffers = cap.value("buffers", 2u);
            cfg.capture.latest_only = cap.value("latest_only", true);
            cfg.capture.stale_after_ms = cap.value("stale_after_ms", 12.0);
            cfg.capture.display_index = cap.value("display_index", 0u);
            cfg.capture.allow_cross_adapter_copy = cap.value("allow_cross_adapter_copy", false);

            // Perception
            const auto& perc = j["perception"];
            cfg.perception.plugin = perc.value("plugin", "yolo11n_aimlabs");
            cfg.perception.precision = perc.value("precision", "fp16");
            cfg.perception.batch = perc.value("batch", 1u);
            cfg.perception.input_width = perc.value("input_width", 640u);
            cfg.perception.input_height = perc.value("input_height", 384u);
            cfg.perception.max_targets = perc.value("max_targets", 64u);
            cfg.perception.cuda_graph = perc.value("cuda_graph", true);
            cfg.perception.confidence_floor = perc.value("confidence_floor", 0.20);
            cfg.perception.nms_iou_threshold = perc.value("nms_iou_threshold", 0.45);
            cfg.perception.model_path = perc.value("model_path", "");
            cfg.perception.model_sha256 = perc.value("model_sha256", "");

            // Tracking
            const auto& trk = j["tracking"];
            if (trk.contains("models") && trk["models"].is_array()) {
                cfg.tracking.models = trk["models"].get<std::vector<std::string>>();
            }
            cfg.tracking.association = trk.value("association", "hungarian");
            cfg.tracking.max_missed_frames = trk.value("max_missed_frames", 3u);
            cfg.tracking.immediate_confidence = trk.value("immediate_confidence", 0.85);
            cfg.tracking.require_second_observation_below = trk.value("require_second_observation_below", 0.85);
            cfg.tracking.engage_if_uncertainty_within_radius = trk.value("engage_if_uncertainty_within_radius", true);
            cfg.tracking.process_noise_scale = trk.value("process_noise_scale", 1.0);
            cfg.tracking.measurement_noise_scale = trk.value("measurement_noise_scale", 1.0);
            cfg.tracking.gating_threshold_chi2 = trk.value("gating_threshold_chi2", 9.21);

            // Policy
            const auto& pol = j["policy"];
            cfg.policy.plugin = pol.value("plugin", "deterministic_utility");
            cfg.policy.objective = pol.value("objective", "raw_score");
            cfg.policy.horizon_targets = pol.value("horizon_targets", 3u);
            cfg.policy.switch_hysteresis = pol.value("switch_hysteresis", 0.08);
            cfg.policy.target_lead_time_ms = pol.value("target_lead_time_ms", 0.0);

            // Trajectory
            const auto& traj = j["trajectory"];
            cfg.trajectory.small_error_mode = traj.value("small_error_mode", "direct_feedforward");
            cfg.trajectory.large_error_mode = traj.value("large_error_mode", "jerk_limited");
            cfg.trajectory.terminal_mode = traj.value("terminal_mode", "critically_damped_pd");
            cfg.trajectory.optional_profile = traj.value("optional_profile", "minimum_jerk");
            cfg.trajectory.small_error_threshold_px = traj.value("small_error_threshold_px", 15.0);
            cfg.trajectory.max_velocity_counts_per_s = traj.value("max_velocity_counts_per_s", 50000.0);
            cfg.trajectory.max_acceleration_counts_per_s2 = traj.value("max_acceleration_counts_per_s2", 500000.0);
            cfg.trajectory.max_jerk_counts_per_s3 = traj.value("max_jerk_counts_per_s3", 10000000.0);
            cfg.trajectory.pd_kp = traj.value("pd_kp", 1.0);
            cfg.trajectory.pd_kd = traj.value("pd_kd", 0.1);

            // Actuator
            const auto& act = j["actuator"];
            cfg.actuator.backend = act.value("backend", "sendinput");
            cfg.actuator.scheduler_hz = act.value("scheduler_hz", 1000u);
            cfg.actuator.relative_counts = act.value("relative_counts", true);
            cfg.actuator.cancel_superseded = act.value("cancel_superseded", true);
            cfg.actuator.counts_per_pixel_x = act.value("counts_per_pixel_x", 1.0);
            cfg.actuator.counts_per_pixel_y = act.value("counts_per_pixel_y", 1.0);
            cfg.actuator.hid_com_port = act.value("hid_com_port", "");
            cfg.actuator.hid_baud_rate = act.value("hid_baud_rate", 115200u);

            // Safety
            const auto& safe = j["safety"];
            cfg.safety.require_foreground_match = safe.value("require_foreground_match", true);
            cfg.safety.target_process_name = safe.value("target_process_name", "Aimlab_tb.exe");
            cfg.safety.target_window_title = safe.value("target_window_title", "Aimlabs");
            cfg.safety.require_emergency_stop = safe.value("require_emergency_stop", true);
            cfg.safety.emergency_stop_key = safe.value("emergency_stop_key", "F12");
            cfg.safety.fail_closed = safe.value("fail_closed", true);
            cfg.safety.max_delta_counts_per_dispatch = safe.value("max_delta_counts_per_dispatch", 500u);
            cfg.safety.max_active_engagement_seconds = safe.value("max_active_engagement_seconds", 60.0);

            return true;
        } catch (const std::exception& e) {
            errors.push_back(std::string("Deserialization exception: ") + e.what());
            return false;
        }
    }

    /// @brief Convert ResolvedConfig struct to canonical nlohmann::json
    static nlohmann::json to_json(const ResolvedConfig& cfg) {
        nlohmann::json j;
        j["schema_version"] = cfg.schema_version;

        j["runtime"] = {
            {"internal_deadline_ms", cfg.runtime.internal_deadline_ms},
            {"target_p99_ms", cfg.runtime.target_p99_ms},
            {"warmup_iterations", cfg.runtime.warmup_iterations},
            {"allocation_audit", cfg.runtime.allocation_audit},
            {"thread_priority", cfg.runtime.thread_priority},
            {"cpu_affinity_mask", cfg.runtime.cpu_affinity_mask}
        };

        j["capture"] = {
            {"backend", cfg.capture.backend},
            {"fallback", cfg.capture.fallback},
            {"source_width", cfg.capture.source_width},
            {"source_height", cfg.capture.source_height},
            {"pixel_format", cfg.capture.pixel_format},
            {"buffers", cfg.capture.buffers},
            {"latest_only", cfg.capture.latest_only},
            {"stale_after_ms", cfg.capture.stale_after_ms},
            {"display_index", cfg.capture.display_index},
            {"allow_cross_adapter_copy", cfg.capture.allow_cross_adapter_copy}
        };

        j["perception"] = {
            {"plugin", cfg.perception.plugin},
            {"precision", cfg.perception.precision},
            {"batch", cfg.perception.batch},
            {"input_width", cfg.perception.input_width},
            {"input_height", cfg.perception.input_height},
            {"max_targets", cfg.perception.max_targets},
            {"cuda_graph", cfg.perception.cuda_graph},
            {"confidence_floor", cfg.perception.confidence_floor},
            {"nms_iou_threshold", cfg.perception.nms_iou_threshold},
            {"model_path", cfg.perception.model_path},
            {"model_sha256", cfg.perception.model_sha256}
        };

        j["tracking"] = {
            {"models", cfg.tracking.models},
            {"association", cfg.tracking.association},
            {"max_missed_frames", cfg.tracking.max_missed_frames},
            {"immediate_confidence", cfg.tracking.immediate_confidence},
            {"require_second_observation_below", cfg.tracking.require_second_observation_below},
            {"engage_if_uncertainty_within_radius", cfg.tracking.engage_if_uncertainty_within_radius},
            {"process_noise_scale", cfg.tracking.process_noise_scale},
            {"measurement_noise_scale", cfg.tracking.measurement_noise_scale},
            {"gating_threshold_chi2", cfg.tracking.gating_threshold_chi2}
        };

        j["policy"] = {
            {"plugin", cfg.policy.plugin},
            {"objective", cfg.policy.objective},
            {"horizon_targets", cfg.policy.horizon_targets},
            {"switch_hysteresis", cfg.policy.switch_hysteresis},
            {"target_lead_time_ms", cfg.policy.target_lead_time_ms}
        };

        j["trajectory"] = {
            {"small_error_mode", cfg.trajectory.small_error_mode},
            {"large_error_mode", cfg.trajectory.large_error_mode},
            {"terminal_mode", cfg.trajectory.terminal_mode},
            {"optional_profile", cfg.trajectory.optional_profile},
            {"small_error_threshold_px", cfg.trajectory.small_error_threshold_px},
            {"max_velocity_counts_per_s", cfg.trajectory.max_velocity_counts_per_s},
            {"max_acceleration_counts_per_s2", cfg.trajectory.max_acceleration_counts_per_s2},
            {"max_jerk_counts_per_s3", cfg.trajectory.max_jerk_counts_per_s3},
            {"pd_kp", cfg.trajectory.pd_kp},
            {"pd_kd", cfg.trajectory.pd_kd}
        };

        j["actuator"] = {
            {"backend", cfg.actuator.backend},
            {"scheduler_hz", cfg.actuator.scheduler_hz},
            {"relative_counts", cfg.actuator.relative_counts},
            {"cancel_superseded", cfg.actuator.cancel_superseded},
            {"counts_per_pixel_x", cfg.actuator.counts_per_pixel_x},
            {"counts_per_pixel_y", cfg.actuator.counts_per_pixel_y},
            {"hid_com_port", cfg.actuator.hid_com_port},
            {"hid_baud_rate", cfg.actuator.hid_baud_rate}
        };

        j["safety"] = {
            {"require_foreground_match", cfg.safety.require_foreground_match},
            {"target_process_name", cfg.safety.target_process_name},
            {"target_window_title", cfg.safety.target_window_title},
            {"require_emergency_stop", cfg.safety.require_emergency_stop},
            {"emergency_stop_key", cfg.safety.emergency_stop_key},
            {"fail_closed", cfg.safety.fail_closed},
            {"max_delta_counts_per_dispatch", cfg.safety.max_delta_counts_per_dispatch},
            {"max_active_engagement_seconds", cfg.safety.max_active_engagement_seconds}
        };

        return j;
    }

    /// @brief Load and merge an ordered list of YAML file paths
    static LoadResult load_from_files(const std::vector<std::filesystem::path>& layer_paths) {
        LoadResult result{};
        nlohmann::json merged = nlohmann::json::object();

        for (const auto& p : layer_paths) {
            try {
                if (!std::filesystem::exists(p)) {
                    result.errors.push_back("Configuration file not found: " + p.string());
                    return result;
                }
                const YAML::Node yaml_doc = YAML::LoadFile(p.string());
                const nlohmann::json layer_json = yaml_to_json(yaml_doc);
                merge_json(merged, layer_json);
                // Store normalized POSIX path
                std::string posix_path = p.generic_string();
                result.source_layers.push_back(posix_path);
            } catch (const std::exception& e) {
                result.errors.push_back("Failed to parse layer '" + p.string() + "': " + e.what());
                return result;
            }
        }

        return process_merged_json(merged, result);
    }

    /// @brief Load and merge an ordered list of YAML strings
    static LoadResult load_from_yaml_strings(
        const std::vector<std::string>& yaml_contents,
        const std::vector<std::string>& layer_names = {})
    {
        LoadResult result{};
        nlohmann::json merged = nlohmann::json::object();

        for (std::size_t i = 0; i < yaml_contents.size(); ++i) {
            const std::string name = (i < layer_names.size()) ? layer_names[i] : ("layer_" + std::to_string(i));
            try {
                const YAML::Node yaml_doc = YAML::Load(yaml_contents[i]);
                const nlohmann::json layer_json = yaml_to_json(yaml_doc);
                merge_json(merged, layer_json);
                result.source_layers.push_back(name);
            } catch (const std::exception& e) {
                result.errors.push_back("Failed to parse YAML string for '" + name + "': " + e.what());
                return result;
            }
        }

        return process_merged_json(merged, result);
    }

private:
    static LoadResult process_merged_json(const nlohmann::json& merged, LoadResult result) {
        result.raw_json = merged;

        // 1. Strict schema validation
        if (!validate_json_schema(merged, result.errors)) {
            result.success = false;
            return result;
        }

        // 2. Struct deserialization
        if (!from_json(merged, result.config, result.errors)) {
            result.success = false;
            return result;
        }

        // 3. Range validation and parameter clamping
        if (!Validator::validate_and_clamp(result.config, result.warnings, result.errors)) {
            result.success = false;
            return result;
        }

        // 4. Canonical JSON SHA-256 computation
        const nlohmann::json canonical_json = to_json(result.config);
        const std::string canonical_str = canonical_json.dump();
        result.sha256_hash = Sha256::hash_string(canonical_str);

        result.success = true;
        return result;
    }
};

} // namespace aim::config
