// include/aim/scenario/scenario_profile_loader.hpp
#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "aim/config/config_loader.hpp"
#include "aim/config/sha256.hpp"
#include "aim/interfaces/scenario_adapter.hpp"
#include "aim/scenario/scenario_adapter.hpp"

namespace aim::scenario {

struct ScenarioProfileLoadResult {
    bool success{false};
    /// @brief Structural/range verdict from @ref ScenarioAdapter::validate().
    /// Parse and schema-shape errors report as @c ScenarioProfileStatus::ok with
    /// @c success == false; inspect @c errors for those.
    ScenarioProfileStatus status{ScenarioProfileStatus::ok};
    /// @brief The parsed profile on success; a default-constructed, capability-free
    /// profile on any failure, so a caller that ignores @c success still gets a
    /// fail-closed adapter.
    ScenarioProfile profile{};
    nlohmann::json raw_json{};
    std::string sha256_hash{};
    std::string source{};
    std::vector<std::string> errors{};
};

/// @brief Strict YAML -> @ref ScenarioProfile loader.
///
/// Mirrors @ref aim::config::ConfigLoader: unknown keys are rejected, required
/// keys are enforced, ranges are checked, and a canonical SHA-256 is emitted.
/// Any failure yields @c success == false and an empty (fail-closed) profile.
class ScenarioProfileLoader {
public:
    /// @brief The only scenario profile document version this loader accepts.
    /// Mirrors `schema_version` in schemas/config/scenario_profile.schema.json.
    static constexpr int kSupportedSchemaVersion = 1;

    static ScenarioProfileLoadResult load_from_yaml_string(const std::string& yaml_text,
                                                           const std::string& source_name = "<string>") {
        ScenarioProfileLoadResult result{};
        result.source = source_name;
        try {
            const YAML::Node doc = YAML::Load(yaml_text);
            const nlohmann::json j = aim::config::ConfigLoader::yaml_to_json(doc);
            return finalize(j, std::move(result));
        } catch (const std::exception& e) {
            result.errors.emplace_back(std::string("YAML parse error: ") + e.what());
            return result;
        }
    }

    static ScenarioProfileLoadResult load_from_file(const std::filesystem::path& path) {
        ScenarioProfileLoadResult result{};
        result.source = path.generic_string();
        if (!std::filesystem::exists(path)) {
            result.errors.emplace_back("Scenario profile file not found: " + path.generic_string());
            return result;
        }
        try {
            const YAML::Node doc = YAML::LoadFile(path.string());
            const nlohmann::json j = aim::config::ConfigLoader::yaml_to_json(doc);
            return finalize(j, std::move(result));
        } catch (const std::exception& e) {
            result.errors.emplace_back(std::string("Failed to parse '") + path.generic_string() + "': " + e.what());
            return result;
        }
    }

private:
    static void reject_unknown_keys(const nlohmann::json& obj,
                                    const std::string& scope,
                                    const std::set<std::string>& allowed,
                                    std::vector<std::string>& errors) {
        if (!obj.is_object()) {
            errors.push_back("Section '" + scope + "' must be a mapping");
            return;
        }
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            if (!allowed.contains(it.key())) {
                errors.push_back("Unknown key in '" + scope + "': '" + it.key() + "'");
            }
        }
    }

    static bool as_bool(const nlohmann::json& obj, const char* key, bool fallback,
                        const std::string& scope, std::vector<std::string>& errors) {
        if (!obj.contains(key)) {
            return fallback;
        }
        if (!obj.at(key).is_boolean()) {
            errors.push_back("Key '" + scope + "." + key + "' must be a boolean");
            return fallback;
        }
        return obj.at(key).get<bool>();
    }

    static double as_number(const nlohmann::json& obj, const char* key, double fallback,
                            const std::string& scope, std::vector<std::string>& errors) {
        if (!obj.contains(key)) {
            return fallback;
        }
        if (!obj.at(key).is_number()) {
            errors.push_back("Key '" + scope + "." + key + "' must be a number");
            return fallback;
        }
        return obj.at(key).get<double>();
    }

    static std::string as_string(const nlohmann::json& obj, const char* key, const std::string& fallback,
                                 const std::string& scope, std::vector<std::string>& errors) {
        if (!obj.contains(key)) {
            return fallback;
        }
        if (!obj.at(key).is_string()) {
            errors.push_back("Key '" + scope + "." + key + "' must be a string");
            return fallback;
        }
        return obj.at(key).get<std::string>();
    }

    static ScenarioProfileLoadResult finalize(const nlohmann::json& j, ScenarioProfileLoadResult result) {
        result.raw_json = j;

        if (!j.is_object()) {
            result.errors.emplace_back("Scenario profile root must be a mapping");
            return result;
        }

        static const std::set<std::string> kRootKeys{
            "schema_version", "profile_id", "default_target_value", "max_engagement_seconds",
            "capabilities", "foreground", "calibration_seed", "crosshair"};
        reject_unknown_keys(j, "<root>", kRootKeys, result.errors);

        // Versioned like every other standalone document family in this repository, so
        // a future migration can be detected rather than silently mis-parsed.
        if (!j.contains("schema_version")) {
            result.errors.emplace_back("Missing required key '<root>.schema_version'");
        } else if (!j.at("schema_version").is_number_integer()) {
            result.errors.emplace_back("Key '<root>.schema_version' must be an integer");
        } else if (j.at("schema_version").get<int>() != kSupportedSchemaVersion) {
            result.errors.emplace_back(
                "Unsupported '<root>.schema_version': expected " +
                std::to_string(kSupportedSchemaVersion) + ", got " +
                std::to_string(j.at("schema_version").get<int>()));
        }

        if (!j.contains("profile_id")) {
            result.errors.emplace_back("Missing required key '<root>.profile_id'");
        }

        ScenarioProfile profile{};
        profile.profile_id = as_string(j, "profile_id", "", "<root>", result.errors);
        profile.default_target_value =
            static_cast<float>(as_number(j, "default_target_value", 1.0, "<root>", result.errors));
        profile.max_engagement_seconds =
            static_cast<float>(as_number(j, "max_engagement_seconds", 0.0, "<root>", result.errors));

        if (j.contains("capabilities")) {
            const nlohmann::json& c = j.at("capabilities");
            static const std::set<std::string> kCapKeys{
                "provides_target_value", "provides_crosshair_context", "provides_calibration_seed",
                "provides_foreground_identity", "reports_scenario_completion"};
            reject_unknown_keys(c, "capabilities", kCapKeys, result.errors);
            profile.capabilities.provides_target_value =
                as_bool(c, "provides_target_value", false, "capabilities", result.errors);
            profile.capabilities.provides_crosshair_context =
                as_bool(c, "provides_crosshair_context", false, "capabilities", result.errors);
            profile.capabilities.provides_calibration_seed =
                as_bool(c, "provides_calibration_seed", false, "capabilities", result.errors);
            profile.capabilities.provides_foreground_identity =
                as_bool(c, "provides_foreground_identity", false, "capabilities", result.errors);
            profile.capabilities.reports_scenario_completion =
                as_bool(c, "reports_scenario_completion", false, "capabilities", result.errors);
        }

        if (j.contains("foreground")) {
            const nlohmann::json& f = j.at("foreground");
            static const std::set<std::string> kFgKeys{"process_name", "window_title_substring", "require_match"};
            reject_unknown_keys(f, "foreground", kFgKeys, result.errors);
            profile.foreground.process_name = as_string(f, "process_name", "", "foreground", result.errors);
            profile.foreground.window_title_substring =
                as_string(f, "window_title_substring", "", "foreground", result.errors);
            profile.foreground.require_match = as_bool(f, "require_match", true, "foreground", result.errors);
        }

        if (j.contains("calibration_seed")) {
            const nlohmann::json& s = j.at("calibration_seed");
            static const std::set<std::string> kSeedKeys{
                "counts_per_pixel_x", "counts_per_pixel_y", "fov_horizontal_deg", "in_game_sensitivity", "valid"};
            reject_unknown_keys(s, "calibration_seed", kSeedKeys, result.errors);
            profile.calibration_seed.counts_per_pixel_x =
                static_cast<float>(as_number(s, "counts_per_pixel_x", 0.0, "calibration_seed", result.errors));
            profile.calibration_seed.counts_per_pixel_y =
                static_cast<float>(as_number(s, "counts_per_pixel_y", 0.0, "calibration_seed", result.errors));
            profile.calibration_seed.fov_horizontal_deg =
                static_cast<float>(as_number(s, "fov_horizontal_deg", 0.0, "calibration_seed", result.errors));
            profile.calibration_seed.in_game_sensitivity =
                static_cast<float>(as_number(s, "in_game_sensitivity", 0.0, "calibration_seed", result.errors));
            profile.calibration_seed.valid = as_bool(s, "valid", false, "calibration_seed", result.errors);
        }

        if (j.contains("crosshair")) {
            const nlohmann::json& x = j.at("crosshair");
            static const std::set<std::string> kXhKeys{
                "center_norm_x", "center_norm_y", "center_px_x", "center_px_y", "valid"};
            reject_unknown_keys(x, "crosshair", kXhKeys, result.errors);
            profile.crosshair.center_norm.x =
                static_cast<float>(as_number(x, "center_norm_x", 0.0, "crosshair", result.errors));
            profile.crosshair.center_norm.y =
                static_cast<float>(as_number(x, "center_norm_y", 0.0, "crosshair", result.errors));
            profile.crosshair.center_px.x =
                static_cast<float>(as_number(x, "center_px_x", 960.0, "crosshair", result.errors));
            profile.crosshair.center_px.y =
                static_cast<float>(as_number(x, "center_px_y", 540.0, "crosshair", result.errors));
            profile.crosshair.valid = as_bool(x, "valid", true, "crosshair", result.errors);
        }

        if (!result.errors.empty()) {
            return result;
        }

        // Structural + range validation is centralized in ScenarioAdapter so that the
        // C++ contract and schemas/config/scenario_profile.schema.json cannot diverge.
        const ScenarioProfileStatus status = ScenarioAdapter::validate(profile);
        if (status != ScenarioProfileStatus::ok) {
            result.status = status;
            result.errors.emplace_back(
                std::string("Scenario profile failed structural/range validation: ") +
                std::string(aim::to_string(status)));
            return result;
        }

        result.profile = profile;
        result.sha256_hash = aim::config::Sha256::hash_string(canonical_json(profile).dump());
        result.success = true;
        return result;
    }

    /// @brief Deterministic canonical representation for hashing (key order fixed).
    static nlohmann::json canonical_json(const ScenarioProfile& p) {
        nlohmann::json j;
        j["profile_id"] = p.profile_id;
        j["default_target_value"] = p.default_target_value;
        j["max_engagement_seconds"] = p.max_engagement_seconds;
        j["capabilities"] = {
            {"provides_target_value", p.capabilities.provides_target_value},
            {"provides_crosshair_context", p.capabilities.provides_crosshair_context},
            {"provides_calibration_seed", p.capabilities.provides_calibration_seed},
            {"provides_foreground_identity", p.capabilities.provides_foreground_identity},
            {"reports_scenario_completion", p.capabilities.reports_scenario_completion},
        };
        j["foreground"] = {
            {"process_name", p.foreground.process_name},
            {"window_title_substring", p.foreground.window_title_substring},
            {"require_match", p.foreground.require_match},
        };
        j["calibration_seed"] = {
            {"counts_per_pixel_x", p.calibration_seed.counts_per_pixel_x},
            {"counts_per_pixel_y", p.calibration_seed.counts_per_pixel_y},
            {"fov_horizontal_deg", p.calibration_seed.fov_horizontal_deg},
            {"in_game_sensitivity", p.calibration_seed.in_game_sensitivity},
            {"valid", p.calibration_seed.valid},
        };
        j["crosshair"] = {
            {"center_norm_x", p.crosshair.center_norm.x},
            {"center_norm_y", p.crosshair.center_norm.y},
            {"center_px_x", p.crosshair.center_px.x},
            {"center_px_y", p.crosshair.center_px.y},
            {"valid", p.crosshair.valid},
        };
        return j;
    }
};

} // namespace aim::scenario
