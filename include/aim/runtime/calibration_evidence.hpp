#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "aim/runtime/visual_calibration.hpp"

namespace aim::runtime {

struct CalibrationRuntimeEvidence {
    std::int32_t maximum_step_counts{0};
    MonotonicNs effect_upper_p50_ns{0};
};

// Private delivery sidecar: bind pulse range and observed response timing to
// the exact measured profile and detector. The public v1 profile stays intact.
inline CalibrationRuntimeEvidence
load_calibration_evidence(const std::filesystem::path &profile_path,
                          const std::string &engine_hash) {
    auto path = profile_path;
    path += ".evidence.json";
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) > 32768U)
        throw std::runtime_error("Measured calibration evidence is missing");
    std::ifstream profile_input(profile_path), evidence_input(path);
    const auto profile = nlohmann::json::parse(profile_input);
    const auto evidence = nlohmann::json::parse(evidence_input);
    if (!evidence.is_object() || evidence.size() != 10 ||
        !evidence.at("schema_version").is_number_integer() ||
        evidence.at("schema_version") != 1 ||
        evidence.at("profile") != profile ||
        evidence.at("engine_sha256") != engine_hash ||
        !evidence.at("training_samples").is_number_unsigned() ||
        !evidence.at("held_out_samples").is_number_unsigned() ||
        evidence.at("training_samples").get<std::uint64_t>() < 12 ||
        evidence.at("held_out_samples").get<std::uint64_t>() < 8 ||
        !evidence.at("maximum_step_counts").is_number_integer() ||
        !evidence.at("effect_upper_p50_ns").is_number_integer() ||
        !evidence.at("effect_upper_p95_ns").is_number_integer() ||
        !evidence.at("effect_lower_p50_ns").is_number_integer() ||
        !evidence.at("largest_effect_bracket_ns").is_number_integer())
        throw std::runtime_error(
            "Calibration evidence does not match the profile/model");
    const auto count = evidence.at("maximum_step_counts").get<double>();
    const auto lower = evidence.at("effect_lower_p50_ns").get<double>();
    const auto upper = evidence.at("effect_upper_p50_ns").get<double>();
    const auto p95 = evidence.at("effect_upper_p95_ns").get<double>();
    const auto bracket = evidence.at("largest_effect_bracket_ns").get<double>();
    if (count < 1 || count > 64 || lower < 0 || upper < lower || upper <= 0 ||
        p95 < upper || p95 > 100'000'000 || bracket <= 0 ||
        bracket > 100'000'000)
        throw std::runtime_error(
            "Measured calibration timing/range is invalid");
    return {static_cast<std::int32_t>(count), static_cast<MonotonicNs>(upper)};
}

inline void save_calibration_measurement(const std::filesystem::path &path,
                                         const VisualCalibrationFit &fit,
                                         const std::string &engine_hash,
                                         const std::string &measured_at) {
    auto sidecar = path;
    sidecar += ".evidence.json";
    if (!fit.succeeded() || std::filesystem::exists(path) ||
        std::filesystem::exists(sidecar))
        throw std::runtime_error(
            "Calibration failed or output profile already exists");
    const auto &p = fit.profile;
    const nlohmann::json profile{{"schema_version", 1},
                                 {"profile_id", p.profile_id},
                                 {"calibrated_at", measured_at},
                                 {"resolution_width", p.resolution_width},
                                 {"resolution_height", p.resolution_height},
                                 {"fov_horizontal_deg", p.fov_horizontal_deg},
                                 {"in_game_sensitivity", p.in_game_sensitivity},
                                 {"counts_per_pixel_x", p.counts_per_pixel_x},
                                 {"counts_per_pixel_y", p.counts_per_pixel_y},
                                 {"deadband_counts", p.deadband_counts},
                                 {"nonlinearity_alpha", p.nonlinearity_alpha},
                                 {"cross_coupling_xy", p.cross_coupling_xy},
                                 {"rmse_pixels", p.rmse_pixels}};
    const nlohmann::json evidence{
        {"schema_version", 1},
        {"profile", profile},
        {"engine_sha256", engine_hash},
        {"training_samples", fit.training_samples},
        {"held_out_samples", fit.held_out_samples},
        {"maximum_step_counts", fit.measured_max_pulse_counts},
        {"effect_lower_p50_ns", fit.effect_lower_p50_ns},
        {"effect_upper_p50_ns", fit.effect_upper_p50_ns},
        {"effect_upper_p95_ns", fit.effect_upper_p95_ns},
        {"largest_effect_bracket_ns", fit.largest_effect_bracket_ns}};
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path), evidence_output(sidecar);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    evidence_output.exceptions(std::ios::failbit | std::ios::badbit);
    output << profile.dump(2) << '\n';
    evidence_output << evidence.dump(2) << '\n';
    output.close();
    evidence_output.close();
}

} // namespace aim::runtime
