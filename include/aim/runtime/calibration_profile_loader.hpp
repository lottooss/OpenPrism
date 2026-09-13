#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

#include "aim/calibration/calibration_model.hpp"

namespace aim::runtime {

inline bool
calibration_matches_session(const calibration::CalibrationProfile &profile,
                            float fov, float sensitivity) noexcept {
    // Same held-out fit tolerance used by CalibrationWizard. Manifest range
    // validity alone is insufficient: rejected fits can still be serialized.
    return calibration::CalibrationModel(profile).is_valid() &&
           profile.rmse_pixels <= 5.0f && profile.resolution_width == 1920 &&
           profile.resolution_height == 1080 &&
           profile.fov_horizontal_deg == fov &&
           profile.in_game_sensitivity == sensitivity;
}

// Cold-path loader for the existing versioned calibration manifest. A profile
// is never inferred from a scenario seed or silently repaired with defaults.
inline calibration::CalibrationProfile
load_calibration_profile(const std::filesystem::path &path) {
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) > 16'384U) {
        throw std::runtime_error(
            "Calibration profile missing or exceeds 16 KiB");
    }
    std::ifstream input(path);
    const auto data = nlohmann::json::parse(input);
    constexpr std::array<std::string_view, 13> keys{
        "schema_version",      "profile_id",         "calibrated_at",
        "resolution_width",    "resolution_height",  "fov_horizontal_deg",
        "in_game_sensitivity", "counts_per_pixel_x", "counts_per_pixel_y",
        "deadband_counts",     "nonlinearity_alpha", "cross_coupling_xy",
        "rmse_pixels"};
    if (!data.is_object() || data.size() != keys.size()) {
        throw std::runtime_error(
            "Calibration profile must contain exactly the version 1 fields");
    }
    for (const auto key : keys) {
        if (!data.contains(key))
            throw std::runtime_error("Missing calibration field");
    }
    if (!data.at("schema_version").is_number_integer() ||
        data.at("schema_version") != 1 ||
        !data.at("resolution_width").is_number_unsigned() ||
        !data.at("resolution_height").is_number_unsigned() ||
        !data.at("calibrated_at").is_string() ||
        data.at("calibrated_at").get<std::string>().empty()) {
        throw std::runtime_error(
            "Invalid calibration version, dimensions, or measurement date");
    }
    if (data.at("resolution_width").get<double>() < 320.0 ||
        data.at("resolution_width").get<double>() > 7680.0 ||
        data.at("resolution_height").get<double>() < 240.0 ||
        data.at("resolution_height").get<double>() > 4320.0) {
        throw std::runtime_error("Calibration dimensions out of bounds");
    }
    calibration::CalibrationProfile profile{};
    constexpr std::array<std::string_view, 8> numeric_keys{
        "fov_horizontal_deg", "in_game_sensitivity", "counts_per_pixel_x",
        "counts_per_pixel_y", "deadband_counts",     "nonlinearity_alpha",
        "cross_coupling_xy",  "rmse_pixels"};
    constexpr std::array<double, 8> minimum{30.0, 0.001, 0.001, 0.001,
                                            0.0,  -1.0,  -0.5,  0.0};
    constexpr std::array<double, 8> maximum{150.0, 100.0, 100.0, 100.0,
                                            10.0,  1.0,   0.5,   50.0};
    for (std::size_t i = 0; i < numeric_keys.size(); ++i) {
        const auto &value = data.at(numeric_keys[i]);
        if (!value.is_number() || !std::isfinite(value.get<double>()) ||
            value.get<double>() < minimum[i] ||
            value.get<double>() > maximum[i])
            throw std::runtime_error(
                "Calibration numeric fields must be finite JSON numbers");
    }
    profile.profile_id = data.at("profile_id").get<std::string>();
    if (profile.profile_id.size() < 3 || profile.profile_id.size() > 64 ||
        !std::all_of(
            profile.profile_id.begin(), profile.profile_id.end(), [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '-';
            })) {
        throw std::runtime_error("Invalid calibration profile ID");
    }
    profile.resolution_width = data.at("resolution_width").get<std::uint32_t>();
    profile.resolution_height =
        data.at("resolution_height").get<std::uint32_t>();
    profile.fov_horizontal_deg = data.at("fov_horizontal_deg").get<float>();
    profile.in_game_sensitivity = data.at("in_game_sensitivity").get<float>();
    profile.counts_per_pixel_x = data.at("counts_per_pixel_x").get<float>();
    profile.counts_per_pixel_y = data.at("counts_per_pixel_y").get<float>();
    profile.deadband_counts = data.at("deadband_counts").get<float>();
    profile.nonlinearity_alpha = data.at("nonlinearity_alpha").get<float>();
    profile.cross_coupling_xy = data.at("cross_coupling_xy").get<float>();
    profile.rmse_pixels = data.at("rmse_pixels").get<float>();
    if (!calibration::CalibrationModel(profile).is_valid()) {
        throw std::runtime_error("Calibration values fail numeric validation");
    }
    return profile;
}

} // namespace aim::runtime
