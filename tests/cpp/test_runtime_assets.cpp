#include <filesystem>
#include <fstream>
#include <iostream>

#include "aim/runtime/calibration_evidence.hpp"
#include "aim/runtime/calibration_profile_loader.hpp"

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      "aim-runtime-calibration-test.json";
    const nlohmann::json valid{{"schema_version", 1},
                               {"profile_id", "measured_test"},
                               {"calibrated_at", "2026-09-12T00:00:00Z"},
                               {"resolution_width", 1920},
                               {"resolution_height", 1080},
                               {"fov_horizontal_deg", 103.0},
                               {"in_game_sensitivity", 1.0},
                               {"counts_per_pixel_x", 1.25},
                               {"counts_per_pixel_y", 1.5},
                               {"deadband_counts", 0.0},
                               {"nonlinearity_alpha", 0.0},
                               {"cross_coupling_xy", 0.0},
                               {"rmse_pixels", 0.5}};
    const auto accepts = [&](const nlohmann::json &data) {
        {
            std::ofstream output(path);
            output << data.dump();
        }
        try {
            const auto profile = aim::runtime::load_calibration_profile(path);
            return profile.counts_per_pixel_x == 1.25f;
        } catch (const std::exception &) {
            return false;
        }
    };
    bool passed = accepts(valid);
    auto profile = aim::runtime::load_calibration_profile(path);
    passed = passed &&
             aim::runtime::calibration_matches_session(profile, 103.0f, 1.0f);
    passed = passed &&
             !aim::runtime::calibration_matches_session(profile, 90.0f, 1.0f);
    profile.rmse_pixels = 49.0f;
    passed = passed &&
             !aim::runtime::calibration_matches_session(profile, 103.0f, 1.0f);
    auto bad = valid;
    bad["extra"] = true;
    passed = passed && !accepts(bad);
    bad = valid;
    bad.erase("counts_per_pixel_y");
    passed = passed && !accepts(bad);
    bad = valid;
    bad["schema_version"] = 1.5;
    passed = passed && !accepts(bad);
    bad = valid;
    bad["resolution_width"] = 4294969216ULL; // must not wrap to 1920
    passed = passed && !accepts(bad);
    bad = valid;
    bad["counts_per_pixel_x"] = nullptr;
    passed = passed && !accepts(bad);
    bad = valid;
    bad["counts_per_pixel_x"] = true;
    passed = passed && !accepts(bad);
    bad = valid;
    bad["rmse_pixels"] = false;
    passed = passed && !accepts(bad);
    bad = valid;
    bad["counts_per_pixel_x"] = 1e300;
    passed = passed && !accepts(bad);
    bad = valid;
    bad["calibrated_at"] = "";
    passed = passed && !accepts(bad);
    // Physical startup must bind timing/range evidence to the exact profile
    // and engine, even if each file separately contains valid numeric values.
    passed = passed && accepts(valid);
    auto sidecar = path;
    sidecar += ".evidence.json";
    const nlohmann::json evidence{{"schema_version", 1},
                                  {"profile", valid},
                                  {"engine_sha256", "abc"},
                                  {"training_samples", 16U},
                                  {"held_out_samples", 8U},
                                  {"maximum_step_counts", 64},
                                  {"effect_lower_p50_ns", 1'000'000},
                                  {"effect_upper_p50_ns", 4'000'000},
                                  {"effect_upper_p95_ns", 8'000'000},
                                  {"largest_effect_bracket_ns", 6'000'000}};
    const auto accepts_evidence = [&](const nlohmann::json &data) {
        {
            std::ofstream out(sidecar);
            out << data.dump();
        }
        try {
            const auto loaded =
                aim::runtime::load_calibration_evidence(path, "abc");
            return loaded.maximum_step_counts == 64 &&
                   loaded.effect_upper_p50_ns == 4'000'000;
        } catch (const std::exception &) {
            return false;
        }
    };
    passed = passed && accepts_evidence(evidence);
    for (const auto *field :
         {"engine_sha256", "maximum_step_counts", "effect_upper_p50_ns",
          "profile", "held_out_samples"}) {
        bad = evidence;
        bad[field] = nullptr;
        passed = passed && !accepts_evidence(bad);
    }
    bad = evidence;
    bad["profile"]["counts_per_pixel_x"] = 2.0;
    passed = passed && !accepts_evidence(bad);
    bad = evidence;
    bad["maximum_step_counts"] = 4294967360ULL;
    passed = passed && !accepts_evidence(bad);
    bad = evidence;
    bad["held_out_samples"] = 7U;
    passed = passed && !accepts_evidence(bad);
    bad = evidence;
    bad["effect_upper_p50_ns"] = -4'000'000;
    passed = passed && !accepts_evidence(bad);
    bad = evidence;
    bad["effect_lower_p50_ns"] = 9'000'000;
    passed = passed && !accepts_evidence(bad);
    bad = evidence;
    bad["extra"] = 1;
    passed = passed && !accepts_evidence(bad);
    std::filesystem::remove(sidecar);
    std::filesystem::remove(path);
    try {
        static_cast<void>(aim::runtime::load_calibration_profile(path));
        passed = false;
    } catch (const std::exception &) {
    }
    if (!passed) {
        std::cerr << "Runtime calibration preflight regression failed\n";
        return 1;
    }
    return 0;
}
