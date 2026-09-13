// include/aim/sim/scenario.hpp
// Simulation scenario container and definition
#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "aim/sim/types.hpp"

namespace aim::sim {

/// @brief Fully specified simulation scenario containing targets, occluders, and noise models.
struct Scenario {
    ScenarioConfig config{};
    std::vector<TargetSpec> targets{};
    std::vector<TemporalOcclusion> temporal_occlusions{};
    std::vector<RectOccluder> rect_occluders{};
    std::vector<CircleOccluder> circle_occluders{};

    [[nodiscard]] std::string_view name() const noexcept { return config.name; }
    [[nodiscard]] std::uint64_t seed() const noexcept { return config.seed; }
    [[nodiscard]] double cadence_hz() const noexcept { return config.cadence_hz; }
    [[nodiscard]] MonotonicNs start_time_ns() const noexcept { return config.start_time_ns; }
    [[nodiscard]] MonotonicNs duration_ns() const noexcept { return config.duration_ns; }
    [[nodiscard]] std::uint32_t screen_width() const noexcept { return config.source_width; }
    [[nodiscard]] std::uint32_t screen_height() const noexcept { return config.source_height; }
    [[nodiscard]] std::size_t target_count() const noexcept { return targets.size(); }
};

} // namespace aim::sim
