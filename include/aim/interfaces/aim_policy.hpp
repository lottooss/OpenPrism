// include/aim/interfaces/aim_policy.hpp
#pragma once

#include <cstdint>
#include <string>
#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim {

struct PolicyContract {
    std::string policy_name{"deterministic_utility"};
    std::string version{"1.0.0"};
    float fov_horizontal_deg{103.0f};
    float max_angular_speed_deg_s{720.0f};
};

struct PolicyManifest {
    std::string config_layer{};
    float reward_weight_distance{1.0f};
    float reward_weight_confidence{2.0f};
    float switch_hysteresis_penalty{1.5f};
    float fire_authorization_threshold_px{8.0f};
};

struct CrosshairState {
    PixelPoint center_px{960.0f, 540.0f};
    NormalizedPoint center_norm{0.0f, 0.0f};
    bool is_recoil_active{false};
};

struct PolicyInput {
    CorrelationId correlation_id{};
    MonotonicNs decision_time_ns{0};
    CrosshairState crosshair{};
    // Borrowed for choose() only; policy implementations must not retain it.
    const bus::TrackedTargetBatch* tracks{nullptr};
};

/// @brief Primary aim policy abstraction choosing target and computing aim intent.
class IAimPolicy {
public:
    virtual ~IAimPolicy() = default;
    [[nodiscard]] virtual PolicyContract contract() const noexcept = 0;
    virtual bool initialize(const PolicyManifest& manifest) noexcept = 0;
    virtual bool choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept = 0;
    virtual void reset() noexcept = 0;
};

} // namespace aim
