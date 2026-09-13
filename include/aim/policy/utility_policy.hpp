// include/aim/policy/utility_policy.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/interfaces/aim_policy.hpp"

namespace aim::policy {

struct UtilityWeights {
    float weight_value{1.0f};
    float weight_confidence{1.5f};
    float weight_distance{0.005f};     // Penalty per pixel of distance
    float weight_uncertainty{0.10f};    // Penalty per px of covariance trace
    float switch_hysteresis_bonus{1.5f};// Bonus for maintaining currently engaged target
    float min_engagement_confidence{0.50f};
    float fire_authorization_threshold_px{8.0f};
    float screen_width_px{1920.0f};
    float screen_height_px{1080.0f};
};

/// @brief Production domain-neutral raw-score utility aim policy.
/// Zero-allocation deterministic target selection.
class UtilityAimPolicy : public IAimPolicy {
public:
    explicit UtilityAimPolicy(UtilityWeights weights = {})
        : weights_(weights) {}

    [[nodiscard]] PolicyContract contract() const noexcept override {
        return PolicyContract{
            "utility_aim_policy",
            "1.0.0",
            103.0f,
            720.0f
        };
    }

    bool initialize(const PolicyManifest& manifest) noexcept override {
        weights_.weight_distance = manifest.reward_weight_distance * 0.005f;
        weights_.weight_confidence = manifest.reward_weight_confidence;
        weights_.switch_hysteresis_bonus = manifest.switch_hysteresis_penalty;
        weights_.fire_authorization_threshold_px = manifest.fire_authorization_threshold_px;
        reset();
        return true;
    }

    void reset() noexcept override {
        current_target_id_ = 0;
        last_decision_time_ns_ = 0;
    }

    bool choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept override;

    [[nodiscard]] TrackId current_target_id() const noexcept { return current_target_id_; }
    [[nodiscard]] const UtilityWeights& weights() const noexcept { return weights_; }
    void set_weights(const UtilityWeights& weights) noexcept { weights_ = weights; }

private:
    UtilityWeights weights_;
    TrackId current_target_id_{0};
    MonotonicNs last_decision_time_ns_{0};
};

/// @brief Baseline Policy 1: Greedy Nearest Target (no hysteresis, no value weighting).
class NearestTargetPolicy : public IAimPolicy {
public:
    [[nodiscard]] PolicyContract contract() const noexcept override {
        return PolicyContract{"nearest_target_baseline", "1.0.0", 103.0f, 720.0f};
    }
    bool initialize(const PolicyManifest&) noexcept override { return true; }
    void reset() noexcept override {}
    bool choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept override;
};

/// @brief Baseline Policy 2: Left-to-Right Spatial Order Policy.
class SpatialSweepPolicy : public IAimPolicy {
public:
    [[nodiscard]] PolicyContract contract() const noexcept override {
        return PolicyContract{"spatial_sweep_baseline", "1.0.0", 103.0f, 720.0f};
    }
    bool initialize(const PolicyManifest&) noexcept override { return true; }
    void reset() noexcept override {}
    bool choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept override;
};

} // namespace aim::policy
