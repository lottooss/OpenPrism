// include/aim/tracking/gated_hungarian.hpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/types.hpp"

namespace aim::tracking {

constexpr std::size_t kMaxAssociationDimension = 64;
constexpr float kGatedCostInfinity = 1e9f;

struct AssociationPair {
    std::uint32_t track_idx{0};
    std::uint32_t observation_idx{0};
    float cost{0.0f};
};

struct AssociationResult {
    std::uint32_t num_matches{0};
    std::array<AssociationPair, kMaxAssociationDimension> matches{};

    std::uint32_t num_unassigned_tracks{0};
    std::array<std::uint32_t, kMaxAssociationDimension> unassigned_tracks{};

    std::uint32_t num_unassigned_observations{0};
    std::array<std::uint32_t, kMaxAssociationDimension> unassigned_observations{};
};

struct AssociationConfig {
    float gate_distance_px{60.0f};
    float max_mahalanobis_sq{25.0f}; // ~5 sigma gate
    float max_radius_diff_ratio{0.50f};
    float position_weight{1.0f};
    float radius_weight{0.2f};
};

struct TrackHypothesis {
    float x{0.0f};
    float y{0.0f};
    float radius{0.0f};
    Covariance2D position_covariance{1.0f, 0.0f, 1.0f};
    std::uint32_t class_id{0};
};

/// @brief Gated Hungarian algorithm for multi-target tracking data association.
/// Allocation-free fixed matrix solver.
class GatedHungarianAssociator {
public:
    explicit GatedHungarianAssociator(AssociationConfig config = {});

    /// @brief Solve optimal assignment between N tracks and M observations with impossible match pre-gating.
    void associate(
        const TrackHypothesis* tracks,
        std::size_t num_tracks,
        const bus::TargetObservation* observations,
        std::size_t num_observations,
        AssociationResult& out_result) noexcept;

    /// @brief Compute pairwise gating and distance cost. Returns kGatedCostInfinity if gated out.
    [[nodiscard]] float compute_cost(
        const TrackHypothesis& track,
        const bus::TargetObservation& obs) const noexcept;

private:
    AssociationConfig config_;

    // Fixed pre-allocated Hungarian cost matrix & assignment buffers (zero allocations)
    std::array<std::array<float, kMaxAssociationDimension>, kMaxAssociationDimension> cost_matrix_{};
    std::array<int, kMaxAssociationDimension> track_to_obs_{};
    std::array<int, kMaxAssociationDimension> obs_to_track_{};
    std::array<float, kMaxAssociationDimension> u_{};
    std::array<float, kMaxAssociationDimension> v_{};
    std::array<int, kMaxAssociationDimension> p_{};
    std::array<float, kMaxAssociationDimension> minv_{};
    std::array<bool, kMaxAssociationDimension> used_{};
};

} // namespace aim::tracking
