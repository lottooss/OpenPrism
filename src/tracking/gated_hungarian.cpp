// src/tracking/gated_hungarian.cpp
#include "aim/tracking/gated_hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aim::tracking {

GatedHungarianAssociator::GatedHungarianAssociator(AssociationConfig config)
    : config_(config) {}

float GatedHungarianAssociator::compute_cost(
    const TrackHypothesis& track,
    const bus::TargetObservation& obs) const noexcept {
    // 1. Semantic class ID compatibility gate
    if (track.class_id != 0 && obs.semantic_id != 0 && track.class_id != obs.semantic_id) {
        return kGatedCostInfinity;
    }

    // 2. Spatial Euclidean distance gate
    const float dx = track.x - obs.center_px.x;
    const float dy = track.y - obs.center_px.y;
    const float dist_sq = dx * dx + dy * dy;
    const float dist = std::sqrt(dist_sq);

    if (dist > config_.gate_distance_px) {
        return kGatedCostInfinity;
    }

    // 3. Radius difference ratio gate
    const float r_track = track.radius;
    const float r_obs = obs.effective_radius_px;
    float radius_penalty = 0.0f;
    if (r_track > 0.0f && r_obs > 0.0f) {
        const float max_r = std::max(r_track, r_obs);
        const float r_diff = std::abs(r_track - r_obs);
        if (r_diff / max_r > config_.max_radius_diff_ratio) {
            return kGatedCostInfinity;
        }
        radius_penalty = r_diff;
    }

    // 4. Mahalanobis distance covariance gating
    const float s_xx = std::max(1e-4f, track.position_covariance.xx + obs.covariance_px2.xx);
    const float s_xy = track.position_covariance.xy + obs.covariance_px2.xy;
    const float s_yy = std::max(1e-4f, track.position_covariance.yy + obs.covariance_px2.yy);

    const float det = s_xx * s_yy - s_xy * s_xy;
    if (det > 1e-6f) {
        const float inv_det = 1.0f / det;
        const float inv_s_xx = s_yy * inv_det;
        const float inv_s_xy = -s_xy * inv_det;
        const float inv_s_yy = s_xx * inv_det;

        const float d_m_sq = dx * (inv_s_xx * dx + inv_s_xy * dy) + dy * (inv_s_xy * dx + inv_s_yy * dy);
        if (d_m_sq > config_.max_mahalanobis_sq) {
            return kGatedCostInfinity;
        }
    }

    // 5. Linear weighted combination of spatial and size cost
    const float total_cost = config_.position_weight * dist + config_.radius_weight * radius_penalty;
    return std::max(0.0f, total_cost);
}

void GatedHungarianAssociator::associate(
    const TrackHypothesis* tracks,
    std::size_t num_tracks,
    const bus::TargetObservation* observations,
    std::size_t num_observations,
    AssociationResult& out_result) noexcept {
    out_result.num_matches = 0;
    out_result.num_unassigned_tracks = 0;
    out_result.num_unassigned_observations = 0;

    const std::size_t n = std::min(num_tracks, kMaxAssociationDimension);
    const std::size_t m = std::min(num_observations, kMaxAssociationDimension);

    if (n == 0) {
        for (std::size_t j = 0; j < m; ++j) {
            out_result.unassigned_observations[out_result.num_unassigned_observations++] =
                static_cast<std::uint32_t>(j);
        }
        return;
    }

    if (m == 0) {
        for (std::size_t i = 0; i < n; ++i) {
            out_result.unassigned_tracks[out_result.num_unassigned_tracks++] =
                static_cast<std::uint32_t>(i);
        }
        return;
    }

    const std::size_t dim = std::max(n, m);

    // 1. Build square cost matrix with gating values
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            if (i < n && j < m) {
                cost_matrix_[i][j] = compute_cost(tracks[i], observations[j]);
            } else {
                cost_matrix_[i][j] = kGatedCostInfinity;
            }
        }
    }

    // 2. Exact Hungarian Algorithm (O(dim^3) Kuhn-Munkres minimum cost bipartite matching)
    std::array<float, kMaxAssociationDimension + 1> u{};
    std::array<float, kMaxAssociationDimension + 1> v{};
    std::array<int, kMaxAssociationDimension + 1> p{};
    std::array<int, kMaxAssociationDimension + 1> way{};
    std::array<float, kMaxAssociationDimension + 1> minv{};
    std::array<bool, kMaxAssociationDimension + 1> used{};

    for (std::size_t i = 1; i <= dim; ++i) {
        p[0] = static_cast<int>(i);
        std::size_t j0 = 0;
        minv.fill(std::numeric_limits<float>::max());
        used.fill(false);
        do {
            used[j0] = true;
            const std::size_t i0 = static_cast<std::size_t>(p[j0]);
            float delta = std::numeric_limits<float>::max();
            std::size_t j1 = 0;
            for (std::size_t j = 1; j <= dim; ++j) {
                if (!used[j]) {
                    const float cur = cost_matrix_[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = static_cast<int>(j0);
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (std::size_t j = 0; j <= dim; ++j) {
                if (used[j]) {
                    u[static_cast<std::size_t>(p[j])] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const std::size_t j1 = static_cast<std::size_t>(way[j0]);
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // 3. Extract assignments and filter out gated pairs
    std::array<bool, kMaxAssociationDimension> track_matched{};
    std::array<bool, kMaxAssociationDimension> obs_matched{};

    for (std::size_t j = 1; j <= dim; ++j) {
        const int assigned_row_1based = p[j];
        if (assigned_row_1based > 0) {
            const std::size_t r = static_cast<std::size_t>(assigned_row_1based - 1);
            const std::size_t c = j - 1;
            if (r < n && c < m) {
                const float c_val = cost_matrix_[r][c];
                if (c_val < kGatedCostInfinity - 1e4f) {
                    AssociationPair pair{};
                    pair.track_idx = static_cast<std::uint32_t>(r);
                    pair.observation_idx = static_cast<std::uint32_t>(c);
                    pair.cost = c_val;
                    out_result.matches[out_result.num_matches++] = pair;
                    track_matched[r] = true;
                    obs_matched[c] = true;
                }
            }
        }
    }

    // 4. Collect unassigned tracks
    for (std::size_t i = 0; i < n; ++i) {
        if (!track_matched[i]) {
            out_result.unassigned_tracks[out_result.num_unassigned_tracks++] =
                static_cast<std::uint32_t>(i);
        }
    }

    // 5. Collect unassigned observations
    for (std::size_t j = 0; j < m; ++j) {
        if (!obs_matched[j]) {
            out_result.unassigned_observations[out_result.num_unassigned_observations++] =
                static_cast<std::uint32_t>(j);
        }
    }
}

} // namespace aim::tracking
