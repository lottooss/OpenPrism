// include/aim/sim/target_simulator.hpp
// Zero-allocation deterministic synthetic target observation generator and bus publisher
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/core/types.hpp"
#include "aim/sim/prng.hpp"
#include "aim/sim/replay_clock.hpp"
#include "aim/sim/scenario.hpp"
#include "aim/sim/trajectory.hpp"
#include "aim/sim/types.hpp"

namespace aim::sim {

/// @brief Primary synthetic target observation generator and bus publisher.
/// Executes trajectory kinematics, spatial/temporal occlusions, and Gaussian measurement noise
/// directly onto preallocated buffers with zero heap allocations on the hot path.
class TargetSimulator {
public:
    explicit TargetSimulator(const Scenario& scenario) noexcept
        : scenario_(scenario),
          rng_(scenario.seed()),
          replay_clock_(scenario.cadence_hz(), scenario.start_time_ns()) {}

    [[nodiscard]] const Scenario& scenario() const noexcept { return scenario_; }
    [[nodiscard]] ReplayClock& clock() noexcept { return replay_clock_; }
    [[nodiscard]] const ReplayClock& clock() const noexcept { return replay_clock_; }
    [[nodiscard]] DeterministicRng& rng() noexcept { return rng_; }

    [[nodiscard]] std::uint64_t total_generated_batches() const noexcept { return total_generated_batches_; }
    [[nodiscard]] std::uint64_t total_generated_observations() const noexcept { return total_generated_observations_; }
    [[nodiscard]] std::uint64_t total_dropped_occlusions() const noexcept { return total_dropped_occlusions_; }

    /// @brief Evaluates all active trajectories and generates observation batch for timestamp_ns.
    std::size_t generate_batch(MonotonicNs timestamp_ns, bus::TargetObservationBatch& out_batch) noexcept {
        out_batch.clear();
        out_batch.schema_major = 1;
        out_batch.schema_minor = 0;
        out_batch.header.sequence_id = replay_clock_.sequence_id() + 1;
        out_batch.header.source_timestamp_ns = timestamp_ns;
        out_batch.header.pipeline_run_id = scenario_.config.pipeline_run_id;
        out_batch.header.flags = static_cast<std::uint32_t>(CorrelationFlags::synthetic);
        out_batch.source_id = scenario_.config.source_id;
        out_batch.frame_id = replay_clock_.frame_id();
        out_batch.captured_at_ns = timestamp_ns;
        out_batch.published_at_ns = timestamp_ns + scenario_.config.synthetic_perception_latency_ns;
        out_batch.source_width = scenario_.config.source_width;
        out_batch.source_height = scenario_.config.source_height;

        const double elapsed_sec = (timestamp_ns >= scenario_.config.start_time_ns)
            ? static_cast<double>(timestamp_ns - scenario_.config.start_time_ns) / 1'000'000'000.0
            : 0.0;

        for (const auto& target_spec : scenario_.targets) {
            // Check lifecycle window
            if (timestamp_ns < target_spec.spawn_time_ns || timestamp_ns >= target_spec.despawn_time_ns) {
                continue;
            }

            // Kinematic evaluation
            const TrajectoryState state = evaluate_trajectory(
                target_spec,
                elapsed_sec,
                scenario_.config.source_width,
                scenario_.config.source_height
            );

            if (state.is_despawned || !state.is_active) {
                continue;
            }

            bool occluded_drop = false;
            bool is_predicted = false;
            bool is_partial = false;
            float occlusion_ratio = 0.0f;

            // 1. Evaluate temporal occlusions
            for (const auto& temp_occ : scenario_.temporal_occlusions) {
                if (temp_occ.target_id == target_spec.target_id &&
                    timestamp_ns >= temp_occ.start_time_ns &&
                    timestamp_ns <= temp_occ.end_time_ns) {
                    if (temp_occ.mode == OcclusionMode::drop) {
                        occluded_drop = true;
                    } else if (temp_occ.mode == OcclusionMode::predicted) {
                        is_predicted = true;
                    } else if (temp_occ.mode == OcclusionMode::partial) {
                        is_partial = true;
                    }
                }
            }

            // 2. Evaluate rectangular spatial occluders
            const BoundingBox target_bbox = make_bounding_box(state.pos_px, state.radius_px);
            const float target_area = 4.0f * state.radius_px * state.radius_px;

            for (const auto& rect_occ : scenario_.rect_occluders) {
                if (timestamp_ns >= rect_occ.start_time_ns && timestamp_ns <= rect_occ.end_time_ns) {
                    const float inter_area = compute_bbox_intersection_area(target_bbox, rect_occ.bounds_px);
                    if (inter_area > 0.0f && target_area > 0.0f) {
                        const float ratio = (std::min)(1.0f, inter_area / target_area);
                        occlusion_ratio = (std::max)(occlusion_ratio, ratio);

                        if (ratio >= 0.75f) {
                            if (rect_occ.mode == OcclusionMode::drop) {
                                if (scenario_.config.emit_predicted_when_occluded) {
                                    is_predicted = true;
                                } else {
                                    occluded_drop = true;
                                }
                            } else if (rect_occ.mode == OcclusionMode::predicted) {
                                is_predicted = true;
                            } else {
                                is_partial = true;
                            }
                        } else {
                            is_partial = true;
                        }
                    }
                }
            }

            // 3. Evaluate circular spatial occluders
            for (const auto& circle_occ : scenario_.circle_occluders) {
                if (timestamp_ns >= circle_occ.start_time_ns && timestamp_ns <= circle_occ.end_time_ns) {
                    const float dx = state.pos_px.x - circle_occ.center_px.x;
                    const float dy = state.pos_px.y - circle_occ.center_px.y;
                    const float dist_sq = dx * dx + dy * dy;
                    const float rad_sum = state.radius_px + circle_occ.radius_px;

                    if (dist_sq < (rad_sum * rad_sum)) {
                        const float dist = std::sqrt(dist_sq);
                        if (dist + state.radius_px <= circle_occ.radius_px) {
                            // Fully inside circle occluder
                            occlusion_ratio = 1.0f;
                            if (circle_occ.mode == OcclusionMode::drop) {
                                if (scenario_.config.emit_predicted_when_occluded) {
                                    is_predicted = true;
                                } else {
                                    occluded_drop = true;
                                }
                            } else if (circle_occ.mode == OcclusionMode::predicted) {
                                is_predicted = true;
                            } else {
                                is_partial = true;
                            }
                        } else {
                            // Partial overlap
                            occlusion_ratio = (std::max)(occlusion_ratio, 0.5f);
                            is_partial = true;
                        }
                    }
                }
            }

            // Handle drop mode
            if (occluded_drop) {
                ++total_dropped_occlusions_;
                continue;
            }

            // 4. Noise and Uncertainty Model
            const float pos_noise_sigma = (target_spec.noise.position_stddev_px > 0.0f)
                ? target_spec.noise.position_stddev_px
                : scenario_.config.default_noise.position_stddev_px;

            const float vel_noise_sigma = (target_spec.noise.velocity_stddev_px_per_s > 0.0f)
                ? target_spec.noise.velocity_stddev_px_per_s
                : scenario_.config.default_noise.velocity_stddev_px_per_s;

            const float base_conf = (target_spec.noise.base_confidence > 0.0f)
                ? target_spec.noise.base_confidence
                : scenario_.config.default_noise.base_confidence;

            // Dropout check
            const float dropout_prob = (target_spec.noise.dropout_probability > 0.0f)
                ? target_spec.noise.dropout_probability
                : scenario_.config.default_noise.dropout_probability;

            if (dropout_prob > 0.0f && rng_.next_uniform_f32() < dropout_prob) {
                ++total_dropped_occlusions_;
                continue;
            }

            PixelPoint obs_pos = state.pos_px;
            if (pos_noise_sigma > 0.0f) {
                const auto noise = rng_.next_gaussian_pair(0.0f, pos_noise_sigma);
                obs_pos.x += noise.first;
                obs_pos.y += noise.second;
            }

            PixelVelocity obs_vel = state.vel_px_per_s;
            if (vel_noise_sigma > 0.0f) {
                const auto vnoise = rng_.next_gaussian_pair(0.0f, vel_noise_sigma);
                obs_vel.x_per_s += vnoise.first;
                obs_vel.y_per_s += vnoise.second;
            }

            Visibility vis = Visibility::visible;
            float conf = base_conf;
            float var_px2 = (pos_noise_sigma > 0.0f) ? (pos_noise_sigma * pos_noise_sigma) : 0.01f;

            if (is_predicted) {
                vis = Visibility::predicted;
                conf = base_conf * 0.30f;
                var_px2 += 25.0f; // Inflated uncertainty
            } else if (is_partial) {
                vis = Visibility::partial;
                conf = base_conf * (1.0f - 0.5f * occlusion_ratio);
                var_px2 *= (1.0f + 3.0f * occlusion_ratio * occlusion_ratio);
            }

            // Construct canonical TargetObservation
            bus::TargetObservation obs{};
            obs.source_id = target_spec.target_id;
            obs.frame_id = replay_clock_.frame_id();
            obs.captured_at_ns = timestamp_ns;
            obs.center_px = obs_pos;
            obs.center_norm = pixel_to_normalized(
                obs_pos,
                scenario_.config.source_width,
                scenario_.config.source_height
            );
            obs.bbox_px = make_bounding_box(obs_pos, state.radius_px);
            obs.effective_radius_px = state.radius_px;
            obs.confidence = conf;
            obs.covariance_px2 = Covariance2D{
                .xx = var_px2,
                .xy = 0.0f,
                .yy = var_px2
            };
            obs.velocity = bus::VelocityHint{
                .pixels_per_second = obs_vel,
                .confidence = (vel_noise_sigma > 0.0f) ? 0.90f : 1.0f
            };
            obs.visibility = vis;
            obs.target_value = target_spec.target_value;
            obs.semantic_id = target_spec.semantic_id;

            if (out_batch.add_target(obs)) {
                ++total_generated_observations_;
            }
        }

        ++total_generated_batches_;
        return out_batch.target_count;
    }

    /// @brief Generates observation batch for current clock tick without advancing.
    [[nodiscard]] bus::TargetObservationBatch generate_current_batch() noexcept {
        generate_batch(replay_clock_.now_ns(), preallocated_batch_);
        return preallocated_batch_;
    }

    /// @brief Advances simulation by 1 tick and returns generated batch.
    [[nodiscard]] bus::TargetObservationBatch step() noexcept {
        generate_batch(replay_clock_.now_ns(), preallocated_batch_);
        replay_clock_.advance_tick();
        return preallocated_batch_;
    }

    /// @brief Advances simulation by 1 tick and publishes directly to SPSC ring buffer (wait-free, zero alloc).
    template <std::size_t Capacity>
    std::uint64_t step_and_publish(bus::LatestSpscRing<bus::TargetObservationBatch, Capacity>& ring) noexcept {
        generate_batch(replay_clock_.now_ns(), preallocated_batch_);
        const std::uint64_t seq = ring.push(preallocated_batch_);
        replay_clock_.advance_tick();
        return seq;
    }

    /// @brief Resets simulation state and PRNG seed.
    void reset(std::uint64_t new_seed = 0) noexcept {
        const std::uint64_t s = (new_seed != 0) ? new_seed : scenario_.config.seed;
        rng_.reseed(s);
        replay_clock_.reset(scenario_.config.start_time_ns);
        total_generated_batches_ = 0;
        total_generated_observations_ = 0;
        total_dropped_occlusions_ = 0;
    }

private:
    Scenario scenario_;
    DeterministicRng rng_;
    ReplayClock replay_clock_;
    bus::TargetObservationBatch preallocated_batch_{};

    std::uint64_t total_generated_batches_{0};
    std::uint64_t total_generated_observations_{0};
    std::uint64_t total_dropped_occlusions_{0};
};

} // namespace aim::sim
