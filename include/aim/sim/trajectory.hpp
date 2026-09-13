// include/aim/sim/trajectory.hpp
// Closed-form and kinematic trajectory generators for synthetic target motion
#pragma once

#include <cmath>
#include <cstdint>
#include "aim/sim/types.hpp"

namespace aim::sim {

/// @brief Evaluates boundary interaction for 1D coordinate in interval [min_val, max_val].
inline void apply_1d_boundary(
    float raw_pos,
    float raw_vel,
    float min_val,
    float max_val,
    BoundaryBehavior behavior,
    float& out_pos,
    float& out_vel,
    bool& out_despawned
) noexcept {
    const float span = max_val - min_val;
    if (span <= 0.0f) {
        out_pos = min_val;
        out_vel = 0.0f;
        return;
    }

    switch (behavior) {
        case BoundaryBehavior::bounce: {
            const float u = raw_pos - min_val;
            const float period = 2.0f * span;
            float m = std::fmod(u, period);
            if (m < 0.0f) m += period;

            if (m <= span) {
                out_pos = min_val + m;
                out_vel = raw_vel;
            } else {
                out_pos = min_val + 2.0f * span - m;
                out_vel = -raw_vel;
            }
            break;
        }

        case BoundaryBehavior::wrap: {
            const float u = raw_pos - min_val;
            float m = std::fmod(u, span);
            if (m < 0.0f) m += span;
            out_pos = min_val + m;
            out_vel = raw_vel;
            break;
        }

        case BoundaryBehavior::clamp: {
            if (raw_pos < min_val) {
                out_pos = min_val;
                out_vel = 0.0f;
            } else if (raw_pos > max_val) {
                out_pos = max_val;
                out_vel = 0.0f;
            } else {
                out_pos = raw_pos;
                out_vel = raw_vel;
            }
            break;
        }

        case BoundaryBehavior::despawn: {
            if (raw_pos < min_val || raw_pos > max_val) {
                out_pos = raw_pos;
                out_vel = raw_vel;
                out_despawned = true;
            } else {
                out_pos = raw_pos;
                out_vel = raw_vel;
            }
            break;
        }

        case BoundaryBehavior::none:
        default: {
            out_pos = raw_pos;
            out_vel = raw_vel;
            break;
        }
    }
}

/// @brief Evaluates target state for a given elapsed simulation time.
[[nodiscard]] inline TrajectoryState evaluate_trajectory(
    const TargetSpec& spec,
    double elapsed_sec,
    std::uint32_t screen_width = 1920,
    std::uint32_t screen_height = 1080
) noexcept {
    TrajectoryState state;
    state.radius_px = spec.radius_px;
    state.is_active = true;
    state.is_despawned = false;

    const float t = static_cast<float>(elapsed_sec);
    const float min_x = spec.radius_px;
    const float max_x = static_cast<float>(screen_width) - spec.radius_px;
    const float min_y = spec.radius_px;
    const float max_y = static_cast<float>(screen_height) - spec.radius_px;

    switch (spec.trajectory_type) {
        case TrajectoryType::stationary: {
            state.pos_px = spec.initial_pos_px;
            state.vel_px_per_s = {0.0f, 0.0f};
            state.accel_px_per_s2 = {0.0f, 0.0f};
            break;
        }

        case TrajectoryType::linear_cv:
        case TrajectoryType::bouncing_box: {
            const float raw_x = spec.initial_pos_px.x + spec.velocity_px_per_s.x_per_s * t;
            const float raw_y = spec.initial_pos_px.y + spec.velocity_px_per_s.y_per_s * t;
            bool despawned = false;

            apply_1d_boundary(raw_x, spec.velocity_px_per_s.x_per_s, min_x, max_x, spec.boundary, state.pos_px.x, state.vel_px_per_s.x_per_s, despawned);
            apply_1d_boundary(raw_y, spec.velocity_px_per_s.y_per_s, min_y, max_y, spec.boundary, state.pos_px.y, state.vel_px_per_s.y_per_s, despawned);

            state.accel_px_per_s2 = {0.0f, 0.0f};
            if (despawned) {
                state.is_despawned = true;
                state.is_active = false;
            }
            break;
        }

        case TrajectoryType::linear_ca: {
            const float raw_x = spec.initial_pos_px.x + spec.velocity_px_per_s.x_per_s * t + 0.5f * spec.accel_px_per_s2.x_per_s2 * t * t;
            const float raw_y = spec.initial_pos_px.y + spec.velocity_px_per_s.y_per_s * t + 0.5f * spec.accel_px_per_s2.y_per_s2 * t * t;
            const float raw_vx = spec.velocity_px_per_s.x_per_s + spec.accel_px_per_s2.x_per_s2 * t;
            const float raw_vy = spec.velocity_px_per_s.y_per_s + spec.accel_px_per_s2.y_per_s2 * t;
            bool despawned = false;

            apply_1d_boundary(raw_x, raw_vx, min_x, max_x, spec.boundary, state.pos_px.x, state.vel_px_per_s.x_per_s, despawned);
            apply_1d_boundary(raw_y, raw_vy, min_y, max_y, spec.boundary, state.pos_px.y, state.vel_px_per_s.y_per_s, despawned);

            state.accel_px_per_s2 = spec.accel_px_per_s2;
            if (despawned) {
                state.is_despawned = true;
                state.is_active = false;
            }
            break;
        }

        case TrajectoryType::sinusoidal_strafe: {
            constexpr float kTwoPi = 6.28318530717958647692f;
            const float omega_x = kTwoPi * spec.frequency_hz;
            const float omega_y = (spec.amplitude_y_px > 0.0f) ? (kTwoPi * spec.frequency_hz * 0.5f) : 0.0f;

            const float phase = spec.phase_rad;
            state.pos_px.x = spec.initial_pos_px.x + spec.amplitude_x_px * sin_f32(omega_x * t + phase);
            state.pos_px.y = spec.initial_pos_px.y + spec.amplitude_y_px * cos_f32(omega_y * t + phase);

            state.vel_px_per_s.x_per_s = spec.amplitude_x_px * omega_x * cos_f32(omega_x * t + phase);
            state.vel_px_per_s.y_per_s = -spec.amplitude_y_px * omega_y * sin_f32(omega_y * t + phase);

            state.accel_px_per_s2.x_per_s2 = -spec.amplitude_x_px * omega_x * omega_x * sin_f32(omega_x * t + phase);
            state.accel_px_per_s2.y_per_s2 = -spec.amplitude_y_px * omega_y * omega_y * cos_f32(omega_y * t + phase);
            break;
        }

        case TrajectoryType::circular_orbit: {
            constexpr float kTwoPi = 6.28318530717958647692f;
            const float omega = kTwoPi * spec.frequency_hz;
            const float radius = spec.amplitude_x_px;
            const float angle = omega * t + spec.phase_rad;

            state.pos_px.x = spec.initial_pos_px.x + radius * cos_f32(angle);
            state.pos_px.y = spec.initial_pos_px.y + radius * sin_f32(angle);

            state.vel_px_per_s.x_per_s = -radius * omega * sin_f32(angle);
            state.vel_px_per_s.y_per_s = radius * omega * cos_f32(angle);

            state.accel_px_per_s2.x_per_s2 = -radius * omega * omega * cos_f32(angle);
            state.accel_px_per_s2.y_per_s2 = -radius * omega * omega * sin_f32(angle);
            break;
        }

        case TrajectoryType::sudden_cut: {
            const double cut_interval_s = static_cast<double>(spec.cut_interval_ns) / 1'000'000'000.0;
            if (cut_interval_s <= 0.0) {
                // Degenerate to linear CV
                state.pos_px.x = spec.initial_pos_px.x + spec.velocity_px_per_s.x_per_s * t;
                state.pos_px.y = spec.initial_pos_px.y + spec.velocity_px_per_s.y_per_s * t;
                state.vel_px_per_s = spec.velocity_px_per_s;
                state.accel_px_per_s2 = {0.0f, 0.0f};
                break;
            }

            const auto step_index = static_cast<std::uint64_t>(elapsed_sec / cut_interval_s);
            const double remainder_s = elapsed_sec - static_cast<double>(step_index) * cut_interval_s;
            const float dt_rem = static_cast<float>(remainder_s);
            const float cut_dt = static_cast<float>(cut_interval_s);

            const float v0_x = spec.velocity_px_per_s.x_per_s;
            const float v0_y = spec.velocity_px_per_s.y_per_s;
            const float speed = std::sqrt(v0_x * v0_x + v0_y * v0_y);
            const float theta0 = (speed > 1e-6f) ? atan2_f32(v0_y, v0_x) : 0.0f;

            // Fast path for 180-degree reversal
            constexpr float kPi = 3.14159265358979323846f;
            if (std::abs(spec.cut_angle_rad - kPi) < 1e-4f) {
                if ((step_index % 2) == 0) {
                    state.pos_px.x = spec.initial_pos_px.x + v0_x * dt_rem;
                    state.pos_px.y = spec.initial_pos_px.y + v0_y * dt_rem;
                    state.vel_px_per_s = spec.velocity_px_per_s;
                } else {
                    state.pos_px.x = spec.initial_pos_px.x + v0_x * (cut_dt - dt_rem);
                    state.pos_px.y = spec.initial_pos_px.y + v0_y * (cut_dt - dt_rem);
                    state.vel_px_per_s.x_per_s = -v0_x;
                    state.vel_px_per_s.y_per_s = -v0_y;
                }
            } else {
                // General arbitrary angle rotation
                float cur_x = spec.initial_pos_px.x;
                float cur_y = spec.initial_pos_px.y;

                for (std::uint64_t j = 0; j < step_index; ++j) {
                    const float angle_j = theta0 + static_cast<float>(j) * spec.cut_angle_rad;
                    cur_x += speed * cos_f32(angle_j) * cut_dt;
                    cur_y += speed * sin_f32(angle_j) * cut_dt;
                }

                const float angle_cur = theta0 + static_cast<float>(step_index) * spec.cut_angle_rad;
                const float cur_vx = speed * cos_f32(angle_cur);
                const float cur_vy = speed * sin_f32(angle_cur);

                state.pos_px.x = cur_x + cur_vx * dt_rem;
                state.pos_px.y = cur_y + cur_vy * dt_rem;
                state.vel_px_per_s.x_per_s = cur_vx;
                state.vel_px_per_s.y_per_s = cur_vy;
            }

            state.accel_px_per_s2 = {0.0f, 0.0f};
            break;
        }

        case TrajectoryType::piecewise:
        default: {
            state.pos_px = spec.initial_pos_px;
            state.vel_px_per_s = spec.velocity_px_per_s;
            state.accel_px_per_s2 = spec.accel_px_per_s2;
            break;
        }
    }

    return state;
}

} // namespace aim::sim
