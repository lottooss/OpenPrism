// src/trajectory/trajectory_planner.cpp
#include "aim/trajectory/trajectory_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aim::trajectory {

namespace {

// Minimum-jerk polynomial trajectory profile (Flash & Hogan)
// s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 for tau in [0, 1]
inline float min_jerk_s(float tau) noexcept {
    const float t2 = tau * tau;
    const float t3 = t2 * tau;
    const float t4 = t3 * tau;
    const float t5 = t4 * tau;
    return 10.0f * t3 - 15.0f * t4 + 6.0f * t5;
}

// First derivative: s_dot(tau) = (30*tau^2 - 60*tau^3 + 30*tau^4) / T
inline float min_jerk_s_dot(float tau, float T) noexcept {
    if (T <= 1e-6f) return 0.0f;
    const float t2 = tau * tau;
    const float t3 = t2 * tau;
    const float t4 = t3 * tau;
    return (30.0f * t2 - 60.0f * t3 + 30.0f * t4) / T;
}

} // namespace

bool HybridTrajectoryPlanner::plan(const bus::AimIntent& intent,
                                   const ControlState& state,
                                   const PlannerLimits& limits,
                                   TrajectoryPlan& out_plan) noexcept {
    out_plan = TrajectoryPlan{};
    out_plan.plan_id = next_plan_id_++;
    out_plan.correlation_id = intent.header;
    is_canceled_ = false;

    if (intent.target_track_id == 0) {
        curve_active_ = false;
        out_plan.point_count = 0;
        return true;
    }

    // Displacement vector from current crosshair to target aim point
    const float total_dx = intent.target_aim_px.x - state.current_crosshair_px.x;
    const float total_dy = intent.target_aim_px.y - state.current_crosshair_px.y;
    const float dist = std::hypot(total_dx, total_dy);
    const auto positive_finite = [](float value) { return std::isfinite(value) && value > 0.0f; };
    if (!std::isfinite(total_dx) || !std::isfinite(total_dy) || !std::isfinite(dist) ||
        !positive_finite(config_.time_step_s) || config_.time_step_s < 0.001f ||
        config_.time_step_s > 0.010f || !positive_finite(config_.min_movement_time_s) ||
        !positive_finite(config_.max_movement_time_s) || config_.max_movement_time_s > 1.0f ||
        !positive_finite(config_.counts_per_pixel) || config_.counts_per_pixel > 100.0f ||
        !positive_finite(limits.max_velocity_px_s) || !positive_finite(limits.max_accel_px_s2) ||
        !positive_finite(limits.max_jerk_px_s3) || !positive_finite(limits.small_error_threshold_px) ||
        static_cast<double>(dist) * config_.counts_per_pixel > (std::numeric_limits<std::int32_t>::max)()) {
        curve_active_ = false;
        return false;
    }

    // If already at target (within subpixel tolerance), emit empty / single zero plan
    if (dist < 0.10f) {
        curve_active_ = false;
        out_plan.point_count = 0;
        return true;
    }

    const MonotonicNs now_ns = intent.command_deadline_ns > 10'000'000LL
        ? intent.command_deadline_ns - 10'000'000LL
        : 0;
    out_plan.start_time_ns = now_ns;
    const auto step_ns = static_cast<MonotonicNs>(std::llround(static_cast<double>(config_.time_step_s) * 1e9));
    if (now_ns > (std::numeric_limits<MonotonicNs>::max)() - step_ns * static_cast<MonotonicNs>(kMaxTrajectoryPoints)) return false;

    // Refresh the source provenance without restarting a stationary target's
    // motion phase. Feedback is allowed one count of visual quantization error.
    const float elapsed_s = curve_active_ && now_ns >= curve_start_ns_
        ? static_cast<float>(static_cast<double>(now_ns - curve_start_ns_) * 1e-9) : 0.0f;
    const float progress = curve_active_ ? min_jerk_s(std::clamp(elapsed_s / curve_duration_s_, 0.0f, 1.0f)) : 0.0f;
    if (curve_active_ && (intent.target_track_id != curve_track_id_ || now_ns <= last_plan_time_ns_))
        curve_active_ = false;
    if (curve_active_ && (elapsed_s >= curve_duration_s_ ||
        std::hypot(total_dx - curve_dx_ * (1.0f - progress),
                   total_dy - curve_dy_ * (1.0f - progress)) > 2.0f)) {
        const float length_sq = curve_dx_ * curve_dx_ + curve_dy_ * curve_dy_;
        const float remaining = (total_dx * curve_dx_ + total_dy * curve_dy_) / length_sq;
        const float lateral = std::hypot(total_dx - curve_dx_ * remaining,
                                          total_dy - curve_dy_ * remaining);
        if (!std::isfinite(remaining) || remaining < 0.0f || remaining > 1.0f || lateral > 2.0f) {
            curve_active_ = false;
        } else {
            // A canceled stale plan never executed its tail. Rejoin the same
            // bounded curve at observed progress instead of assuming wall time
            // executed it, or repeatedly restarting at zero velocity.
            const float observed_progress = 1.0f - remaining;
            float low = 0.0f, high = 1.0f;
            for (unsigned iteration = 0; iteration < 20; ++iteration) {
                const float middle = (low + high) * 0.5f;
                if (min_jerk_s(middle) < observed_progress) low = middle;
                else high = middle;
            }
            const auto observed_phase_ns = static_cast<MonotonicNs>(
                static_cast<double>((low + high) * 0.5f * curve_duration_s_) * 1e9);
            curve_start_ns_ = now_ns - observed_phase_ns;
        }
    }
    last_plan_time_ns_ = now_ns;

    // CASE 1: Direct Feed-Forward (Small Errors)
    if (!curve_active_ && dist <= limits.small_error_threshold_px) {
        // Compute critically damped 1-step or 2-step correction
        const float step_count_x = total_dx * config_.counts_per_pixel;
        const float step_count_y = total_dy * config_.counts_per_pixel;

        // Micro-step 1
        TrajectoryPoint pt{};
        pt.target_time_ns = now_ns + step_ns;
        pt.position_px = intent.target_aim_px;
        pt.velocity_px_s = PixelVelocity{total_dx / config_.time_step_s, total_dy / config_.time_step_s};
        pt.step_delta_x_counts = static_cast<std::int32_t>(std::round(step_count_x));
        pt.step_delta_y_counts = static_cast<std::int32_t>(std::round(step_count_y));

        out_plan.points[0] = pt;
        out_plan.point_count = 1;
        out_plan.end_time_ns = pt.target_time_ns;
        return true;
    }

    // CASE 2: Jerk-Limited Minimum-Jerk Profile (Large Motion)
    // Compute required movement duration T to satisfy velocity, acceleration, and jerk limits:
    // v_max = 1.875 * D / T   => T >= 1.875 * D / v_lim
    // a_max = 5.774 * D / T^2 => T >= sqrt(5.774 * D / a_lim)
    // j_max = 60.0  * D / T^3 => T >= cbrt(60.0 * D / j_lim)

    const float t_vel = 1.875f * dist / std::max(1.0f, limits.max_velocity_px_s);
    const float t_acc = std::sqrt(5.774f * dist / std::max(1.0f, limits.max_accel_px_s2));
    const float t_jerk = std::cbrt(60.0f * dist / std::max(1.0f, limits.max_jerk_px_s3));

    if (!curve_active_) {
        curve_duration_s_ = std::max({config_.min_movement_time_s, t_vel, t_acc, t_jerk});
        curve_dx_ = total_dx;
        curve_dy_ = total_dy;
        curve_start_ns_ = now_ns;
        curve_track_id_ = intent.target_track_id;
        curve_active_ = true;
    }
    const float T = curve_duration_s_;
    const float offset_s = static_cast<float>(static_cast<double>(now_ns - curve_start_ns_) * 1e-9);
    const float initial_progress = min_jerk_s(std::clamp(offset_s / T, 0.0f, 1.0f));

    // Number of 1 kHz steps to plan in this batch (up to buffer capacity)
    const std::uint32_t total_steps = static_cast<std::uint32_t>(std::clamp(
        std::ceil((T - offset_s) / config_.time_step_s), 1.0f, static_cast<float>(kMaxTrajectoryPoints)));
    const std::uint32_t num_steps = std::clamp<std::uint32_t>(
        total_steps,
        1,
        static_cast<std::uint32_t>(kMaxTrajectoryPoints)
    );

    std::int32_t cumulative_counts_x = 0;
    std::int32_t cumulative_counts_y = 0;

    const float target_total_counts_x = curve_dx_ * config_.counts_per_pixel;
    const float target_total_counts_y = curve_dy_ * config_.counts_per_pixel;

    for (std::uint32_t i = 1; i <= num_steps; ++i) {
        const float t = offset_s + static_cast<float>(i) * config_.time_step_s;
        const float tau = std::min(1.0f, t / T);

        const float s = min_jerk_s(tau) - initial_progress;
        const float s_dot = min_jerk_s_dot(tau, T);

        const float cur_x = state.current_crosshair_px.x + curve_dx_ * s;
        const float cur_y = state.current_crosshair_px.y + curve_dy_ * s;

        const float vel_x = curve_dx_ * s_dot;
        const float vel_y = curve_dy_ * s_dot;

        // Compute ideal cumulative counts up to step i
        const std::int32_t ideal_cum_x = static_cast<std::int32_t>(std::round(target_total_counts_x * s));
        const std::int32_t ideal_cum_y = static_cast<std::int32_t>(std::round(target_total_counts_y * s));

        // Incremental counts for this step
        const std::int32_t delta_counts_x = ideal_cum_x - cumulative_counts_x;
        const std::int32_t delta_counts_y = ideal_cum_y - cumulative_counts_y;

        cumulative_counts_x = ideal_cum_x;
        cumulative_counts_y = ideal_cum_y;

        TrajectoryPoint pt{};
        pt.target_time_ns = now_ns + static_cast<MonotonicNs>(i) * step_ns;
        pt.position_px = PixelPoint{cur_x, cur_y};
        pt.velocity_px_s = PixelVelocity{vel_x, vel_y};
        pt.step_delta_x_counts = delta_counts_x;
        pt.step_delta_y_counts = delta_counts_y;

        out_plan.points[i - 1] = pt;
    }

    out_plan.point_count = num_steps;
    out_plan.end_time_ns = out_plan.points[num_steps - 1].target_time_ns;
    return true;
}

} // namespace aim::trajectory
