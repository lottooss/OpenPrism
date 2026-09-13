// include/aim/sim/types.hpp
// Type definitions, enums, and data contracts for the Deterministic Target Simulator
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim::sim {

/// @brief binary32 transcendentals evaluated in binary64 and narrowed once.
///
/// The single-precision libm entry points are not correctly rounded and disagree between
/// implementations: on glibc 2.42, `cos_f32(0.99483f)` returns `0x3f0b6d75` while the
/// correctly-rounded result is `0x3f0b6d76`. Calling them directly would make this
/// simulator's "bit-exact deterministic replay" contract depend on which C runtime the
/// binary happens to link, and would put it permanently out of step with `tools/sim/`,
/// whose Python `math` module is binary64. Evaluating in double and rounding once keeps
/// the result stable across MSVC, GCC, and Clang. `aim::sim::DeterministicRng` already
/// takes this approach for Box-Muller; these helpers apply it to the kinematics.
///
/// `sqrt` needs no such helper: IEEE-754 requires correct rounding, and binary64 carries
/// enough bits that narrowing a binary64 square root never double-rounds.
[[nodiscard]] inline float sin_f32(float radians) noexcept {
    return static_cast<float>(std::sin(static_cast<double>(radians)));
}

[[nodiscard]] inline float cos_f32(float radians) noexcept {
    return static_cast<float>(std::cos(static_cast<double>(radians)));
}

[[nodiscard]] inline float atan2_f32(float y, float x) noexcept {
    return static_cast<float>(std::atan2(static_cast<double>(y), static_cast<double>(x)));
}


/// @brief Trajectory generation mathematical models.
enum class TrajectoryType : std::uint8_t {
    stationary       = 0,
    linear_cv        = 1, // Constant velocity
    linear_ca        = 2, // Constant acceleration
    sinusoidal_strafe= 3, // Harmonic 2D sinusoidal strafe
    circular_orbit   = 4, // 2D circular orbit
    sudden_cut       = 5, // Discrete directional cuts / step reversals
    bouncing_box     = 6, // Linear motion with specular boundary reflections
    piecewise        = 7  // Piecewise waypoint path
};

/// @brief Boundary interaction behaviors when reaching screen bounds.
enum class BoundaryBehavior : std::uint8_t {
    none             = 0, // Fly off-screen unrestricted
    bounce           = 1, // Specular elastic reflection
    wrap             = 2, // Toroidal screen wrapping
    clamp            = 3, // Clamped to boundary limits
    despawn          = 4  // Despawned immediately on boundary touch
};

/// @brief Occlusion handling mode.
enum class OcclusionMode : std::uint8_t {
    drop             = 0, // Omitted completely from observation batch
    predicted        = 1, // Present with Visibility::predicted and inflated covariance
    partial          = 2  // Present with Visibility::partial and scaled confidence
};

/// @brief Axis-aligned rectangular occlusion zone.
struct RectOccluder {
    BoundingBox bounds_px{};
    MonotonicNs start_time_ns{0};
    MonotonicNs end_time_ns{std::numeric_limits<std::int64_t>::max()};
    OcclusionMode mode{OcclusionMode::drop};
};

/// @brief Circular occlusion zone.
struct CircleOccluder {
    PixelPoint center_px{};
    float radius_px{0.0f};
    MonotonicNs start_time_ns{0};
    MonotonicNs end_time_ns{std::numeric_limits<std::int64_t>::max()};
    OcclusionMode mode{OcclusionMode::drop};
};

/// @brief Scheduled temporal occlusion for a specific target.
struct TemporalOcclusion {
    std::uint64_t target_id{0};
    MonotonicNs start_time_ns{0};
    MonotonicNs end_time_ns{std::numeric_limits<std::int64_t>::max()};
    OcclusionMode mode{OcclusionMode::drop};
};

/// @brief Measurement noise and perception uncertainty configuration.
struct NoiseConfig {
    float position_stddev_px{0.0f};
    float velocity_stddev_px_per_s{0.0f};
    float radius_stddev_px{0.0f};
    float base_confidence{0.98f};
    float confidence_jitter{0.0f};
    float dropout_probability{0.0f};
};

/// @brief Specification for a synthetic target in the simulation scenario.
struct TargetSpec {
    std::uint64_t target_id{1};
    std::uint32_t semantic_id{0};
    TrajectoryType trajectory_type{TrajectoryType::stationary};
    BoundaryBehavior boundary{BoundaryBehavior::bounce};

    PixelPoint initial_pos_px{960.0f, 540.0f};
    PixelVelocity velocity_px_per_s{0.0f, 0.0f};
    PixelAcceleration accel_px_per_s2{0.0f, 0.0f};
    float radius_px{20.0f};
    float target_value{1.0f};

    // Harmonic / Sinusoidal / Circular parameters
    float amplitude_x_px{0.0f};
    float amplitude_y_px{0.0f};
    float frequency_hz{1.0f};
    float phase_rad{0.0f};

    // Sudden Cut parameters
    MonotonicNs cut_interval_ns{0};
    float cut_angle_rad{3.14159265358979323846f}; // 180° reversal by default

    // Target Lifecycle
    MonotonicNs spawn_time_ns{0};
    MonotonicNs despawn_time_ns{std::numeric_limits<std::int64_t>::max()};

    // Noise overrides
    NoiseConfig noise{};
};

/// @brief Global simulation scenario configuration.
struct ScenarioConfig {
    std::string name{"default"};
    std::uint64_t seed{133742ULL};
    double cadence_hz{144.0};
    MonotonicNs start_time_ns{1'000'000'000LL}; // 1.0 s default
    MonotonicNs duration_ns{30'000'000'000LL};   // 30.0 s default
    std::uint32_t source_width{1920};
    std::uint32_t source_height{1080};
    std::uint64_t source_id{1};
    std::uint32_t pipeline_run_id{100};
    MonotonicNs synthetic_perception_latency_ns{2'500'000LL}; // 2.5 ms
    NoiseConfig default_noise{};
    bool emit_predicted_when_occluded{true};
};

/// @brief State of an evaluated target at an instantaneous simulation timestamp.
struct TrajectoryState {
    PixelPoint pos_px{0.0f, 0.0f};
    PixelVelocity vel_px_per_s{0.0f, 0.0f};
    PixelAcceleration accel_px_per_s2{0.0f, 0.0f};
    float radius_px{20.0f};
    bool is_active{true};
    bool is_despawned{false};
};

// =============================================================================
// Coordinate System Math Utilities (Subpixel Precision Guarantee)
// =============================================================================

/// @brief Converts top-left pixel coordinates to normalized center [-1.0, 1.0] coordinates.
[[nodiscard]] constexpr NormalizedPoint pixel_to_normalized(
    PixelPoint px,
    std::uint32_t width = 1920,
    std::uint32_t height = 1080
) noexcept {
    const float half_w = static_cast<float>(width) * 0.5f;
    const float half_h = static_cast<float>(height) * 0.5f;
    return NormalizedPoint{
        .x = (half_w > 0.0f) ? ((px.x - half_w) / half_w) : 0.0f,
        .y = (half_h > 0.0f) ? ((px.y - half_h) / half_h) : 0.0f
    };
}

/// @brief Converts normalized center [-1.0, 1.0] coordinates to top-left pixel coordinates.
[[nodiscard]] constexpr PixelPoint normalized_to_pixel(
    NormalizedPoint norm,
    std::uint32_t width = 1920,
    std::uint32_t height = 1080
) noexcept {
    const float half_w = static_cast<float>(width) * 0.5f;
    const float half_h = static_cast<float>(height) * 0.5f;
    return PixelPoint{
        .x = (norm.x + 1.0f) * half_w,
        .y = (norm.y + 1.0f) * half_h
    };
}

/// @brief Computes axis-aligned bounding box from center position and radius.
[[nodiscard]] constexpr BoundingBox make_bounding_box(PixelPoint center, float radius) noexcept {
    return BoundingBox{
        .left = center.x - radius,
        .top = center.y - radius,
        .right = center.x + radius,
        .bottom = center.y + radius
    };
}

/// @brief Computes 2D geometric intersection area between two axis-aligned bounding boxes.
[[nodiscard]] inline float compute_bbox_intersection_area(
    const BoundingBox& a,
    const BoundingBox& b
) noexcept {
    const float inter_left = (std::max)(a.left, b.left);
    const float inter_top = (std::max)(a.top, b.top);
    const float inter_right = (std::min)(a.right, b.right);
    const float inter_bottom = (std::min)(a.bottom, b.bottom);

    const float inter_w = (std::max)(0.0f, inter_right - inter_left);
    const float inter_h = (std::max)(0.0f, inter_bottom - inter_top);
    return inter_w * inter_h;
}

} // namespace aim::sim
