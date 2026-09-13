// include/aim/sim/scenario_builder.hpp
// Fluent programmatic builder for synthetic simulation scenarios and standard presets
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include "aim/sim/scenario.hpp"
#include "aim/sim/types.hpp"

namespace aim::sim {

/// @brief Fluent C++ builder for constructing deterministic scenarios without external file I/O.
class ScenarioBuilder {
public:
    explicit ScenarioBuilder(std::string_view name = "scenario") {
        scenario_.config.name = std::string(name);
    }

    ScenarioBuilder& with_name(std::string_view name) {
        scenario_.config.name = std::string(name);
        return *this;
    }

    ScenarioBuilder& with_seed(std::uint64_t seed) noexcept {
        scenario_.config.seed = seed;
        return *this;
    }

    ScenarioBuilder& with_cadence_hz(double hz) noexcept {
        if (hz > 0.0) scenario_.config.cadence_hz = hz;
        return *this;
    }

    ScenarioBuilder& with_start_time_ns(MonotonicNs start_ns) noexcept {
        scenario_.config.start_time_ns = start_ns;
        return *this;
    }

    ScenarioBuilder& with_duration_ns(MonotonicNs duration_ns) noexcept {
        scenario_.config.duration_ns = duration_ns;
        return *this;
    }

    ScenarioBuilder& with_duration_seconds(double duration_s) noexcept {
        scenario_.config.duration_ns = static_cast<MonotonicNs>(duration_s * 1'000'000'000.0);
        return *this;
    }

    ScenarioBuilder& with_screen_size(std::uint32_t width, std::uint32_t height) noexcept {
        scenario_.config.source_width = width;
        scenario_.config.source_height = height;
        return *this;
    }

    ScenarioBuilder& with_source_id(std::uint64_t source_id) noexcept {
        scenario_.config.source_id = source_id;
        return *this;
    }

    ScenarioBuilder& with_pipeline_run_id(std::uint32_t run_id) noexcept {
        scenario_.config.pipeline_run_id = run_id;
        return *this;
    }

    ScenarioBuilder& with_perception_latency_ns(MonotonicNs latency_ns) noexcept {
        scenario_.config.synthetic_perception_latency_ns = latency_ns;
        return *this;
    }

    ScenarioBuilder& with_default_noise(const NoiseConfig& noise) noexcept {
        scenario_.config.default_noise = noise;
        return *this;
    }

    ScenarioBuilder& with_position_noise(float stddev_px) noexcept {
        scenario_.config.default_noise.position_stddev_px = stddev_px;
        return *this;
    }

    ScenarioBuilder& with_velocity_noise(float stddev_px_per_s) noexcept {
        scenario_.config.default_noise.velocity_stddev_px_per_s = stddev_px_per_s;
        return *this;
    }

    ScenarioBuilder& with_emit_predicted_when_occluded(bool emit) noexcept {
        scenario_.config.emit_predicted_when_occluded = emit;
        return *this;
    }

    ScenarioBuilder& add_target(const TargetSpec& spec) {
        scenario_.targets.push_back(spec);
        return *this;
    }

    ScenarioBuilder& add_stationary_target(
        std::uint64_t id,
        PixelPoint pos,
        float radius = 20.0f,
        MonotonicNs spawn_ns = 0,
        MonotonicNs despawn_ns = std::numeric_limits<std::int64_t>::max()
    ) {
        TargetSpec spec;
        spec.target_id = id;
        spec.trajectory_type = TrajectoryType::stationary;
        spec.initial_pos_px = pos;
        spec.radius_px = radius;
        spec.spawn_time_ns = spawn_ns;
        spec.despawn_time_ns = despawn_ns;
        return add_target(spec);
    }

    ScenarioBuilder& add_linear_target(
        std::uint64_t id,
        PixelPoint pos,
        PixelVelocity vel,
        float radius = 20.0f,
        BoundaryBehavior boundary = BoundaryBehavior::bounce,
        MonotonicNs spawn_ns = 0,
        MonotonicNs despawn_ns = std::numeric_limits<std::int64_t>::max()
    ) {
        TargetSpec spec;
        spec.target_id = id;
        spec.trajectory_type = TrajectoryType::linear_cv;
        spec.initial_pos_px = pos;
        spec.velocity_px_per_s = vel;
        spec.radius_px = radius;
        spec.boundary = boundary;
        spec.spawn_time_ns = spawn_ns;
        spec.despawn_time_ns = despawn_ns;
        return add_target(spec);
    }

    ScenarioBuilder& add_accelerating_target(
        std::uint64_t id,
        PixelPoint pos,
        PixelVelocity vel,
        PixelAcceleration accel,
        float radius = 20.0f,
        BoundaryBehavior boundary = BoundaryBehavior::bounce,
        MonotonicNs spawn_ns = 0,
        MonotonicNs despawn_ns = std::numeric_limits<std::int64_t>::max()
    ) {
        TargetSpec spec;
        spec.target_id = id;
        spec.trajectory_type = TrajectoryType::linear_ca;
        spec.initial_pos_px = pos;
        spec.velocity_px_per_s = vel;
        spec.accel_px_per_s2 = accel;
        spec.radius_px = radius;
        spec.boundary = boundary;
        spec.spawn_time_ns = spawn_ns;
        spec.despawn_time_ns = despawn_ns;
        return add_target(spec);
    }

    ScenarioBuilder& add_sinusoidal_target(
        std::uint64_t id,
        PixelPoint center,
        float amp_x,
        float amp_y,
        float freq_hz,
        float phase_rad = 0.0f,
        float radius = 20.0f,
        MonotonicNs spawn_ns = 0,
        MonotonicNs despawn_ns = std::numeric_limits<std::int64_t>::max()
    ) {
        TargetSpec spec;
        spec.target_id = id;
        spec.trajectory_type = TrajectoryType::sinusoidal_strafe;
        spec.initial_pos_px = center;
        spec.amplitude_x_px = amp_x;
        spec.amplitude_y_px = amp_y;
        spec.frequency_hz = freq_hz;
        spec.phase_rad = phase_rad;
        spec.radius_px = radius;
        spec.spawn_time_ns = spawn_ns;
        spec.despawn_time_ns = despawn_ns;
        return add_target(spec);
    }

    ScenarioBuilder& add_circular_target(
        std::uint64_t id,
        PixelPoint center,
        float orbit_radius,
        float freq_hz,
        float phase_rad = 0.0f,
        float target_radius = 20.0f,
        MonotonicNs spawn_ns = 0,
        MonotonicNs despawn_ns = std::numeric_limits<std::int64_t>::max()
    ) {
        TargetSpec spec;
        spec.target_id = id;
        spec.trajectory_type = TrajectoryType::circular_orbit;
        spec.initial_pos_px = center;
        spec.amplitude_x_px = orbit_radius;
        spec.frequency_hz = freq_hz;
        spec.phase_rad = phase_rad;
        spec.radius_px = target_radius;
        spec.spawn_time_ns = spawn_ns;
        spec.despawn_time_ns = despawn_ns;
        return add_target(spec);
    }

    ScenarioBuilder& add_sudden_cut_target(
        std::uint64_t id,
        PixelPoint pos,
        PixelVelocity vel,
        MonotonicNs cut_interval_ns,
        float cut_angle_rad = 3.14159265358979323846f,
        float radius = 20.0f,
        MonotonicNs spawn_ns = 0,
        MonotonicNs despawn_ns = std::numeric_limits<std::int64_t>::max()
    ) {
        TargetSpec spec;
        spec.target_id = id;
        spec.trajectory_type = TrajectoryType::sudden_cut;
        spec.initial_pos_px = pos;
        spec.velocity_px_per_s = vel;
        spec.cut_interval_ns = cut_interval_ns;
        spec.cut_angle_rad = cut_angle_rad;
        spec.radius_px = radius;
        spec.spawn_time_ns = spawn_ns;
        spec.despawn_time_ns = despawn_ns;
        return add_target(spec);
    }


    ScenarioBuilder& add_temporal_occlusion(
        std::uint64_t target_id,
        MonotonicNs start_ns,
        MonotonicNs end_ns,
        OcclusionMode mode = OcclusionMode::drop
    ) {
        scenario_.temporal_occlusions.push_back(TemporalOcclusion{
            .target_id = target_id,
            .start_time_ns = start_ns,
            .end_time_ns = end_ns,
            .mode = mode
        });
        return *this;
    }

    ScenarioBuilder& add_temporal_occlusion_seconds(
        std::uint64_t target_id,
        double start_s,
        double end_s,
        OcclusionMode mode = OcclusionMode::drop
    ) {
        return add_temporal_occlusion(
            target_id,
            static_cast<MonotonicNs>(start_s * 1'000'000'000.0),
            static_cast<MonotonicNs>(end_s * 1'000'000'000.0),
            mode
        );
    }

    ScenarioBuilder& add_rect_occluder(
        BoundingBox bounds,
        MonotonicNs start_ns = 0,
        MonotonicNs end_ns = std::numeric_limits<std::int64_t>::max(),
        OcclusionMode mode = OcclusionMode::drop
    ) {
        scenario_.rect_occluders.push_back(RectOccluder{
            .bounds_px = bounds,
            .start_time_ns = start_ns,
            .end_time_ns = end_ns,
            .mode = mode
        });
        return *this;
    }

    ScenarioBuilder& add_circle_occluder(
        PixelPoint center,
        float radius,
        MonotonicNs start_ns = 0,
        MonotonicNs end_ns = std::numeric_limits<std::int64_t>::max(),
        OcclusionMode mode = OcclusionMode::drop
    ) {
        scenario_.circle_occluders.push_back(CircleOccluder{
            .center_px = center,
            .radius_px = radius,
            .start_time_ns = start_ns,
            .end_time_ns = end_ns,
            .mode = mode
        });
        return *this;
    }

    [[nodiscard]] Scenario build() const {
        return scenario_;
    }

    // =========================================================================
    // Standard Scenario Presets
    // =========================================================================

    /// @brief 3-target static Gridshot benchmark scenario.
    static Scenario make_gridshot_preset(std::uint64_t seed = 42) {
        return ScenarioBuilder("gridshot_preset")
            .with_seed(seed)
            .with_cadence_hz(144.0)
            .with_duration_seconds(60.0)
            .add_stationary_target(1, {800.0f, 450.0f}, 25.0f)
            .add_stationary_target(2, {960.0f, 540.0f}, 25.0f)
            .add_stationary_target(3, {1120.0f, 630.0f}, 25.0f)
            .build();
    }

    /// @brief Strafing and reversing target scenario.
    static Scenario make_strafe_track_preset(std::uint64_t seed = 42) {
        return ScenarioBuilder("strafe_track_preset")
            .with_seed(seed)
            .with_cadence_hz(144.0)
            .with_duration_seconds(30.0)
            .add_sinusoidal_target(1, {960.0f, 540.0f}, 350.0f, 0.0f, 1.2f, 0.0f, 22.0f)
            .add_sudden_cut_target(2, {600.0f, 400.0f}, {400.0f, 0.0f}, 1'500'000'000LL, 3.14159265f, 20.0f)
            .build();
    }

    /// @brief Occluded target tracking scenario.
    static Scenario make_occlusion_preset(std::uint64_t seed = 42) {
        return ScenarioBuilder("occlusion_preset")
            .with_seed(seed)
            .with_cadence_hz(144.0)
            .with_duration_seconds(20.0)
            .add_linear_target(1, {400.0f, 540.0f}, {300.0f, 0.0f}, 24.0f, BoundaryBehavior::bounce)
            .add_rect_occluder(BoundingBox{850.0f, 400.0f, 1070.0f, 680.0f}, 0, std::numeric_limits<std::int64_t>::max(), OcclusionMode::predicted)
            .add_temporal_occlusion_seconds(1, 5.0, 7.0, OcclusionMode::drop)
            .build();
    }

    /// @brief High density multi-target batch saturation scenario (up to 64 targets).
    static Scenario make_density_preset(std::uint64_t seed = 42, std::size_t target_count = 64) {
        ScenarioBuilder b("density_preset");
        b.with_seed(seed).with_cadence_hz(144.0).with_duration_seconds(10.0);
        const std::size_t count = (std::min)(target_count, std::size_t{64});

        for (std::size_t i = 0; i < count; ++i) {
            const float angle = static_cast<float>(i) * (6.2831853f / static_cast<float>(count));
            const float radius = 200.0f + static_cast<float>(i % 4) * 80.0f;
            const float cx = 960.0f + radius * cos_f32(angle);
            const float cy = 540.0f + radius * sin_f32(angle);
            const float vx = -200.0f * sin_f32(angle);
            const float vy = 200.0f * cos_f32(angle);

            b.add_linear_target(
                static_cast<std::uint64_t>(i + 1),
                {cx, cy},
                {vx, vy},
                18.0f,
                BoundaryBehavior::bounce
            );
        }
        return b.build();
    }

private:
    Scenario scenario_{};
};

} // namespace aim::sim
