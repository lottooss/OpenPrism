"""tools/sim/scenario.py
Scenario definitions and ScenarioBuilder for fluent Python scenario construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from tools.bus.schema_bindings import BBoxf, Vec2f
from tools.sim.prng import f32
from tools.sim.trajectory import cos_f32, sin_f32
from tools.sim.types import (
    BoundaryBehavior,
    CircleOccluder,
    NoiseConfig,
    OcclusionMode,
    RectOccluder,
    ScenarioConfig,
    TargetSpec,
    TemporalOcclusion,
    TrajectoryType,
)


@dataclass
class Scenario:
    config: ScenarioConfig = field(default_factory=ScenarioConfig)
    targets: List[TargetSpec] = field(default_factory=list)
    temporal_occlusions: List[TemporalOcclusion] = field(default_factory=list)
    rect_occluders: List[RectOccluder] = field(default_factory=list)
    circle_occluders: List[CircleOccluder] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def seed(self) -> int:
        return self.config.seed

    @property
    def cadence_hz(self) -> float:
        return self.config.cadence_hz

    @property
    def start_time_ns(self) -> int:
        return self.config.start_time_ns

    @property
    def duration_ns(self) -> int:
        return self.config.duration_ns

    @property
    def screen_width(self) -> int:
        return self.config.source_width

    @property
    def screen_height(self) -> int:
        return self.config.source_height

    @property
    def target_count(self) -> int:
        return len(self.targets)


class ScenarioBuilder:
    """Fluent builder for constructing simulation scenarios."""

    def __init__(self, name: str = "scenario") -> None:
        self._scenario = Scenario(config=ScenarioConfig(name=name))

    def with_name(self, name: str) -> ScenarioBuilder:
        self._scenario.config.name = name
        return self

    def with_seed(self, seed: int) -> ScenarioBuilder:
        self._scenario.config.seed = seed
        return self

    def with_cadence_hz(self, hz: float) -> ScenarioBuilder:
        if hz > 0:
            self._scenario.config.cadence_hz = hz
        return self

    def with_start_time_ns(self, start_ns: int) -> ScenarioBuilder:
        self._scenario.config.start_time_ns = start_ns
        return self

    def with_duration_ns(self, duration_ns: int) -> ScenarioBuilder:
        self._scenario.config.duration_ns = duration_ns
        return self

    def with_duration_seconds(self, duration_s: float) -> ScenarioBuilder:
        self._scenario.config.duration_ns = int(duration_s * 1_000_000_000.0)
        return self

    def with_screen_size(self, width: int, height: int) -> ScenarioBuilder:
        self._scenario.config.source_width = width
        self._scenario.config.source_height = height
        return self

    def with_source_id(self, source_id: int) -> ScenarioBuilder:
        self._scenario.config.source_id = source_id
        return self

    def with_pipeline_run_id(self, run_id: int) -> ScenarioBuilder:
        self._scenario.config.pipeline_run_id = run_id
        return self

    def with_perception_latency_ns(self, latency_ns: int) -> ScenarioBuilder:
        self._scenario.config.synthetic_perception_latency_ns = latency_ns
        return self

    def with_default_noise(self, noise: NoiseConfig) -> ScenarioBuilder:
        self._scenario.config.default_noise = noise
        return self

    def with_position_noise(self, stddev_px: float) -> ScenarioBuilder:
        self._scenario.config.default_noise.position_stddev_px = stddev_px
        return self

    def with_velocity_noise(self, stddev_px_per_s: float) -> ScenarioBuilder:
        self._scenario.config.default_noise.velocity_stddev_px_per_s = stddev_px_per_s
        return self

    def with_emit_predicted_when_occluded(self, emit: bool) -> ScenarioBuilder:
        self._scenario.config.emit_predicted_when_occluded = emit
        return self

    def add_target(self, spec: TargetSpec) -> ScenarioBuilder:
        self._scenario.targets.append(spec)
        return self

    def add_stationary_target(
        self,
        target_id: int,
        pos: Vec2f,
        radius: float = 20.0,
        spawn_ns: int = 0,
        despawn_ns: int = 0x7FFFFFFFFFFFFFFF,
    ) -> ScenarioBuilder:
        spec = TargetSpec(
            target_id=target_id,
            trajectory_type=TrajectoryType.STATIONARY,
            initial_pos_px=pos,
            radius_px=radius,
            spawn_time_ns=spawn_ns,
            despawn_time_ns=despawn_ns,
        )
        return self.add_target(spec)

    def add_linear_target(
        self,
        target_id: int,
        pos: Vec2f,
        vel: Vec2f,
        radius: float = 20.0,
        boundary: BoundaryBehavior = BoundaryBehavior.BOUNCE,
        spawn_ns: int = 0,
        despawn_ns: int = 0x7FFFFFFFFFFFFFFF,
    ) -> ScenarioBuilder:
        spec = TargetSpec(
            target_id=target_id,
            trajectory_type=TrajectoryType.LINEAR_CV,
            initial_pos_px=pos,
            velocity_px_s=vel,
            radius_px=radius,
            boundary=boundary,
            spawn_time_ns=spawn_ns,
            despawn_time_ns=despawn_ns,
        )
        return self.add_target(spec)

    def add_accelerating_target(
        self,
        target_id: int,
        pos: Vec2f,
        vel: Vec2f,
        accel: Vec2f,
        radius: float = 20.0,
        boundary: BoundaryBehavior = BoundaryBehavior.BOUNCE,
        spawn_ns: int = 0,
        despawn_ns: int = 0x7FFFFFFFFFFFFFFF,
    ) -> ScenarioBuilder:
        spec = TargetSpec(
            target_id=target_id,
            trajectory_type=TrajectoryType.LINEAR_CA,
            initial_pos_px=pos,
            velocity_px_s=vel,
            accel_px_s2=accel,
            radius_px=radius,
            boundary=boundary,
            spawn_time_ns=spawn_ns,
            despawn_time_ns=despawn_ns,
        )
        return self.add_target(spec)

    def add_sinusoidal_target(
        self,
        target_id: int,
        center: Vec2f,
        amp_x: float,
        amp_y: float,
        freq_hz: float,
        phase_rad: float = 0.0,
        radius: float = 20.0,
        spawn_ns: int = 0,
        despawn_ns: int = 0x7FFFFFFFFFFFFFFF,
    ) -> ScenarioBuilder:
        spec = TargetSpec(
            target_id=target_id,
            trajectory_type=TrajectoryType.SINUSOIDAL_STRAFE,
            initial_pos_px=center,
            amplitude_x_px=amp_x,
            amplitude_y_px=amp_y,
            frequency_hz=freq_hz,
            phase_rad=phase_rad,
            radius_px=radius,
            spawn_time_ns=spawn_ns,
            despawn_time_ns=despawn_ns,
        )
        return self.add_target(spec)

    def add_circular_target(
        self,
        target_id: int,
        center: Vec2f,
        orbit_radius: float,
        freq_hz: float,
        phase_rad: float = 0.0,
        target_radius: float = 20.0,
        spawn_ns: int = 0,
        despawn_ns: int = 0x7FFFFFFFFFFFFFFF,
    ) -> ScenarioBuilder:
        spec = TargetSpec(
            target_id=target_id,
            trajectory_type=TrajectoryType.CIRCULAR_ORBIT,
            initial_pos_px=center,
            amplitude_x_px=orbit_radius,
            frequency_hz=freq_hz,
            phase_rad=phase_rad,
            radius_px=target_radius,
            spawn_time_ns=spawn_ns,
            despawn_time_ns=despawn_ns,
        )
        return self.add_target(spec)

    def add_sudden_cut_target(
        self,
        target_id: int,
        pos: Vec2f,
        vel: Vec2f,
        cut_interval_ns: int,
        cut_angle_rad: float = 3.14159265358979323846,
        radius: float = 20.0,
        spawn_ns: int = 0,
        despawn_ns: int = 0x7FFFFFFFFFFFFFFF,
    ) -> ScenarioBuilder:
        spec = TargetSpec(
            target_id=target_id,
            trajectory_type=TrajectoryType.SUDDEN_CUT,
            initial_pos_px=pos,
            velocity_px_s=vel,
            cut_interval_ns=cut_interval_ns,
            cut_angle_rad=cut_angle_rad,
            radius_px=radius,
            spawn_time_ns=spawn_ns,
            despawn_time_ns=despawn_ns,
        )
        return self.add_target(spec)

    def add_temporal_occlusion(
        self,
        target_id: int,
        start_ns: int,
        end_ns: int,
        mode: OcclusionMode = OcclusionMode.DROP,
    ) -> ScenarioBuilder:
        self._scenario.temporal_occlusions.append(
            TemporalOcclusion(
                target_id=target_id,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                mode=mode,
            )
        )
        return self

    def add_rect_occluder(
        self,
        bounds: BBoxf,
        start_ns: int = 0,
        end_ns: int = 0x7FFFFFFFFFFFFFFF,
        mode: OcclusionMode = OcclusionMode.DROP,
    ) -> ScenarioBuilder:
        self._scenario.rect_occluders.append(
            RectOccluder(
                bounds_px=bounds,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                mode=mode,
            )
        )
        return self

    def add_circle_occluder(
        self,
        center: Vec2f,
        radius: float,
        start_ns: int = 0,
        end_ns: int = 0x7FFFFFFFFFFFFFFF,
        mode: OcclusionMode = OcclusionMode.DROP,
    ) -> ScenarioBuilder:
        self._scenario.circle_occluders.append(
            CircleOccluder(
                center_px=center,
                radius_px=radius,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                mode=mode,
            )
        )
        return self

    def build(self) -> Scenario:
        return self._scenario

    @staticmethod
    def make_gridshot_preset(seed: int = 42) -> Scenario:
        return (
            ScenarioBuilder("gridshot_preset")
            .with_seed(seed)
            .with_cadence_hz(144.0)
            .with_duration_seconds(60.0)
            .add_stationary_target(1, Vec2f(800.0, 450.0), 25.0)
            .add_stationary_target(2, Vec2f(960.0, 540.0), 25.0)
            .add_stationary_target(3, Vec2f(1120.0, 630.0), 25.0)
            .build()
        )

    @staticmethod
    def make_strafe_track_preset(seed: int = 42) -> Scenario:
        return (
            ScenarioBuilder("strafe_track_preset")
            .with_seed(seed)
            .with_cadence_hz(144.0)
            .with_duration_seconds(30.0)
            .add_sinusoidal_target(1, Vec2f(960.0, 540.0), 350.0, 0.0, 1.2, 0.0, 22.0)
            .add_sudden_cut_target(2, Vec2f(600.0, 400.0), Vec2f(400.0, 0.0), 1_500_000_000, 3.14159265, 20.0)
            .build()
        )

    @staticmethod
    def make_occlusion_preset(seed: int = 42) -> Scenario:
        return (
            ScenarioBuilder("occlusion_preset")
            .with_seed(seed)
            .with_cadence_hz(144.0)
            .with_duration_seconds(20.0)
            .add_linear_target(1, Vec2f(400.0, 540.0), Vec2f(300.0, 0.0), 24.0, BoundaryBehavior.BOUNCE)
            .add_rect_occluder(BBoxf(850.0, 400.0, 1070.0, 680.0), 0, 0x7FFFFFFFFFFFFFFF, OcclusionMode.PREDICTED)
            .add_temporal_occlusion(1, 5_000_000_000, 7_000_000_000, OcclusionMode.DROP)
            .build()
        )

    @staticmethod
    def make_density_preset(seed: int = 42, target_count: int = 64) -> Scenario:
        b = ScenarioBuilder("density_preset").with_seed(seed).with_cadence_hz(144.0).with_duration_seconds(10.0)
        count = min(target_count, 64)
        for i in range(count):
            # Mirrors make_density_preset in include/aim/sim/scenario_builder.hpp, which
            # evaluates in binary32 and rounds after every operation.
            angle = f32(f32(i) * f32(f32(6.2831853) / f32(count)))
            radius = f32(200.0 + f32(f32(i % 4) * 80.0))
            cx = f32(960.0 + f32(radius * cos_f32(angle)))
            cy = f32(540.0 + f32(radius * sin_f32(angle)))
            vx = f32(-200.0 * sin_f32(angle))
            vy = f32(200.0 * cos_f32(angle))
            b.add_linear_target(i + 1, Vec2f(cx, cy), Vec2f(vx, vy), 18.0, BoundaryBehavior.BOUNCE)
        return b.build()
