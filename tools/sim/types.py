"""tools/sim/types.py
Data structures, enums, and mathematical coordinate helpers for target simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from tools.sim.prng import f32
from tools.bus.schema_bindings import (
    BBoxf,
    Vec2f,
)


class TrajectoryType(IntEnum):
    STATIONARY = 0
    LINEAR_CV = 1
    LINEAR_CA = 2
    SINUSOIDAL_STRAFE = 3
    CIRCULAR_ORBIT = 4
    SUDDEN_CUT = 5
    BOUNCING_BOX = 6
    PIECEWISE = 7


class BoundaryBehavior(IntEnum):
    NONE = 0
    BOUNCE = 1
    WRAP = 2
    CLAMP = 3
    DESPAWN = 4


class OcclusionMode(IntEnum):
    DROP = 0
    PREDICTED = 1
    PARTIAL = 2


@dataclass(slots=True)
class RectOccluder:
    bounds_px: BBoxf
    start_time_ns: int = 0
    end_time_ns: int = 0x7FFFFFFFFFFFFFFF
    mode: OcclusionMode = OcclusionMode.DROP


@dataclass(slots=True)
class CircleOccluder:
    center_px: Vec2f
    radius_px: float = 0.0
    start_time_ns: int = 0
    end_time_ns: int = 0x7FFFFFFFFFFFFFFF
    mode: OcclusionMode = OcclusionMode.DROP


@dataclass(slots=True)
class TemporalOcclusion:
    target_id: int
    start_time_ns: int = 0
    end_time_ns: int = 0x7FFFFFFFFFFFFFFF
    mode: OcclusionMode = OcclusionMode.DROP


@dataclass(slots=True)
class NoiseConfig:
    position_stddev_px: float = 0.0
    velocity_stddev_px_per_s: float = 0.0
    radius_stddev_px: float = 0.0
    base_confidence: float = 0.98
    confidence_jitter: float = 0.0
    dropout_probability: float = 0.0


@dataclass(slots=True)
class TargetSpec:
    target_id: int = 1
    semantic_id: int = 0
    trajectory_type: TrajectoryType = TrajectoryType.STATIONARY
    boundary: BoundaryBehavior = BoundaryBehavior.BOUNCE
    initial_pos_px: Vec2f = field(default_factory=lambda: Vec2f(960.0, 540.0))
    velocity_px_s: Vec2f = field(default_factory=lambda: Vec2f(0.0, 0.0))
    accel_px_s2: Vec2f = field(default_factory=lambda: Vec2f(0.0, 0.0))
    radius_px: float = 20.0
    target_value: float = 1.0

    # Harmonic / Sinusoidal / Circular parameters
    amplitude_x_px: float = 0.0
    amplitude_y_px: float = 0.0
    frequency_hz: float = 1.0
    phase_rad: float = 0.0

    # Sudden Cut parameters
    cut_interval_ns: int = 0
    cut_angle_rad: float = 3.14159265358979323846

    # Target Lifecycle
    spawn_time_ns: int = 0
    despawn_time_ns: int = 0x7FFFFFFFFFFFFFFF

    # Noise override
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass(slots=True)
class ScenarioConfig:
    name: str = "default"
    seed: int = 133742
    cadence_hz: float = 144.0
    start_time_ns: int = 1_000_000_000
    duration_ns: int = 30_000_000_000
    source_width: int = 1920
    source_height: int = 1080
    source_id: int = 1
    pipeline_run_id: int = 100
    synthetic_perception_latency_ns: int = 2_500_000
    default_noise: NoiseConfig = field(default_factory=NoiseConfig)
    emit_predicted_when_occluded: bool = True


@dataclass(slots=True)
class TrajectoryState:
    pos_px: Vec2f = field(default_factory=lambda: Vec2f(0.0, 0.0))
    vel_px_per_s: Vec2f = field(default_factory=lambda: Vec2f(0.0, 0.0))
    accel_px_s2: Vec2f = field(default_factory=lambda: Vec2f(0.0, 0.0))
    radius_px: float = 20.0
    is_active: bool = True
    is_despawned: bool = False


# ==============================================================================
# Coordinate Transformation Math Helpers
# ==============================================================================

def pixel_to_normalized(px: Vec2f, width: int = 1920, height: int = 1080) -> Vec2f:
    """Converts top-left pixel coordinates (1080p) to normalized center [-1.0, 1.0]."""
    half_w = f32(f32(width) * 0.5)
    half_h = f32(f32(height) * 0.5)
    norm_x = f32(f32(f32(px.x) - half_w) / half_w) if half_w > 0 else 0.0
    norm_y = f32(f32(f32(px.y) - half_h) / half_h) if half_h > 0 else 0.0
    return Vec2f(norm_x, norm_y)


def normalized_to_pixel(norm: Vec2f, width: int = 1920, height: int = 1080) -> Vec2f:
    """Converts normalized center [-1.0, 1.0] to top-left pixel coordinates."""
    half_w = f32(f32(width) * 0.5)
    half_h = f32(f32(height) * 0.5)
    return Vec2f(f32(f32(f32(norm.x) + 1.0) * half_w), f32(f32(f32(norm.y) + 1.0) * half_h))


def make_bounding_box(center: Vec2f, radius: float) -> BBoxf:
    """Computes axis-aligned bounding box from center and radius."""
    cx, cy, r = f32(center.x), f32(center.y), f32(radius)
    return BBoxf(
        left=f32(cx - r),
        top=f32(cy - r),
        right=f32(cx + r),
        bottom=f32(cy + r),
    )


def compute_bbox_intersection_area(a: BBoxf, b: BBoxf) -> float:
    """Computes 2D intersection area of two bounding boxes."""
    inter_left = max(f32(a.left), f32(b.left))
    inter_top = max(f32(a.top), f32(b.top))
    inter_right = min(f32(a.right), f32(b.right))
    inter_bottom = min(f32(a.bottom), f32(b.bottom))

    inter_w = max(0.0, f32(inter_right - inter_left))
    inter_h = max(0.0, f32(inter_bottom - inter_top))
    return f32(inter_w * inter_h)
