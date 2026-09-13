"""tools/sim/__init__.py
Deterministic Target Simulator and Replay Clock package.
"""

from tools.sim.types import (
    TrajectoryType,
    BoundaryBehavior,
    OcclusionMode,
    TargetSpec,
    ScenarioConfig,
    TrajectoryState,
    RectOccluder,
    CircleOccluder,
    TemporalOcclusion,
    NoiseConfig,
    pixel_to_normalized,
    normalized_to_pixel,
    make_bounding_box,
    compute_bbox_intersection_area,
)
from tools.sim.prng import DeterministicRng
from tools.sim.replay_clock import ReplayClock
from tools.sim.trajectory import evaluate_trajectory, apply_1d_boundary
from tools.sim.scenario import Scenario, ScenarioBuilder
from tools.sim.target_simulator import TargetSimulator

__all__ = [
    "TrajectoryType",
    "BoundaryBehavior",
    "OcclusionMode",
    "TargetSpec",
    "ScenarioConfig",
    "TrajectoryState",
    "RectOccluder",
    "CircleOccluder",
    "TemporalOcclusion",
    "NoiseConfig",
    "pixel_to_normalized",
    "normalized_to_pixel",
    "make_bounding_box",
    "compute_bbox_intersection_area",
    "DeterministicRng",
    "ReplayClock",
    "evaluate_trajectory",
    "apply_1d_boundary",
    "Scenario",
    "ScenarioBuilder",
    "TargetSimulator",
]
