"""Policy Manifest Validator and Builder (Milestone M5-03)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "schemas"
    / "manifest"
    / "policy_manifest.schema.json"
)


@dataclass
class PolicyContracts:
    input_schema: str = "bus::TrackedTargetBatch"
    output_schema: str = "bus::AimIntent"
    coordinate_system: str = "pixel_top_left"


@dataclass
class PolicyHyperparameters:
    reward_weight_distance: float = 1.0
    reward_weight_confidence: float = 2.0
    switch_hysteresis_penalty: float = 1.5
    fire_authorization_threshold_px: float = 8.0
    receding_horizon_steps: int = 4


@dataclass
class PolicySafetyBounds:
    max_angular_speed_deg_s: float = 720.0
    max_pixel_distance_px: float = 1920.0
    require_confirmed_track: bool = True
    min_engagement_confidence: float = 0.50


@dataclass
class PolicyManifestData:
    schema_version: int = 1
    policy_id: str = "deterministic_utility_v1"
    policy_type: str = "deterministic_utility"
    version: str = "1.0.0"
    author: str = "OpenPrism Core Team"
    license: str = "PolyForm-Noncommercial-1.0.0"
    description: str = "Domain-neutral utility aim policy with switch hysteresis"
    contracts: PolicyContracts = field(default_factory=PolicyContracts)
    hyperparameters: PolicyHyperparameters = field(
        default_factory=PolicyHyperparameters
    )
    safety_bounds: PolicySafetyBounds = field(default_factory=PolicySafetyBounds)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_policy_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema: Dict[str, Any] = json.load(f)
        return schema


def validate_policy_manifest(
    data: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> bool:
    if schema is None:
        schema = load_policy_schema()
    jsonschema.validate(instance=data, schema=schema)
    return True
