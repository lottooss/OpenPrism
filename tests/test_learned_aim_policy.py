# tests/test_learned_aim_policy.py
"""Unit and regression tests for Replaceable Learned AimPolicy (Milestone M9-04)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import jsonschema
import yaml

from tools.policy.learned_aim_policy import (
    LearnedAimPolicy,
    LearnedPolicyWeights,
    PolicyTrackedTarget,
    default_policy_weights,
    train_numpy_imitation_policy,
)
from tools.policy.policy_manifest import validate_policy_manifest


def test_policy_manifests_conform_to_schema() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    det_manifest_path = repo_root / "manifests" / "policy" / "deterministic_utility_manifest.json"
    learned_manifest_path = repo_root / "manifests" / "policy" / "learned_imitation_manifest.json"

    assert det_manifest_path.is_file(), f"Missing {det_manifest_path}"
    assert learned_manifest_path.is_file(), f"Missing {learned_manifest_path}"

    with open(det_manifest_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)
    with open(learned_manifest_path, "r", encoding="utf-8") as f:
        learned_data = json.load(f)

    assert validate_policy_manifest(det_data) is True
    assert validate_policy_manifest(learned_data) is True


def test_learned_policy_config_conforms_to_schema() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    schema_path = repo_root / "schemas" / "config" / "policy.schema.json"
    config_path = repo_root / "configs" / "policy" / "learned_imitation.yaml"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    assert "policy" in config_data
    jsonschema.validate(instance=config_data["policy"], schema=schema)
    assert config_data["policy"]["plugin"] == "learned_imitation"


def test_safety_bounds_cannot_be_bypassed() -> None:
    policy = LearnedAimPolicy()

    # Case 1: Target is outside fire authorization distance (> 8.0 px)
    t1 = PolicyTrackedTarget(track_id=1, x=980.0, y=540.0, confidence=0.99, is_confirmed=True)  # 20 px away
    intent = policy.choose(960.0, 540.0, [t1])
    assert intent is not None
    assert intent["target_track_id"] == 1
    assert intent["error_distance_px"] == 20.0
    assert not intent["authorize_fire"], "Cannot authorize fire outside distance threshold"

    # Case 2: Target is within 8px distance but unconfirmed
    t2 = PolicyTrackedTarget(track_id=2, x=964.0, y=540.0, confidence=0.99, is_confirmed=False)  # 4 px away
    intent2 = policy.choose(960.0, 540.0, [t2])
    assert intent2 is not None
    assert not intent2["authorize_fire"], "Cannot authorize fire on unconfirmed target"

    # Case 3: Target is within 8px distance, confirmed, but confidence < 0.50
    t3 = PolicyTrackedTarget(track_id=3, x=962.0, y=540.0, confidence=0.45, is_confirmed=True)  # 2 px away
    intent3 = policy.choose(960.0, 540.0, [t3])
    # Low-confidence target must be completely filtered out
    assert intent3 is None, "Cannot engage target with confidence < min_engagement_confidence"


def test_zero_perception_coupling() -> None:
    policy = LearnedAimPolicy()
    # Passing an arbitrary non-canonical object or raw image array must raise error
    with pytest.raises((TypeError, AttributeError)):
        # Pass invalid raw matrix/object
        policy.choose(960.0, 540.0, ["raw_opencv_mat_image"])  # type: ignore


def test_weights_serialization_roundtrip_parity() -> None:
    weights = default_policy_weights(seed=123)
    data = weights.to_dict()
    restored = LearnedPolicyWeights.from_dict(data)

    policy_orig = LearnedAimPolicy(weights=weights)
    policy_restored = LearnedAimPolicy(weights=restored)

    t1 = PolicyTrackedTarget(track_id=10, x=940.0, y=520.0, confidence=0.90, target_value=1.0)
    t2 = PolicyTrackedTarget(track_id=20, x=1100.0, y=600.0, confidence=0.80, target_value=0.8)

    res1 = policy_orig.choose(960.0, 540.0, [t1, t2])
    res2 = policy_restored.choose(960.0, 540.0, [t1, t2])

    assert res1 == res2


def test_cross_domain_evaluation_parity() -> None:
    # Train short policy model
    weights = train_numpy_imitation_policy(num_samples=1000, epochs=8, seed=42)
    policy = LearnedAimPolicy(weights=weights)

    # Domain A (circular target)
    tA = PolicyTrackedTarget(track_id=1, x=965.0, y=540.0, confidence=0.95, covariance_trace=8.0)
    resA = policy.choose(960.0, 540.0, [tA])
    assert resA is not None
    assert resA["target_track_id"] == 1

    # Domain B (humanoid target, anisotropic covariance)
    tB = PolicyTrackedTarget(track_id=2, x=965.0, y=540.0, confidence=0.95, covariance_trace=20.0)
    resB = policy.choose(960.0, 540.0, [tB])
    assert resB is not None
    assert resB["target_track_id"] == 2
