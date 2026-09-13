"""Unit tests for Policy Manifest Validation (Milestone M5-03)."""

import jsonschema
import pytest
from tools.policy.policy_manifest import (
    PolicyManifestData,
    validate_policy_manifest,
)


def test_valid_policy_manifest() -> None:
    manifest = PolicyManifestData()
    assert validate_policy_manifest(manifest.to_dict()) is True


def test_invalid_policy_type_fails() -> None:
    manifest = PolicyManifestData()
    data = manifest.to_dict()
    data["policy_type"] = "unauthorized_cheat_policy"

    with pytest.raises(jsonschema.ValidationError):
        validate_policy_manifest(data)


def test_invalid_contracts_fails() -> None:
    manifest = PolicyManifestData()
    data = manifest.to_dict()
    data["contracts"]["input_schema"] = "raw_opencv_mat"  # Must fail closed

    with pytest.raises(jsonschema.ValidationError):
        validate_policy_manifest(data)


def test_safety_bounds_enforcement() -> None:
    manifest = PolicyManifestData()
    data = manifest.to_dict()
    data["safety_bounds"]["max_angular_speed_deg_s"] = 50000.0  # Exceeds max 2000.0

    with pytest.raises(jsonschema.ValidationError):
        validate_policy_manifest(data)
