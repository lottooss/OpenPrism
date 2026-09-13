# tests/test_plugin_manifest.py
"""Tests for OpenPrism Plugin Manifest JSON Schema and Parser."""

import copy
import json
import tempfile
from pathlib import Path

import jsonschema
import pytest

from tools.plugin.manifest import (
    PluginManifest,
    compute_file_sha256,
    get_schema,
    load_manifest_file,
    validate_manifest_dict,
    verify_binary_sha256,
)
from tools.plugin.contract_validator import PluginContractValidator

SAMPLE_VALID_MANIFEST = {
    "schema_version": 1,
    "name": "yolo11n_perception",
    "version": "1.0.0",
    "abi_version": 65536,
    "kind": "perception",
    "author": "Aim Research Team",
    "license": "MIT",
    "description": "YOLO11n TensorRT perception plugin for high-speed target detection",
    "entry_point": "yolo11n_perception.dll",
    "checksum": {
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "size_bytes": 1048576,
    },
    "contracts": {
        "input_schema": {
            "identifier": "AFR1",
            "major_min": 1,
            "major_max": 1,
        },
        "output_schema": {
            "identifier": "AOB1",
            "major_min": 1,
            "major_max": 1,
        },
    },
    "capabilities": [
        "bounding_box",
        "gpu_direct",
        "fp16_inference",
        "confidence_calibrated",
    ],
    "models": [
        {
            "name": "yolo11n_trt",
            "path": "models/yolo11n.engine",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "framework": "tensorrt",
            "precision": "fp16",
            "input_dimensions": [1, 3, 384, 640],
        }
    ],
    "resource_requirements": {
        "max_latency_budget_ns": 3000000,
        "max_output_buffer_bytes": 65536,
        "gpu_vram_bytes": 268435456,
        "requires_cuda": True,
    },
}


def test_schema_loads_successfully():
    schema = get_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "AimPluginManifest"
    assert schema["additionalProperties"] is False


def test_valid_manifest_passes_validation():
    validate_manifest_dict(SAMPLE_VALID_MANIFEST)
    manifest = PluginManifest(SAMPLE_VALID_MANIFEST)
    assert manifest.name == "yolo11n_perception"
    assert manifest.kind == "perception"
    assert manifest.abi_version == 65536
    assert manifest.entry_point == "yolo11n_perception.dll"
    assert manifest.input_schema_identifier == "AFR1"
    assert manifest.output_schema_identifier == "AOB1"
    assert "bounding_box" in manifest.capabilities


def test_rejects_missing_required_fields():
    required_fields = [
        "schema_version",
        "name",
        "version",
        "abi_version",
        "kind",
        "author",
        "license",
        "entry_point",
        "checksum",
        "contracts",
        "capabilities",
    ]

    for field in required_fields:
        bad_manifest = copy.deepcopy(SAMPLE_VALID_MANIFEST)
        del bad_manifest[field]
        with pytest.raises(jsonschema.ValidationError):
            validate_manifest_dict(bad_manifest)


def test_rejects_unknown_properties():
    bad_manifest = copy.deepcopy(SAMPLE_VALID_MANIFEST)
    bad_manifest["unknown_field"] = "malicious_payload"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_dict(bad_manifest)


def test_rejects_invalid_kind():
    bad_manifest = copy.deepcopy(SAMPLE_VALID_MANIFEST)
    bad_manifest["kind"] = "magic_wand"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_dict(bad_manifest)


def test_rejects_invalid_fourcc_identifier():
    bad_manifest = copy.deepcopy(SAMPLE_VALID_MANIFEST)
    bad_manifest["contracts"]["input_schema"]["identifier"] = "INVALID"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_dict(bad_manifest)


def test_rejects_invalid_checksum_format():
    bad_manifest = copy.deepcopy(SAMPLE_VALID_MANIFEST)
    bad_manifest["checksum"]["sha256"] = "not_a_valid_hex_string"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_dict(bad_manifest)


def test_sha256_binary_verification():
    with tempfile.NamedTemporaryFile(suffix=".dll", delete=False) as tmp:
        tmp.write(b"MOCK_DLL_BINARY_CONTENT_FOR_HASH_TEST")
        tmp_path = Path(tmp.name)

    try:
        expected_hash = compute_file_sha256(tmp_path)
        manifest_data = copy.deepcopy(SAMPLE_VALID_MANIFEST)
        manifest_data["checksum"]["sha256"] = expected_hash
        manifest_data["checksum"]["size_bytes"] = tmp_path.stat().st_size

        assert verify_binary_sha256(manifest_data, tmp_path) is True

        manifest_data["checksum"]["size_bytes"] += 1
        assert verify_binary_sha256(manifest_data, tmp_path) is False
        manifest_data["checksum"]["size_bytes"] -= 1

        manifest_data["checksum"]["size_bytes"] = 0
        assert verify_binary_sha256(manifest_data, tmp_path) is False
        manifest_data["checksum"]["size_bytes"] = tmp_path.stat().st_size

        manifest_data["checksum"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        assert verify_binary_sha256(manifest_data, tmp_path) is False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_manifest_file_json_and_yaml():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "plugin.json"
        yaml_path = Path(tmp_dir) / "plugin.yaml"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_VALID_MANIFEST, f)

        import yaml

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_VALID_MANIFEST, f)

        data_json = load_manifest_file(json_path)
        data_yaml = load_manifest_file(yaml_path)

        assert data_json["name"] == "yolo11n_perception"
        assert data_yaml["name"] == "yolo11n_perception"


def test_checksum_mismatch_fails_before_library_load(monkeypatch, tmp_path):
    binary_path = tmp_path / "untrusted.dll"
    binary_path.write_bytes(b"not a native library")

    manifest_data = copy.deepcopy(SAMPLE_VALID_MANIFEST)
    manifest_data["entry_point"] = binary_path.name
    manifest_data["checksum"]["size_bytes"] = binary_path.stat().st_size
    manifest_data["checksum"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "plugin.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    loader_called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("unverified binary was loaded")

    monkeypatch.setattr("tools.plugin.contract_validator.PluginWrapper", fail_if_loaded)
    result = PluginContractValidator.validate_plugin(manifest_path, binary_path)

    assert result.is_valid is False
    assert result.checksum_valid is False
    assert loader_called is False
