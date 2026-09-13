# tools/plugin/manifest.py
"""Plugin Manifest Loading and Schema Validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

import jsonschema
import yaml

MANIFEST_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "schemas"
    / "manifest"
    / "plugin_manifest.schema.json"
)

_CACHED_SCHEMA: Optional[Dict[str, Any]] = None
_CACHED_VALIDATOR: Optional[jsonschema.Draft202012Validator] = None


def get_schema() -> Dict[str, Any]:
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is None:
        if not MANIFEST_SCHEMA_PATH.is_file():
            raise FileNotFoundError(f"Schema file not found: {MANIFEST_SCHEMA_PATH}")
        with open(MANIFEST_SCHEMA_PATH, "r", encoding="utf-8") as f:
            _CACHED_SCHEMA = json.load(f)
    return _CACHED_SCHEMA


def get_validator() -> jsonschema.Draft202012Validator:
    global _CACHED_VALIDATOR
    if _CACHED_VALIDATOR is None:
        schema = get_schema()
        _CACHED_VALIDATOR = jsonschema.Draft202012Validator(schema)
    return _CACHED_VALIDATOR


def validate_manifest_dict(manifest_dict: Dict[str, Any]) -> None:
    """Validate manifest dictionary against JSON Schema Draft 2020-12.

    Raises jsonschema.ValidationError on failure.
    """
    validator = get_validator()
    validator.validate(manifest_dict)


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_binary_sha256(manifest_dict: Dict[str, Any], binary_path: Path | str) -> bool:
    """Verify SHA-256 and declared byte size before a binary is loaded."""
    path = Path(binary_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    checksum = manifest_dict.get("checksum")
    if not isinstance(checksum, dict):
        return False
    expected_hash_value = checksum.get("sha256")
    if not isinstance(expected_hash_value, str) or not expected_hash_value:
        return False
    expected_hash = expected_hash_value.lower()
    expected_size = checksum.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size < 1 or path.stat().st_size != expected_size:
        return False
    actual_hash = compute_file_sha256(path).lower()
    return actual_hash == expected_hash


def load_manifest_file(manifest_path: Path | str) -> Dict[str, Any]:
    """Load and validate manifest from YAML or JSON file."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be an object/dict in {path}")

    validate_manifest_dict(data)
    return cast(Dict[str, Any], data)


class PluginManifest:
    """Strongly-typed wrapper around manifest data."""

    def __init__(self, data: Dict[str, Any]):
        validate_manifest_dict(data)
        self.data = data

    @classmethod
    def from_file(cls, manifest_path: Path | str) -> PluginManifest:
        return cls(load_manifest_file(manifest_path))

    @property
    def schema_version(self) -> int:
        return cast(int, self.data["schema_version"])

    @property
    def name(self) -> str:
        return cast(str, self.data["name"])

    @property
    def version(self) -> str:
        return cast(str, self.data["version"])

    @property
    def abi_version(self) -> int:
        return cast(int, self.data["abi_version"])

    @property
    def kind(self) -> str:
        return cast(str, self.data["kind"])

    @property
    def author(self) -> str:
        return cast(str, self.data["author"])

    @property
    def license(self) -> str:
        return cast(str, self.data["license"])

    @property
    def entry_point(self) -> str:
        return cast(str, self.data["entry_point"])

    @property
    def sha256(self) -> str:
        return cast(str, self.data["checksum"]["sha256"])

    @property
    def input_schema_identifier(self) -> str:
        return cast(str, self.data["contracts"]["input_schema"]["identifier"])

    @property
    def output_schema_identifier(self) -> str:
        return cast(str, self.data["contracts"]["output_schema"]["identifier"])

    @property
    def capabilities(self) -> list[str]:
        return list(self.data["capabilities"])

    def verify_binary(self, binary_path: Path | str) -> bool:
        return verify_binary_sha256(self.data, binary_path)
