"""Layered YAML configuration loading, deep merging, and strict schema validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from tools.config.hashing import compute_config_sha256


class ConfigValidationError(Exception):
    """Raised when a configuration fails JSON schema or semantic validation."""

    def __init__(self, message: str, path: str | None = None, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.details = details or []


def deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively deep merge source into target.

    - Nested dictionaries are merged recursively.
    - Lists / sequences in source atomically replace those in target.
    - Scalar values in source overwrite those in target.
    """
    result = copy.deepcopy(target)
    for key, value in source.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif isinstance(value, list):
            result[key] = copy.deepcopy(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class LayeredConfigLoader:
    """Deterministic 7-tier layered configuration loader with JSON Schema validation."""

    DEFAULT_SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas" / "config"

    def __init__(self, schemas_dir: Path | None = None) -> None:
        self.schemas_dir = schemas_dir or self.DEFAULT_SCHEMAS_DIR

    @staticmethod
    def parse_yaml_string(content: str) -> dict[str, Any]:
        """Parse a YAML string into a Python dictionary."""
        if not content.strip():
            return {}
        try:
            data = yaml.safe_load(content)
        except Exception as e:
            raise ConfigValidationError(f"Invalid YAML syntax: {e}") from e
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigValidationError(f"Expected YAML mapping at root, got {type(data).__name__}")
        return data

    @classmethod
    def parse_yaml_file(cls, file_path: str | Path) -> dict[str, Any]:
        """Read and parse a YAML file from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration layer file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ConfigValidationError(f"Failed to read configuration file '{path}': {e}") from e
        return cls.parse_yaml_string(text)

    def load_layered(
        self,
        layer_paths: list[str | Path],
        validate: bool = True,
        clamp: bool = True,
    ) -> tuple[dict[str, Any], str, list[str]]:
        """Load and merge an ordered sequence of configuration layer files.

        Returns:
            (resolved_config_dict, config_sha256, normalized_source_layer_paths)
        """
        merged: dict[str, Any] = {}
        source_layers: list[str] = []

        for layer_path in layer_paths:
            path = Path(layer_path)
            layer_dict = self.parse_yaml_file(path)
            merged = deep_merge(merged, layer_dict)
            source_layers.append(path.as_posix())

        if clamp:
            from tools.config.hot_reload import clamp_tuning_parameters
            merged, _ = clamp_tuning_parameters(merged)

        if validate:
            self.validate(merged)

        config_sha256 = compute_config_sha256(merged)
        return merged, config_sha256, source_layers

    def load_layered_strings(
        self,
        yaml_strings: list[str],
        layer_names: list[str] | None = None,
        validate: bool = True,
        clamp: bool = True,
    ) -> tuple[dict[str, Any], str, list[str]]:
        """Load and merge an ordered sequence of YAML strings."""
        merged: dict[str, Any] = {}
        source_layers: list[str] = []

        for idx, y_str in enumerate(yaml_strings):
            name = layer_names[idx] if layer_names and idx < len(layer_names) else f"layer_{idx}"
            layer_dict = self.parse_yaml_string(y_str)
            merged = deep_merge(merged, layer_dict)
            source_layers.append(name)

        if clamp:
            from tools.config.hot_reload import clamp_tuning_parameters
            merged, _ = clamp_tuning_parameters(merged)

        if validate:
            self.validate(merged)

        config_sha256 = compute_config_sha256(merged)
        return merged, config_sha256, source_layers

    def validate(self, config_dict: dict[str, Any]) -> None:
        """Validate a configuration dictionary against root.schema.json and sub-schemas."""
        root_schema_path = self.schemas_dir / "root.schema.json"
        if not root_schema_path.is_file():
            raise FileNotFoundError(f"Root schema file not found at: {root_schema_path}")

        try:
            root_schema = json.loads(root_schema_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ConfigValidationError(f"Failed to parse root schema: {e}") from e

        # Load all domain schemas into store for reference resolution
        schema_store: dict[str, Any] = {}
        for schema_file in self.schemas_dir.glob("*.schema.json"):
            try:
                schema_content = json.loads(schema_file.read_text(encoding="utf-8"))
                schema_store[schema_file.name] = schema_content
                if "$id" in schema_content:
                    schema_store[schema_content["$id"]] = schema_content
            except Exception as e:
                raise ConfigValidationError(f"Failed to parse sub-schema '{schema_file.name}': {e}") from e

        # Set up validator with schema resolver or referencing registry
        try:
            # Modern referencing library approach if available
            import referencing
            from referencing.jsonschema import DRAFT202012

            resources = []
            for k, sc in schema_store.items():
                if isinstance(sc, dict) and "$id" in sc:
                    resources.append((sc["$id"], DRAFT202012.create_resource(sc)))
                resources.append((k, DRAFT202012.create_resource(sc)))
            registry = referencing.Registry().with_resources(resources)
            validator = jsonschema.Draft202012Validator(root_schema, registry=registry)
        except Exception:
            # Fallback to RefResolver
            from jsonschema import RefResolver
            base_uri = root_schema_path.as_uri()
            resolver = RefResolver(base_uri=base_uri, referrer=root_schema, store=schema_store)
            validator = jsonschema.Draft202012Validator(root_schema, resolver=resolver)

        errors = list(validator.iter_errors(config_dict))
        if errors:
            error_messages = []
            for err in errors:
                path = ".".join(str(p) for p in err.absolute_path) or "<root>"
                error_messages.append(f"At '{path}': {err.message}")
            summary = f"Configuration validation failed with {len(errors)} error(s):\n" + "\n".join(f" - {msg}" for msg in error_messages)
            first_path = ".".join(str(p) for p in errors[0].absolute_path) or "<root>"
            raise ConfigValidationError(summary, path=first_path, details=error_messages)
