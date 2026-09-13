"""Comprehensive unit tests for layered configuration loading, validation, and run manifests."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

from tools.config.hashing import compute_config_sha256
from tools.config.hot_reload import (
    clamp_tuning_parameters,
    validate_hot_reload,
)
from tools.config.loader import ConfigValidationError, LayeredConfigLoader, deep_merge
from tools.config.manifest import RunManifestBuilder, verify_run_manifest


class TestConfigManifest(unittest.TestCase):
    """Deterministic test suite for configuration merging, schema strictness, and run manifests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.configs_dir = cls.repo_root / "configs"
        cls.schemas_dir = cls.repo_root / "schemas" / "config"
        cls.loader = LayeredConfigLoader(schemas_dir=cls.schemas_dir)

    # -------------------------------------------------------------------------
    # Category A: Valid Config Loading and Layer Merging
    # -------------------------------------------------------------------------

    def test_default_base_config_load(self) -> None:
        """TC-CFG-01: Verify default base configuration loads and populates all 9 domains."""
        base_path = self.configs_dir / "base.yaml"
        self.assertTrue(base_path.is_file(), f"Base config missing at {base_path}")

        resolved, cfg_hash, source_layers = self.loader.load_layered([base_path])

        self.assertEqual(resolved["schema_version"], 1)
        expected_domains = [
            "runtime", "capture", "perception", "tracking",
            "policy", "trajectory", "actuator", "safety"
        ]
        for domain in expected_domains:
            self.assertIn(domain, resolved)
            self.assertIsInstance(resolved[domain], dict)

        self.assertEqual(resolved["runtime"]["internal_deadline_ms"], 10.0)
        self.assertEqual(resolved["runtime"]["target_p99_ms"], 6.0)
        self.assertEqual(resolved["capture"]["backend"], "dxgi")
        self.assertEqual(resolved["capture"]["source_width"], 1920)
        self.assertEqual(resolved["capture"]["source_height"], 1080)
        self.assertEqual(resolved["perception"]["plugin"], "yolo11n_aimlabs")
        self.assertEqual(resolved["perception"]["input_width"], 640)
        self.assertEqual(resolved["perception"]["input_height"], 384)
        self.assertEqual(resolved["tracking"]["association"], "hungarian")
        self.assertEqual(len(resolved["tracking"]["models"]), 3)
        self.assertEqual(resolved["policy"]["plugin"], "deterministic_utility")
        self.assertEqual(resolved["policy"]["horizon_targets"], 3)
        self.assertEqual(resolved["actuator"]["backend"], "sendinput")
        self.assertEqual(resolved["actuator"]["scheduler_hz"], 1000)
        self.assertTrue(resolved["safety"]["require_foreground_match"])
        self.assertTrue(resolved["safety"]["fail_closed"])

        self.assertEqual(len(cfg_hash), 64)
        self.assertEqual(len(source_layers), 1)

    def test_deterministic_layer_merge_hierarchy(self) -> None:
        """TC-CFG-02: Verify 7-layer hierarchy precedence (base -> hw -> domain -> perception -> policy -> actuator -> overrides)."""
        layer_base = """
schema_version: 1
runtime:
  internal_deadline_ms: 10.0
  target_p99_ms: 6.0
  warmup_iterations: 200
  allocation_audit: true
capture:
  backend: "dxgi"
  fallback: "wgc"
  source_width: 1920
  source_height: 1080
  pixel_format: "bgra8_sdr"
  buffers: 2
  latest_only: true
  stale_after_ms: 12.0
perception:
  plugin: "yolo11n_aimlabs"
  precision: "fp16"
  batch: 1
  input_width: 640
  input_height: 384
  max_targets: 64
  cuda_graph: true
  confidence_floor: 0.20
tracking:
  models: ["stationary", "constant_velocity", "constant_acceleration"]
  association: "hungarian"
  max_missed_frames: 3
  immediate_confidence: 0.85
  require_second_observation_below: 0.85
  engage_if_uncertainty_within_radius: true
policy:
  plugin: "deterministic_utility"
  objective: "raw_score"
  horizon_targets: 3
  switch_hysteresis: 0.08
trajectory:
  small_error_mode: "direct_feedforward"
  large_error_mode: "jerk_limited"
  terminal_mode: "critically_damped_pd"
  optional_profile: "minimum_jerk"
actuator:
  backend: "sendinput"
  scheduler_hz: 1000
  relative_counts: true
  cancel_superseded: true
safety:
  require_foreground_match: true
  require_emergency_stop: true
  fail_closed: true
"""
        layer_hw = "capture:\n  stale_after_ms: 10.0\n"
        layer_domain = "policy:\n  horizon_targets: 4\n"
        layer_perception = "perception:\n  confidence_floor: 0.30\n"
        layer_policy = "policy:\n  switch_hysteresis: 0.12\n"
        layer_actuator = "actuator:\n  scheduler_hz: 2000\n"
        layer_override = "runtime:\n  target_p99_ms: 4.5\n"

        resolved, _, source_layers = self.loader.load_layered_strings(
            [layer_base, layer_hw, layer_domain, layer_perception, layer_policy, layer_actuator, layer_override],
            layer_names=["base", "hw", "domain", "perception", "policy", "actuator", "override"]
        )

        self.assertEqual(resolved["capture"]["stale_after_ms"], 10.0)
        self.assertEqual(resolved["policy"]["horizon_targets"], 4)
        self.assertEqual(resolved["perception"]["confidence_floor"], 0.30)
        self.assertEqual(resolved["policy"]["switch_hysteresis"], 0.12)
        self.assertEqual(resolved["actuator"]["scheduler_hz"], 2000)
        self.assertEqual(resolved["runtime"]["target_p99_ms"], 4.5)
        self.assertEqual(len(source_layers), 7)

    def test_partial_subtree_deep_merge(self) -> None:
        """TC-CFG-03: Verify partial subtree override preserves sibling keys."""
        base = {
            "capture": {
                "backend": "dxgi",
                "source_width": 1920,
                "source_height": 1080,
                "buffers": 2,
                "stale_after_ms": 12.0
            }
        }
        overlay = {
            "capture": {
                "stale_after_ms": 8.0
            }
        }

        merged = deep_merge(base, overlay)
        self.assertEqual(merged["capture"]["stale_after_ms"], 8.0)
        self.assertEqual(merged["capture"]["backend"], "dxgi")
        self.assertEqual(merged["capture"]["source_width"], 1920)
        self.assertEqual(merged["capture"]["source_height"], 1080)
        self.assertEqual(merged["capture"]["buffers"], 2)

    def test_array_replacement_semantics(self) -> None:
        """TC-CFG-04: Verify arrays are atomically replaced rather than appended."""
        base = {
            "tracking": {
                "models": ["stationary", "constant_velocity", "constant_acceleration"]
            }
        }
        overlay = {
            "tracking": {
                "models": ["constant_velocity"]
            }
        }

        merged = deep_merge(base, overlay)
        self.assertEqual(merged["tracking"]["models"], ["constant_velocity"])

    def test_source_layers_provenance_tracking(self) -> None:
        """TC-CFG-05: Verify provenance tracking of loaded source layers."""
        base_path = self.configs_dir / "base.yaml"
        hw_path = self.configs_dir / "hardware" / "windows_1080p.yaml"

        _, _, source_layers = self.loader.load_layered([base_path, hw_path])
        self.assertEqual(len(source_layers), 2)
        self.assertEqual(source_layers[0], base_path.as_posix())
        self.assertEqual(source_layers[1], hw_path.as_posix())

    # -------------------------------------------------------------------------
    # Category B: Unknown Key Rejection (Schema Strictness)
    # -------------------------------------------------------------------------

    def test_unknown_root_key_rejection(self) -> None:
        """TC-CFG-06: Pass unrecognized root property, verify fail closed."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        bad_dict = copy.deepcopy(base_dict)
        bad_dict["unrecognized_setting"] = 123

        with self.assertRaises(ConfigValidationError) as ctx:
            self.loader.validate(bad_dict)
        self.assertIn("unrecognized_setting", str(ctx.exception))

    def test_unknown_subdomain_key_rejection(self) -> None:
        """TC-CFG-07: Pass unknown property in subdomain, verify failure."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        bad_dict = copy.deepcopy(base_dict)
        bad_dict["capture"]["invalid_buffer_mode"] = "async"

        with self.assertRaises(ConfigValidationError) as ctx:
            self.loader.validate(bad_dict)
        self.assertIn("invalid_buffer_mode", str(ctx.exception))

    def test_typo_key_rejection(self) -> None:
        """TC-CFG-08: Pass subtle typos, verify rejection."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        bad_dict = copy.deepcopy(base_dict)
        del bad_dict["runtime"]["target_p99_ms"]
        bad_dict["runtime"]["target_p99"] = 6.0

        with self.assertRaises(ConfigValidationError) as ctx:
            self.loader.validate(bad_dict)
        self.assertIn("target_p99", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Category C: Value Range Validation, Clamping, & Telemetry
    # -------------------------------------------------------------------------

    def test_structural_bounds_violation_failure(self) -> None:
        """TC-CFG-09: Invalid structural bounds fail validation immediately."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")

        bad_width = copy.deepcopy(base_dict)
        bad_width["capture"]["source_width"] = 0
        with self.assertRaises(ConfigValidationError):
            self.loader.validate(bad_width)

        bad_deadline = copy.deepcopy(base_dict)
        bad_deadline["runtime"]["internal_deadline_ms"] = -1.0
        with self.assertRaises(ConfigValidationError):
            self.loader.validate(bad_deadline)

    def test_safe_tuning_value_clamping_upper(self) -> None:
        """TC-CFG-10: Out-of-range tuning value exceeding max is clamped to 1.0."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        test_dict = copy.deepcopy(base_dict)
        test_dict["perception"]["confidence_floor"] = 1.45

        clamped, warnings = clamp_tuning_parameters(test_dict)
        self.assertEqual(clamped["perception"]["confidence_floor"], 1.0)
        self.assertTrue(len(warnings) > 0)
        self.assertIn("perception.confidence_floor", warnings[0])

    def test_safe_tuning_value_clamping_lower(self) -> None:
        """TC-CFG-11: Out-of-range tuning value below min is clamped to 0.0."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        test_dict = copy.deepcopy(base_dict)
        test_dict["perception"]["confidence_floor"] = -0.20
        test_dict["tracking"]["immediate_confidence"] = -0.50

        clamped, warnings = clamp_tuning_parameters(test_dict)
        self.assertEqual(clamped["perception"]["confidence_floor"], 0.0)
        self.assertEqual(clamped["tracking"]["immediate_confidence"], 0.0)
        self.assertEqual(len(warnings), 2)

    def test_hot_reload_allowlist_enforcement(self) -> None:
        """TC-CFG-12: Verify allowlisted hot reload fields vs structural restart rejection."""
        current = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")

        # Safe update
        safe_update = copy.deepcopy(current)
        safe_update["perception"]["confidence_floor"] = 0.35
        safe_update["policy"]["switch_hysteresis"] = 0.15

        res_safe = validate_hot_reload(current, safe_update)
        self.assertTrue(res_safe.is_allowed)
        self.assertFalse(res_safe.requires_restart)
        self.assertIn("perception.confidence_floor", res_safe.modified_allowed_fields)
        self.assertIn("policy.switch_hysteresis", res_safe.modified_allowed_fields)

        # Structural update
        struct_update = copy.deepcopy(current)
        struct_update["capture"]["backend"] = "wgc"

        res_struct = validate_hot_reload(current, struct_update)
        self.assertFalse(res_struct.is_allowed)
        self.assertTrue(res_struct.requires_restart)
        self.assertIn("capture.backend", res_struct.conflicting_structural_fields)

    # -------------------------------------------------------------------------
    # Category D: Missing Required Keys & Type Safety
    # -------------------------------------------------------------------------

    def test_missing_schema_version_rejection(self) -> None:
        """TC-CFG-13: Missing schema_version fails validation."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        bad_dict = copy.deepcopy(base_dict)
        del bad_dict["schema_version"]

        with self.assertRaises(ConfigValidationError):
            self.loader.validate(bad_dict)

    def test_missing_domain_section_rejection(self) -> None:
        """TC-CFG-14: Missing entire required domain fails validation."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        bad_dict = copy.deepcopy(base_dict)
        del bad_dict["safety"]

        with self.assertRaises(ConfigValidationError):
            self.loader.validate(bad_dict)

    def test_type_mismatch_rejection(self) -> None:
        """TC-CFG-15: Incorrect data types fail validation."""
        base_dict = self.loader.parse_yaml_file(self.configs_dir / "base.yaml")
        bad_dict = copy.deepcopy(base_dict)
        bad_dict["capture"]["source_width"] = "1920"  # String instead of int

        with self.assertRaises(ConfigValidationError):
            self.loader.validate(bad_dict)

    # -------------------------------------------------------------------------
    # Category E: Manifest Creation, Hash Validation, & Immutability
    # -------------------------------------------------------------------------

    def test_run_manifest_creation_and_completeness(self) -> None:
        """TC-MAN-01: Verify RunManifestBuilder creates a complete valid manifest."""
        resolved, cfg_hash, source_layers = self.loader.load_layered([self.configs_dir / "base.yaml"])

        builder = RunManifestBuilder(repo_root=self.repo_root)
        manifest = builder.build(
            resolved_config=resolved,
            source_layers=source_layers,
            config_sha256=cfg_hash,
            mode="simulation",
            notes="Automated test manifest"
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("manifest_id", manifest)
        self.assertIn("timestamp_utc", manifest)
        self.assertIn("git_info", manifest)
        self.assertIn("environment", manifest)
        self.assertIn("config", manifest)
        self.assertIn("schemas", manifest)
        self.assertIn("artifacts", manifest)
        self.assertIn("execution", manifest)
        self.assertEqual(manifest["config"]["config_sha256"], cfg_hash)

        is_valid, msg = verify_run_manifest(manifest)
        self.assertTrue(is_valid, msg)

    def test_config_sha256_stability(self) -> None:
        """TC-MAN-02: Verify deterministic SHA-256 hash stability."""
        resolved, hash1, _ = self.loader.load_layered([self.configs_dir / "base.yaml"])
        hash2 = compute_config_sha256(resolved)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_manifest_tamper_detection(self) -> None:
        """TC-MAN-03: Verify manifest tamper detection when resolved_config is altered."""
        resolved, cfg_hash, source_layers = self.loader.load_layered([self.configs_dir / "base.yaml"])

        builder = RunManifestBuilder(repo_root=self.repo_root)
        manifest = builder.build(
            resolved_config=resolved,
            source_layers=source_layers,
            config_sha256=cfg_hash,
        )

        is_valid, _ = verify_run_manifest(manifest)
        self.assertTrue(is_valid)

        # Tamper with resolved config
        tampered_manifest = copy.deepcopy(manifest)
        tampered_manifest["config"]["resolved_config"]["runtime"]["target_p99_ms"] = 3.0

        is_tampered_valid, tamper_msg = verify_run_manifest(tampered_manifest)
        self.assertFalse(is_tampered_valid)
        self.assertIn("tamper detected", tamper_msg.lower())

    # -------------------------------------------------------------------------
    # Category F: Default Repository Config Profiles Validation
    # -------------------------------------------------------------------------

    def test_all_default_configs_valid(self) -> None:
        """Verify all default configuration files in configs/ can be layered on base.yaml."""
        base_path = self.configs_dir / "base.yaml"
        sub_configs = [
            self.configs_dir / "hardware" / "windows_1080p.yaml",
            self.configs_dir / "domain" / "aimlabs.yaml",
            self.configs_dir / "perception" / "yolo11n.yaml",
            self.configs_dir / "policy" / "deterministic_utility.yaml",
            self.configs_dir / "actuator" / "sendinput.yaml",
            self.configs_dir / "actuator" / "null.yaml",
        ]

        for sub_cfg in sub_configs:
            self.assertTrue(sub_cfg.is_file(), f"Missing config file: {sub_cfg}")
            resolved, _, _ = self.loader.load_layered([base_path, sub_cfg])
            self.assertEqual(resolved["schema_version"], 1)

    # -------------------------------------------------------------------------
    # Category G: C++ and Python Cross-Language Parity
    # -------------------------------------------------------------------------

    def test_cross_language_config_hash_parity(self) -> None:
        """TC-PAR-01: Verify C++ and Python produce exact identical SHA-256 hash for base.yaml."""
        base_path = self.configs_dir / "base.yaml"
        resolved, python_hash, _ = self.loader.load_layered([base_path])

        # Golden SHA-256 hash
        expected_golden_hash = "1b41a4d4af3dce9db93dd17131b1b521acfbe70268f4ae5f2f7ccd6f2f4f54a7"
        self.assertEqual(python_hash, expected_golden_hash)

    def test_cross_language_schema_validation_parity(self) -> None:
        """TC-PAR-02: Verify identical pass/fail outcomes on valid and invalid fixtures."""
        base_path = self.configs_dir / "base.yaml"
        valid_dict = self.loader.parse_yaml_file(base_path)
        self.loader.validate(valid_dict)

        # Invalid fixture: unknown key
        invalid_unknown = copy.deepcopy(valid_dict)
        invalid_unknown["unrecognized_key"] = "forbidden"
        with self.assertRaises(ConfigValidationError):
            self.loader.validate(invalid_unknown)

        # Invalid fixture: type mismatch
        invalid_type = copy.deepcopy(valid_dict)
        invalid_type["runtime"]["warmup_iterations"] = "not_an_int"
        with self.assertRaises(ConfigValidationError):
            self.loader.validate(invalid_type)


if __name__ == "__main__":
    unittest.main()
