"""Adversarial stress-test suite for Milestone M1-03: Layered Configuration and Run Manifests.

Empirically tests:
1. Complex multi-tier YAML merge precedence and array replacement semantics.
2. Exhaustive unknown key rejection at root and across all 9 domains.
3. Type-coercion fuzzing across all primitives (strings, ints, floats, booleans, arrays, objects, null).
4. Boundary values and parameter clamping.
5. Hot-reload allowlist permutation testing.
6. Run manifest cryptographic tamper detection and schema conformance.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from typing import Any

from tools.config.hot_reload import (
    ALLOWLISTED_HOT_RELOAD_FIELDS,
    clamp_tuning_parameters,
    validate_hot_reload,
)
from tools.config.loader import ConfigValidationError, LayeredConfigLoader, deep_merge
from tools.config.manifest import RunManifestBuilder, verify_run_manifest


class TestConfigAdversarial(unittest.TestCase):
    """Adversarial challenger tests for layered configuration and run manifests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.configs_dir = cls.repo_root / "configs"
        cls.schemas_dir = cls.repo_root / "schemas" / "config"
        cls.manifest_schema_path = cls.repo_root / "schemas" / "manifest" / "run_manifest.schema.json"
        cls.loader = LayeredConfigLoader(schemas_dir=cls.schemas_dir)
        cls.base_config = cls.loader.parse_yaml_file(cls.configs_dir / "base.yaml")

    def get_clean_config(self) -> dict[str, Any]:
        """Return a fresh deep copy of base configuration."""
        return copy.deepcopy(self.base_config)

    # =========================================================================
    # SECTION 1: Layered Merge Precedence & Array Stress Tests
    # =========================================================================

    def test_deep_ten_layer_precedence_merge(self) -> None:
        """Stress-test: 10 layers with overlapping overrides at multiple depths."""
        layers = [
            # Layer 0: Base
            (self.configs_dir / "base.yaml").read_text(encoding="utf-8"),
            # Layer 1: Hardware
            "capture:\n  stale_after_ms: 11.5\n  buffers: 3\n",
            # Layer 2: Domain
            "policy:\n  horizon_targets: 5\n  objective: nearest_target\n",
            # Layer 3: Perception
            "perception:\n  confidence_floor: 0.35\n  nms_iou_threshold: 0.50\n",
            # Layer 4: Tracking
            "tracking:\n  models: [stationary, constant_acceleration]\n  immediate_confidence: 0.90\n",
            # Layer 5: Policy override
            "policy:\n  switch_hysteresis: 0.15\n",
            # Layer 6: Trajectory
            "trajectory:\n  small_error_threshold_px: 20.0\n  max_velocity_counts_per_s: 60000.0\n",
            # Layer 7: Actuator
            "actuator:\n  scheduler_hz: 2000\n",
            # Layer 8: Safety
            "safety:\n  max_delta_counts_per_dispatch: 750\n",
            # Layer 9: Experiment Override (overriding multiple previous layers)
            "runtime:\n  target_p99_ms: 5.2\npolicy:\n  horizon_targets: 2\ntracking:\n  models: [constant_velocity]\n"
        ]

        resolved, cfg_hash, source_layers = self.loader.load_layered_strings(
            layers,
            layer_names=[f"L{i}" for i in range(len(layers))]
        )

        # Verify final winning values
        self.assertEqual(resolved["capture"]["stale_after_ms"], 11.5)
        self.assertEqual(resolved["capture"]["buffers"], 3)
        self.assertEqual(resolved["policy"]["objective"], "nearest_target")
        self.assertEqual(resolved["policy"]["horizon_targets"], 2)  # Layer 9 won over Layer 2
        self.assertEqual(resolved["perception"]["confidence_floor"], 0.35)
        self.assertEqual(resolved["perception"]["nms_iou_threshold"], 0.50)
        self.assertEqual(resolved["tracking"]["models"], ["constant_velocity"])  # Layer 9 atomic replace
        self.assertEqual(resolved["tracking"]["immediate_confidence"], 0.90)
        self.assertEqual(resolved["policy"]["switch_hysteresis"], 0.15)
        self.assertEqual(resolved["trajectory"]["small_error_threshold_px"], 20.0)
        self.assertEqual(resolved["trajectory"]["max_velocity_counts_per_s"], 60000.0)
        self.assertEqual(resolved["actuator"]["scheduler_hz"], 2000)
        self.assertEqual(resolved["safety"]["max_delta_counts_per_dispatch"], 750)
        self.assertEqual(resolved["runtime"]["target_p99_ms"], 5.2)

        self.assertEqual(len(source_layers), 10)
        self.assertEqual(len(cfg_hash), 64)

    def test_array_replacement_with_empty_and_single_element(self) -> None:
        """Verify tracking.models array replacement works with empty array (schema rejects minItems 1) and single element."""
        # Single element array should succeed
        cfg1 = self.get_clean_config()
        cfg1["tracking"]["models"] = ["stationary"]
        self.loader.validate(cfg1)

        # Empty array should fail schema validation (minItems: 1)
        cfg_empty = self.get_clean_config()
        cfg_empty["tracking"]["models"] = []
        with self.assertRaises(ConfigValidationError):
            self.loader.validate(cfg_empty)

    def test_array_elements_uniqueness(self) -> None:
        """Verify duplicate items in tracking.models array fail uniqueItems schema constraint."""
        cfg_dup = self.get_clean_config()
        cfg_dup["tracking"]["models"] = ["stationary", "stationary"]
        with self.assertRaises(ConfigValidationError):
            self.loader.validate(cfg_dup)

    def test_deep_merge_non_dict_overwrites(self) -> None:
        """Verify deep_merge behavior when replacing subtrees or scalars."""
        base = {"a": {"b": 1, "c": 2}, "d": [1, 2, 3]}

        # Override dict with scalar
        overlay1 = {"a": 42}
        res1 = deep_merge(base, overlay1)
        self.assertEqual(res1["a"], 42)

        # Override scalar with dict
        overlay2 = {"d": {"x": 10}}
        res2 = deep_merge(base, overlay2)
        self.assertEqual(res2["d"], {"x": 10})

    # =========================================================================
    # SECTION 2: Exhaustive Unknown Key Rejection across All Domains
    # =========================================================================

    def test_unknown_keys_in_all_nine_domains(self) -> None:
        """Empirically test that unknown keys are strictly rejected in every domain."""
        domains = [
            "runtime",
            "capture",
            "perception",
            "tracking",
            "policy",
            "trajectory",
            "actuator",
            "safety",
        ]

        for domain in domains:
            cfg = self.get_clean_config()
            cfg[domain]["_adversarial_unknown_key_999"] = "exploit"
            with self.assertRaises(ConfigValidationError, msg=f"Failed to reject unknown key in domain '{domain}'") as ctx:
                self.loader.validate(cfg)
            self.assertIn("_adversarial_unknown_key_999", str(ctx.exception))

    def test_unknown_keys_at_root_hierarchy(self) -> None:
        """Test unknown keys at root level with various patterns."""
        invalid_keys = [
            "extra_field",
            "RUNTIME",  # Case variant
            "Capture",  # Case variant
            "__proto__",
            "constructor",
            " ",
            "aim_policy",
            "debug",
            "telemetry_extended",
        ]

        for key in invalid_keys:
            cfg = self.get_clean_config()
            cfg[key] = True
            with self.assertRaises(ConfigValidationError, msg=f"Failed to reject root key '{key}'") as ctx:
                self.loader.validate(cfg)
            self.assertIn(key, str(ctx.exception))

    def test_case_sensitivity_in_subdomains(self) -> None:
        """Verify that camelCase or UPPERCASE variants of known keys are rejected as unknown."""
        case_tests = [
            ("runtime", "internalDeadlineMs", 10.0),
            ("runtime", "TARGET_P99_MS", 6.0),
            ("capture", "Source_Width", 1920),
            ("perception", "ConfidenceFloor", 0.20),
            ("tracking", "Max_Missed_Frames", 3),
            ("policy", "SwitchHysteresis", 0.08),
            ("actuator", "Scheduler_Hz", 1000),
            ("safety", "FailClosed", True),
        ]

        for domain, bad_key, val in case_tests:
            cfg = self.get_clean_config()
            cfg[domain][bad_key] = val
            with self.assertRaises(ConfigValidationError, msg=f"Failed to reject case variation '{domain}.{bad_key}'") as ctx:
                self.loader.validate(cfg)
            self.assertIn(bad_key, str(ctx.exception))

    # =========================================================================
    # SECTION 3: Type-Coercion Fuzzing
    # =========================================================================

    def test_string_coercion_where_numbers_required(self) -> None:
        """Verify strings formatted as numbers are rejected where integer or float required."""
        test_cases = [
            ("runtime", "internal_deadline_ms", "10.0"),
            ("runtime", "target_p99_ms", "6.0"),
            ("runtime", "warmup_iterations", "200"),
            ("capture", "source_width", "1920"),
            ("capture", "source_height", "1080"),
            ("capture", "buffers", "2"),
            ("capture", "stale_after_ms", "12.0"),
            ("perception", "input_width", "640"),
            ("perception", "input_height", "384"),
            ("perception", "confidence_floor", "0.20"),
            ("tracking", "max_missed_frames", "3"),
            ("tracking", "immediate_confidence", "0.85"),
            ("policy", "horizon_targets", "3"),
            ("policy", "switch_hysteresis", "0.08"),
            ("trajectory", "small_error_threshold_px", "15.0"),
            ("actuator", "scheduler_hz", "1000"),
            ("safety", "max_delta_counts_per_dispatch", "500"),
        ]

        for domain, field_name, bad_value in test_cases:
            cfg = self.get_clean_config()
            cfg[domain][field_name] = bad_value
            with self.assertRaises(ConfigValidationError, msg=f"String '{bad_value}' was accepted for numeric {domain}.{field_name}"):
                self.loader.validate(cfg)

    def test_boolean_coercion_where_numbers_required(self) -> None:
        """Verify boolean values (True/False) are rejected where numbers required."""
        test_cases = [
            ("runtime", "internal_deadline_ms", True),
            ("runtime", "warmup_iterations", False),
            ("capture", "source_width", True),
            ("perception", "confidence_floor", False),
            ("tracking", "max_missed_frames", True),
            ("actuator", "scheduler_hz", False),
        ]

        for domain, field_name, bad_value in test_cases:
            cfg = self.get_clean_config()
            cfg[domain][field_name] = bad_value
            with self.assertRaises(ConfigValidationError, msg=f"Boolean '{bad_value}' was accepted for {domain}.{field_name}"):
                self.loader.validate(cfg)

    def test_numeric_coercion_where_boolean_required(self) -> None:
        """Verify 1 / 0 integers are rejected where boolean required."""
        test_cases = [
            ("runtime", "allocation_audit", 1),
            ("capture", "latest_only", 1),
            ("capture", "allow_cross_adapter_copy", 0),
            ("perception", "cuda_graph", 1),
            ("tracking", "engage_if_uncertainty_within_radius", 0),
            ("actuator", "relative_counts", 1),
            ("actuator", "cancel_superseded", 0),
            ("safety", "require_foreground_match", 1),
            ("safety", "fail_closed", 0),
        ]

        for domain, field_name, bad_value in test_cases:
            cfg = self.get_clean_config()
            cfg[domain][field_name] = bad_value
            with self.assertRaises(ConfigValidationError, msg=f"Integer '{bad_value}' was accepted for boolean {domain}.{field_name}"):
                self.loader.validate(cfg)

    def test_float_where_strict_integer_required(self) -> None:
        """Verify float values with fractional parts are rejected where integer required."""
        test_cases = [
            ("runtime", "warmup_iterations", 200.5),
            ("capture", "source_width", 1920.7),
            ("capture", "buffers", 2.5),
            ("perception", "batch", 1.5),
            ("perception", "input_width", 640.2),
            ("tracking", "max_missed_frames", 3.14),
            ("policy", "horizon_targets", 3.9),
            ("actuator", "scheduler_hz", 1000.1),
        ]

        for domain, field_name, bad_value in test_cases:
            cfg = self.get_clean_config()
            cfg[domain][field_name] = bad_value
            with self.assertRaises(ConfigValidationError, msg=f"Float '{bad_value}' was accepted for integer {domain}.{field_name}"):
                self.loader.validate(cfg)

    def test_null_none_rejection_for_required_fields(self) -> None:
        """Verify None/null is rejected for all standard primitive types."""
        test_cases = [
            ("schema_version", None),
            ("runtime", None),
            ("capture", "backend", None),
            ("capture", "source_width", None),
            ("perception", "confidence_floor", None),
            ("tracking", "models", None),
            ("policy", "objective", None),
            ("trajectory", "small_error_mode", None),
            ("actuator", "scheduler_hz", None),
            ("safety", "fail_closed", None),
        ]

        for item in test_cases:
            cfg = self.get_clean_config()
            if len(item) == 2:
                field_name, val = item
                cfg[field_name] = val
            else:
                domain, field_name, val = item
                cfg[domain][field_name] = val
            with self.assertRaises(ConfigValidationError, msg=f"None/null was accepted for {item}"):
                self.loader.validate(cfg)

    def test_complex_types_where_primitives_expected(self) -> None:
        """Verify arrays and dicts are rejected where primitive values expected."""
        test_cases = [
            ("capture", "backend", ["dxgi"]),
            ("capture", "source_width", {"width": 1920}),
            ("perception", "confidence_floor", [0.20]),
            ("actuator", "scheduler_hz", {"hz": 1000}),
            ("safety", "fail_closed", {"enabled": True}),
        ]

        for domain, field_name, bad_value in test_cases:
            cfg = self.get_clean_config()
            cfg[domain][field_name] = bad_value
            with self.assertRaises(ConfigValidationError, msg=f"Complex type was accepted for {domain}.{field_name}"):
                self.loader.validate(cfg)

    # =========================================================================
    # SECTION 4: Boundary Values, Enum Constraints & Clamping
    # =========================================================================

    def test_enum_constraints_rejection(self) -> None:
        """Verify invalid enum values are strictly rejected across all domains."""
        enum_tests = [
            ("capture", "backend", "d3d11"),
            ("capture", "backend", "DXGI"),
            ("capture", "fallback", "dxgi"),
            ("capture", "pixel_format", "rgb24"),
            ("perception", "precision", "bf16"),
            ("tracking", "association", "gnn"),
            ("policy", "objective", "max_score"),
            ("trajectory", "small_error_mode", "quadratic"),
            ("trajectory", "large_error_mode", "step"),
            ("trajectory", "terminal_mode", "pid"),
            ("actuator", "backend", "interception"),
        ]

        for domain, field_name, invalid_enum in enum_tests:
            cfg = self.get_clean_config()
            cfg[domain][field_name] = invalid_enum
            with self.assertRaises(ConfigValidationError, msg=f"Invalid enum '{invalid_enum}' was accepted for {domain}.{field_name}"):
                self.loader.validate(cfg)

    def test_extreme_clamping_and_warning_generation(self) -> None:
        """Verify extreme continuous values are clamped to limits with warnings."""
        cfg = self.get_clean_config()
        cfg["perception"]["confidence_floor"] = 999.0
        cfg["perception"]["nms_iou_threshold"] = -50.0
        cfg["tracking"]["immediate_confidence"] = 10.0
        cfg["tracking"]["require_second_observation_below"] = -1.0
        cfg["policy"]["switch_hysteresis"] = 100.0
        cfg["runtime"]["warmup_iterations"] = 50000
        cfg["trajectory"]["small_error_threshold_px"] = 1000.0
        cfg["trajectory"]["max_velocity_counts_per_s"] = 5.0
        cfg["trajectory"]["max_acceleration_counts_per_s2"] = 1e12
        cfg["actuator"]["scheduler_hz"] = 50

        clamped, warnings = clamp_tuning_parameters(cfg)

        self.assertEqual(clamped["perception"]["confidence_floor"], 1.0)
        self.assertEqual(clamped["perception"]["nms_iou_threshold"], 0.0)
        self.assertEqual(clamped["tracking"]["immediate_confidence"], 1.0)
        self.assertEqual(clamped["tracking"]["require_second_observation_below"], 0.0)
        self.assertEqual(clamped["policy"]["switch_hysteresis"], 10.0)
        self.assertEqual(clamped["runtime"]["warmup_iterations"], 10000)
        self.assertEqual(clamped["trajectory"]["small_error_threshold_px"], 500.0)
        self.assertEqual(clamped["trajectory"]["max_velocity_counts_per_s"], 100.0)
        self.assertEqual(clamped["trajectory"]["max_acceleration_counts_per_s2"], 100000000.0)
        self.assertEqual(clamped["actuator"]["scheduler_hz"], 100)

        self.assertEqual(len(warnings), 10)

    # =========================================================================
    # SECTION 5: Hot Reload Exhaustive Allowlist Permutation Testing
    # =========================================================================

    def test_all_allowlisted_fields_individually(self) -> None:
        """Verify each of the 17 allowlisted hot-reload fields succeeds individually."""
        allowed_mutations: dict[str, Any] = {
            "perception.confidence_floor": 0.35,
            "perception.nms_iou_threshold": 0.60,
            "tracking.max_missed_frames": 5,
            "tracking.immediate_confidence": 0.90,
            "tracking.require_second_observation_below": 0.75,
            "tracking.engage_if_uncertainty_within_radius": False,
            "tracking.process_noise_scale": 1.5,
            "tracking.measurement_noise_scale": 0.8,
            "tracking.gating_threshold_chi2": 11.34,
            "policy.switch_hysteresis": 0.12,
            "policy.horizon_targets": 4,
            "policy.target_lead_time_ms": 5.0,
            "runtime.internal_deadline_ms": 8.0,
            "runtime.target_p99_ms": 5.0,
            "safety.fail_closed": False,
            "safety.max_delta_counts_per_dispatch": 600,
            "safety.max_active_engagement_seconds": 120.0,
        }

        self.assertEqual(len(allowed_mutations), len(ALLOWLISTED_HOT_RELOAD_FIELDS))

        current = self.get_clean_config()

        for field_path, new_val in allowed_mutations.items():
            updated = copy.deepcopy(current)
            domain, key = field_path.split(".", 1)
            updated[domain][key] = new_val

            result = validate_hot_reload(current, updated)
            self.assertTrue(result.is_allowed, f"Field '{field_path}' should be allowed to hot-reload")
            self.assertFalse(result.requires_restart, f"Field '{field_path}' unexpectedly required restart")
            self.assertIn(field_path, result.modified_allowed_fields)
            self.assertEqual(len(result.conflicting_structural_fields), 0)

    def test_all_structural_fields_trigger_restart(self) -> None:
        """Verify structural fields in every domain trigger requires_restart=True."""
        structural_mutations: list[tuple[str, str, Any]] = [
            ("runtime", "warmup_iterations", 300),
            ("runtime", "allocation_audit", False),
            ("runtime", "thread_priority", "normal"),
            ("capture", "backend", "wgc"),
            ("capture", "fallback", "none"),
            ("capture", "source_width", 2560),
            ("capture", "source_height", 1440),
            ("capture", "pixel_format", "rgba8_sdr"),
            ("capture", "buffers", 4),
            ("capture", "latest_only", False),
            ("perception", "plugin", "custom_yolo"),
            ("perception", "precision", "fp32"),
            ("perception", "input_width", 1280),
            ("perception", "input_height", 768),
            ("perception", "cuda_graph", False),
            ("tracking", "models", ["stationary"]),
            ("tracking", "association", "greedy"),
            ("policy", "plugin", "heuristic_policy"),
            ("policy", "objective", "nearest_target"),
            ("trajectory", "small_error_mode", "linear"),
            ("actuator", "backend", "usb_hid"),
            ("actuator", "scheduler_hz", 2000),
            ("safety", "target_process_name", "custom_game.exe"),
            ("safety", "emergency_stop_key", "F11"),
        ]

        current = self.get_clean_config()

        for domain, key, new_val in structural_mutations:
            updated = copy.deepcopy(current)
            updated[domain][key] = new_val

            result = validate_hot_reload(current, updated)
            self.assertFalse(result.is_allowed, f"Structural field '{domain}.{key}' should not be hot-reloadable")
            self.assertTrue(result.requires_restart, f"Structural field '{domain}.{key}' must require restart")
            self.assertIn(f"{domain}.{key}", result.conflicting_structural_fields)

    # =========================================================================
    # SECTION 6: Manifest Cryptographic Tamper Detection & Integrity
    # =========================================================================

    def test_manifest_schema_validation_and_tamper(self) -> None:
        """Verify RunManifest adheres to run_manifest.schema.json and detects tampering."""
        resolved, cfg_hash, source_layers = self.loader.load_layered([self.configs_dir / "base.yaml"])

        builder = RunManifestBuilder(repo_root=self.repo_root)
        manifest = builder.build(
            resolved_config=resolved,
            source_layers=source_layers,
            config_sha256=cfg_hash,
            mode="benchmark",
            notes="Adversarial validation test"
        )

        # Validate against run_manifest.schema.json
        import jsonschema
        manifest_schema = json.loads(self.manifest_schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(manifest_schema).validate(manifest)

        # Verify integrity check passes
        is_valid, msg = verify_run_manifest(manifest)
        self.assertTrue(is_valid, msg)

        # Test single bit tamper in SHA-256
        tampered_hash = copy.deepcopy(manifest)
        # Flip first character of hash
        orig_char = tampered_hash["config"]["config_sha256"][0]
        flipped_char = '0' if orig_char != '0' else '1'
        tampered_hash["config"]["config_sha256"] = flipped_char + tampered_hash["config"]["config_sha256"][1:]

        is_valid_t1, msg_t1 = verify_run_manifest(tampered_hash)
        self.assertFalse(is_valid_t1)
        self.assertIn("tamper detected", msg_t1.lower())

        # Test tampering deep field inside resolved_config
        tampered_field = copy.deepcopy(manifest)
        tampered_field["config"]["resolved_config"]["safety"]["fail_closed"] = False
        is_valid_t2, msg_t2 = verify_run_manifest(tampered_field)
        self.assertFalse(is_valid_t2)
        self.assertIn("tamper detected", msg_t2.lower())


if __name__ == "__main__":
    unittest.main()
