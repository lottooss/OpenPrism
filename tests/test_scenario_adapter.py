"""Scenario profile schema, config, and C++/schema parity tests (Milestone M7-01).

The C++ side (``tests/cpp/test_scenario_adapter.cpp``) proves the adapter's runtime
behaviour. This suite proves the *declarative* half: that the shipped YAML profiles
validate, that the schema is genuinely strict, and — most importantly — that the
range constants compiled into ``ScenarioAdapter::validate()`` are identical to the
bounds declared in the JSON Schema, so the two validators cannot drift apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
import unittest

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "config" / "scenario_profile.schema.json"
PROFILES_DIR = REPO_ROOT / "configs" / "scenario"
CONTRACT_HEADER = REPO_ROOT / "include" / "aim" / "interfaces" / "scenario_adapter.hpp"

# Every capability flag, paired with the profile data it obliges the profile to carry.
# Mirrors the ``capability_without_data`` branches of ScenarioAdapter::validate().
CAPABILITY_OBLIGATIONS = {
    "provides_calibration_seed": ("calibration_seed", "valid"),
    "provides_crosshair_context": ("crosshair", "valid"),
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML profile, asserting it is a mapping."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AssertionError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


class TestScenarioProfileSchema(unittest.TestCase):
    """The schema itself must be well formed and strict."""

    schema: dict[str, Any]
    validator: jsonschema.Draft202012Validator

    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

    def test_schema_is_valid_draft_2020_12(self) -> None:
        """TC-SCN-01: The schema compiles under the dialect it declares."""
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        jsonschema.Draft202012Validator.check_schema(self.schema)
        self.assertEqual(self.schema["title"], "AimAgentScenarioProfile")

    def test_schema_rejects_unknown_keys_everywhere(self) -> None:
        """TC-SCN-02: additionalProperties is false at the root and in every section."""
        self.assertFalse(self.schema["additionalProperties"])
        for name, section in self.schema["properties"].items():
            if section.get("type") == "object":
                self.assertFalse(
                    section.get("additionalProperties", True),
                    f"section '{name}' must reject unknown keys",
                )

    def test_schema_requires_version_and_profile_id(self) -> None:
        """TC-SCN-03: schema_version and profile_id are the mandatory keys.

        Every standalone document family in this repository is versioned. Because
        the schema sets additionalProperties=false and the C++ loader whitelists root
        keys, a version added later would invalidate every profile already written -
        so it has to be present from the first release.
        """
        self.assertEqual(self.schema["required"], ["schema_version", "profile_id"])
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], 1)
        errors = list(self.validator.iter_errors({}))
        self.assertTrue(errors, "an empty document must not validate")
        self.assertTrue(
            list(self.validator.iter_errors({"profile_id": "x"})),
            "a document without schema_version must not validate",
        )
        self.assertTrue(
            list(self.validator.iter_errors({"schema_version": 2, "profile_id": "x"})),
            "a future schema_version must not validate",
        )


class TestShippedProfiles(unittest.TestCase):
    """Every profile under configs/scenario/ must validate and stay domain-neutral."""

    schema: dict[str, Any]
    validator: jsonschema.Draft202012Validator
    profiles: list[Path]

    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.validator = jsonschema.Draft202012Validator(cls.schema)
        cls.profiles = sorted(PROFILES_DIR.glob("*.yaml"))

    def test_expected_profiles_exist(self) -> None:
        """TC-SCN-04: The generic and Aimlabs profiles are both shipped."""
        names = {path.name for path in self.profiles}
        self.assertIn("generic.yaml", names)
        self.assertIn("aimlabs.yaml", names)

    def test_all_profiles_validate(self) -> None:
        """TC-SCN-05: Every shipped profile satisfies the schema."""
        self.assertTrue(self.profiles, "no scenario profiles found")
        for path in self.profiles:
            with self.subTest(profile=path.name):
                errors = sorted(self.validator.iter_errors(load_yaml(path)), key=str)
                self.assertEqual(
                    errors, [], f"{path.name}: " + "; ".join(e.message for e in errors)
                )

    def test_capability_flags_are_backed_by_data(self) -> None:
        """TC-SCN-06: A profile never advertises a capability it cannot honour.

        Mirrors ScenarioAdapter::validate()'s ``capability_without_data`` branches.
        """
        for path in self.profiles:
            doc = load_yaml(path)
            caps = doc.get("capabilities", {})
            with self.subTest(profile=path.name):
                for flag, (section, key) in CAPABILITY_OBLIGATIONS.items():
                    if caps.get(flag, False):
                        self.assertTrue(
                            doc.get(section, {}).get(key, False),
                            f"{path.name} advertises {flag} but {section}.{key} is not set",
                        )
                if caps.get("provides_target_value", False):
                    self.assertGreater(doc.get("default_target_value", 1.0), 0.0)
                if caps.get("reports_scenario_completion", False):
                    self.assertGreater(doc.get("max_engagement_seconds", 0.0), 0.0)
                if caps.get("provides_foreground_identity", False):
                    foreground = doc.get("foreground", {})
                    # A window title alone is not an authorization boundary: any
                    # application can set its own title. A constrained identity must
                    # therefore name a process, matching ScenarioAdapter::validate().
                    self.assertTrue(
                        bool(foreground.get("process_name"))
                        or not foreground.get("require_match", True),
                        f"{path.name} constrains an identity without naming a process",
                    )

    def test_generic_profile_carries_no_domain_semantics(self) -> None:
        """TC-SCN-07: The domain-neutral profile names no application."""
        doc = load_yaml(PROFILES_DIR / "generic.yaml")
        self.assertEqual(doc["profile_id"], "generic_targets")
        foreground = doc.get("foreground", {})
        self.assertEqual(foreground.get("process_name", ""), "")
        self.assertEqual(foreground.get("window_title_substring", ""), "")
        # Explicitly unconstrained rather than accidentally unconfirmable.
        self.assertFalse(foreground.get("require_match", True))
        self.assertFalse(doc.get("capabilities", {}).get("reports_scenario_completion", False))

    def test_aimlabs_profile_is_thin_and_injection_free(self) -> None:
        """TC-SCN-08: The Aimlabs profile carries identity and seeds, nothing more.

        The acceptance criterion is that no game state is read. Structurally that
        means the profile may only contain the keys the schema declares - there is
        no score, recoil, weapon, movement, or memory-offset surface to read from.
        """
        doc = load_yaml(PROFILES_DIR / "aimlabs.yaml")
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["profile_id"], "aimlabs_gridshot")
        self.assertEqual(doc["foreground"]["process_name"], "Aimlab_tb.exe")
        self.assertEqual(doc["foreground"]["window_title_substring"], "aimlab")
        self.assertTrue(doc["foreground"]["require_match"])
        self.assertEqual(doc["max_engagement_seconds"], 60.0)
        self.assertTrue(doc["calibration_seed"]["valid"])

        schema_keys = set(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["properties"])
        self.assertEqual(set(doc) - schema_keys, set())
        forbidden = {"score", "recoil", "weapon", "movement", "memory", "offset", "address", "pid"}
        self.assertEqual(forbidden & set(doc), set())

    def test_engagement_window_agrees_with_safety_config(self) -> None:
        """TC-SCN-09: The Aimlabs window matches the safety layer's engagement cap.

        configs/domain/aimlabs.yaml is the authority enforced by SafetySupervisor;
        the scenario profile must not quietly disagree with it.
        """
        profile = load_yaml(PROFILES_DIR / "aimlabs.yaml")
        domain = load_yaml(REPO_ROOT / "configs" / "domain" / "aimlabs.yaml")
        safety = domain.get("safety", {})
        self.assertEqual(profile["max_engagement_seconds"], safety["max_active_engagement_seconds"])
        self.assertEqual(profile["foreground"]["process_name"], safety["target_process_name"])
        self.assertEqual(profile["foreground"]["window_title_substring"], safety["target_window_title"])


class TestSchemaMatchesCppContract(unittest.TestCase):
    """The compiled C++ bounds and the JSON Schema bounds must be identical.

    ``ScenarioAdapter::validate()`` and the JSON Schema are two independent
    validators over the same data. If they disagree, YAML rejected by one is
    accepted by the other. These tests read the constants straight out of the
    header so the drift is caught at test time rather than in the field.
    """

    schema: dict[str, Any]
    limits: dict[str, float]

    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.limits = cls._parse_limits(CONTRACT_HEADER.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_limits(source: str) -> dict[str, float]:
        """Extract the ``aim::scenario_limits`` constants from the contract header."""
        block = re.search(
            r"namespace scenario_limits \{(.*?)\}\s*//\s*namespace scenario_limits",
            source,
            re.DOTALL,
        )
        if block is None:
            raise AssertionError("scenario_limits namespace not found in the contract header")
        pattern = re.compile(
            r"inline constexpr (?:float|std::size_t) (k\w+)\s*=\s*([0-9.]+)f?;"
        )
        limits = {name: float(value) for name, value in pattern.findall(block.group(1))}
        if not limits:
            raise AssertionError("no scenario_limits constants parsed")
        return limits

    def test_all_limits_were_parsed(self) -> None:
        """TC-SCN-10: Every constant the parity test relies on is present."""
        expected = {
            "kMaxProfileIdLength",
            "kMaxIdentityLength",
            "kMaxTargetValue",
            "kMaxEngagementSeconds",
            "kMaxCountsPerPixel",
            "kMaxInGameSensitivity",
            "kMaxFovDegExclusive",
            "kMaxCrosshairPxX",
            "kMaxCrosshairPxY",
        }
        self.assertEqual(expected - set(self.limits), set())

    def test_scalar_bounds_match(self) -> None:
        """TC-SCN-11: Root and nested numeric maxima agree with the C++ constants."""
        props = self.schema["properties"]
        self.assertEqual(props["profile_id"]["maxLength"], self.limits["kMaxProfileIdLength"])
        self.assertEqual(props["default_target_value"]["maximum"], self.limits["kMaxTargetValue"])
        self.assertEqual(
            props["max_engagement_seconds"]["maximum"], self.limits["kMaxEngagementSeconds"]
        )

        seed = props["calibration_seed"]["properties"]
        self.assertEqual(seed["counts_per_pixel_x"]["maximum"], self.limits["kMaxCountsPerPixel"])
        self.assertEqual(seed["counts_per_pixel_y"]["maximum"], self.limits["kMaxCountsPerPixel"])
        self.assertEqual(
            seed["in_game_sensitivity"]["maximum"], self.limits["kMaxInGameSensitivity"]
        )
        self.assertEqual(
            seed["fov_horizontal_deg"]["exclusiveMaximum"], self.limits["kMaxFovDegExclusive"]
        )

        crosshair = props["crosshair"]["properties"]
        self.assertEqual(crosshair["center_px_x"]["maximum"], self.limits["kMaxCrosshairPxX"])
        self.assertEqual(crosshair["center_px_y"]["maximum"], self.limits["kMaxCrosshairPxY"])
        self.assertEqual(crosshair["center_norm_x"]["minimum"], -1.0)
        self.assertEqual(crosshair["center_norm_x"]["maximum"], 1.0)
        self.assertEqual(crosshair["center_norm_y"]["minimum"], -1.0)
        self.assertEqual(crosshair["center_norm_y"]["maximum"], 1.0)

        foreground = props["foreground"]["properties"]
        self.assertEqual(foreground["process_name"]["maxLength"], self.limits["kMaxIdentityLength"])
        self.assertEqual(
            foreground["window_title_substring"]["maxLength"], self.limits["kMaxIdentityLength"]
        )

    def test_profile_id_pattern_matches_the_cpp_validator(self) -> None:
        """TC-SCN-12: The schema pattern is the one profile_id_is_well_formed() implements."""
        pattern = self.schema["properties"]["profile_id"]["pattern"]
        self.assertEqual(pattern, "^[a-z0-9][a-z0-9_]*$")
        compiled = re.compile(pattern)
        for accepted in ("generic_targets", "aimlabs_gridshot", "a", "0abc", "a_b_c"):
            self.assertIsNotNone(compiled.match(accepted), accepted)
        for rejected in ("", "Aimlabs", "aim labs", "_leading", "trailing-dash", "dots.here"):
            self.assertIsNone(compiled.match(rejected), rejected)

    def test_capability_flags_match_the_cpp_struct(self) -> None:
        """TC-SCN-13: The schema declares exactly the ScenarioCapabilities fields."""
        source = CONTRACT_HEADER.read_text(encoding="utf-8")
        struct = re.search(r"struct ScenarioCapabilities \{(.*?)\n\};", source, re.DOTALL)
        assert struct is not None
        cpp_flags = set(re.findall(r"bool (\w+)\{false\};", struct.group(1)))
        schema_flags = set(self.schema["properties"]["capabilities"]["properties"])
        self.assertEqual(cpp_flags, schema_flags)


if __name__ == "__main__":
    unittest.main()
