# tools/plugin/contract_validator.py
"""Automated Contract & ABI Validator for OpenPrism Dynamic Plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from tools.plugin.abi import (
    AIM_PLUGIN_ABI_VERSION_V1,
    AimPluginKind,
    AimStatusCode,
    PluginWrapper,
    abi_major,
    fourcc_to_uint32,
)
from tools.plugin.manifest import (
    load_manifest_file,
    verify_binary_sha256,
)


@dataclass
class ValidationResult:
    is_valid: bool = True
    manifest_valid: bool = True
    checksum_valid: bool = True
    abi_negotiation_valid: bool = True
    lifecycle_valid: bool = True
    buffer_bounds_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PluginContractValidator:
    """Validates third-party plugin manifests, checksums, ABI conformance, and lifecycle."""

    @staticmethod
    def validate_manifest(manifest_path: Path | str) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        errors = []
        try:
            data = load_manifest_file(manifest_path)
            return True, data, []
        except (jsonschema.ValidationError, ValueError, FileNotFoundError) as e:
            errors.append(f"Manifest validation error: {e}")
            return False, None, errors

    @staticmethod
    def validate_checksum(manifest_dict: Dict[str, Any], binary_path: Path | str) -> Tuple[bool, List[str]]:
        errors = []
        try:
            valid = verify_binary_sha256(manifest_dict, binary_path)
            if not valid:
                errors.append(
                    f"Checksum mismatch: binary hash does not match manifest hash {manifest_dict.get('checksum', {}).get('sha256')}"
                )
            return valid, errors
        except Exception as e:
            errors.append(f"Checksum verification failed: {e}")
            return False, errors

    @classmethod
    def validate_plugin(
        cls,
        manifest_path: Path | str,
        binary_path: Path | str,
        sample_input: bytes = b"",
    ) -> ValidationResult:
        result = ValidationResult()

        # Step 1: Manifest Schema Check
        m_ok, manifest_data, m_errs = cls.validate_manifest(manifest_path)
        if not m_ok or not manifest_data:
            result.manifest_valid = False
            result.is_valid = False
            result.errors.extend(m_errs)
            return result

        result.metadata["manifest"] = manifest_data

        # Step 2: Binary Checksum
        c_ok, c_errs = cls.validate_checksum(manifest_data, binary_path)
        if not c_ok:
            result.checksum_valid = False
            result.is_valid = False
            result.errors.extend(c_errs)
            # Loading an unverified library would execute untrusted code. Fail closed.
            return result

        # Step 3: Dynamic ABI Loading and Lifecycle
        telemetry_records: List[bytes] = []
        error_records: List[Tuple[int, str]] = []

        def _on_telemetry(buf: bytes) -> None:
            telemetry_records.append(buf)

        def _on_error(code: int, msg: str) -> None:
            error_records.append((code, msg))

        wrapper: Optional[PluginWrapper] = None
        try:
            wrapper = PluginWrapper(
                binary_path,
                emit_telemetry=_on_telemetry,
                report_error=_on_error,
            )

            # Check ABI Handshake
            if abi_major(wrapper.reported_abi) != abi_major(AIM_PLUGIN_ABI_VERSION_V1):
                result.abi_negotiation_valid = False
                result.is_valid = False
                result.errors.append(
                    f"Reported ABI version 0x{wrapper.reported_abi:08X} does not match host 0x{AIM_PLUGIN_ABI_VERSION_V1:08X}"
                )

            manifest_abi = int(manifest_data["abi_version"])
            expected_kind = AimPluginKind[manifest_data["kind"].upper()]
            descriptor = wrapper.get_descriptor()
            if abi_major(manifest_abi) != abi_major(wrapper.reported_abi):
                result.abi_negotiation_valid = False
                result.is_valid = False
                result.errors.append("Manifest ABI major does not match the loaded plugin")
            if wrapper.api.kind != expected_kind:
                result.abi_negotiation_valid = False
                result.is_valid = False
                result.errors.append("Manifest plugin kind does not match the API table")
            if descriptor is None:
                result.abi_negotiation_valid = False
                result.is_valid = False
                result.errors.append("Plugin descriptor is missing or malformed")
            else:
                expected_input = fourcc_to_uint32(
                    manifest_data["contracts"]["input_schema"]["identifier"]
                )
                expected_output = fourcc_to_uint32(
                    manifest_data["contracts"]["output_schema"]["identifier"]
                )
                if descriptor.input_schema_fourcc != expected_input:
                    result.abi_negotiation_valid = False
                    result.is_valid = False
                    result.errors.append("Manifest input schema does not match the plugin descriptor")
                if descriptor.output_schema_fourcc != expected_output:
                    result.abi_negotiation_valid = False
                    result.is_valid = False
                    result.errors.append("Manifest output schema does not match the plugin descriptor")

            if not result.abi_negotiation_valid:
                return result

            # Check Lifecycle: start -> process -> stop -> destroy
            status_start = wrapper.start(b"{}")
            if status_start != AimStatusCode.OK:
                result.lifecycle_valid = False
                result.is_valid = False
                result.errors.append(f"Plugin start() failed with status {status_start.name}")

            # Process 1: Normal call
            status_proc, out_bytes = wrapper.process(sample_input, output_capacity=65536)
            if status_proc != AimStatusCode.OK:
                result.lifecycle_valid = False
                result.is_valid = False
                result.errors.append(f"Plugin process() failed with status {status_proc.name}")

            # Bounds Check: Insufficient capacity
            status_small, _ = wrapper.process(sample_input, output_capacity=1)
            # If the plugin outputs data, capacity=1 should return ERROR_BUFFER_TOO_SMALL or ERROR_OUT_OF_BOUNDS
            if len(out_bytes) > 1 and status_small not in (
                AimStatusCode.ERROR_BUFFER_TOO_SMALL,
                AimStatusCode.ERROR_OUT_OF_BOUNDS,
            ):
                result.buffer_bounds_valid = False
                result.is_valid = False
                result.errors.append(
                    f"Plugin did not return ERROR_BUFFER_TOO_SMALL for capacity 1 (returned {status_small.name})"
                )

            wrapper.stop()

        except Exception as e:
            result.is_valid = False
            result.lifecycle_valid = False
            result.errors.append(f"Plugin execution exception: {e}")
        finally:
            if wrapper is not None:
                wrapper.destroy()

        result.metadata["telemetry_count"] = len(telemetry_records)
        result.metadata["reported_errors"] = error_records
        return result
