# tests/test_plugin_abi.py
"""Tests for OpenPrism Dynamic Plugin C ABI Python Bindings."""

import ctypes
import os
from pathlib import Path

import pytest

from tools.plugin.abi import (
    AIM_PLUGIN_ABI_VERSION_V1,
    AimHostServicesV1,
    AimPluginApiV1,
    AimPluginDescriptorV1,
    AimPluginKind,
    AimStatusCode,
    PluginWrapper,
    abi_major,
    abi_minor,
    fourcc_to_uint32,
)


def test_abi_version_constants():
    assert AIM_PLUGIN_ABI_VERSION_V1 == 0x00010000
    assert abi_major(AIM_PLUGIN_ABI_VERSION_V1) == 1
    assert abi_minor(AIM_PLUGIN_ABI_VERSION_V1) == 0
    assert abi_major(0x00020005) == 2
    assert abi_minor(0x00020005) == 5
    assert fourcc_to_uint32("AOB1") == 0x31424F41
    with pytest.raises(ValueError):
        fourcc_to_uint32("TOO-LONG")


def test_plugin_kind_enums():
    assert AimPluginKind.UNKNOWN == 0
    assert AimPluginKind.PERCEPTION == 1
    assert AimPluginKind.TRACKER == 2
    assert AimPluginKind.POLICY == 3
    assert AimPluginKind.TRAJECTORY == 4
    assert AimPluginKind.ACTUATOR == 5
    assert AimPluginKind.SCENARIO == 6


def test_status_code_enums():
    assert AimStatusCode.OK == 0
    assert AimStatusCode.ERROR_INVALID_ARGUMENT == 1
    assert AimStatusCode.ERROR_BUFFER_TOO_SMALL == 2
    assert AimStatusCode.ERROR_ABI_MISMATCH == 3
    assert AimStatusCode.ERROR_UNINITIALIZED == 4
    assert AimStatusCode.ERROR_ALREADY_INITIALIZED == 5
    assert AimStatusCode.ERROR_OUT_OF_BOUNDS == 6
    assert AimStatusCode.ERROR_CORRUPTED_DATA == 7
    assert AimStatusCode.ERROR_EXECUTION_FAILED == 8
    assert AimStatusCode.ERROR_HARDWARE_FAULT == 9
    assert AimStatusCode.ERROR_EMERGENCY_STOP == 10
    assert AimStatusCode.ERROR_NOT_SUPPORTED == 11


def test_struct_sizes_and_alignment():
    # AimHostServicesV1: 1 uint32 + 3 function pointers (with 4 bytes padding on 64-bit)
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(AimHostServicesV1) == 32
        assert ctypes.sizeof(AimPluginDescriptorV1) == 48
        assert ctypes.sizeof(AimPluginApiV1) == 56


def find_built_dll(name: str) -> Path | None:
    search_dirs = [
        Path("build/windows-msvc-release"),
        Path("build/windows-msvc-release/bin"),
        Path("build/windows-msvc-release/Release"),
        Path("build/windows-msvc-debug"),
        Path("build/windows-msvc-debug/bin"),
        Path("build/windows-msvc-debug/Debug"),
        Path("build-wsl"),
        Path("build"),
        Path("."),
    ]
    suffixes = [".dll"] if os.name == "nt" else [f"lib{name}.so", f"{name}.so", ".so"]
    for d in search_dirs:
        for s in suffixes:
            candidate = d / (f"{name}{s}" if s.startswith(".") else s)
            if candidate.is_file():
                return candidate.resolve()
    return None


def test_plugin_wrapper_with_mock_dll():
    perc_dll = find_built_dll("aim_mock_perception_plugin")
    if not perc_dll:
        pytest.skip("aim_mock_perception_plugin.dll not yet compiled")

    telemetry_received = []

    def on_telem(buf: bytes):
        telemetry_received.append(buf)

    with PluginWrapper(perc_dll, emit_telemetry=on_telem) as wrapper:
        assert wrapper.reported_abi == AIM_PLUGIN_ABI_VERSION_V1
        assert wrapper.api.kind == AimPluginKind.PERCEPTION

        desc = wrapper.get_descriptor()
        assert desc is not None
        assert desc.name.decode("utf-8") == "mock_perception"

        # Start plugin
        status = wrapper.start(b"{}")
        assert status == AimStatusCode.OK
        assert wrapper.is_active is True

        # Process 10 frames
        for _ in range(10):
            status, out_data = wrapper.process(b"", output_capacity=65536)
            assert status == AimStatusCode.OK
            assert len(out_data) > 0

        # Truncation check
        status, _ = wrapper.process(b"", output_capacity=10)
        assert status == AimStatusCode.ERROR_BUFFER_TOO_SMALL

        status, _ = wrapper.process(b"", output_capacity=-1)
        assert status == AimStatusCode.ERROR_INVALID_ARGUMENT

        wrapper.stop()
        assert wrapper.is_active is False
