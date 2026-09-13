# tools/plugin/abi.py
"""OpenPrism Dynamic Plugin C ABI Types and ctypes Wrappers."""

from __future__ import annotations

import ctypes
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import Callable, Optional, Tuple

AIM_PLUGIN_ABI_VERSION_V1 = 0x00010000


def abi_major(version: int) -> int:
    return (version >> 16) & 0xFFFF


def abi_minor(version: int) -> int:
    return version & 0xFFFF


def fourcc_to_uint32(identifier: str) -> int:
    """Convert an exact four-character ASCII schema identifier to little-endian uint32."""
    if len(identifier) != 4 or not identifier.isascii():
        raise ValueError("schema identifier must contain exactly four ASCII characters")
    encoded = identifier.encode("ascii")
    return int.from_bytes(encoded, byteorder="little", signed=False)


class AimPluginKind(IntEnum):
    UNKNOWN = 0
    PERCEPTION = 1
    TRACKER = 2
    POLICY = 3
    TRAJECTORY = 4
    ACTUATOR = 5
    SCENARIO = 6


class AimStatusCode(IntEnum):
    OK = 0
    ERROR_INVALID_ARGUMENT = 1
    ERROR_BUFFER_TOO_SMALL = 2
    ERROR_ABI_MISMATCH = 3
    ERROR_UNINITIALIZED = 4
    ERROR_ALREADY_INITIALIZED = 5
    ERROR_OUT_OF_BOUNDS = 6
    ERROR_CORRUPTED_DATA = 7
    ERROR_EXECUTION_FAILED = 8
    ERROR_HARDWARE_FAULT = 9
    ERROR_EMERGENCY_STOP = 10
    ERROR_NOT_SUPPORTED = 11


# Host Service Callbacks
EmitTelemetryFn = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32)
MonotonicTimeNsFn = ctypes.CFUNCTYPE(ctypes.c_int64)
ReportErrorFn = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_char_p)


class AimHostServicesV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("emit_telemetry", EmitTelemetryFn),
        ("monotonic_time_ns", MonotonicTimeNsFn),
        ("report_error", ReportErrorFn),
    ]


class AimPluginDescriptorV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("author", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("input_schema_fourcc", ctypes.c_uint32),
        ("output_schema_fourcc", ctypes.c_uint32),
    ]


# Plugin API Callbacks
StartFn = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32)
ProcessFn = ctypes.CFUNCTYPE(
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
)
StopFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
DestroyFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
GetDescriptorFn = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(AimPluginDescriptorV1))


class AimPluginApiV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("instance", ctypes.c_void_p),
        ("start", StartFn),
        ("process", ProcessFn),
        ("stop", StopFn),
        ("destroy", DestroyFn),
        ("get_descriptor", GetDescriptorFn),
    ]


class PluginWrapper:
    """Python ctypes loader and wrapper for OpenPrism native plugins."""

    def __init__(
        self,
        dll_path: Path | str,
        emit_telemetry: Optional[Callable[[bytes], None]] = None,
        report_error: Optional[Callable[[int, str], None]] = None,
    ):
        self.dll_path = Path(dll_path).resolve()
        if not self.dll_path.is_file():
            raise FileNotFoundError(f"Plugin library not found: {self.dll_path}")

        self._lib = ctypes.CDLL(str(self.dll_path))

        # Check export functions
        if not hasattr(self._lib, "aim_plugin_abi_version") or not hasattr(self._lib, "aim_plugin_create_v1"):
            raise ValueError(f"Plugin library {self.dll_path} does not export required factory functions")

        self._lib.aim_plugin_abi_version.restype = ctypes.c_uint32
        self._lib.aim_plugin_abi_version.argtypes = []

        self._lib.aim_plugin_create_v1.restype = ctypes.c_uint32
        self._lib.aim_plugin_create_v1.argtypes = [
            ctypes.POINTER(AimHostServicesV1),
            ctypes.POINTER(AimPluginApiV1),
        ]

        # Phase 1: ABI Version Handshake
        self.reported_abi = self._lib.aim_plugin_abi_version()
        if abi_major(self.reported_abi) != abi_major(AIM_PLUGIN_ABI_VERSION_V1):
            raise RuntimeError(
                f"Plugin ABI mismatch: reported 0x{self.reported_abi:08X}, expected major 0x{abi_major(AIM_PLUGIN_ABI_VERSION_V1):04X}"
            )

        # Build Host Services
        self._user_emit_telemetry = emit_telemetry
        self._user_report_error = report_error

        def _c_emit_telemetry(
            buf_ptr: ctypes._Pointer[ctypes.c_uint8], size: int
        ) -> None:
            try:
                if self._user_emit_telemetry and buf_ptr and size > 0:
                    data = bytes(ctypes.string_at(buf_ptr, size))
                    self._user_emit_telemetry(data)
            except Exception:
                # Python exceptions must never unwind through a native callback boundary.
                return

        def _c_monotonic_time_ns() -> int:
            import time
            return time.monotonic_ns()

        def _c_report_error(code: int, msg_ptr: bytes | None) -> None:
            try:
                if self._user_report_error:
                    msg = msg_ptr.decode("utf-8", errors="replace") if msg_ptr else ""
                    self._user_report_error(code, msg)
            except Exception:
                # Python exceptions must never unwind through a native callback boundary.
                return

        self._cb_emit_telemetry = EmitTelemetryFn(_c_emit_telemetry)
        self._cb_monotonic_time_ns = MonotonicTimeNsFn(_c_monotonic_time_ns)
        self._cb_report_error = ReportErrorFn(_c_report_error)

        self.host_services = AimHostServicesV1(
            abi_version=AIM_PLUGIN_ABI_VERSION_V1,
            emit_telemetry=self._cb_emit_telemetry,
            monotonic_time_ns=self._cb_monotonic_time_ns,
            report_error=self._cb_report_error,
        )

        # Phase 2: Create plugin instance
        self.api = AimPluginApiV1()
        status = self._lib.aim_plugin_create_v1(
            ctypes.byref(self.host_services),
            ctypes.byref(self.api),
        )

        if status != AimStatusCode.OK or not self.api.instance:
            status_name = AimStatusCode(status).name if status in AimStatusCode._value2member_map_ else "UNKNOWN"
            raise RuntimeError(f"aim_plugin_create_v1 failed with status code {status} ({status_name})")

        required_callbacks = (
            self.api.start,
            self.api.process,
            self.api.stop,
            self.api.destroy,
            self.api.get_descriptor,
        )
        api_valid = (
            abi_major(self.api.abi_version) == abi_major(AIM_PLUGIN_ABI_VERSION_V1)
            and self.api.kind in {kind.value for kind in AimPluginKind if kind is not AimPluginKind.UNKNOWN}
            and all(bool(callback) for callback in required_callbacks)
        )
        if not api_valid:
            if self.api.destroy and self.api.instance:
                self.api.destroy(self.api.instance)
                self.api.instance = None
            raise RuntimeError("plugin returned a malformed or incompatible AimPluginApiV1 table")

        self.is_active = False

    def start(self, config_bytes: bytes = b"") -> AimStatusCode:
        if not self.api.start or not self.api.instance:
            return AimStatusCode.ERROR_UNINITIALIZED
        if self.is_active:
            return AimStatusCode.ERROR_ALREADY_INITIALIZED
        if len(config_bytes) > ctypes.c_uint32(-1).value:
            return AimStatusCode.ERROR_OUT_OF_BOUNDS

        buf = (ctypes.c_uint8 * len(config_bytes)).from_buffer_copy(config_bytes) if config_bytes else None
        res = self.api.start(self.api.instance, buf, len(config_bytes))
        status = AimStatusCode(res) if res in AimStatusCode._value2member_map_ else AimStatusCode.ERROR_CORRUPTED_DATA
        if status == AimStatusCode.OK:
            self.is_active = True
        return status

    def process(self, input_bytes: bytes, output_capacity: int = 65536) -> Tuple[AimStatusCode, bytes]:
        if not self.api.process or not self.api.instance or not self.is_active:
            return AimStatusCode.ERROR_UNINITIALIZED, b""

        if output_capacity < 0 or output_capacity > ctypes.c_uint32(-1).value:
            return AimStatusCode.ERROR_INVALID_ARGUMENT, b""
        if len(input_bytes) > ctypes.c_uint32(-1).value:
            return AimStatusCode.ERROR_OUT_OF_BOUNDS, b""

        in_buf = (ctypes.c_uint8 * len(input_bytes)).from_buffer_copy(input_bytes) if input_bytes else None
        out_buf = (ctypes.c_uint8 * output_capacity)()
        out_size = ctypes.c_uint32(0)

        res = self.api.process(
            self.api.instance,
            in_buf,
            len(input_bytes),
            out_buf,
            output_capacity,
            ctypes.byref(out_size),
        )

        if (
            out_size.value > output_capacity
            and res not in (
                AimStatusCode.ERROR_BUFFER_TOO_SMALL,
                AimStatusCode.ERROR_OUT_OF_BOUNDS,
            )
        ):
            return AimStatusCode.ERROR_CORRUPTED_DATA, b""

        status = AimStatusCode(res) if res in AimStatusCode._value2member_map_ else AimStatusCode.ERROR_CORRUPTED_DATA
        if status == AimStatusCode.OK:
            actual_size = out_size.value
            return status, bytes(out_buf[:actual_size])
        return status, b""

    def stop(self) -> None:
        if self.is_active and self.api.stop and self.api.instance:
            self.api.stop(self.api.instance)
            self.is_active = False

    def destroy(self) -> None:
        self.stop()
        if self.api.destroy and self.api.instance:
            self.api.destroy(self.api.instance)
            self.api.instance = None

    def get_descriptor(self) -> Optional[AimPluginDescriptorV1]:
        if not self.api.get_descriptor or not self.api.instance:
            return None
        desc = AimPluginDescriptorV1()
        res = self.api.get_descriptor(self.api.instance, ctypes.byref(desc))
        if res == AimStatusCode.OK:
            if (
                abi_major(desc.abi_version) != abi_major(AIM_PLUGIN_ABI_VERSION_V1)
                or desc.kind != self.api.kind
                or not desc.name
                or not desc.version
            ):
                return None
            return desc
        return None

    def __enter__(self) -> PluginWrapper:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.destroy()
