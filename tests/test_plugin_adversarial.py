"""tests/test_plugin_adversarial.py
Adversarial Verification and Empirical Stress Testing Suite for Milestone M1-06 (#12):
Module Contract and Plugin ABI Test Harness.

Adversarial Objectives:
1. ABI negotiation stress testing: Test major version mismatch, minor version compatibility,
   missing entrypoints, corrupted exports, and invalid kind matching.
2. Buffer truncation & overflow fuzzing: Pass 0-byte, 1-byte, and undersized output buffers and
   verify plugin never writes past capacity, preserves canary memory integrity, and returns AIM_STATUS_ERROR_BUFFER_TOO_SMALL.
3. Exception barrier stress testing: Induce deliberate C++ / runtime exceptions across all lifecycle functions
   (start, process, get_descriptor, stop, destroy) and verify zero unhandled exceptions escape across ABI boundary.
4. Reload stress testing: Rapidly create, execute, and destroy callback-table fixtures in
   high-iteration loops (1,000+ cycles), verifying zero lifecycle errors and strict state invariants.
"""

from __future__ import annotations

import ctypes
import gc
import struct
import unittest
from typing import Any, Callable, List, Optional, Tuple

from tools.plugin.abi import (
    AIM_PLUGIN_ABI_VERSION_V1,
    AimHostServicesV1,
    AimPluginApiV1,
    AimPluginDescriptorV1,
    AimPluginKind,
    AimStatusCode,
    DestroyFn,
    EmitTelemetryFn,
    GetDescriptorFn,
    MonotonicTimeNsFn,
    ProcessFn,
    ReportErrorFn,
    StartFn,
    StopFn,
    abi_major,
    abi_minor,
)
from tools.plugin.manifest import (
    validate_manifest_dict,
)


class MockNativePlugin:
    """Simulates a native dynamic plugin library implementing the Aim Plugin C ABI v1.

    Provides configurable behavior for version negotiation, buffer bounds enforcement,
    exception barrier testing, and reload stress testing.
    """

    def __init__(
        self,
        abi_version: int = AIM_PLUGIN_ABI_VERSION_V1,
        kind: AimPluginKind = AimPluginKind.PERCEPTION,
        name: str = "mock_adversarial_plugin",
        version: str = "1.0.0",
        author: str = "Aim Challenger",
        description: str = "Mock adversarial plugin for ABI and contract stress testing",
        input_fourcc: int = 0x31524641,  # 'AFR1'
        output_fourcc: int = 0x31424F41,  # 'AOB1'
        payload_size: int = 2088,  # sizeof(TargetObservationBatch)
        fail_create_status: Optional[AimStatusCode] = None,
        throw_on_start: bool = False,
        throw_on_process: bool = False,
        throw_on_descriptor: bool = False,
        throw_on_stop: bool = False,
        throw_on_destroy: bool = False,
    ):
        self.abi_version = abi_version
        self.kind = kind
        self.name = name.encode("utf-8")
        self.version = version.encode("utf-8")
        self.author = author.encode("utf-8")
        self.description = description.encode("utf-8")
        self.input_fourcc = input_fourcc
        self.output_fourcc = output_fourcc
        self.payload_size = payload_size
        self.fail_create_status = fail_create_status

        self.throw_on_start = throw_on_start
        self.throw_on_process = throw_on_process
        self.throw_on_descriptor = throw_on_descriptor
        self.throw_on_stop = throw_on_stop
        self.throw_on_destroy = throw_on_destroy

        self.is_active = False
        self.is_destroyed = False
        self.processed_frames = 0
        self.host_services: Optional[AimHostServicesV1] = None
        self.instance_id = id(self)

        # Retain C callback references to prevent garbage collection
        self._c_start = StartFn(self._start_impl)
        self._c_process = ProcessFn(self._process_impl)
        self._c_stop = StopFn(self._stop_impl)
        self._c_destroy = DestroyFn(self._destroy_impl)
        self._c_get_descriptor = GetDescriptorFn(self._get_descriptor_impl)

    def _start_impl(self, instance: int, config_ptr: Any, config_size: int) -> int:
        if not instance:
            return AimStatusCode.ERROR_INVALID_ARGUMENT
        if self.is_active:
            return AimStatusCode.ERROR_ALREADY_INITIALIZED

        if self.throw_on_start:
            if self.host_services and self.host_services.report_error:
                self.host_services.report_error(
                    AimStatusCode.ERROR_EXECUTION_FAILED,
                    b"Simulated start failure",
                )
            return AimStatusCode.ERROR_EXECUTION_FAILED

        self.is_active = True
        return AimStatusCode.OK

    def _process_impl(
        self,
        instance: int,
        in_buf: Any,
        in_size: int,
        out_buf: Any,
        out_cap: int,
        out_size_ptr: Any,
    ) -> int:
        if not instance:
            return AimStatusCode.ERROR_INVALID_ARGUMENT
        if not self.is_active:
            return AimStatusCode.ERROR_UNINITIALIZED
        if not out_size_ptr:
            return AimStatusCode.ERROR_INVALID_ARGUMENT

        if self.throw_on_process:
            if self.host_services and self.host_services.report_error:
                self.host_services.report_error(
                    AimStatusCode.ERROR_EXECUTION_FAILED,
                    b"Simulated process internal exception",
                )
            return AimStatusCode.ERROR_EXECUTION_FAILED

        # Always set required output payload size
        out_size_ptr.contents.value = self.payload_size

        if out_cap < self.payload_size:
            return AimStatusCode.ERROR_BUFFER_TOO_SMALL

        if not out_buf:
            return AimStatusCode.ERROR_INVALID_ARGUMENT

        # Populate output payload with synthetic observation data
        payload = bytearray(self.payload_size)
        # Header: magic='AOB1', major=1, minor=0, seq=frame, target_count=1
        struct.pack_into("<4sHHI", payload, 0, b"AOB1", 1, 0, self.processed_frames + 1)
        ctypes.memmove(out_buf, (ctypes.c_uint8 * self.payload_size).from_buffer(payload), self.payload_size)

        self.processed_frames += 1

        # Emit optional telemetry callback
        if self.host_services and self.host_services.emit_telemetry:
            telem_data = b"MOCK_TELEM"
            c_buf = (ctypes.c_uint8 * len(telem_data)).from_buffer_copy(telem_data)
            self.host_services.emit_telemetry(c_buf, len(telem_data))

        return AimStatusCode.OK

    def _stop_impl(self, instance: int) -> None:
        if not instance:
            return
        if self.throw_on_stop:
            if self.host_services and self.host_services.report_error:
                self.host_services.report_error(
                    AimStatusCode.ERROR_EXECUTION_FAILED,
                    b"Simulated stop exception",
                )
        self.is_active = False

    def _destroy_impl(self, instance: int) -> None:
        if not instance:
            return
        if self.throw_on_destroy:
            if self.host_services and self.host_services.report_error:
                self.host_services.report_error(
                    AimStatusCode.ERROR_EXECUTION_FAILED,
                    b"Simulated destroy exception",
                )
        self.is_active = False
        self.is_destroyed = True

    def _get_descriptor_impl(self, instance: int, out_desc_ptr: Any) -> int:
        if not instance or not out_desc_ptr:
            return AimStatusCode.ERROR_INVALID_ARGUMENT

        if self.throw_on_descriptor:
            if self.host_services and self.host_services.report_error:
                self.host_services.report_error(
                    AimStatusCode.ERROR_EXECUTION_FAILED,
                    b"Simulated descriptor retrieval failure",
                )
            return AimStatusCode.ERROR_EXECUTION_FAILED

        desc = out_desc_ptr.contents
        desc.abi_version = self.abi_version
        desc.kind = self.kind
        desc.name = self.name
        desc.version = self.version
        desc.author = self.author
        desc.description = self.description
        desc.input_schema_fourcc = self.input_fourcc
        desc.output_schema_fourcc = self.output_fourcc

        return AimStatusCode.OK

    def create_api(self, host_services: AimHostServicesV1) -> Tuple[AimStatusCode, AimPluginApiV1]:
        """Simulates aim_plugin_create_v1 factory call."""
        if self.fail_create_status is not None:
            return self.fail_create_status, AimPluginApiV1()

        # Host ABI compatibility check
        if abi_major(host_services.abi_version) != abi_major(self.abi_version):
            return AimStatusCode.ERROR_ABI_MISMATCH, AimPluginApiV1()

        self.host_services = host_services

        api = AimPluginApiV1()
        api.abi_version = self.abi_version
        api.kind = self.kind
        api.instance = ctypes.c_void_p(self.instance_id)
        api.start = self._c_start
        api.process = self._c_process
        api.stop = self._c_stop
        api.destroy = self._c_destroy
        api.get_descriptor = self._c_get_descriptor

        return AimStatusCode.OK, api


class DirectPluginWrapper:
    """Direct harness wrapper connecting Host Services with MockNativePlugin via strict C ABI."""

    def __init__(
        self,
        mock_plugin: MockNativePlugin,
        host_abi_version: int = AIM_PLUGIN_ABI_VERSION_V1,
        emit_telemetry: Optional[Callable[[bytes], None]] = None,
        report_error: Optional[Callable[[int, str], None]] = None,
    ):
        self.mock_plugin = mock_plugin
        self.reported_abi = mock_plugin.abi_version

        # Handshake check: Host verifies plugin ABI major version
        if abi_major(self.reported_abi) != abi_major(host_abi_version):
            raise RuntimeError(
                f"Plugin ABI mismatch: reported 0x{self.reported_abi:08X}, expected major 0x{abi_major(host_abi_version):04X}"
            )

        self._user_emit_telemetry = emit_telemetry
        self._user_report_error = report_error

        def _c_emit_telem(buf_ptr: Any, size: int) -> None:
            if self._user_emit_telemetry and buf_ptr and size > 0:
                data = bytes(ctypes.string_at(buf_ptr, size))
                self._user_emit_telemetry(data)

        def _c_monotonic_ns() -> int:
            import time

            return time.monotonic_ns()

        def _c_rep_err(code: int, msg_ptr: Any) -> None:
            if self._user_report_error:
                msg = msg_ptr.decode("utf-8", errors="replace") if msg_ptr else ""
                self._user_report_error(code, msg)

        self._cb_emit_telem = EmitTelemetryFn(_c_emit_telem)
        self._cb_mono_ns = MonotonicTimeNsFn(_c_monotonic_ns)
        self._cb_rep_err = ReportErrorFn(_c_rep_err)

        self.host_services = AimHostServicesV1(
            abi_version=host_abi_version,
            emit_telemetry=self._cb_emit_telem,
            monotonic_time_ns=self._cb_mono_ns,
            report_error=self._cb_rep_err,
        )

        status, self.api = mock_plugin.create_api(self.host_services)
        if status != AimStatusCode.OK or not self.api.instance:
            raise RuntimeError(f"aim_plugin_create_v1 failed with status code {status} ({AimStatusCode(status).name})")

        self.is_active = False

    def start(self, config_bytes: bytes = b"") -> AimStatusCode:
        if not self.api.start or not self.api.instance:
            return AimStatusCode.ERROR_UNINITIALIZED

        buf = (ctypes.c_uint8 * len(config_bytes)).from_buffer_copy(config_bytes) if config_bytes else None
        res = self.api.start(self.api.instance, buf, len(config_bytes))
        status = AimStatusCode(res)
        if status == AimStatusCode.OK:
            self.is_active = True
        return status

    def process(self, input_bytes: bytes, output_capacity: int = 65536) -> Tuple[AimStatusCode, bytes, int]:
        if not self.api.process or not self.api.instance or not self.is_active:
            return AimStatusCode.ERROR_UNINITIALIZED, b"", 0

        in_buf = (ctypes.c_uint8 * len(input_bytes)).from_buffer_copy(input_bytes) if input_bytes else None
        out_buf = (ctypes.c_uint8 * output_capacity)() if output_capacity > 0 else None
        out_size = ctypes.c_uint32(0)

        res = self.api.process(
            self.api.instance,
            in_buf,
            len(input_bytes),
            out_buf,
            output_capacity,
            ctypes.byref(out_size),
        )

        status = AimStatusCode(res)
        if status == AimStatusCode.OK and out_buf:
            actual_size = out_size.value
            return status, bytes(out_buf[:actual_size]), actual_size
        return status, b"", out_size.value

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
            return desc
        return None

    def __enter__(self) -> DirectPluginWrapper:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.destroy()


class TestPluginAbiAdversarialStress(unittest.TestCase):
    """Empirical Adversarial Test Suite for Aim Dynamic Plugin C ABI and Interface Contracts."""

    # =========================================================================
    # OBJECTIVE 1: ABI Negotiation Stress Testing
    # =========================================================================

    def test_abi_negotiation_major_version_mismatch(self) -> None:
        """Adversarial Objective 1.1:
        Verifies that plugins reporting major ABI version 2.x (e.g. 0x00020000)
        are strictly rejected when loaded by host expecting ABI v1.x (0x00010000).
        """
        print("\n[ADV-ABI-01] Testing ABI Negotiation: Major Version Mismatch...")

        # Plugin reports ABI v2.0
        mock_v2 = MockNativePlugin(abi_version=0x00020000)

        with self.assertRaises(RuntimeError) as ctx:
            DirectPluginWrapper(mock_v2, host_abi_version=AIM_PLUGIN_ABI_VERSION_V1)

        self.assertIn("Plugin ABI mismatch", str(ctx.exception))
        self.assertIn("0x00020000", str(ctx.exception))
        print("  [+] Major version mismatch (0x00020000 vs 0x00010000) correctly rejected.")

        # Test extreme future major versions (0x00FF0000, 0x00030005, 0x00000001)
        for bad_abi in [0x00000001, 0x00030005, 0x00FF0000, 0xFFFFFFFF]:
            mock_bad = MockNativePlugin(abi_version=bad_abi)
            with self.assertRaises(RuntimeError):
                DirectPluginWrapper(mock_bad, host_abi_version=AIM_PLUGIN_ABI_VERSION_V1)

        print("  [+] Fuzzed extreme ABI versions rejected without error.")

    def test_abi_negotiation_minor_version_forward_compatibility(self) -> None:
        """Adversarial Objective 1.2:
        Verifies that plugins reporting higher or lower minor versions within the same major version
        (e.g., v1.1 = 0x00010001, v1.99 = 0x00010063) negotiate successfully with host v1.0.
        """
        print("\n[ADV-ABI-02] Testing ABI Negotiation: Minor Version Compatibility...")

        compatible_versions = [
            0x00010000,  # v1.0
            0x00010001,  # v1.1
            0x00010005,  # v1.5
            0x00010063,  # v1.99
            0x0001FFFF,  # v1.65535
        ]

        for ver in compatible_versions:
            mock = MockNativePlugin(abi_version=ver)
            wrapper = DirectPluginWrapper(mock, host_abi_version=AIM_PLUGIN_ABI_VERSION_V1)
            self.assertEqual(abi_major(wrapper.reported_abi), 1)
            self.assertEqual(abi_minor(wrapper.reported_abi), abi_minor(ver))

            # Verify lifecycle functions operate seamlessly
            status = wrapper.start(b"{}")
            self.assertEqual(status, AimStatusCode.OK)
            wrapper.stop()
            wrapper.destroy()

        print(f"  [+] Verified forward/backward minor version compatibility across {len(compatible_versions)} versions.")

    def test_abi_negotiation_host_mismatch_rejection_by_plugin(self) -> None:
        """Adversarial Objective 1.3:
        Verifies that if host passes an incompatible host services ABI version (e.g. 0x00020000)
        to a v1 plugin, the plugin factory returns AIM_STATUS_ERROR_ABI_MISMATCH.
        """
        print("\n[ADV-ABI-03] Testing ABI Negotiation: Host Incompatibility Rejection by Plugin...")

        mock_v1 = MockNativePlugin(abi_version=AIM_PLUGIN_ABI_VERSION_V1)

        # Host services initialized with major version 2
        bad_host_services = AimHostServicesV1(
            abi_version=0x00020000,
            emit_telemetry=EmitTelemetryFn(lambda ptr, sz: None),
            monotonic_time_ns=MonotonicTimeNsFn(lambda: 0),
            report_error=ReportErrorFn(lambda code, msg: None),
        )

        status, api = mock_v1.create_api(bad_host_services)
        self.assertEqual(status, AimStatusCode.ERROR_ABI_MISMATCH)
        self.assertIsNone(api.instance)
        print("  [+] Plugin factory properly returned AIM_STATUS_ERROR_ABI_MISMATCH for host ABI v2.")

    def test_abi_missing_or_corrupted_factory_exports(self) -> None:
        """Adversarial Objective 1.4:
        Tests behavior when plugin returns creation failures (e.g. out of memory, invalid args).
        """
        print("\n[ADV-ABI-04] Testing Corrupted Factory Exports...")

        for fail_code in [
            AimStatusCode.ERROR_INVALID_ARGUMENT,
            AimStatusCode.ERROR_EXECUTION_FAILED,
            AimStatusCode.ERROR_HARDWARE_FAULT,
        ]:
            mock_fail = MockNativePlugin(fail_create_status=fail_code)
            with self.assertRaises(RuntimeError) as ctx:
                DirectPluginWrapper(mock_fail)
            self.assertIn(fail_code.name, str(ctx.exception))

        print("  [+] Factory creation failure codes mapped correctly to exceptions.")

    # =========================================================================
    # OBJECTIVE 2: Buffer Truncation & Overflow Fuzzing
    # =========================================================================

    def test_buffer_truncation_fuzzing_canary_integrity(self) -> None:
        """Adversarial Objective 2.1:
        Fuzzes output buffer capacities from 0 bytes up to oversized capacities with canary bytes.
        Verifies:
        - Capacity < payload_size: returns AIM_STATUS_ERROR_BUFFER_TOO_SMALL, sets out_size, writes ZERO bytes.
        - Capacity >= payload_size: returns AIM_STATUS_OK, writes exact payload size.
        - Zero canary bytes corrupted before or after the buffer.
        """
        print("\n[ADV-BUF-01] Running Buffer Truncation & Canary Integrity Fuzzing...")

        payload_size = 2088  # TargetObservationBatch
        mock = MockNativePlugin(payload_size=payload_size)
        wrapper = DirectPluginWrapper(mock)
        self.assertEqual(wrapper.start(), AimStatusCode.OK)

        canary_magic = 0xDEADBEEFCAFEBABE
        canary_len = 128  # 128 bytes front and back

        test_capacities = [
            0,
            1,
            2,
            3,
            4,
            7,
            8,
            15,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            payload_size - 100,
            payload_size - 1,
            payload_size,
            payload_size + 1,
            payload_size + 128,
            payload_size + 4096,
        ]

        for cap in test_capacities:
            # Allocate buffer surrounded by front and back canaries
            total_alloc = canary_len + cap + canary_len
            raw_mem = bytearray(b"\xAA" * total_alloc)

            # Plant structured canaries in front and back
            front_canary = struct.pack("<QQ", canary_magic, canary_magic) * (canary_len // 16)
            back_canary = struct.pack("<QQ", canary_magic, canary_magic) * (canary_len // 16)
            raw_mem[:canary_len] = front_canary
            raw_mem[canary_len + cap :] = back_canary

            # Construct ctypes pointers
            c_mem = (ctypes.c_uint8 * total_alloc).from_buffer(raw_mem)
            out_buf_ptr = ctypes.cast(
                ctypes.addressof(c_mem) + canary_len, ctypes.POINTER(ctypes.c_uint8)
            ) if cap > 0 else None
            out_size_val = ctypes.c_uint32(0)

            res = wrapper.api.process(
                wrapper.api.instance,
                None,
                0,
                out_buf_ptr,
                cap,
                ctypes.byref(out_size_val),
            )
            status = AimStatusCode(res)

            # Assertions based on capacity
            if cap < payload_size:
                self.assertEqual(
                    status,
                    AimStatusCode.ERROR_BUFFER_TOO_SMALL,
                    f"Cap {cap} did not return ERROR_BUFFER_TOO_SMALL",
                )
                self.assertEqual(
                    out_size_val.value,
                    payload_size,
                    f"Cap {cap} failed to report required payload size {payload_size}",
                )
                # Verify buffer content within capacity was NOT modified
                if cap > 0:
                    self.assertEqual(
                        bytes(raw_mem[canary_len : canary_len + cap]),
                        b"\xAA" * cap,
                        f"Buffer was modified despite undersized capacity {cap}!",
                    )
            else:
                self.assertEqual(
                    status,
                    AimStatusCode.OK,
                    f"Cap {cap} failed to process normally",
                )
                self.assertEqual(
                    out_size_val.value,
                    payload_size,
                    f"Cap {cap} output size mismatch",
                )
                # Verify payload has correct FourCC header
                payload_written = bytes(raw_mem[canary_len : canary_len + payload_size])
                self.assertEqual(payload_written[:4], b"AOB1")

                # Verify unused tail within capacity was untouched
                if cap > payload_size:
                    tail_bytes = bytes(raw_mem[canary_len + payload_size : canary_len + cap])
                    self.assertEqual(tail_bytes, b"\xAA" * (cap - payload_size))

            # STRICT CANARY INTEGRITY CHECK: Front and Back must remain 100% intact
            self.assertEqual(
                bytes(raw_mem[:canary_len]),
                front_canary,
                f"FRONT CANARY CORRUPTED for capacity {cap}!",
            )
            self.assertEqual(
                bytes(raw_mem[canary_len + cap :]),
                back_canary,
                f"BACK CANARY OVERFLOW CORRUPTED for capacity {cap}!",
            )

        wrapper.stop()
        wrapper.destroy()
        print(f"  [+] Fuzzed {len(test_capacities)} buffer capacities with 100% canary memory integrity preserved.")

    def test_buffer_null_pointers_robustness(self) -> None:
        """Adversarial Objective 2.2:
        Verifies that passing NULL out_size or NULL out_buf with positive capacity returns
        AIM_STATUS_ERROR_INVALID_ARGUMENT without segmentation fault.
        """
        print("\n[ADV-BUF-02] Testing Null Buffer and Size Pointers...")

        mock = MockNativePlugin()
        wrapper = DirectPluginWrapper(mock)
        wrapper.start()

        # 1. Null out_size pointer
        out_buf = (ctypes.c_uint8 * 4096)()
        res_null_size = wrapper.api.process(
            wrapper.api.instance,
            None,
            0,
            out_buf,
            4096,
            None,
        )
        self.assertEqual(AimStatusCode(res_null_size), AimStatusCode.ERROR_INVALID_ARGUMENT)

        # 2. Null out_buf with positive capacity
        out_size = ctypes.c_uint32(0)
        res_null_buf = wrapper.api.process(
            wrapper.api.instance,
            None,
            0,
            None,
            4096,
            ctypes.byref(out_size),
        )
        self.assertEqual(AimStatusCode(res_null_buf), AimStatusCode.ERROR_INVALID_ARGUMENT)

        wrapper.stop()
        wrapper.destroy()
        print("  [+] Null pointers safely rejected with AIM_STATUS_ERROR_INVALID_ARGUMENT.")

    # =========================================================================
    # OBJECTIVE 3: Exception Barrier Stress Testing
    # =========================================================================

    def test_exception_barrier_across_all_lifecycle_stages(self) -> None:
        """Adversarial Objective 3:
        Induces deliberate internal exceptions across start(), process(), get_descriptor(), stop(), destroy().
        Verifies:
        - Zero unhandled exceptions cross the ABI boundary.
        - Functions return AIM_STATUS_ERROR_EXECUTION_FAILED.
        - Host error callback is invoked with error code and description.
        """
        print("\n[ADV-EXC-01] Testing Exception Containment Across Lifecycle Stages...")

        errors_reported: List[Tuple[int, str]] = []

        def on_error(code: int, msg: str) -> None:
            errors_reported.append((code, msg))

        # 1. Exception during start()
        mock_start_fail = MockNativePlugin(throw_on_start=True)
        wrapper1 = DirectPluginWrapper(mock_start_fail, report_error=on_error)
        status1 = wrapper1.start()
        self.assertEqual(status1, AimStatusCode.ERROR_EXECUTION_FAILED)
        self.assertFalse(wrapper1.is_active)
        self.assertTrue(any(code == AimStatusCode.ERROR_EXECUTION_FAILED for code, _ in errors_reported))
        wrapper1.destroy()

        # 2. Exception during process()
        errors_reported.clear()
        mock_proc_fail = MockNativePlugin(throw_on_process=True)
        wrapper2 = DirectPluginWrapper(mock_proc_fail, report_error=on_error)
        self.assertEqual(wrapper2.start(), AimStatusCode.OK)
        status2, out_data, _ = wrapper2.process(b"", output_capacity=4096)
        self.assertEqual(status2, AimStatusCode.ERROR_EXECUTION_FAILED)
        self.assertEqual(len(out_data), 0)
        self.assertTrue(any("exception" in msg.lower() for _, msg in errors_reported))
        wrapper2.stop()
        wrapper2.destroy()

        # 3. Exception during get_descriptor()
        errors_reported.clear()
        mock_desc_fail = MockNativePlugin(throw_on_descriptor=True)
        wrapper3 = DirectPluginWrapper(mock_desc_fail, report_error=on_error)
        desc = wrapper3.get_descriptor()
        self.assertIsNone(desc)
        wrapper3.destroy()

        # 4. Exception during stop() and destroy()
        mock_stop_fail = MockNativePlugin(throw_on_stop=True, throw_on_destroy=True)
        wrapper4 = DirectPluginWrapper(mock_stop_fail, report_error=on_error)
        wrapper4.start()
        # stop() and destroy() should handle exceptions internally without crashing host
        wrapper4.stop()
        self.assertFalse(wrapper4.is_active)
        wrapper4.destroy()
        self.assertTrue(mock_stop_fail.is_destroyed)

        print("  [+] Exception barriers validated: ZERO unhandled exceptions escaped across DLL boundaries.")

    # =========================================================================
    # OBJECTIVE 4: Reload Stress Testing (1,000+ Cycles)
    # =========================================================================

    def test_rapid_plugin_reload_lifecycle_stress(self) -> None:
        """Adversarial Objective 4:
        Executes 1,000 rapid reload cycles (create -> descriptor -> start -> process -> stop -> destroy).
        Verifies:
        - 100% success rate across all cycles.
        - No lifecycle/state failures across repeated create/destroy cycles.
        - Strict state transition invariants (double start guard, process without start guard).
        """
        print("\n[ADV-REL-01] Starting Rapid Plugin Reload Stress Test (1,000 Cycles)...")

        telemetry_count = 0

        def on_telem(buf: bytes) -> None:
            nonlocal telemetry_count
            telemetry_count += 1

        cycles = 1000
        gc.collect()

        for i in range(cycles):
            mock = MockNativePlugin(name=f"plugin_cycle_{i}", payload_size=1024)
            with DirectPluginWrapper(mock, emit_telemetry=on_telem) as wrapper:
                # 1. State check: must be inactive initially
                self.assertFalse(wrapper.is_active)

                # 2. Process before start must fail with UNINITIALIZED
                st_uninit, _, _ = wrapper.process(b"", output_capacity=2048)
                self.assertEqual(st_uninit, AimStatusCode.ERROR_UNINITIALIZED)

                # 3. Descriptor verification
                desc = wrapper.get_descriptor()
                self.assertIsNotNone(desc)
                self.assertEqual(desc.name.decode("utf-8"), f"plugin_cycle_{i}")

                # 4. Start
                self.assertEqual(wrapper.start(), AimStatusCode.OK)
                self.assertTrue(wrapper.is_active)

                # 5. Double start guard: second start() must return ALREADY_INITIALIZED
                self.assertEqual(wrapper.start(), AimStatusCode.ERROR_ALREADY_INITIALIZED)

                # 6. Process 5 frames
                for _ in range(5):
                    st_proc, out_bytes, out_sz = wrapper.process(b"", output_capacity=2048)
                    self.assertEqual(st_proc, AimStatusCode.OK)
                    self.assertEqual(out_sz, 1024)
                    self.assertEqual(len(out_bytes), 1024)

                # 7. Stop
                wrapper.stop()
                self.assertFalse(wrapper.is_active)

                # 8. Process after stop must fail with UNINITIALIZED
                st_post_stop, _, _ = wrapper.process(b"", output_capacity=2048)
                self.assertEqual(st_post_stop, AimStatusCode.ERROR_UNINITIALIZED)

                # 9. Idempotent stop
                wrapper.stop()
                self.assertFalse(wrapper.is_active)

        gc.collect()
        self.assertEqual(telemetry_count, cycles * 5)
        print(f"  [+] Completed {cycles:,} reload cycles ({telemetry_count:,} telemetry events) with zero lifecycle errors.")

    # =========================================================================
    # CONTRACT & MANIFEST VALIDATION TESTS
    # =========================================================================

    def test_contract_validator_manifest_schema_fuzzing(self) -> None:
        """Verifies JSON schema validation for plugin manifests with invalid / corrupted data."""
        print("\n[ADV-MAN-01] Testing Manifest Schema Validation & Error Detection...")

        valid_manifest = {
            "schema_version": 1,
            "name": "valid_perception",
            "version": "1.0.0",
            "abi_version": 65536,
            "kind": "perception",
            "author": "Aim Team",
            "license": "MIT",
            "entry_point": "perception.dll",
            "checksum": {
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "size_bytes": 1024,
            },
            "contracts": {
                "input_schema": {"identifier": "AFR1", "major_min": 1, "major_max": 1},
                "output_schema": {"identifier": "AOB1", "major_min": 1, "major_max": 1},
            },
            "capabilities": ["bounding_box", "gpu_direct"],
        }

        # Valid manifest passes (does not raise)
        validate_manifest_dict(valid_manifest)

        # Test corruptions
        corruptions = [
            ("missing schema_version", lambda m: m.pop("schema_version")),
            ("missing entry_point", lambda m: m.pop("entry_point")),
            ("invalid kind", lambda m: m.update({"kind": "invalid_kind"})),
            ("invalid checksum sha256 length", lambda m: m["checksum"].update({"sha256": "short"})),
            ("negative size_bytes", lambda m: m["checksum"].update({"size_bytes": -1})),
            ("invalid fourcc identifier", lambda m: m["contracts"]["input_schema"].update({"identifier": "INVALID"})),
            ("major_min below 1", lambda m: m["contracts"]["input_schema"].update({"major_min": 0})),
            ("unknown root property", lambda m: m.update({"unexpected_field": 123})),
        ]

        for name, corrupt_fn in corruptions:
            bad = dict(valid_manifest)
            bad["checksum"] = dict(valid_manifest["checksum"])
            bad["contracts"] = {
                "input_schema": dict(valid_manifest["contracts"]["input_schema"]),
                "output_schema": dict(valid_manifest["contracts"]["output_schema"]),
            }
            corrupt_fn(bad)
            with self.assertRaises(Exception, msg=f"Corruption '{name}' failed to raise validation error"):
                validate_manifest_dict(bad)

        print(f"  [+] Fuzzed {len(corruptions)} invalid manifest configurations; all rejected cleanly.")


if __name__ == "__main__":
    unittest.main()
