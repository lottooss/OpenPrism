"""Unit tests for the hardware probe and topology evaluation logic."""

import json
import unittest

from tools.probe.hardware_probe import (
    CpuInfo,
    CudaDeviceInfo,
    DisplayOutputInfo,
    EnvironmentManifest,
    GpuAdapterInfo,
    PowerInfo,
    SystemInfo,
    evaluate_topology,
    format_luid,
)


class TestHardwareProbe(unittest.TestCase):
    """Test suite for deterministic hardware probe and topology analysis."""

    def setUp(self) -> None:
        self.igpu_luid = "0x00000000:00000001"
        self.dgpu_luid = "0x00000000:00000002"

        self.amd_gpu = GpuAdapterInfo(
            index=0,
            description="AMD Radeon(TM) Graphics",
            vendor_id_hex="0x1002",
            device_id_hex="0x1681",
            sub_sys_id_hex="0x00000000",
            revision=0,
            dedicated_video_memory_mb=483,
            dedicated_system_memory_mb=0,
            shared_system_memory_mb=7805,
            luid_low=0x00000001,
            luid_high=0,
            luid=self.igpu_luid,
            is_software=False,
        )

        self.nvidia_gpu = GpuAdapterInfo(
            index=1,
            description="NVIDIA GeForce RTX 4060 Laptop GPU",
            vendor_id_hex="0x10de",
            device_id_hex="0x28e0",
            sub_sys_id_hex="0x00000000",
            revision=0,
            dedicated_video_memory_mb=7956,
            dedicated_system_memory_mb=0,
            shared_system_memory_mb=7805,
            luid_low=0x00000002,
            luid_high=0,
            luid=self.dgpu_luid,
            is_software=False,
        )

        self.cuda_rtx4060 = CudaDeviceInfo(
            device_index=0,
            name="NVIDIA GeForce RTX 4060 Laptop GPU",
            compute_capability_major=8,
            compute_capability_minor=9,
            multiprocessor_count=24,
            clock_rate_mhz=2250.0,
            total_memory_mb=8187,
            luid_low=0x00000002,
            luid_high=0,
            luid=self.dgpu_luid,
            pci_bus_id="00000000:01:00.0",
        )

    def test_format_luid(self) -> None:
        """Verify LUID formatting into normalized hex strings."""
        self.assertEqual(format_luid(0x1, 0), "0x00000000:00000001")
        self.assertEqual(format_luid(0x2, 0), "0x00000000:00000002")
        self.assertEqual(format_luid(0xDEADBEEF, 0x12345678), "0x12345678:deadbeef")

    def test_same_adapter_topology(self) -> None:
        """Verify topology analysis when display and CUDA share the same adapter (direct zero-copy)."""
        display = DisplayOutputInfo(
            device_name="\\\\.\\DISPLAY1",
            monitor_name="Generic PnP Monitor",
            is_attached=True,
            is_primary=True,
            width_px=1920,
            height_px=1080,
            refresh_rate_hz=144,
            bits_per_pixel=32,
            desktop_bounds={"left": 0, "top": 0, "right": 1920, "bottom": 1080},
            adapter_luid=self.dgpu_luid,
            adapter_description="NVIDIA GeForce RTX 4060 Laptop GPU",
        )

        result = evaluate_topology([display], [self.nvidia_gpu], [self.cuda_rtx4060])

        self.assertEqual(result.topology_type, "SAME_ADAPTER_DIRECT")
        self.assertTrue(result.luid_match)
        self.assertTrue(result.zero_copy_capture_possible)
        self.assertFalse(result.requires_cross_adapter_fallback)
        self.assertEqual(result.primary_display_adapter_luid, self.dgpu_luid)
        self.assertEqual(result.primary_cuda_adapter_luid, self.dgpu_luid)

    def test_cross_adapter_hybrid_topology(self) -> None:
        """Verify topology analysis for hybrid laptop (iGPU display + dGPU CUDA)."""
        display = DisplayOutputInfo(
            device_name="\\\\.\\DISPLAY1",
            monitor_name="Generic PnP Monitor",
            is_attached=True,
            is_primary=True,
            width_px=1920,
            height_px=1080,
            refresh_rate_hz=144,
            bits_per_pixel=32,
            desktop_bounds={"left": 0, "top": 0, "right": 1920, "bottom": 1080},
            adapter_luid=self.igpu_luid,
            adapter_description="AMD Radeon(TM) Graphics",
        )

        result = evaluate_topology(
            [display], [self.amd_gpu, self.nvidia_gpu], [self.cuda_rtx4060]
        )

        self.assertEqual(result.topology_type, "CROSS_ADAPTER_HYBRID")
        self.assertFalse(result.luid_match)
        self.assertFalse(result.zero_copy_capture_possible)
        self.assertTrue(result.requires_cross_adapter_fallback)
        self.assertEqual(result.primary_display_adapter_luid, self.igpu_luid)
        self.assertEqual(result.primary_cuda_adapter_luid, self.dgpu_luid)
        self.assertIn("Hybrid GPU laptop topology detected", result.diagnosis)
        self.assertTrue(len(result.recommendations) >= 2)

    def test_no_attached_displays(self) -> None:
        """Verify topology analysis when no displays are attached (headless)."""
        result = evaluate_topology([], [self.nvidia_gpu], [self.cuda_rtx4060])

        self.assertEqual(result.topology_type, "NO_ATTACHED_DISPLAYS")
        self.assertFalse(result.luid_match)
        self.assertFalse(result.zero_copy_capture_possible)
        self.assertTrue(result.requires_cross_adapter_fallback)

    def test_all_displays_detached(self) -> None:
        """Verify topology analysis when display list is non-empty but all displays are detached."""
        detached_display = DisplayOutputInfo(
            device_name="\\\\.\\DISPLAY1",
            monitor_name="Detached Monitor",
            is_attached=False,
            is_primary=False,
            width_px=0,
            height_px=0,
            refresh_rate_hz=0,
            bits_per_pixel=0,
            desktop_bounds={"left": 0, "top": 0, "right": 0, "bottom": 0},
            adapter_luid=self.dgpu_luid,
            adapter_description="NVIDIA GeForce RTX 4060 Laptop GPU",
        )

        result = evaluate_topology([detached_display], [self.nvidia_gpu], [self.cuda_rtx4060])
        self.assertEqual(result.topology_type, "NO_ATTACHED_DISPLAYS")
        self.assertFalse(result.zero_copy_capture_possible)

    def test_no_cuda_device(self) -> None:
        """Verify topology analysis when no CUDA device is detected."""
        display = DisplayOutputInfo(
            device_name="\\\\.\\DISPLAY1",
            monitor_name="Generic PnP Monitor",
            is_attached=True,
            is_primary=True,
            width_px=1920,
            height_px=1080,
            refresh_rate_hz=144,
            bits_per_pixel=32,
            desktop_bounds={"left": 0, "top": 0, "right": 1920, "bottom": 1080},
            adapter_luid=self.igpu_luid,
            adapter_description="AMD Radeon(TM) Graphics",
        )

        result = evaluate_topology([display], [self.amd_gpu], [])

        self.assertEqual(result.topology_type, "NO_CUDA_DEVICE")
        self.assertFalse(result.luid_match)
        self.assertFalse(result.zero_copy_capture_possible)
        self.assertFalse(result.requires_cross_adapter_fallback)
        self.assertIn("No NVIDIA CUDA device", result.diagnosis)

    def test_environment_manifest_serialization(self) -> None:
        """Verify EnvironmentManifest serialization to dictionary and valid JSON."""
        sys_info = SystemInfo(
            os_caption="Microsoft Windows 11 Home",
            os_version="10.0.26100",
            os_build="26100",
            architecture="AMD64",
            hostname="EXAMPLE-HOST",
            probe_timestamp_utc="2026-08-23T19:50:00Z",
        )
        cpu_info = CpuInfo(
            name="AMD Ryzen 7 7735HS with Radeon Graphics",
            cores=8,
            logical_processors=16,
            max_clock_mhz=3201,
            architecture="AMD64",
        )
        power_info = PowerInfo(
            ac_line_status="Online",
            battery_percent=80,
            power_scheme_guid="381b4222-f694-41f0-9685-ff5bb260df2e",
            power_scheme_name="Balanced",
        )
        display = DisplayOutputInfo(
            device_name="\\\\.\\DISPLAY1",
            monitor_name="Example Monitor",
            is_attached=True,
            is_primary=True,
            width_px=1920,
            height_px=1080,
            refresh_rate_hz=144,
            bits_per_pixel=32,
            desktop_bounds={"left": 0, "top": 0, "right": 1920, "bottom": 1080},
            adapter_luid=self.igpu_luid,
            adapter_description="AMD Radeon(TM) Graphics",
        )
        top = evaluate_topology([display], [self.amd_gpu, self.nvidia_gpu], [self.cuda_rtx4060])

        manifest = EnvironmentManifest(
            schema_version=1,
            system=sys_info,
            cpu=cpu_info,
            gpus=[self.amd_gpu, self.nvidia_gpu],
            displays=[display],
            cuda_devices=[self.cuda_rtx4060],
            power=power_info,
            topology=top,
        )

        manifest_dict = manifest.to_dict()
        self.assertEqual(manifest_dict["schema_version"], 1)
        self.assertEqual(manifest_dict["system"]["hostname"], "EXAMPLE-HOST")
        self.assertEqual(manifest_dict["cpu"]["cores"], 8)
        self.assertEqual(manifest_dict["topology"]["topology_type"], "CROSS_ADAPTER_HYBRID")

        raw_json = manifest.to_json()
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["topology"]["luid_match"], False)
        self.assertEqual(parsed["displays"][0]["refresh_rate_hz"], 144)


if __name__ == "__main__":
    unittest.main()
