"""Unit tests for DXGI and WGC capture capability probe."""

import json
import os
import unittest

from tools.probe.capture_probe import (
    CaptureCapabilityReport,
    DxgiAdapterCaptureProbe,
    WgcCapabilityProbe,
    compute_cadence_metrics,
    get_hresult_name,
    probe_wgc_capabilities,
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_INVALID_CALL,
    DXGI_ERROR_UNSUPPORTED,
    DXGI_ERROR_WAIT_TIMEOUT,
    E_ACCESSDENIED,
    S_OK,
)


class TestCaptureProbe(unittest.TestCase):
    """Test suite for capture capabilities and metrics calculation."""

    def test_hresult_naming(self) -> None:
        """Verify HRESULT error codes are decoded to human-readable names."""
        self.assertIn("S_OK", get_hresult_name(S_OK))
        self.assertEqual(get_hresult_name(DXGI_ERROR_ACCESS_LOST), "DXGI_ERROR_ACCESS_LOST")
        self.assertEqual(get_hresult_name(DXGI_ERROR_WAIT_TIMEOUT), "DXGI_ERROR_WAIT_TIMEOUT")
        self.assertEqual(get_hresult_name(DXGI_ERROR_INVALID_CALL), "DXGI_ERROR_INVALID_CALL")
        self.assertEqual(get_hresult_name(DXGI_ERROR_UNSUPPORTED), "DXGI_ERROR_UNSUPPORTED")
        self.assertIn("E_ACCESSDENIED", get_hresult_name(E_ACCESSDENIED))

    def test_cadence_metrics_computation(self) -> None:
        """Verify 144 Hz frame cadence metrics computation."""
        cad = compute_cadence_metrics(refresh_hz=144, samples=100)
        self.assertEqual(cad.target_refresh_hz, 144)
        self.assertAlmostEqual(cad.nominal_interval_ms, 6.944, places=3)
        self.assertEqual(cad.total_samples, 100)
        self.assertTrue(cad.cadence_stable_144hz)
        self.assertGreater(cad.p50_interval_ms, 6.0)
        self.assertLess(cad.p50_interval_ms, 8.0)
        self.assertGreaterEqual(cad.p99_interval_ms, cad.p50_interval_ms)

        # Invalid arguments must raise ValueError
        with self.assertRaises(ValueError):
            compute_cadence_metrics(refresh_hz=0, samples=100)
        with self.assertRaises(ValueError):
            compute_cadence_metrics(refresh_hz=144, samples=0)

    def test_wgc_capabilities(self) -> None:
        """Verify WGC capability detection."""
        wgc = probe_wgc_capabilities()
        self.assertEqual(wgc.is_supported_on_os, os.name == "nt")
        self.assertEqual(wgc.os_build_minimum_met, os.name == "nt")
        self.assertEqual(wgc.graphics_capture_session_api_present, os.name == "nt")
        self.assertTrue(wgc.supports_cross_adapter_interop)
        self.assertTrue(wgc.recommended_for_hybrid_laptops)

    def test_capture_report_serialization(self) -> None:
        """Verify CaptureCapabilityReport serialization to dictionary and valid JSON."""
        dxgi_probe = DxgiAdapterCaptureProbe(
            adapter_index=0,
            adapter_name="AMD Radeon(TM) Graphics",
            adapter_luid="0x00000000:00000001",
            has_outputs=True,
            output_count=1,
            d3d11_device_created=True,
            d3d11_feature_level="0xb000",
            duplication_supported=False,
            duplication_hresult_hex="0x80070005",
            duplication_hresult_name="E_ACCESSDENIED",
            source_width_px=1920,
            source_height_px=1080,
            refresh_rate_hz=144,
            format_name="DXGI_FORMAT_B8G8R8A8_UNORM (87)",
            is_driving_display=True,
            requires_cross_adapter_copy=True,
        )

        wgc_probe = WgcCapabilityProbe(
            is_supported_on_os=True,
            os_build_minimum_met=True,
            graphics_capture_session_api_present=True,
            is_cursor_capture_toggle_supported=True,
            is_border_toggle_supported=True,
            supports_cross_adapter_interop=True,
            recommended_for_hybrid_laptops=True,
        )

        cadence = compute_cadence_metrics(refresh_hz=144, samples=50)

        report = CaptureCapabilityReport(
            schema_version=1,
            timestamp_utc="2026-08-23T20:00:00Z",
            target_host="EXAMPLE-HOST",
            os_caption="Windows 11",
            active_power_scheme="Balanced",
            dxgi_probes=[dxgi_probe],
            wgc_probe=wgc_probe,
            cadence_metrics=cadence,
            primary_capture_recommendation="DXGI on iGPU with staging copy",
            fallback_capture_recommendation="WGC GraphicsCaptureSession",
            zero_copy_feasible_now=False,
            notes=["Note 1", "Note 2"],
        )

        d = report.to_dict()
        self.assertEqual(d["schema_version"], 1)
        self.assertEqual(d["target_host"], "EXAMPLE-HOST")
        self.assertEqual(len(d["dxgi_probes"]), 1)

        raw_json = report.to_json()
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["cadence_metrics"]["target_refresh_hz"], 144)


if __name__ == "__main__":
    unittest.main()
