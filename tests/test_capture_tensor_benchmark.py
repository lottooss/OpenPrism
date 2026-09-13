"""Unit tests for capture-to-tensor benchmarking and acceptance reporting.

Milestone M2-06 test suite.
"""

from __future__ import annotations

import json
import unittest

from tools.probe.capture_tensor_benchmark import (
    CaptureToTensorBenchmark,
    LatencyPercentiles,
)


class TestCaptureTensorBenchmark(unittest.TestCase):
    """Test suite for CaptureToTensorBenchmark and statistical aggregations."""

    def test_empty_samples(self) -> None:
        """Verify handling of zero samples."""
        stats = LatencyPercentiles.from_samples([])
        self.assertEqual(stats.sample_count, 0)
        self.assertEqual(stats.p50_ms, 0.0)
        self.assertEqual(stats.p99_ms, 0.0)

    def test_percentile_calculation_accuracy(self) -> None:
        """Verify accurate calculation of p50, p95, p99, min, max, avg."""
        # 100 samples from 0.01 to 1.00 ms
        samples = [i * 0.01 for i in range(1, 101)]
        stats = LatencyPercentiles.from_samples(samples)

        self.assertEqual(stats.sample_count, 100)
        self.assertAlmostEqual(stats.min_ms, 0.01, delta=1e-4)
        self.assertAlmostEqual(stats.max_ms, 1.00, delta=1e-4)
        self.assertAlmostEqual(stats.avg_ms, 0.505, delta=1e-4)
        self.assertAlmostEqual(stats.p50_ms, 0.505, delta=1e-2)
        self.assertAlmostEqual(stats.p95_ms, 0.9505, delta=1e-2)
        self.assertAlmostEqual(stats.p99_ms, 0.9901, delta=1e-2)

    def test_benchmark_acceptance_gate(self) -> None:
        """Verify benchmark reports passing status when p99 <= 1.0 ms."""
        bench = CaptureToTensorBenchmark(target_p99_ms=1.0, stale_cutoff_ms=10.0)

        # 1000 simulated frames with 0.45 ms total latency
        for i in range(1000):
            t_arrival = i * 6_944_444
            t_acq = 100_000      # 0.1 ms
            t_pre = 350_000      # 0.35 ms
            t_now = t_arrival + t_acq + t_pre # 0.45 ms total
            ok = bench.record_frame(t_arrival, t_acq, t_pre, t_now, is_valid=True)
            self.assertTrue(ok)

        report = bench.generate_report()
        self.assertTrue(report.p99_passed)
        self.assertEqual(report.frames_acquired, 1000)
        self.assertEqual(report.frames_dropped_stale, 0)
        self.assertEqual(report.frames_dropped_invalid, 0)
        self.assertAlmostEqual(report.capture_to_tensor_latency.p99_ms, 0.45, delta=1e-3)

    def test_stale_and_invalid_frame_filtering(self) -> None:
        """Verify stale frames (>10 ms) and invalid frames are recorded and rejected."""
        bench = CaptureToTensorBenchmark(target_p99_ms=1.0, stale_cutoff_ms=10.0)

        # Valid frame (0.5 ms old)
        ok1 = bench.record_frame(1_000_000, 100_000, 400_000, 1_500_000, is_valid=True)
        self.assertTrue(ok1)

        # Stale frame (15.0 ms old > 10.0 ms cutoff)
        ok2 = bench.record_frame(1_000_000, 100_000, 400_000, 16_000_000, is_valid=True)
        self.assertFalse(ok2)

        # Invalid frame (e.g. faulted capture)
        ok3 = bench.record_frame(1_000_000, 0, 0, 1_000_000, is_valid=False)
        self.assertFalse(ok3)

        bench.record_recovery_event()

        report = bench.generate_report()
        self.assertEqual(report.total_frames_processed, 3)
        self.assertEqual(report.frames_acquired, 2)
        self.assertEqual(report.frames_dropped_stale, 1)
        self.assertEqual(report.frames_dropped_invalid, 1)
        self.assertEqual(report.recovery_events_count, 1)

    def test_json_serialization(self) -> None:
        """Verify report serializes cleanly to JSON."""
        bench = CaptureToTensorBenchmark()
        bench.record_frame(1_000_000, 100_000, 300_000, 1_400_000, is_valid=True)
        report = bench.generate_report()

        raw_json = report.to_json()
        parsed = json.loads(raw_json)

        self.assertEqual(parsed["pipeline_name"], "capture_to_tensor")
        self.assertEqual(parsed["target_p99_ms"], 1.0)
        self.assertTrue(parsed["p99_passed"])
        self.assertEqual(parsed["total_frames_processed"], 1)


if __name__ == "__main__":
    unittest.main()
