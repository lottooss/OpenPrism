"""Unit tests for monotonic clock, QPC conversion, stage timing, and correlation ID propagation."""

import unittest

from tools.timing.clock import FakeClock, QpcClock, qpc_to_ns_euclidean
from tools.timing.stage_timer import (
    CorrelationFlags,
    CorrelationId,
    FixedTelemetryBuffer,
    PipelineStage,
    ScopedStageTimer,
    StageTimestampEvent,
)
from tools.timing.telemetry import HotLoopTelemetryEvent


class TestClockAndTiming(unittest.TestCase):
    """Test suite for timing core, clock conversions, and stage timing."""

    def test_qpc_to_ns_euclidean_precision(self) -> None:
        """Verify QPC to nanoseconds conversion for 10 MHz and non-10 MHz frequencies."""
        # 10 MHz (standard Windows invariant TSC QPF)
        qpf_10mhz = 10_000_000
        self.assertEqual(qpc_to_ns_euclidean(0, qpf_10mhz), 0)
        self.assertEqual(qpc_to_ns_euclidean(10_000_000, qpf_10mhz), 1_000_000_000)
        self.assertEqual(qpc_to_ns_euclidean(5_000_000, qpf_10mhz), 500_000_000)
        self.assertEqual(qpc_to_ns_euclidean(1234, qpf_10mhz), 123_400)

        # 24 MHz (HPET / non-standard QPF)
        qpf_24mhz = 24_000_000
        self.assertEqual(qpc_to_ns_euclidean(24_000_000, qpf_24mhz), 1_000_000_000)
        self.assertEqual(qpc_to_ns_euclidean(12_000_000, qpf_24mhz), 500_000_000)

        # Edge case 0 QPF
        self.assertEqual(qpc_to_ns_euclidean(100, 0), 0)

    def test_qpc_overflow_safety(self) -> None:
        """Verify overflow safety at high QPC values that would overflow naive 64-bit uint multiplication."""
        qpf_10mhz = 10_000_000

        # Naive (qpc * 10^9) overflows at qpc >= 18_446_744_073 (approx 30.7 minutes uptime)
        # Test 1 hour, 1 day, 100 days, 10 years of uptime
        one_hour_qpc = 3600 * qpf_10mhz
        one_day_qpc = 86400 * qpf_10mhz
        ten_years_qpc = 10 * 365 * 86400 * qpf_10mhz

        self.assertEqual(qpc_to_ns_euclidean(one_hour_qpc, qpf_10mhz), 3_600_000_000_000)
        self.assertEqual(qpc_to_ns_euclidean(one_day_qpc, qpf_10mhz), 86_400_000_000_000)
        self.assertEqual(
            qpc_to_ns_euclidean(ten_years_qpc, qpf_10mhz),
            10 * 365 * 86400 * 1_000_000_000,
        )

    def test_fake_clock_monotonicity(self) -> None:
        """Verify FakeClock deterministic step, advance, and monotonic invariant."""
        clock = FakeClock(initial_ns=1_000_000_000)
        self.assertEqual(clock.now_ns(), 1_000_000_000)

        clock.advance_ns(500_000)
        self.assertEqual(clock.now_ns(), 1_000_500_000)

        clock.advance_ms(2.5)
        self.assertEqual(clock.now_ns(), 1_003_000_000)

        clock.set_ns(2_000_000_000)
        self.assertEqual(clock.now_ns(), 2_000_000_000)

        with self.assertRaises(ValueError):
            clock.advance_ns(-100)

        with self.assertRaises(ValueError):
            clock.set_ns(1_500_000_000)

    def test_qpc_clock_monotonicity(self) -> None:
        """Verify real QpcClock produces strictly non-decreasing timestamps."""
        clock = QpcClock()
        t1 = clock.now_ns()
        self.assertGreater(t1, 0)

        prev = t1
        for _ in range(1000):
            current = clock.now_ns()
            self.assertGreaterEqual(current, prev, "QpcClock must be non-decreasing")
            prev = current

    def test_scoped_stage_timer(self) -> None:
        """Verify ScopedStageTimer records exact durations with FakeClock."""
        clock = FakeClock(initial_ns=100_000_000)
        buffer = FixedTelemetryBuffer(capacity=10)
        cid = CorrelationId(sequence_id=42, source_timestamp_ns=100_000_000, pipeline_run_id=1)

        with ScopedStageTimer(PipelineStage.PREPROCESS, cid, clock, buffer):
            clock.advance_ns(300_000)  # 0.3 ms

        self.assertEqual(len(buffer), 1)
        event = buffer.events[0]
        self.assertEqual(event.stage, PipelineStage.PREPROCESS)
        self.assertEqual(event.start_ns, 100_000_000)
        self.assertEqual(event.end_ns, 100_300_000)
        self.assertEqual(event.duration_ns, 300_000)
        self.assertAlmostEqual(event.duration_ms, 0.3, places=4)
        self.assertEqual(event.correlation_id.sequence_id, 42)

    def test_fixed_telemetry_buffer_capacity(self) -> None:
        """Verify FixedTelemetryBuffer bounds capacity and drops excess without error."""
        buffer = FixedTelemetryBuffer(capacity=3)
        cid = CorrelationId(sequence_id=1, source_timestamp_ns=100)
        event = StageTimestampEvent(correlation_id=cid, stage=PipelineStage.PREPROCESS, start_ns=100, end_ns=200)

        self.assertTrue(buffer.record(event))
        self.assertTrue(buffer.record(event))
        self.assertTrue(buffer.record(event))
        # 4th record must be dropped (returns False)
        self.assertFalse(buffer.record(event))
        self.assertEqual(len(buffer), 3)

        buffer.reset()
        self.assertEqual(len(buffer), 0)

    def test_synthetic_pipeline_correlation_propagation(self) -> None:
        """Verify correlation ID survives unmodified across an entire 6-stage aiming pipeline."""
        clock = FakeClock(initial_ns=1_000_000_000)
        buffer = FixedTelemetryBuffer(capacity=32)

        capture_arrival_ns = clock.now_ns()
        cid = CorrelationId(
            sequence_id=101,
            source_timestamp_ns=capture_arrival_ns,
            pipeline_run_id=7,
            flags=CorrelationFlags.SYNTHETIC,
        )

        # Stage 1: Capture acquire
        with ScopedStageTimer(PipelineStage.CAPTURE_ARRIVAL, cid, clock, buffer):
            clock.advance_ns(150_000)  # 0.15 ms

        # Stage 2: Fused GPU preprocess
        with ScopedStageTimer(PipelineStage.PREPROCESS, cid, clock, buffer):
            clock.advance_ns(300_000)  # 0.30 ms

        # Stage 3: TensorRT inference & decode
        with ScopedStageTimer(PipelineStage.PERCEPTION_INFER, cid, clock, buffer):
            clock.advance_ns(1_800_000)  # 1.80 ms

        # Stage 4: Kalman association & tracking
        with ScopedStageTimer(PipelineStage.TRACKING_KALMAN, cid, clock, buffer):
            clock.advance_ns(50_000)  # 0.05 ms

        # Stage 5: Policy target selection & prediction
        with ScopedStageTimer(PipelineStage.AIM_POLICY_PREDICT, cid, clock, buffer):
            clock.advance_ns(60_000)  # 0.06 ms

        # Stage 6: Trajectory planning & actuation dispatch
        with ScopedStageTimer(PipelineStage.ACTUATION_DISPATCH, cid, clock, buffer):
            clock.advance_ns(80_000)  # 0.08 ms

        events = buffer.events
        self.assertEqual(len(events), 6)

        expected_stages = [
            PipelineStage.CAPTURE_ARRIVAL,
            PipelineStage.PREPROCESS,
            PipelineStage.PERCEPTION_INFER,
            PipelineStage.TRACKING_KALMAN,
            PipelineStage.AIM_POLICY_PREDICT,
            PipelineStage.ACTUATION_DISPATCH,
        ]

        for i, ev in enumerate(events):
            self.assertEqual(ev.stage, expected_stages[i])
            # Invariant: CorrelationId is identical and intact
            self.assertEqual(ev.correlation_id.sequence_id, 101)
            self.assertEqual(ev.correlation_id.source_timestamp_ns, 1_000_000_000)
            self.assertEqual(ev.correlation_id.pipeline_run_id, 7)
            self.assertTrue(ev.correlation_id.is_synthetic)

        # End-to-end total latency from capture arrival to dispatch
        total_latency_ns = events[-1].end_ns - cid.source_timestamp_ns
        total_latency_ms = total_latency_ns / 1_000_000.0

        self.assertEqual(total_latency_ns, 2_440_000)
        self.assertAlmostEqual(total_latency_ms, 2.44, places=2)
        # Verify it meets the <= 6 ms p99 engineering target
        self.assertLessEqual(total_latency_ms, 6.0)

    def test_hot_loop_telemetry_event(self) -> None:
        """Verify HotLoopTelemetryEvent construction, total latency calculation, and dict serialization."""
        cid = CorrelationId(sequence_id=55, source_timestamp_ns=1_000_000_000)
        telemetry = HotLoopTelemetryEvent(
            correlation_id=cid,
            capture_arrival_ns=1_000_000_000,
            preprocess_done_ns=1_000_300_000,
            inference_done_ns=1_002_100_000,
            tracking_done_ns=1_002_150_000,
            policy_done_ns=1_002_210_000,
            dispatch_done_ns=1_002_290_000,
            detected_targets=3,
            tracked_targets=2,
            dropped_or_stale=False,
        )

        self.assertEqual(telemetry.total_latency_ns, 2_290_000)
        self.assertAlmostEqual(telemetry.total_latency_ms, 2.29, places=3)

        d = telemetry.to_dict()
        self.assertEqual(d["sequence_id"], 55)
        self.assertEqual(d["detected_targets"], 3)
        self.assertEqual(d["total_latency_ms"], 2.29)


if __name__ == "__main__":
    unittest.main()
