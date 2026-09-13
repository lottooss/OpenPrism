"""Capture-to-tensor latency benchmark, percentile evaluation, and telemetry report generator.

Milestone M2-06 acceptance gate tooling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass(frozen=True)
class LatencyPercentiles:
    """Statistical summary of pipeline latency measurements in milliseconds."""

    sample_count: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = 0.0
    avg_ms: float = 0.0
    std_dev_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples_ms: List[float]) -> LatencyPercentiles:
        """Computes percentiles and summary statistics from latency sample list."""
        if not samples_ms:
            return cls()

        arr = np.array(samples_ms, dtype=np.float64)
        return cls(
            sample_count=len(samples_ms),
            p50_ms=float(np.percentile(arr, 50.0)),
            p95_ms=float(np.percentile(arr, 95.0)),
            p99_ms=float(np.percentile(arr, 99.0)),
            max_ms=float(np.max(arr)),
            min_ms=float(np.min(arr)),
            avg_ms=float(np.mean(arr)),
            std_dev_ms=float(np.std(arr)),
        )


@dataclass
class CaptureToTensorReport:
    """Comprehensive milestone benchmark and soak report."""

    pipeline_name: str = "capture_to_tensor"
    target_p99_ms: float = 1.0
    p99_passed: bool = False
    total_frames_processed: int = 0
    frames_acquired: int = 0
    frames_dropped_stale: int = 0
    frames_dropped_invalid: int = 0
    recovery_events_count: int = 0

    capture_to_tensor_latency: LatencyPercentiles = field(default_factory=LatencyPercentiles)
    stage_acquire_latency: LatencyPercentiles = field(default_factory=LatencyPercentiles)
    stage_preprocess_latency: LatencyPercentiles = field(default_factory=LatencyPercentiles)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CaptureToTensorBenchmark:
    """Evaluation harness calculating pipeline acceptance percentiles."""

    def __init__(self, target_p99_ms: float = 1.0, stale_cutoff_ms: float = 10.0) -> None:
        self.target_p99_ms = target_p99_ms
        self.stale_cutoff_ms = stale_cutoff_ms

        self.capture_to_tensor_samples: List[float] = []
        self.acquire_samples: List[float] = []
        self.preprocess_samples: List[float] = []

        self.total_frames = 0
        self.frames_acquired = 0
        self.frames_dropped_stale = 0
        self.frames_dropped_invalid = 0
        self.recovery_events = 0

    def record_frame(
        self,
        capture_arrival_ns: int,
        acquire_duration_ns: int,
        preprocess_duration_ns: int,
        now_ns: int,
        is_valid: bool = True,
    ) -> bool:
        """Records a single frame execution tick."""
        self.total_frames += 1

        if not is_valid:
            self.frames_dropped_invalid += 1
            return False

        self.frames_acquired += 1

        latency_ms = (now_ns - capture_arrival_ns) / 1_000_000.0
        if latency_ms > self.stale_cutoff_ms:
            self.frames_dropped_stale += 1
            return False

        acquire_ms = acquire_duration_ns / 1_000_000.0
        preprocess_ms = preprocess_duration_ns / 1_000_000.0

        self.capture_to_tensor_samples.append(latency_ms)
        self.acquire_samples.append(acquire_ms)
        self.preprocess_samples.append(preprocess_ms)
        return True

    def record_recovery_event(self) -> None:
        """Increments recovery fault transition counter."""
        self.recovery_events += 1

    def generate_report(self) -> CaptureToTensorReport:
        """Compiles complete statistical report and evaluates acceptance criteria."""
        c2t_stats = LatencyPercentiles.from_samples(self.capture_to_tensor_samples)
        acq_stats = LatencyPercentiles.from_samples(self.acquire_samples)
        pre_stats = LatencyPercentiles.from_samples(self.preprocess_samples)

        p99_passed = (
            c2t_stats.sample_count > 0 and c2t_stats.p99_ms <= self.target_p99_ms
        )

        return CaptureToTensorReport(
            target_p99_ms=self.target_p99_ms,
            p99_passed=p99_passed,
            total_frames_processed=self.total_frames,
            frames_acquired=self.frames_acquired,
            frames_dropped_stale=self.frames_dropped_stale,
            frames_dropped_invalid=self.frames_dropped_invalid,
            recovery_events_count=self.recovery_events,
            capture_to_tensor_latency=c2t_stats,
            stage_acquire_latency=acq_stats,
            stage_preprocess_latency=pre_stats,
        )
