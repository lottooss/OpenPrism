"""End-to-end hot loop telemetry structures."""

from __future__ import annotations

import dataclasses
from typing import Dict, Any

from tools.timing.stage_timer import CorrelationId


@dataclasses.dataclass(slots=True)
class HotLoopTelemetryEvent:
    correlation_id: CorrelationId
    capture_arrival_ns: int = 0
    preprocess_done_ns: int = 0
    inference_done_ns: int = 0
    tracking_done_ns: int = 0
    policy_done_ns: int = 0
    dispatch_done_ns: int = 0
    detected_targets: int = 0
    tracked_targets: int = 0
    dropped_or_stale: bool = False

    @property
    def total_latency_ns(self) -> int:
        return max(0, self.dispatch_done_ns - self.capture_arrival_ns)

    @property
    def total_latency_ms(self) -> float:
        return self.total_latency_ns / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.correlation_id.sequence_id,
            "source_timestamp_ns": self.correlation_id.source_timestamp_ns,
            "pipeline_run_id": self.correlation_id.pipeline_run_id,
            "flags": int(self.correlation_id.flags),
            "capture_arrival_ns": self.capture_arrival_ns,
            "preprocess_done_ns": self.preprocess_done_ns,
            "inference_done_ns": self.inference_done_ns,
            "tracking_done_ns": self.tracking_done_ns,
            "policy_done_ns": self.policy_done_ns,
            "dispatch_done_ns": self.dispatch_done_ns,
            "total_latency_ms": self.total_latency_ms,
            "detected_targets": self.detected_targets,
            "tracked_targets": self.tracked_targets,
            "dropped_or_stale": self.dropped_or_stale,
        }
