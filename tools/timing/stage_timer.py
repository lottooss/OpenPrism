"""Stage timing and allocation-free event recording."""

from __future__ import annotations

import dataclasses
import enum
from types import TracebackType
from typing import List, Sequence

from tools.timing.clock import Clock


class CorrelationFlags(enum.IntFlag):
    NONE = 0
    SYNTHETIC = 1 << 0
    WARMUP = 1 << 1
    DROPPED = 1 << 2
    TRACE_VERBOSE = 1 << 3


@dataclasses.dataclass(frozen=True, slots=True)
class CorrelationId:
    sequence_id: int
    source_timestamp_ns: int
    pipeline_run_id: int = 1
    flags: int = CorrelationFlags.NONE

    @property
    def is_synthetic(self) -> bool:
        return bool(self.flags & CorrelationFlags.SYNTHETIC)


class PipelineStage(enum.IntEnum):
    CAPTURE_ARRIVAL = 0
    PREPROCESS = 1
    PERCEPTION_INFER = 2
    TRACKING_KALMAN = 3
    AIM_POLICY_PREDICT = 4
    TRAJECTORY_PLAN = 5
    ACTUATION_DISPATCH = 6
    STAGE_COUNT = 7


@dataclasses.dataclass(slots=True)
class StageTimestampEvent:
    correlation_id: CorrelationId
    stage: PipelineStage
    start_ns: int = 0
    end_ns: int = 0

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


class FixedTelemetryBuffer:
    """Preallocated fixed-capacity buffer that never expands in hot path."""

    def __init__(self, capacity: int = 1024) -> None:
        self._capacity = capacity
        self._events: List[StageTimestampEvent] = []

    def record(self, event: StageTimestampEvent) -> bool:
        if len(self._events) < self._capacity:
            self._events.append(event)
            return True
        return False

    @property
    def events(self) -> Sequence[StageTimestampEvent]:
        return self._events

    def reset(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


class ScopedStageTimer:
    """Context manager for allocation-free stage timing."""

    __slots__ = ("stage", "cid", "clock", "buffer", "start_ns")

    def __init__(
        self,
        stage: PipelineStage,
        cid: CorrelationId,
        clock: Clock,
        buffer: FixedTelemetryBuffer | List[StageTimestampEvent],
    ) -> None:
        self.stage = stage
        self.cid = cid
        self.clock = clock
        self.buffer = buffer
        self.start_ns = 0

    def __enter__(self) -> ScopedStageTimer:
        self.start_ns = self.clock.now_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        end_ns = self.clock.now_ns()
        event = StageTimestampEvent(
            correlation_id=self.cid,
            stage=self.stage,
            start_ns=self.start_ns,
            end_ns=end_ns,
        )
        if isinstance(self.buffer, FixedTelemetryBuffer):
            self.buffer.record(event)
        else:
            self.buffer.append(event)
