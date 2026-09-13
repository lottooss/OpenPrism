"""Emergency Stop latch and safety evaluation contracts."""

from __future__ import annotations

import dataclasses
import enum


class SafetyReason(enum.IntEnum):
    NONE = 0
    EMERGENCY_STOP_TRIGGERED = 1
    STALE_DATA = 2
    LOST_FOCUS = 3
    INVALID_CALIBRATION = 4
    OUT_OF_BOUNDS = 5
    DRIVER_ERROR = 6
    OPERATOR_REQUESTED = 7


@dataclasses.dataclass(frozen=True, slots=True)
class ResetToken:
    token_id: str
    generated_at_ns: int = 0
    is_valid: bool = True


class EmergencyStopLatch:
    """Thread-safe latched emergency-stop state machine."""

    def __init__(self) -> None:
        self._is_latched = False
        self._reason = SafetyReason.NONE

    def trigger(self, reason: SafetyReason = SafetyReason.EMERGENCY_STOP_TRIGGERED) -> None:
        self._reason = reason
        self._is_latched = True

    @property
    def is_latched(self) -> bool:
        return self._is_latched

    @property
    def reason(self) -> SafetyReason:
        return self._reason

    def try_reset(self, token: ResetToken) -> bool:
        if not token.is_valid or not token.token_id.strip():
            return False
        self._reason = SafetyReason.NONE
        self._is_latched = False
        return True
