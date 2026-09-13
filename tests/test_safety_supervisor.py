"""Unit tests for centralized fail-closed Safety Supervisor (Milestone M6-05)."""

from dataclasses import dataclass
from enum import IntEnum


class SafetyReason(IntEnum):
    NONE = 0
    EMERGENCY_STOP_TRIGGERED = 1
    STALE_DATA = 2
    LOST_FOCUS = 3
    INVALID_CALIBRATION = 4
    OUT_OF_BOUNDS = 5


@dataclass
class ActuationCommand:
    sequence_id: int = 0
    generated_at_ns: int = 0
    delta_x_counts: int = 0
    delta_y_counts: int = 0


class SafetySupervisor:

    def __init__(
        self,
        max_command_age_ns: int = 10_000_000,
        max_single_step_counts: int = 150,
        authorized_token: str = "super_secret_token",
    ) -> None:
        self.max_command_age_ns = max_command_age_ns
        self.max_counts = max_single_step_counts
        self.authorized_token = authorized_token
        self.is_latched = False
        self.last_reason: SafetyReason = SafetyReason.NONE
        self.last_seq = 0
        self.is_foreground = True
        self.is_calibrated = True

    def trigger_emergency_stop(self, reason: SafetyReason) -> None:
        self.is_latched = True
        self.last_reason = reason

    def try_reset(self, token: str) -> bool:
        if token == self.authorized_token:
            self.is_latched = False
            self.last_reason = SafetyReason.NONE
            return True
        return False

    def check_actuation_safety(self, cmd: ActuationCommand, now_ns: int) -> bool:
        if self.is_latched:
            return False

        if not self.is_foreground:
            self.trigger_emergency_stop(SafetyReason.LOST_FOCUS)
            return False

        if not self.is_calibrated:
            self.trigger_emergency_stop(SafetyReason.INVALID_CALIBRATION)
            return False

        if cmd.sequence_id > 0 and cmd.sequence_id <= self.last_seq:
            self.last_reason = SafetyReason.STALE_DATA
            return False
        self.last_seq = cmd.sequence_id

        if cmd.generated_at_ns > 0 and now_ns > cmd.generated_at_ns + self.max_command_age_ns:
            self.last_reason = SafetyReason.STALE_DATA
            return False

        if abs(cmd.delta_x_counts) > self.max_counts or abs(cmd.delta_y_counts) > self.max_counts:
            self.trigger_emergency_stop(SafetyReason.OUT_OF_BOUNDS)
            return False

        return True


    def get_last_reason(self) -> SafetyReason:
        return self.last_reason


def test_supervisor_all_gates() -> None:
    supervisor = SafetySupervisor()

    # 1. Valid command passes
    cmd1 = ActuationCommand(sequence_id=1, generated_at_ns=1_000_000_000, delta_x_counts=20)
    assert supervisor.check_actuation_safety(cmd1, now_ns=1_005_000_000)

    # 2. Duplicate sequence fails
    assert not supervisor.check_actuation_safety(cmd1, now_ns=1_006_000_000)
    assert supervisor.get_last_reason() == SafetyReason.STALE_DATA

    # 3. Oversized command latches e-stop
    cmd2 = ActuationCommand(sequence_id=2, generated_at_ns=1_010_000_000, delta_x_counts=500)
    assert not supervisor.check_actuation_safety(cmd2, now_ns=1_011_000_000)
    assert supervisor.is_latched
    assert supervisor.get_last_reason() == SafetyReason.OUT_OF_BOUNDS

    # 4. Unauthorized reset fails
    assert not supervisor.try_reset("wrong")
    assert supervisor.is_latched

    # 5. Authorized reset succeeds
    assert supervisor.try_reset("super_secret_token")
    assert not supervisor.is_latched
