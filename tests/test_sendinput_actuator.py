"""Unit tests for SendInput Actuator and Command Contract (Milestone M6-01)."""

from dataclasses import dataclass, field
from enum import IntEnum
import time
from typing import List


class MouseButton(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    MIDDLE = 3


class ButtonAction(IntEnum):
    NONE = 0
    PRESS = 1
    RELEASE = 2
    CLICK = 3


class SubmitResult(IntEnum):
    SUBMITTED = 0
    REJECTED_LATCHED = 1
    REJECTED_STALE = 2
    REJECTED_UNINITIALIZED = 3


@dataclass
class ButtonTransition:
    button: MouseButton = MouseButton.NONE
    action: ButtonAction = ButtonAction.NONE


@dataclass
class ActuationCommand:
    sequence_id: int = 0
    generated_at_ns: int = 0
    desired_apply_time_ns: int = 0
    delta_x_counts: int = 0
    delta_y_counts: int = 0
    button_transition: ButtonTransition = field(
        default_factory=ButtonTransition
    )


class MockSendInputActuator:

    def __init__(self, max_tolerated_lag_ns: int = 10_000_000) -> None:
        self.max_tolerated_lag_ns = max_tolerated_lag_ns
        self.is_initialized = False
        self.is_active = False
        self.is_latched = False
        self.last_sequence_id = 0
        self.pressed_mask = 0
        self.cumulative_x = 0
        self.cumulative_y = 0
        self.dispatches: List[ActuationCommand] = []

    def initialize(self) -> bool:
        self.is_initialized = True
        return True

    def start(self) -> bool:
        if not self.is_initialized or self.is_latched:
            return False
        self.is_active = True
        return True

    def emergency_stop(self) -> None:
        self.is_latched = True
        self.pressed_mask = 0  # Fail-safe release

    def reset_emergency_stop(self) -> None:
        self.is_latched = False

    def submit(self, cmd: ActuationCommand) -> SubmitResult:
        if not self.is_initialized or not self.is_active:
            return SubmitResult.REJECTED_UNINITIALIZED
        if self.is_latched:
            return SubmitResult.REJECTED_LATCHED

        if cmd.sequence_id > 0 and cmd.sequence_id <= self.last_sequence_id:
            return SubmitResult.REJECTED_STALE

        now_ns = time.time_ns()
        target_time = (
            cmd.desired_apply_time_ns
            if cmd.desired_apply_time_ns > 0
            else cmd.generated_at_ns
        )
        if (
            target_time > 0
            and now_ns > target_time + self.max_tolerated_lag_ns
        ):
            return SubmitResult.REJECTED_STALE

        btn = cmd.button_transition
        if btn.button != MouseButton.NONE:
            bit = 1 << int(btn.button)
            if btn.action == ButtonAction.PRESS:
                self.pressed_mask |= bit
            elif btn.action == ButtonAction.RELEASE:
                self.pressed_mask &= ~bit

        self.last_sequence_id = cmd.sequence_id
        self.cumulative_x += cmd.delta_x_counts
        self.cumulative_y += cmd.delta_y_counts
        self.dispatches.append(cmd)
        return SubmitResult.SUBMITTED


def test_sendinput_monotonic_and_bounds() -> None:
    actuator = MockSendInputActuator()
    assert actuator.initialize()
    assert actuator.start()

    cmd1 = ActuationCommand(sequence_id=1, delta_x_counts=10, delta_y_counts=5)
    assert actuator.submit(cmd1) == SubmitResult.SUBMITTED

    # Duplicate sequence must be rejected
    cmd2 = ActuationCommand(sequence_id=1, delta_x_counts=20)
    assert actuator.submit(cmd2) == SubmitResult.REJECTED_STALE

    cmd3 = ActuationCommand(
        sequence_id=2, delta_x_counts=-5, delta_y_counts=15
    )
    assert actuator.submit(cmd3) == SubmitResult.SUBMITTED

    assert actuator.cumulative_x == 5
    assert actuator.cumulative_y == 20


def test_sendinput_emergency_stop_and_release() -> None:
    actuator = MockSendInputActuator()
    assert actuator.initialize()
    assert actuator.start()

    cmd_press = ActuationCommand(
        sequence_id=1,
        button_transition=ButtonTransition(
            MouseButton.LEFT, ButtonAction.PRESS
        ),
    )
    assert actuator.submit(cmd_press) == SubmitResult.SUBMITTED
    assert actuator.pressed_mask & (1 << int(MouseButton.LEFT))

    # Emergency stop releases held buttons
    actuator.emergency_stop()
    assert actuator.is_latched
    assert actuator.pressed_mask == 0

    # Commands rejected while latched
    cmd2 = ActuationCommand(sequence_id=2, delta_x_counts=10)
    assert actuator.submit(cmd2) == SubmitResult.REJECTED_LATCHED
