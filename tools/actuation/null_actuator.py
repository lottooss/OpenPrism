"""NullActuator implementation recording commands without OS input."""

from __future__ import annotations

import dataclasses
import enum
from typing import List, Optional

from tools.actuation.safety import EmergencyStopLatch, ResetToken, SafetyReason
from tools.timing.stage_timer import CorrelationId


class MouseButton(enum.IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    MIDDLE = 3
    EXTRA1 = 4
    EXTRA2 = 5


class ButtonAction(enum.IntEnum):
    NONE = 0
    PRESS = 1
    RELEASE = 2
    CLICK = 3


@dataclasses.dataclass(frozen=True, slots=True)
class ButtonTransition:
    button: MouseButton = MouseButton.NONE
    action: ButtonAction = ButtonAction.NONE


@dataclasses.dataclass(frozen=True, slots=True)
class ActuationCommand:
    sequence_id: int
    correlation_id: CorrelationId
    generated_at_ns: int
    desired_apply_time_ns: int
    delta_x_counts: int = 0
    delta_y_counts: int = 0
    button_transition: ButtonTransition = dataclasses.field(default_factory=ButtonTransition)


class SubmitResult(enum.IntEnum):
    SUBMITTED = 0
    REJECTED_LATCHED = 1
    REJECTED_STALE = 2
    REJECTED_UNINITIALIZED = 3


@dataclasses.dataclass(slots=True)
class ActuatorConfig:
    backend: str = "null"
    scheduler_hz: int = 1000
    relative_counts: bool = True
    cancel_superseded: bool = True
    require_emergency_stop: bool = True


@dataclasses.dataclass(slots=True)
class ActuatorHealth:
    is_active: bool = False
    is_latched: bool = False
    total_commands_submitted: int = 0
    total_commands_rejected: int = 0
    total_commands_cancelled: int = 0
    last_dispatch_ns: int = 0
    pressed_buttons_mask: int = 0


class NullActuator:
    """Simulated non-actuating command sink for testing, replays, and safety verification."""

    def __init__(self) -> None:
        self._config = ActuatorConfig()
        self._latch = EmergencyStopLatch()
        self._is_initialized = False
        self._is_active = False
        self._pending_command: Optional[ActuationCommand] = None
        self._recorded_commands: List[ActuationCommand] = []
        self._total_submitted = 0
        self._total_rejected = 0
        self._total_cancelled = 0
        self._last_dispatch_ns = 0
        self._pressed_buttons_mask = 0

    def initialize(self, config: ActuatorConfig) -> bool:
        self._config = config
        self._is_initialized = True
        return True

    def start(self) -> bool:
        if not self._is_initialized or self._latch.is_latched:
            return False
        self._is_active = True
        return True

    def submit_latest(self, command: ActuationCommand) -> SubmitResult:
        if not self._is_initialized or not self._is_active:
            self._total_rejected += 1
            return SubmitResult.REJECTED_UNINITIALIZED

        if self._latch.is_latched:
            self._total_rejected += 1
            return SubmitResult.REJECTED_LATCHED

        # Update simulated button state
        btn = command.button_transition
        if btn.button != MouseButton.NONE:
            bit = 1 << int(btn.button)
            if btn.action in (ButtonAction.PRESS, ButtonAction.CLICK):
                self._pressed_buttons_mask |= bit
            if btn.action in (ButtonAction.RELEASE, ButtonAction.CLICK):
                self._pressed_buttons_mask &= ~bit

        self._pending_command = command
        self._recorded_commands.append(command)
        self._last_dispatch_ns = (
            command.desired_apply_time_ns
            if command.desired_apply_time_ns > 0
            else command.generated_at_ns
        )
        self._total_submitted += 1
        return SubmitResult.SUBMITTED

    def cancel_pending(self) -> None:
        if self._pending_command is not None:
            self._pending_command = None
            self._total_cancelled += 1

    def emergency_stop(self, reason: SafetyReason = SafetyReason.EMERGENCY_STOP_TRIGGERED) -> None:
        self._latch.trigger(reason)
        if self._pending_command is not None:
            self._pending_command = None
            self._total_cancelled += 1
        # Fail-safe: release all button states immediately
        self._pressed_buttons_mask = 0

    def reset_emergency_stop(self, token: ResetToken) -> bool:
        return self._latch.try_reset(token)

    def shutdown(self) -> None:
        self._is_active = False
        self._pending_command = None
        # Fail-safe: release all button states on shutdown
        self._pressed_buttons_mask = 0

    @property
    def health(self) -> ActuatorHealth:
        return ActuatorHealth(
            is_active=self._is_active,
            is_latched=self._latch.is_latched,
            total_commands_submitted=self._total_submitted,
            total_commands_rejected=self._total_rejected,
            total_commands_cancelled=self._total_cancelled,
            last_dispatch_ns=self._last_dispatch_ns,
            pressed_buttons_mask=self._pressed_buttons_mask,
        )

    @property
    def recorded_commands(self) -> List[ActuationCommand]:
        return list(self._recorded_commands)

    def clear_recorded_commands(self) -> None:
        self._recorded_commands.clear()
