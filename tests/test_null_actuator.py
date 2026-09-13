"""Unit tests for NullActuator, latched emergency stop, and token-based reset."""

import unittest

from tools.actuation.null_actuator import (
    ActuationCommand,
    ActuatorConfig,
    ButtonAction,
    ButtonTransition,
    MouseButton,
    NullActuator,
    SubmitResult,
)
from tools.actuation.safety import ResetToken, SafetyReason
from tools.timing.stage_timer import CorrelationFlags, CorrelationId


class TestNullActuator(unittest.TestCase):
    """Test suite for NullActuator command sink and emergency stop safety latch."""

    def setUp(self) -> None:
        self.actuator = NullActuator()
        self.config = ActuatorConfig(
            backend="null",
            scheduler_hz=1000,
            relative_counts=True,
            cancel_superseded=True,
            require_emergency_stop=True,
        )
        self.cid = CorrelationId(
            sequence_id=1,
            source_timestamp_ns=1_000_000_000,
            pipeline_run_id=10,
            flags=CorrelationFlags.SYNTHETIC,
        )

    def test_uninitialized_and_inactive_rejection(self) -> None:
        """Verify submitting commands before initialize/start is cleanly rejected."""
        cmd = ActuationCommand(
            sequence_id=1,
            correlation_id=self.cid,
            generated_at_ns=1_000_000_000,
            desired_apply_time_ns=1_000_500_000,
            delta_x_counts=10,
            delta_y_counts=-5,
        )

        res = self.actuator.submit_latest(cmd)
        self.assertEqual(res, SubmitResult.REJECTED_UNINITIALIZED)

        self.actuator.initialize(self.config)
        # Initialized but not started
        res2 = self.actuator.submit_latest(cmd)
        self.assertEqual(res2, SubmitResult.REJECTED_UNINITIALIZED)

    def test_command_recording_without_os_input(self) -> None:
        """Verify NullActuator records timestamped commands without OS input."""
        self.actuator.initialize(self.config)
        self.assertTrue(self.actuator.start())

        cmd1 = ActuationCommand(
            sequence_id=1,
            correlation_id=self.cid,
            generated_at_ns=1_000_000_000,
            desired_apply_time_ns=1_000_500_000,
            delta_x_counts=12,
            delta_y_counts=-8,
            button_transition=ButtonTransition(button=MouseButton.LEFT, action=ButtonAction.PRESS),
        )

        cmd2 = ActuationCommand(
            sequence_id=2,
            correlation_id=self.cid,
            generated_at_ns=1_001_000_000,
            desired_apply_time_ns=1_001_500_000,
            delta_x_counts=5,
            delta_y_counts=2,
            button_transition=ButtonTransition(button=MouseButton.LEFT, action=ButtonAction.RELEASE),
        )

        self.assertEqual(self.actuator.submit_latest(cmd1), SubmitResult.SUBMITTED)
        self.assertEqual(self.actuator.submit_latest(cmd2), SubmitResult.SUBMITTED)

        records = self.actuator.recorded_commands
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].delta_x_counts, 12)
        self.assertEqual(records[0].delta_y_counts, -8)
        self.assertEqual(records[1].delta_x_counts, 5)

        health = self.actuator.health
        self.assertTrue(health.is_active)
        self.assertFalse(health.is_latched)
        self.assertEqual(health.total_commands_submitted, 2)
        self.assertEqual(health.last_dispatch_ns, 1_001_500_000)

    def test_cancel_pending_work(self) -> None:
        """Verify canceling pending work decrements pending state and increments cancelled count."""
        self.actuator.initialize(self.config)
        self.actuator.start()

        cmd = ActuationCommand(
            sequence_id=1,
            correlation_id=self.cid,
            generated_at_ns=1_000_000_000,
            desired_apply_time_ns=1_001_000_000,
            delta_x_counts=15,
            delta_y_counts=0,
        )

        self.actuator.submit_latest(cmd)
        self.actuator.cancel_pending()

        health = self.actuator.health
        self.assertEqual(health.total_commands_submitted, 1)
        self.assertEqual(health.total_commands_cancelled, 1)

    def test_emergency_stop_latches_and_rejects(self) -> None:
        """Verify emergency stop cancels work, latches fail-closed, and rejects future submissions."""
        self.actuator.initialize(self.config)
        self.actuator.start()

        cmd = ActuationCommand(
            sequence_id=1,
            correlation_id=self.cid,
            generated_at_ns=1_000_000_000,
            desired_apply_time_ns=1_000_500_000,
            delta_x_counts=5,
            button_transition=ButtonTransition(button=MouseButton.LEFT, action=ButtonAction.PRESS),
        )
        self.actuator.submit_latest(cmd)
        self.assertEqual(self.actuator.health.pressed_buttons_mask, (1 << MouseButton.LEFT))

        # Trigger Emergency Stop
        self.actuator.emergency_stop(SafetyReason.EMERGENCY_STOP_TRIGGERED)

        # Invariant: latch is engaged, button mask is cleared immediately
        health = self.actuator.health
        self.assertTrue(health.is_latched)
        self.assertEqual(health.pressed_buttons_mask, 0)
        self.assertEqual(health.total_commands_cancelled, 1)

        # Subsequent submissions must be rejected
        cmd2 = ActuationCommand(
            sequence_id=2,
            correlation_id=self.cid,
            generated_at_ns=1_001_000_000,
            desired_apply_time_ns=1_001_500_000,
            delta_x_counts=10,
        )
        res = self.actuator.submit_latest(cmd2)
        self.assertEqual(res, SubmitResult.REJECTED_LATCHED)
        self.assertEqual(self.actuator.health.total_commands_rejected, 1)

    def test_reset_emergency_stop_requires_valid_token(self) -> None:
        """Verify emergency stop latch can only be reset with an explicit valid token."""
        self.actuator.initialize(self.config)
        self.actuator.start()
        self.actuator.emergency_stop()

        self.assertTrue(self.actuator.health.is_latched)
        self.assertFalse(self.actuator.start(), "start() must return False while emergency stop is latched")

        # Invalid token (empty string)
        invalid_tok1 = ResetToken(token_id="", is_valid=True)
        self.assertFalse(self.actuator.reset_emergency_stop(invalid_tok1))
        self.assertTrue(self.actuator.health.is_latched)

        # Invalid token flag
        invalid_tok2 = ResetToken(token_id="RESET_SEC_123", is_valid=False)
        self.assertFalse(self.actuator.reset_emergency_stop(invalid_tok2))
        self.assertTrue(self.actuator.health.is_latched)

        # Valid token
        valid_tok = ResetToken(token_id="AUTH_OP_RESET_99", is_valid=True)
        self.assertTrue(self.actuator.reset_emergency_stop(valid_tok))
        self.assertFalse(self.actuator.health.is_latched)

        # Submissions now accepted again
        cmd = ActuationCommand(
            sequence_id=3,
            correlation_id=self.cid,
            generated_at_ns=1_002_000_000,
            desired_apply_time_ns=1_002_500_000,
            delta_x_counts=2,
        )
        self.assertEqual(self.actuator.submit_latest(cmd), SubmitResult.SUBMITTED)

    def test_shutdown_releases_button_state(self) -> None:
        """Verify shutdown releases all button states and deactivates actuator."""
        self.actuator.initialize(self.config)
        self.actuator.start()

        cmd = ActuationCommand(
            sequence_id=1,
            correlation_id=self.cid,
            generated_at_ns=1_000_000_000,
            desired_apply_time_ns=1_000_500_000,
            button_transition=ButtonTransition(button=MouseButton.RIGHT, action=ButtonAction.PRESS),
        )
        self.actuator.submit_latest(cmd)
        self.assertEqual(self.actuator.health.pressed_buttons_mask, (1 << MouseButton.RIGHT))

        self.actuator.shutdown()
        health = self.actuator.health
        self.assertFalse(health.is_active)
        self.assertEqual(health.pressed_buttons_mask, 0)


if __name__ == "__main__":
    unittest.main()
