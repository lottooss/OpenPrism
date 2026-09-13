"""Unit tests for counts-to-pixels calibration wizard (Milestone M6-03)."""

import pytest
import jsonschema

from tools.calibration.calibration_wizard import (
    CalibrationWizard,
    CalibrationResult,
    SimulatedCalibrationEnvironment,
)


def test_calibration_wizard_nominal_success() -> None:
    env = SimulatedCalibrationEnvironment(
        true_counts_per_pixel_x=1.25,
        true_counts_per_pixel_y=1.25,
        true_deadband=1.0,
        true_nonlinearity=0.01,
        true_cross_coupling=0.005,
    )
    wizard = CalibrationWizard(actuator=env, observer=env)
    result = wizard.run_calibration(profile_id="test_profile_01")

    assert result.is_valid
    assert abs(result.counts_per_pixel_x - 1.25) < 0.15
    assert abs(result.counts_per_pixel_y - 1.25) < 0.15
    assert result.rmse_pixels <= 2.0
    assert result.deadband_counts >= 0.0


def test_calibration_wizard_foreground_failure() -> None:
    env = SimulatedCalibrationEnvironment(is_foreground=False)
    wizard = CalibrationWizard(actuator=env, observer=env)

    with pytest.raises(PermissionError, match="authorized foreground"):
        wizard.run_calibration()


def test_calibration_wizard_emergency_stop_unarmed() -> None:
    env = SimulatedCalibrationEnvironment(is_armed=False)
    wizard = CalibrationWizard(actuator=env, observer=env)

    with pytest.raises(RuntimeError, match="emergency stop"):
        wizard.run_calibration()


class InstrumentedEnvironment(SimulatedCalibrationEnvironment):
    def __init__(self, fail_at: int = 0) -> None:
        super().__init__(true_deadband=0.0, true_nonlinearity=0.0, noise_std_px=0.0)
        self.fail_at = fail_at
        self.dispatches = 0
        self.observations = 0

    def dispatch_relative(self, dx_counts: int, dy_counts: int) -> bool:
        self.dispatches += 1
        if self.dispatches == self.fail_at:
            return False
        return super().dispatch_relative(dx_counts, dy_counts)

    def observe_pixel_displacement(self) -> tuple[float, float]:
        self.observations += 1
        return super().observe_pixel_displacement()


@pytest.mark.parametrize("fail_at", [1, 2, 3, 22])
def test_failed_dispatch_cannot_be_used_as_fit_or_held_out_evidence(fail_at: int) -> None:
    # One deadband pulse, twenty signed axis pulses, then held-out motions.
    env = InstrumentedEnvironment(fail_at=fail_at)
    with pytest.raises(RuntimeError, match="dispatch failed"):
        CalibrationWizard(env, env).run_calibration()
    assert env.dispatches == fail_at
    assert env.observations == fail_at - 1


@pytest.mark.parametrize("gate", ["is_foreground", "is_armed"])
@pytest.mark.parametrize("lose_at", [1, 26])
def test_authorization_is_rechecked_for_every_pulse(gate: str, lose_at: int) -> None:
    class LosingAuthorization(InstrumentedEnvironment):
        def observe_pixel_displacement(self) -> tuple[float, float]:
            observation = super().observe_pixel_displacement()
            if self.observations == lose_at:
                setattr(self, gate, False)
            return observation

    env = LosingAuthorization()
    with pytest.raises((PermissionError, RuntimeError)):
        CalibrationWizard(env, env).run_calibration()
    assert env.dispatches == lose_at
    assert env.observations == lose_at


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("axis", [0, 1])
def test_nonfinite_observation_rejects_profile(invalid: float, axis: int) -> None:
    class InvalidObserver(InstrumentedEnvironment):
        def observe_pixel_displacement(self) -> tuple[float, float]:
            return (invalid, 0.0) if axis == 0 else (0.0, invalid)

    env = InvalidObserver()
    with pytest.raises(ValueError, match="nonfinite"):
        CalibrationWizard(env, env).run_calibration()
    assert env.dispatches == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -1.0, 151.0])
def test_invalid_context_rejects_before_dispatch(invalid: float) -> None:
    env = InstrumentedEnvironment()
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        CalibrationWizard(env, env, fov_deg=invalid).run_calibration()
    assert env.dispatches == 0


@pytest.mark.parametrize("field", [
    "fov_horizontal_deg", "in_game_sensitivity", "counts_per_pixel_x", "counts_per_pixel_y",
    "deadband_counts", "nonlinearity_alpha", "cross_coupling_xy", "rmse_pixels",
])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_result_rejects_every_nonfinite_numeric_field(field: str, invalid: float) -> None:
    result = CalibrationResult(
        schema_version=1, profile_id="test_profile", calibrated_at="",
        resolution_width=1920, resolution_height=1080, fov_horizontal_deg=103.0,
        in_game_sensitivity=1.0, counts_per_pixel_x=1.0, counts_per_pixel_y=1.0,
        deadband_counts=0.0, nonlinearity_alpha=0.0, cross_coupling_xy=0.0, rmse_pixels=0.0,
    )
    setattr(result, field, invalid)
    with pytest.raises(ValueError, match="nonfinite"):
        result.validate_schema()
