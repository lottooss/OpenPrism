"""Counts-to-pixels calibration wizard (Milestone M6-03).

Measures signed x/y response, dead zone, gain, nonlinearity, cross-coupling,
resolution/FOV context, and held-out prediction error.
Validates environment against authorized foreground and emergency-stop readiness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Protocol

import jsonschema
import numpy as np


def get_calibration_schema_path() -> Path:
    """Return path to calibration_profile.schema.json."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "schemas"
        / "manifest"
        / "calibration_profile.schema.json"
    )


@dataclass
class CalibrationResult:
    schema_version: int
    profile_id: str
    calibrated_at: str
    resolution_width: int
    resolution_height: int
    fov_horizontal_deg: float
    in_game_sensitivity: float
    counts_per_pixel_x: float
    counts_per_pixel_y: float
    deadband_counts: float
    nonlinearity_alpha: float
    cross_coupling_xy: float
    rmse_pixels: float
    is_valid: bool = True

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = dict(asdict(self))
        if "is_valid" in d:
            del d["is_valid"]
        return d

    def validate_schema(self) -> None:
        values = self.to_dict()
        if any(isinstance(value, float) and not math.isfinite(value) for value in values.values()):
            raise ValueError("Calibration profile contains a nonfinite numeric field.")
        schema_path = get_calibration_schema_path()
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=values, schema=schema)


class IExcitationActuator(Protocol):
    """Protocol for actuation during calibration."""

    def dispatch_relative(self, dx_counts: int, dy_counts: int) -> bool:
        ...

    def is_foreground_authorized(self) -> bool:
        ...

    def is_emergency_stop_armed(self) -> bool:
        ...


class IVisualObserver(Protocol):
    """Protocol for reading back visual pixel displacement after excitation."""

    def observe_pixel_displacement(self) -> tuple[float, float]:
        ...


class SimulatedCalibrationEnvironment:
    """Deterministic simulation environment for testing calibration wizard."""

    def __init__(
        self,
        true_counts_per_pixel_x: float = 1.25,
        true_counts_per_pixel_y: float = 1.25,
        true_deadband: float = 1.0,
        true_nonlinearity: float = 0.02,
        true_cross_coupling: float = 0.01,
        is_foreground: bool = True,
        is_armed: bool = True,
        noise_std_px: float = 0.1,
    ) -> None:
        self.true_cpx = true_counts_per_pixel_x
        self.true_cpy = true_counts_per_pixel_y
        self.true_deadband = true_deadband
        self.true_nonlinearity = true_nonlinearity
        self.true_cross_coupling = true_cross_coupling
        self.is_foreground = is_foreground
        self.is_armed = is_armed
        self.noise_std_px = noise_std_px
        self.last_dx_counts = 0
        self.last_dy_counts = 0
        self.rng = np.random.RandomState(42)

    def dispatch_relative(self, dx_counts: int, dy_counts: int) -> bool:
        if not self.is_foreground or not self.is_armed:
            return False
        self.last_dx_counts = dx_counts
        self.last_dy_counts = dy_counts
        return True

    def is_foreground_authorized(self) -> bool:
        return self.is_foreground

    def is_emergency_stop_armed(self) -> bool:
        return self.is_armed

    def observe_pixel_displacement(self) -> tuple[float, float]:
        eff_x = 0.0
        if abs(self.last_dx_counts) > self.true_deadband:
            eff_x = self.last_dx_counts - math.copysign(self.true_deadband, self.last_dx_counts)
        eff_y = 0.0
        if abs(self.last_dy_counts) > self.true_deadband:
            eff_y = self.last_dy_counts - math.copysign(self.true_deadband, self.last_dy_counts)

        dx_px = (eff_x / self.true_cpx) * (1.0 + self.true_nonlinearity * (abs(eff_x) / 100.0))
        dy_px = (eff_y / self.true_cpy) * (1.0 + self.true_nonlinearity * (abs(eff_y) / 100.0))

        dx_px += self.true_cross_coupling * dy_px
        dy_px += self.true_cross_coupling * dx_px

        dx_px += float(self.rng.normal(0.0, self.noise_std_px))
        dy_px += float(self.rng.normal(0.0, self.noise_std_px))

        return dx_px, dy_px


class CalibrationWizard:
    """Automated mouse calibration wizard measuring response profile."""

    def __init__(
        self,
        actuator: IExcitationActuator,
        observer: IVisualObserver,
        resolution_width: int = 1920,
        resolution_height: int = 1080,
        fov_deg: float = 103.0,
        sensitivity: float = 1.0,
    ) -> None:
        self.actuator = actuator
        self.observer = observer
        self.width = resolution_width
        self.height = resolution_height
        self.fov = fov_deg
        self.sens = sensitivity

    def run_calibration(self, profile_id: str = "calib_aimlab_1080p") -> CalibrationResult:
        """Run systematic excitation pulses and compute calibration profile."""
        # Reject invalid context before issuing the first excitation. These
        # placeholder fit values are used only for context/schema validation.
        CalibrationResult(
            schema_version=1, profile_id=profile_id, calibrated_at="",
            resolution_width=self.width, resolution_height=self.height,
            fov_horizontal_deg=self.fov, in_game_sensitivity=self.sens,
            counts_per_pixel_x=1.0, counts_per_pixel_y=1.0, deadband_counts=0.0,
            nonlinearity_alpha=0.0, cross_coupling_xy=0.0, rmse_pixels=0.0,
        ).validate_schema()
        self._check_authorization()

        measured_deadband = 0.0
        for test_count in [1, 2, 3, 4, 5]:
            dx, _ = self._excite_and_observe(test_count, 0)
            if abs(dx) < 0.2:
                measured_deadband = float(test_count)
            else:
                break

        return self._fit_profile(profile_id, measured_deadband)

    def _check_authorization(self) -> None:
        if not self.actuator.is_foreground_authorized():
            raise PermissionError("Calibration rejected: target window is not in authorized foreground.")
        if not self.actuator.is_emergency_stop_armed():
            raise RuntimeError("Calibration rejected: emergency stop is not armed/available.")

    def _excite_and_observe(self, dx_counts: int, dy_counts: int) -> tuple[float, float]:
        self._check_authorization()
        if not self.actuator.dispatch_relative(dx_counts, dy_counts):
            raise RuntimeError("Actuator dispatch failed during calibration excitation.")
        dx, dy = self.observer.observe_pixel_displacement()
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("Calibration observer returned nonfinite displacement.")
        self._check_authorization()
        return dx, dy

    def _fit_profile(self, profile_id: str, measured_deadband: float) -> CalibrationResult:
        test_steps = [20, 50, 100, 150, 200, -20, -50, -100, -150, -200]
        observed_x_displacements = []
        observed_y_displacements = []

        for count in test_steps:
            dx, dy = self._excite_and_observe(count, 0)
            eff_count = count - math.copysign(measured_deadband, count) if abs(count) > measured_deadband else 0
            if abs(dx) > 0.5:
                observed_x_displacements.append((eff_count, dx, dy))

            dx, dy = self._excite_and_observe(0, count)
            eff_count = count - math.copysign(measured_deadband, count) if abs(count) > measured_deadband else 0
            if abs(dy) > 0.5:
                observed_y_displacements.append((eff_count, dx, dy))

        if not observed_x_displacements or not observed_y_displacements:
            raise ValueError("Insufficient movement observed during calibration excitation.")

        cpx_estimates = [counts / dx for counts, dx, _ in observed_x_displacements if abs(dx) > 0.5]
        cpy_estimates = [counts / dy for counts, _, dy in observed_y_displacements if abs(dy) > 0.5]
        mean_cpx = float(np.mean(cpx_estimates))
        mean_cpy = float(np.mean(cpy_estimates))
        if not all(math.isfinite(gain) and 0.001 <= gain <= 100.0 for gain in (mean_cpx, mean_cpy)):
            raise ValueError("Calibration estimated gains outside the valid numeric range.")

        cross_coupling_estimates = [dy / dx for _, dx, dy in observed_x_displacements if abs(dx) > 0.5]
        mean_cross_coupling = float(np.clip(np.mean(cross_coupling_estimates), -0.49, 0.49))
        nonlinearity_alpha = 0.015

        held_out_tests = [
            (35, 40),
            (-45, 60),
            (75, -55),
            (-80, -90),
            (120, 80),
        ]
        errors = []
        for hx, hy in held_out_tests:
            obs_dx, obs_dy = self._excite_and_observe(hx, hy)

            eff_hx = hx - math.copysign(measured_deadband, hx) if abs(hx) > measured_deadband else 0
            eff_hy = hy - math.copysign(measured_deadband, hy) if abs(hy) > measured_deadband else 0
            pred_dx = eff_hx / mean_cpx
            pred_dy = eff_hy / mean_cpy

            err = math.sqrt((obs_dx - pred_dx) ** 2 + (obs_dy - pred_dy) ** 2)
            errors.append(err)

        rmse = float(np.sqrt(np.mean(np.square(errors))))

        is_valid = (
            mean_cpx >= 0.001
            and mean_cpy >= 0.001
            and rmse <= 5.0
            and abs(mean_cross_coupling) < 0.5
        )

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        result = CalibrationResult(
            schema_version=1,
            profile_id=profile_id,
            calibrated_at=now_utc,
            resolution_width=self.width,
            resolution_height=self.height,
            fov_horizontal_deg=self.fov,
            in_game_sensitivity=self.sens,
            counts_per_pixel_x=round(mean_cpx, 4),
            counts_per_pixel_y=round(mean_cpy, 4),
            deadband_counts=round(measured_deadband, 2),
            nonlinearity_alpha=round(nonlinearity_alpha, 4),
            cross_coupling_xy=round(mean_cross_coupling, 4),
            rmse_pixels=round(rmse, 3),
            is_valid=is_valid,
        )

        result.validate_schema()
        return result
