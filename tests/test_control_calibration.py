"""Unit tests for bounded online visual response calibration (Milestone M6-04)."""

import math
from dataclasses import dataclass


@dataclass
class OnlineCalibrationConfig:
    initial_cpi: float = 800.0
    initial_counts_per_pixel: float = 1.0
    max_drift_fraction: float = 0.25
    max_step_delta: float = 0.02
    min_magnitude_px: float = 3.0
    outlier_ratio_threshold: float = 0.35
    learning_rate: float = 0.1


class ControlCalibration:

    def __init__(self, config: OnlineCalibrationConfig | None = None) -> None:
        self.config = config or OnlineCalibrationConfig()
        self.baseline_cpp = self.config.initial_counts_per_pixel
        self.current_cpp = self.baseline_cpp
        self.successful_updates = 0
        self.rejected_outliers = 0

    def update_feedback(self, exp_x: float, exp_y: float, obs_x: float, obs_y: float) -> bool:
        exp_mag = math.sqrt(exp_x * exp_x + exp_y * exp_y)
        obs_mag = math.sqrt(obs_x * obs_x + obs_y * obs_y)

        if exp_mag < self.config.min_magnitude_px or obs_mag < 0.5:
            self.rejected_outliers += 1
            return False

        dot = exp_x * obs_x + exp_y * obs_y
        if dot <= 0.0:
            self.rejected_outliers += 1
            return False

        rel_diff = abs(obs_mag - exp_mag) / exp_mag
        if rel_diff > self.config.outlier_ratio_threshold:
            self.rejected_outliers += 1
            return False

        ratio = exp_mag / obs_mag
        target_cpp = self.baseline_cpp * ratio
        delta = (target_cpp - self.current_cpp) * self.config.learning_rate
        clamped_delta = max(-self.config.max_step_delta, min(self.config.max_step_delta, delta))

        new_cpp = self.current_cpp + clamped_delta
        min_allowed = self.baseline_cpp * (1.0 - self.config.max_drift_fraction)
        max_allowed = self.baseline_cpp * (1.0 + self.config.max_drift_fraction)
        self.current_cpp = max(min_allowed, min(max_allowed, new_cpp))
        self.successful_updates += 1
        return True


def test_online_calibration_convergence() -> None:
    calib = ControlCalibration(OnlineCalibrationConfig(initial_counts_per_pixel=1.0, learning_rate=0.1))

    for _ in range(25):
        assert calib.update_feedback(100.0, 0.0, 91.0, 0.0)

    assert abs(calib.current_cpp - 1.10) < 0.05
    assert calib.successful_updates == 25


def test_online_calibration_outlier_rejection() -> None:
    calib = ControlCalibration(OnlineCalibrationConfig(initial_counts_per_pixel=1.0))

    # 50% outlier error
    assert not calib.update_feedback(100.0, 0.0, 50.0, 0.0)
    assert calib.rejected_outliers == 1
    assert calib.current_cpp == 1.0
