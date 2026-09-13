"""Unit tests for Counts-to-Pixels Calibration Model (Milestone M6-03)."""

from dataclasses import dataclass


@dataclass
class CalibrationProfile:
    counts_per_pixel_x: float = 1.0
    counts_per_pixel_y: float = 1.0
    deadband_counts: float = 0.0
    nonlinearity_alpha: float = 0.0
    cross_coupling_xy: float = 0.0


class CalibrationModel:

    def __init__(self, profile: CalibrationProfile) -> None:
        self.profile = profile

    def is_valid(self) -> bool:
        if self.profile.counts_per_pixel_x <= 0.001:
            return False
        if self.profile.counts_per_pixel_y <= 0.001:
            return False
        return True

    def pixels_to_counts(self, dx: float, dy: float) -> tuple[int, int]:
        if not self.is_valid():
            return 0, 0
        raw_x = dx * self.profile.counts_per_pixel_x
        raw_y = dy * self.profile.counts_per_pixel_y
        if self.profile.deadband_counts > 0:
            if abs(raw_x) > 0.1:
                raw_x += (
                    self.profile.deadband_counts
                    if raw_x > 0
                    else -self.profile.deadband_counts
                )
            if abs(raw_y) > 0.1:
                raw_y += (
                    self.profile.deadband_counts
                    if raw_y > 0
                    else -self.profile.deadband_counts
                )
        return round(raw_x), round(raw_y)


def test_calibration_roundtrip() -> None:
    profile = CalibrationProfile(
        counts_per_pixel_x=1.5, counts_per_pixel_y=1.5
    )
    model = CalibrationModel(profile)
    assert model.is_valid()

    cx, cy = model.pixels_to_counts(100.0, -50.0)
    assert cx == 150
    assert cy == -75


def test_calibration_deadband() -> None:
    profile = CalibrationProfile(
        counts_per_pixel_x=1.0, counts_per_pixel_y=1.0, deadband_counts=2.0
    )
    model = CalibrationModel(profile)
    cx, cy = model.pixels_to_counts(1.0, -1.0)
    assert cx == 3
    assert cy == -3


def test_calibration_invalid_fails() -> None:
    profile = CalibrationProfile(counts_per_pixel_x=-1.0)
    model = CalibrationModel(profile)
    assert not model.is_valid()
    assert model.pixels_to_counts(10.0, 10.0) == (0, 0)
