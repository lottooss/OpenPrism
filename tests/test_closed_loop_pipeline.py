"""Unit tests for closed-loop aiming pipeline and uncertainty-aware shot gating (Milestone M7-02)."""

import math
from dataclasses import dataclass
from enum import IntEnum


class ShotGateDecision(IntEnum):
    AUTHORIZED = 0
    REJECTED_UNCERTAINTY_EXCEEDED = 1
    REJECTED_LOW_CONFIDENCE = 2
    REJECTED_ALIGNMENT_MISS = 3
    REJECTED_DEADLINE_EXPIRED = 4
    REJECTED_SAFETY_LATCHED = 5
    REJECTED_NO_TARGET = 6


@dataclass
class Covariance2D:
    xx: float
    xy: float
    yy: float


@dataclass
class TrackedTarget:
    track_id: int
    center_x: float
    center_y: float
    effective_radius_px: float
    confidence: float
    covariance: Covariance2D
    prediction_time_ns: int


class ShotGate:

    def __init__(
        self,
        min_confidence: float = 0.80,
        max_uncertainty_radius_ratio: float = 1.0,
        max_alignment_radius_ratio: float = 1.10,
        max_observation_age_ns: int = 10_000_000,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_uncertainty_ratio = max_uncertainty_radius_ratio
        self.max_alignment_ratio = max_alignment_radius_ratio
        self.max_age_ns = max_observation_age_ns

    @staticmethod
    def compute_uncertainty_sigma(cov: Covariance2D) -> float:
        tr = cov.xx + cov.yy
        diff = (cov.xx - cov.yy) * 0.5
        disc = math.sqrt(max(0.0, diff * diff + cov.xy * cov.xy))
        lambda_max = tr * 0.5 + disc
        return math.sqrt(max(0.0, lambda_max))

    def evaluate(
        self,
        target: TrackedTarget | None,
        crosshair_x: float,
        crosshair_y: float,
        now_ns: int,
        is_safety_latched: bool = False,
    ) -> tuple[ShotGateDecision, bool]:
        if target is None:
            return ShotGateDecision.REJECTED_NO_TARGET, False

        if is_safety_latched:
            return ShotGateDecision.REJECTED_SAFETY_LATCHED, False

        if target.prediction_time_ns > 0 and now_ns > 0:
            age = now_ns - target.prediction_time_ns
            if age > self.max_age_ns or age < -50_000_000:
                return ShotGateDecision.REJECTED_DEADLINE_EXPIRED, False

        if target.confidence < self.min_confidence:
            return ShotGateDecision.REJECTED_LOW_CONFIDENCE, False

        sigma = self.compute_uncertainty_sigma(target.covariance)
        if sigma > target.effective_radius_px * self.max_uncertainty_ratio:
            return ShotGateDecision.REJECTED_UNCERTAINTY_EXCEEDED, False

        dist = math.sqrt(
            (target.center_x - crosshair_x) ** 2 + (target.center_y - crosshair_y) ** 2
        )
        if dist > target.effective_radius_px * self.max_alignment_ratio:
            return ShotGateDecision.REJECTED_ALIGNMENT_MISS, False

        return ShotGateDecision.AUTHORIZED, True


def test_shot_gate_nominal_authorization() -> None:
    gate = ShotGate()
    target = TrackedTarget(
        track_id=1,
        center_x=962.0,
        center_y=541.0,
        effective_radius_px=15.0,
        confidence=0.95,
        covariance=Covariance2D(xx=4.0, xy=0.0, yy=4.0), # sigma = 2.0 px
        prediction_time_ns=1_000_000_000,
    )

    decision, authorized = gate.evaluate(
        target, crosshair_x=960.0, crosshair_y=540.0, now_ns=1_002_000_000
    )
    assert decision == ShotGateDecision.AUTHORIZED
    assert authorized


def test_shot_gate_uncertainty_rejection() -> None:
    gate = ShotGate()
    target = TrackedTarget(
        track_id=2,
        center_x=960.0,
        center_y=540.0,
        effective_radius_px=10.0,
        confidence=0.95,
        covariance=Covariance2D(xx=225.0, xy=0.0, yy=225.0), # sigma = 15.0 px > 10.0 px!
        prediction_time_ns=1_000_000_000,
    )

    decision, authorized = gate.evaluate(
        target, crosshair_x=960.0, crosshair_y=540.0, now_ns=1_002_000_000
    )
    assert decision == ShotGateDecision.REJECTED_UNCERTAINTY_EXCEEDED
    assert not authorized


def test_shot_gate_alignment_miss_rejection() -> None:
    gate = ShotGate()
    target = TrackedTarget(
        track_id=3,
        center_x=1100.0,
        center_y=540.0, # 140 px away
        effective_radius_px=15.0,
        confidence=0.95,
        covariance=Covariance2D(xx=1.0, xy=0.0, yy=1.0),
        prediction_time_ns=1_000_000_000,
    )

    decision, authorized = gate.evaluate(
        target, crosshair_x=960.0, crosshair_y=540.0, now_ns=1_002_000_000
    )
    assert decision == ShotGateDecision.REJECTED_ALIGNMENT_MISS
    assert not authorized
