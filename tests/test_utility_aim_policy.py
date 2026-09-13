"""Unit tests for Domain-Neutral Utility Aim Policy (Milestone M5-01)."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrackedTarget:
    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.95
    target_value: float = 1.0
    is_confirmed: bool = True


class UtilityAimPolicyPy:
    def __init__(
        self,
        weight_value: float = 1.0,
        weight_confidence: float = 1.5,
        weight_distance: float = 0.005,
        switch_hysteresis: float = 1.5,
        fire_threshold_px: float = 8.0,
    ):
        self.w_val = weight_value
        self.w_conf = weight_confidence
        self.w_dist = weight_distance
        self.hysteresis = switch_hysteresis
        self.fire_thresh = fire_threshold_px
        self.current_locked_id: Optional[int] = None

    def choose(
        self, crosshair_x: float, crosshair_y: float, tracks: List[TrackedTarget]
    ) -> Optional[Dict[str, Any]]:
        if not tracks:
            self.current_locked_id = None
            return None

        best_utility = -float("inf")
        best_target = None

        for t in tracks:
            dist = math.hypot(t.x - crosshair_x, t.y - crosshair_y)
            conf_mult = 1.2 if t.is_confirmed else 0.8
            utility = (
                self.w_val * t.target_value * conf_mult
                + self.w_conf * t.confidence
                - self.w_dist * dist
            )

            if (
                self.current_locked_id is not None
                and t.track_id == self.current_locked_id
            ):
                utility += self.hysteresis

            if utility > best_utility:
                best_utility = utility
                best_target = t

        if best_target is None:
            self.current_locked_id = None
            return None

        self.current_locked_id = best_target.track_id
        dist = math.hypot(best_target.x - crosshair_x, best_target.y - crosshair_y)
        authorize_fire = dist <= self.fire_thresh and best_target.is_confirmed

        return {
            "target_track_id": best_target.track_id,
            "aim_x": best_target.x,
            "aim_y": best_target.y,
            "error_distance_px": dist,
            "authorize_fire": authorize_fire,
            "utility_score": best_utility,
        }


def test_utility_policy_target_ranking() -> None:
    policy = UtilityAimPolicyPy()
    t1 = TrackedTarget(track_id=1, x=980.0, y=540.0)  # 20px
    t2 = TrackedTarget(track_id=2, x=1200.0, y=540.0)  # 240px

    intent = policy.choose(960.0, 540.0, [t1, t2])
    assert intent is not None
    assert intent["target_track_id"] == 1
    assert intent["error_distance_px"] == 20.0
    assert not intent["authorize_fire"]


def test_utility_policy_hysteresis_anti_thrashing() -> None:
    policy = UtilityAimPolicyPy()
    t1 = TrackedTarget(track_id=1, x=930.0, y=540.0)  # 30px
    t2 = TrackedTarget(track_id=2, x=995.0, y=540.0)  # 35px

    # Frame 1: Lock onto Target 1
    intent1 = policy.choose(960.0, 540.0, [t1, t2])
    assert intent1 is not None
    assert intent1["target_track_id"] == 1

    # Frame 2: Crosshair shifts closer to Target 2 (dist to T1=35, dist to T2=30)
    # Hysteresis should keep lock on T1
    intent2 = policy.choose(965.0, 540.0, [t1, t2])
    assert intent2 is not None
    assert intent2["target_track_id"] == 1


def test_utility_policy_fire_authorization() -> None:
    policy = UtilityAimPolicyPy()
    t1 = TrackedTarget(
        track_id=1, x=964.0, y=540.0, is_confirmed=True
    )  # 4px error <= 8px
    intent = policy.choose(960.0, 540.0, [t1])
    assert intent is not None
    assert intent["authorize_fire"] is True

    t_tentative = TrackedTarget(
        track_id=2, x=964.0, y=540.0, is_confirmed=False
    )
    intent2 = policy.choose(960.0, 540.0, [t_tentative])
    assert intent2 is not None
    assert intent2["authorize_fire"] is False
