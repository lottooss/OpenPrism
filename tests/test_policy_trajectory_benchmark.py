"""Replay benchmark verification for Milestone M5 (M5-04)."""

from typing import Optional


class Target:

    def __init__(self, tid: int, x: float, y: float, vx: float):
        self.id = tid
        self.x = x
        self.y = y
        self.vx = vx


def test_policy_hysteresis_beats_nearest_switches() -> None:
    dt = 1.0 / 144.0
    t1 = Target(1, 800.0, 540.0, 45.0)
    t2 = Target(2, 1120.0, 545.0, -45.0)

    crosshair_util_x = 960.0
    crosshair_near_x = 960.0

    utility_locked: Optional[int] = None
    nearest_locked: Optional[int] = None

    utility_switches = 0
    nearest_switches = 0

    for _ in range(1000):
        t1.x += t1.vx * dt
        t2.x += t2.vx * dt

        # Nearest target with moving crosshair
        d1_near = abs(t1.x - crosshair_near_x)
        d2_near = abs(t2.x - crosshair_near_x)
        nearest_id = 1 if d1_near < d2_near else 2
        if nearest_locked is not None and nearest_id != nearest_locked:
            nearest_switches += 1
        nearest_locked = nearest_id
        # Move nearest crosshair towards nearest target
        crosshair_near_x += 0.05 * (
            (t1.x if nearest_id == 1 else t2.x) - crosshair_near_x
        )

        # Utility target with hysteresis and moving crosshair
        d1_util = abs(t1.x - crosshair_util_x)
        d2_util = abs(t2.x - crosshair_util_x)
        u1 = 1.0 - 0.005 * d1_util
        u2 = 1.0 - 0.005 * d2_util
        if utility_locked == 1:
            u1 += 1.5
        elif utility_locked == 2:
            u2 += 1.5

        util_id = 1 if u1 >= u2 else 2
        if utility_locked is not None and util_id != utility_locked:
            utility_switches += 1
        utility_locked = util_id
        # Move utility crosshair towards utility target
        crosshair_util_x += 0.05 * (
            (t1.x if util_id == 1 else t2.x) - crosshair_util_x
        )

    # Hysteresis prevents thrashing at crossing point
    assert utility_switches <= nearest_switches
    assert utility_switches == 0
    assert nearest_switches >= 1
