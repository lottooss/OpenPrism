"""Unit tests for Hybrid Trajectory Planner (Milestone M5-02)."""

import math
from typing import List, Tuple


def min_jerk_profile(tau: float) -> Tuple[float, float]:
    """Flash & Hogan minimum-jerk profile: s(tau) and s_dot(tau)."""
    tau = min(1.0, max(0.0, tau))
    t2 = tau * tau
    t3 = t2 * tau
    t4 = t3 * tau
    t5 = t4 * tau
    s = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    s_dot = 30.0 * t2 - 60.0 * t3 + 30.0 * t4
    return s, s_dot


def plan_trajectory_py(
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    max_vel: float = 5000.0,
    max_acc: float = 50000.0,
    max_jerk: float = 500000.0,
    dt: float = 0.001,
) -> List[Tuple[int, int]]:
    dx = target_x - start_x
    dy = target_y - start_y
    dist = math.hypot(dx, dy)

    if dist <= 6.0:
        return [(round(dx), round(dy))]

    t_vel = 1.875 * dist / max_vel
    t_acc = math.sqrt(5.774 * dist / max_acc)
    t_jerk = (60.0 * dist / max_jerk) ** (1.0 / 3.0)

    T = max(0.005, t_vel, t_acc, t_jerk)
    T = min(0.032, T)

    num_steps = max(1, math.ceil(T / dt))
    steps = []
    cum_x = 0
    cum_y = 0

    for i in range(1, num_steps + 1):
        tau = min(1.0, (i * dt) / T)
        s, _ = min_jerk_profile(tau)
        ideal_x = round(dx * s)
        ideal_y = round(dy * s)

        delta_x = ideal_x - cum_x
        delta_y = ideal_y - cum_y
        cum_x = ideal_x
        cum_y = ideal_y
        steps.append((delta_x, delta_y))

    return steps


def test_direct_feed_forward() -> None:
    steps = plan_trajectory_py(100.0, 100.0, 104.0, 102.0)
    assert len(steps) == 1
    assert steps[0] == (4, 2)


def test_minimum_jerk_count_conservation() -> None:
    steps = plan_trajectory_py(0.0, 0.0, 250.0, 100.0)
    assert len(steps) > 5

    total_x = sum(s[0] for s in steps)
    total_y = sum(s[1] for s in steps)

    assert total_x == 250
    assert total_y == 100
