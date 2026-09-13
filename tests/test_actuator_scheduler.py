"""Unit tests for Actuator Scheduler and Plan Superseding (Milestone M6-02)."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrajectoryPoint:
    target_time_ns: int = 0
    step_delta_x_counts: int = 0
    step_delta_y_counts: int = 0


@dataclass
class TrajectoryPlan:
    plan_id: int = 0
    point_count: int = 0
    points: List[TrajectoryPoint] = field(default_factory=list)


class MockScheduler:

    def __init__(self) -> None:
        self.active_plan: Optional[TrajectoryPlan] = None
        self.current_step_idx = 0
        self.dispatched_counts_x: List[int] = []
        self.plans_superseded = 0

    def submit_plan(self, plan: TrajectoryPlan) -> None:
        if (
            self.active_plan is not None
            and self.current_step_idx < self.active_plan.point_count
        ):
            self.plans_superseded += 1
        self.active_plan = plan
        self.current_step_idx = 0

    def tick(self) -> bool:
        if (
            self.active_plan is None
            or self.current_step_idx >= self.active_plan.point_count
        ):
            return False
        pt = self.active_plan.points[self.current_step_idx]
        self.dispatched_counts_x.append(pt.step_delta_x_counts)
        self.current_step_idx += 1
        return True


def test_scheduler_superseding() -> None:
    scheduler = MockScheduler()

    plan1 = TrajectoryPlan(
        plan_id=1,
        point_count=5,
        points=[
            TrajectoryPoint(step_delta_x_counts=10) for _ in range(5)
        ],
    )
    scheduler.submit_plan(plan1)

    # Tick 2 steps of Plan 1
    assert scheduler.tick()
    assert scheduler.tick()

    # Submit Plan 2 (supersedes remainder of Plan 1)
    plan2 = TrajectoryPlan(
        plan_id=2,
        point_count=3,
        points=[
            TrajectoryPoint(step_delta_x_counts=50) for _ in range(3)
        ],
    )
    scheduler.submit_plan(plan2)

    # Tick all steps of Plan 2
    assert scheduler.tick()
    assert scheduler.tick()
    assert scheduler.tick()
    assert not scheduler.tick()  # Finished

    assert scheduler.plans_superseded == 1
    assert scheduler.dispatched_counts_x == [10, 10, 50, 50, 50]
