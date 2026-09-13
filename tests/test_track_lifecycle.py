"""Unit and state machine tests for multi-target tracking lifecycle (M4-03)."""

from dataclasses import dataclass
import enum
import numpy as np


class TrackStatePy(enum.Enum):
    tentative = 0
    confirmed = 1
    occluded = 2
    coasting = 3
    lost = 4


@dataclass
class ObservationPy:
    x: float
    y: float
    radius: float = 15.0
    confidence: float = 0.95
    semantic_id: int = 1


@dataclass
class TrackPy:
    track_id: int
    state: TrackStatePy
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 15.0
    confidence: float = 0.95
    total_visible: int = 1
    total_missed: int = 0
    consecutive_hits: int = 1
    consecutive_misses: int = 0


class MultiTargetTrackerPy:
    def __init__(
        self,
        immediate_confirm_conf: float = 0.85,
        max_missed_frames: int = 3,
        confidence_decay: float = 0.80,
    ) -> None:
        self.immediate_confirm_conf = immediate_confirm_conf
        self.max_missed_frames = max_missed_frames
        self.confidence_decay = confidence_decay
        self.next_id = 1
        self.tracks: list[TrackPy] = []

    def process(self, observations: list[ObservationPy], dt: float = 1.0 / 144.0) -> list[TrackPy]:
        # 1. Predict
        for t in self.tracks:
            t.x += t.vx * dt
            t.y += t.vy * dt

        # 2. Match nearest
        matched_tracks = set()
        matched_obs = set()

        for t_idx, t in enumerate(self.tracks):
            best_o = -1
            best_dist = 60.0
            for o_idx, o in enumerate(observations):
                if o_idx in matched_obs:
                    continue
                dist = np.hypot(t.x - o.x, t.y - o.y)
                if dist < best_dist:
                    best_dist = dist
                    best_o = o_idx

            if best_o != -1:
                o = observations[best_o]
                # Update velocity estimate
                t.vx = (o.x - t.x) / dt if t.total_visible > 0 else 0.0
                t.vy = (o.y - t.y) / dt if t.total_visible > 0 else 0.0
                t.x = o.x
                t.y = o.y
                t.total_visible += 1
                t.consecutive_hits += 1
                t.consecutive_misses = 0
                t.confidence = o.confidence
                if t.state == TrackStatePy.tentative:
                    if t.consecutive_hits >= 2 or o.confidence >= self.immediate_confirm_conf:
                        t.state = TrackStatePy.confirmed
                elif t.state == TrackStatePy.occluded:
                    t.state = TrackStatePy.confirmed
                matched_tracks.add(t_idx)
                matched_obs.add(best_o)

        # 3. Handle misses
        alive_tracks = []
        for t_idx, t in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                t.total_missed += 1
                t.consecutive_misses += 1
                t.consecutive_hits = 0
                t.confidence *= self.confidence_decay
                if t.state == TrackStatePy.tentative:
                    t.state = TrackStatePy.lost
                elif t.state == TrackStatePy.confirmed:
                    t.state = TrackStatePy.occluded
                    alive_tracks.append(t)
                elif t.state == TrackStatePy.occluded:
                    if t.consecutive_misses <= self.max_missed_frames:
                        alive_tracks.append(t)
                    else:
                        t.state = TrackStatePy.lost
            else:
                alive_tracks.append(t)

        # 4. Handle unassigned obs
        for o_idx, o in enumerate(observations):
            if o_idx not in matched_obs:
                state = (
                    TrackStatePy.confirmed
                    if o.confidence >= self.immediate_confirm_conf
                    else TrackStatePy.tentative
                )
                new_t = TrackPy(
                    track_id=self.next_id,
                    state=state,
                    x=o.x,
                    y=o.y,
                    radius=o.radius,
                    confidence=o.confidence,
                )
                self.next_id += 1
                alive_tracks.append(new_t)

        self.tracks = alive_tracks
        return self.tracks


def test_immediate_and_tentative_lifecycle() -> None:
    tracker = MultiTargetTrackerPy()

    # High conf -> confirmed
    res1 = tracker.process([ObservationPy(100.0, 100.0, confidence=0.90)])
    assert len(res1) == 1
    assert res1[0].state == TrackStatePy.confirmed

    # Low conf -> tentative
    res2 = tracker.process([ObservationPy(100.0, 100.0, confidence=0.90), ObservationPy(300.0, 300.0, confidence=0.60)])
    assert len(res2) == 2
    assert res2[1].state == TrackStatePy.tentative

    # Hit tentative again -> confirmed
    res3 = tracker.process([ObservationPy(100.0, 100.0, confidence=0.90), ObservationPy(302.0, 300.0, confidence=0.60)])
    assert len(res3) == 2
    assert res3[1].state == TrackStatePy.confirmed


def test_occlusion_and_recovery() -> None:
    tracker = MultiTargetTrackerPy()

    tracker.process([ObservationPy(100.0, 100.0, confidence=0.95)])
    t_id = tracker.tracks[0].track_id

    # Miss 1, 2, 3 frames
    tracker.process([])
    assert tracker.tracks[0].state == TrackStatePy.occluded
    tracker.process([])
    assert tracker.tracks[0].state == TrackStatePy.occluded
    tracker.process([])
    assert tracker.tracks[0].state == TrackStatePy.occluded

    # Recover on frame 4
    tracker.process([ObservationPy(101.0, 100.0, confidence=0.90)])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == t_id
    assert tracker.tracks[0].state == TrackStatePy.confirmed
