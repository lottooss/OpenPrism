"""Unit and numerical tests for Gated Hungarian multi-target data association."""

from dataclasses import dataclass
import numpy as np


@dataclass
class TrackHypothesisPy:
    x: float
    y: float
    radius: float
    cov_xx: float = 1.0
    cov_xy: float = 0.0
    cov_yy: float = 1.0
    class_id: int = 1


@dataclass
class ObservationPy:
    x: float
    y: float
    radius: float
    cov_xx: float = 1.0
    cov_xy: float = 0.0
    cov_yy: float = 1.0
    semantic_id: int = 1


def hungarian_matching_py(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Exact O(N^3) Kuhn-Munkres minimum weight bipartite matching in pure Python."""
    n, m = cost_matrix.shape
    dim = max(n, m)
    # Pad matrix to square
    a = np.full((dim, dim), 1e9, dtype=np.float64)
    a[:n, :m] = cost_matrix

    u = np.zeros(dim + 1, dtype=np.float64)
    v = np.zeros(dim + 1, dtype=np.float64)
    p = np.zeros(dim + 1, dtype=np.int64)
    way = np.zeros(dim + 1, dtype=np.int64)

    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = np.full(dim + 1, 1e18, dtype=np.float64)
        used = np.zeros(dim + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 1e18
            j1 = 0
            for j in range(1, dim + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(dim + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = []
    for j in range(1, dim + 1):
        row = p[j] - 1
        col = j - 1
        if row < n and col < m:
            pairs.append((row, col))
    return pairs


class GatedHungarianAssociatorPy:
    """Python reference implementation of GatedHungarianAssociator."""

    def __init__(
        self,
        gate_distance_px: float = 60.0,
        max_mahalanobis_sq: float = 16.0,
        max_radius_diff_ratio: float = 0.50,
        position_weight: float = 1.0,
        radius_weight: float = 0.2,
    ) -> None:
        self.gate_distance_px = gate_distance_px
        self.max_mahalanobis_sq = max_mahalanobis_sq
        self.max_radius_diff_ratio = max_radius_diff_ratio
        self.position_weight = position_weight
        self.radius_weight = radius_weight
        self.inf_cost = 1e9

    def compute_cost(self, track: TrackHypothesisPy, obs: ObservationPy) -> float:
        # 1. Semantic class ID gate
        if track.class_id != 0 and obs.semantic_id != 0 and track.class_id != obs.semantic_id:
            return self.inf_cost

        # 2. Euclidean distance gate
        dx = track.x - obs.x
        dy = track.y - obs.y
        dist = np.hypot(dx, dy)
        if dist > self.gate_distance_px:
            return self.inf_cost

        # 3. Radius difference ratio gate
        radius_penalty = 0.0
        if track.radius > 0.0 and obs.radius > 0.0:
            max_r = max(track.radius, obs.radius)
            r_diff = abs(track.radius - obs.radius)
            if r_diff / max_r > self.max_radius_diff_ratio:
                return self.inf_cost
            radius_penalty = r_diff

        # 4. Mahalanobis distance gate
        s_xx = max(1e-4, track.cov_xx + obs.cov_xx)
        s_xy = track.cov_xy + obs.cov_xy
        s_yy = max(1e-4, track.cov_yy + obs.cov_yy)

        det = s_xx * s_yy - s_xy * s_xy
        if det > 1e-6:
            inv_det = 1.0 / det
            inv_s_xx = s_yy * inv_det
            inv_s_xy = -s_xy * inv_det
            inv_s_yy = s_xx * inv_det

            d_m_sq = dx * (inv_s_xx * dx + inv_s_xy * dy) + dy * (inv_s_xy * dx + inv_s_yy * dy)
            if d_m_sq > self.max_mahalanobis_sq:
                return self.inf_cost

        cost = self.position_weight * dist + self.radius_weight * radius_penalty
        return max(0.0, float(cost))

    def associate(
        self,
        tracks: list[TrackHypothesisPy],
        observations: list[ObservationPy],
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        n = len(tracks)
        m = len(observations)

        if n == 0:
            return [], [], list(range(m))
        if m == 0:
            return [], list(range(n)), []

        cost_matrix = np.full((n, m), self.inf_cost, dtype=np.float64)
        for i, trk in enumerate(tracks):
            for j, obs in enumerate(observations):
                cost_matrix[i, j] = self.compute_cost(trk, obs)

        assigned_pairs = hungarian_matching_py(cost_matrix)

        matches: list[tuple[int, int, float]] = []
        matched_tracks = set()
        matched_obs = set()

        for r, c in assigned_pairs:
            cost = cost_matrix[r, c]
            if cost < self.inf_cost - 1e4:
                matches.append((int(r), int(c), float(cost)))
                matched_tracks.add(int(r))
                matched_obs.add(int(c))

        unassigned_tracks = [i for i in range(n) if i not in matched_tracks]
        unassigned_obs = [j for j in range(m) if j not in matched_obs]

        return matches, unassigned_tracks, unassigned_obs


def test_perfect_1to1_matching() -> None:
    associator = GatedHungarianAssociatorPy()
    tracks = [
        TrackHypothesisPy(100.0, 100.0, 15.0),
        TrackHypothesisPy(300.0, 300.0, 15.0),
        TrackHypothesisPy(500.0, 500.0, 15.0),
    ]
    obs = [
        ObservationPy(101.0, 100.0, 15.0),
        ObservationPy(299.0, 301.0, 15.0),
        ObservationPy(500.0, 499.0, 15.0),
    ]

    matches, unassigned_t, unassigned_o = associator.associate(tracks, obs)
    assert len(matches) == 3
    assert len(unassigned_t) == 0
    assert len(unassigned_o) == 0
    for r, c, cost in matches:
        assert r == c
        assert cost < 2.0


def test_distance_and_radius_gating() -> None:
    associator = GatedHungarianAssociatorPy(gate_distance_px=50.0, max_radius_diff_ratio=0.30)

    # 1. Dist gate
    trk1 = [TrackHypothesisPy(100.0, 100.0, 15.0)]
    obs1 = [ObservationPy(170.0, 100.0, 15.0)]
    matches, unassigned_t, unassigned_o = associator.associate(trk1, obs1)
    assert len(matches) == 0
    assert unassigned_t == [0]
    assert unassigned_o == [0]

    # 2. Radius gate
    trk2 = [TrackHypothesisPy(100.0, 100.0, 20.0)]
    obs2 = [ObservationPy(100.0, 100.0, 10.0)]
    matches, unassigned_t, unassigned_o = associator.associate(trk2, obs2)
    assert len(matches) == 0
    assert unassigned_t == [0]
    assert unassigned_o == [0]


def test_unequal_dimensions() -> None:
    associator = GatedHungarianAssociatorPy()

    # 3 tracks, 1 obs
    tracks = [
        TrackHypothesisPy(100.0, 100.0, 15.0),
        TrackHypothesisPy(300.0, 300.0, 15.0),
        TrackHypothesisPy(500.0, 500.0, 15.0),
    ]
    obs = [ObservationPy(302.0, 298.0, 15.0)]
    matches, unassigned_t, unassigned_o = associator.associate(tracks, obs)
    assert len(matches) == 1
    assert matches[0][0] == 1
    assert matches[0][1] == 0
    assert len(unassigned_t) == 2
    assert len(unassigned_o) == 0
