"""Unit and numerical tests for Kalman kinematics, Joseph-form stability, and extrapolation."""

import numpy as np


class KalmanSimulation2D:
    """Python reference simulator for 2D position, velocity, and acceleration tracking."""

    def __init__(self, q_pos: float = 10.0, q_vel: float = 50.0, r_val: float = 2.0) -> None:
        self.x = np.zeros(4, dtype=np.float64)  # [px, py, vx, vy]
        self.p = np.eye(4, dtype=np.float64) * 100.0
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_val = r_val

    def initialize(self, x: float, y: float) -> None:
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.p = np.diag([2.0, 2.0, 500.0, 500.0])

    def predict(self, dt: float) -> None:
        f = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.x = f @ self.x
        q = np.diag([self.q_pos * dt, self.q_pos * dt, self.q_vel * dt, self.q_vel * dt])
        self.p = f @ self.p @ f.T + q

    def update(self, z_x: float, z_y: float) -> float:
        h = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        r = np.eye(2) * self.r_val
        y = np.array([z_x, z_y]) - h @ self.x
        s = h @ self.p @ h.T + r
        inv_s = np.linalg.inv(s)
        k = self.p @ h.T @ inv_s

        self.x = self.x + k @ y
        i_kh = np.eye(4) - k @ h
        # Joseph form
        self.p = i_kh @ self.p @ i_kh.T + k @ r @ k.T
        # Symmetrize
        self.p = 0.5 * (self.p + self.p.T)

        d_m_sq = float(y.T @ inv_s @ y)
        return d_m_sq

    def extrapolate(self, horizon: float) -> tuple[float, float]:
        px = self.x[0] + horizon * self.x[2]
        py = self.x[1] + horizon * self.x[3]
        return float(px), float(py)


def test_kalman_reference_simulation() -> None:
    """Verify linear trajectory estimation and forward extrapolation."""
    sim = KalmanSimulation2D()
    sim.initialize(100.0, 200.0)

    dt = 1.0 / 144.0
    vx_true = 250.0
    vy_true = -100.0

    cx = 100.0
    cy = 200.0
    for _ in range(80):
        cx += vx_true * dt
        cy += vy_true * dt
        sim.predict(dt)
        d_m = sim.update(cx, cy)
        assert d_m >= 0.0

    assert abs(sim.x[0] - cx) < 1.0
    assert abs(sim.x[1] - cy) < 1.0
    assert abs(sim.x[2] - vx_true) < 15.0
    assert abs(sim.x[3] - vy_true) < 15.0

    # Extrapolate 20ms
    ex_x, ex_y = sim.extrapolate(0.020)
    assert abs(ex_x - (cx + vx_true * 0.020)) < 1.5
    assert abs(ex_y - (cy + vy_true * 0.020)) < 1.5
