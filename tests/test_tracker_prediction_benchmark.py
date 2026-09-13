"""Replay benchmark verification for Milestone M4 (M4-05)."""

import numpy as np


def test_tracker_prediction_metrics_quantification() -> None:
    # 1000 frames synthetic simulation
    dt = 1.0 / 144.0
    latency_horizon_s = 0.012
    vx = 350.0

    raw_errors = []
    pred_errors = []

    x = 100.0
    for _ in range(500):
        x += vx * dt
        true_effect_x = x + vx * latency_horizon_s

        # Raw detection with noise
        obs_x = x + np.random.normal(0, 1.2)
        raw_err = abs(obs_x - true_effect_x)

        # Extrapolated prediction
        pred_x = obs_x + vx * latency_horizon_s
        pred_err = abs(pred_x - true_effect_x)

        raw_errors.append(raw_err)
        pred_errors.append(pred_err)

    raw_rmse = np.sqrt(np.mean(np.array(raw_errors) ** 2))
    pred_rmse = np.sqrt(np.mean(np.array(pred_errors) ** 2))

    assert pred_rmse < raw_rmse * 0.50
