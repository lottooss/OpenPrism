"""Unit tests for adaptive command-effect latency estimation and forward prediction extrapolation (M4-04)."""

import numpy as np


class CommandEffectLatencyEstimatorPy:
    def __init__(
        self,
        min_latency_ns: int = 1_500_000,
        max_latency_ns: int = 25_000_000,
        default_latency_ns: int = 6_000_000,
        ewma_alpha: float = 0.15,
        ewma_var_alpha: float = 0.10,
        max_std_dev_ns: float = 3_000_000.0,
        warmup_samples: int = 5,
    ) -> None:
        self.min_latency_ns = min_latency_ns
        self.max_latency_ns = max_latency_ns
        self.default_latency_ns = default_latency_ns
        self.ewma_alpha = ewma_alpha
        self.ewma_var_alpha = ewma_var_alpha
        self.max_std_dev_ns = max_std_dev_ns
        self.warmup_samples = warmup_samples

        self.estimated_latency_ns = float(default_latency_ns)
        self.variance_ns2 = 0.0
        self.sample_count = 0
        self.is_stable = False

    def update(self, measured_latency_ns: int) -> None:
        clamped = max(self.min_latency_ns, min(self.max_latency_ns, measured_latency_ns))
        sample_f = float(clamped)

        if self.sample_count == 0:
            self.estimated_latency_ns = sample_f
            self.variance_ns2 = 0.0
            self.sample_count = 1
            self.is_stable = (self.warmup_samples <= 1)
            return

        diff = sample_f - self.estimated_latency_ns
        self.estimated_latency_ns += self.ewma_alpha * diff
        self.variance_ns2 = (1.0 - self.ewma_var_alpha) * self.variance_ns2 + self.ewma_var_alpha * (diff * diff)
        self.sample_count += 1

        std_dev = np.sqrt(max(0.0, self.variance_ns2))
        self.is_stable = (self.sample_count >= self.warmup_samples) and (std_dev <= self.max_std_dev_ns)


def extrapolate_py(
    x: float,
    y: float,
    vx: float,
    vy: float,
    ax: float,
    ay: float,
    horizon_s: float,
    max_horizon_s: float = 0.050,
) -> tuple[float, float]:
    h = max(0.0, min(max_horizon_s, horizon_s))
    pred_x = x + vx * h + 0.5 * ax * (h * h)
    pred_y = y + vy * h + 0.5 * ay * (h * h)
    return float(pred_x), float(pred_y)


def test_latency_estimator_outliers() -> None:
    estimator = CommandEffectLatencyEstimatorPy(min_latency_ns=2_000_000, max_latency_ns=20_000_000, warmup_samples=4)

    for _ in range(10):
        estimator.update(5_000_000)

    assert estimator.is_stable
    assert abs(estimator.estimated_latency_ns - 5_000_000) < 200_000

    # Large outlier is clamped
    estimator.update(100_000_000)
    assert estimator.estimated_latency_ns <= 8_500_000


def test_extrapolation_accuracy() -> None:
    # 400 px/s moving target, 15ms horizon
    vx = 400.0
    vy = 100.0
    horizon = 0.015

    true_x = 100.0 + vx * horizon
    true_y = 200.0 + vy * horizon

    pred_x, pred_y = extrapolate_py(100.0, 200.0, vx, vy, 0.0, 0.0, horizon)

    raw_err = np.hypot(100.0 - true_x, 200.0 - true_y)
    pred_err = np.hypot(pred_x - true_x, pred_y - true_y)

    assert raw_err > 6.0
    assert pred_err < 1e-6
