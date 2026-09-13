# tools/policy/learned_aim_policy.py
"""Neural behavioral cloning imitation policy conforming to canonical AimPolicy contracts (M9-04)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PolicyTrackedTarget:
    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.95
    target_value: float = 1.0
    is_confirmed: bool = True
    covariance_trace: float = 4.0  # px^2


@dataclass
class LearnedPolicyWeights:
    """Network weights for a 2-layer MLP (10 -> 32 -> 16 -> 1)."""
    w1: np.ndarray  # [10, 32]
    b1: np.ndarray  # [32]
    w2: np.ndarray  # [32, 16]
    b2: np.ndarray  # [16]
    w_score: np.ndarray  # [16, 1]
    b_score: np.ndarray  # [1]
    w_fire: np.ndarray   # [16, 1]
    b_fire: np.ndarray   # [1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w_score": self.w_score.tolist(),
            "b_score": self.b_score.tolist(),
            "w_fire": self.w_fire.tolist(),
            "b_fire": self.b_fire.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LearnedPolicyWeights:
        return cls(
            w1=np.array(data["w1"], dtype=np.float32),
            b1=np.array(data["b1"], dtype=np.float32),
            w2=np.array(data["w2"], dtype=np.float32),
            b2=np.array(data["b2"], dtype=np.float32),
            w_score=np.array(data["w_score"], dtype=np.float32),
            b_score=np.array(data["b_score"], dtype=np.float32),
            w_fire=np.array(data["w_fire"], dtype=np.float32),
            b_fire=np.array(data["b_fire"], dtype=np.float32),
        )


def default_policy_weights(seed: int = 42) -> LearnedPolicyWeights:
    """Initialize weights with He-normal distribution."""
    rng = np.random.default_rng(seed)
    return LearnedPolicyWeights(
        w1=rng.normal(0, np.sqrt(2.0 / 10), (10, 32)).astype(np.float32),
        b1=np.zeros(32, dtype=np.float32),
        w2=rng.normal(0, np.sqrt(2.0 / 32), (32, 16)).astype(np.float32),
        b2=np.zeros(16, dtype=np.float32),
        w_score=rng.normal(0, np.sqrt(2.0 / 16), (16, 1)).astype(np.float32),
        b_score=np.zeros(1, dtype=np.float32),
        w_fire=rng.normal(0, np.sqrt(2.0 / 16), (16, 1)).astype(np.float32),
        b_fire=np.zeros(1, dtype=np.float32),
    )


class LearnedAimPolicy:
    """Production candidate learned imitation policy.

    Replaces the deterministic utility policy behind the canonical IAimPolicy contract.
    Operates strictly on canonical tracked target observations with zero perception coupling.
    Guarantees that policy outputs cannot bypass safety bounds.
    """

    FEATURE_DIM = 10

    def __init__(
        self,
        weights: Optional[LearnedPolicyWeights] = None,
        fire_threshold_px: float = 8.0,
        min_engagement_confidence: float = 0.50,
        screen_width: float = 1920.0,
        screen_height: float = 1080.0,
    ) -> None:
        self.weights = weights if weights is not None else default_policy_weights()
        self.fire_threshold_px = fire_threshold_px
        self.min_engagement_confidence = min_engagement_confidence
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.current_locked_id: Optional[int] = None

    def extract_features(
        self,
        crosshair_x: float,
        crosshair_y: float,
        target: PolicyTrackedTarget,
    ) -> np.ndarray:
        """Extract canonical 10-dimensional normalized feature vector for a tracked target."""
        dx = target.x - crosshair_x
        dy = target.y - crosshair_y
        dist = math.hypot(dx, dy)

        norm_dx = dx / (self.screen_width * 0.5)
        norm_dy = dy / (self.screen_height * 0.5)
        norm_dist = dist / self.screen_width
        norm_vx = target.vx / 1000.0
        norm_vy = target.vy / 1000.0
        conf = target.confidence
        val = target.target_value
        is_conf = 1.0 if target.is_confirmed else 0.0
        unc = min(target.covariance_trace / 100.0, 1.0)
        is_cur = 1.0 if (self.current_locked_id is not None and target.track_id == self.current_locked_id) else 0.0

        return np.array(
            [norm_dx, norm_dy, norm_dist, norm_vx, norm_vy, conf, val, is_conf, unc, is_cur],
            dtype=np.float32,
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy forward pass: [N, 10] -> ([N, 1], [N, 1])."""
        # Layer 1
        h1 = np.maximum(0, x @ self.weights.w1 + self.weights.b1)
        # Layer 2
        h2 = np.maximum(0, h1 @ self.weights.w2 + self.weights.b2)
        # Score head (linear)
        scores = h2 @ self.weights.w_score + self.weights.b_score
        # Fire head (sigmoid)
        fire_logits = h2 @ self.weights.w_fire + self.weights.b_fire
        fire_probs = 1.0 / (1.0 + np.exp(-np.clip(fire_logits, -15.0, 15.0)))
        return scores, fire_probs

    def choose(
        self,
        crosshair_x: float,
        crosshair_y: float,
        tracks: List[PolicyTrackedTarget],
    ) -> Optional[Dict[str, Any]]:
        """Choose target and produce AimIntent conforming strictly to canonical contracts."""
        if not tracks:
            self.current_locked_id = None
            return None

        valid_candidates: List[PolicyTrackedTarget] = []
        feature_list: List[np.ndarray] = []

        for t in tracks:
            if t.confidence < self.min_engagement_confidence:
                continue
            feats = self.extract_features(crosshair_x, crosshair_y, t)
            valid_candidates.append(t)
            feature_list.append(feats)

        if not valid_candidates:
            self.current_locked_id = None
            return None

        feat_tensor = np.stack(feature_list)
        scores, fire_probs = self.forward(feat_tensor)

        best_idx = int(np.argmax(scores.flatten()))
        best_target = valid_candidates[best_idx]
        best_score = float(scores.flatten()[best_idx])
        model_fire_prob = float(fire_probs.flatten()[best_idx])

        self.current_locked_id = best_target.track_id
        dist = math.hypot(best_target.x - crosshair_x, best_target.y - crosshair_y)

        # Policy CANNOT bypass safety supervisor bounds:
        # Fire authorization requires: within pixel distance threshold, confirmed track, and confidence
        authorize_fire = (
            dist <= self.fire_threshold_px
            and best_target.is_confirmed
            and best_target.confidence >= self.min_engagement_confidence
            and model_fire_prob >= 0.25
        )

        if dist > 40.0:
            mode = "flick"
        elif dist <= self.fire_threshold_px:
            mode = "micro_correction"
        else:
            mode = "tracking"

        return {
            "target_track_id": best_target.track_id,
            "aim_x": best_target.x,
            "aim_y": best_target.y,
            "error_distance_px": dist,
            "authorize_fire": authorize_fire,
            "utility_score": best_score,
            "mode": mode,
            "confidence": best_target.confidence,
        }

    def reset(self) -> None:
        self.current_locked_id = None


def train_numpy_imitation_policy(
    num_samples: int = 5000,
    epochs: int = 25,
    lr: float = 0.01,
    seed: int = 42,
) -> LearnedPolicyWeights:
    """Train neural imitation policy on oracle deterministic utility selections using NumPy."""
    rng = np.random.default_rng(seed)
    weights = default_policy_weights(seed=seed)

    features_list: List[np.ndarray] = []
    target_scores: List[float] = []
    target_fires: List[float] = []

    # Training policy instance to extract features
    temp_policy = LearnedAimPolicy(weights=weights)

    for _ in range(num_samples):
        cx, cy = 960.0, 540.0
        # Stratified sampling: 25% of samples near crosshair to learn fire threshold
        if rng.random() < 0.25:
            r = float(rng.uniform(0.0, 15.0))
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            tx = cx + r * math.cos(theta)
            ty = cy + r * math.sin(theta)
        else:
            tx = float(rng.uniform(200, 1720))
            ty = float(rng.uniform(150, 930))
        dist = math.hypot(tx - cx, ty - cy)
        conf = float(rng.uniform(0.5, 1.0))
        val = float(rng.choice([0.8, 1.0, 1.2]))
        is_conf = bool(rng.random() > 0.1)
        unc = float(rng.choice([8.0, 20.0, float(rng.uniform(1.0, 25.0))]))
        is_cur = bool(rng.random() > 0.7)

        target = PolicyTrackedTarget(
            track_id=1,
            x=tx,
            y=ty,
            vx=float(rng.uniform(-50, 50)),
            vy=float(rng.uniform(-50, 50)),
            confidence=conf,
            target_value=val,
            is_confirmed=is_conf,
            covariance_trace=unc,
        )

        state_mult = 1.2 if is_conf else 0.8
        oracle_score = val * state_mult + 1.5 * conf - 0.005 * dist - 0.10 * math.sqrt(unc)
        if is_cur:
            oracle_score += 1.5
        oracle_fire = 1.0 if (dist <= 8.0 and is_conf and conf >= 0.5) else 0.0

        if is_cur:
            temp_policy.current_locked_id = 1
        else:
            temp_policy.current_locked_id = None
        feats = temp_policy.extract_features(cx, cy, target)

        features_list.append(feats)
        target_scores.append(oracle_score)
        target_fires.append(oracle_fire)

    x = np.stack(features_list)
    y_score = np.array(target_scores, dtype=np.float32).reshape(-1, 1)
    y_fire = np.array(target_fires, dtype=np.float32).reshape(-1, 1)

    # Mini-batch gradient descent with momentum
    batch_size = 64
    n_batches = len(x) // batch_size

    # Velocity accumulators for momentum
    v_w1, v_b1 = np.zeros_like(weights.w1), np.zeros_like(weights.b1)
    v_w2, v_b2 = np.zeros_like(weights.w2), np.zeros_like(weights.b2)
    v_ws, v_bs = np.zeros_like(weights.w_score), np.zeros_like(weights.b_score)
    v_wf, v_bf = np.zeros_like(weights.w_fire), np.zeros_like(weights.b_fire)
    momentum = 0.9

    for _ in range(epochs):
        perm = rng.permutation(len(x))
        x_shuf = x[perm]
        ys_shuf = y_score[perm]
        yf_shuf = y_fire[perm]

        for b in range(n_batches):
            xb = x_shuf[b * batch_size : (b + 1) * batch_size]
            ysb = ys_shuf[b * batch_size : (b + 1) * batch_size]
            yfb = yf_shuf[b * batch_size : (b + 1) * batch_size]

            # Forward
            z1 = xb @ weights.w1 + weights.b1
            h1 = np.maximum(0, z1)
            z2 = h1 @ weights.w2 + weights.b2
            h2 = np.maximum(0, z2)

            pred_score = h2 @ weights.w_score + weights.b_score
            fire_logits = h2 @ weights.w_fire + weights.b_fire
            pred_fire = 1.0 / (1.0 + np.exp(-np.clip(fire_logits, -15.0, 15.0)))

            # Gradients for score head (MSE)
            d_score = (2.0 / batch_size) * (pred_score - ysb)
            d_w_score = h2.T @ d_score
            d_b_score = np.sum(d_score, axis=0)

            # Gradients for fire head (BCE with logits)
            d_fire = (1.0 / batch_size) * (pred_fire - yfb)
            d_w_fire = h2.T @ d_fire
            d_b_fire = np.sum(d_fire, axis=0)

            # Backprop into Layer 2
            d_h2 = d_score @ weights.w_score.T + d_fire @ weights.w_fire.T
            d_z2 = d_h2 * (z2 > 0)
            d_w2 = h1.T @ d_z2
            d_b2 = np.sum(d_z2, axis=0)

            # Backprop into Layer 1
            d_h1 = d_z2 @ weights.w2.T
            d_z1 = d_h1 * (z1 > 0)
            d_w1 = xb.T @ d_z1
            d_b1 = np.sum(d_z1, axis=0)

            # Apply momentum updates
            v_w1 = momentum * v_w1 - lr * d_w1
            v_b1 = momentum * v_b1 - lr * d_b1
            v_w2 = momentum * v_w2 - lr * d_w2
            v_b2 = momentum * v_b2 - lr * d_b2
            v_ws = momentum * v_ws - lr * d_w_score
            v_bs = momentum * v_bs - lr * d_b_score
            v_wf = momentum * v_wf - lr * d_w_fire
            v_bf = momentum * v_bf - lr * d_b_fire

            weights.w1 += v_w1
            weights.b1 += v_b1
            weights.w2 += v_w2
            weights.b2 += v_b2
            weights.w_score += v_ws
            weights.b_score += v_bs
            weights.w_fire += v_wf
            weights.b_fire += v_bf

    return weights
