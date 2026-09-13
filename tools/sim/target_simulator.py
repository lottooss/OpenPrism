"""tools/sim/target_simulator.py
Python deterministic target simulator producing TargetObservationBatchData.
"""

from __future__ import annotations

import math
from typing import Optional

from tools.bus.schema_bindings import (
    CorrelationFlags,
    CorrelationHeader,
    Covariance2f,
    TargetObservationBatchData,
    TargetObservationData,
    Vec2f,
    Visibility,
)
from tools.sim.prng import DeterministicRng, f32
from tools.sim.replay_clock import ReplayClock
from tools.sim.scenario import Scenario
from tools.sim.trajectory import evaluate_trajectory
from tools.sim.types import (
    OcclusionMode,
    TrajectoryState,
    compute_bbox_intersection_area,
    make_bounding_box,
    pixel_to_normalized,
)


class TargetSimulator:
    """Deterministic synthetic target observation batch generator matching C++ simulator."""

    def __init__(self, scenario: Scenario) -> None:
        self._scenario: Scenario = scenario
        self._rng: DeterministicRng = DeterministicRng(scenario.seed)
        self._clock: ReplayClock = ReplayClock(scenario.cadence_hz, scenario.start_time_ns)
        self._total_generated_batches: int = 0
        self._total_generated_observations: int = 0
        self._total_dropped_occlusions: int = 0

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def clock(self) -> ReplayClock:
        return self._clock

    @property
    def rng(self) -> DeterministicRng:
        return self._rng

    @property
    def total_generated_batches(self) -> int:
        return self._total_generated_batches

    @property
    def total_generated_observations(self) -> int:
        return self._total_generated_observations

    @property
    def total_dropped_occlusions(self) -> int:
        return self._total_dropped_occlusions

    def generate_batch(self, timestamp_ns: int) -> TargetObservationBatchData:
        """Generates TargetObservationBatchData at the specified simulation timestamp."""
        elapsed_sec = (
            float(timestamp_ns - self._scenario.start_time_ns) / 1_000_000_000.0
            if timestamp_ns >= self._scenario.start_time_ns
            else 0.0
        )

        header = CorrelationHeader(
            sequence_id=self._clock.sequence_id + 1,
            source_timestamp_ns=timestamp_ns,
            pipeline_run_id=self._scenario.config.pipeline_run_id,
            flags=int(CorrelationFlags.SYNTHETIC),
        )

        batch = TargetObservationBatchData(
            header=header,
            source_id=self._scenario.config.source_id,
            frame_id=self._clock.frame_id,
            captured_at_ns=timestamp_ns,
            published_at_ns=timestamp_ns + self._scenario.config.synthetic_perception_latency_ns,
            source_width=self._scenario.config.source_width,
            source_height=self._scenario.config.source_height,
            targets=[],
        )

        for target_spec in self._scenario.targets:
            # Check lifecycle window
            if timestamp_ns < target_spec.spawn_time_ns or timestamp_ns >= target_spec.despawn_time_ns:
                continue

            state: TrajectoryState = evaluate_trajectory(
                target_spec,
                elapsed_sec,
                self._scenario.config.source_width,
                self._scenario.config.source_height,
            )

            if state.is_despawned or not state.is_active:
                continue

            occluded_drop = False
            is_predicted = False
            is_partial = False
            occlusion_ratio = f32(0.0)

            # 1. Evaluate temporal occlusions
            for temp_occ in self._scenario.temporal_occlusions:
                if (
                    temp_occ.target_id == target_spec.target_id
                    and temp_occ.start_time_ns <= timestamp_ns <= temp_occ.end_time_ns
                ):
                    if temp_occ.mode == OcclusionMode.DROP:
                        occluded_drop = True
                    elif temp_occ.mode == OcclusionMode.PREDICTED:
                        is_predicted = True
                    elif temp_occ.mode == OcclusionMode.PARTIAL:
                        is_partial = True

            # 2. Evaluate rectangular occluders
            target_bbox = make_bounding_box(state.pos_px, state.radius_px)
            target_area = f32(f32(4.0 * state.radius_px) * state.radius_px)

            for rect_occ in self._scenario.rect_occluders:
                if rect_occ.start_time_ns <= timestamp_ns <= rect_occ.end_time_ns:
                    inter_area = compute_bbox_intersection_area(target_bbox, rect_occ.bounds_px)
                    if inter_area > 0 and target_area > 0:
                        ratio = min(1.0, f32(inter_area / target_area))
                        occlusion_ratio = max(occlusion_ratio, ratio)
                        if ratio >= 0.75:
                            if rect_occ.mode == OcclusionMode.DROP:
                                if self._scenario.config.emit_predicted_when_occluded:
                                    is_predicted = True
                                else:
                                    occluded_drop = True
                            elif rect_occ.mode == OcclusionMode.PREDICTED:
                                is_predicted = True
                            else:
                                is_partial = True
                        else:
                            is_partial = True

            # 3. Evaluate circular occluders
            for circle_occ in self._scenario.circle_occluders:
                if circle_occ.start_time_ns <= timestamp_ns <= circle_occ.end_time_ns:
                    dx = f32(state.pos_px.x - f32(circle_occ.center_px.x))
                    dy = f32(state.pos_px.y - f32(circle_occ.center_px.y))
                    dist_sq = f32(f32(dx * dx) + f32(dy * dy))
                    rad_sum = f32(state.radius_px + f32(circle_occ.radius_px))
                    if dist_sq < f32(rad_sum * rad_sum):
                        dist = f32(math.sqrt(dist_sq))
                        if f32(dist + state.radius_px) <= circle_occ.radius_px:
                            occlusion_ratio = 1.0
                            if circle_occ.mode == OcclusionMode.DROP:
                                if self._scenario.config.emit_predicted_when_occluded:
                                    is_predicted = True
                                else:
                                    occluded_drop = True
                            elif circle_occ.mode == OcclusionMode.PREDICTED:
                                is_predicted = True
                            else:
                                is_partial = True
                        else:
                            occlusion_ratio = max(occlusion_ratio, 0.5)
                            is_partial = True

            # Handle drop
            if occluded_drop:
                self._total_dropped_occlusions += 1
                continue

            # 4. Noise & Uncertainty Model
            pos_noise_sigma = f32(
                target_spec.noise.position_stddev_px
                if target_spec.noise.position_stddev_px > 0
                else self._scenario.config.default_noise.position_stddev_px
            )
            vel_noise_sigma = f32(
                target_spec.noise.velocity_stddev_px_per_s
                if target_spec.noise.velocity_stddev_px_per_s > 0
                else self._scenario.config.default_noise.velocity_stddev_px_per_s
            )
            base_conf = f32(
                target_spec.noise.base_confidence
                if target_spec.noise.base_confidence > 0
                else self._scenario.config.default_noise.base_confidence
            )
            dropout_prob = f32(
                target_spec.noise.dropout_probability
                if target_spec.noise.dropout_probability > 0
                else self._scenario.config.default_noise.dropout_probability
            )

            if dropout_prob > 0 and self._rng.next_uniform_f32() < dropout_prob:
                self._total_dropped_occlusions += 1
                continue

            obs_x = state.pos_px.x
            obs_y = state.pos_px.y
            if pos_noise_sigma > 0:
                nx, ny = self._rng.next_gaussian_pair(0.0, pos_noise_sigma)
                obs_x = f32(obs_x + nx)
                obs_y = f32(obs_y + ny)

            obs_vx = state.vel_px_per_s.x
            obs_vy = state.vel_px_per_s.y
            if vel_noise_sigma > 0:
                vnx, vny = self._rng.next_gaussian_pair(0.0, vel_noise_sigma)
                obs_vx = f32(obs_vx + vnx)
                obs_vy = f32(obs_vy + vny)

            vis = Visibility.VISIBLE
            conf = base_conf
            var_px2 = f32(pos_noise_sigma * pos_noise_sigma) if pos_noise_sigma > 0 else f32(0.01)

            if is_predicted:
                vis = Visibility.PREDICTED
                conf = f32(base_conf * f32(0.30))
                var_px2 = f32(var_px2 + 25.0)
            elif is_partial:
                vis = Visibility.PARTIAL
                conf = f32(base_conf * f32(1.0 - f32(0.5 * occlusion_ratio)))
                var_px2 = f32(var_px2 * f32(1.0 + f32(f32(3.0 * occlusion_ratio) * occlusion_ratio)))

            obs_pos = Vec2f(obs_x, obs_y)
            obs_vel = Vec2f(obs_vx, obs_vy)

            obs = TargetObservationData(
                source_id=target_spec.target_id,
                frame_id=self._clock.frame_id,
                captured_at_ns=timestamp_ns,
                center_px=obs_pos,
                center_norm=pixel_to_normalized(obs_pos, self._scenario.config.source_width, self._scenario.config.source_height),
                bbox_px=make_bounding_box(obs_pos, state.radius_px),
                effective_radius_px=state.radius_px,
                confidence=conf,
                covariance_px2=Covariance2f(xx=var_px2, xy=0.0, yy=var_px2),
                velocity_px_per_s=obs_vel,
                velocity_confidence=f32(0.90) if vel_noise_sigma > 0 else 1.0,
                visibility=vis,
                target_value=f32(target_spec.target_value),
                semantic_id=target_spec.semantic_id,
            )

            if len(batch.targets) < 64:
                batch.targets.append(obs)
                self._total_generated_observations += 1

        self._total_generated_batches += 1
        return batch

    def generate_current_batch(self) -> TargetObservationBatchData:
        return self.generate_batch(self._clock.now_ns())

    def step(self) -> TargetObservationBatchData:
        batch = self.generate_batch(self._clock.now_ns())
        self._clock.advance_tick()
        return batch

    def reset(self, new_seed: Optional[int] = None) -> None:
        seed = new_seed if new_seed is not None else self._scenario.config.seed
        self._rng.reseed(seed)
        self._clock.reset(self._scenario.config.start_time_ns)
        self._total_generated_batches = 0
        self._total_generated_observations = 0
        self._total_dropped_occlusions = 0
