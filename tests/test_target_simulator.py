"""tests/test_target_simulator.py
Comprehensive Python unit tests for Deterministic Target Simulator and Replay Clock (Milestone M1-05 #11).
"""

from __future__ import annotations

import math
import unittest

from tools.bus.ring import LatestSpscRing
from tools.bus.schema_bindings import (
    BBoxf,
    TargetObservationBatchData,
    Vec2f,
    Visibility,
)
from tools.sim.prng import DeterministicRng
from tools.sim.replay_clock import ReplayClock
from tools.sim.scenario import ScenarioBuilder
from tools.sim.target_simulator import TargetSimulator
from tools.sim.trajectory import evaluate_trajectory
from tools.sim.types import (
    BoundaryBehavior,
    OcclusionMode,
    TargetSpec,
    TrajectoryType,
    normalized_to_pixel,
    pixel_to_normalized,
)


class TestTargetSimulator(unittest.TestCase):
    """Test suite verifying simulation kinematics, replay clock, determinism, occlusions, and ring bus publishing."""

    def test_coordinate_transforms_and_subpixel_accuracy(self) -> None:
        """TC-SIM-01: Verifies coordinate conversions between 1080p and [-1, 1] normalized."""
        width = 1920
        height = 1080

        # Center (960, 540) -> (0, 0)
        c_norm = pixel_to_normalized(Vec2f(960.0, 540.0), width, height)
        self.assertAlmostEqual(c_norm.x, 0.0, places=6)
        self.assertAlmostEqual(c_norm.y, 0.0, places=6)

        c_px = normalized_to_pixel(c_norm, width, height)
        self.assertAlmostEqual(c_px.x, 960.0, places=6)
        self.assertAlmostEqual(c_px.y, 540.0, places=6)

        # Top-left (0, 0) -> (-1, -1)
        tl_norm = pixel_to_normalized(Vec2f(0.0, 0.0), width, height)
        self.assertAlmostEqual(tl_norm.x, -1.0, places=6)
        self.assertAlmostEqual(tl_norm.y, -1.0, places=6)

        # Bottom-right (1920, 1080) -> (1, 1)
        br_norm = pixel_to_normalized(Vec2f(1920.0, 1080.0), width, height)
        self.assertAlmostEqual(br_norm.x, 1.0, places=6)
        self.assertAlmostEqual(br_norm.y, 1.0, places=6)

        # 10,000 subpixel roundtrip checks
        max_err = 0.0
        for ix in range(101):
            for iy in range(101):
                px = Vec2f(float(ix) * 19.2, float(iy) * 10.8)
                norm = pixel_to_normalized(px, width, height)
                self.assertTrue(-1.0001 <= norm.x <= 1.0001)
                self.assertTrue(-1.0001 <= norm.y <= 1.0001)

                rt = normalized_to_pixel(norm, width, height)
                err_x = abs(rt.x - px.x)
                err_y = abs(rt.y - px.y)
                max_err = max(max_err, err_x, err_y)

        # Matches the 1e-3 px bound in TC-SIM-01 (tests/cpp/test_target_simulator.cpp).
        # This assertion used to demand 1e-5, which only held because the Python helpers
        # computed in binary64 while the shipped C++ path computes in binary32. Tightening
        # it again would mean this suite is grading a more precise implementation than the
        # one that actually runs.
        self.assertLess(max_err, 1e-3)

    def test_trajectory_kinematics_cv_ca_sinusoidal_circular_cut_bounce(self) -> None:
        """TC-SIM-02: Verifies all closed-form kinematic formulas and boundary interaction."""
        # 1. Stationary
        stat_spec = TargetSpec(
            trajectory_type=TrajectoryType.STATIONARY,
            initial_pos_px=Vec2f(500.0, 400.0),
        )
        st = evaluate_trajectory(stat_spec, 5.0)
        self.assertEqual(st.pos_px.x, 500.0)
        self.assertEqual(st.pos_px.y, 400.0)
        self.assertEqual(st.vel_px_per_s.x, 0.0)
        self.assertEqual(st.vel_px_per_s.y, 0.0)

        # 2. Linear Constant Velocity (CV)
        cv_spec = TargetSpec(
            trajectory_type=TrajectoryType.LINEAR_CV,
            boundary=BoundaryBehavior.NONE,
            initial_pos_px=Vec2f(100.0, 200.0),
            velocity_px_s=Vec2f(50.0, -20.0),
        )
        st_cv = evaluate_trajectory(cv_spec, 4.0)
        self.assertAlmostEqual(st_cv.pos_px.x, 300.0, places=5)
        self.assertAlmostEqual(st_cv.pos_px.y, 120.0, places=5)

        # 3. Linear Constant Acceleration (CA)
        ca_spec = TargetSpec(
            trajectory_type=TrajectoryType.LINEAR_CA,
            boundary=BoundaryBehavior.NONE,
            initial_pos_px=Vec2f(0.0, 0.0),
            velocity_px_s=Vec2f(10.0, 0.0),
            accel_px_s2=Vec2f(5.0, 0.0),
        )
        st_ca = evaluate_trajectory(ca_spec, 2.0)
        # x = 0 + 10*2 + 0.5*5*4 = 30; vx = 10 + 5*2 = 20
        self.assertAlmostEqual(st_ca.pos_px.x, 30.0, places=5)
        self.assertAlmostEqual(st_ca.vel_px_per_s.x, 20.0, places=5)

        # 4. Sinusoidal Strafe
        sin_spec = TargetSpec(
            trajectory_type=TrajectoryType.SINUSOIDAL_STRAFE,
            initial_pos_px=Vec2f(960.0, 540.0),
            amplitude_x_px=300.0,
            amplitude_y_px=0.0,
            frequency_hz=1.0,
        )
        st_sin_qtr = evaluate_trajectory(sin_spec, 0.25)
        self.assertAlmostEqual(st_sin_qtr.pos_px.x, 1260.0, places=2)
        self.assertAlmostEqual(st_sin_qtr.vel_px_per_s.x, 0.0, places=2)

        # 5. Circular Orbit
        circ_spec = TargetSpec(
            trajectory_type=TrajectoryType.CIRCULAR_ORBIT,
            initial_pos_px=Vec2f(960.0, 540.0),
            amplitude_x_px=150.0,
            frequency_hz=0.5,
        )
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            st_c = evaluate_trajectory(circ_spec, t)
            dx = st_c.pos_px.x - 960.0
            dy = st_c.pos_px.y - 540.0
            r = math.sqrt(dx * dx + dy * dy)
            self.assertAlmostEqual(r, 150.0, places=3)

        # 6. Specular Bounce
        bounce_spec = TargetSpec(
            trajectory_type=TrajectoryType.LINEAR_CV,
            boundary=BoundaryBehavior.BOUNCE,
            radius_px=20.0,
            initial_pos_px=Vec2f(1800.0, 540.0),
            velocity_px_s=Vec2f(200.0, 0.0),
        )
        # Touch at t=0.5s (1900)
        st_bt = evaluate_trajectory(bounce_spec, 0.5)
        self.assertAlmostEqual(st_bt.pos_px.x, 1900.0, places=4)

        # Rebound at t=1.0s (1800, vx = -200)
        st_br = evaluate_trajectory(bounce_spec, 1.0)
        self.assertAlmostEqual(st_br.pos_px.x, 1800.0, places=4)
        self.assertEqual(st_br.vel_px_per_s.x, -200.0)

        # 7. Sudden Cut
        cut_spec = TargetSpec(
            trajectory_type=TrajectoryType.SUDDEN_CUT,
            initial_pos_px=Vec2f(500.0, 500.0),
            velocity_px_s=Vec2f(100.0, 0.0),
            cut_interval_ns=1_000_000_000,
        )
        st_cut05 = evaluate_trajectory(cut_spec, 0.5)
        self.assertAlmostEqual(st_cut05.pos_px.x, 550.0, places=4)
        self.assertEqual(st_cut05.vel_px_per_s.x, 100.0)

        st_cut15 = evaluate_trajectory(cut_spec, 1.5)
        self.assertAlmostEqual(st_cut15.pos_px.x, 550.0, places=4)
        self.assertEqual(st_cut15.vel_px_per_s.x, -100.0)

    def test_replay_clock_144hz_zero_drift_and_controls(self) -> None:
        """TC-SIM-03: Verifies 144 Hz frame cadence and zero long-term timestamp drift."""
        t0 = 1_000_000_000
        clock = ReplayClock(144.0, t0)

        self.assertEqual(clock.now_ns(), t0)
        self.assertEqual(clock.tick_index, 0)

        # Step 144 frames -> exactly 1 second (1,000,000,000 ns)
        clock.step(144)
        self.assertEqual(clock.tick_index, 144)
        self.assertEqual(clock.elapsed_ns, 1_000_000_000)
        self.assertEqual(clock.now_ns(), 2_000_000_000)

        # 10,000 frames drift verification
        clock.reset(t0)
        for i in range(1, 10001):
            clock.step(1)
            expected_s = float(i) / 144.0
            expected_ns = t0 + int(expected_s * 1_000_000_000.0)
            self.assertLessEqual(abs(clock.now_ns() - expected_ns), 1)

        # Pause / Resume
        clock.reset(t0)
        clock.pause()
        self.assertTrue(clock.is_paused)
        clock.step(10)
        self.assertEqual(clock.tick_index, 0)
        self.assertEqual(clock.now_ns(), t0)

        clock.resume()
        self.assertFalse(clock.is_paused)
        clock.step(1)
        self.assertEqual(clock.tick_index, 1)

        # Rate scale (2.0x)
        clock.reset(t0)
        clock.set_rate_scale(2.0)
        clock.step(144)
        self.assertEqual(clock.elapsed_ns, 2_000_000_000)

    def test_prng_splitmix64_determinism_and_reseed(self) -> None:
        """TC-SIM-04: Verifies PRNG deterministic sequence and cross-run bit parity."""
        seed = 0x123456789ABCDEF0
        rng1 = DeterministicRng(seed)
        rng2 = DeterministicRng(seed)

        for _ in range(5000):
            self.assertEqual(rng1.next_u64(), rng2.next_u64())

        rng1.reseed(seed)
        rng2.reseed(seed)
        for _ in range(5000):
            self.assertEqual(rng1.next_uniform_f32(), rng2.next_uniform_f32())
            self.assertEqual(rng1.next_gaussian_pair(), rng2.next_gaussian_pair())

    def test_target_lifecycle_spawning_and_despawning(self) -> None:
        """TC-SIM-05: Verifies active target window [spawn_time_ns, despawn_time_ns)."""
        t0 = 1_000_000_000
        scen = (
            ScenarioBuilder("lifecycle")
            .with_seed(100)
            .with_start_time_ns(t0)
            .add_stationary_target(1, Vec2f(500.0, 500.0), 20.0, spawn_ns=t0, despawn_ns=t0 + 2_000_000_000)
            .add_stationary_target(2, Vec2f(700.0, 500.0), 20.0, spawn_ns=t0 + 1_000_000_000, despawn_ns=t0 + 4_000_000_000)
            .build()
        )

        sim = TargetSimulator(scen)

        # t = 1.0s: target 1 only
        b1 = sim.generate_batch(t0)
        self.assertEqual(len(b1.targets), 1)
        self.assertEqual(b1.targets[0].source_id, 1)

        # t = 2.5s: target 1 and 2
        b2 = sim.generate_batch(t0 + 1_500_000_000)
        self.assertEqual(len(b2.targets), 2)

        # t = 3.5s: target 2 only
        b3 = sim.generate_batch(t0 + 2_500_000_000)
        self.assertEqual(len(b3.targets), 1)
        self.assertEqual(b3.targets[0].source_id, 2)

        # t = 5.0s: all despawned
        b4 = sim.generate_batch(t0 + 4_000_000_000)
        self.assertEqual(len(b4.targets), 0)

    def test_temporal_and_spatial_occlusions(self) -> None:
        """TC-SIM-06: Verifies temporal drop and spatial predicted occlusions."""
        t0 = 1_000_000_000
        scen = (
            ScenarioBuilder("occlusion")
            .with_seed(200)
            .with_start_time_ns(t0)
            .with_emit_predicted_when_occluded(True)
            .add_linear_target(1, Vec2f(700.0, 500.0), Vec2f(100.0, 0.0), 20.0, boundary=BoundaryBehavior.NONE)
            .add_rect_occluder(
                bounds=BBoxf(left=900.0, top=400.0, right=1020.0, bottom=600.0),
                start_ns=t0,
                end_ns=t0 + 10_000_000_000,
                mode=OcclusionMode.PREDICTED,
            )
            .build()
        )

        sim = TargetSimulator(scen)

        # Before occluder (x=700): visible
        b0 = sim.generate_batch(t0)
        self.assertEqual(b0.targets[0].visibility, Visibility.VISIBLE)

        # Inside occluder (x=960 at t=2.6s): predicted
        b_occ = sim.generate_batch(t0 + 2_600_000_000)
        self.assertEqual(b_occ.targets[0].visibility, Visibility.PREDICTED)
        self.assertLess(b_occ.targets[0].confidence, 0.50)
        self.assertGreater(b_occ.targets[0].covariance_px2.xx, 20.0)

        # Past occluder (x=1200 at t=5.0s): visible
        b_clear = sim.generate_batch(t0 + 5_000_000_000)
        self.assertEqual(b_clear.targets[0].visibility, Visibility.VISIBLE)

    def test_scenario_builder_and_standard_presets(self) -> None:
        """TC-SIM-07: Verifies standard presets (Gridshot, Strafe Track, Occlusion, Density)."""
        gridshot = ScenarioBuilder.make_gridshot_preset()
        self.assertEqual(gridshot.target_count, 3)

        strafe = ScenarioBuilder.make_strafe_track_preset()
        self.assertEqual(strafe.target_count, 2)

        density = ScenarioBuilder.make_density_preset(target_count=64)
        self.assertEqual(density.target_count, 64)

        sim_dense = TargetSimulator(density)
        b_dense = sim_dense.step()
        self.assertEqual(len(b_dense.targets), 64)

    def test_bus_ring_integration_and_latest_wins(self) -> None:
        """TC-SIM-08: Verifies integration with LatestSpscRing."""
        scen = ScenarioBuilder.make_strafe_track_preset(42)
        sim = TargetSimulator(scen)
        ring: LatestSpscRing[TargetObservationBatchData] = LatestSpscRing(capacity=16)

        # Publish 500 batches to ring
        for i in range(1, 501):
            batch = sim.step()
            seq = ring.push(batch)
            self.assertEqual(seq, i)

        self.assertEqual(ring.latest_sequence, 500)

        # Read latest
        ok, read_batch, last_seq, drops = ring.try_read_latest()
        self.assertTrue(ok)
        self.assertIsNotNone(read_batch)
        if read_batch is not None:
            self.assertEqual(last_seq, 500)
            self.assertEqual(drops, 499)
            self.assertEqual(len(read_batch.targets), 2)


if __name__ == "__main__":
    unittest.main()
