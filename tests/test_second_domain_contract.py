"""Unit and contract tests for second-domain perception source and zero circle assumption (Milestone M8-02)."""

from pathlib import Path
import unittest
import yaml

from tools.bus.ring import (
    ChannelId,
    SharedMemoryRingInspector,
)
from tools.bus.schema_bindings import (
    BBoxf,
    CorrelationFlags,
    CorrelationHeader,
    Covariance2f,
    TargetObservationBatchData,
    TargetObservationData,
    Vec2f,
    Visibility,
    deserialize_target_observation_batch,
    serialize_target_observation_batch,
)
from tools.bus.shm_observation_publisher import (
    SharedMemoryObservationPublisher,
)


class TestSecondDomainContract(unittest.TestCase):
    """Verifies second-domain perception contracts and non-circular target observations."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_second_domain_scenario_config(self) -> None:
        scenario_path = self.repo_root / "configs" / "scenario" / "second_domain.yaml"
        self.assertTrue(scenario_path.exists(), f"Missing {scenario_path}")

        with open(scenario_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["schema_version"], 1)
        self.assertEqual(cfg["profile_id"], "tactical_humanoid_sim")
        self.assertEqual(cfg["foreground"]["process_name"], "tactical_sim.exe")
        self.assertEqual(cfg["foreground"]["window_title_substring"], "Tactical Domain")
        self.assertTrue(cfg["capabilities"]["provides_foreground_identity"])
        self.assertEqual(cfg["calibration_seed"]["counts_per_pixel_x"], 1.50)
        self.assertEqual(cfg["calibration_seed"]["counts_per_pixel_y"], 1.50)
        self.assertEqual(cfg["calibration_seed"]["fov_horizontal_deg"], 90.0)

    def test_second_domain_config_invariance(self) -> None:
        domain_sec_path = self.repo_root / "configs" / "domain" / "second_domain.yaml"
        domain_aim_path = self.repo_root / "configs" / "domain" / "aimlabs.yaml"
        self.assertTrue(domain_sec_path.exists())
        self.assertTrue(domain_aim_path.exists())

        with open(domain_sec_path, "r", encoding="utf-8") as f:
            cfg_sec = yaml.safe_load(f)
        with open(domain_aim_path, "r", encoding="utf-8") as f:
            cfg_aim = yaml.safe_load(f)

        # Policy plugin and core objective must remain completely invariant
        self.assertEqual(cfg_sec["policy"]["plugin"], cfg_aim["policy"]["plugin"])
        self.assertEqual(cfg_sec["policy"]["objective"], cfg_aim["policy"]["objective"])

    def test_humanoid_batch_flatbuffers_roundtrip(self) -> None:
        cid = CorrelationHeader(
            sequence_id=42,
            source_timestamp_ns=1_000_000_000,
            pipeline_run_id=888,
            flags=CorrelationFlags.NONE,
        )

        # Humanoid target: w=24, h=72 (1:3 aspect ratio), anisotropic covariance xx=4.0, yy=36.0
        humanoid_obs = TargetObservationData(
            source_id=1,
            frame_id=42,
            captured_at_ns=1_000_000_000,
            center_px=Vec2f(960.0, 540.0),
            center_norm=Vec2f(0.0, 0.0),
            bbox_px=BBoxf(948.0, 504.0, 972.0, 576.0),
            effective_radius_px=12.0,
            confidence=0.96,
            covariance_px2=Covariance2f(4.0, 0.0, 36.0),
            velocity_px_per_s=Vec2f(120.0, 0.0),
            velocity_confidence=0.85,
            visibility=Visibility.VISIBLE,
            target_value=1.0,
            semantic_id=2,  # Tactical / Humanoid target
        )

        batch_out = TargetObservationBatchData(
            header=cid,
            source_id=1,
            frame_id=42,
            captured_at_ns=1_000_000_000,
            published_at_ns=1_000_000_000,
            source_width=1920,
            source_height=1080,
            targets=[humanoid_obs],
            schema_major=1,
            schema_minor=0,
        )

        wire_bytes = serialize_target_observation_batch(batch_out)
        self.assertTrue(len(wire_bytes) > 0)

        # Deserialize and verify exact match
        batch_in = deserialize_target_observation_batch(wire_bytes)
        self.assertEqual(batch_in.schema_major, 1)
        self.assertEqual(batch_in.header.sequence_id, 42)
        self.assertEqual(len(batch_in.targets), 1)

        t_in = batch_in.targets[0]
        self.assertEqual(t_in.source_id, 1)
        self.assertEqual(t_in.effective_radius_px, 12.0)
        # Verify tall bounding box
        box_w = t_in.bbox_px.right - t_in.bbox_px.left
        box_h = t_in.bbox_px.bottom - t_in.bbox_px.top
        self.assertAlmostEqual(box_w, 24.0, places=3)
        self.assertAlmostEqual(box_h, 72.0, places=3)
        # Verify anisotropic covariance
        self.assertGreater(t_in.covariance_px2.yy, t_in.covariance_px2.xx * 8.0)
        self.assertEqual(t_in.semantic_id, 2)

    def test_shm_publisher_humanoid_stream(self) -> None:
        shm_name = "aim_test_py_pub_humanoid"
        run_id = 999

        with SharedMemoryObservationPublisher(
            shm_name=shm_name, pipeline_run_id=run_id, create=True
        ) as pub:
            obs = TargetObservationData(
                source_id=5,
                frame_id=1,
                captured_at_ns=1_000_000_000,
                center_px=Vec2f(960.0, 540.0),
                center_norm=Vec2f(0.0, 0.0),
                bbox_px=BBoxf(948.0, 504.0, 972.0, 576.0),
                effective_radius_px=12.0,
                confidence=0.98,
                covariance_px2=Covariance2f(4.0, 0.0, 16.0),
                velocity_px_per_s=Vec2f(0.0, 0.0),
                velocity_confidence=0.0,
                visibility=Visibility.VISIBLE,
                target_value=1.0,
                semantic_id=2,
            )

            cid = CorrelationHeader(
                sequence_id=1,
                source_timestamp_ns=1_000_000_000,
                pipeline_run_id=run_id,
                flags=CorrelationFlags.NONE,
            )
            batch = TargetObservationBatchData(
                header=cid,
                source_id=1,
                frame_id=1,
                captured_at_ns=1_000_000_000,
                published_at_ns=1_000_000_000,
                source_width=1920,
                source_height=1080,
                targets=[obs],
                schema_major=1,
                schema_minor=0,
            )

            written_seq = pub.write_observation_batch(batch)
            self.assertEqual(written_seq, 1)

            # Inspect memory
            hdr = SharedMemoryRingInspector.parse_control_header(bytes(pub._shm[:128]))
            self.assertEqual(hdr.producer_head_seq, 1)
            self.assertEqual(hdr.channel_id, int(ChannelId.TARGET_OBSERVATION))


if __name__ == "__main__":
    unittest.main()
