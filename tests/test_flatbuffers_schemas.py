"""Unit tests for FlatBuffers v1 bus schemas, binary magic tokens, and serialization."""

import os
import unittest
from tools.bus.schema_bindings import (
    ActuationCommandData,
    ButtonAction,
    ButtonTransitionData,
    CorrelationFlags,
    CorrelationHeader,
    Covariance2f,
    SchemaValidationError,
    FrameDescriptorData,
    FramePixelFormat,
    MouseButton,
    TargetObservationBatchData,
    TargetObservationData,
    Vec2f,
    BBoxf,
    Visibility,
    deserialize_actuation_command,
    deserialize_frame_descriptor,
    deserialize_target_observation_batch,
    serialize_actuation_command,
    serialize_frame_descriptor,
    serialize_target_observation_batch,
    validate_file_identifier,
    TARGET_OBSERVATION_BATCH_IDENTIFIER,
)


class TestFlatBuffersSchemas(unittest.TestCase):
    """Test suite verifying v1 FlatBuffers schemas, file identifiers, and coordinate semantics."""

    def setUp(self) -> None:
        self.cid = CorrelationHeader(
            sequence_id=42,
            source_timestamp_ns=1_000_000_000,
            pipeline_run_id=1,
            flags=CorrelationFlags.SYNTHETIC,
        )

    def test_schema_files_exist(self) -> None:
        """Verify all 7 canonical FlatBuffers schema files exist under schemas/bus/."""
        expected_schemas = [
            "common.fbs",
            "frame_descriptor.fbs",
            "target_observation.fbs",
            "tracked_target.fbs",
            "aim_intent.fbs",
            "actuation_command.fbs",
            "telemetry_event.fbs",
        ]
        base_dir = os.path.join(os.path.dirname(__file__), "..", "schemas", "bus")
        for s in expected_schemas:
            path = os.path.join(base_dir, s)
            self.assertTrue(os.path.isfile(path), f"Missing schema file: {s}")

    def test_released_v1_snapshot_exists(self) -> None:
        """Verify released v1 baseline exists for backward-conformance verification."""
        rel_dir = os.path.join(os.path.dirname(__file__), "..", "schemas", "bus", "released", "v1")
        self.assertTrue(os.path.isdir(rel_dir), "Missing released/v1 directory")
        self.assertTrue(os.path.isfile(os.path.join(rel_dir, "common.fbs")))
        self.assertTrue(os.path.isfile(os.path.join(rel_dir, "target_observation.fbs")))

    def test_frame_descriptor_serialization_and_magic(self) -> None:
        """Verify FrameDescriptor serializes with magic identifier 'AFR1'."""
        frame_data = FrameDescriptorData(
            header=self.cid,
            frame_id=101,
            captured_at_ns=1_000_000_000,
            width=1920,
            height=1080,
            format=FramePixelFormat.B8G8R8A8_UNORM,
            pool_slot_index=3,
            shared_nt_handle=0x12345678,
            adapter_luid=0x10688,
            is_keyframe=False,
        )
        buf = serialize_frame_descriptor(frame_data)

        self.assertGreater(len(buf), 16)
        self.assertTrue(validate_file_identifier(buf, b"AFR1"))
        self.assertEqual(buf[4:8], b"AFR1")
        decoded = deserialize_frame_descriptor(buf)
        self.assertEqual(decoded.header, self.cid)
        self.assertEqual(decoded.frame_id, 101)
        self.assertEqual(decoded.pool_slot_index, 3)
        self.assertEqual(decoded.shared_nt_handle, 0x12345678)
        self.assertEqual(decoded.adapter_luid, 0x10688)
        self.assertFalse(decoded.is_keyframe)

    def test_actuation_command_serialization_and_magic(self) -> None:
        """Verify ActuationCommand serializes with magic identifier 'AAC1'."""
        cmd_data = ActuationCommandData(
            header=self.cid,
            generated_at_ns=1_000_500_000,
            desired_apply_time_ns=1_001_000_000,
            delta_x_counts=25,
            delta_y_counts=-14,
            button_transition=ButtonTransitionData(
                button=MouseButton.LEFT,
                action=ButtonAction.PRESS,
            ),
            cancel_superseded=False,
        )
        buf = serialize_actuation_command(cmd_data)

        self.assertGreater(len(buf), 16)
        self.assertTrue(validate_file_identifier(buf, b"AAC1"))
        self.assertEqual(buf[4:8], b"AAC1")
        decoded = deserialize_actuation_command(buf)
        self.assertEqual(decoded.header, self.cid)
        self.assertEqual(decoded.delta_x_counts, 25)
        self.assertEqual(decoded.delta_y_counts, -14)
        self.assertEqual(decoded.button_transition.button, MouseButton.LEFT)
        self.assertEqual(decoded.button_transition.action, ButtonAction.PRESS)
        self.assertFalse(decoded.cancel_superseded)

    def test_identifier_version_and_malformed_buffers_fail_closed(self) -> None:
        frame = FrameDescriptorData(
            header=self.cid,
            frame_id=101,
            captured_at_ns=1_000_000_000,
            schema_major=2,
        )
        unsupported = serialize_frame_descriptor(frame)
        with self.assertRaises(SchemaValidationError):
            deserialize_frame_descriptor(unsupported)

        corrupt_identifier = bytearray(unsupported)
        corrupt_identifier[4:8] = b"BAD!"
        with self.assertRaises(SchemaValidationError):
            deserialize_frame_descriptor(bytes(corrupt_identifier))

        with self.assertRaises(SchemaValidationError):
            deserialize_frame_descriptor(b"\x08\x00\x00\x00AFR1")

        self.assertFalse(validate_file_identifier(unsupported, b"TOO-LONG"))

    def test_coordinate_transforms_and_normalization(self) -> None:
        """Verify pixel to normalized [-1, 1] coordinate domain conversion."""
        width = 1920
        height = 1080

        def to_norm(px_x: float, px_y: float) -> Vec2f:
            norm_x = (px_x - (width / 2.0)) / (width / 2.0)
            norm_y = (px_y - (height / 2.0)) / (height / 2.0)
            return Vec2f(norm_x, norm_y)

        # Center of screen -> (0, 0)
        c = to_norm(960.0, 540.0)
        self.assertAlmostEqual(c.x, 0.0, places=5)
        self.assertAlmostEqual(c.y, 0.0, places=5)

        # Top-Left -> (-1, -1)
        tl = to_norm(0.0, 0.0)
        self.assertAlmostEqual(tl.x, -1.0, places=5)
        self.assertAlmostEqual(tl.y, -1.0, places=5)

        # Bottom-Right -> (1, 1)
        br = to_norm(1920.0, 1080.0)
        self.assertAlmostEqual(br.x, 1.0, places=5)
        self.assertAlmostEqual(br.y, 1.0, places=5)

    def test_target_observation_batch_data_structures(self) -> None:
        """Verify TargetObservationBatch structure and fields."""
        obs = TargetObservationData(
            source_id=1,
            frame_id=101,
            captured_at_ns=1_000_000_000,
            center_px=Vec2f(960.0, 540.0),
            center_norm=Vec2f(0.0, 0.0),
            bbox_px=BBoxf(940.0, 520.0, 980.0, 560.0),
            effective_radius_px=20.0,
            confidence=0.985,
            covariance_px2=Covariance2f(1.0, 0.0, 1.0),
            velocity_px_per_s=Vec2f(100.0, -50.0),
            visibility=Visibility.VISIBLE,
            target_value=1.0,
            semantic_id=1,
        )

        batch = TargetObservationBatchData(
            header=self.cid,
            source_id=1,
            frame_id=101,
            captured_at_ns=1_000_000_000,
            published_at_ns=1_002_000_000,
            source_width=1920,
            source_height=1080,
            targets=[obs],
        )

        self.assertEqual(batch.schema_major, 1)
        self.assertEqual(batch.schema_minor, 0)
        self.assertEqual(len(batch.targets), 1)
        self.assertEqual(batch.targets[0].confidence, 0.985)
        self.assertEqual(batch.targets[0].center_px.x, 960.0)

        # Test FlatBuffers binary serialization
        buf = serialize_target_observation_batch(batch)
        self.assertGreater(len(buf), 32)
        self.assertTrue(validate_file_identifier(buf, TARGET_OBSERVATION_BATCH_IDENTIFIER))
        self.assertEqual(buf[4:8], b"AOB1")

        # Test FlatBuffers binary deserialization and round-trip parity
        decoded = deserialize_target_observation_batch(buf)
        self.assertEqual(decoded.schema_major, 1)
        self.assertEqual(decoded.schema_minor, 0)
        self.assertEqual(decoded.header, self.cid)
        self.assertEqual(decoded.source_id, 1)
        self.assertEqual(decoded.frame_id, 101)
        self.assertEqual(decoded.captured_at_ns, 1_000_000_000)
        self.assertEqual(decoded.published_at_ns, 1_002_000_000)
        self.assertEqual(decoded.source_width, 1920)
        self.assertEqual(decoded.source_height, 1080)
        self.assertEqual(len(decoded.targets), 1)

        d_obs = decoded.targets[0]
        self.assertEqual(d_obs.source_id, 1)
        self.assertEqual(d_obs.frame_id, 101)
        self.assertEqual(d_obs.captured_at_ns, 1_000_000_000)
        self.assertAlmostEqual(d_obs.center_px.x, 960.0, places=5)
        self.assertAlmostEqual(d_obs.center_px.y, 540.0, places=5)
        self.assertAlmostEqual(d_obs.center_norm.x, 0.0, places=5)
        self.assertAlmostEqual(d_obs.center_norm.y, 0.0, places=5)
        self.assertAlmostEqual(d_obs.bbox_px.left, 940.0, places=5)
        self.assertAlmostEqual(d_obs.bbox_px.right, 980.0, places=5)
        self.assertAlmostEqual(d_obs.effective_radius_px, 20.0, places=5)
        self.assertAlmostEqual(d_obs.confidence, 0.985, places=5)
        self.assertAlmostEqual(d_obs.covariance_px2.xx, 1.0, places=5)
        self.assertAlmostEqual(d_obs.velocity_px_per_s.x, 100.0, places=5)
        self.assertEqual(d_obs.visibility, Visibility.VISIBLE)
        self.assertAlmostEqual(d_obs.target_value, 1.0, places=5)
        self.assertEqual(d_obs.semantic_id, 1)



if __name__ == "__main__":
    unittest.main()
