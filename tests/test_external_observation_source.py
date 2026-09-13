"""Unit and integration tests for external observation source over shared memory (Milestone M8-01)."""

from pathlib import Path
import subprocess
import time
import unittest

from tools.bus.ring import (
    SHM_RING_MAGIC,
    SHM_VERSION_MAJOR,
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
)
from tools.bus.shm_observation_publisher import (
    SharedMemoryObservationPublisher,
)


class TestExternalObservationSource(unittest.TestCase):
    """Verifies SharedMemoryObservationPublisher and cross-language C++ ingestion."""

    def test_shm_publisher_lifecycle_and_seqlock(self) -> None:
        shm_name = "aim_test_py_pub_lifecycle"
        run_id = 777

        with SharedMemoryObservationPublisher(
            shm_name=shm_name, pipeline_run_id=run_id, create=True
        ) as pub:
            # Read header directly
            hdr = SharedMemoryRingInspector.parse_control_header(bytes(pub._shm[:128]))
            self.assertEqual(hdr.magic, SHM_RING_MAGIC)
            self.assertEqual(hdr.version_major, SHM_VERSION_MAJOR)
            self.assertEqual(hdr.channel_id, int(ChannelId.TARGET_OBSERVATION))
            self.assertEqual(hdr.pipeline_run_id, run_id)
            self.assertEqual(hdr.ring_capacity, 16)
            self.assertEqual(hdr.producer_head_seq, 0)

            # Publish a batch
            cid = CorrelationHeader(
                sequence_id=1,
                source_timestamp_ns=1_000_000,
                pipeline_run_id=run_id,
                flags=CorrelationFlags.NONE,
            )
            obs = TargetObservationData(
                source_id=1,
                frame_id=10,
                captured_at_ns=1_000_000,
                center_px=Vec2f(960.0, 540.0),
                center_norm=Vec2f(0.0, 0.0),
                bbox_px=BBoxf(940.0, 520.0, 980.0, 560.0),
                effective_radius_px=15.0,
                confidence=0.99,
                covariance_px2=Covariance2f(1.0, 0.0, 1.0),
                velocity_px_per_s=Vec2f(0.0, 0.0),
                velocity_confidence=0.0,
                visibility=Visibility.VISIBLE,
                target_value=1.0,
                semantic_id=0,
            )
            batch = TargetObservationBatchData(
                header=cid,
                source_id=1,
                frame_id=10,
                captured_at_ns=1_000_000,
                published_at_ns=1_001_000,
                source_width=1920,
                source_height=1080,
                targets=[obs],
            )

            seq = pub.write_observation_batch(batch, publish_time_ns=1_001_000)
            self.assertEqual(seq, 1)

            # Check updated control header
            hdr_after = SharedMemoryRingInspector.parse_control_header(bytes(pub._shm[:128]))
            self.assertEqual(hdr_after.producer_head_seq, 1)
            self.assertEqual(hdr_after.writer_heartbeat_ns, 1_001_000)

            # Check slot header
            slot_hdr = SharedMemoryRingInspector.parse_slot_header(bytes(pub._shm), offset=128)
            self.assertEqual(slot_hdr.seq_before, 2)
            self.assertEqual(slot_hdr.seq_after, 2)
            self.assertEqual(slot_hdr.magic_identifier, 0x31424F41)
            self.assertEqual(slot_hdr.published_at_ns, 1_001_000)

    def test_cross_language_cpp_consumer_ingestion(self) -> None:
        """Publishes observations from Python and verifies consumption via C++ binary."""
        shm_name = "aim_test_cross_lang_shm"
        run_id = 888

        repo_root = Path(__file__).resolve().parent.parent
        cpp_bin_release = (
            repo_root
            / "build"
            / "windows-msvc-release"
            / "aim_external_observation_source_tests.exe"
        )
        if not cpp_bin_release.is_file():
            cpp_bin_release = (
                repo_root
                / "build"
                / "windows-msvc-debug"
                / "aim_external_observation_source_tests.exe"
            )

        self.assertTrue(cpp_bin_release.is_file(), f"Binary not found: {cpp_bin_release}")

        with SharedMemoryObservationPublisher(
            shm_name=shm_name, pipeline_run_id=run_id, create=True
        ) as pub:
            now_ns = time.perf_counter_ns()
            pub.update_heartbeat(now_ns)

            cid = CorrelationHeader(
                sequence_id=42,
                source_timestamp_ns=now_ns,
                pipeline_run_id=run_id,
                flags=CorrelationFlags.NONE,
            )
            obs = TargetObservationData(
                source_id=1,
                frame_id=200,
                captured_at_ns=now_ns,
                center_px=Vec2f(800.0, 450.0),
                center_norm=Vec2f(-0.1667, -0.1667),
                bbox_px=BBoxf(780.0, 430.0, 820.0, 470.0),
                effective_radius_px=20.0,
                confidence=0.95,
                covariance_px2=Covariance2f(1.0, 0.0, 1.0),
                velocity_px_per_s=Vec2f(10.0, 0.0),
                velocity_confidence=0.8,
                visibility=Visibility.VISIBLE,
                target_value=1.0,
                semantic_id=1,
            )
            batch = TargetObservationBatchData(
                header=cid,
                source_id=1,
                frame_id=200,
                captured_at_ns=now_ns,
                published_at_ns=now_ns + 500_000,
                source_width=1920,
                source_height=1080,
                targets=[obs],
            )
            pub.write_observation_batch(batch, publish_time_ns=now_ns + 500_000)

            # Execute C++ verification
            res = subprocess.run(
                [str(cpp_bin_release), "--verify-shm", shm_name, "42"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                res.returncode,
                0,
                f"C++ verification failed:\nstdout: {res.stdout}\nstderr: {res.stderr}",
            )
            self.assertIn("Verified IPC segment", res.stdout)


if __name__ == "__main__":
    unittest.main()
