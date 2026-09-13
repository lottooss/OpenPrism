"""Unit and adversarial tests for dataset splitter, zero session leakage, and slice distribution auditing."""

import pytest

from tools.data.dataset_splitter import DatasetSplitter, DatasetSplitterConfig
from tools.data.recording_index import (
    CorrelationIdInfo,
    FrameEntry,
    RecordingSession,
    SessionStatistics,
    SourceInfo,
    StorageInfo,
)


@pytest.fixture
def mixed_dataset_sessions() -> list[RecordingSession]:
    """Create a diverse mixture of 8 synthetic and 4 real sessions across various tasks."""
    sessions = []

    # 4 Real sessions
    for i in range(1, 5):
        sess_id = f"00000000-1111-4222-8333-{i:012d}"
        frames = [
            FrameEntry(
                sample_id=f"{sess_id}_{f}",
                frame_sequence_id=f,
                timestamp_ns=1_000_000_000 + f * 6_944_444,
                relative_path=f"real_{i:02d}/frame_{f:04d}.png",
                sha256=f"a{i:02d}{f:04d}".ljust(64, "0"),
                width=1920,
                height=1080,
                correlation_id=CorrelationIdInfo(f, 1_000_000_000 + f * 6_944_444, 1, 0),
                label_path=f"real_{i:02d}/frame_{f:04d}.txt",
                label_sha256="b".ljust(64, "0"),
                is_annotated=True,
            )
            for f in range(1, 6)  # 5 frames each
        ]
        sess = RecordingSession(
            schema_version=1,
            session_id=sess_id,
            scenario_name=f"real_task_{i}",
            task_type="flick" if i <= 2 else "tracking",
            visual_profile="default" if i % 2 == 0 else "high_contrast",
            source=SourceInfo(1920, 1080, 144.0),
            storage=StorageInfo(recording_dir=f"real_{i:02d}"),
            statistics=SessionStatistics(
                start_timestamp_utc="2026-08-30T10:00:00Z",
                end_timestamp_utc="2026-08-30T10:01:00Z",
                start_time_ns=1_000_000_000,
                end_time_ns=1_000_000_000 + 5 * 6_944_444,
                duration_ms=34.72,
                total_frames_captured=5,
                total_frames_recorded=5,
                total_frames_dropped=0,
                total_bytes=5 * 1024,
            ),
            frames=frames,
        )
        sessions.append(sess)

    # 8 Synthetic sessions
    for i in range(5, 13):
        sess_id = f"00000000-2222-4333-8444-{i:012d}"
        frames = [
            FrameEntry(
                sample_id=f"{sess_id}_{f}",
                frame_sequence_id=f,
                timestamp_ns=2_000_000_000 + f * 6_944_444,
                relative_path=f"synth_{i:02d}/frame_{f:04d}.png",
                sha256=f"c{i:02d}{f:04d}".ljust(64, "0"),
                width=1920,
                height=1080,
                correlation_id=CorrelationIdInfo(f, 2_000_000_000 + f * 6_944_444, 2, 0),
                label_path=f"synth_{i:02d}/frame_{f:04d}.txt",
                label_sha256="d".ljust(64, "0"),
                is_annotated=True,
            )
            for f in range(1, 11)  # 10 frames each
        ]
        sess = RecordingSession(
            schema_version=1,
            session_id=sess_id,
            scenario_name="synthetic_grid",
            task_type="synthetic",
            visual_profile="aimlabs_grid" if i % 2 == 0 else "dark_mode",
            source=SourceInfo(1920, 1080, 144.0),
            storage=StorageInfo(recording_dir=f"synth_{i:02d}"),
            statistics=SessionStatistics(
                start_timestamp_utc="2026-08-30T10:00:00Z",
                end_timestamp_utc="2026-08-30T10:01:00Z",
                start_time_ns=2_000_000_000,
                end_time_ns=2_000_000_000 + 10 * 6_944_444,
                duration_ms=69.44,
                total_frames_captured=10,
                total_frames_recorded=10,
                total_frames_dropped=0,
                total_bytes=10 * 1024,
            ),
            frames=frames,
        )
        sessions.append(sess)

    return sessions


def test_dataset_splitter_zero_session_leakage(mixed_dataset_sessions: list[RecordingSession]) -> None:
    """CRITICAL TEST: Verify 100% session isolation with zero cross-split leakage."""
    cfg = DatasetSplitterConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        holdout_real_for_test=True,
        random_seed=42,
    )
    splitter = DatasetSplitter(cfg)
    manifest, report = splitter.split_sessions(mixed_dataset_sessions)

    assert report.zero_leakage_verified
    assert report.real_holdout_verified
    assert report.total_samples == 100  # 4 * 5 real + 8 * 10 synthetic = 100 samples
    assert report.train_samples + report.val_samples + report.test_samples == 100

    # Ensure no single session ID appears in multiple splits
    session_splits: dict[str, set[str]] = {}
    for sample in manifest.samples:
        session_splits.setdefault(sample.session_id, set()).add(sample.split)

    for sid, splits in session_splits.items():
        assert len(splits) == 1, f"Session {sid} was leaked across splits: {splits}"

    # Verify real held-out test distribution has real frames
    assert report.test_slices.domain_real_count > 0 or report.val_slices.domain_real_count > 0


def test_dataset_splitter_deterministic_hashing(mixed_dataset_sessions: list[RecordingSession]) -> None:
    """Verify that identical random seeds produce identical split allocations and hashes."""
    cfg1 = DatasetSplitterConfig(random_seed=12345)
    cfg2 = DatasetSplitterConfig(random_seed=12345)
    cfg3 = DatasetSplitterConfig(random_seed=99999)

    splitter1 = DatasetSplitter(cfg1)
    splitter2 = DatasetSplitter(cfg2)
    splitter3 = DatasetSplitter(cfg3)

    m1, r1 = splitter1.split_sessions(mixed_dataset_sessions)
    m2, r2 = splitter2.split_sessions(mixed_dataset_sessions)
    m3, r3 = splitter3.split_sessions(mixed_dataset_sessions)

    assert r1.split_hash == r2.split_hash
    assert [s.split for s in m1.samples] == [s.split for s in m2.samples]
    assert r1.split_hash != r3.split_hash


def test_dataset_splitter_empty_sessions_rejected() -> None:
    """Verify that attempting to split an empty session list raises ValueError."""
    splitter = DatasetSplitter()
    with pytest.raises(ValueError, match="Cannot split empty session list"):
        splitter.split_sessions([])
