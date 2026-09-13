"""Unit and adversarial tests for dataset manifest creation, session-aware split allocation, and git cleanliness."""

import json
from pathlib import Path
import pytest

from tools.data.dataset_manifest import (
    DatasetManifest,
    DatasetManifestBuilder,
    DatasetSample,
    DatasetStatistics,
    SplitPolicy,
    get_dataset_schema_path,
)
from tools.data.recording_index import (
    CorrelationIdInfo,
    FrameEntry,
    RecordingSession,
    SessionStatistics,
    SourceInfo,
    StorageInfo,
)


@pytest.fixture
def sample_sessions() -> list[RecordingSession]:
    """Create 6 distinct synthetic recording sessions across flick, tracking, and switching tasks."""
    sessions = []
    task_types = ["flick", "flick", "tracking", "tracking", "switching", "switching"]
    profiles = ["default", "high_contrast", "dark", "default", "high_contrast", "custom"]

    for i in range(1, 7):
        sess_id = f"11111111-2222-4333-8444-{i:012d}"
        frames = [
            FrameEntry(
                sample_id=f"{sess_id}_{f}",
                frame_sequence_id=f,
                timestamp_ns=1_000_000_000 + f * 6_944_444,
                relative_path=f"session_{i:02d}/frame_{f:04d}.png",
                sha256=f"a{i:02d}{f:04d}".ljust(64, "0"),
                width=1920,
                height=1080,
                correlation_id=CorrelationIdInfo(f, 1_000_000_000 + f * 6_944_444, 1, 0),
                label_path=f"session_{i:02d}/frame_{f:04d}.txt",
                label_sha256="b".ljust(64, "0"),
                is_annotated=True,
            )
            for f in range(1, 11)  # 10 frames per session
        ]
        sess = RecordingSession(
            schema_version=1,
            session_id=sess_id,
            scenario_name=f"task_{task_types[i-1]}",
            task_type=task_types[i-1],
            visual_profile=profiles[i-1],
            source=SourceInfo(1920, 1080, 144.0),
            storage=StorageInfo(recording_dir=f"session_{i:02d}"),
            statistics=SessionStatistics(
                start_timestamp_utc="2026-08-30T10:00:00Z",
                end_timestamp_utc="2026-08-30T10:01:00Z",
                start_time_ns=1_000_000_000,
                end_time_ns=1_000_000_000 + 10 * 6_944_444,
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


def test_dataset_schema_file_exists() -> None:
    """Verify that dataset_manifest.schema.json exists and is valid JSON."""
    schema_path = get_dataset_schema_path()
    assert schema_path.exists(), f"Dataset schema not found: {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert schema.get("title") == "AimAgentDatasetManifest"


def test_dataset_manifest_roundtrip(tmp_path: Path) -> None:
    """Test creating, validating, saving, and loading a DatasetManifest."""
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id="99999999-8888-4777-8666-555555555555",
        dataset_name="aim_reference_benchmarks",
        version="1.0.0",
        created_at_utc="2026-08-30T10:00:00Z",
        description="Benchmark target dataset",
        license="MIT",
        classes=["target_sphere"],
        split_policy=SplitPolicy(
            strategy="session_aware",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        ),
        source_sessions=["11111111-2222-4333-8444-000000000001"],
        statistics=DatasetStatistics(
            total_samples=1,
            train_count=1,
            val_count=0,
            test_count=0,
            total_annotations=1,
            class_distribution={"target_sphere": 1},
            dataset_sha256="c000000000000000000000000000000000000000000000000000000000000000",
        ),
        samples=[
            DatasetSample(
                sample_id="11111111-2222-4333-8444-000000000001_1",
                session_id="11111111-2222-4333-8444-000000000001",
                task_type="flick",
                visual_profile="default",
                split="train",
                image_path="session_01/frame_0001.png",
                image_sha256="a000000000000000000000000000000000000000000000000000000000000000",
                label_path="session_01/frame_0001.txt",
                label_sha256="b000000000000000000000000000000000000000000000000000000000000000",
                annotation_count=1,
                width=1920,
                height=1080,
            )
        ],
    )

    manifest.validate_schema()

    out_file = tmp_path / "dataset_manifest.json"
    manifest.save_json(out_file)
    assert out_file.exists()

    loaded = DatasetManifest.load_json(out_file)
    assert loaded.dataset_id == manifest.dataset_id
    assert loaded.statistics.total_samples == 1
    assert loaded.classes == ["target_sphere"]


def test_session_aware_split_no_leakage(sample_sessions: list[RecordingSession]) -> None:
    """CRITICAL TEST: Ensure zero cross-split leakage by assigning whole sessions to splits."""
    policy = SplitPolicy(
        strategy="session_aware",
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        random_seed=1337,
    )

    manifest = DatasetManifestBuilder.build_from_sessions(
        sessions=sample_sessions,
        dataset_name="aim_leakage_free_targets",
        split_policy=policy,
    )

    assert manifest.statistics.total_samples == 60  # 6 sessions * 10 frames
    assert manifest.statistics.train_count > 0
    assert manifest.statistics.val_count > 0
    assert manifest.statistics.test_count > 0
    assert (
        manifest.statistics.train_count
        + manifest.statistics.val_count
        + manifest.statistics.test_count
        == 60
    )

    # Check that each session ID is strictly inside ONE split
    session_splits: dict[str, set[str]] = {}
    for sample in manifest.samples:
        session_splits.setdefault(sample.session_id, set()).add(sample.split)

    for sess_id, splits in session_splits.items():
        assert len(splits) == 1, f"Data leakage! Session {sess_id} spanned splits {splits}"

    report = DatasetManifestBuilder.validate_manifest(manifest)
    assert report.is_valid
    assert not report.session_leakage_detected
    assert len(report.errors) == 0


def test_deterministic_split_with_seed(sample_sessions: list[RecordingSession]) -> None:
    """Verify that identical random seeds produce bitwise identical split allocations."""
    policy1 = SplitPolicy(random_seed=42)
    policy2 = SplitPolicy(random_seed=42)

    manifest1 = DatasetManifestBuilder.build_from_sessions(
        sessions=sample_sessions, split_policy=policy1
    )
    manifest2 = DatasetManifestBuilder.build_from_sessions(
        sessions=sample_sessions, split_policy=policy2
    )

    splits1 = [s.split for s in manifest1.samples]
    splits2 = [s.split for s in manifest2.samples]
    assert splits1 == splits2


def test_invalid_split_ratios_rejected() -> None:
    """Verify that split ratios that do not sum to 1.0 are rejected."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        policy = SplitPolicy(train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)
        policy.validate()


def test_git_cleanliness_verification() -> None:
    """Verify that Git cleanliness audit rejects any tracked dataset/recording binaries."""
    manifest = DatasetManifest()
    is_clean, errors = DatasetManifestBuilder.verify_git_clean(manifest)
    assert is_clean, f"Git clean violation detected: {errors}"
    assert len(errors) == 0


def test_adversarial_session_leakage_detection(sample_sessions: list[RecordingSession]) -> None:
    """Adversarial test: verify that manually corrupted manifests with cross-split session leakage are detected and rejected."""
    policy = SplitPolicy()
    manifest = DatasetManifestBuilder.build_from_sessions(
        sessions=sample_sessions, split_policy=policy
    )

    # Manually inject data leakage: change the split of one sample belonging to session 1 to 'test'
    target_session = sample_sessions[0].session_id
    for s in manifest.samples:
        if s.session_id == target_session:
            s.split = "test"  # Original was train!
            break

    report = DatasetManifestBuilder.validate_manifest(manifest)
    assert not report.is_valid
    assert report.session_leakage_detected
    assert any("Adjacent-frame data leakage detected" in err for err in report.errors)
