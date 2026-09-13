"""Unit and adversarial tests for recording indexer, schema validation, and corruption detection."""

import json
from pathlib import Path
import pytest
import jsonschema

from tools.data.recording_index import (
    CorrelationIdInfo,
    FrameEntry,
    RecordingIndexer,
    RecordingSession,
    SessionStatistics,
    SourceInfo,
    StorageInfo,
    get_schema_path,
)


@pytest.fixture
def temp_recording_dir(tmp_path: Path) -> Path:
    """Create a temporary recording directory with simulated binary frame files."""
    rec_dir = tmp_path / "session_rec_001"
    rec_dir.mkdir()

    for i in range(1, 11):
        frame_file = rec_dir / f"frame_{i:04d}.bin"
        # Write deterministic non-empty binary payload
        frame_file.write_bytes(bytes([(i * 17 + j) % 256 for j in range(256)]))

    return rec_dir


def test_schema_file_exists() -> None:
    """Verify that recording_session.schema.json exists and is valid JSON."""
    schema_path = get_schema_path()
    assert schema_path.exists(), f"Schema file not found: {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert schema.get("title") == "AimAgentRecordingSessionManifest"


def test_recording_session_dataclass_roundtrip(tmp_path: Path) -> None:
    """Test creating, validating, saving, and loading a RecordingSession."""
    session = RecordingSession(
        schema_version=1,
        session_id="11111111-2222-4333-8444-555555555555",
        scenario_name="gridshot_100",
        task_type="flick",
        visual_profile="default",
        notes="Calibration run",
        source=SourceInfo(
            resolution_width=1920,
            resolution_height=1080,
            target_fps=144.0,
            color_format="b8g8r8a8_unorm",
            adapter_luid="0x00000000000184A2",
            capture_backend="dxgi_duplication",
        ),
        storage=StorageInfo(
            recording_dir="session_001",
            format="raw_binary",
            data_retention_days=30,
        ),
        statistics=SessionStatistics(
            start_timestamp_utc="2026-08-30T10:00:00Z",
            end_timestamp_utc="2026-08-30T10:01:00Z",
            start_time_ns=1_000_000_000,
            end_time_ns=61_000_000_000,
            duration_ms=60000.0,
            total_frames_captured=8640,
            total_frames_recorded=8640,
            total_frames_dropped=0,
            total_bytes=8640 * 1024,
        ),
        frames=[
            FrameEntry(
                sample_id="11111111-2222-4333-8444-555555555555_1",
                frame_sequence_id=1,
                timestamp_ns=1_000_000_000,
                relative_path="frame_0001.bin",
                sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                width=1920,
                height=1080,
                correlation_id=CorrelationIdInfo(1, 1_000_000_000, 1, 0),
                is_annotated=False,
            )
        ],
    )

    # Validate against JSON schema
    session.validate_schema()

    manifest_path = tmp_path / "session_manifest.json"
    session.save_json(manifest_path)
    assert manifest_path.exists()

    loaded = RecordingSession.load_json(manifest_path)
    assert loaded.session_id == session.session_id
    assert loaded.scenario_name == session.scenario_name
    assert len(loaded.frames) == 1
    assert loaded.frames[0].sha256 == session.frames[0].sha256


def test_recording_indexer_index_directory(temp_recording_dir: Path, tmp_path: Path) -> None:
    """Test indexing a directory of simulated raw frame files."""
    session = RecordingIndexer.index_directory(
        directory=temp_recording_dir,
        session_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        scenario_name="spidershot",
        task_type="flick",
        visual_profile="high_contrast",
        target_fps=144.0,
    )

    assert session.session_id == "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    assert session.scenario_name == "spidershot"
    assert session.task_type == "flick"
    assert session.visual_profile == "high_contrast"
    assert len(session.frames) == 10
    assert session.statistics.total_frames_recorded == 10
    assert session.statistics.total_bytes == 10 * 256

    # Verify each frame has a valid SHA-256
    for f in session.frames:
        assert len(f.sha256) == 64
        assert f.sample_id.startswith(session.session_id)

    # Save and validate
    out_json = tmp_path / "indexed_session.json"
    session.save_json(out_json)

    report = RecordingIndexer.validate_session(session, base_dir=temp_recording_dir)
    assert report.is_valid
    assert len(report.errors) == 0
    assert report.verified_files == 10
    assert report.missing_files == 0
    assert report.corrupted_files == 0


def test_recording_indexer_detects_corrupted_hash(temp_recording_dir: Path) -> None:
    """Verify that validator catches files modified on disk."""
    session = RecordingIndexer.index_directory(
        directory=temp_recording_dir,
        session_id="22222222-3333-4444-8555-666666666666",
    )

    # Corrupt one file on disk
    corrupted_file = temp_recording_dir / session.frames[0].relative_path
    corrupted_file.write_bytes(b"CORRUPTED_MODIFIED_CONTENT")

    report = RecordingIndexer.validate_session(session, base_dir=temp_recording_dir)
    assert not report.is_valid
    assert report.corrupted_files == 1
    assert any("Corrupted frame hash" in err for err in report.errors)


def test_recording_indexer_detects_missing_file(temp_recording_dir: Path) -> None:
    """Verify that validator catches deleted or missing files."""
    session = RecordingIndexer.index_directory(
        directory=temp_recording_dir,
        session_id="33333333-4444-4555-8666-777777777777",
    )

    # Delete one file
    missing_file = temp_recording_dir / session.frames[2].relative_path
    missing_file.unlink()

    report = RecordingIndexer.validate_session(session, base_dir=temp_recording_dir)
    assert not report.is_valid
    assert report.missing_files == 1
    assert any("Missing frame file on disk" in err for err in report.errors)


def test_recording_indexer_detects_duplicate_sequence_and_non_monotonic_timestamps() -> None:
    """Adversarial test: detect duplicate sequence IDs and non-monotonic timestamps."""
    session = RecordingSession(
        session_id="44444444-5555-4666-8777-888888888888",
        frames=[
            FrameEntry(
                sample_id="44444444-5555-4666-8777-888888888888_1",
                frame_sequence_id=1,
                timestamp_ns=2_000_000_000,
                relative_path="frame_01.bin",
                sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                width=1920,
                height=1080,
            ),
            FrameEntry(
                sample_id="44444444-5555-4666-8777-888888888888_2",
                frame_sequence_id=1,  # Duplicate sequence ID!
                timestamp_ns=1_000_000_000,  # Non-monotonic (time went backwards)!
                relative_path="frame_02.bin",
                sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                width=1920,
                height=1080,
            ),
        ],
    )

    report = RecordingIndexer.validate_session(session, verify_hashes=False)
    assert not report.is_valid
    assert report.duplicate_sequences == 1
    assert any("Duplicate frame sequence ID detected: 1" in err for err in report.errors)
    assert any("Non-monotonic timestamp detected" in err for err in report.errors)


def test_schema_rejects_unknown_task_type() -> None:
    """Verify that schema rejects unknown task types."""
    session_data = {
        "schema_version": 1,
        "session_id": "55555555-6666-4777-8888-999999999999",
        "scenario_name": "gridshot",
        "task_type": "invalid_unsupported_task",
        "visual_profile": "default",
        "source": {
            "resolution_width": 1920,
            "resolution_height": 1080,
            "target_fps": 144.0,
            "color_format": "b8g8r8a8_unorm",
            "adapter_luid": "0x0000000000000000",
            "capture_backend": "dxgi_duplication",
        },
        "storage": {
            "recording_dir": "rec",
            "format": "raw_binary",
            "data_retention_days": 30,
        },
        "statistics": {
            "start_timestamp_utc": "2026-08-30T10:00:00Z",
            "end_timestamp_utc": "2026-08-30T10:00:01Z",
            "start_time_ns": 0,
            "end_time_ns": 1000,
            "duration_ms": 1.0,
            "total_frames_captured": 0,
            "total_frames_recorded": 0,
            "total_frames_dropped": 0,
            "total_bytes": 0,
        },
        "frames": [],
    }

    schema_file = get_schema_path()
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=session_data, schema=schema)
