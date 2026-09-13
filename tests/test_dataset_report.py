"""Unit and golden tests for dataset validation report generation (Markdown & JSON)."""

import json
from pathlib import Path

from tools.data.dataset_report import generate_validation_report
from tools.data.dataset_splitter import DatasetSplitter, DatasetSplitterConfig
from tools.data.recording_index import (
    CorrelationIdInfo,
    FrameEntry,
    RecordingSession,
    SessionStatistics,
    SourceInfo,
    StorageInfo,
)


def test_generate_validation_report_end_to_end(tmp_path: Path) -> None:
    """Test generating full markdown and JSON reports for a partitioned dataset."""
    sessions = []
    for i in range(1, 4):
        sess_id = f"00000000-0000-4000-8000-{i:012d}"
        frames = [
            FrameEntry(
                sample_id=f"{sess_id}_{f}",
                frame_sequence_id=f,
                timestamp_ns=1_000_000_000 + f * 6_944_444,
                relative_path=f"sess_{i}/f_{f}.png",
                sha256="a" * 64,
                width=1920,
                height=1080,
                correlation_id=CorrelationIdInfo(f, 1_000_000_000, 1, 0),
                label_path=f"sess_{i}/f_{f}.txt",
                label_sha256="b" * 64,
                is_annotated=True,
            )
            for f in range(1, 4)
        ]
        sess = RecordingSession(
            schema_version=1,
            session_id=sess_id,
            scenario_name=f"task_{i}",
            task_type="flick" if i == 1 else "synthetic",
            visual_profile="aimlabs_grid",
            source=SourceInfo(1920, 1080, 144.0),
            storage=StorageInfo(recording_dir=f"sess_{i}"),
            statistics=SessionStatistics(
                start_timestamp_utc="2026-08-30T10:00:00Z",
                end_timestamp_utc="2026-08-30T10:01:00Z",
                start_time_ns=1_000_000_000,
                end_time_ns=1_000_000_000 + 3 * 6_944_444,
                duration_ms=20.83,
                total_frames_captured=3,
                total_frames_recorded=3,
                total_frames_dropped=0,
                total_bytes=3 * 1024,
            ),
            frames=frames,
        )
        sessions.append(sess)

    splitter = DatasetSplitter(DatasetSplitterConfig(random_seed=42))
    manifest, dist_report = splitter.split_sessions(sessions)

    out_md = tmp_path / "reports" / "validation.md"
    out_json = tmp_path / "reports" / "validation.json"

    md_str, json_dict = generate_validation_report(
        manifest=manifest,
        distribution_report=dist_report,
        output_md_path=out_md,
        output_json_path=out_json,
    )

    assert out_md.exists()
    assert out_json.exists()

    # Markdown assertions
    assert "# Dataset Validation & Slice Distribution Report:" in md_str
    assert "Session Leakage Protection" in md_str
    assert "PASSED (0 Leakage)" in md_str
    assert "Train" in md_str
    assert "Validation" in md_str
    assert "Held-Out Test" in md_str

    # JSON assertions
    with open(out_json, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert loaded_json["dataset_id"] == manifest.dataset_id
    assert loaded_json["zero_session_leakage"] is True
    assert loaded_json["statistics"]["total_samples"] == 9
    assert "distribution_slices" in loaded_json
