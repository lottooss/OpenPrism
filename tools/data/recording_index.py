"""Recording index management, session metadata indexing, and offline validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import jsonschema


def get_schema_path() -> Path:
    """Return the absolute path to recording_session.schema.json."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "schemas"
        / "manifest"
        / "recording_session.schema.json"
    )


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hexadecimal digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class CorrelationIdInfo:
    sequence_id: int = 0
    source_timestamp_ns: int = 0
    pipeline_run_id: int = 0
    flags: int = 0


@dataclass
class FrameEntry:
    sample_id: str
    frame_sequence_id: int
    timestamp_ns: int
    relative_path: str
    sha256: str
    width: int
    height: int
    correlation_id: CorrelationIdInfo = field(default_factory=CorrelationIdInfo)
    label_path: str | None = None
    label_sha256: str | None = None
    is_annotated: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sample_id": self.sample_id,
            "frame_sequence_id": self.frame_sequence_id,
            "timestamp_ns": self.timestamp_ns,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "correlation_id": asdict(self.correlation_id),
            "is_annotated": self.is_annotated,
        }
        if self.label_path is not None:
            d["label_path"] = self.label_path
        if self.label_sha256 is not None:
            d["label_sha256"] = self.label_sha256
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameEntry:
        corr_data = data.get("correlation_id", {})
        corr = CorrelationIdInfo(
            sequence_id=corr_data.get("sequence_id", 0),
            source_timestamp_ns=corr_data.get("source_timestamp_ns", 0),
            pipeline_run_id=corr_data.get("pipeline_run_id", 0),
            flags=corr_data.get("flags", 0),
        )
        return cls(
            sample_id=data["sample_id"],
            frame_sequence_id=data["frame_sequence_id"],
            timestamp_ns=data["timestamp_ns"],
            relative_path=data["relative_path"],
            sha256=data["sha256"],
            width=data["width"],
            height=data["height"],
            correlation_id=corr,
            label_path=data.get("label_path"),
            label_sha256=data.get("label_sha256"),
            is_annotated=data.get("is_annotated", False),
        )


@dataclass
class SourceInfo:
    resolution_width: int = 1920
    resolution_height: int = 1080
    target_fps: float = 144.0
    color_format: str = "b8g8r8a8_unorm"
    adapter_luid: str = "0x0000000000000000"
    capture_backend: str = "dxgi_duplication"


@dataclass
class StorageInfo:
    recording_dir: str = "recordings"
    format: str = "raw_binary"
    data_retention_days: int = 30


@dataclass
class SessionStatistics:
    start_timestamp_utc: str = ""
    end_timestamp_utc: str = ""
    start_time_ns: int = 0
    end_time_ns: int = 0
    duration_ms: float = 0.0
    total_frames_captured: int = 0
    total_frames_recorded: int = 0
    total_frames_dropped: int = 0
    total_bytes: int = 0


@dataclass
class RecordingSession:
    schema_version: int = 1
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str = "gridshot"
    task_type: str = "flick"
    visual_profile: str = "default"
    notes: str = ""
    source: SourceInfo = field(default_factory=SourceInfo)
    storage: StorageInfo = field(default_factory=StorageInfo)
    statistics: SessionStatistics = field(default_factory=SessionStatistics)
    frames: list[FrameEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "scenario_name": self.scenario_name,
            "task_type": self.task_type,
            "visual_profile": self.visual_profile,
            "notes": self.notes,
            "source": asdict(self.source),
            "storage": asdict(self.storage),
            "statistics": asdict(self.statistics),
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordingSession:
        source = SourceInfo(**data["source"])
        storage = StorageInfo(**data["storage"])
        statistics = SessionStatistics(**data["statistics"])
        frames = [FrameEntry.from_dict(f) for f in data.get("frames", [])]
        return cls(
            schema_version=data.get("schema_version", 1),
            session_id=data["session_id"],
            scenario_name=data["scenario_name"],
            task_type=data["task_type"],
            visual_profile=data["visual_profile"],
            notes=data.get("notes", ""),
            source=source,
            storage=storage,
            statistics=statistics,
            frames=frames,
        )

    def validate_schema(self) -> None:
        """Validate this session dictionary representation against the JSON schema."""
        schema_file = get_schema_path()
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=self.to_dict(), schema=schema)

    def save_json(self, output_path: Path | str) -> None:
        """Save this session to a JSON file after schema validation."""
        self.validate_schema()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, input_path: Path | str) -> RecordingSession:
        """Load and validate a RecordingSession from a JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = cls.from_dict(data)
        session.validate_schema()
        return session


@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_frames: int = 0
    verified_files: int = 0
    missing_files: int = 0
    corrupted_files: int = 0
    duplicate_sequences: int = 0


class RecordingIndexer:
    """Utility for scanning, indexing, and validating recording directories."""

    @staticmethod
    def index_directory(
        directory: Path | str,
        session_id: str | None = None,
        scenario_name: str = "gridshot",
        task_type: str = "flick",
        visual_profile: str = "default",
        resolution: tuple[int, int] = (1920, 1080),
        target_fps: float = 144.0,
        color_format: str = "b8g8r8a8_unorm",
        retention_days: int = 30,
        notes: str = "",
    ) -> RecordingSession:
        """Scan a recording folder, compute hashes for binary/image files, and generate a RecordingSession."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Recording directory does not exist: {dir_path}")

        session_uuid = session_id or str(uuid.uuid4())
        frame_files = sorted(
            [
                p
                for p in dir_path.iterdir()
                if p.is_file()
                and p.suffix.lower() in [".bin", ".png", ".raw", ".dat"]
                and not p.name.endswith(".json")
            ]
        )

        frames: list[FrameEntry] = []
        total_bytes = 0
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        start_time_ns = 1_000_000_000
        time_step_ns = int(1_000_000_000 / target_fps)

        for seq_id, file_path in enumerate(frame_files, start=1):
            file_bytes = file_path.stat().st_size
            total_bytes += file_bytes
            sha256_hex = compute_file_sha256(file_path)
            frame_ts_ns = start_time_ns + (seq_id - 1) * time_step_ns

            entry = FrameEntry(
                sample_id=f"{session_uuid}_{seq_id}",
                frame_sequence_id=seq_id,
                timestamp_ns=frame_ts_ns,
                relative_path=file_path.name,
                sha256=sha256_hex,
                width=resolution[0],
                height=resolution[1],
                correlation_id=CorrelationIdInfo(
                    sequence_id=seq_id,
                    source_timestamp_ns=frame_ts_ns,
                    pipeline_run_id=1,
                    flags=0,
                ),
                is_annotated=False,
            )
            frames.append(entry)

        end_time_ns = (
            start_time_ns + len(frames) * time_step_ns if frames else start_time_ns
        )
        duration_ms = (
            (end_time_ns - start_time_ns) / 1_000_000.0 if frames else 0.0
        )

        source = SourceInfo(
            resolution_width=resolution[0],
            resolution_height=resolution[1],
            target_fps=target_fps,
            color_format=color_format,
            capture_backend="dxgi_duplication",
        )
        storage = StorageInfo(
            recording_dir=str(dir_path.name),
            format="raw_binary",
            data_retention_days=retention_days,
        )
        statistics = SessionStatistics(
            start_timestamp_utc=now_iso,
            end_timestamp_utc=now_iso,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            duration_ms=duration_ms,
            total_frames_captured=len(frames),
            total_frames_recorded=len(frames),
            total_frames_dropped=0,
            total_bytes=total_bytes,
        )

        session = RecordingSession(
            schema_version=1,
            session_id=session_uuid,
            scenario_name=scenario_name,
            task_type=task_type,
            visual_profile=visual_profile,
            notes=notes,
            source=source,
            storage=storage,
            statistics=statistics,
            frames=frames,
        )
        session.validate_schema()
        return session

    @staticmethod
    def validate_session(
        session: RecordingSession,
        base_dir: Path | str | None = None,
        verify_hashes: bool = True,
    ) -> ValidationReport:
        """Thoroughly audit a RecordingSession for schema conformance, missing files, corrupted hashes, and duplicate frames."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Schema Validation
        try:
            session.validate_schema()
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")

        # 2. Sequence and Timestamp Analysis
        seen_seqs: set[int] = set()
        seen_timestamps: set[int] = set()
        last_ts = -1

        missing_files = 0
        corrupted_files = 0
        verified_files = 0
        duplicate_sequences = 0

        target_base = Path(base_dir) if base_dir else Path(session.storage.recording_dir)

        for frame in session.frames:
            # Check duplicate sequence IDs
            if frame.frame_sequence_id in seen_seqs:
                duplicate_sequences += 1
                errors.append(
                    f"Duplicate frame sequence ID detected: {frame.frame_sequence_id}"
                )
            seen_seqs.add(frame.frame_sequence_id)

            # Check duplicate or non-monotonic timestamps
            if frame.timestamp_ns in seen_timestamps:
                warnings.append(
                    f"Duplicate timestamp detected: {frame.timestamp_ns} ns for sequence {frame.frame_sequence_id}"
                )
            seen_timestamps.add(frame.timestamp_ns)

            if last_ts >= 0 and frame.timestamp_ns < last_ts:
                errors.append(
                    f"Non-monotonic timestamp detected at seq {frame.frame_sequence_id}: {frame.timestamp_ns} < {last_ts}"
                )
            last_ts = frame.timestamp_ns

            # 3. File existence and SHA-256 verification
            if target_base.exists():
                frame_file = target_base / frame.relative_path
                if not frame_file.exists():
                    missing_files += 1
                    errors.append(f"Missing frame file on disk: {frame_file}")
                elif verify_hashes:
                    actual_sha = compute_file_sha256(frame_file)
                    if actual_sha.lower() != frame.sha256.lower():
                        corrupted_files += 1
                        errors.append(
                            f"Corrupted frame hash at seq {frame.frame_sequence_id}: expected {frame.sha256}, got {actual_sha}"
                        )
                    else:
                        verified_files += 1
                else:
                    verified_files += 1

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            total_frames=len(session.frames),
            verified_files=verified_files,
            missing_files=missing_files,
            corrupted_files=corrupted_files,
            duplicate_sequences=duplicate_sequences,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenPrism Recording Indexer and Session Manifest CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    idx_parser = subparsers.add_parser("index", help="Index a recording folder")
    idx_parser.add_argument(
        "--dir", required=True, help="Path to recording folder"
    )
    idx_parser.add_argument(
        "--output", required=True, help="Output manifest JSON path"
    )
    idx_parser.add_argument("--scenario", default="gridshot", help="Scenario name")
    idx_parser.add_argument(
        "--task-type",
        default="flick",
        choices=[
            "flick",
            "tracking",
            "switching",
            "synthetic",
            "calibration",
            "custom",
        ],
    )
    idx_parser.add_argument(
        "--visual-profile", default="default", help="Visual profile label"
    )
    idx_parser.add_argument(
        "--fps", type=float, default=144.0, help="Target recording FPS"
    )

    # validate command
    val_parser = subparsers.add_parser("validate", help="Validate a session manifest")
    val_parser.add_argument(
        "--manifest", required=True, help="Path to session manifest JSON"
    )
    val_parser.add_argument(
        "--base-dir", default=None, help="Base directory of recorded frame files"
    )

    args = parser.parse_args()

    if args.command == "index":
        session = RecordingIndexer.index_directory(
            directory=args.dir,
            scenario_name=args.scenario,
            task_type=args.task_type,
            visual_profile=args.visual_profile,
            target_fps=args.fps,
        )
        session.save_json(args.output)
        print(
            f"Successfully indexed session {session.session_id} with {len(session.frames)} frames -> {args.output}"
        )

    elif args.command == "validate":
        session = RecordingSession.load_json(args.manifest)
        report = RecordingIndexer.validate_session(
            session, base_dir=args.base_dir
        )
        print(f"Validation Result: {'PASSED' if report.is_valid else 'FAILED'}")
        print(f"Total Frames: {report.total_frames}")
        print(f"Verified Files: {report.verified_files}")
        print(f"Missing Files: {report.missing_files}")
        print(f"Corrupted Files: {report.corrupted_files}")
        if report.errors:
            print("Errors:")
            for err in report.errors:
                print(f"  - {err}")
        if not report.is_valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
