"""Dataset manifest creation, session-aware leakage-safe splitting, and Git-clean verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import jsonschema

from tools.data.recording_index import RecordingSession


def get_dataset_schema_path() -> Path:
    """Return the absolute path to dataset_manifest.schema.json."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "schemas"
        / "manifest"
        / "dataset_manifest.schema.json"
    )


@dataclass
class SplitPolicy:
    strategy: str = "session_aware"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total:.4f}: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )


@dataclass
class DatasetSample:
    sample_id: str
    session_id: str
    task_type: str
    visual_profile: str
    split: str  # "train", "val", "test"
    image_path: str
    image_sha256: str
    label_path: str
    label_sha256: str
    annotation_count: int
    width: int
    height: int


@dataclass
class DatasetStatistics:
    total_samples: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    total_annotations: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    dataset_sha256: str = ""


@dataclass
class DatasetManifest:
    schema_version: int = 1
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = "aim_reference_targets"
    version: str = "1.0.0"
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    description: str = "Canonical reference target dataset with session-aware leakage-safe splits"
    license: str = "PolyForm-Noncommercial-1.0.0"
    classes: list[str] = field(default_factory=lambda: ["target_sphere"])
    split_policy: SplitPolicy = field(default_factory=SplitPolicy)
    source_sessions: list[str] = field(default_factory=list)
    statistics: DatasetStatistics = field(default_factory=DatasetStatistics)
    samples: list[DatasetSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "created_at_utc": self.created_at_utc,
            "description": self.description,
            "license": self.license,
            "classes": self.classes,
            "split_policy": asdict(self.split_policy),
            "source_sessions": self.source_sessions,
            "statistics": asdict(self.statistics),
            "samples": [asdict(s) for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        split_policy = SplitPolicy(**data["split_policy"])
        statistics = DatasetStatistics(**data["statistics"])
        samples = [DatasetSample(**s) for s in data.get("samples", [])]
        return cls(
            schema_version=data.get("schema_version", 1),
            dataset_id=data["dataset_id"],
            dataset_name=data["dataset_name"],
            version=data["version"],
            created_at_utc=data["created_at_utc"],
            description=data.get("description", ""),
            license=data["license"],
            classes=data.get("classes", ["target_sphere"]),
            split_policy=split_policy,
            source_sessions=data.get("source_sessions", []),
            statistics=statistics,
            samples=samples,
        )

    def validate_schema(self) -> None:
        """Validate this dataset manifest against the JSON schema."""
        schema_file = get_dataset_schema_path()
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=self.to_dict(), schema=schema)

    def save_json(self, output_path: Path | str) -> None:
        """Save this dataset manifest to JSON file after schema validation."""
        self.validate_schema()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, input_path: Path | str) -> DatasetManifest:
        """Load and validate a DatasetManifest from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls.from_dict(data)
        manifest.validate_schema()
        return manifest


@dataclass
class ManifestValidationReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_samples: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    session_leakage_detected: bool = False
    git_clean: bool = True


class DatasetManifestBuilder:
    """Builder for session-aware dataset manifests with leakage protection and Git cleanliness."""

    @staticmethod
    def build_from_sessions(
        sessions: list[RecordingSession],
        dataset_name: str = "aim_reference_targets",
        version: str = "1.0.0",
        classes: list[str] | None = None,
        split_policy: SplitPolicy | None = None,
        description: str = "Canonical reference target dataset",
        license_name: str = "PolyForm-Noncommercial-1.0.0",
    ) -> DatasetManifest:
        """Build a DatasetManifest from a list of RecordingSession objects enforcing session-aware splits."""
        if not sessions:
            raise ValueError("Cannot build a dataset manifest from an empty session list")

        policy = split_policy or SplitPolicy()
        policy.validate()
        target_classes = classes or ["target_sphere"]

        # Sort sessions deterministically by session_id
        sorted_sessions = sorted(sessions, key=lambda s: s.session_id)

        # Shuffle session IDs with the fixed random seed to assign whole sessions to splits
        rng = random.Random(policy.random_seed)
        shuffled_sessions = list(sorted_sessions)
        rng.shuffle(shuffled_sessions)

        num_sessions = len(shuffled_sessions)
        train_end = max(1, int(round(num_sessions * policy.train_ratio)))
        val_end = max(train_end, int(round(num_sessions * (policy.train_ratio + policy.val_ratio))))

        session_split_map: dict[str, str] = {}
        for i, s in enumerate(shuffled_sessions):
            if i < train_end:
                session_split_map[s.session_id] = "train"
            elif i < val_end:
                session_split_map[s.session_id] = "val"
            else:
                session_split_map[s.session_id] = "test"

        # If few sessions exist, ensure all configured splits have at least one session if requested
        if num_sessions >= 3 and len(set(session_split_map.values())) < 3:
            session_split_map[shuffled_sessions[0].session_id] = "train"
            session_split_map[shuffled_sessions[1].session_id] = "val"
            session_split_map[shuffled_sessions[2].session_id] = "test"

        samples: list[DatasetSample] = []
        train_count = 0
        val_count = 0
        test_count = 0
        total_annotations = 0
        class_dist: dict[str, int] = {c: 0 for c in target_classes}

        hasher = hashlib.sha256()

        for s in sorted_sessions:
            split = session_split_map[s.session_id]
            for f in s.frames:
                label_path = f.label_path or f"{Path(f.relative_path).stem}.txt"
                label_sha = f.label_sha256 or "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

                sample = DatasetSample(
                    sample_id=f.sample_id,
                    session_id=s.session_id,
                    task_type=s.task_type,
                    visual_profile=s.visual_profile,
                    split=split,
                    image_path=f.relative_path,
                    image_sha256=f.sha256,
                    label_path=label_path,
                    label_sha256=label_sha,
                    annotation_count=1 if f.is_annotated else 0,
                    width=f.width,
                    height=f.height,
                )
                samples.append(sample)

                if split == "train":
                    train_count += 1
                elif split == "val":
                    val_count += 1
                elif split == "test":
                    test_count += 1

                if f.is_annotated:
                    total_annotations += 1
                    class_dist[target_classes[0]] = class_dist.get(target_classes[0], 0) + 1

                hasher.update(f.sample_id.encode("utf-8"))
                hasher.update(f.sha256.encode("utf-8"))
                hasher.update(split.encode("utf-8"))

        dataset_sha256 = hasher.hexdigest()

        stats = DatasetStatistics(
            total_samples=len(samples),
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            total_annotations=total_annotations,
            class_distribution=class_dist,
            dataset_sha256=dataset_sha256,
        )

        manifest = DatasetManifest(
            schema_version=1,
            dataset_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            version=version,
            description=description,
            license=license_name,
            classes=target_classes,
            split_policy=policy,
            source_sessions=[s.session_id for s in sorted_sessions],
            statistics=stats,
            samples=samples,
        )
        manifest.validate_schema()
        return manifest

    @staticmethod
    def verify_git_clean(manifest: DatasetManifest, repo_root: Path | str | None = None) -> tuple[bool, list[str]]:
        """Verify that NO dataset sample images or binary files are tracked in Git."""
        errors: list[str] = []
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent.parent

        try:
            res = subprocess.run(
                ["git", "ls-files", "datasets/", "recordings/"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            tracked_files = set(res.stdout.strip().splitlines()) if res.stdout else set()
            if tracked_files:
                for tf in sorted(tracked_files):
                    errors.append(f"Git-clean violation: Dataset/recording artifact is tracked by git: {tf}")
        except Exception as e:
            errors.append(f"Could not execute git ls-files to verify git cleanliness: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_manifest(
        manifest: DatasetManifest,
        base_dir: Path | str | None = None,
        repo_root: Path | str | None = None,
    ) -> ManifestValidationReport:
        """Validate dataset manifest schema, check for session leakage across splits, and verify git cleanliness."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Schema validation
        try:
            manifest.validate_schema()
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")

        # 2. Session Leakage Check: verify that no single session appears in multiple splits
        session_to_splits: dict[str, set[str]] = {}
        for sample in manifest.samples:
            session_to_splits.setdefault(sample.session_id, set()).add(sample.split)

        leakage_detected = False
        for sess_id, splits in session_to_splits.items():
            if len(splits) > 1:
                leakage_detected = True
                errors.append(
                    f"Adjacent-frame data leakage detected for session {sess_id}: present in multiple splits {sorted(splits)}"
                )

        # 3. Git clean check
        is_git_clean, git_errors = DatasetManifestBuilder.verify_git_clean(manifest, repo_root=repo_root)
        errors.extend(git_errors)

        # 4. File existence checks if base_dir is supplied
        if base_dir:
            b_path = Path(base_dir)
            for s in manifest.samples:
                img_f = b_path / s.image_path
                if not img_f.exists():
                    warnings.append(f"Sample image missing on disk: {img_f}")

        is_valid = len(errors) == 0
        return ManifestValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            total_samples=manifest.statistics.total_samples,
            train_count=manifest.statistics.train_count,
            val_count=manifest.statistics.val_count,
            test_count=manifest.statistics.test_count,
            session_leakage_detected=leakage_detected,
            git_clean=is_git_clean,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenPrism Dataset Manifest CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a dataset manifest from session JSON manifests")
    build_parser.add_argument("--sessions-dir", required=True, help="Directory containing session JSON manifests")
    build_parser.add_argument("--output", required=True, help="Output dataset manifest JSON path")
    build_parser.add_argument("--name", default="aim_reference_targets", help="Dataset name")
    build_parser.add_argument("--train-ratio", type=float, default=0.70)
    build_parser.add_argument("--val-ratio", type=float, default=0.15)
    build_parser.add_argument("--test-ratio", type=float, default=0.15)
    build_parser.add_argument("--seed", type=int, default=42)

    val_parser = subparsers.add_parser("validate", help="Validate a dataset manifest JSON")
    val_parser.add_argument("--manifest", required=True, help="Path to dataset manifest JSON")

    args = parser.parse_args()

    if args.command == "build":
        sess_dir = Path(args.sessions_dir)
        sessions = [
            RecordingSession.load_json(p)
            for p in sess_dir.glob("*.json")
            if not p.name.startswith("dataset_")
        ]
        if not sessions:
            print(f"No session JSON manifests found in {sess_dir}")
            raise SystemExit(1)

        policy = SplitPolicy(
            strategy="session_aware",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )
        dataset_manifest = DatasetManifestBuilder.build_from_sessions(
            sessions=sessions,
            dataset_name=args.name,
            split_policy=policy,
        )
        dataset_manifest.save_json(args.output)
        print(f"Saved dataset manifest -> {args.output} with {dataset_manifest.statistics.total_samples} samples")

    elif args.command == "validate":
        manifest = DatasetManifest.load_json(args.manifest)
        report = DatasetManifestBuilder.validate_manifest(manifest)
        print(f"Validation: {'PASSED' if report.is_valid else 'FAILED'}")
        print(f"Total Samples: {report.total_samples} (Train: {report.train_count}, Val: {report.val_count}, Test: {report.test_count})")
        print(f"Session Leakage Detected: {report.session_leakage_detected}")
        print(f"Git Clean: {report.git_clean}")
        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  - {e}")
        if not report.is_valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
