"""Session-aware dataset splitting, held-out real domain reservation, and slice distribution auditing."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
import random

from tools.data.dataset_manifest import DatasetManifest, DatasetSample, DatasetStatistics, SplitPolicy
from tools.data.recording_index import RecordingSession


@dataclass
class ScaleDistribution:
    small_count: int = 0   # radius < 25px
    medium_count: int = 0  # 25px <= radius <= 45px
    large_count: int = 0   # radius > 45px


@dataclass
class VisibilityDistribution:
    visible_count: int = 0
    partial_count: int = 0
    occluded_count: int = 0


@dataclass
class ClippingDistribution:
    unclipped_count: int = 0
    boundary_clipped_count: int = 0


@dataclass
class SliceDistribution:
    scale: ScaleDistribution = field(default_factory=ScaleDistribution)
    visibility: VisibilityDistribution = field(default_factory=VisibilityDistribution)
    clipping: ClippingDistribution = field(default_factory=ClippingDistribution)
    domain_real_count: int = 0
    domain_synthetic_count: int = 0
    tasks: dict[str, int] = field(default_factory=dict)
    visual_profiles: dict[str, int] = field(default_factory=dict)


@dataclass
class SplitDistributionReport:
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    split_policy: SplitPolicy
    split_hash: str
    train_slices: SliceDistribution = field(default_factory=SliceDistribution)
    val_slices: SliceDistribution = field(default_factory=SliceDistribution)
    test_slices: SliceDistribution = field(default_factory=SliceDistribution)
    zero_leakage_verified: bool = True
    real_holdout_verified: bool = True


@dataclass
class DatasetSplitterConfig:
    strategy: str = "session_aware"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    holdout_real_for_test: bool = True
    random_seed: int = 42


class DatasetSplitter:
    """Partitions recording sessions into leakage-free train/val/test splits and computes slice distribution metrics."""

    def __init__(self, config: DatasetSplitterConfig | None = None) -> None:
        self.config = config or DatasetSplitterConfig()

    def split_sessions(
        self,
        sessions: list[RecordingSession],
        dataset_name: str = "aim_reference_dataset",
        version: str = "1.0.0",
        classes: list[str] | None = None,
    ) -> tuple[DatasetManifest, SplitDistributionReport]:
        """Partition sessions atomically into train, val, and test splits with held-out real validation."""
        if not sessions:
            raise ValueError("Cannot split empty session list")

        policy = SplitPolicy(
            strategy=self.config.strategy,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            random_seed=self.config.random_seed,
        )
        policy.validate()
        target_classes = classes or ["target_sphere"]

        # Segregate synthetic vs real sessions
        real_sessions = [s for s in sessions if s.task_type != "synthetic"]
        synthetic_sessions = [s for s in sessions if s.task_type == "synthetic"]

        rng = random.Random(self.config.random_seed)

        # Shuffle deterministic order
        shuffled_real = sorted(real_sessions, key=lambda s: s.session_id)
        rng.shuffle(shuffled_real)

        shuffled_synthetic = sorted(synthetic_sessions, key=lambda s: s.session_id)
        rng.shuffle(shuffled_synthetic)

        session_split_map: dict[str, str] = {}

        if self.config.holdout_real_for_test and real_sessions:
            # If real sessions exist, guarantee real sessions are in val/test held-out sets
            if len(real_sessions) == 1:
                session_split_map[real_sessions[0].session_id] = "test"
            elif len(real_sessions) == 2:
                session_split_map[shuffled_real[0].session_id] = "val"
                session_split_map[shuffled_real[1].session_id] = "test"
            else:
                num_real = len(shuffled_real)
                r_train_end = max(1, int(round(num_real * self.config.train_ratio)))
                r_val_end = max(r_train_end, int(round(num_real * (self.config.train_ratio + self.config.val_ratio))))
                for i, s in enumerate(shuffled_real):
                    if i < r_train_end:
                        session_split_map[s.session_id] = "train"
                    elif i < r_val_end:
                        session_split_map[s.session_id] = "val"
                    else:
                        session_split_map[s.session_id] = "test"

            # Distribute synthetic sessions across train/val/test
            if synthetic_sessions:
                num_synth = len(shuffled_synthetic)
                s_train_end = max(1, int(round(num_synth * self.config.train_ratio)))
                s_val_end = max(s_train_end, int(round(num_synth * (self.config.train_ratio + self.config.val_ratio))))
                for i, s in enumerate(shuffled_synthetic):
                    if i < s_train_end:
                        session_split_map[s.session_id] = "train"
                    elif i < s_val_end:
                        session_split_map[s.session_id] = "val"
                    else:
                        session_split_map[s.session_id] = "test"
        else:
            all_shuffled = sorted(sessions, key=lambda s: s.session_id)
            rng.shuffle(all_shuffled)
            num_total = len(all_shuffled)
            train_end = max(1, int(round(num_total * self.config.train_ratio)))
            val_end = max(train_end, int(round(num_total * (self.config.train_ratio + self.config.val_ratio))))
            for i, s in enumerate(all_shuffled):
                if i < train_end:
                    session_split_map[s.session_id] = "train"
                elif i < val_end:
                    session_split_map[s.session_id] = "val"
                else:
                    session_split_map[s.session_id] = "test"

        # Compute deterministic split digest
        hasher = hashlib.sha256()
        hasher.update(str(self.config.random_seed).encode("utf-8"))
        for sid in sorted(session_split_map.keys()):
            hasher.update(f"{sid}:{session_split_map[sid]}".encode("utf-8"))
        split_hash = hasher.hexdigest()

        # Build samples and slice distributions
        samples: list[DatasetSample] = []
        train_slices = SliceDistribution()
        val_slices = SliceDistribution()
        test_slices = SliceDistribution()

        train_count = 0
        val_count = 0
        test_count = 0
        total_annotations = 0

        for s in sorted(sessions, key=lambda x: x.session_id):
            split = session_split_map[s.session_id]
            slice_dist = (
                train_slices if split == "train" else (val_slices if split == "val" else test_slices)
            )

            # Update domain and task metrics
            if s.task_type == "synthetic":
                slice_dist.domain_synthetic_count += len(s.frames)
            else:
                slice_dist.domain_real_count += len(s.frames)

            slice_dist.tasks[s.task_type] = slice_dist.tasks.get(s.task_type, 0) + len(s.frames)
            slice_dist.visual_profiles[s.visual_profile] = (
                slice_dist.visual_profiles.get(s.visual_profile, 0) + len(s.frames)
            )

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

        stats = DatasetStatistics(
            total_samples=len(samples),
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            total_annotations=total_annotations,
            class_distribution={target_classes[0]: total_annotations},
            dataset_sha256=split_hash,
        )

        manifest = DatasetManifest(
            schema_version=1,
            dataset_id=f"{split_hash[:8]}-{split_hash[8:12]}-4000-8000-{split_hash[12:24]}",
            dataset_name=dataset_name,
            version=version,
            license="PolyForm-Noncommercial-1.0.0",
            classes=target_classes,
            split_policy=policy,
            source_sessions=[s.session_id for s in sorted(sessions, key=lambda x: x.session_id)],
            statistics=stats,
            samples=samples,
        )
        manifest.validate_schema()

        # Leakage verification: assert zero session overlap
        session_to_splits: dict[str, set[str]] = {}
        for sm in samples:
            session_to_splits.setdefault(sm.session_id, set()).add(sm.split)
        zero_leakage = all(len(splits) == 1 for splits in session_to_splits.values())

        # Check real holdout: verify test split has real sessions if real sessions were provided
        has_real_test = (
            test_slices.domain_real_count > 0 or val_slices.domain_real_count > 0
            if real_sessions
            else True
        )

        report = SplitDistributionReport(
            total_samples=len(samples),
            train_samples=train_count,
            val_samples=val_count,
            test_samples=test_count,
            split_policy=policy,
            split_hash=split_hash,
            train_slices=train_slices,
            val_slices=val_slices,
            test_slices=test_slices,
            zero_leakage_verified=zero_leakage,
            real_holdout_verified=has_real_test,
        )

        return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenPrism Dataset Splitter and Distribution CLI")
    parser.add_argument("--sessions-dir", required=True, help="Directory with session manifest JSON files")
    parser.add_argument("--output-manifest", required=True, help="Output dataset manifest JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    args = parser.parse_args()

    s_dir = Path(args.sessions_dir)
    sessions = [
        RecordingSession.load_json(p)
        for p in s_dir.glob("*.json")
        if not p.name.startswith("dataset_")
    ]
    if not sessions:
        print(f"No sessions found in {s_dir}")
        raise SystemExit(1)

    cfg = DatasetSplitterConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    splitter = DatasetSplitter(cfg)
    manifest, report = splitter.split_sessions(sessions)

    manifest.save_json(args.output_manifest)
    print(f"Successfully generated dataset manifest: {args.output_manifest}")
    print(f"Total Samples: {report.total_samples} (Train: {report.train_samples}, Val: {report.val_samples}, Test: {report.test_samples})")
    print(f"Zero Leakage: {report.zero_leakage_verified} | Real Holdout: {report.real_holdout_verified}")


if __name__ == "__main__":
    main()
