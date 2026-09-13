"""Perception model manifest dataclasses, schema validation, and artifact tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import jsonschema


def get_model_schema_path() -> Path:
    """Return path to schemas/manifest/model_manifest.schema.json."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "schemas"
        / "manifest"
        / "model_manifest.schema.json"
    )


@dataclass
class InputTensorSpec:
    name: str = "images"
    shape: list[int] = field(default_factory=lambda: [1, 3, 384, 640])
    dtype: str = "float32"
    color_format: str = "RGB"
    layout: str = "NCHW"
    normalization: str = "zero_to_one"


@dataclass
class OutputTensorSpec:
    name: str = "output0"
    shape: list[int] = field(default_factory=lambda: [1, 5, 5040])
    dtype: str = "float32"
    classes: list[str] = field(default_factory=lambda: ["target_sphere"])


@dataclass
class DatasetProvenance:
    dataset_id: str
    dataset_name: str
    dataset_sha256: str
    train_samples: int
    val_samples: int
    test_samples: int


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "AdamW"
    seed: int = 42


@dataclass
class PerceptionMetrics:
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1_score: float
    mean_center_error_px: float
    expected_calibration_error: float
    p95_center_error_px: float = 0.0
    max_center_error_px: float = 0.0


@dataclass
class SliceMetric:
    sample_count: int
    precision: float
    recall: float
    f1_score: float
    mean_center_error_px: float


@dataclass
class ModelArtifact:
    artifact_type: str
    path: str
    sha256: str
    size_bytes: int
    description: str = ""


@dataclass
class ModelManifest:
    schema_version: int
    model_id: str
    model_name: str
    architecture: str
    version: str
    license: str
    input_tensor: InputTensorSpec
    output_tensor: OutputTensorSpec
    dataset_provenance: DatasetProvenance
    training_config: TrainingConfig
    metrics: PerceptionMetrics
    scenario_slices: dict[str, SliceMetric]
    model_artifacts: list[ModelArtifact]
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    )
    git_info: dict[str, Any] = field(
        default_factory=lambda: {
            "commit_hash": "0000000000000000000000000000000000000000",
            "branch": "main",
            "is_dirty": False,
        }
    )

    def validate_schema(self) -> None:
        """Validate instance strictly against JSON Schema."""
        schema_path = get_model_schema_path()
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        jsonschema.validate(instance=self.to_dict(), schema=schema)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        d = asdict(self)
        # Ensure scenario_slices dict items are formatted as dicts
        d["scenario_slices"] = {
            k: asdict(v) if isinstance(v, SliceMetric) else v
            for k, v in self.scenario_slices.items()
        }
        return d

    def save_json(self, file_path: Path | str) -> None:
        """Validate and write model manifest to JSON file."""
        self.validate_schema()
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: Path | str) -> ModelManifest:
        """Load and validate model manifest from JSON file."""
        p = Path(file_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate the original mapping before dataclass conversion can discard unknown keys.
        schema = json.loads(get_model_schema_path().read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)

        input_t = InputTensorSpec(**data["input_tensor"])
        output_t = OutputTensorSpec(**data["output_tensor"])
        prov = DatasetProvenance(**data["dataset_provenance"])
        train_cfg = TrainingConfig(**data["training_config"])
        metrics = PerceptionMetrics(**data["metrics"])
        slices = {k: SliceMetric(**v) for k, v in data.get("scenario_slices", {}).items()}
        artifacts = [ModelArtifact(**a) for a in data.get("model_artifacts", [])]

        manifest = cls(
            schema_version=data["schema_version"],
            model_id=data["model_id"],
            model_name=data["model_name"],
            architecture=data["architecture"],
            version=data["version"],
            license=data["license"],
            input_tensor=input_t,
            output_tensor=output_t,
            dataset_provenance=prov,
            training_config=train_cfg,
            metrics=metrics,
            scenario_slices=slices,
            model_artifacts=artifacts,
            created_at_utc=data.get("created_at_utc", ""),
            git_info=data.get("git_info", {}),
        )
        manifest.validate_schema()
        return manifest
