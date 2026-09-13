"""Unit and adversarial tests for perception model manifest and schema validation."""

import json
from pathlib import Path
import pytest

from tools.perception.model_manifest import (
    DatasetProvenance,
    InputTensorSpec,
    ModelArtifact,
    ModelManifest,
    OutputTensorSpec,
    PerceptionMetrics,
    SliceMetric,
    TrainingConfig,
    get_model_schema_path,
)


@pytest.fixture
def valid_model_manifest() -> ModelManifest:
    return ModelManifest(
        schema_version=1,
        model_id="11111111-2222-4333-8444-555555555555",
        model_name="yolo11n_aimlabs_reference",
        architecture="yolo11n",
        version="1.0.0",
        license="AGPL-3.0",
        input_tensor=InputTensorSpec(
            name="images",
            shape=[1, 3, 384, 640],
            dtype="float32",
            color_format="RGB",
            layout="NCHW",
            normalization="zero_to_one",
        ),
        output_tensor=OutputTensorSpec(
            name="output0",
            shape=[1, 5, 5040],
            dtype="float32",
            classes=["target_sphere"],
        ),
        dataset_provenance=DatasetProvenance(
            dataset_id="22222222-3333-4444-8555-666666666666",
            dataset_name="aim_reference_dataset",
            dataset_sha256="a" * 64,
            train_samples=700,
            val_samples=150,
            test_samples=150,
        ),
        training_config=TrainingConfig(
            epochs=100,
            batch_size=16,
            learning_rate=0.01,
            optimizer="AdamW",
            seed=42,
        ),
        metrics=PerceptionMetrics(
            map50=0.985,
            map50_95=0.880,
            precision=0.965,
            recall=0.970,
            f1_score=0.967,
            mean_center_error_px=1.25,
            p95_center_error_px=2.80,
            max_center_error_px=4.50,
            expected_calibration_error=0.042,
        ),
        scenario_slices={
            "aimlabs_grid": SliceMetric(
                sample_count=50,
                precision=0.98,
                recall=0.99,
                f1_score=0.985,
                mean_center_error_px=1.10,
            ),
            "dark_mode": SliceMetric(
                sample_count=50,
                precision=0.95,
                recall=0.96,
                f1_score=0.955,
                mean_center_error_px=1.40,
            ),
        },
        model_artifacts=[
            ModelArtifact(
                artifact_type="weights_pt",
                path="weights/yolo11n_best.pt",
                sha256="b" * 64,
                size_bytes=5_200_000,
                description="PyTorch trained weights",
            ),
            ModelArtifact(
                artifact_type="model_onnx",
                path="models/yolo11n_384x640.onnx",
                sha256="c" * 64,
                size_bytes=5_100_000,
                description="Fixed-shape ONNX model",
            ),
        ],
    )


def test_model_schema_exists() -> None:
    """Verify that model_manifest.schema.json exists and is valid JSON schema."""
    schema_p = get_model_schema_path()
    assert schema_p.exists()

    with open(schema_p, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert schema.get("title") == "AimAgentModelManifest"


def test_model_manifest_validation(valid_model_manifest: ModelManifest) -> None:
    """Verify valid model manifest passes JSON schema validation."""
    valid_model_manifest.validate_schema()


def test_model_manifest_roundtrip_file(
    valid_model_manifest: ModelManifest, tmp_path: Path
) -> None:
    """Verify saving to and loading from JSON file preserves all fields."""
    file_path = tmp_path / "model_manifest.json"
    valid_model_manifest.save_json(file_path)
    assert file_path.exists()

    loaded = ModelManifest.load_json(file_path)
    assert loaded.model_id == valid_model_manifest.model_id
    assert loaded.architecture == "yolo11n"
    assert loaded.metrics.precision == valid_model_manifest.metrics.precision
    assert loaded.metrics.mean_center_error_px == 1.25
    assert len(loaded.model_artifacts) == 2
    assert "aimlabs_grid" in loaded.scenario_slices


def test_model_manifest_invalid_uuid(valid_model_manifest: ModelManifest) -> None:
    """Detect invalid non-UUID model IDs."""
    valid_model_manifest.model_id = "invalid-uuid-format"
    with pytest.raises(Exception):
        valid_model_manifest.validate_schema()


def test_load_rejects_unknown_raw_keys(tmp_path: Path, valid_model_manifest: ModelManifest) -> None:
    import jsonschema
    data = valid_model_manifest.to_dict()
    data["unexpected_trust_override"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        ModelManifest.load_json(path)
