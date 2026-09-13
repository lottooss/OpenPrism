"""Unit and adversarial tests for TensorRT FP16 perception runner configuration and manifest validation."""

from tools.perception.model_manifest import (
    DatasetProvenance,
    InputTensorSpec,
    ModelArtifact,
    ModelManifest,
    OutputTensorSpec,
    PerceptionMetrics,
    TrainingConfig,
)


def test_tensorrt_runner_manifest_configuration() -> None:
    """Verify that a valid TensorRT FP16 engine artifact manifest passes validation."""
    manifest = ModelManifest(
        schema_version=1,
        model_id="99999999-8888-4777-8666-555555555555",
        model_name="yolo11n_aimlabs_reference",
        architecture="yolo11n",
        version="1.0.0",
        license="AGPL-3.0",
        input_tensor=InputTensorSpec(
            name="images",
            shape=[1, 3, 384, 640],
            dtype="float16",
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
            dataset_id="11111111-2222-4333-8444-555555555555",
            dataset_name="aim_reference_dataset",
            dataset_sha256="0" * 64,
            train_samples=100,
            val_samples=20,
            test_samples=20,
        ),
        training_config=TrainingConfig(),
        metrics=PerceptionMetrics(
            map50=0.98,
            map50_95=0.88,
            precision=0.96,
            recall=0.97,
            f1_score=0.965,
            mean_center_error_px=1.20,
            expected_calibration_error=0.04,
        ),
        scenario_slices={},
        model_artifacts=[
            ModelArtifact(
                artifact_type="engine_trt",
                path="models/yolo11n_fp16.engine",
                sha256="1" * 64,
                size_bytes=6_500_000,
                description="TensorRT FP16 engine with CUDA Graph support",
            )
        ],
    )
    manifest.validate_schema()
    assert manifest.model_artifacts[0].artifact_type == "engine_trt"
    assert manifest.input_tensor.dtype == "float16"
