"""Unit and adversarial tests for ONNX fixed tensor contract validation and framework parity auditing."""

import numpy as np
import pytest
from pathlib import Path

from tools.perception.model_manifest import (
    DatasetProvenance,
    PerceptionMetrics,
    SliceMetric,
)
from tools.perception.onnx_exporter import (
    FrameworkParityVerifier,
    OnnxContractValidator,
    OnnxExportConfig,
    build_reference_onnx_manifest,
)


def test_onnx_tensor_contract_valid() -> None:
    """Verify that exact batch-1 640x384 input and 5040 anchor output pass validation."""
    cfg = OnnxExportConfig()
    errors = OnnxContractValidator.validate_tensor_contract(
        input_name="images",
        input_shape=[1, 3, 384, 640],
        output_name="output0",
        output_shape=[1, 5, 5040],
        config=cfg,
    )
    assert len(errors) == 0


def test_onnx_tensor_contract_invalid_shapes() -> None:
    """Detect and reject dynamic batch or unexpected tensor resolutions fail-closed."""
    cfg = OnnxExportConfig()

    # Dynamic batch or wrong resolution
    errors = OnnxContractValidator.validate_tensor_contract(
        input_name="images",
        input_shape=[-1, 3, 384, 640],  # Dynamic batch
        output_name="output0",
        output_shape=[1, 5, 5040],
        config=cfg,
    )
    assert len(errors) > 0
    assert any("Invalid input shape" in e for e in errors)


def test_onnx_operator_compatibility_disallowed_ops() -> None:
    """Detect and reject non-deterministic or graph-internal NMS operators."""
    valid_ops = ["Conv", "Relu", "Add", "Concat", "Transpose", "Reshape", "Sigmoid"]
    assert len(OnnxContractValidator.check_operator_compatibility(valid_ops)) == 0

    invalid_ops = ["Conv", "NonMaxSuppression", "RandomNormal"]
    errors = OnnxContractValidator.check_operator_compatibility(invalid_ops)
    assert len(errors) == 2
    assert any("NonMaxSuppression" in e for e in errors)
    assert any("RandomNormal" in e for e in errors)


def test_framework_parity_perfect_match() -> None:
    """Verify parity check succeeds on bitwise identical tensors."""
    ref = np.random.RandomState(42).randn(1, 5, 5040).astype(np.float32)
    onnx = ref.copy()

    res = FrameworkParityVerifier.compute_parity_metrics(ref, onnx)
    assert res.is_valid
    assert res.l_inf_error == 0.0
    assert res.relative_error == 0.0
    assert len(res.errors) == 0


def test_framework_parity_within_tolerance() -> None:
    """Verify parity check succeeds with minor floating point precision drift within 1e-4."""
    ref = np.random.RandomState(42).randn(1, 5, 5040).astype(np.float32)
    # Add tiny perturbation 1e-5
    onnx = ref + np.float32(1e-5)

    res = FrameworkParityVerifier.compute_parity_metrics(ref, onnx)
    assert res.is_valid
    assert res.l_inf_error <= 1e-4


def test_framework_parity_exceeding_tolerance() -> None:
    """Detect and fail parity check when numerical difference exceeds 1e-4."""
    ref = np.random.RandomState(42).randn(1, 5, 5040).astype(np.float32)
    # Add large perturbation 1e-2
    onnx = ref + np.float32(1e-2)

    res = FrameworkParityVerifier.compute_parity_metrics(ref, onnx)
    assert not res.is_valid
    assert res.l_inf_error > 1e-4
    assert len(res.errors) > 0


def test_build_reference_onnx_manifest() -> None:
    """Verify building and validating an exported ONNX model manifest."""
    dataset_prov = DatasetProvenance(
        dataset_id="11111111-2222-4333-8444-555555555555",
        dataset_name="aim_reference_dataset",
        dataset_sha256="e" * 64,
        train_samples=700,
        val_samples=150,
        test_samples=150,
    )
    metrics = PerceptionMetrics(
        map50=0.98,
        map50_95=0.88,
        precision=0.96,
        recall=0.97,
        f1_score=0.965,
        mean_center_error_px=1.20,
        expected_calibration_error=0.04,
    )
    slices = {
        "aimlabs_grid": SliceMetric(
            sample_count=50,
            precision=0.98,
            recall=0.99,
            f1_score=0.985,
            mean_center_error_px=1.10,
        )
    }

    manifest = build_reference_onnx_manifest(
        model_id="22222222-3333-4444-8555-666666666666",
        onnx_path="models/yolo11n_384x640.onnx",
        onnx_sha256="f" * 64,
        onnx_size_bytes=5_200_000,
        metrics=metrics,
        scenario_slices=slices,
        dataset_prov=dataset_prov,
    )

    assert manifest.architecture == "yolo11n"
    assert manifest.input_tensor.shape == [1, 3, 384, 640]
    assert manifest.output_tensor.shape == [1, 5, 5040]
    assert len(manifest.model_artifacts) == 1
    assert manifest.model_artifacts[0].artifact_type == "model_onnx"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_parity_rejects_nonfinite_output(value: float) -> None:
    ref = np.zeros((1, 5, 5040), dtype=np.float32)
    actual = ref.copy()
    actual[0, 0, 0] = value
    assert not FrameworkParityVerifier.compute_parity_metrics(ref, actual).is_valid


def test_parity_rejects_equal_but_wrong_shapes() -> None:
    value = np.zeros((0,), dtype=np.float32)
    assert not FrameworkParityVerifier.compute_parity_metrics(value, value).is_valid


def test_export_rejects_untrusted_checkpoint_before_loading(tmp_path: Path) -> None:
    from tools.perception.onnx_exporter import export_trusted_torchscript
    path = tmp_path / "untrusted.pt"
    path.write_bytes(b"not executable")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_trusted_torchscript(path, "0" * 64, tmp_path / "input.npy", tmp_path / "out.onnx")
