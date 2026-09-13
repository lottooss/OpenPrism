"""Fixed-shape ONNX export validation, graph topology checking, and framework parity auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.perception.model_manifest import (
    DatasetProvenance,
    InputTensorSpec,
    ModelArtifact,
    ModelManifest,
    OutputTensorSpec,
    PerceptionMetrics,
    SliceMetric,
    TrainingConfig,
)


@dataclass
class OnnxExportConfig:
    input_shape: tuple[int, int, int, int] = (1, 3, 384, 640)
    input_name: str = "images"
    output_names: list[str] = field(default_factory=lambda: ["output0"])
    output_shape: tuple[int, int, int] = (1, 5, 5040)
    opset_version: int = 17
    dtype: str = "float32"
    tolerance_l_inf: float = 1e-4
    tolerance_relative: float = 1e-3


@dataclass
class ParityVerificationResult:
    is_valid: bool
    l_inf_error: float
    relative_error: float
    max_allowed_l_inf: float
    max_allowed_relative: float
    input_shape: list[int]
    output_shape: list[int]
    errors: list[str] = field(default_factory=list)


class OnnxContractValidator:
    """Validates ONNX model contracts, fixed batch/resolution requirements, and operator constraints."""

    DISALLOWED_OPERATORS = {
        "NonMaxSuppression",  # NMS inside graph introduces dynamic shapes/branches; decoded in host/C++ plugin
        "RandomUniform",
        "RandomNormal",
        "Loop",
        "If",
    }

    @classmethod
    def validate_tensor_contract(
        cls,
        input_name: str,
        input_shape: list[int],
        output_name: str,
        output_shape: list[int],
        config: OnnxExportConfig,
    ) -> list[str]:
        """Validate that tensor names and fixed shapes adhere strictly to the technical blueprint."""
        errors: list[str] = []

        if input_shape != list(config.input_shape):
            errors.append(
                f"Invalid input shape {input_shape}; expected strictly fixed {list(config.input_shape)}"
            )

        if output_shape != list(config.output_shape):
            errors.append(
                f"Invalid output shape {output_shape}; expected strictly fixed {list(config.output_shape)}"
            )

        if input_name != config.input_name:
            errors.append(
                f"Invalid input tensor name '{input_name}'; expected '{config.input_name}'"
            )

        if output_name not in config.output_names:
            errors.append(
                f"Invalid output tensor name '{output_name}'; expected one of {config.output_names}"
            )

        return errors

    @classmethod
    def check_operator_compatibility(cls, operators_present: list[str]) -> list[str]:
        """Verify graph contains no disallowed dynamic or non-deterministic operators."""
        errors: list[str] = []
        for op in operators_present:
            if op in cls.DISALLOWED_OPERATORS:
                errors.append(
                    f"Disallowed operator '{op}' detected in graph. TensorRT engine requires fixed deterministic ops."
                )
        return errors


class FrameworkParityVerifier:
    """Verifies numerical equivalence between PyTorch reference model and ONNX exported graph."""

    @staticmethod
    def compute_parity_metrics(
        reference_output: np.ndarray,
        onnx_output: np.ndarray,
        config: OnnxExportConfig | None = None,
    ) -> ParityVerificationResult:
        """Compute L_inf and relative error between PyTorch reference and ONNX predictions."""
        cfg = config or OnnxExportConfig()
        errors: list[str] = []

        if (reference_output.shape != onnx_output.shape or
                reference_output.shape != cfg.output_shape or
                not np.isfinite(reference_output).all() or not np.isfinite(onnx_output).all() or
                not math.isfinite(cfg.tolerance_l_inf) or cfg.tolerance_l_inf < 0 or
                not math.isfinite(cfg.tolerance_relative) or cfg.tolerance_relative < 0):
            errors.append(
                f"Invalid shape, nonfinite tensor or tolerance: reference {reference_output.shape}, ONNX {onnx_output.shape}"
            )
            return ParityVerificationResult(
                is_valid=False,
                l_inf_error=float("inf"),
                relative_error=float("inf"),
                max_allowed_l_inf=cfg.tolerance_l_inf,
                max_allowed_relative=cfg.tolerance_relative,
                input_shape=list(cfg.input_shape),
                output_shape=list(cfg.output_shape),
                errors=errors,
            )

        diff = np.abs(reference_output - onnx_output)
        l_inf = float(np.max(diff))

        norm_ref = float(np.linalg.norm(reference_output))
        norm_diff = float(np.linalg.norm(diff))
        rel_error = norm_diff / (norm_ref + 1e-9)

        if l_inf > cfg.tolerance_l_inf:
            errors.append(
                f"L_inf error {l_inf:.6e} exceeds tolerance threshold {cfg.tolerance_l_inf:.6e}"
            )

        if rel_error > cfg.tolerance_relative:
            errors.append(
                f"Relative error {rel_error:.6e} exceeds tolerance threshold {cfg.tolerance_relative:.6e}"
            )

        is_valid = len(errors) == 0

        return ParityVerificationResult(
            is_valid=is_valid,
            l_inf_error=l_inf,
            relative_error=rel_error,
            max_allowed_l_inf=cfg.tolerance_l_inf,
            max_allowed_relative=cfg.tolerance_relative,
            input_shape=list(cfg.input_shape),
            output_shape=list(cfg.output_shape),
            errors=errors,
        )


def build_reference_onnx_manifest(
    model_id: str,
    onnx_path: str,
    onnx_sha256: str,
    onnx_size_bytes: int,
    metrics: PerceptionMetrics,
    scenario_slices: dict[str, SliceMetric],
    dataset_prov: DatasetProvenance,
    train_cfg: TrainingConfig | None = None,
) -> ModelManifest:
    """Build and validate a production ModelManifest for an exported ONNX model."""
    manifest = ModelManifest(
        schema_version=1,
        model_id=model_id,
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
        dataset_provenance=dataset_prov,
        training_config=train_cfg or TrainingConfig(),
        metrics=metrics,
        scenario_slices=scenario_slices,
        model_artifacts=[
            ModelArtifact(
                artifact_type="model_onnx",
                path=onnx_path,
                sha256=onnx_sha256,
                size_bytes=onnx_size_bytes,
                description="Fixed-shape batch-1 640x384 FP32 ONNX graph",
            )
        ],
    )
    manifest.validate_schema()
    return manifest


def export_trusted_torchscript(checkpoint: Path, expected_sha256: str, golden_inputs: Path,
                              output: Path, config: OnnxExportConfig | None = None) -> dict[str, Any]:
    """Export an explicitly supplied trusted TorchScript model and run real ONNX CPU parity.

    Does not download models, train, install packages, or claim TensorRT parity. Golden
    input archive is a float32 array shaped [N, 1, 3, 384, 640], never labels/predictions.
    TorchScript is executable: caller supplies the trusted artifact hash before loading.
    """
    cfg = config or OnnxExportConfig()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("Trusted checkpoint SHA-256 mismatch")
    inputs = np.load(golden_inputs, allow_pickle=False)
    if (inputs.dtype != np.float32 or inputs.ndim != 5 or len(inputs) == 0 or
            tuple(inputs.shape[1:]) != cfg.input_shape or not np.isfinite(inputs).all() or
            np.min(inputs) < 0 or np.max(inputs) > 1):
        raise ValueError("Golden inputs must be finite [N,1,3,384,640] RGB float32 in [0,1]")
    try:
        torch = importlib.import_module("torch")
        onnx = importlib.import_module("onnx")
        ort = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise RuntimeError("Use the locked project ML extra; torch, onnx and onnxruntime are required") from exc
    model = torch.jit.load(str(checkpoint), map_location="cpu").eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(inputs[0]), str(output),
                          input_names=[cfg.input_name], output_names=cfg.output_names,
                          opset_version=cfg.opset_version, dynamic_axes=None, dynamo=False)
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("Expected exactly one input and output")
    inp, out = graph.graph.input[0], graph.graph.output[0]
    errors = OnnxContractValidator.validate_tensor_contract(
        inp.name, [dim.dim_value for dim in inp.type.tensor_type.shape.dim],
        out.name, [dim.dim_value for dim in out.type.tensor_type.shape.dim], cfg)
    errors += OnnxContractValidator.check_operator_compatibility([node.op_type for node in graph.graph.node])
    if inp.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or out.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        errors.append("ONNX input/output must be float32; TensorRT engine I/O format is set at build time")
    if errors:
        raise ValueError("; ".join(errors))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    results: list[dict[str, Any]] = []
    for golden_input in inputs:
        with torch.inference_mode():
            reference = model(torch.from_numpy(golden_input))
        if not isinstance(reference, torch.Tensor):
            raise ValueError("Model must return the single raw detection tensor")
        candidate = session.run(cfg.output_names, {cfg.input_name: golden_input})[0]
        result = FrameworkParityVerifier.compute_parity_metrics(reference.cpu().numpy(), candidate, cfg)
        if not result.is_valid:
            raise ValueError("ONNX golden parity failed: " + "; ".join(result.errors))
        results.append(asdict(result))
    return {"schema_version": 1, "evidence_kind": "measured", "checkpoint_sha256": digest,
            "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "golden_inputs_sha256": hashlib.sha256(golden_inputs.read_bytes()).hexdigest(),
            "torch_version": torch.__version__, "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__, "opset": cfg.opset_version,
            "pytorch_onnx_parity_passed": True, "tensorrt_parity_passed": False,
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trusted local TorchScript and measure ONNX golden parity")
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--golden-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = export_trusted_torchscript(args.torchscript, args.sha256, args.golden_inputs, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
