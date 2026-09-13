"""Perception models, manifests, ONNX/TensorRT builders, and evaluation tooling."""

from tools.perception.eval_yolo import DetectionPrediction, FailureCase, MatchResult, PerceptionEvaluator
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
from tools.perception.onnx_exporter import (
    FrameworkParityVerifier,
    OnnxContractValidator,
    OnnxExportConfig,
    ParityVerificationResult,
    build_reference_onnx_manifest,
)

__all__ = [
    "DatasetProvenance",
    "DetectionPrediction",
    "FailureCase",
    "FrameworkParityVerifier",
    "InputTensorSpec",
    "MatchResult",
    "ModelArtifact",
    "ModelManifest",
    "OnnxContractValidator",
    "OnnxExportConfig",
    "OutputTensorSpec",
    "ParityVerificationResult",
    "PerceptionEvaluator",
    "PerceptionMetrics",
    "SliceMetric",
    "TrainingConfig",
    "build_reference_onnx_manifest",
]
