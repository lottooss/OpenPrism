"""Evaluate supplied detector measurements; synthetic fixtures never certify acceptance.

Input schema_version 1: metadata, session_splits (train/val/test session IDs),
artifacts ({name: {path, sha256}}), and samples. Each sample has sample_id,
session_id, image_sha256, scenario/size/motion/occlusion slice labels,
inference_latency_ms, targets and predictions. Boxes use 1920x1080 pixel xyxy;
targets/predictions contain class_id, bbox_xyxy and center_px; predictions also
contain confidence. Latency must come from CUDA events after declared warmup.
Artifact paths are relative to the evidence JSON, and are hashed before use.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.perception.eval_yolo import PerceptionEvaluator


@dataclass
class PerceptionBenchmarkSummary:
    timestamp_utc: str
    evidence_kind: str
    gpu_device: str
    model_name: str
    architecture: str
    precision: float
    recall: float
    f1_score: float
    mean_center_error_px: float
    median_center_error_px: float | None
    p95_center_error_px: float | None
    max_center_error_px: float | None
    expected_calibration_error: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    target_p99_ms: float = 2.50
    accuracy_gate_passed: bool = False
    latency_gate_passed: bool = False
    total_eval_samples: int = 0
    scenario_slices: dict[str, dict[str, float | None]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_sha256: str = ""
    # Benchmark numbers alone cannot certify parity, allocation/fault tests or the live milestone.
    milestone_acceptance_passed: bool = False


def _finite(value: Any, name: str, lower: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < lower:
        raise ValueError(f"{name} must be finite and >= {lower}")
    return result


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _box(obj: dict[str, Any], prediction: bool = False) -> None:
    if type(obj.get("class_id")) is not int or obj["class_id"] < 0:
        raise ValueError("Invalid class_id")
    box, center = obj.get("bbox_xyxy"), obj.get("center_px")
    if not isinstance(box, list) or len(box) != 4 or not isinstance(center, list) or len(center) != 2:
        raise ValueError("Expected pixel xyxy box and xy center")
    for number in box + center:
        _finite(number, "box coordinate")
    if not (box[0] < box[2] <= 1920 and box[1] < box[3] <= 1080 and
            box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]):
        raise ValueError("Invalid box/center geometry")
    if prediction and _finite(obj.get("confidence"), "confidence") > 1:
        raise ValueError("Confidence exceeds one")


def _validate(data: dict[str, Any], base: Path, synthetic: bool) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("Expected evidence schema_version 1")
    splits = data.get("session_splits", {})
    sets: list[set[str]] = []
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"Missing session split {name}")
        if len(set(values)) != len(values):
            raise ValueError("Duplicate session within split")
        sets.append(set(values))
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Session leakage across train/val/test")
    metadata = data.get("metadata", {})
    if not synthetic:
        if metadata.get("evidence_kind") != "measured" or metadata.get("timing_method") != "cuda_events":
            raise ValueError("Measured CUDA-event evidence required; use --synthetic only for fixtures")
        for key in ("gpu_device", "model_name", "architecture", "dataset_revision", "dataset_license",
                    "model_license", "tensorrt_version", "cuda_version", "driver_version", "os",
                    "training_config", "training_seed", "opset", "concurrent_workload", "thermal_soak_seconds"):
            if key not in metadata or metadata[key] == "" or metadata[key] is None:
                raise ValueError(f"Missing evidence metadata: {key}")
        if type(metadata.get("warmup_iterations")) is not int or metadata["warmup_iterations"] <= 0:
            raise ValueError("Positive measured warmup_iterations required")
        if type(metadata["training_seed"]) is not int or not isinstance(metadata["training_config"], dict):
            raise ValueError("Training seed/config must be recorded")
        if type(metadata["concurrent_workload"]) is not bool:
            raise ValueError("concurrent_workload must be boolean")
        _finite(metadata["thermal_soak_seconds"], "thermal_soak_seconds")
        for name in ("dataset_manifest", "checkpoint", "onnx", "engine", "parity_report"):
            artifact = data.get("artifacts", {}).get(name, {})
            if not _hash(artifact.get("sha256")) or not artifact.get("path"):
                raise ValueError(f"Missing hashed artifact {name}")
            path = base / artifact["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError(f"Artifact missing or hash mismatch: {name}")
        manifest_path = base / data["artifacts"]["dataset_manifest"]["path"]
        dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
        if dataset.get("session_splits") != splits or dataset.get("dataset_revision") != metadata["dataset_revision"]:
            raise ValueError("Dataset revision/session split provenance mismatch")
        dataset_samples = dataset.get("samples", [])
        all_hashes: set[str] = set()
        all_ids: set[str] = set()
        held_out: dict[str, dict[str, Any]] = {}
        for sample in dataset_samples:
            sid, image_hash = sample.get("sample_id"), sample.get("image_sha256")
            if (not isinstance(sid, str) or not sid or sid in all_ids or
                    not _hash(image_hash) or image_hash in all_hashes):
                raise ValueError("Dataset sample/image overlap or missing identity")
            all_ids.add(sid)
            all_hashes.add(image_hash)
            if sample.get("session_id") not in set.union(*sets):
                raise ValueError("Dataset sample lacks a declared session")
            if sample["session_id"] in sets[2]:
                held_out[sid] = sample
        measured = data.get("samples", [])
        if not held_out or {s.get("sample_id") for s in measured} != set(held_out):
            raise ValueError("Evaluation must cover the complete declared held-out dataset")
        for sample in measured:
            ground_truth = held_out[sample["sample_id"]]
            for key in ("session_id", "image_sha256", "targets", "scenario", "size", "motion", "occlusion"):
                if sample.get(key) != ground_truth.get(key):
                    raise ValueError(f"Evaluation ground truth differs from dataset manifest: {key}")
        parity_path = base / data["artifacts"]["parity_report"]["path"]
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if (parity.get("pytorch_onnx_parity_passed") is not True or
                parity.get("tensorrt_parity_passed") is not True or
                not _hash(parity.get("golden_inputs_sha256"))):
            raise ValueError("Measured PyTorch/ONNX/TensorRT golden parity report required")
        for name in ("checkpoint", "onnx", "engine"):
            if parity.get(f"{name}_sha256") != data["artifacts"][name]["sha256"]:
                raise ValueError("Parity was measured against different model artifacts")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("No measured samples")
    ids: set[str] = set()
    hashes: set[str] = set()
    for sample in samples:
        sid = sample.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("Missing/duplicate sample ID")
        ids.add(sid)
        if sample.get("session_id") not in sets[2]:
            raise ValueError("Evaluation sample is not in the held-out test sessions")
        image_hash = sample.get("image_sha256")
        if not _hash(image_hash) or image_hash in hashes:
            raise ValueError("Missing/duplicate evaluation image hash")
        hashes.add(image_hash)
        if _finite(sample.get("inference_latency_ms"), "inference latency") <= 0:
            raise ValueError("Latency must be positive")
        for dimension in ("scenario", "size", "motion", "occlusion"):
            if not isinstance(sample.get(dimension), str) or not sample[dimension]:
                raise ValueError(f"Missing slice {dimension}")
        for key in ("targets", "predictions"):
            if not isinstance(sample.get(key), list):
                raise ValueError(f"Missing {key}")
            for obj in sample[key]:
                _box(obj, key == "predictions")


def _evaluate(samples: list[dict[str, Any]]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    true_positive = 0
    false_positive = 0
    target_count = 0
    errors: list[float] = []
    confidences: list[float] = []
    correct: list[int] = []
    failures: list[dict[str, Any]] = []
    for sample in samples:
        targets = sample["targets"]
        target_count += len(targets)
        matched: set[int] = set()
        for pred in sorted(sample["predictions"], key=lambda p: p["confidence"], reverse=True):
            if pred["confidence"] < 0.25:
                continue
            candidates = [(PerceptionEvaluator.compute_iou(pred["bbox_xyxy"], target["bbox_xyxy"]), i)
                          for i, target in enumerate(targets)
                          if i not in matched and target["class_id"] == pred["class_id"]]
            iou, idx = max(candidates, default=(0.0, -1))
            is_match = iou >= 0.5 and idx >= 0
            confidences.append(pred["confidence"])
            correct.append(int(is_match))
            if is_match:
                matched.add(idx)
                true_positive += 1
                error = math.dist(pred["center_px"], targets[idx]["center_px"])
                errors.append(error)
                if error > 2.0:
                    failures.append({"sample_id": sample["sample_id"], "kind": "center_error", "error_px": error})
            else:
                false_positive += 1
                failures.append({"sample_id": sample["sample_id"], "kind": "false_positive", "prediction": pred})
        for i, target in enumerate(targets):
            if i not in matched:
                failures.append({"sample_id": sample["sample_id"], "kind": "false_negative", "target": target})
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(target_count, 1)
    return {
        "sample_count": float(len(samples)), "precision": precision, "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_center_error_px": float(np.mean(errors)) if errors else 0.0,
        "median_center_error_px": float(np.median(errors)) if errors else None,
        "p95_center_error_px": float(np.percentile(errors, 95)) if errors else None,
        "max_center_error_px": max(errors) if errors else None,
        "expected_calibration_error": PerceptionEvaluator.compute_ece(confidences, correct),
    }, failures


class PerceptionBenchmark:
    @classmethod
    def run_benchmark(cls, iterations: int = 1000, *, evidence_path: Path | str | None = None,
                      synthetic: bool = False) -> PerceptionBenchmarkSummary:
        if evidence_path is None and not synthetic:
            raise ValueError("Supply measured --evidence JSON or explicitly select --synthetic")
        if evidence_path is not None and synthetic:
            raise ValueError("Measured evidence and synthetic mode are mutually exclusive")
        evidence_hash = ""
        if synthetic:
            if iterations <= 0:
                raise ValueError("iterations must be positive")
            target = {"class_id": 0, "bbox_xyxy": [100.0, 100.0, 120.0, 120.0], "center_px": [110.0, 110.0]}
            data: dict[str, Any] = {
                "schema_version": 1,
                "metadata": {"evidence_kind": "synthetic", "gpu_device": "none (fixture)",
                             "model_name": "fixture", "architecture": "none", "warmup_iterations": 0},
                "session_splits": {"train": ["fixture_train"], "val": ["fixture_val"], "test": ["fixture_test"]},
                "samples": [{"sample_id": str(i), "session_id": "fixture_test",
                             "image_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                             "scenario": "fixture", "size": "fixture", "motion": "fixture", "occlusion": "fixture",
                             "targets": [target], "predictions": [{**target, "confidence": 1.0}],
                             "inference_latency_ms": 1.0} for i in range(iterations)],
            }
            base = Path.cwd()
        else:
            path = Path(str(evidence_path))
            raw = path.read_bytes()
            evidence_hash = hashlib.sha256(raw).hexdigest()
            data = json.loads(raw)
            base = path.parent
        _validate(data, base, synthetic)
        samples, metadata = data["samples"], data["metadata"]
        metrics, failures = _evaluate(samples)
        slices: dict[str, dict[str, float | None]] = {}
        for dimension in ("scenario", "size", "motion", "occlusion"):
            for label in sorted({s[dimension] for s in samples}):
                slices[f"{dimension}:{label}"] = _evaluate([s for s in samples if s[dimension] == label])[0]
        latency = [s["inference_latency_ms"] for s in samples]
        p50, p95, p99 = (float(np.percentile(latency, q)) for q in (50, 95, 99))
        median = metrics["median_center_error_px"]
        return PerceptionBenchmarkSummary(
            timestamp_utc=datetime.now(timezone.utc).isoformat(), evidence_kind="synthetic" if synthetic else "measured",
            gpu_device=metadata["gpu_device"], model_name=metadata["model_name"], architecture=metadata["architecture"],
            precision=float(metrics["precision"] or 0), recall=float(metrics["recall"] or 0),
            f1_score=float(metrics["f1_score"] or 0), mean_center_error_px=float(metrics["mean_center_error_px"] or 0),
            median_center_error_px=median, p95_center_error_px=metrics["p95_center_error_px"],
            max_center_error_px=metrics["max_center_error_px"],
            expected_calibration_error=float(metrics["expected_calibration_error"] or 0),
            latency_p50_ms=p50, latency_p95_ms=p95, latency_p99_ms=p99, latency_max_ms=max(latency),
            accuracy_gate_passed=not synthetic and float(metrics["precision"] or 0) >= 0.995 and
                float(metrics["recall"] or 0) >= 0.99 and median is not None and median <= 2.0,
            latency_gate_passed=not synthetic and p99 <= 2.5 and metadata["concurrent_workload"] is True,
            total_eval_samples=len(samples), scenario_slices=slices, failures=failures,
            metadata=metadata, evidence_sha256=evidence_hash,
        )

    @classmethod
    def generate_markdown_report(cls, summary: PerceptionBenchmarkSummary,
                                 output_path: Path | str | None = None) -> str:
        median = "unavailable" if summary.median_center_error_px is None else f"{summary.median_center_error_px:.3f} px"
        lines = ["# Perception Evidence Report", "", f"Evidence: **{summary.evidence_kind}**.",
                 f"Model: {summary.model_name}. GPU: {summary.gpu_device}. Samples: {summary.total_eval_samples}.",
                 "Synthetic values are fixture data and cannot establish acceptance." if summary.evidence_kind == "synthetic"
                 else "Statistics computed from supplied measured records; artifact hashes verified. Hardware provenance still requires review.",
                 "", "| Metric | Threshold | Result |", "|---|---|---|",
                 f"| Recall on Held-Out Set | >=99.0% | {summary.recall:.4%} |",
                 f"| Precision on Held-Out Set | >=99.5% | {summary.precision:.4%} |",
                 f"| Median Center Error | <=2 px | {median} |",
                 f"| Inference Latency (p99) | <=2.5 ms | {summary.latency_p99_ms:.4f} ms |",
                 f"| ECE | reported | {summary.expected_calibration_error:.5f} |", "",
                 f"Latency p50/p95/p99/max (ms): {summary.latency_p50_ms:.4f} / {summary.latency_p95_ms:.4f} / "
                 f"{summary.latency_p99_ms:.4f} / {summary.latency_max_ms:.4f}.",
                 f"Accuracy numeric gate: {summary.accuracy_gate_passed}. Concurrent latency gate: {summary.latency_gate_passed}.",
                 f"Failure records: {len(summary.failures)}; complete failures, metadata and slices are in the JSON.", "",
                 "**Milestone acceptance is NOT certified by this report.** Review framework parity, real held-out dataset provenance, "
                 "concurrent-load measurements, allocation instrumentation and fault/lifetime tests separately."]
        report = "\n".join(lines) + "\n"
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evidence", type=Path)
    mode.add_argument("--synthetic", action="store_true")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    summary = PerceptionBenchmark.run_benchmark(args.iterations, evidence_path=args.evidence, synthetic=args.synthetic)
    PerceptionBenchmark.generate_markdown_report(summary, args.output_md)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(summary), indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
