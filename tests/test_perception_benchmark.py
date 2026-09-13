"""Measured evidence regressions: fixture values must never certify a detector."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.probe.perception_benchmark import PerceptionBenchmark


def measured_fixture(tmp_path: Path) -> tuple[Path, dict]:
    target = {"class_id": 0, "bbox_xyxy": [100, 100, 140, 140], "center_px": [120, 120]}
    samples = [{"sample_id": str(i), "session_id": "test", "image_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                "scenario": "grid", "size": "small", "motion": "static", "occlusion": "visible",
                "targets": [target], "predictions": [{**target, "confidence": 1}],
                "inference_latency_ms": value} for i, value in enumerate([1, 1.5, 2])]
    data = {"schema_version": 1,
            "session_splits": {"train": ["train"], "val": ["val"], "test": ["test"]},
            "metadata": {"evidence_kind": "measured", "timing_method": "cuda_events", "gpu_device": "test fixture",
                         "model_name": "test", "architecture": "test", "dataset_revision": "revision1",
                         "dataset_license": "fixture", "model_license": "fixture", "tensorrt_version": "10",
                         "cuda_version": "12", "driver_version": "fixture", "os": "fixture", "training_config": {},
                         "training_seed": 42, "opset": 17, "concurrent_workload": True,
                         "thermal_soak_seconds": 30, "warmup_iterations": 50},
            "samples": samples, "artifacts": {}}
    dataset = {"dataset_revision": "revision1", "session_splits": data["session_splits"], "samples": samples}
    def artifact(name: str, contents: bytes) -> None:
        path = tmp_path / name
        path.write_bytes(contents)
        data["artifacts"][name] = {"path": name, "sha256": hashlib.sha256(contents).hexdigest()}
    artifact("dataset_manifest", json.dumps(dataset).encode())
    for name in ("checkpoint", "onnx", "engine"):
        artifact(name, name.encode())
    parity = {"pytorch_onnx_parity_passed": True, "tensorrt_parity_passed": True,
              "golden_inputs_sha256": "1" * 64,
              **{f"{name}_sha256": data["artifacts"][name]["sha256"] for name in ("checkpoint", "onnx", "engine")}}
    artifact("parity_report", json.dumps(parity).encode())
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, data


def test_default_requires_evidence() -> None:
    with pytest.raises(ValueError, match="Supply measured"):
        PerceptionBenchmark.run_benchmark(iterations=25)


def test_synthetic_never_passes(tmp_path: Path) -> None:
    summary = PerceptionBenchmark.run_benchmark(iterations=25, synthetic=True)
    assert summary.precision == 1  # Even perfect fixture numbers do not pass acceptance.
    assert not summary.accuracy_gate_passed
    assert not summary.latency_gate_passed
    assert not summary.milestone_acceptance_passed
    report = PerceptionBenchmark.generate_markdown_report(summary, tmp_path / "report.md")
    assert "synthetic" in report
    assert "NOT certified" in report
    assert "99.5%" in report


def test_measured_values_and_hash_provenance(tmp_path: Path) -> None:
    path, _ = measured_fixture(tmp_path)
    summary = PerceptionBenchmark.run_benchmark(evidence_path=path)
    assert summary.latency_p50_ms == 1.5
    assert summary.latency_p95_ms == pytest.approx(1.95)
    assert summary.latency_p99_ms == pytest.approx(1.99)
    assert summary.latency_max_ms == 2
    assert summary.median_center_error_px == 0
    assert summary.accuracy_gate_passed and summary.latency_gate_passed
    assert not summary.milestone_acceptance_passed
    assert len(summary.scenario_slices) == 4
    assert summary.evidence_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["session_splits"]["train"].append("test"), "leakage"),
    (lambda d: d["samples"][0].update(inference_latency_ms=float("nan")), "finite"),
    (lambda d: d["samples"][0]["predictions"][0].update(confidence=float("nan")), "finite"),
    (lambda d: d["samples"].pop(), "complete"),
    (lambda d: d["samples"][0].update(image_sha256="f" * 64), "ground truth"),
    (lambda d: d["artifacts"]["engine"].update(sha256="f" * 64), "hash mismatch"),
    (lambda d: d["metadata"].pop("training_seed"), "training_seed"),
])
def test_invalid_evidence_rejected(tmp_path: Path, mutation, match: str) -> None:
    path, data = measured_fixture(tmp_path)
    mutation(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        PerceptionBenchmark.run_benchmark(evidence_path=path)


def test_precision_threshold_and_slice_false_positives(tmp_path: Path) -> None:
    path, data = measured_fixture(tmp_path)
    false_positive = copy.deepcopy(data["samples"][0]["predictions"][0])
    false_positive["bbox_xyxy"] = [200, 200, 240, 240]
    false_positive["center_px"] = [220, 220]
    data["samples"][0]["predictions"].append(false_positive)
    data["metadata"]["concurrent_workload"] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    summary = PerceptionBenchmark.run_benchmark(evidence_path=path)
    assert summary.precision == .75
    assert summary.scenario_slices["scenario:grid"]["precision"] == .75
    assert not summary.accuracy_gate_passed and not summary.latency_gate_passed
    assert summary.failures[0]["kind"] == "false_positive"


def test_median_is_computed_not_derived_from_mean(tmp_path: Path) -> None:
    path, data = measured_fixture(tmp_path)
    for sample, error in zip(data["samples"], [0, 0, 9], strict=True):
        sample["predictions"][0]["center_px"] = [120 + error, 120]
    path.write_text(json.dumps(data), encoding="utf-8")
    summary = PerceptionBenchmark.run_benchmark(evidence_path=path)
    assert summary.mean_center_error_px == 3
    assert summary.median_center_error_px == 0
