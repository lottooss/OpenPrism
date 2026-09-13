"""Unit tests for TensorRT INT8 calibrator, dynamic range scaling, cache serialization, and evaluation (Milestone M9-02)."""

import numpy as np
import pytest

from tools.benchmark.eval_int8_vs_fp16_benchmark import run_int8_evaluation
from tools.perception.int8_calibrator import (
    Int8CalibrationConfig,
    RepresentativeCalibrationSet,
    TensorDynamicRange,
    TensorRtCalibrationCache,
)


def test_int8_calibration_config() -> None:
    """Verify default calibration configuration values and limits."""
    cfg = Int8CalibrationConfig()
    assert cfg.dataset_id == "aimlabs_int8_calibration_v1"
    assert cfg.calibration_samples == 500
    assert cfg.batch_size == 8
    assert cfg.input_shape == (1, 3, 384, 640)
    assert cfg.algorithm == "EntropyCalibration2"
    assert cfg.max_precision_loss_pp == 0.25
    assert cfg.max_recall_loss_pp == 0.25


def test_dynamic_range_quantize_dequantize() -> None:
    """Verify symmetric uniform INT8 quantization, bounds [-127, 127], and numerical recovery."""
    dr = TensorDynamicRange(
        tensor_name="test_activation",
        dynamic_range_min=-10.0,
        dynamic_range_max=10.0,
        scale=10.0 / 127.0,
    )

    # Values within dynamic range
    test_vals = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    q = dr.quantize(test_vals)
    assert q.dtype == np.int8
    assert q[0] == -127
    assert q[2] == 0
    assert q[4] == 127

    recovered = dr.dequantize(q)
    assert np.allclose(test_vals, recovered, atol=0.1)

    # Clamping behavior for out-of-range activations
    overflow_vals = np.array([-50.0, 50.0], dtype=np.float32)
    q_clamped = dr.quantize(overflow_vals)
    assert q_clamped[0] == -127
    assert q_clamped[1] == 127


def test_calibration_set_generation() -> None:
    """Verify calibration frame generation produces normalized 384x640 NCHW frames."""
    cfg = Int8CalibrationConfig(calibration_samples=4)
    calib_set = RepresentativeCalibrationSet(cfg)

    frames, sample_ids = calib_set.generate_calibration_batch(seed=99)
    assert frames.shape == (4, 3, 384, 640)
    assert frames.dtype == np.float32
    assert np.all(frames >= 0.0) and np.all(frames <= 1.0)
    assert len(sample_ids) == 4
    assert sample_ids[0] == "calib_0000"

    manifest = calib_set.compute_calibration_manifest(frames, sample_ids)
    assert manifest.sample_count == 4
    assert len(manifest.dataset_sha256) == 64
    assert "images" in manifest.dynamic_ranges
    assert "output0" in manifest.dynamic_ranges


def test_tensorrt_calibration_cache_serialization_roundtrip() -> None:
    """Verify TensorRT ASCII calibration cache formatting and IEEE-754 hex round-trip."""
    dr_dict = {
        "images": {"scale": 0.0078740157},
        "conv1_output": {"scale": 0.0452189},
        "output0": {"scale": 0.0625},
    }

    cache_str = TensorRtCalibrationCache.serialize_cache(dr_dict)
    assert cache_str.startswith("TRT-8600-EntropyCalibration2")
    assert "images:" in cache_str
    assert "output0:" in cache_str

    deserialized = TensorRtCalibrationCache.deserialize_cache(cache_str)
    assert "images" in deserialized
    assert "conv1_output" in deserialized
    assert "output0" in deserialized

    assert deserialized["images"] == pytest.approx(0.0078740157, rel=1e-4)
    assert deserialized["output0"] == pytest.approx(0.0625, rel=1e-5)


def test_tensorrt_cache_invalid_header_fails_closed() -> None:
    """Detect and reject malformed calibration cache headers."""
    invalid_cache = "CORRUPT_HEADER_V1\nimages: 3c010203\n"
    with pytest.raises(ValueError, match="Invalid TensorRT calibration cache header"):
        TensorRtCalibrationCache.deserialize_cache(invalid_cache)


def test_int8_vs_fp16_benchmark_execution() -> None:
    """Verify execution of full INT8 vs FP16 evaluation suite."""
    result = run_int8_evaluation(samples=20, calib_samples=4, seed=42)
    assert result.eval_samples == 20
    assert "fp16" in result.profiles
    assert "int8_calibrated" in result.profiles

    fp16 = result.profiles["fp16"]
    int8 = result.profiles["int8_calibrated"]

    assert fp16.meets_acceptance_criteria is True
    assert int8.meets_acceptance_criteria is True
    assert int8.latency_p99_ms < fp16.latency_p99_ms
    assert result.triggering_bottleneck_active is False
    assert "RETAIN_FP16_REFERENCE_PROFILE" in result.decision
