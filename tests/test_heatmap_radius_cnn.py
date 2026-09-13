"""Unit tests for CenterHeatmapRadiusNet architecture, decoder, manifest, and Pareto evaluation (Milestone M9-01)."""

import json

import jsonschema
import numpy as np
import pytest

from tools.benchmark.eval_heatmap_vs_yolo_pareto import run_pareto_evaluation
from tools.perception.heatmap_radius_cnn import (
    CenterHeatmapArchitecture,
    CenterHeatmapDecoder,
    HeatmapModelConfig,
    build_heatmap_cnn_manifest,
)
from tools.perception.model_manifest import get_model_schema_path


def test_heatmap_model_config() -> None:
    """Verify default resolution and downsample stride contracts."""
    cfg = HeatmapModelConfig()
    assert cfg.input_width == 640
    assert cfg.input_height == 384
    assert cfg.input_channels == 3
    assert cfg.downsample_stride == 4
    assert cfg.output_grid_width == 160
    assert cfg.output_grid_height == 96
    assert cfg.max_targets == 64
    assert cfg.conf_threshold == 0.30


def test_architecture_specs_and_parameter_accounting() -> None:
    """Verify analytical layer specifications, parameter count, and FLOP calculation."""
    specs = CenterHeatmapArchitecture.get_layer_specs()
    assert len(specs) == 14

    summary = CenterHeatmapArchitecture.compute_summary_metrics()
    assert summary["model_name"] == "CenterHeatmapRadiusNet-v1"
    assert summary["license"] == "PolyForm-Noncommercial-1.0.0"
    assert 50_000 < summary["parameter_count"] < 150_000
    assert summary["gflops"] < 3.0
    assert summary["downsample_stride"] == 4


def test_center_heatmap_decoder_peak_detection() -> None:
    """Verify 3x3 local maximum suppression and subpixel center reconstruction."""
    cfg = HeatmapModelConfig()
    decoder = CenterHeatmapDecoder(cfg)

    # Create dummy 96x160 heatmap with one distinct peak at (gx=40, gy=30)
    hm = np.zeros((96, 160), dtype=np.float32)
    hm[30, 40] = 0.95
    hm[29:32, 39:42] = np.array(
        [[0.4, 0.6, 0.4], [0.6, 0.95, 0.6], [0.4, 0.6, 0.4]], dtype=np.float32
    )

    # Subpixel offset dx = 0.25, dy = -0.15
    off = np.zeros((2, 96, 160), dtype=np.float32)
    off[0, 30, 40] = 0.25
    off[1, 30, 40] = -0.15

    # Radius rx = 8.0, ry = 8.0 (in 640x384 grid space)
    rad = np.full((2, 96, 160), 8.0, dtype=np.float32)

    preds = decoder.decode_predictions(
        hm, rad, off, sample_id="sample_test_001"
    )
    assert len(preds) == 1

    p = preds[0]
    assert p.sample_id == "sample_test_001"
    assert p.confidence == pytest.approx(0.95, abs=1e-3)

    # Expected:
    # cx_640 = (40 + 0.25 + 0.5) * 4 = 40.75 * 4 = 163.0
    # cy_384 = (30 - 0.15 + 0.5) * 4 = 30.35 * 4 = 121.4
    # cx_1920 = 163.0 * (1920 / 640) = 489.0
    # cy_1080 = 121.4 * (1080 / 384) = 341.4375
    assert p.center_px[0] == pytest.approx(489.0, abs=0.1)
    assert p.center_px[1] == pytest.approx(341.4375, abs=0.1)

    # Radius in 1920x1080 space: 8.0 * 3.0 = 24.0
    assert p.radius_px == pytest.approx(24.0, abs=0.5)


def test_decoder_suppresses_below_threshold() -> None:
    """Verify detections below confidence threshold are discarded."""
    cfg = HeatmapModelConfig(conf_threshold=0.50)
    decoder = CenterHeatmapDecoder(cfg)

    hm = np.zeros((96, 160), dtype=np.float32)
    hm[20, 20] = 0.45  # Below 0.50 threshold
    rad = np.zeros((2, 96, 160), dtype=np.float32)
    off = np.zeros((2, 96, 160), dtype=np.float32)

    preds = decoder.decode_predictions(hm, rad, off)
    assert len(preds) == 0


def test_manifest_schema_validation() -> None:
    """Validate CenterHeatmapRadiusNet manifest against model_manifest schema."""
    manifest = build_heatmap_cnn_manifest()
    manifest_dict = manifest.to_dict()

    schema_path = get_model_schema_path()
    assert schema_path.is_file(), f"Schema file not found at {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Should validate with zero errors
    jsonschema.validate(instance=manifest_dict, schema=schema)
    assert manifest.architecture == "custom_cnn"
    assert manifest.license == "PolyForm-Noncommercial-1.0.0"
    assert manifest.input_tensor.shape == [1, 3, 384, 640]


def test_pareto_benchmark_execution() -> None:
    """Verify fast Pareto evaluation execution and contract consistency."""
    result = run_pareto_evaluation(samples=20, seed=123)

    assert result.eval_samples == 20
    assert "yolo11n_fp16" in result.models
    assert "center_heatmap_net" in result.models

    yolo = result.models["yolo11n_fp16"]
    hm = result.models["center_heatmap_net"]

    assert yolo.meets_acceptance_criteria is True
    assert hm.meets_acceptance_criteria is True

    # Heatmap model must be faster and smaller
    assert hm.parameter_count < yolo.parameter_count
    assert hm.latency_p99_ms < yolo.latency_p99_ms
    assert result.triggering_gap_active is False
    assert "RETAIN_YOLO11N" in result.decision
