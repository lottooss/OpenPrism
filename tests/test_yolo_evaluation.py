"""Unit and adversarial tests for YOLO perception evaluation, center error, ECE calibration, and failure gallery."""

from pathlib import Path

from tools.data.synthetic_generator import (
    FrameAnnotation,
    ProvenanceInfo,
    TargetAnnotation,
)
from tools.perception.eval_yolo import (
    DetectionPrediction,
    PerceptionEvaluator,
)
from tools.perception.model_manifest import (
    DatasetProvenance,
    InputTensorSpec,
    ModelManifest,
    OutputTensorSpec,
    TrainingConfig,
)


def test_compute_iou() -> None:
    """Test standard intersection-over-union math."""
    box_a = [100.0, 100.0, 200.0, 200.0]  # area 10000
    box_b = [150.0, 100.0, 250.0, 200.0]  # area 10000, intersection 50 * 100 = 5000, union = 15000 -> 1/3

    iou = PerceptionEvaluator.compute_iou(box_a, box_b)
    assert abs(iou - (1.0 / 3.0)) < 1e-4

    # Disjoint boxes
    box_c = [300.0, 300.0, 400.0, 400.0]
    assert PerceptionEvaluator.compute_iou(box_a, box_c) == 0.0


def test_compute_ece() -> None:
    """Test Expected Calibration Error calculation."""
    confidences = [0.9, 0.85, 0.8, 0.75, 0.2, 0.1]
    accuracies = [1, 1, 1, 0, 0, 0]

    ece = PerceptionEvaluator.compute_ece(confidences, accuracies, num_bins=5)
    assert 0.0 <= ece <= 1.0


def test_evaluate_detections_metrics_and_slices(tmp_path: Path) -> None:
    """Test end-to-end detection evaluation with ground truth annotations."""
    # 2 Sample frames
    annot1 = FrameAnnotation(
        schema_version=1,
        sample_id="sample_01",
        image_path="sample_01.png",
        image_sha256="a" * 64,
        width=1920,
        height=1080,
        targets=[
            TargetAnnotation(
                target_id=1,
                class_id=0,
                class_name="target_sphere",
                center_px=[960.0, 540.0],
                radius_px=30.0,
                bbox_xyxy=[930.0, 510.0, 990.0, 570.0],
                bbox_yolo=[0.5, 0.5, 60.0 / 1920.0, 60.0 / 1080.0],
                visibility="visible",
                visibility_ratio=1.0,
            )
        ],
        provenance=ProvenanceInfo(
            generator_name="synthetic_target_generator",
            random_seed=123,
            background_theme="aimlabs_grid",
            augmentations_applied=[],
            created_at_utc="2026-08-30T10:00:00Z",
        ),
    )

    annot2 = FrameAnnotation(
        schema_version=1,
        sample_id="sample_02",
        image_path="sample_02.png",
        image_sha256="b" * 64,
        width=1920,
        height=1080,
        targets=[
            TargetAnnotation(
                target_id=2,
                class_id=0,
                class_name="target_sphere",
                center_px=[400.0, 300.0],
                radius_px=25.0,
                bbox_xyxy=[375.0, 275.0, 425.0, 325.0],
                bbox_yolo=[400.0 / 1920.0, 300.0 / 1080.0, 50.0 / 1920.0, 50.0 / 1080.0],
                visibility="visible",
                visibility_ratio=1.0,
            )
        ],
        provenance=ProvenanceInfo(
            generator_name="synthetic_target_generator",
            random_seed=456,
            background_theme="dark_mode",
            augmentations_applied=[],
            created_at_utc="2026-08-30T10:00:00Z",
        ),
    )

    predictions = [
        # Match for sample 1 (slight 1.0px center error)
        DetectionPrediction(
            sample_id="sample_01",
            class_id=0,
            confidence=0.95,
            bbox_xyxy=[931.0, 510.0, 991.0, 570.0],
            center_px=[961.0, 540.0],
            radius_px=30.0,
        ),
        # Match for sample 2 (exact center)
        DetectionPrediction(
            sample_id="sample_02",
            class_id=0,
            confidence=0.92,
            bbox_xyxy=[375.0, 275.0, 425.0, 325.0],
            center_px=[400.0, 300.0],
            radius_px=25.0,
        ),
        # 1 False Positive
        DetectionPrediction(
            sample_id="sample_02",
            class_id=0,
            confidence=0.40,
            bbox_xyxy=[1500.0, 800.0, 1550.0, 850.0],
            center_px=[1525.0, 825.0],
            radius_px=25.0,
        ),
    ]

    metrics, failure_cases, slices = PerceptionEvaluator.evaluate_detections(
        annotations=[annot1, annot2],
        predictions=predictions,
        iou_threshold=0.50,
        conf_threshold=0.25,
    )

    # Both true positives precede the low-confidence false positive: integrated AP is 1,
    # while single-threshold precision is 2/3. AP must not be their product.
    assert metrics.map50 == 1.0
    assert metrics.map50_95 == 1.0
    assert slices["dark_mode"].precision == 0.5
    assert metrics.precision == 2.0 / 3.0  # 2 TP, 1 FP
    assert metrics.recall == 1.0           # 2 TP / 2 GT
    assert 0.0 <= metrics.mean_center_error_px <= 2.0
    assert 0.0 <= metrics.expected_calibration_error <= 1.0

    assert "aimlabs_grid" in slices
    assert "dark_mode" in slices
    assert len(failure_cases) == 1  # 1 false positive

    # Test report generation
    manifest = ModelManifest(
        schema_version=1,
        model_id="33333333-4444-4555-8666-777777777777",
        model_name="test_detector",
        architecture="yolo11n",
        version="1.0.0",
        license="AGPL-3.0",
        input_tensor=InputTensorSpec(),
        output_tensor=OutputTensorSpec(),
        dataset_provenance=DatasetProvenance(
            dataset_id="44444444-5555-4666-8777-888888888888",
            dataset_name="test_dataset",
            dataset_sha256="d" * 64,
            train_samples=2,
            val_samples=1,
            test_samples=1,
        ),
        training_config=TrainingConfig(),
        metrics=metrics,
        scenario_slices=slices,
        model_artifacts=[],
    )

    out_md = tmp_path / "eval_report.md"
    report_md = PerceptionEvaluator.generate_evaluation_report(
        manifest=manifest,
        failure_cases=failure_cases,
        output_md_path=out_md,
    )

    assert out_md.exists()
    assert "# Perception Model Evaluation Report:" in report_md
    assert "Precision" in report_md
    assert "Scenario Slice Breakdown" in report_md
