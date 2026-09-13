"""Unit and adversarial tests for annotation verification, geometric auditing, and visual overlay generator."""

from pathlib import Path
import numpy as np
import pytest

from tools.data.annotation_verifier import AnnotationVerifier
from tools.data.synthetic_generator import (
    FrameAnnotation,
    IgnoreRegion,
    SyntheticGeneratorConfig,
    SyntheticTargetGenerator,
    TargetAnnotation,
    save_png,
)


@pytest.fixture
def valid_target() -> TargetAnnotation:
    return TargetAnnotation(
        target_id=0,
        class_id=0,
        class_name="target_sphere",
        center_px=[960.0, 540.0],
        radius_px=30.0,
        bbox_xyxy=[930.0, 510.0, 990.0, 570.0],
        bbox_yolo=[0.5, 0.5, 60.0 / 1920.0, 60.0 / 1080.0],
        visibility="visible",
        visibility_ratio=1.0,
    )


def test_verify_target_valid(valid_target: TargetAnnotation) -> None:
    """Verify that geometrically consistent targets pass validation with zero errors."""
    errs = AnnotationVerifier.verify_target(
        target=valid_target, frame_width=1920, frame_height=1080, sample_id="sample_01"
    )
    assert len(errs) == 0


def test_verify_target_center_out_of_bounds(valid_target: TargetAnnotation) -> None:
    """Detect target centers placed outside frame boundaries."""
    valid_target.center_px = [2500.0, 540.0]
    errs = AnnotationVerifier.verify_target(
        target=valid_target, frame_width=1920, frame_height=1080
    )
    assert len(errs) > 0
    assert any(e.error_type == "center_out_of_bounds" for e in errs)


def test_verify_target_invalid_radius(valid_target: TargetAnnotation) -> None:
    """Detect negative or zero radii."""
    valid_target.radius_px = -5.0
    errs = AnnotationVerifier.verify_target(
        target=valid_target, frame_width=1920, frame_height=1080
    )
    assert len(errs) > 0
    assert any(e.error_type == "invalid_radius" for e in errs)


def test_verify_target_degenerate_bbox(valid_target: TargetAnnotation) -> None:
    """Detect inverted or degenerate bounding boxes."""
    valid_target.bbox_xyxy = [1000.0, 500.0, 900.0, 600.0]  # xmin > xmax
    errs = AnnotationVerifier.verify_target(
        target=valid_target, frame_width=1920, frame_height=1080
    )
    assert len(errs) > 0
    assert any(e.error_type == "degenerate_bbox" for e in errs)


def test_verify_target_invalid_yolo_normalization(valid_target: TargetAnnotation) -> None:
    """Detect YOLO normalized boxes exceeding [0, 1]."""
    valid_target.bbox_yolo = [1.5, 0.5, 0.2, 0.2]  # xc > 1.0
    errs = AnnotationVerifier.verify_target(
        target=valid_target, frame_width=1920, frame_height=1080
    )
    assert len(errs) > 0
    assert any(e.error_type == "invalid_yolo_normalization" for e in errs)


def test_verify_target_invalid_visibility_ratio(valid_target: TargetAnnotation) -> None:
    """Detect visibility ratios out of range [0, 1]."""
    valid_target.visibility_ratio = 1.25
    errs = AnnotationVerifier.verify_target(
        target=valid_target, frame_width=1920, frame_height=1080
    )
    assert len(errs) > 0
    assert any(e.error_type == "invalid_visibility_ratio" for e in errs)


def test_render_visual_audit_overlay(valid_target: TargetAnnotation, tmp_path: Path) -> None:
    """Test generating visual audit overlay image."""
    base_img = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    annotation = FrameAnnotation(
        schema_version=1,
        sample_id="test_vis",
        image_path="test_vis.png",
        image_sha256="a" * 64,
        width=1920,
        height=1080,
        targets=[valid_target],
        ignore_regions=[
            IgnoreRegion(
                region_id="ch",
                region_type="crosshair",
                bbox_xyxy=[955.0, 535.0, 965.0, 545.0],
            )
        ],
    )

    overlay = AnnotationVerifier.render_visual_audit_overlay(base_img, annotation)
    assert overlay.shape == (1080, 1920, 3)

    out_file = tmp_path / "audit_overlay.png"
    save_png(out_file, overlay)
    assert out_file.exists()


def test_audit_dataset_directory(tmp_path: Path) -> None:
    """Test end-to-end directory auditing on a generated dataset batch."""
    cfg = SyntheticGeneratorConfig(width=640, height=384, min_targets=2, max_targets=4)
    generator = SyntheticTargetGenerator(cfg)

    dataset_dir = tmp_path / "test_audit_dataset"
    generator.generate_dataset_batch(output_dir=dataset_dir, num_samples=5, base_seed=777)

    report = AnnotationVerifier.audit_dataset_directory(dataset_dir)
    assert report.is_valid
    assert report.total_samples == 5
    assert report.total_targets >= 10
    assert report.valid_targets > 0
    assert len(report.errors) == 0
