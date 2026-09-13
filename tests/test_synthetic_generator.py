"""Unit and deterministic tests for synthetic target generator, domain randomizer, and YOLO exports."""

import json
from pathlib import Path

from tools.data.synthetic_generator import (
    SyntheticGeneratorConfig,
    SyntheticTargetGenerator,
    get_annotation_schema_path,
)


def test_annotation_schema_exists() -> None:
    """Verify that annotation_format.schema.json exists and is valid JSON."""
    schema_path = get_annotation_schema_path()
    assert schema_path.exists(), f"Schema not found: {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert schema.get("title") == "AimAgentAnnotationFormat"


def test_synthetic_target_generator_single_frame() -> None:
    """Test generating a single synthetic frame and verifying mathematical ground truth."""
    cfg = SyntheticGeneratorConfig(
        width=1920,
        height=1080,
        min_targets=3,
        max_targets=5,
        min_radius=20.0,
        max_radius=50.0,
    )
    generator = SyntheticTargetGenerator(cfg)

    img_rgb, annot = generator.generate_sample(seed=1337, sample_id="test_sample_001")

    assert img_rgb.shape == (1080, 1920, 3)
    assert annot.sample_id == "test_sample_001"
    assert annot.width == 1920
    assert annot.height == 1080
    assert 3 <= len(annot.targets) <= 5
    assert len(annot.image_sha256) == 64

    # Validate against JSON schema
    annot.validate_schema()

    # Check each target's geometric consistency
    for t in annot.targets:
        assert t.class_id == 0
        assert t.class_name == "target_sphere"
        assert 20.0 <= t.radius_px <= 50.0

        cx, cy = t.center_px
        assert 0 <= cx <= 1920
        assert 0 <= cy <= 1080

        xmin, ymin, xmax, ymax = t.bbox_xyxy
        assert xmin < xmax
        assert ymin < ymax
        assert 0 <= xmin <= 1920
        assert 0 <= ymin <= 1080
        assert 0 <= xmax <= 1920
        assert 0 <= ymax <= 1080

        # Check YOLO normalized coords
        x_norm, y_norm, w_norm, h_norm = t.bbox_yolo
        assert 0.0 <= x_norm <= 1.0
        assert 0.0 <= y_norm <= 1.0
        assert 0.0 < w_norm <= 1.0
        assert 0.0 < h_norm <= 1.0

        # Check visibility label and ratio
        assert t.visibility in ["visible", "partial", "occluded"]
        assert 0.0 <= t.visibility_ratio <= 1.0

    # Check provenance
    assert annot.provenance is not None
    assert annot.provenance.random_seed == 1337
    assert annot.provenance.background_theme in cfg.background_themes


def test_synthetic_target_generator_seed_reproducibility() -> None:
    """Verify that identical random seeds produce bitwise identical images and annotations."""
    cfg = SyntheticGeneratorConfig()
    generator = SyntheticTargetGenerator(cfg)

    img1, annot1 = generator.generate_sample(seed=424242, sample_id="synth_424242")
    img2, annot2 = generator.generate_sample(seed=424242, sample_id="synth_424242")

    assert (img1 == img2).all()
    assert annot1.image_sha256 == annot2.image_sha256
    assert annot1.provenance is not None
    assert annot2.provenance is not None
    assert annot1.provenance.random_seed == annot2.provenance.random_seed
    assert annot1.provenance.background_theme == annot2.provenance.background_theme
    assert annot1.provenance.augmentations_applied == annot2.provenance.augmentations_applied


def test_synthetic_target_generator_batch_generation(tmp_path: Path) -> None:
    """Test generating a batch dataset on disk with image, label, and annotation files."""
    cfg = SyntheticGeneratorConfig(width=640, height=384, min_targets=1, max_targets=3)
    generator = SyntheticTargetGenerator(cfg)

    out_dir = tmp_path / "synthetic_batch"
    annots = generator.generate_dataset_batch(output_dir=out_dir, num_samples=5, base_seed=100)

    assert len(annots) == 5

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    annots_dir = out_dir / "annotations"

    assert images_dir.exists()
    assert labels_dir.exists()
    assert annots_dir.exists()

    for i in range(5):
        sid = f"synth_000100_{i:06d}"
        img_file = images_dir / f"{sid}.png"
        txt_file = labels_dir / f"{sid}.txt"
        json_file = annots_dir / f"{sid}.json"

        assert img_file.exists()
        assert txt_file.exists()
        assert json_file.exists()

        # Load and validate JSON annotation
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["sample_id"] == sid

        # Check YOLO label lines
        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            assert parts[0] == "0"  # class_id
            for val in parts[1:]:
                float_val = float(val)
                assert 0.0 <= float_val <= 1.0
