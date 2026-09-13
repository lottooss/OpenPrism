"""Annotation verification tool, geometric consistency auditing, and visual overlay generator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
import math
from pathlib import Path
import random

import jsonschema
import numpy as np

from tools.data.synthetic_generator import (
    FrameAnnotation,
    IgnoreRegion,
    TargetAnnotation,
    save_png,
)


@dataclass
class TargetAuditError:
    sample_id: str
    target_id: int
    error_type: str
    message: str


@dataclass
class DatasetAuditReport:
    is_valid: bool
    total_samples: int = 0
    total_targets: int = 0
    valid_targets: int = 0
    occluded_targets: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    target_errors: list[TargetAuditError] = field(default_factory=list)
    class_distribution: dict[int, int] = field(default_factory=dict)


class AnnotationVerifier:
    """Verifies geometric validity and creates visual audit overlays for bounding boxes and circle targets."""

    @staticmethod
    def verify_target(
        target: TargetAnnotation,
        frame_width: int,
        frame_height: int,
        sample_id: str = "",
    ) -> list[TargetAuditError]:
        """Audit geometric consistency of a single target annotation."""
        errors: list[TargetAuditError] = []

        # 1. Check center bounds
        cx, cy = target.center_px
        if not (0 <= cx <= frame_width) or not (0 <= cy <= frame_height):
            errors.append(
                TargetAuditError(
                    sample_id=sample_id,
                    target_id=target.target_id,
                    error_type="center_out_of_bounds",
                    message=f"Center ({cx}, {cy}) outside frame boundaries ({frame_width}x{frame_height})",
                )
            )

        # 2. Check radius
        if target.radius_px <= 0:
            errors.append(
                TargetAuditError(
                    sample_id=sample_id,
                    target_id=target.target_id,
                    error_type="invalid_radius",
                    message=f"Radius {target.radius_px} must be strictly positive",
                )
            )

        # 3. Check bounding box coordinates
        xmin, ymin, xmax, ymax = target.bbox_xyxy
        if xmin < 0 or ymin < 0 or xmax > frame_width or ymax > frame_height:
            errors.append(
                TargetAuditError(
                    sample_id=sample_id,
                    target_id=target.target_id,
                    error_type="bbox_out_of_bounds",
                    message=f"Bounding box [{xmin}, {ymin}, {xmax}, {ymax}] out of frame ({frame_width}x{frame_height})",
                )
            )

        if xmin >= xmax or ymin >= ymax:
            errors.append(
                TargetAuditError(
                    sample_id=sample_id,
                    target_id=target.target_id,
                    error_type="degenerate_bbox",
                    message=f"Bounding box degenerate: width={xmax - xmin}, height={ymax - ymin}",
                )
            )

        # 4. Check YOLO normalized bounds
        xc, yc, w, h = target.bbox_yolo
        if not (0.0 <= xc <= 1.0) or not (0.0 <= yc <= 1.0) or not (0.0 < w <= 1.0) or not (0.0 < h <= 1.0):
            errors.append(
                TargetAuditError(
                    sample_id=sample_id,
                    target_id=target.target_id,
                    error_type="invalid_yolo_normalization",
                    message=f"YOLO normalized bbox [{xc}, {yc}, {w}, {h}] must be inside [0, 1]",
                )
            )

        # 5. Check visibility ratio
        if not (0.0 <= target.visibility_ratio <= 1.0):
            errors.append(
                TargetAuditError(
                    sample_id=sample_id,
                    target_id=target.target_id,
                    error_type="invalid_visibility_ratio",
                    message=f"Visibility ratio {target.visibility_ratio} must be in [0.0, 1.0]",
                )
            )

        return errors

    @staticmethod
    def verify_frame_annotation(annotation: FrameAnnotation) -> list[str]:
        """Audit a complete FrameAnnotation object."""
        errors: list[str] = []

        try:
            annotation.validate_schema()
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error for {annotation.sample_id}: {e.message}")

        for t in annotation.targets:
            t_errs = AnnotationVerifier.verify_target(
                target=t,
                frame_width=annotation.width,
                frame_height=annotation.height,
                sample_id=annotation.sample_id,
            )
            for te in t_errs:
                errors.append(f"[{te.sample_id} Target {te.target_id}] {te.error_type}: {te.message}")

        return errors

    @staticmethod
    def render_visual_audit_overlay(
        image_rgb: np.ndarray,
        annotation: FrameAnnotation,
    ) -> np.ndarray:
        """Render debug visualization with bounding boxes, circle outlines, center dots, and ignore regions onto a numpy RGB array."""
        h, w, c = image_rgb.shape
        overlay = image_rgb.copy()

        # 1. Draw Ignore Regions in Orange
        for region in annotation.ignore_regions:
            xmin, ymin, xmax, ymax = [int(v) for v in region.bbox_xyxy]
            xmin = max(0, min(w - 1, xmin))
            xmax = max(0, min(w - 1, xmax))
            ymin = max(0, min(h - 1, ymin))
            ymax = max(0, min(h - 1, ymax))

            orange = np.array([255, 140, 0], dtype=np.uint8)
            # Outline rectangle
            overlay[ymin:ymin + 2, xmin:xmax] = orange
            overlay[ymax - 2:ymax, xmin:xmax] = orange
            overlay[ymin:ymax, xmin:xmin + 2] = orange
            overlay[ymin:ymax, xmax - 2:xmax] = orange

        # 2. Draw Target Bounding Boxes & Circle Outlines
        for t in annotation.targets:
            if t.visibility == "visible":
                box_color = np.array([0, 255, 0], dtype=np.uint8)       # Green
            elif t.visibility == "partial":
                box_color = np.array([255, 255, 0], dtype=np.uint8)     # Yellow
            else:
                box_color = np.array([255, 50, 50], dtype=np.uint8)      # Red

            xmin, ymin, xmax, ymax = [int(round(v)) for v in t.bbox_xyxy]
            xmin = max(0, min(w - 1, xmin))
            xmax = max(0, min(w - 1, xmax))
            ymin = max(0, min(h - 1, ymin))
            ymax = max(0, min(h - 1, ymax))

            # Bounding box lines
            overlay[ymin:min(h, ymin + 2), xmin:xmax] = box_color
            overlay[max(0, ymax - 2):ymax, xmin:xmax] = box_color
            overlay[ymin:ymax, xmin:min(w, xmin + 2)] = box_color
            overlay[ymin:ymax, max(0, xmax - 2):xmax] = box_color

            # Draw Center Dot (Magenta)
            cx, cy = int(round(t.center_px[0])), int(round(t.center_px[1]))
            cy1, cy2 = max(0, cy - 2), min(h, cy + 3)
            cx1, cx2 = max(0, cx - 2), min(w, cx + 3)
            overlay[cy1:cy2, cx1:cx2] = np.array([255, 0, 255], dtype=np.uint8)

            # Circle outline (cyan)
            radius = t.radius_px
            for angle_deg in range(0, 360, 5):
                rad = math.radians(angle_deg)
                px = int(round(cx + radius * math.cos(rad)))
                py = int(round(cy + radius * math.sin(rad)))
                if 0 <= px < w and 0 <= py < h:
                    overlay[py, px] = np.array([0, 220, 255], dtype=np.uint8)

        return overlay

    @staticmethod
    def audit_dataset_directory(
        dataset_dir: Path | str,
    ) -> DatasetAuditReport:
        """Scan and audit all annotations in a dataset folder."""
        d_path = Path(dataset_dir)
        annots_dir = d_path / "annotations"

        if not annots_dir.exists():
            annots_dir = d_path

        annot_files = sorted(list(annots_dir.glob("*.json")))

        total_samples = len(annot_files)
        total_targets = 0
        valid_targets = 0
        occluded_targets = 0
        all_errors: list[str] = []
        all_warnings: list[str] = []
        target_errors: list[TargetAuditError] = []
        class_dist: dict[int, int] = {}

        for p in annot_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)

                targets = [TargetAnnotation(**t) for t in data.get("targets", [])]
                ignore_regions = [IgnoreRegion(**r) for r in data.get("ignore_regions", [])]
                provenance = None

                annot = FrameAnnotation(
                    schema_version=data.get("schema_version", 1),
                    sample_id=data.get("sample_id", p.stem),
                    image_path=data.get("image_path", ""),
                    image_sha256=data.get("image_sha256", ""),
                    width=data.get("width", 1920),
                    height=data.get("height", 1080),
                    targets=targets,
                    ignore_regions=ignore_regions,
                    provenance=provenance,
                )

                errs = AnnotationVerifier.verify_frame_annotation(annot)
                all_errors.extend(errs)

                for t in annot.targets:
                    total_targets += 1
                    class_dist[t.class_id] = class_dist.get(t.class_id, 0) + 1
                    if t.visibility == "occluded":
                        occluded_targets += 1
                    else:
                        valid_targets += 1

            except Exception as e:
                all_errors.append(f"Failed to parse annotation file {p}: {e}")

        is_valid = len(all_errors) == 0
        return DatasetAuditReport(
            is_valid=is_valid,
            total_samples=total_samples,
            total_targets=total_targets,
            valid_targets=valid_targets,
            occluded_targets=occluded_targets,
            errors=all_errors,
            warnings=all_warnings,
            target_errors=target_errors,
            class_distribution=class_dist,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenPrism Annotation Verification & Visual Audit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit dataset annotations and generate visual inspection overlays")
    audit_parser.add_argument("--dataset-dir", required=True, help="Directory containing dataset images and annotations")
    audit_parser.add_argument("--visual-audit-dir", default=None, help="Directory to output visual audit overlay images")
    audit_parser.add_argument("--num-visual-samples", type=int, default=5, help="Number of random visual samples to render")

    args = parser.parse_args()

    if args.command == "audit":
        dataset_path = Path(args.dataset_dir)
        report = AnnotationVerifier.audit_dataset_directory(dataset_path)

        print(f"Dataset Audit Result: {'PASSED' if report.is_valid else 'FAILED'}")
        print(f"Total Samples: {report.total_samples}")
        print(f"Total Targets: {report.total_targets} (Valid: {report.valid_targets}, Occluded: {report.occluded_targets})")
        print(f"Class Distribution: {report.class_distribution}")

        if report.errors:
            print("Errors:")
            for err in report.errors[:20]:
                print(f"  - {err}")

        # Generate visual audit overlays
        if args.visual_audit_dir and report.total_samples > 0:
            out_vis = Path(args.visual_audit_dir)
            out_vis.mkdir(parents=True, exist_ok=True)

            annots_dir = dataset_path / "annotations"
            annot_files = sorted(list(annots_dir.glob("*.json")))

            num_samples = min(args.num_visual_samples, len(annot_files))
            sampled_files = random.sample(annot_files, num_samples)

            for p in sampled_files:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)

                targets = [TargetAnnotation(**t) for t in data.get("targets", [])]
                ignore_regions = [IgnoreRegion(**r) for r in data.get("ignore_regions", [])]
                annot = FrameAnnotation(
                    schema_version=data.get("schema_version", 1),
                    sample_id=data.get("sample_id", p.stem),
                    image_path=data.get("image_path", ""),
                    image_sha256=data.get("image_sha256", ""),
                    width=data.get("width", 1920),
                    height=data.get("height", 1080),
                    targets=targets,
                    ignore_regions=ignore_regions,
                )

                # Render overlay on empty or synthetic canvas
                canvas = np.full((annot.height, annot.width, 3), 40, dtype=np.uint8)
                overlay = AnnotationVerifier.render_visual_audit_overlay(canvas, annot)
                save_png(out_vis / f"audit_{p.stem}.png", overlay)

            print(f"Saved {num_samples} visual audit overlay images -> {out_vis}")

        if not report.is_valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
