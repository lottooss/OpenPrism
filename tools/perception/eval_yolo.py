"""Perception model evaluation pipeline computing calibration (ECE), center error, precision/recall, and scenario slices."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from tools.data.synthetic_generator import FrameAnnotation
from tools.perception.model_manifest import (
    ModelManifest,
    PerceptionMetrics,
    SliceMetric,
)


@dataclass
class DetectionPrediction:
    sample_id: str
    class_id: int
    confidence: float
    bbox_xyxy: list[float]  # [xmin, ymin, xmax, ymax]
    center_px: list[float]  # [cx, cy]
    radius_px: float = 0.0


@dataclass
class MatchResult:
    is_true_positive: bool
    confidence: float
    iou: float
    center_error_px: float
    ground_truth_id: int | None
    predicted_target: DetectionPrediction


@dataclass
class FailureCase:
    sample_id: str
    failure_type: str  # "false_positive", "false_negative", "large_center_error"
    confidence: float
    center_error_px: float
    bbox_xyxy: list[float]
    details: str = ""


class PerceptionEvaluator:
    """Evaluates perception predictions against mathematical ground truth."""

    @staticmethod
    def compute_iou(box_a: list[float], box_b: list[float]) -> float:
        """Compute Intersection-over-Union (IoU) between two [xmin, ymin, xmax, ymax] boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_ece(
        confidences: list[float],
        accuracies: list[int],
        num_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE) across confidence bins."""
        if not confidences or len(confidences) != len(accuracies):
            return 0.0

        n = len(confidences)
        bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
        ece = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            bin_indices = [
                idx
                for idx, c in enumerate(confidences)
                if bin_lower <= c < bin_upper or (i == num_bins - 1 and bin_lower <= c <= bin_upper)
            ]

            bin_size = len(bin_indices)
            if bin_size > 0:
                bin_acc = sum(accuracies[idx] for idx in bin_indices) / bin_size
                bin_conf = sum(confidences[idx] for idx in bin_indices) / bin_size
                ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return ece

    @classmethod
    def average_precision(cls, annotations: list[FrameAnnotation],
                          predictions: list[DetectionPrediction], threshold: float) -> float:
        """101-point interpolated AP, averaged over ground-truth classes."""
        classes = sorted({t.class_id for a in annotations for t in a.targets if t.visibility != "occluded"})
        scores: list[float] = []
        for class_id in classes:
            ground_truth = {a.sample_id: [t for t in a.targets
                            if t.visibility != "occluded" and t.class_id == class_id] for a in annotations}
            count = sum(len(items) for items in ground_truth.values())
            matched: dict[str, set[int]] = {}
            tp = 0
            fp = 0
            curve: list[tuple[float, float]] = []
            for prediction in sorted((p for p in predictions if p.class_id == class_id),
                                     key=lambda p: p.confidence, reverse=True):
                used = matched.setdefault(prediction.sample_id, set())
                choices = [(cls.compute_iou(prediction.bbox_xyxy, t.bbox_xyxy), i)
                           for i, t in enumerate(ground_truth.get(prediction.sample_id, [])) if i not in used]
                iou, idx = max(choices, default=(0.0, -1))
                if idx >= 0 and iou >= threshold:
                    used.add(idx)
                    tp += 1
                else:
                    fp += 1
                curve.append((tp / count, tp / (tp + fp)))
            scores.append(sum(max((p for r, p in curve if r >= step / 100), default=0.0)
                              for step in range(101)) / 101)
        return sum(scores) / len(scores) if scores else 0.0

    @classmethod
    def evaluate_detections(
        cls,
        annotations: list[FrameAnnotation],
        predictions: list[DetectionPrediction],
        iou_threshold: float = 0.50,
        conf_threshold: float = 0.25,
    ) -> tuple[PerceptionMetrics, list[FailureCase], dict[str, SliceMetric]]:
        """Perform evaluation across all samples, calculate metrics, failure cases, and scenario slices."""
        pred_by_sample: dict[str, list[DetectionPrediction]] = {}
        for p in predictions:
            if p.confidence >= conf_threshold:
                pred_by_sample.setdefault(p.sample_id, []).append(p)

        all_matches: list[MatchResult] = []
        failure_cases: list[FailureCase] = []
        center_errors: list[float] = []
        confidences_for_ece: list[float] = []
        accuracies_for_ece: list[int] = []

        total_gt_targets = 0
        scenario_groups: dict[str, list[tuple[int, int, int, list[float]]]] = {}  # slice -> [(gt_count, tp_count, fp_count, center_errs)]

        for annot in annotations:
            sid = annot.sample_id
            gts = [t for t in annot.targets if t.visibility != "occluded"]
            total_gt_targets += len(gts)
            preds = pred_by_sample.get(sid, [])

            # Identify scenario slice key
            scenario_key = (
                annot.provenance.background_theme
                if annot.provenance and annot.provenance.background_theme
                else "standard"
            )

            # Sort predictions by confidence descending
            sorted_preds = sorted(preds, key=lambda x: x.confidence, reverse=True)
            matched_gts: set[int] = set()

            tp_count = 0
            slice_center_errs: list[float] = []

            for p in sorted_preds:
                best_iou = 0.0
                best_gt_idx = -1

                for idx, gt in enumerate(gts):
                    if idx in matched_gts or gt.class_id != p.class_id:
                        continue
                    iou = cls.compute_iou(p.bbox_xyxy, gt.bbox_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    matched_gts.add(best_gt_idx)
                    gt_target = gts[best_gt_idx]

                    # Center L2 distance error
                    dx = p.center_px[0] - gt_target.center_px[0]
                    dy = p.center_px[1] - gt_target.center_px[1]
                    err_px = math.sqrt(dx * dx + dy * dy)
                    center_errors.append(err_px)
                    slice_center_errs.append(err_px)

                    tp_count += 1
                    all_matches.append(
                        MatchResult(
                            is_true_positive=True,
                            confidence=p.confidence,
                            iou=best_iou,
                            center_error_px=err_px,
                            ground_truth_id=gt_target.target_id,
                            predicted_target=p,
                        )
                    )
                    confidences_for_ece.append(p.confidence)
                    accuracies_for_ece.append(1)

                    if err_px > 5.0:
                        failure_cases.append(
                            FailureCase(
                                sample_id=sid,
                                failure_type="large_center_error",
                                confidence=p.confidence,
                                center_error_px=err_px,
                                bbox_xyxy=p.bbox_xyxy,
                                details=f"Center error {err_px:.2f}px exceeds 5.0px threshold",
                            )
                        )
                else:
                    # False Positive
                    all_matches.append(
                        MatchResult(
                            is_true_positive=False,
                            confidence=p.confidence,
                            iou=best_iou,
                            center_error_px=0.0,
                            ground_truth_id=None,
                            predicted_target=p,
                        )
                    )
                    confidences_for_ece.append(p.confidence)
                    accuracies_for_ece.append(0)
                    failure_cases.append(
                        FailureCase(
                            sample_id=sid,
                            failure_type="false_positive",
                            confidence=p.confidence,
                            center_error_px=0.0,
                            bbox_xyxy=p.bbox_xyxy,
                            details=f"Unmatched false positive with confidence {p.confidence:.3f}",
                        )
                    )

            # Check False Negatives
            for idx, gt in enumerate(gts):
                if idx not in matched_gts:
                    failure_cases.append(
                        FailureCase(
                            sample_id=sid,
                            failure_type="false_negative",
                            confidence=0.0,
                            center_error_px=0.0,
                            bbox_xyxy=gt.bbox_xyxy,
                            details=f"Missed target sphere at center {gt.center_px}",
                        )
                    )

            # Record scenario slice metrics
            scenario_groups.setdefault(scenario_key, []).append(
                (len(gts), tp_count, len(preds) - tp_count, slice_center_errs)
            )

        total_tp = sum(1 for m in all_matches if m.is_true_positive)
        total_fp = sum(1 for m in all_matches if not m.is_true_positive)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt_targets if total_gt_targets > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        mean_center_err = sum(center_errors) / len(center_errors) if center_errors else 0.0
        sorted_errs = sorted(center_errors)
        p95_idx = int(len(sorted_errs) * 0.95)
        p95_center_err = sorted_errs[p95_idx] if sorted_errs else 0.0
        max_center_err = max(center_errors) if center_errors else 0.0

        ece = cls.compute_ece(confidences_for_ece, accuracies_for_ece)

        metrics = PerceptionMetrics(
            map50=cls.average_precision(annotations, predictions, 0.50),
            map50_95=sum(cls.average_precision(annotations, predictions, 0.50 + i * 0.05)
                         for i in range(10)) / 10,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_center_error_px=mean_center_err,
            p95_center_error_px=p95_center_err,
            max_center_error_px=max_center_err,
            expected_calibration_error=ece,
        )

        # Build scenario slice metrics
        scenario_slices: dict[str, SliceMetric] = {}
        for skey, items in scenario_groups.items():
            s_gt = sum(item[0] for item in items)
            s_tp = sum(item[1] for item in items)
            s_errs: list[float] = []
            for item in items:
                s_errs.extend(item[3])

            s_fp = sum(item[2] for item in items)
            s_prec = s_tp / (s_tp + s_fp) if s_tp + s_fp else 0.0
            s_rec = s_tp / s_gt if s_gt > 0 else 0.0
            s_f1 = 2 * (s_prec * s_rec) / (s_prec + s_rec) if s_prec + s_rec else 0.0
            s_mean_err = sum(s_errs) / len(s_errs) if s_errs else 0.0

            scenario_slices[skey] = SliceMetric(
                sample_count=len(items),
                precision=min(1.0, s_prec),
                recall=min(1.0, s_rec),
                f1_score=min(1.0, s_f1),
                mean_center_error_px=s_mean_err,
            )

        return metrics, failure_cases, scenario_slices

    @classmethod
    def generate_evaluation_report(
        cls,
        manifest: ModelManifest,
        failure_cases: list[FailureCase],
        output_md_path: Path | str | None = None,
    ) -> str:
        """Generate rich Markdown evaluation summary report."""
        m = manifest.metrics
        md_lines: list[str] = [
            f"# Perception Model Evaluation Report: {manifest.model_name} ({manifest.architecture})",
            "",
            f"**Model ID**: `{manifest.model_id}`  ",
            f"**Version**: `{manifest.version}`  ",
            f"**License**: `{manifest.license}`  ",
            f"**Input Tensor**: `{manifest.input_tensor.shape}` ({manifest.input_tensor.dtype} {manifest.input_tensor.layout})  ",
            "",
            "## 1. Overall Performance Metrics",
            "",
            "| Metric | Value | Gate Threshold | Status |",
            "|---|---|---|---|",
            f"| **Precision** | {m.precision:.4f} ({m.precision * 100:.1f}%) | ≥ 0.90 | {'✅ PASS' if m.precision >= 0.90 else '⚠️ WARN'} |",
            f"| **Recall** | {m.recall:.4f} ({m.recall * 100:.1f}%) | ≥ 0.90 | {'✅ PASS' if m.recall >= 0.90 else '⚠️ WARN'} |",
            f"| **F1 Score** | {m.f1_score:.4f} | ≥ 0.90 | {'✅ PASS' if m.f1_score >= 0.90 else '⚠️ WARN'} |",
            f"| **Mean Center Error** | {m.mean_center_error_px:.2f} px | ≤ 3.0 px | {'✅ PASS' if m.mean_center_error_px <= 3.0 else '⚠️ WARN'} |",
            f"| **P95 Center Error** | {m.p95_center_error_px:.2f} px | ≤ 5.0 px | {'✅ PASS' if m.p95_center_error_px <= 5.0 else '⚠️ WARN'} |",
            f"| **Expected Calibration Error (ECE)** | {m.expected_calibration_error:.4f} | ≤ 0.15 | {'✅ PASS' if m.expected_calibration_error <= 0.15 else '⚠️ WARN'} |",
            "",
            "## 2. Scenario Slice Breakdown",
            "",
            "| Scenario / Domain Slice | Samples | Precision | Recall | F1 | Mean Center Error |",
            "|---|---|---|---|---|---|",
        ]

        for sname, sm in sorted(manifest.scenario_slices.items()):
            md_lines.append(
                f"| `{sname}` | {sm.sample_count} | {sm.precision:.3f} | {sm.recall:.3f} | {sm.f1_score:.3f} | {sm.mean_center_error_px:.2f} px |"
            )

        md_lines.extend([
            "",
            "## 3. Failure Gallery Summary",
            "",
            f"Total Failure Incidents: **{len(failure_cases)}**",
            "",
            "| Sample ID | Type | Confidence | Center Error | Description |",
            "|---|---|---|---|---|",
        ])

        for fc in failure_cases[:10]:
            md_lines.append(
                f"| `{fc.sample_id}` | `{fc.failure_type}` | {fc.confidence:.3f} | {fc.center_error_px:.2f} px | {fc.details} |"
            )

        report_md = "\n".join(md_lines) + "\n"

        if output_md_path:
            p = Path(output_md_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(report_md, encoding="utf-8")

        return report_md
