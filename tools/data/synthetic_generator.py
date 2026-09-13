"""Procedural synthetic target sphere renderer, domain randomizer, and ground-truth annotation generator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from typing import Any
import uuid
import zlib

import jsonschema
import numpy as np


def get_annotation_schema_path() -> Path:
    """Return the absolute path to annotation_format.schema.json."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "schemas"
        / "manifest"
        / "annotation_format.schema.json"
    )


def save_png(file_path: Path | str, img_rgb: np.ndarray) -> None:
    """Write an RGB uint8 numpy array to a valid standard PNG file using pure zlib and struct."""
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.clip(0, 255).astype(np.uint8)

    h, w, c = img_rgb.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels (RGB), got shape {img_rgb.shape}")

    # Prepend filter byte 0 (None) to each row
    raw_rows = [b"\x00" + img_rgb[r].tobytes() for r in range(h)]
    raw_bytes = b"".join(raw_rows)

    compressor = zlib.compressobj(level=6)
    compressed = compressor.compress(raw_bytes) + compressor.flush()

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")

    png_bytes = b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend
    with open(file_path, "wb") as f:
        f.write(png_bytes)


@dataclass
class TargetAnnotation:
    target_id: int
    class_id: int
    class_name: str
    center_px: list[float]  # [x, y]
    radius_px: float
    bbox_xyxy: list[float]  # [xmin, ymin, xmax, ymax]
    bbox_yolo: list[float]  # [xcenter, ycenter, width, height] normalized to [0, 1]
    visibility: str  # "visible", "partial", "occluded"
    visibility_ratio: float  # [0.0, 1.0]
    occluded_by: list[int] = field(default_factory=list)

    def to_yolo_line(self) -> str:
        """Format target as YOLO format string."""
        x, y, w, h = self.bbox_yolo
        return f"{self.class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


@dataclass
class IgnoreRegion:
    region_id: str
    region_type: str  # "crosshair", "hud_overlay", "weapon_model", "custom"
    bbox_xyxy: list[float]


@dataclass
class ProvenanceInfo:
    generator_name: str
    random_seed: int
    background_theme: str
    augmentations_applied: list[str]
    created_at_utc: str
    source_session_id: str = ""


@dataclass
class FrameAnnotation:
    schema_version: int = 1
    sample_id: str = ""
    image_path: str = ""
    image_sha256: str = ""
    width: int = 1920
    height: int = 1080
    targets: list[TargetAnnotation] = field(default_factory=list)
    ignore_regions: list[IgnoreRegion] = field(default_factory=list)
    provenance: ProvenanceInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
            "targets": [asdict(t) for t in self.targets],
            "ignore_regions": [asdict(r) for r in self.ignore_regions],
        }
        if self.provenance is not None:
            d["provenance"] = asdict(self.provenance)
        return d

    def validate_schema(self) -> None:
        """Validate annotation structure against JSON schema."""
        schema_path = get_annotation_schema_path()
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=self.to_dict(), schema=schema)

    def save_json(self, path: Path | str) -> None:
        self.validate_schema()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yolo_txt(self, path: Path | str, min_visibility: float = 0.20) -> None:
        """Export YOLO format text annotations, filtering out heavily occluded targets."""
        lines = [
            t.to_yolo_line()
            for t in self.targets
            if t.visibility_ratio >= min_visibility and t.visibility != "occluded"
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


@dataclass
class SyntheticGeneratorConfig:
    width: int = 1920
    height: int = 1080
    min_targets: int = 1
    max_targets: int = 5
    min_radius: float = 18.0
    max_radius: float = 55.0
    target_colors: list[tuple[int, int, int]] = field(
        default_factory=lambda: [
            (0, 220, 255),    # Aimlabs Cyan
            (255, 60, 60),    # Ruby Red
            (50, 255, 80),    # Neon Green
            (255, 170, 0),    # Amber Orange
            (220, 80, 255),   # Magenta Purple
        ]
    )
    background_themes: list[str] = field(
        default_factory=lambda: [
            "aimlabs_grid",
            "dark_mode",
            "high_contrast",
            "textured_concrete",
        ]
    )
    enable_motion_blur: bool = True
    enable_flash_burst: bool = True
    enable_sensor_noise: bool = True
    enable_crosshair: bool = True
    enable_hud_overlay: bool = True
    enable_occlusions: bool = True


class SyntheticTargetGenerator:
    """Procedural 3D-shaded sphere renderer, domain randomizer, and annotation generator."""

    def __init__(self, config: SyntheticGeneratorConfig | None = None) -> None:
        self.config = config or SyntheticGeneratorConfig()

    def _render_background(self, theme: str, rng: random.Random) -> np.ndarray:
        """Render a synthetic background canvas with procedural lighting and grids."""
        w, h = self.config.width, self.config.height

        if theme == "aimlabs_grid":
            base_val = rng.randint(45, 60)
            img_arr = np.full((h, w, 3), base_val, dtype=np.uint8)
            grid_spacing = rng.randint(60, 100)
            grid_color = np.array([base_val + 25, base_val + 25, base_val + 35], dtype=np.uint8)
            img_arr[::grid_spacing, :, :] = grid_color
            img_arr[:, ::grid_spacing, :] = grid_color

        elif theme == "dark_mode":
            base_val = rng.randint(15, 30)
            img_arr = np.full((h, w, 3), base_val, dtype=np.uint8)
            grid_spacing = rng.randint(80, 120)
            img_arr[::grid_spacing, :, :] = base_val + 15
            img_arr[:, ::grid_spacing, :] = base_val + 15

        elif theme == "high_contrast":
            base_val = rng.randint(200, 230)
            img_arr = np.full((h, w, 3), base_val, dtype=np.uint8)
            grid_spacing = rng.randint(70, 110)
            img_arr[::grid_spacing, :, :] = base_val - 35
            img_arr[:, ::grid_spacing, :] = base_val - 35

        else:  # textured_concrete
            noise = np.random.RandomState(rng.randint(0, 1_000_000)).normal(100, 15, (h, w)).clip(0, 255).astype(np.uint8)
            img_arr = np.stack([noise, noise, noise], axis=-1)

        return img_arr

    def _render_shaded_sphere(
        self,
        radius: float,
        color: tuple[int, int, int],
        light_dir: tuple[float, float, float] = (-0.5, -0.5, 0.707),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render a single 3D shaded sphere with Lambertian diffuse + Phong specular highlight."""
        r_int = int(math.ceil(radius))
        size = 2 * r_int + 1
        y, x = np.ogrid[-r_int:r_int + 1, -r_int:r_int + 1]
        dist_sq = x * x + y * y
        mask = dist_sq <= radius * radius

        sphere_rgb = np.zeros((size, size, 3), dtype=np.float32)

        # Normalized coordinates on the sphere surface
        z = np.sqrt(np.maximum(0.0, radius * radius - dist_sq))
        nx = x / radius
        ny = y / radius
        nz = z / radius

        # Normalize light vector
        lx, ly, lz = light_dir
        l_mag = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / l_mag, ly / l_mag, lz / l_mag

        # Lambertian diffuse
        dot = np.maximum(0.0, -(nx * lx + ny * ly + nz * lz))
        diffuse = 0.35 + 0.65 * dot

        # Phong specular highlight
        rz = 2 * dot * nz + lz
        specular = np.power(np.maximum(0.0, -rz), 16) * 0.55

        for c in range(3):
            sphere_rgb[:, :, c] = np.clip((color[c] / 255.0) * diffuse + specular, 0.0, 1.0) * 255.0

        return sphere_rgb.astype(np.uint8), mask

    def generate_sample(
        self,
        seed: int | None = None,
        sample_id: str | None = None,
        image_filename: str = "frame.png",
    ) -> tuple[np.ndarray, FrameAnnotation]:
        """Generate a single synthetic frame with domain randomizations and mathematical ground truth."""
        actual_seed = seed if seed is not None else random.randint(0, 2_147_483_647)
        rng = random.Random(actual_seed)
        np_rng = np.random.RandomState(actual_seed)

        w, h = self.config.width, self.config.height
        sid = sample_id or str(uuid.uuid4())
        theme = rng.choice(self.config.background_themes)

        canvas = self._render_background(theme, rng).astype(np.float32)

        num_targets = rng.randint(self.config.min_targets, self.config.max_targets)
        depth_buffer = np.full((h, w), -1, dtype=np.int32)
        target_masks: list[np.ndarray] = []
        target_infos: list[dict[str, Any]] = []

        # Sort targets by random depth z (front to back)
        z_orders = sorted([rng.uniform(1.0, 10.0) for _ in range(num_targets)])

        for tid, z_depth in enumerate(z_orders):
            radius = rng.uniform(self.config.min_radius, self.config.max_radius)
            cx = rng.uniform(radius + 50.0, w - radius - 50.0)
            cy = rng.uniform(radius + 50.0, h - radius - 50.0)
            color = rng.choice(self.config.target_colors)

            sphere_rgb, local_mask = self._render_shaded_sphere(radius, color)
            r_int = int(math.ceil(radius))

            paste_x = int(round(cx - r_int))
            paste_y = int(round(cy - r_int))

            # Destination bounding box inside canvas
            y_start = max(0, paste_y)
            y_end = min(h, paste_y + sphere_rgb.shape[0])
            x_start = max(0, paste_x)
            x_end = min(w, paste_x + sphere_rgb.shape[1])

            sy_start = y_start - paste_y
            sy_end = sy_start + (y_end - y_start)
            sx_start = x_start - paste_x
            sx_end = sx_start + (x_end - x_start)

            sub_mask = local_mask[sy_start:sy_end, sx_start:sx_end]
            sub_sphere = sphere_rgb[sy_start:sy_end, sx_start:sx_end]

            # Composite onto canvas
            target_region = canvas[y_start:y_end, x_start:x_end]
            target_region[sub_mask] = sub_sphere[sub_mask]
            canvas[y_start:y_end, x_start:x_end] = target_region

            # Global boolean mask for this target
            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[y_start:y_end, x_start:x_end] = sub_mask
            target_masks.append(full_mask)

            # Update depth buffer
            depth_buffer[full_mask] = tid

            # Calculate exact bounding box
            xmin = max(0.0, cx - radius)
            ymin = max(0.0, cy - radius)
            xmax = min(float(w), cx + radius)
            ymax = min(float(h), cy + radius)

            # YOLO box (normalized center_x, center_y, width, height in [0, 1])
            bbox_yolo = [
                ((xmin + xmax) / 2.0) / w,
                ((ymin + ymax) / 2.0) / h,
                (xmax - xmin) / w,
                (ymax - ymin) / h,
            ]

            target_infos.append(
                {
                    "target_id": tid,
                    "center_px": [round(cx, 3), round(cy, 3)],
                    "radius_px": round(radius, 3),
                    "bbox_xyxy": [round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3)],
                    "bbox_yolo": [round(v, 6) for v in bbox_yolo],
                    "total_pixels": int(np.count_nonzero(full_mask)),
                }
            )

        # Calculate visibility & mutual occlusions
        target_annotations: list[TargetAnnotation] = []
        for tid, tinfo in enumerate(target_infos):
            initial_pixels = tinfo["total_pixels"]
            if initial_pixels == 0:
                continue

            visible_pixels = int(np.count_nonzero(depth_buffer == tid))
            visibility_ratio = visible_pixels / float(initial_pixels)

            # Determine occluded_by list
            occluded_by: list[int] = []
            for other_tid in range(tid + 1, len(target_infos)):
                overlap = np.logical_and(target_masks[tid], depth_buffer == other_tid)
                if np.any(overlap):
                    occluded_by.append(other_tid)

            if visibility_ratio >= 0.85:
                vis_label = "visible"
            elif visibility_ratio >= 0.15:
                vis_label = "partial"
            else:
                vis_label = "occluded"

            annot = TargetAnnotation(
                target_id=tinfo["target_id"],
                class_id=0,
                class_name="target_sphere",
                center_px=tinfo["center_px"],
                radius_px=tinfo["radius_px"],
                bbox_xyxy=tinfo["bbox_xyxy"],
                bbox_yolo=tinfo["bbox_yolo"],
                visibility=vis_label,
                visibility_ratio=round(visibility_ratio, 4),
                occluded_by=occluded_by,
            )
            target_annotations.append(annot)

        # Ignore regions (HUD, Crosshairs, weapon models)
        ignore_regions: list[IgnoreRegion] = []
        augmentations_applied: list[str] = []

        # Photometric & Synthetic Augmentations
        # 1. Motion Blur (simple 1D kernel along motion vector)
        if self.config.enable_motion_blur and rng.random() < 0.35:
            ksize = rng.choice([3, 5, 7])
            kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
            pad = ksize // 2
            padded = np.pad(canvas, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            # Simple spatial filtering
            blurred = np.zeros_like(canvas)
            for c in range(3):
                for dy in range(ksize):
                    for dx in range(ksize):
                        blurred[:, :, c] += padded[dy:dy + h, dx:dx + w, c] * kernel[dy, dx]
            canvas = blurred
            augmentations_applied.append(f"motion_blur_k{ksize}")

        # 2. Flash / Illumination Burst
        if self.config.enable_flash_burst and rng.random() < 0.20:
            flash_intensity = rng.randint(20, 60)
            canvas = np.clip(canvas + flash_intensity, 0.0, 255.0)
            augmentations_applied.append(f"flash_burst_{flash_intensity}")

        # 3. Crosshair Overlay & Ignore Region
        if self.config.enable_crosshair:
            cx_center, cy_center = int(w // 2), int(h // 2)
            ch_size = rng.randint(6, 12)
            # Draw green crosshair lines
            canvas[cy_center - 1:cy_center + 1, cx_center - ch_size:cx_center + ch_size, :] = [0, 255, 0]
            canvas[cy_center - ch_size:cy_center + ch_size, cx_center - 1:cx_center + 1, :] = [0, 255, 0]
            ignore_regions.append(
                IgnoreRegion(
                    region_id="crosshair_center",
                    region_type="crosshair",
                    bbox_xyxy=[float(cx_center - ch_size - 2), float(cy_center - ch_size - 2), float(cx_center + ch_size + 2), float(cy_center + ch_size + 2)],
                )
            )
            augmentations_applied.append("crosshair_hud")

        # 4. HUD Score Overlay & Ignore Region
        if self.config.enable_hud_overlay and rng.random() < 0.50:
            hx1, hy1, hx2, hy2 = int(w // 2 - 100), 20, int(w // 2 + 100), 60
            canvas[hy1:hy2, hx1:hx2, :] = canvas[hy1:hy2, hx1:hx2, :] * 0.3 + 30.0
            ignore_regions.append(
                IgnoreRegion(
                    region_id="score_hud",
                    region_type="hud_overlay",
                    bbox_xyxy=[float(hx1), float(hy1), float(hx2), float(hy2)],
                )
            )
            augmentations_applied.append("score_hud")

        # 5. Gaussian Sensor Noise
        if self.config.enable_sensor_noise and rng.random() < 0.30:
            noise_sigma = rng.uniform(2.0, 6.0)
            noise = np_rng.normal(0, noise_sigma, (h, w, 3))
            canvas = np.clip(canvas + noise, 0.0, 255.0)
            augmentations_applied.append(f"sensor_noise_{noise_sigma:.2f}")

        final_rgb = np.clip(canvas, 0, 255).astype(np.uint8)

        # Compute SHA-256 hash of RGB bytes
        sha256_hash = hashlib.sha256(final_rgb.tobytes()).hexdigest()

        provenance = ProvenanceInfo(
            generator_name="SyntheticTargetGenerator_v1",
            random_seed=actual_seed,
            background_theme=theme,
            augmentations_applied=augmentations_applied,
            created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source_session_id=sid,
        )

        frame_annot = FrameAnnotation(
            schema_version=1,
            sample_id=sid,
            image_path=image_filename,
            image_sha256=sha256_hash,
            width=w,
            height=h,
            targets=target_annotations,
            ignore_regions=ignore_regions,
            provenance=provenance,
        )
        frame_annot.validate_schema()

        return final_rgb, frame_annot

    def generate_dataset_batch(
        self,
        output_dir: Path | str,
        num_samples: int = 100,
        base_seed: int = 42,
    ) -> list[FrameAnnotation]:
        """Generate a batch of synthetic dataset samples, writing images, YOLO txts, and JSON annotations."""
        out_path = Path(output_dir)
        images_dir = out_path / "images"
        labels_dir = out_path / "labels"
        annots_dir = out_path / "annotations"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        annots_dir.mkdir(parents=True, exist_ok=True)

        annotations: list[FrameAnnotation] = []

        for i in range(num_samples):
            seed = base_seed + i
            sample_id = f"synth_{base_seed:06d}_{i:06d}"
            img_rel_name = f"images/{sample_id}.png"

            img_rgb, annot = self.generate_sample(
                seed=seed,
                sample_id=sample_id,
                image_filename=img_rel_name,
            )

            # Save PNG image using pure zlib encoder
            save_png(images_dir / f"{sample_id}.png", img_rgb)

            # Save YOLO text format
            annot.save_yolo_txt(labels_dir / f"{sample_id}.txt")

            # Save rich JSON annotation
            annot.save_json(annots_dir / f"{sample_id}.json")

            annotations.append(annot)

        return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenPrism Synthetic Target Generator CLI")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated dataset")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of synthetic frames to render")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")

    args = parser.parse_args()

    cfg = SyntheticGeneratorConfig(width=args.width, height=args.height)
    generator = SyntheticTargetGenerator(cfg)

    print(f"Generating {args.num_samples} synthetic samples in {args.output_dir} (seed={args.seed})...")
    annots = generator.generate_dataset_batch(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        base_seed=args.seed,
    )
    print(f"Successfully generated {len(annots)} samples.")


if __name__ == "__main__":
    main()
