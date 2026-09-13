"""Reproducible synthetic YOLO11n bootstrap; never certifies real-scene accuracy.

The generated checkpoint is an isolated AGPL-3.0 research artifact. Real sessions
must still validate it with the existing perception benchmark before acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import yaml

from tools.data.synthetic_generator import SyntheticGeneratorConfig, SyntheticTargetGenerator


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset(root: Path, count: int, seed: int) -> Path:
    """Use distinct procedural scene seeds for training and validation."""
    generator = SyntheticTargetGenerator(SyntheticGeneratorConfig(
        width=640, height=360, min_targets=0, max_targets=6,
        min_radius=6.0, max_radius=23.0, enable_occlusions=False,
    ))
    provenance: dict[str, Any] = {"source": "procedural-only", "seed": seed, "splits": {}}
    for split, size, offset in (("train", count, 0), ("val", max(32, count // 4), 1_000_000)):
        annotations = generator.generate_dataset_batch(root / split, size, seed + offset)
        provenance["splits"][split] = [
            {"sample_id": a.sample_id, "sha256": a.image_sha256,
             "seed": a.provenance.random_seed if a.provenance else None}
            for a in annotations
        ]
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    data = root / "dataset.yaml"
    data.write_text(yaml.safe_dump({"path": str(root.resolve()), "train": "train/images",
                                    "val": "val/images", "names": {0: "target_sphere"}}), encoding="utf-8")
    return data


def add_training_scene(root: Path, scene: Path, annotations: Path, seed: int) -> dict[str, Any]:
    """Augment one explicitly annotated local scene into training only.

    This scene and every derivative are excluded from validation. Independent
    live scenes are still required to establish real-scene accuracy.
    """
    # Pillow belongs to the optional ML environment, like torch/onnx below.
    Image = importlib.import_module("PIL.Image")
    document = json.loads(annotations.read_text(encoding="utf-8"))
    if set(document) != {"image_sha256", "targets"} or document["image_sha256"] != sha256(scene):
        raise ValueError("Training annotations must bind the exact scene image")
    original = Image.open(scene).convert("RGB")
    width, height = original.size
    source = original.resize((640, 360), Image.Resampling.LANCZOS)
    boxes = []
    for target in document["targets"]:
        if set(target) != {"x", "y", "radius"}:
            raise ValueError("Training targets require x/y/radius in source pixels")
        x, y, radius = (float(target[k]) for k in ("x", "y", "radius"))
        if not (np.isfinite([x, y, radius]).all() and radius > 0 and
                radius < x < width - radius and radius < y < height - radius):
            raise ValueError("Training target must be finite and fully inside the scene")
        boxes.append((x * 640 / width, y * 360 / height,
                      radius * 640 / width, radius * 360 / height))
    if not boxes:
        raise ValueError("At least one independently annotated target is required")
    rng = np.random.default_rng(seed)
    for index in range(128):
        scale = float(rng.uniform(0.85, 1.15))
        dx, dy = float(rng.uniform(-45, 45)), float(rng.uniform(-25, 25))
        tx, ty = 320 * (1 - scale) + dx, 180 * (1 - scale) + dy
        transformed = source.transform((640, 360), Image.Transform.AFFINE,
            (1 / scale, 0, -tx / scale, 0, 1 / scale, -ty / scale),
            resample=Image.Resampling.BILINEAR, fillcolor=(18, 18, 24))
        name = f"local_training_{index:04d}"
        transformed.save(root / "train/images" / f"{name}.png")
        lines = [f"0 {(x * scale + tx) / 640:.8f} {(y * scale + ty) / 360:.8f} "
                 f"{2 * rx * scale / 640:.8f} {2 * ry * scale / 360:.8f}"
                 for x, y, rx, ry in boxes]
        (root / "train/labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"source_image_sha256": sha256(scene), "annotations_sha256": sha256(annotations),
            "source_sessions": 1, "augmented_training_images": 128,
            "held_out_real_validation": False}


def write_golden_outputs(root: Path, checkpoint: Path, onnx_path: Path) -> float:
    """Compare both frameworks using the exact values delivered by FP16 input."""
    Image = importlib.import_module("PIL.Image")
    ultra = importlib.import_module("ultralytics")
    torch = importlib.import_module("torch")
    ort = importlib.import_module("onnxruntime")
    torch.set_num_threads(8)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = ultra.YOLO(str(checkpoint)).model.eval().float()
    inputs, outputs, errors = [], [], []
    for frame in sorted((root / "dataset/val/images").glob("*.png"))[:8]:
        rgb = np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32) / 255.0
        tensor = np.full((1, 3, 384, 640), 114.0 / 255.0, dtype=np.float32)
        tensor[0, :, 12:372, :] = rgb.transpose(2, 0, 1)
        # TensorRT consumes FP16. Its reference must use the same rounded input
        # values, rather than comparing two different input representations.
        tensor = tensor.astype(np.float16).astype(np.float32)
        with torch.no_grad():
            result = reference(torch.from_numpy(tensor))
            expected = (result[0] if isinstance(result, tuple) else result).numpy()
        actual = session.run(["output0"], {"images": tensor})[0]
        if actual.shape != (1, 5, 5040) or not np.isfinite(actual).all():
            raise RuntimeError("Export violates the native detector contract")
        inputs.append(tensor)
        outputs.append(expected)
        errors.append(float(np.max(np.abs(actual - expected))))
    if len(inputs) != 8:
        raise ValueError("Eight independent procedural validation inputs are required")
    np.save(root / "golden-inputs.npy", np.stack(inputs))
    np.save(root / "golden-pytorch.npy", np.stack(outputs))
    np.stack(inputs).astype("<f2").tofile(root / "golden-inputs.fp16.bin")
    np.stack(outputs).astype("<f4").tofile(root / "golden-pytorch.fp32.bin")
    return max(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=114119)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--training-scene", type=Path)
    parser.add_argument("--training-annotations", type=Path)
    args = parser.parse_args()
    if args.samples < 32 or args.epochs < 1:
        parser.error("at least 32 training samples and one epoch are required")
    if args.checkpoint and not re.fullmatch(r"[0-9a-fA-F]{64}", args.checkpoint_sha256 or ""):
        parser.error("a trusted --checkpoint-sha256 is required with --checkpoint")
    if bool(args.training_scene) != bool(args.training_annotations):
        parser.error("--training-scene and --training-annotations must be supplied together")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Do not enable cloud experiment tracking or upload generated training data.
    os.environ["YOLO_CONFIG_DIR"] = str(root / "ultralytics-settings")
    (root / "ultralytics-settings").mkdir(exist_ok=True)
    ultra = importlib.import_module("ultralytics")
    torch = importlib.import_module("torch")
    torch.set_num_threads(8)
    ultra.settings.update({"sync": False, "wandb": False, "comet": False,
                           "mlflow": False, "clearml": False, "neptune": False})
    data = prepare_dataset(root / "dataset", args.samples, args.seed)
    local_training = (add_training_scene(root / "dataset", args.training_scene,
                      args.training_annotations, args.seed)
                      if args.training_scene else None)
    # Verify bytes before PyTorch checkpoint deserialization. The default is a
    # pinned official upstream artifact, never an implicit mutable local weight.
    initial = args.checkpoint or root / "yolo11n.pt"
    initial_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
    expected_hash = args.checkpoint_sha256 if args.checkpoint else (
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1")
    if not args.checkpoint and not initial.exists():
        urlretrieve(initial_url, initial)
    initial_hash = sha256(initial)
    if initial_hash != expected_hash.lower():
        raise RuntimeError("Initial checkpoint SHA-256 mismatch")
    model = ultra.YOLO(str(initial))
    model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=8,
                device=args.device, workers=0, seed=args.seed, deterministic=True,
                rect=True, freeze=10, optimizer="AdamW", lr0=0.002,
                mosaic=0.0, mixup=0.0, scale=0.2, translate=0.1,
                hsv_h=0.03, hsv_s=0.3, hsv_v=0.3, close_mosaic=0,
                project=str(root), name="training", exist_ok=False,
                plots=False, save=True, cache=False, patience=args.epochs)
    checkpoint = Path(model.trainer.best)
    trained = ultra.YOLO(str(checkpoint))
    onnx_path = Path(trained.export(format="onnx", imgsz=[384, 640], batch=1,
                                  dynamic=False, simplify=False, opset=17,
                                  nms=False, device="cpu"))
    max_error = write_golden_outputs(root, checkpoint, onnx_path)
    report = {"schema_version": 1, "kind": "local-refinement-candidate" if local_training else "synthetic-bootstrap-candidate",
              "real_scene_acceptance": False, "license": "AGPL-3.0",
              "seed": args.seed, "epochs": args.epochs, "training_samples": args.samples,
              "validation_samples": max(32, args.samples // 4),
              "torch_version": torch.__version__, "ultralytics_version": ultra.__version__,
              "initial_checkpoint_sha256": initial_hash,
              "initial_checkpoint_source": str(initial) if args.checkpoint else initial_url,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "onnx": str(onnx_path), "onnx_sha256": sha256(onnx_path),
              "golden_input_values": "exact-fp16-representable",
              "pytorch_onnx_max_absolute_error": max_error,
              "pytorch_onnx_strict_parity_passed": max_error <= 1e-4,
              "tensorrt_parity_passed": False}
    if local_training:
        report["local_training"] = local_training
        report["training_samples"] += 128
    (root / "bootstrap-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
