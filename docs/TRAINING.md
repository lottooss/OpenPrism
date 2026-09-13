# Training and evaluating OpenPrism

Training is offline. The runtime does not learn automatically from each test run, and calibration is parameter estimation rather than detector training.

1. Collect authorized examples covering target size, color, motion, background and occlusion. Annotate centers and extents consistently with the perception contract.
2. Split data by recording session before augmentation. Adjacent frames and transformed copies of one screenshot must not leak across training and validation sets.
3. Train a compatible detector using the optional ML environment: `uv sync --frozen --project python --extra ml`. The tooling under `tools/data/` and `tools/perception/` handles dataset manifests, splitting, model export and evaluation. Inspect each tool's entry point before supplying your own data paths.
4. Evaluate center error, recall, confidence calibration and false engagement by scenario slice, not only aggregate accuracy. Keep an independently collected real held-out set.
5. Export to ONNX, compare golden outputs against PyTorch, then build and validate a TensorRT engine for the intended GPU/runtime. Record model hash, training configuration, export settings, data revision and license.
6. Run deterministic replay and NullActuator checks before physical input. Recalibrate when display or sensitivity settings change. Measure end-to-end latency under concurrent game load.

The reference detector and the decision maker are separate. A stronger detector can improve observations without replacing tracking or control. Policy-imitation tooling under `tools/policy/` is experimental; it is not automatically enabled by training a detector.

For speed, measure the limiting stage first. Compare p50/p95/p99, center error and false engagement at the same workload. Lower precision or a smaller model should be accepted only after parity and quality checks; a faster engine that shifts target centers can make control worse.

Some comparison helpers under `tools/benchmark/` contain illustrative baseline values and synthetic evaluation inputs. Their reports are fixtures for research workflows, not evidence of a fresh hardware measurement or real held-out accuracy. Use actual recorded measurements when making performance claims.

The local demo reference was trained with procedural examples and variants of one real training screenshot. That is insufficient evidence of real-world generalization. No model weights, training data or engines are included in this repository. Ultralytics-derived artifacts retain their original AGPL/commercial licensing; the repository license does not replace it.
