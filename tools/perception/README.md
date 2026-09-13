# Real perception evidence workflow

The repository contains no trained checkpoint, exported ONNX, trusted TensorRT
engine, or authorized held-out recording dataset. These commands consume external
artifacts. They do not download models, install SDKs, train a model, or certify M3.
Use the existing locked Python environment and approved ML extra; keep assets out
of Git. Raw Ultralytics checkpoints must first be converted by their approved
training/export workflow into a TorchScript module returning a single raw tensor.

## Export and golden parity

Supply a trusted TorchScript checkpoint and a NumPy golden-input file shaped
`[N, 1, 3, 384, 640]`, float32 RGB NCHW, in `[0, 1]`. The model must return
`[1, 5, 5040]` raw detections. No labels are used to produce predictions.

```powershell
uv run --project python python -m tools.perception.onnx_exporter --torchscript D:/aim-assets/model.torchscript --sha256 <trusted-sha256> --golden-inputs D:/aim-assets/golden-inputs.npy --output D:/aim-assets/model.onnx --report D:/aim-assets/onnx-parity.json
```

This executes PyTorch and ONNX Runtime on the same golden tensors. The report
records their versions, opset, artifact/input hashes and actual parity errors.
TensorRT parity stays false until measured separately against those same inputs.
The optional ML extra is required; this tool never installs missing packages.

The native runner requires TensorRT 10, fixed `images` FP16 linear device input
`[1,3,384,640]`, and `output0` FP32 linear device output `[1,5,5040]`.
FP16 internal optimization alone does not guarantee an FP16 input binding.
Configure native builds with the explicitly trusted `AIM_TENSORRT_ROOT`; public
headers alone permit a compile check but cannot provide a runnable SDK/backend.
Graph-enabled warmup must capture real `enqueueV3` successfully; failure is not
silently replaced by zero output or a different backend.

Retain an immutable input tensor until its inference ticket is collected or the
runner has shut down. Complete producer work before enqueue. The runner keeps one
inference in flight; a busy enqueue returns `inference_timeout` without queuing.
`try_collect` queries the event covering inference and pinned output copy. The
optional `last_inference_ms()` is measured by CUDA events around inference only;
it excludes input/output copies. Warmup and shutdown may drain the stream, while
enqueue and collection do not synchronize it.

## Local TensorRT build and golden validation

Install the approved TensorRT 10 Windows SDK in a trusted local directory and
configure CMake with `-DAIM_TENSORRT_ROOT=D:/tools/TensorRT-10.8.0.43`. An SDK build
also creates `aim_engine_validation`. SDK and CUDA DLL directories are supplied
by the caller; the script changes only its process search path and restores it.
It does not install packages or redistribute SDK binaries.

```powershell
./tools/perception/build_engine.ps1 -OnnxPath D:/aim-assets/model.onnx -OnnxSha256 <trusted-sha256> -EnginePath D:/aim-assets/model.engine -ValidationExecutable build/windows-msvc-release/aim_engine_validation.exe -GoldenInputs D:/aim-assets/golden-inputs.fp16.bin -GoldenReference D:/aim-assets/golden-pytorch.fp32.bin -ReportPath D:/aim-assets/tensorrt-parity.json -TensorRtSdkRoot D:/tools/TensorRT-10.8.0.43 -CudaRoot D:/tools/cuda -Frames 8
```

For a component-based CUDA installation, pass any additional cuBLAS/NVRTC DLL
directories with `-AdditionalDllDirectories`. The input file contains exactly
`Frames * 3 * 384 * 640` little-endian IEEE float16 RGB values in `[0,1]`; the
reference contains exactly `Frames * 5 * 5040` little-endian float32 values, in
the same frame order. Preserve the source tensors and preprocessing metadata
with these raw files. Reference outputs should be measured using the supplied
golden tensors, not generated from labels.

The script verifies the supplied ONNX hash, runs `trtexec` with the required
external bindings, then validates the hash-verified engine through real
`enqueueV3` CUDA graphs. `-InternalPrecision fp16` allows FP16 internal
optimization; the default `-InternalPrecision fp32` disables TF32 and FP16 internal tactics
while retaining the same FP16 input binding. Precision failures require a more
accurate engine, not a wider acceptance threshold. The builder uses optimization
level 0 by default; raise `-BuilderOptimizationLevel` only when measured latency
justifies the additional build work. The fixed FP16 conversion/inference
tolerances are maximum all-anchor Euclidean center error <=0.25 model pixels
(<=0.75 source pixels), maximum confidence error <=0.005, and matching decoded
NMS results. These tolerances are separate from strict FP32 ONNX parity; neither
test changes or replaces the other's threshold.

By default validation runs 50 warmups and 20 iterations of each frame, retaining
the raw TensorRT outputs, every CUDA-event timing sample, p50/p95/p99/max,
artifact hashes, hardware/runtime metadata, and a build log/manifest beside the
artifacts. Timing covers GPU inference, excluding copies and other pipeline
stages. A parity failure returns a failure code while preserving its evidence.
The tool reports concurrent workload as unknown (`null`); it does not infer an
idle GPU or an active game benchmark from process presence. Record workload
conditions separately before using these timings as acceptance evidence.
This offline check does not certify detector accuracy, concurrent-game latency,
or a milestone. Synthetic training/golden data remain synthetic evidence even
when inference itself runs on the real GPU.

## Measured benchmark records

```powershell
uv run --project python python -m tools.probe.perception_benchmark --evidence D:/aim-assets/evidence.json --output-md D:/aim-assets/perception-report.md --output-json D:/aim-assets/perception-report.json
```

The evidence file has `schema_version: 1`, `metadata`, `session_splits`,
`artifacts`, and `samples`. All artifact paths are relative to that file (absolute
external paths also work). `artifacts` must contain `dataset_manifest`,
`checkpoint`, `onnx`, `engine`, and `parity_report`, each with `path` and `sha256`.
All referenced files are hashed. A hash verifies identity, not independent truth
of the supplied measurements or artifact trustworthiness.

`metadata` requires `evidence_kind: measured`, `timing_method: cuda_events`,
`gpu_device`, `model_name`, `architecture`, `dataset_revision`, `dataset_license`,
`model_license`, `tensorrt_version`, `cuda_version`, `driver_version`, `os`,
`training_config`, integer `training_seed`, `opset`, positive `warmup_iterations`,
boolean `concurrent_workload`, and `thermal_soak_seconds`.

`session_splits` maps `train`, `val`, and `test` to nonempty disjoint session ID
arrays. The hashed dataset manifest repeats the same `session_splits` and
`dataset_revision`, and lists every dataset sample in `samples`. Each sample has
`sample_id`, `session_id`, `image_sha256`, `scenario`, `size`, `motion`, `occlusion`,
and `targets`. Sample IDs and image hashes cannot repeat across the dataset.

Each measured `samples` entry repeats that held-out sample's exact ground truth
and slice metadata and adds `inference_latency_ms` and `predictions`. Evaluation
must cover the complete held-out manifest. Targets and predictions contain
`class_id`, `bbox_xyxy` and `center_px` in full 1920x1080 pixels; predictions also
have finite confidence in `[0,1]`. Fully invisible targets should be excluded
from detection labels using the dataset's documented annotation policy; partial
occlusion remains a separately named slice. Empty-target frames are permitted.

The hashed parity report must record true `pytorch_onnx_parity_passed` and
`tensorrt_parity_passed`, a `golden_inputs_sha256`, and matching
`checkpoint_sha256`, `onnx_sha256`, and `engine_sha256`. Retain the measured raw
parity outputs with the external dataset for review.

The evaluator calculates precision/recall, actual median/mean/p95/max center
error, confidence calibration, scenario/size/motion/occlusion slices, failure
records, and measured latency p50/p95/p99/max. Numeric thresholds are precision
>=99.5%, recall >=99.0%, median center error <=2 px and inference p99 <=2.5 ms
under concurrent workload. Even passing these numeric gates never stamps the
entire milestone accepted: allocation, fault, lifetime, calibration, dataset and
hardware provenance evidence still require review.

## Explicit synthetic fixture

```powershell
uv run --project python python -m tools.probe.perception_benchmark --synthetic --iterations 25 --output-md D:/aim-assets/fixture-report.md --output-json D:/aim-assets/fixture-report.json
```

Synthetic mode tests report plumbing only. All acceptance gates remain false,
regardless of its fixture values. There is no default synthetic benchmark and no
automatic overwrite of historical milestone reports.
