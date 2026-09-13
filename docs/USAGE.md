# Running locally

Build and test the core using the root README. Live operation additionally needs a compatible NVIDIA GPU, CUDA/TensorRT, a validated engine and local calibration. The tested SDK combination was TensorRT 10.8 with CUDA 12.6; compatibility with other versions must be checked locally.

Configure the optional native backend with `AIM_TENSORRT_ROOT` pointing to the TensorRT SDK when configuring CMake. Keep SDK DLL directories available to the local process. Do not copy proprietary SDKs into source control.

`scripts/package-aimlabs.ps1` creates a machine-bound package from an engine, native binary, scenario, model/validation reports and model license. Inspect its named parameters and supply paths to your independently validated artifacts. Packaging does not train a model or certify it. No ready-made engine is shipped here.

After creating the local package, launch Aimlabs in the configured controlled scenario, then use the root launcher:

```powershell
.\Start-Aimlabs.cmd -Mode Preflight
.\Start-Aimlabs.cmd -Mode Calibrate
.\Start-Aimlabs.cmd -Duration 30
```

Confirm the currently applied FOV and sensitivity when prompted. Keep resolution and display configuration consistent with the scenario. Calibration is local and must be repeated when relevant settings change. Preflight uses no physical input; calibration and Run require the authorized game window and explicit F11 arming. F12 or Escape stops and releases input.

Start with short runs and inspect the generated reports. Preserve all safety gates. Slow acquisition, stale-frame rejection or parity failures should be investigated through recorded metrics, not hidden by relaxed thresholds.

Hardware probes collect local hostnames, adapter/display identifiers and environment details for diagnosis. Review and redact those fields, absolute paths and any account information before sharing a generated report. The repository's examples use synthetic host identifiers.
