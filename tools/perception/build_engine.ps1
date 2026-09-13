param(
    [Parameter(Mandatory)][string]$OnnxPath,
    [Parameter(Mandatory)][ValidatePattern('^[a-fA-F0-9]{64}$')][string]$OnnxSha256,
    [Parameter(Mandatory)][string]$EnginePath,
    [Parameter(Mandatory)][string]$ValidationExecutable,
    [Parameter(Mandatory)][string]$GoldenInputs,
    [Parameter(Mandatory)][string]$GoldenReference,
    [Parameter(Mandatory)][string]$ReportPath,
    [Parameter(Mandatory)][string]$TensorRtSdkRoot,
    [Parameter(Mandatory)][string]$CudaRoot,
    [string[]]$AdditionalDllDirectories = @(),
    [ValidateSet('fp16', 'fp32')][string]$InternalPrecision = 'fp32',
    [ValidateRange(0,5)][int]$BuilderOptimizationLevel = 0,
    [ValidateRange(1,128)][int]$Frames = 8,
    [ValidateRange(1,1000)][int]$Warmup = 50,
    [ValidateRange(1,1000)][int]$Iterations = 20
)
# Offline local build/evaluation; no package installation, training, input actuation or CI waits.
$ErrorActionPreference = 'Stop'
$onnx = (Resolve-Path -LiteralPath $OnnxPath).Path
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $onnx).Hash -ine $OnnxSha256) { throw 'Trusted ONNX SHA-256 mismatch' }
$trtexec = Join-Path $TensorRtSdkRoot 'bin/trtexec.exe'
$validator = (Resolve-Path -LiteralPath $ValidationExecutable).Path
$inputs = (Resolve-Path -LiteralPath $GoldenInputs).Path
$reference = (Resolve-Path -LiteralPath $GoldenReference).Path
$engine = [IO.Path]::GetFullPath($EnginePath)
$report = [IO.Path]::GetFullPath($ReportPath)
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($engine)) | Out-Null
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($report)) | Out-Null
$buildLog = "$engine.build.log"
$actual = "$report.actual.fp32.bin"
$savedSearchPath = $env:PATH
try {
    $dllDirectories = @((Join-Path $TensorRtSdkRoot 'lib'), (Join-Path $CudaRoot 'bin')) + $AdditionalDllDirectories
    $env:PATH = ($dllDirectories -join ';') + ';' + $savedSearchPath
    $buildArguments = @("--onnx=$onnx", "--saveEngine=$engine", '--inputIOFormats=fp16:chw', '--outputIOFormats=fp32:chw', '--memPoolSize=workspace:128', '--skipInference', "--builderOptimizationLevel=$BuilderOptimizationLevel")
    if ($InternalPrecision -eq 'fp16') { $buildArguments += '--fp16' }
    else { $buildArguments += '--noTF32' }
    & $trtexec @buildArguments *> $buildLog
    if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $buildLog -Tail 25; throw "TensorRT build failed; inspect $buildLog" }
    $engineHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $engine).Hash.ToLowerInvariant()
    $buildManifest = [ordered]@{
        schema_version = 1; onnx_sha256 = $OnnxSha256.ToLowerInvariant(); engine_sha256 = $engineHash
        engine_path = $engine; trtexec_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $trtexec).Hash.ToLowerInvariant()
        tensorrt_sdk_root = $TensorRtSdkRoot; arguments = $buildArguments
        internal_precision = $InternalPrecision
        input_shape = @(1,3,384,640); input_dtype = 'float16'; output_shape = @(1,5,5040); output_dtype = 'float32'
        golden_inputs_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $inputs).Hash.ToLowerInvariant()
        reference_outputs_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $reference).Hash.ToLowerInvariant()
        center_tolerance_model_px = 0.25; confidence_tolerance = 0.005
        detector_accuracy_certified = $false; milestone_acceptance_passed = $false
    }
    $buildManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$engine.build.json" -Encoding utf8
    & $validator "--engine=$engine" "--sha256=$engineHash" "--inputs=$inputs" "--reference=$reference" "--frames=$Frames" "--actual=$actual" "--report=$report" "--warmup=$Warmup" "--iterations=$Iterations"
    $validationExit = $LASTEXITCODE
    Write-Output "Engine: $engine"
    Write-Output "SHA256: $engineHash"
    Write-Output "Validation report: $report"
    if ($validationExit -ne 0) { throw "TensorRT validation failed (exit $validationExit); thresholds were not changed. Inspect report and actual outputs." }
} finally {
    $env:PATH = $savedSearchPath
}
