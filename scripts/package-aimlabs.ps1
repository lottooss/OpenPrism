# Creates a local, machine-bound delivery. SDK DLLs stay in their installed folders.
[CmdletBinding()]
param(
    [string]$Engine,
    [string]$Binary,
    [string]$Scenario,
    [string]$Destination,
    [string]$ModelReport,
    [string]$ValidationReport,
    [string]$ModelLicense,
    [string]$TensorRtRoot = $env:TENSORRT_ROOT,
    [string]$CudaRoot = $env:CUDA_PATH,
    [string[]]$NativeDllDirectories = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$sourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Get-RequiredFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file missing: $Path" }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -eq 0) { throw "Required file is empty: $Path" }
    return $item.FullName
}

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose(); $stream.Dispose() }
}

function Write-ManagedText([string]$Path, [string]$Value) {
    $temp = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    [IO.File]::WriteAllText($temp, $Value, (New-Object Text.UTF8Encoding($false)))
    if (Test-Path -LiteralPath $Path) {
        [IO.File]::Replace($temp, $Path, "$Path.$([Guid]::NewGuid().ToString('N')).previous")
    } else { [IO.File]::Move($temp, $Path) }
}

function Read-Report([string]$Path) {
    if ((Get-Item -LiteralPath $Path).Length -gt 1048576) { throw 'Model/validation reports must not exceed 1 MiB.' }
    try {
        $report = [IO.File]::ReadAllText($Path) | ConvertFrom-Json
        if ($report -isnot [pscustomobject]) { throw 'Report must be a JSON object.' }
        return $report
    }
    catch { throw "Invalid report JSON: $Path" }
}

function Get-ReportField($Report, [string]$Name) {
    if ($null -eq $Report) { return $null }
    $property = $Report.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Copy-Immutable([string]$Source, [string]$Target, [string]$Hash) {
    if (Test-Path -LiteralPath $Target) {
        if ((Get-Sha256 $Target) -ne $Hash) { throw "Existing payload differs; preserving it: $Target" }
        return
    }
    [IO.File]::Copy($Source, $Target, $false)
    if ((Get-Sha256 $Target) -ne $Hash) { throw "Copied payload hash mismatch: $Target" }
}

try {
    if (-not $Engine) { $Engine = Join-Path $sourceRoot 'artifacts/aimlabs-refinement/best.fp32.optimized.engine' }
    if (-not $Binary) { $Binary = Join-Path $sourceRoot 'build/windows-msvc-release/aim_runtime_candidate.exe' }
    if (-not $Scenario) { $Scenario = Join-Path $sourceRoot 'configs/scenario/aimlabs.yaml' }
    if (-not $Destination) { $Destination = Join-Path $sourceRoot 'artifacts/aimlabs-delivery' }
    $Engine = Get-RequiredFile $Engine
    $Binary = Get-RequiredFile $Binary
    $Scenario = Get-RequiredFile $Scenario
    $evidenceFiles = [ordered]@{}
    $modelData = $null
    $validationData = $null
    if ($ModelReport) { $ModelReport = Get-RequiredFile $ModelReport; $modelData = Read-Report $ModelReport; $evidenceFiles.modelReport = $ModelReport }
    if ($ValidationReport) { $ValidationReport = Get-RequiredFile $ValidationReport; $validationData = Read-Report $ValidationReport; $evidenceFiles.validationReport = $ValidationReport }
    if ($ModelLicense) { $ModelLicense = Get-RequiredFile $ModelLicense; $evidenceFiles.modelLicense = $ModelLicense }
    $license = Get-ReportField $modelData 'license'
    $kind = Get-ReportField $modelData 'kind'
    foreach ($label in @($license, $kind)) {
        if ($null -ne $label -and ($label -isnot [string] -or $label -notmatch '^[A-Za-z0-9_.+ -]{1,100}$')) {
            throw 'Model report kind/license must be simple nonempty identifiers.'
        }
    }
    if ($license -like 'AGPL*' -and -not $ModelLicense) { throw 'Supply -ModelLicense with the existing license text for this AGPL model.' }
    $Destination = [IO.Path]::GetFullPath($Destination)
    if ([IO.Path]::GetPathRoot($Destination).TrimEnd('\', '/') -eq $Destination.TrimEnd('\', '/')) {
        throw 'The delivery directory must not be a drive root.'
    }
    if ($Destination.TrimEnd('\', '/') -eq $sourceRoot.TrimEnd('\', '/')) {
        throw 'Use a dedicated delivery directory, not the source checkout.'
    }
    $ancestor = $Destination
    while ($ancestor) {
        if ((Test-Path -LiteralPath $ancestor) -and
            ((Get-Item -LiteralPath $ancestor).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'The delivery directory must not traverse links or junctions.'
        }
        $ancestor = Split-Path -Parent $ancestor
    }

    # Explicit arguments/environment win; these fallbacks cover a per-user SDK install.
    if (-not $TensorRtRoot -and $env:AIM_TENSORRT_ROOT) { $TensorRtRoot = $env:AIM_TENSORRT_ROOT }
    $userTools = Join-Path $env:USERPROFILE 'tools'
    if (-not $TensorRtRoot -and (Test-Path -LiteralPath $userTools -PathType Container)) {
        $sdk = Get-ChildItem -LiteralPath $userTools -Directory -Filter 'TensorRT-*' |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($sdk) { $TensorRtRoot = $sdk.FullName }
    }
    if (-not $CudaRoot) { $CudaRoot = Join-Path $userTools 'cuda_toolkit' }
    $candidates = @($NativeDllDirectories) + @((Split-Path -Parent $Binary))
    if ($TensorRtRoot) { $candidates += @((Join-Path $TensorRtRoot 'lib'), (Join-Path $TensorRtRoot 'bin')) }
    if ($CudaRoot) { $candidates += Join-Path $CudaRoot 'bin' }
    $candidates += @((Join-Path $userTools 'cuda/libcublas/cublas/bin'), (Join-Path $userTools 'cuda/cuda_nvrtc/nvrtc/bin'))
    $candidates += Join-Path $env:WINDIR 'System32'
    $candidates += @($env:PATH -split ';')
    $dllDirectories = @($candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } |
        ForEach-Object { (Get-Item -LiteralPath $_).FullName } | Select-Object -Unique)
    $requiredDlls = @('nvinfer_10.dll', 'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll',
        'nvrtc64_120_0.dll', 'MSVCP140.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll')
    # Keep only dependency directories, avoiding unrelated PATH entries in the private manifest.
    $runtimeDirectories = @()
    foreach ($name in $requiredDlls) {
        $found = $dllDirectories | Where-Object { Test-Path -LiteralPath (Join-Path $_ $name) -PathType Leaf } |
            Select-Object -First 1
        if (-not $found) { throw "Missing $name. Supply -TensorRtRoot, -CudaRoot or -NativeDllDirectories." }
        $runtimeDirectories += $found
    }
    foreach ($dir in $dllDirectories) {
        $builtins = @(Get-ChildItem -LiteralPath $dir -Filter 'nvrtc-builtins64_*.dll' -File)
        foreach ($dll in $builtins) { $requiredDlls += $dll.Name; $runtimeDirectories += $dir }
        if ((Get-ChildItem -LiteralPath $dir -Filter 'nvrtc*.dll' -File | Select-Object -First 1) -or
            (Test-Path -LiteralPath (Join-Path $dir 'nvinfer_plugin_10.dll') -PathType Leaf)) {
            $runtimeDirectories += $dir
        }
    }
    if (-not @($requiredDlls | Where-Object { $_ -like 'nvrtc-builtins64_*.dll' }).Count) {
        throw 'NVRTC builtins DLL missing. Supply its installed folder in -NativeDllDirectories.'
    }
    $requiredDlls = @($requiredDlls | Select-Object -Unique)
    # Explicit directories may supply engine-specific plugins or VC runtime dependencies.
    foreach ($dir in $NativeDllDirectories) {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { throw "Native DLL directory missing: $dir" }
        $runtimeDirectories += (Get-Item -LiteralPath $dir).FullName
    }
    $runtimeDirectories = @($runtimeDirectories | Select-Object -Unique)
    if (@($runtimeDirectories | Where-Object { $_.Contains(';') }).Count) { throw 'Native DLL paths must not contain semicolons.' }

    $engineHash = Get-Sha256 $Engine
    if ($validationData -and (Get-ReportField $validationData 'engine_sha256') -cne $engineHash) {
        throw 'Validation report engine_sha256 does not match the packaged engine.'
    }
    $reportedModelEngine = Get-ReportField $modelData 'engine_sha256'
    if ($reportedModelEngine -and $reportedModelEngine -cne $engineHash) { throw 'Model report engine_sha256 does not match the packaged engine.' }
    $binaryHash = Get-Sha256 $Binary
    $scenarioHash = Get-Sha256 $Scenario
    $payloadRelative = 'payloads/' + $engineHash.Substring(0, 16) + '-' + $binaryHash.Substring(0, 16) + '-' + $scenarioHash.Substring(0, 12)
    $payload = Join-Path $Destination $payloadRelative
    [IO.Directory]::CreateDirectory($payload) | Out-Null
    Copy-Immutable $Engine (Join-Path $payload 'detector.engine') $engineHash
    Copy-Immutable $Binary (Join-Path $payload 'aim_runtime_candidate.exe') $binaryHash
    Copy-Immutable $Scenario (Join-Path $payload 'aimlabs.yaml') $scenarioHash
    $evidenceManifest = [ordered]@{}
    foreach ($entry in $evidenceFiles.GetEnumerator()) {
        $hash = Get-Sha256 $entry.Value
        $extension = '.json'
        if ($entry.Key -eq 'modelLicense') { $extension = '.txt' }
        $relative = "$payloadRelative/evidence/$($entry.Key)-$hash$extension"
        $target = Join-Path $Destination $relative
        [IO.Directory]::CreateDirectory((Split-Path -Parent $target)) | Out-Null
        Copy-Immutable $entry.Value $target $hash
        $evidenceManifest[$entry.Key] = $relative
        $evidenceManifest[$entry.Key + 'Sha256'] = $hash
    }
    [IO.Directory]::CreateDirectory((Join-Path $Destination 'scripts')) | Out-Null
    foreach ($relative in @('Start-Aimlabs.cmd', 'scripts/start-aimlabs.ps1', 'scripts/package-aimlabs.ps1')) {
        $target = Join-Path $Destination $relative
        if (Test-Path -LiteralPath $target) {
            if ((Get-Sha256 $target) -eq (Get-Sha256 (Join-Path $sourceRoot $relative))) { continue }
            [IO.File]::Copy($target, "$target.$([Guid]::NewGuid().ToString('N')).previous", $false)
        }
        [IO.File]::Copy((Join-Path $sourceRoot $relative), $target, $true)
    }
    $manifest = [ordered]@{
        schemaVersion = 1
        localMachineOnly = $true
        candidate = "$payloadRelative/aim_runtime_candidate.exe"
        candidateSha256 = $binaryHash
        engine = "$payloadRelative/detector.engine"
        engineSha256 = $engineHash
        scenario = "$payloadRelative/aimlabs.yaml"
        scenarioSha256 = $scenarioHash
        nativeDllDirectories = $runtimeDirectories
        requiredDlls = $requiredDlls
    }
    foreach ($entry in $evidenceManifest.GetEnumerator()) { $manifest[$entry.Key] = $entry.Value }
    Write-ManagedText (Join-Path $Destination 'runtime.json') ($manifest | ConvertTo-Json -Depth 12)

    $readme = @(
        '# Aimlabs local research candidate', '',
        'This is a local, machine-bound package. runtime.json references installed TensorRT, CUDA and Microsoft runtime folders. SDK binaries are not redistributed; copying this folder to another machine is insufficient.', '',
        'Start Start-Aimlabs.cmd. Confirm the current 1920 x 1080 horizontal FOV and sensitivity when prompted. If both -Fov and -Sensitivity are supplied, those values are treated as explicit confirmation of the current game settings.', '',
        'Run performs measured visual calibration first when a matching profile and evidence are unavailable. Calibration uses small mouse movements without shooting. Focus Aimlabs on visible stationary targets and press F11. Once calibration succeeds, focus Aimlabs and press F11 again for the run. F12 stops and releases input. The default run lasts 30 seconds; use -Duration to change it.', '',
        'Modes: -Mode Preflight checks real capture/inference without input; -Mode Observe observes without calibration or actuation; -Mode Calibrate creates a new measured profile; -Mode Run is the default. Profiles and their evidence are preserved. Settings or model changes require a new matching calibration.', '',
        '## Evidence and limits', '',
        "Engine SHA-256: $engineHash", ''
    )
    if ($kind) { $readme += "Supplied model kind: $kind." } else { $readme += 'No model provenance kind was supplied.' }
    if ($kind -eq 'synthetic-bootstrap-candidate') {
        $trainingCount = Get-ReportField $modelData 'training_samples'
        $validationCount = Get-ReportField $modelData 'validation_samples'
        if ($trainingCount -is [ValueType] -and [double]$trainingCount -gt 0) { $readme += "The model report records $trainingCount procedural training images." }
        if ($validationCount -is [ValueType] -and [double]$validationCount -gt 0) { $readme += "The model report records $validationCount procedural validation images." }
        $readme += 'Procedural examples are not a real held-out Aimlabs accuracy evaluation. This bootstrap model is a research candidate.'
    }
    $readme += 'Export/TensorRT parity and successful preflight do not certify real-scene detector accuracy, aiming accuracy, or full-system latency. This package does not establish the required 30-minute concurrent Aimlabs acceptance run.'
    if ($ValidationReport) { $readme += 'The supplied validation report is copied verbatim and bound to this engine hash; its measurements and limitations remain authoritative.' }
    if ($license) { $readme += "Model license declared by the supplied report: $license." }
    else { $readme += 'No model license identifier was supplied; no license classification is inferred.' }
    if ($license -like 'AGPL*') { $readme += 'The AGPL model and its accompanying license/provenance remain isolated as model artifacts. This local packaging step does not establish permission for a proprietary transition or redistribution.' }
    if ($ModelLicense) { $readme += 'The supplied model license text is included unchanged.' }
    $readme += ''
    foreach ($entry in $evidenceManifest.GetEnumerator()) {
        if ($entry.Key -notlike '*Sha256') { $readme += "- $($entry.Key): $($entry.Value)" }
    }
    Write-ManagedText (Join-Path $Destination 'README.md') (($readme -join [Environment]::NewLine) + [Environment]::NewLine)
    Write-Host "Local delivery ready: $Destination"
    Write-Host 'Installed SDK folders are referenced privately; this package is not portable to another machine.'
    Write-Host 'Run Start-Aimlabs.cmd -Mode Preflight for capture/inference validation without input.'
    Write-Host 'Run Start-Aimlabs.cmd for confirmed settings, measured calibration, then an F11-gated run.'
} catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
