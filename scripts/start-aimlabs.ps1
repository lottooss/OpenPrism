[CmdletBinding()]
param(
    [ValidateSet('Run', 'Observe', 'Calibrate', 'Preflight')][string]$Mode = 'Run',
    [string]$PackageDirectory,
    [ValidateRange(1, 300)][int]$Duration = 30,
    # Supplying both values confirms they are currently applied in Aimlabs.
    # Otherwise an unconfigured or changed session asks through Read-Host.
    [double]$Fov = 0,
    [double]$Sensitivity = 0,
    [string]$AimlabsSettingsPath = (Join-Path $env:USERPROFILE 'AppData/LocalLow/Statespace/aimlab_tb/UserDefault/WindowsPlayer_PlayerSettingsData.json')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$invariant = [Globalization.CultureInfo]::InvariantCulture
$explicitSettings = $PSBoundParameters.ContainsKey('Fov') -and $PSBoundParameters.ContainsKey('Sensitivity')
$jsonArguments = @{}
# PowerShell 7.5 otherwise converts ISO strings into DateTime, changing the profile contract.
if ((Get-Command ConvertFrom-Json).Parameters.ContainsKey('DateKind')) { $jsonArguments.DateKind = 'String' }

function Read-JsonFile([string]$Path, [long]$MaximumBytes = 32768) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or (Get-Item -LiteralPath $Path).Length -gt $MaximumBytes) {
        throw "Missing or oversized JSON file: $Path"
    }
    try { return [IO.File]::ReadAllText($Path) | ConvertFrom-Json @jsonArguments }
    catch { throw "Invalid JSON file: $Path" }
}

function Get-Field($Object, [string]$Name) {
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose(); $stream.Dispose() }
}

function Test-JsonNumber($Value) {
    return ($Value -is [ValueType] -and $Value -isnot [bool] -and
        -not [double]::IsNaN([double]$Value) -and -not [double]::IsInfinity([double]$Value))
}

function Get-PackagePath([string]$Relative, [switch]$AllowMissing) {
    if (-not $Relative -or [IO.Path]::IsPathRooted($Relative)) { throw 'Package paths must be relative.' }
    $path = [IO.Path]::GetFullPath((Join-Path $PackageDirectory $Relative))
    $prefix = $PackageDirectory.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Package path escapes the delivery directory.' }
    # A junction could defeat the lexical containment check, including for a new profile.
    $cursor = $path
    while ($cursor -and $cursor.Length -ge $PackageDirectory.Length) {
        if ((Test-Path -LiteralPath $cursor) -and
            ((Get-Item -LiteralPath $cursor).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Package payload and profile paths must not traverse links or junctions.'
        }
        if ($cursor -eq $PackageDirectory) { break }
        $cursor = Split-Path -Parent $cursor
    }
    if (-not $AllowMissing -and -not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Package file missing: $Relative" }
    return $path
}

function Save-Settings($Value) {
    $temp = "$settingsPath.$([Guid]::NewGuid().ToString('N')).tmp"
    [IO.File]::WriteAllText($temp, ($Value | ConvertTo-Json -Depth 12), (New-Object Text.UTF8Encoding($false)))
    if (Test-Path -LiteralPath $settingsPath) {
        [IO.File]::Replace($temp, $settingsPath, "$settingsPath.$([Guid]::NewGuid().ToString('N')).previous")
    } else { [IO.File]::Move($temp, $settingsPath) }
}

function Parse-Number($Value, [double]$Minimum, [double]$Maximum, [string]$Label) {
    $number = 0.0
    if (-not [double]::TryParse([string]$Value, [Globalization.NumberStyles]::Float, $invariant, [ref]$number) -or
        [double]::IsNaN($number) -or [double]::IsInfinity($number) -or $number -lt $Minimum -or $number -gt $Maximum) {
        throw "$Label must be a finite number between $Minimum and $Maximum (use a decimal point)."
    }
    return [single]$number
}

function Read-Number([string]$Label, $Suggested, [double]$Minimum, [double]$Maximum) {
    $suffix = ''
    if ($null -ne $Suggested) { $suffix = ' [' + ([single]$Suggested).ToString('R', $invariant) + ']' }
    $answer = Read-Host "$Label$suffix"
    if (-not $answer -and $null -ne $Suggested) { $answer = ([single]$Suggested).ToString('R', $invariant) }
    return Parse-Number $answer $Minimum $Maximum $Label
}

function Read-AimlabsSuggestions {
    if (-not (Test-Path -LiteralPath $AimlabsSettingsPath -PathType Leaf)) { return $null }
    # Never print the source document or enumerate otherSettings: it can contain credentials.
    $document = Read-JsonFile $AimlabsSettingsPath 4194304
    $controls = Get-Field $document 'controlSettings'
    $graphics = Get-Field $document 'graphicsSettings'
    $vertical = Get-Field $controls 'lookVertFov'
    $mouse = Get-Field $controls 'mouseSensitivityX'
    if ($null -eq $vertical -or $null -eq $mouse) { return $null }
    $verticalNumber = Parse-Number $vertical 1 179 'Aimlabs vertical FOV'
    $mouseNumber = Parse-Number $mouse 0.001 100 'Aimlabs sensitivity'
    $width = Get-Field $graphics 'screenWidth'
    $height = Get-Field $graphics 'screenHeight'
    if (($null -ne $width -and [int]$width -ne 1920) -or ($null -ne $height -and [int]$height -ne 1080)) {
        throw 'Aimlabs graphics settings must be 1920 x 1080 before using this delivery.'
    }
    $horizontal = 2 * [Math]::Atan([Math]::Tan($verticalNumber * [Math]::PI / 360) * (1920.0 / 1080.0)) * 180 / [Math]::PI
    # The fingerprint contains only settings which affect this session's calibration.
    $selected = [ordered]@{
        lookVertFov = $verticalNumber
        mouseSensitivityX = $mouseNumber
        mouseSensitivityY = Get-Field $controls 'mouseSensitivityY'
        gameProfile = Get-Field $controls 'gameProfile'
        screenWidth = $width
        screenHeight = $height
    }
    $hash = [Security.Cryptography.SHA256]::Create()
    try { $fingerprint = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($selected | ConvertTo-Json -Compress)))).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose() }
    return @{ fov = [single]$horizontal; sensitivity = $mouseNumber; fingerprint = $fingerprint }
}

function Test-Calibration([string]$Path, [string]$ExpectedProfileHash = '', [string]$ExpectedEvidenceHash = '') {
    try {
        $profile = Read-JsonFile $Path 16384
        $evidence = Read-JsonFile "$Path.evidence.json"
        if ($ExpectedProfileHash -and (Get-Sha256 $Path) -ne $ExpectedProfileHash) { return $false }
        if ($ExpectedEvidenceHash -and (Get-Sha256 "$Path.evidence.json") -ne $ExpectedEvidenceHash) { return $false }
        $keys = @('schema_version', 'profile_id', 'calibrated_at', 'resolution_width', 'resolution_height',
            'fov_horizontal_deg', 'in_game_sensitivity', 'counts_per_pixel_x', 'counts_per_pixel_y',
            'deadband_counts', 'nonlinearity_alpha', 'cross_coupling_xy', 'rmse_pixels')
        if (@($profile.PSObject.Properties).Count -ne 13 -or @($evidence.PSObject.Properties).Count -ne 10 -or
            $profile.schema_version -ne 1 -or $evidence.schema_version -ne 1 -or
            $profile.resolution_width -ne 1920 -or $profile.resolution_height -ne 1080 -or
            [single]$profile.fov_horizontal_deg -ne $sessionFov -or [single]$profile.in_game_sensitivity -ne $sessionSensitivity -or
            $evidence.engine_sha256 -cne $engineHash -or $profile.profile_id -cnotmatch '^[A-Za-z0-9_-]{3,64}$' -or
            -not $profile.calibrated_at) { return $false }
        foreach ($key in $keys) {
            if ($null -eq (Get-Field $profile $key) -or (Get-Field $profile $key) -cne (Get-Field $evidence.profile $key)) { return $false }
            if ($key -ne 'profile_id' -and $key -ne 'calibrated_at' -and -not (Test-JsonNumber (Get-Field $profile $key))) { return $false }
        }
        if ($profile.profile_id -isnot [string] -or $profile.calibrated_at -isnot [string] -or
            @($evidence.profile.PSObject.Properties).Count -ne 13) { return $false }
        foreach ($key in @('schema_version', 'training_samples', 'held_out_samples', 'maximum_step_counts',
            'effect_lower_p50_ns', 'effect_upper_p50_ns', 'effect_upper_p95_ns', 'largest_effect_bracket_ns')) {
            $value = Get-Field $evidence $key
            if (-not (Test-JsonNumber $value) -or [double]$value -ne [Math]::Truncate([double]$value)) { return $false }
        }
        $ranges = @{
            counts_per_pixel_x = @(0.001, 100); counts_per_pixel_y = @(0.001, 100)
            deadband_counts = @(0, 10); nonlinearity_alpha = @(-1, 1)
            cross_coupling_xy = @(-0.5, 0.5); rmse_pixels = @(0, 5)
        }
        foreach ($key in $ranges.Keys) {
            Parse-Number (Get-Field $profile $key) $ranges[$key][0] $ranges[$key][1] $key | Out-Null
        }
        if ($evidence.training_samples -lt 12 -or $evidence.held_out_samples -lt 8 -or
            $evidence.maximum_step_counts -lt 1 -or $evidence.maximum_step_counts -gt 64 -or
            $evidence.effect_lower_p50_ns -lt 0 -or $evidence.effect_upper_p50_ns -le 0 -or
            $evidence.effect_upper_p50_ns -lt $evidence.effect_lower_p50_ns -or
            $evidence.effect_upper_p95_ns -lt $evidence.effect_upper_p50_ns -or $evidence.effect_upper_p95_ns -gt 100000000 -or
            $evidence.largest_effect_bracket_ns -le 0 -or $evidence.largest_effect_bracket_ns -gt 100000000) { return $false }
        return $true
    } catch { return $false }
}

function Invoke-Candidate([string[]]$Flags, [string]$ProfilePath = '') {
    $reportDirectory = Get-PackagePath 'reports' -AllowMissing
    [IO.Directory]::CreateDirectory($reportDirectory) | Out-Null
    $reportName = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss') + '-' + [Guid]::NewGuid().ToString('N') + '.json'
    $reportPath = Join-Path $reportDirectory $reportName
    $nativeArgs = @("--assets=$PackageDirectory", "--engine=$engine", "--sha256=$engineHash", "--scenario=$scenario",
        "--fov=$($sessionFov.ToString('R', $invariant))", "--sensitivity=$($sessionSensitivity.ToString('R', $invariant))", "--duration=$Duration", "--report=$reportPath")
    if ($ProfilePath) { $nativeArgs += "--calibration=$ProfilePath" }
    $nativeArgs += $Flags
    $oldPath = $env:PATH
    try {
        $env:PATH = (@($runtime.nativeDllDirectories) + @($oldPath)) -join ';'
        & $candidate @nativeArgs | Out-Host
        return $LASTEXITCODE
    } finally { $env:PATH = $oldPath }
}

try {
    if (-not $PackageDirectory) {
        $base = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
        if (Test-Path -LiteralPath (Join-Path $base 'runtime.json') -PathType Leaf) { $PackageDirectory = $base }
        else { $PackageDirectory = Join-Path $base 'artifacts/aimlabs-delivery' }
    }
    $PackageDirectory = [IO.Path]::GetFullPath($PackageDirectory).TrimEnd('\', '/')
    $runtime = Read-JsonFile (Join-Path $PackageDirectory 'runtime.json')
    if ($runtime.schemaVersion -ne 1 -or -not $runtime.localMachineOnly) { throw 'Unsupported local runtime manifest.' }
    $candidate = Get-PackagePath $runtime.candidate
    $engine = Get-PackagePath $runtime.engine
    $scenario = Get-PackagePath $runtime.scenario
    foreach ($entry in @(@($candidate, $runtime.candidateSha256), @($engine, $runtime.engineSha256), @($scenario, $runtime.scenarioSha256))) {
        if ($entry[1] -cnotmatch '^[0-9a-f]{64}$' -or (Get-Sha256 $entry[0]) -cne $entry[1]) { throw 'Delivery payload hash mismatch; package the trusted build again.' }
    }
    foreach ($name in @('modelReport', 'validationReport', 'modelLicense')) {
        $relative = Get-Field $runtime $name
        $expected = Get-Field $runtime ($name + 'Sha256')
        if ($null -ne $relative -or $null -ne $expected) {
            $path = Get-PackagePath $relative
            if ($expected -cnotmatch '^[0-9a-f]{64}$' -or (Get-Sha256 $path) -cne $expected) {
                throw 'Packaged model evidence hash mismatch; package the trusted evidence again.'
            }
        }
    }
    $engineHash = $runtime.engineSha256
    foreach ($dir in $runtime.nativeDllDirectories) {
        if ($dir.Contains(';') -or -not [IO.Path]::IsPathRooted($dir) -or -not (Test-Path -LiteralPath $dir -PathType Container)) {
            throw 'An installed native SDK directory is unavailable; package this machine again.'
        }
    }
    foreach ($name in $runtime.requiredDlls) {
        if ([IO.Path]::GetFileName($name) -ne $name -or -not (@($runtime.nativeDllDirectories | Where-Object { Test-Path -LiteralPath (Join-Path $_ $name) -PathType Leaf }).Count)) {
            throw "Required native DLL missing: $name"
        }
    }
    $settingsPath = Get-PackagePath 'user-settings.json' -AllowMissing
    $settings = $null
    if (Test-Path -LiteralPath $settingsPath) {
        $settings = Read-JsonFile $settingsPath
        if ($settings.schemaVersion -ne 1) { throw 'Unsupported saved settings; preserve the file and use another package directory.' }
    }
    $suggested = Read-AimlabsSuggestions
    $fingerprint = ''
    if ($suggested) { $fingerprint = $suggested.fingerprint }
    $needsConfirmation = -not $fingerprint -or $null -eq $settings -or (Get-Field $settings 'aimlabsSettingsFingerprint') -cne $fingerprint
    if ($explicitSettings -or $Fov -ne 0 -or $Sensitivity -ne 0) { $needsConfirmation = $true }
    if ($needsConfirmation) {
        Write-Host 'Using Aimlabs at 1920 x 1080. No game settings are modified.'
        $fovSuggestion = Get-Field $settings 'fov'
        $sensitivitySuggestion = Get-Field $settings 'sensitivity'
        if ($suggested) { $fovSuggestion = $suggested.fov; $sensitivitySuggestion = $suggested.sensitivity }
        if ($Fov -ne 0) { $fovSuggestion = Parse-Number $Fov 30 150 'Horizontal FOV' }
        if ($Sensitivity -ne 0) { $sensitivitySuggestion = Parse-Number $Sensitivity 0.001 100 'Sensitivity' }
        if ($explicitSettings) {
            $sessionFov = Parse-Number $Fov 30 150 'Explicit horizontal FOV'
            $sessionSensitivity = Parse-Number $Sensitivity 0.001 100 'Explicit sensitivity'
            Write-Host 'Using the current settings explicitly confirmed through -Fov and -Sensitivity.'
        } else {
            $sessionFov = Read-Number 'Horizontal FOV in degrees' $fovSuggestion 30 150
            $sessionSensitivity = Read-Number 'In-game mouse sensitivity X' $sensitivitySuggestion 0.001 100
            if ((Read-Host 'Are these the settings currently applied in Aimlabs? Type YES to confirm') -cne 'YES') { throw 'Settings were not confirmed; no input was requested.' }
        }
        $previousCalibration = $null
        if ($settings -and (Get-Field $settings 'aimlabsSettingsFingerprint') -ceq $fingerprint -and
            [single]$settings.fov -eq $sessionFov -and [single]$settings.sensitivity -eq $sessionSensitivity) {
            $previousCalibration = Get-Field $settings 'calibration'
        }
        $settings = [pscustomobject][ordered]@{
            schemaVersion = 1; fov = $sessionFov; sensitivity = $sessionSensitivity
            aimlabsSettingsFingerprint = $fingerprint; calibration = $previousCalibration
        }
        Save-Settings $settings
    } else {
        $sessionFov = Parse-Number $settings.fov 30 150 'Saved horizontal FOV'
        $sessionSensitivity = Parse-Number $settings.sensitivity 0.001 100 'Saved sensitivity'
    }
    if ($Mode -eq 'Observe' -or $Mode -eq 'Preflight') {
        $flags = @('--observe')
        if ($Mode -eq 'Preflight') { $flags += '--preflight' }
        exit (Invoke-Candidate $flags)
    }
    $calibrationPath = ''
    $pointer = Get-Field $settings 'calibration'
    if ($pointer -and $pointer.engineSha256 -ceq $engineHash) {
        $previous = Get-PackagePath $pointer.path -AllowMissing
        Get-PackagePath ($pointer.path + '.evidence.json') -AllowMissing | Out-Null
        if (Test-Calibration $previous $pointer.profileSha256 $pointer.evidenceSha256) { $calibrationPath = $previous }
    }
    if ($Mode -eq 'Calibrate' -or -not $calibrationPath) {
        $relative = 'profiles/' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss') + '-' + [Guid]::NewGuid().ToString('N') + '.json'
        $newProfile = Get-PackagePath $relative -AllowMissing
        [IO.Directory]::CreateDirectory((Split-Path -Parent $newProfile)) | Out-Null
        Write-Host 'Measured calibration is required. Focus Aimlabs on visible stationary targets and press F11; F12 stops.'
        Write-Host 'Calibration uses small mouse movements and does not shoot. Existing profiles are preserved.'
        $result = Invoke-Candidate @('--calibrate') $newProfile
        if ($result -ne 0) { exit $result }
        if (-not (Test-Calibration $newProfile)) { throw 'Calibration returned success without a valid matching profile and evidence; run was not started.' }
        $settings.calibration = [pscustomobject]@{
            path = $relative; engineSha256 = $engineHash
            profileSha256 = Get-Sha256 $newProfile
            evidenceSha256 = Get-Sha256 "$newProfile.evidence.json"
        }
        Save-Settings $settings
        $calibrationPath = $newProfile
    }
    if ($Mode -eq 'Calibrate') { Write-Host "Measured profile saved: $calibrationPath"; exit 0 }
    Write-Host 'Run is ready. Focus Aimlabs and press F11 to start; F12 stops and releases input.'
    exit (Invoke-Candidate @('--actuate') $calibrationPath)
} catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
