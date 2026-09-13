# PowerShell wrapper for Hardware & Display/GPU Routing Probe
[CmdletBinding()]
param(
    [string]$OutputPath = "reports/local/environment_manifest.json",
    [switch]$JsonOnly,
    [switch]$CheckZeroCopy
)

$ErrorActionPreference = 'Stop'

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python 3 is required but was not found on PATH."
    exit 1
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$probeScript = Join-Path $repoRoot "tools\probe\hardware_probe.py"

if (-not (Test-Path -LiteralPath $probeScript -PathType Leaf)) {
    Write-Error "Probe script not found at: $probeScript"
    exit 1
}

$argsList = @($probeScript)

if ($OutputPath) {
    $resolvedOutput = Join-Path $repoRoot $OutputPath
    $argsList += @("--output", $resolvedOutput)
}

if ($JsonOnly) {
    $argsList += "--json-only"
}

if ($CheckZeroCopy) {
    $argsList += "--check-zero-copy"
}

& $pythonCmd.Source @argsList
exit $LASTEXITCODE
