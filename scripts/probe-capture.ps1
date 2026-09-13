# PowerShell runner for DXGI & WGC Capture Capability Probe
[CmdletBinding()]
param(
    [string]$OutputPath = "reports/local/capture_report.json",
    [switch]$JsonOnly
)

$ErrorActionPreference = 'Stop'

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python 3 is required but was not found on PATH."
    exit 1
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$probeScript = Join-Path $repoRoot "tools\probe\capture_probe.py"

if (-not (Test-Path -LiteralPath $probeScript -PathType Leaf)) {
    Write-Error "Capture probe script not found at: $probeScript"
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

& $pythonCmd.Source @argsList
exit $LASTEXITCODE
