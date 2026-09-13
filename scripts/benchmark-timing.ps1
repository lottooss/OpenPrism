# PowerShell runner for Clock and Stage-Timing Benchmark
[CmdletBinding()]
param(
    [int]$Iterations = 100000,
    [int]$Warmup = 10000
)

$ErrorActionPreference = 'Stop'

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python 3 is required but was not found on PATH."
    exit 1
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$benchScript = Join-Path $repoRoot "tools\timing\benchmark.py"

if (-not (Test-Path -LiteralPath $benchScript -PathType Leaf)) {
    Write-Error "Benchmark script not found at: $benchScript"
    exit 1
}

& $pythonCmd.Source $benchScript --iterations $Iterations --warmup $Warmup
exit $LASTEXITCODE
