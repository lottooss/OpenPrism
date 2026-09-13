[CmdletBinding()]
param(
    [string]$Flatc = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$schemaRoot = Join-Path $repoRoot "schemas\bus"
$releasedRoot = Join-Path $schemaRoot "released\v1"

if (-not $Flatc) {
    $fromPath = Get-Command flatc -ErrorAction SilentlyContinue
    if ($fromPath) {
        $Flatc = $fromPath.Source
    }
    else {
        $Flatc = Join-Path $repoRoot "build\vcpkg_installed\x64-windows\tools\flatbuffers\flatc.exe"
    }
}
$flatcExecutable = (Resolve-Path -LiteralPath $Flatc).Path

$schemaFiles = @(Get-ChildItem -LiteralPath $schemaRoot -Filter "*.fbs" -File | Sort-Object Name)
foreach ($schemaFile in $schemaFiles) {
    $releasedSchema = Join-Path $releasedRoot $schemaFile.Name
    if (-not (Test-Path -LiteralPath $releasedSchema)) {
        throw "Missing released v1 baseline: $releasedSchema"
    }

    & $flatcExecutable --conform $releasedSchema --conform-includes $releasedRoot -I $schemaRoot $schemaFile.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Schema conformance failed for $($schemaFile.Name)"
    }
}

Write-Output "All $($schemaFiles.Count) canonical bus schemas conform to the released v1 baseline."
