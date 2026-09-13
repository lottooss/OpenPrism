[CmdletBinding()]
param(
    [string]$Flatc = "",
    [switch]$Check
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$expectedFlatcVersion = "25.12.19"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$schemaRoot = Join-Path $repoRoot "schemas\bus"
$cppDestination = Join-Path $repoRoot "generated\flatbuffers\cpp"
$pythonDestination = Join-Path $repoRoot "python\aim\bus\v1"
$stagingRoot = Join-Path $repoRoot ("build\schema-codegen\" + [Guid]::NewGuid().ToString("N"))
$cppStaging = Join-Path $stagingRoot "cpp"
$pythonStaging = Join-Path $stagingRoot "python"

function Resolve-Flatc {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    $fromPath = Get-Command flatc -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $vcpkgCandidate = Join-Path $repoRoot "build\vcpkg_installed\x64-windows\tools\flatbuffers\flatc.exe"
    if (Test-Path -LiteralPath $vcpkgCandidate) {
        return (Resolve-Path -LiteralPath $vcpkgCandidate).Path
    }

    throw "flatc was not found. Pass -Flatc or configure a Windows preset to install the pinned vcpkg manifest first."
}

function Get-GeneratedFiles {
    param(
        [string]$Root,
        [string[]]$Extensions
    )

    if (-not (Test-Path -LiteralPath $Root)) {
        return @()
    }

    return @(Get-ChildItem -LiteralPath $Root -File | Where-Object { $Extensions -contains $_.Extension } | Sort-Object Name)
}

function Normalize-GeneratedText {
    param(
        [string]$Root,
        [string[]]$Extensions
    )

    $utf8WithoutBom = [Text.UTF8Encoding]::new($false)
    foreach ($file in (Get-GeneratedFiles -Root $Root -Extensions $Extensions)) {
        $content = [IO.File]::ReadAllText($file.FullName)
        $normalized = ($content -replace "`r`n?", "`n").TrimEnd([char[]]@("`r", "`n")) + "`n"
        [IO.File]::WriteAllText($file.FullName, $normalized, $utf8WithoutBom)
    }
}

function Sync-Or-CheckDirectory {
    param(
        [string]$Staging,
        [string]$Destination,
        [string[]]$Extensions,
        [switch]$VerifyOnly
    )

    $stagedFiles = Get-GeneratedFiles -Root $Staging -Extensions $Extensions
    if ($stagedFiles.Count -eq 0) {
        throw "flatc produced no expected files under $Staging"
    }

    $destinationFiles = Get-GeneratedFiles -Root $Destination -Extensions $Extensions
    $stagedNames = @($stagedFiles | ForEach-Object Name)
    $destinationNames = @($destinationFiles | ForEach-Object Name)

    if ($VerifyOnly) {
        $nameDifference = Compare-Object -ReferenceObject $stagedNames -DifferenceObject $destinationNames
        if ($nameDifference) {
            throw "Generated file set is stale under $Destination`n$($nameDifference | Out-String)"
        }

        foreach ($stagedFile in $stagedFiles) {
            $destinationFile = Join-Path $Destination $stagedFile.Name
            $stagedHash = (Get-FileHash -LiteralPath $stagedFile.FullName -Algorithm SHA256).Hash
            $destinationHash = (Get-FileHash -LiteralPath $destinationFile -Algorithm SHA256).Hash
            if ($stagedHash -ne $destinationHash) {
                throw "Generated file is stale: $destinationFile"
            }
        }
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($destinationFile in $destinationFiles) {
        if ($stagedNames -notcontains $destinationFile.Name) {
            Remove-Item -LiteralPath $destinationFile.FullName -Force
        }
    }
    foreach ($stagedFile in $stagedFiles) {
        Copy-Item -LiteralPath $stagedFile.FullName -Destination (Join-Path $Destination $stagedFile.Name) -Force
    }
}

$flatcExecutable = Resolve-Flatc -RequestedPath $Flatc
$versionOutput = (& $flatcExecutable --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $versionOutput -ne "flatc version $expectedFlatcVersion") {
    throw "Expected flatc $expectedFlatcVersion, got '$versionOutput' from $flatcExecutable"
}

New-Item -ItemType Directory -Force -Path $cppStaging, $pythonStaging | Out-Null
$schemaFiles = @(Get-ChildItem -LiteralPath $schemaRoot -Filter "*.fbs" -File | Sort-Object Name | ForEach-Object Name)
if ($schemaFiles.Count -ne 7) {
    throw "Expected seven canonical bus schemas, found $($schemaFiles.Count)."
}

Push-Location $schemaRoot
try {
    & $flatcExecutable --cpp --cpp-std c++17 --scoped-enums --filename-suffix _generated -I . -o $cppStaging $schemaFiles
    if ($LASTEXITCODE -ne 0) {
        throw "C++ FlatBuffers generation failed with exit code $LASTEXITCODE"
    }

    & $flatcExecutable --python --python-typing --python-version 3.12 -I . -o $pythonStaging $schemaFiles
    if ($LASTEXITCODE -ne 0) {
        throw "Python FlatBuffers generation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$generatedPythonPackage = Join-Path $pythonStaging "aim\bus\v1"
Normalize-GeneratedText -Root $cppStaging -Extensions @(".h")
Normalize-GeneratedText -Root $generatedPythonPackage -Extensions @(".py", ".pyi")
Sync-Or-CheckDirectory -Staging $cppStaging -Destination $cppDestination -Extensions @(".h") -VerifyOnly:$Check
Sync-Or-CheckDirectory -Staging $generatedPythonPackage -Destination $pythonDestination -Extensions @(".py", ".pyi") -VerifyOnly:$Check

$mode = if ($Check) { "verified" } else { "updated" }
Write-Output "FlatBuffers $expectedFlatcVersion bindings $mode successfully."

$schemaCodegenRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "build\schema-codegen"))
$resolvedStagingRoot = (Resolve-Path -LiteralPath $stagingRoot).Path
$expectedPrefix = $schemaCodegenRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $resolvedStagingRoot.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove unexpected staging path: $resolvedStagingRoot"
}
Remove-Item -LiteralPath $resolvedStagingRoot -Recurse -Force
