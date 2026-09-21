[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Python
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "PowerShell 7 or newer is required."
}

$ValidatedCommit = "4085b007546f820820dbdb7f7f0bba6ebc4b2049"
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$PythonPath = (Resolve-Path -LiteralPath $Python).Path
$StageRoot = Join-Path $RepositoryRoot "build\desktop-integrity-stage"
$SpecPath = Join-Path $RepositoryRoot "packaging\v2_1_desktop.spec"
$ExePath = Join-Path $RepositoryRoot "dist\VCSELContinuousEye\VCSELContinuousEye.exe"

Push-Location $RepositoryRoot
try {
    $WorktreeState = & git status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect git worktree state."
    }
    if ($WorktreeState) {
        throw "Desktop build requires a clean worktree."
    }

    & git merge-base --is-ancestor $ValidatedCommit HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Validated V2.1 B commit is not an ancestor of HEAD."
    }

    $ValidatedScopes = @(
        "configs/ai_tuning/b0.v2_1.continuous.yaml",
        "configs/ai_tuning/runtime.v2_1.continuous.live.yaml",
        "configs/ai_tuning/v2_1_continuous.live.yaml",
        "configs/hardware/vcsel_lab_v2_1_continuous.yaml",
        "src/vcsel_ai_tuning/continuous_live.py",
        "src/vcsel_ai_tuning/runtime.py",
        "src/vcsel_ai_tuning/runtime_config.py",
        "src/vcsel_ai_tuning/campaign",
        "src/vcsel_ai_tuning/hardware",
        "src/vcsel_ai_tuning/v2"
    )
    & git diff --quiet $ValidatedCommit -- $ValidatedScopes
    if ($LASTEXITCODE -ne 0) {
        throw "Protected validated engine/config paths differ from V2.1 B."
    }

    $DesktopCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $DesktopCommit) {
        throw "Unable to resolve the desktop commit."
    }

    & $PythonPath -c "import sys; assert sys.version_info >= (3, 12); import tkinter; import pydantic, yaml, PIL, openai, serial, pyvisa, PyInstaller"
    if ($LASTEXITCODE -ne 0) {
        throw "The selected Python is missing desktop build dependencies."
    }

    $env:PYTHONPATH = Join-Path $RepositoryRoot "src"
    & $PythonPath -m vcsel_ai_tuning.desktop.build_support `
        --repository-root $RepositoryRoot `
        --stage-root $StageRoot `
        --desktop-commit $DesktopCommit
    if ($LASTEXITCODE -ne 0) {
        throw "Integrity staging failed."
    }

    $env:VCSEL_DESKTOP_STAGE = $StageRoot
    & $PythonPath -m PyInstaller --noconfirm --clean $SpecPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
    if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) {
        throw "Packaged executable was not produced: $ExePath"
    }

    & $ExePath --offline-smoke-test
    if ($LASTEXITCODE -ne 0) {
        throw "Packaged executable failed its offline smoke test."
    }

    $ExeHash = (Get-FileHash -LiteralPath $ExePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $ManifestPath = Join-Path $RepositoryRoot "dist\VCSELContinuousEye\_internal\build-manifest.json"
    $ManifestHash = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output "EXE=$ExePath"
    Write-Output "EXE_SHA256=$ExeHash"
    Write-Output "MANIFEST_SHA256=$ManifestHash"
}
finally {
    Pop-Location
}
