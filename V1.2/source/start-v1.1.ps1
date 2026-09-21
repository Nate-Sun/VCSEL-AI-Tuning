[CmdletBinding()]
param(
    [ValidateSet("FAKE", "READ_ONLY", "LIVE_TUNING")]
    [string]$Mode = "FAKE",

    [string]$ConfigPath,

    [ValidateRange(1, 65535)]
    [int]$Port = 8000,

    [switch]$NoBrowser,

    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$HostAddress = "127.0.0.1"
$ProjectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$SourceRoot = Join-Path $ProjectRoot "src"
$ApiEntrypoint = Join-Path $SourceRoot "vcsel_ai_tuning\api\app.py"
$FrontendDist = Join-Path $ProjectRoot "frontend\dist"
$FrontendIndex = Join-Path $FrontendDist "index.html"
$ServerProcess = $null
$LogPath = $null
$StartedAt = [DateTimeOffset]::UtcNow
$EnvironmentNames = @(
    "PYTHONPATH",
    "VCSEL_AI_FRONTEND_DIST",
    "VCSEL_AI_API_HOST",
    "VCSEL_AI_API_PORT",
    "VCSEL_AI_RUNTIME_CONFIG",
    "VCSEL_AI_CAMPAIGN_SERVICE_FACTORY"
)
$SavedEnvironment = @{}
foreach ($Name in $EnvironmentNames) {
    $SavedEnvironment[$Name] = [Environment]::GetEnvironmentVariable(
        $Name,
        [EnvironmentVariableTarget]::Process
    )
}

function Write-LauncherLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    if ($null -ne $script:LogPath) {
        Add-Content -LiteralPath $script:LogPath -Value $Message -Encoding UTF8
    }
}

function Get-LauncherPython {
    $Candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($env:VCSEL_AI_PYTHON)) {
        $Candidates.Add($env:VCSEL_AI_PYTHON)
    }
    $Candidates.Add((Join-Path $ProjectRoot ".venv-ai-tuning\Scripts\python.exe"))
    $Candidates.Add(
        [System.IO.Path]::GetFullPath(
            (Join-Path $ProjectRoot "..\..\.venv-ai-tuning\Scripts\python.exe")
        )
    )
    $SystemPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $SystemPython) {
        $Candidates.Add($SystemPython.Source)
    }

    foreach ($Candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($Candidate) -and
            (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }
    throw (
        "Python was not found. Create .venv-ai-tuning or set " +
        "VCSEL_AI_PYTHON to a Python 3.12+ executable."
    )
}

function Assert-WritableDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    $Probe = Join-Path $Path (".launcher-write-{0}.tmp" -f [Guid]::NewGuid())
    try {
        [System.IO.File]::WriteAllText($Probe, "write-check")
    }
    catch {
        throw "Directory is not writable: $Path"
    }
    finally {
        if (Test-Path -LiteralPath $Probe -PathType Leaf) {
            Remove-Item -LiteralPath $Probe -Force
        }
    }
}

function Assert-PortAvailable {
    param([Parameter(Mandatory = $true)][int]$CandidatePort)

    $Listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $CandidatePort
    )
    try {
        $Listener.Start()
    }
    catch {
        throw (
            "Port $CandidatePort is already in use on $HostAddress. " +
            "Choose another port with -Port."
        )
    }
    finally {
        $Listener.Stop()
    }
}

function Restore-LauncherEnvironment {
    foreach ($Name in $EnvironmentNames) {
        [Environment]::SetEnvironmentVariable(
            $Name,
            $SavedEnvironment[$Name],
            [EnvironmentVariableTarget]::Process
        )
    }
}

try {
    if (-not (Test-Path -LiteralPath $ApiEntrypoint -PathType Leaf)) {
        throw "FastAPI entrypoint does not exist: $ApiEntrypoint"
    }
    if (-not (Test-Path -LiteralPath $FrontendIndex -PathType Leaf)) {
        throw (
            "frontend/dist/index.html is missing. From '$ProjectRoot', run: " +
            "pnpm --dir frontend build. Dependencies must already be installed."
        )
    }
    if ($Mode -ne "FAKE" -and [string]::IsNullOrWhiteSpace($ConfigPath)) {
        throw "$Mode mode requires -ConfigPath with an explicitly reviewed runtime YAML."
    }

    $ResolvedConfigPath = $null
    if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
        $CandidateConfig = $ConfigPath
        if (-not [System.IO.Path]::IsPathRooted($CandidateConfig)) {
            $CandidateConfig = Join-Path (Get-Location).Path $CandidateConfig
        }
        if (-not (Test-Path -LiteralPath $CandidateConfig -PathType Leaf)) {
            throw "Runtime config does not exist: $CandidateConfig"
        }
        $ResolvedConfigPath = (Resolve-Path -LiteralPath $CandidateConfig).Path
    }

    $PythonPath = Get-LauncherPython
    $ExistingPythonPath = $SavedEnvironment["PYTHONPATH"]
    if ([string]::IsNullOrWhiteSpace($ExistingPythonPath)) {
        $env:PYTHONPATH = $SourceRoot
    }
    else {
        $env:PYTHONPATH = "$SourceRoot$([System.IO.Path]::PathSeparator)$ExistingPythonPath"
    }
    $env:VCSEL_AI_FRONTEND_DIST = $FrontendDist
    $env:VCSEL_AI_API_HOST = $HostAddress
    $env:VCSEL_AI_API_PORT = [string]$Port
    Remove-Item Env:\VCSEL_AI_CAMPAIGN_SERVICE_FACTORY -ErrorAction SilentlyContinue
    if ($null -ne $ResolvedConfigPath) {
        $env:VCSEL_AI_RUNTIME_CONFIG = $ResolvedConfigPath
    }
    else {
        Remove-Item Env:\VCSEL_AI_RUNTIME_CONFIG -ErrorAction SilentlyContinue
    }

    $DependencyOutput = & $PythonPath -c (
        "import sys; " +
        "assert sys.version_info >= (3, 12), 'Python 3.12+ is required'; " +
        "import fastapi, uvicorn; " +
        "from vcsel_ai_tuning.api.app import main"
    ) 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Python cannot import the V1.2 FastAPI runtime: " +
            ($DependencyOutput -join [Environment]::NewLine)
        )
    }

    if ($null -ne $ResolvedConfigPath) {
        $ConfigSummaryOutput = & $PythonPath -c (
            "import json, sys; " +
            "from vcsel_ai_tuning.runtime_config import load_runtime_config; " +
            "c = load_runtime_config(sys.argv[1]); " +
            "print(json.dumps({'mode': c.mode.value, " +
            "'sqlite_path': str(c.storage.sqlite_path), " +
            "'artifact_root': str(c.storage.artifact_root)}))"
        ) $ResolvedConfigPath 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw (
                "Runtime config validation failed: " +
                ($ConfigSummaryOutput -join [Environment]::NewLine)
            )
        }
        $ConfigSummary = (
            [string]($ConfigSummaryOutput | Select-Object -Last 1)
        ) | ConvertFrom-Json
        $ConfiguredMode = [string]$ConfigSummary.mode
        if ($ConfiguredMode -ne $Mode) {
            throw (
                "Requested mode $Mode does not match runtime config mode " +
                "$ConfiguredMode."
            )
        }
        $StateDirectory = Split-Path -Parent ([string]$ConfigSummary.sqlite_path)
        $ArtifactDirectory = [string]$ConfigSummary.artifact_root
    }
    else {
        $StateDirectory = Join-Path $ProjectRoot "state\ai_tuning"
        $ArtifactDirectory = Join-Path $ProjectRoot "artifacts\ai_tuning"
    }

    Assert-PortAvailable -CandidatePort $Port
    Assert-WritableDirectory -Path $StateDirectory
    Assert-WritableDirectory -Path $ArtifactDirectory
    Assert-WritableDirectory -Path (Join-Path $ProjectRoot "state\v1.2")
    Assert-WritableDirectory -Path (Join-Path $ProjectRoot "artifacts\v1.2")
    Assert-WritableDirectory -Path (Join-Path $ProjectRoot "logs")

    $LogPath = Join-Path $ProjectRoot (
        "logs\v1.2-launcher-{0}.log" -f $StartedAt.ToString("yyyyMMdd")
    )
    $VersionMatch = Select-String -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") `
        -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    $Version = if ($null -eq $VersionMatch) {
        "unknown"
    }
    else {
        $VersionMatch.Matches[0].Groups[1].Value
    }
    $ConfigForLog = if ($null -eq $ResolvedConfigPath) {
        "<none>"
    }
    else {
        $ResolvedConfigPath
    }
    Write-LauncherLog ("started_at_utc={0}" -f $StartedAt.ToString("o"))
    Write-LauncherLog "version=$Version"
    Write-LauncherLog "mode=$Mode"
    Write-LauncherLog "config=$ConfigForLog"
    Write-LauncherLog "host=$HostAddress"
    Write-LauncherLog "port=$Port"
    Write-LauncherLog "python=$PythonPath"
    Write-LauncherLog "frontend_dist=$FrontendDist"
    Write-LauncherLog "state_directory=$StateDirectory"
    Write-LauncherLog "artifact_directory=$ArtifactDirectory"

    if ($ValidateOnly) {
        Write-LauncherLog "validation=passed"
        Write-Host "Validation passed for VCSEL AI Tuning V1.2 ($Mode)."
        return
    }

    $ProcessInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $ProcessInfo.FileName = $PythonPath
    $ProcessInfo.Arguments = "-m vcsel_ai_tuning.api"
    $ProcessInfo.WorkingDirectory = $ProjectRoot
    $ProcessInfo.UseShellExecute = $false
    $ServerProcess = [System.Diagnostics.Process]::new()
    $ServerProcess.StartInfo = $ProcessInfo
    if (-not $ServerProcess.Start()) {
        throw "FastAPI process could not be started."
    }
    $BaseUrl = "http://${HostAddress}:$Port"
    $Deadline = [DateTimeOffset]::UtcNow.AddSeconds(30)
    $Health = $null
    while ([DateTimeOffset]::UtcNow -lt $Deadline) {
        if ($ServerProcess.HasExited) {
            throw (
                "FastAPI exited before /health became ready " +
                "(exit code $($ServerProcess.ExitCode))."
            )
        }
        try {
            $Health = Invoke-RestMethod `
                -Uri "$BaseUrl/health" `
                -Method Get `
                -TimeoutSec 2
            if ([string]$Health.status -eq "ok") {
                break
            }
        }
        catch {
            $Health = $null
        }
        Start-Sleep -Milliseconds 250
    }
    if ($null -eq $Health) {
        throw "FastAPI did not return a successful /health response within 30 seconds."
    }

    $Capabilities = Invoke-RestMethod `
        -Uri "$BaseUrl/capabilities" `
        -Method Get `
        -TimeoutSec 5
    $ReportedMode = [string]$Capabilities.runtime_mode
    if ($ReportedMode -ne $Mode) {
        throw "Runtime reported $ReportedMode, but launcher requested $Mode."
    }

    Write-LauncherLog ("ready_at_utc={0}" -f [DateTimeOffset]::UtcNow.ToString("o"))
    Write-Host "VCSEL AI Tuning V1.2 is ready."
    Write-Host "Runtime mode: $ReportedMode"
    Write-Host "URL: $BaseUrl"
    Write-Host "Press Ctrl+C to stop."
    if (-not $NoBrowser) {
        $BrowserInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $BrowserInfo.FileName = $BaseUrl
        $BrowserInfo.UseShellExecute = $true
        [void][System.Diagnostics.Process]::Start($BrowserInfo)
    }

    while (-not $ServerProcess.HasExited) {
        Start-Sleep -Milliseconds 500
    }
    if ($ServerProcess.ExitCode -ne 0) {
        throw "FastAPI exited with code $($ServerProcess.ExitCode)."
    }
}
catch {
    Write-LauncherLog "error=$($_.Exception.Message)"
    throw
}
finally {
    if ($null -ne $ServerProcess -and -not $ServerProcess.HasExited) {
        [void]$ServerProcess.CloseMainWindow()
        if (-not $ServerProcess.WaitForExit(3000)) {
            Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
            [void]$ServerProcess.WaitForExit(3000)
        }
    }
    Write-LauncherLog (
        "stopped_at_utc={0}" -f [DateTimeOffset]::UtcNow.ToString("o")
    )
    Restore-LauncherEnvironment
}
