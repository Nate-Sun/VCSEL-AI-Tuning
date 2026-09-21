#requires -Version 7.0
param([switch]$LoadHelpersOnly)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function New-ContinuousCampaignId {
    $CampaignId = 'v21-b-{0}-{1}' -f `
        ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')), `
        ([guid]::NewGuid().ToString('N').Substring(0, 8))
    if ($CampaignId -cnotmatch '^v21-b-[0-9]{8}T[0-9]{9}Z-[0-9a-f]{8}$') {
        throw '内部 Campaign ID 生成失败 / Internal Campaign ID generation failed'
    }
    return $CampaignId
}

function ConvertTo-PositiveDecimalInteger {
    param([AllowEmptyString()][string]$Text)

    if ($null -eq $Text -or $Text -cnotmatch '^[0-9]+$') {
        throw '最大轮数必须是只含十进制数字的正整数 / Max rounds must be a positive decimal integer containing digits only'
    }
    $ParsedValue = [System.Numerics.BigInteger]::Zero
    $Parsed = [System.Numerics.BigInteger]::TryParse(
        $Text,
        [System.Globalization.NumberStyles]::None,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$ParsedValue
    )
    if (-not $Parsed -or $ParsedValue -le [System.Numerics.BigInteger]::Zero) {
        throw '最大轮数必须大于零 / Max rounds must be greater than zero'
    }
    return $Text
}

function ConvertTo-PositiveInvariantDouble {
    param([AllowEmptyString()][string]$Text)

    $ParsedValue = 0.0
    $Parsed = [double]::TryParse(
        $Text,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$ParsedValue
    )
    if (-not $Parsed -or $ParsedValue -le 0 -or
        [double]::IsNaN($ParsedValue) -or [double]::IsInfinity($ParsedValue)) {
        throw '目标 TDECQ 必须是使用小数点的有限正数 / Target TDECQ must be a positive finite number using a decimal point'
    }
    return $ParsedValue.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture)
}

function ConvertFrom-ContinuousJsonResult {
    param(
        [Parameter(Mandatory = $true)][object[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $NonblankLines = @($Lines | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    if ($NonblankLines.Count -ne 1) {
        throw "$Label 必须产生且只能产生一行非空 JSON 结果 / must produce exactly one nonblank JSON result line"
    }
    try {
        $Result = $NonblankLines[0] | ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "$Label 返回了无效 JSON / returned invalid JSON: $($_.Exception.Message)"
    }
    if ($null -eq $Result -or $Result -isnot [pscustomobject]) {
        throw "$Label JSON 结果必须是对象 / JSON result must be an object"
    }
    return $Result
}

function Assert-ContinuousPreflightReport {
    param([Parameter(Mandatory = $true)][pscustomobject]$Report)

    if ($Report.operation -cne 'PREFLIGHT' -or
        $Report.dut_write_performed -ne $false -or
        $Report.hardware_write_performed -ne $false -or
        $Report.hardware_write_scope -cne 'NO_DUT_REGISTER_OR_BERT_WRITE' -or
        $Report.bert_contacted -ne $false -or
        $Report.dca_acquisition_commands_performed -ne $true) {
        throw 'PRE-FLIGHT 报告未证明 DCA 已采集且 DUT/BERT 保持只读；不得进入 COMMISSION / PRE-FLIGHT report did not prove DCA acquisition with read-only DUT/BERT scope; do not continue to COMMISSION'
    }
}

function Get-ContinuousConfigManifest {
    param([Parameter(Mandatory = $true)][string]$LiveConfigPath)

    $ManifestLines = @(& py -3.13 -P -c @'
from hashlib import sha256
from pathlib import Path
from sys import argv
import yaml

repo_root = Path(argv[2]).resolve(strict=True)
live_path = Path(argv[1]).resolve(strict=True)
live_document = yaml.safe_load(live_path.read_text(encoding='utf-8'))
runtime_path = (repo_root / live_document['live']['runtime_config']).resolve(strict=True)
runtime_document = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))
hardware_path = (runtime_path.parent / runtime_document['hardware']['driver_profile']).resolve(strict=True)
b0_path = (runtime_path.parent / runtime_document['b0_path']).resolve(strict=True)
paths = (live_path, runtime_path, hardware_path, b0_path)
if any(not path.is_file() for path in paths):
    raise ValueError('会话 manifest 要求所有配置路径都是普通文件 / Session manifest requires regular configuration files')
digest = sha256()
for path in paths:
    digest.update(str(path).encode('utf-8'))
    digest.update(b'\0')
    digest.update(path.read_bytes())
    digest.update(b'\0')
print(digest.hexdigest())
'@ $LiveConfigPath (Get-Location).Path)
    if ($LASTEXITCODE -ne 0) {
        throw '无法计算会话配置 manifest / Could not calculate the session configuration manifest'
    }
    if ($ManifestLines.Count -ne 1 -or [string]::IsNullOrWhiteSpace([string]$ManifestLines[0])) {
        throw '会话配置 manifest 必须且只能产生一行非空输出 / Session configuration manifest must yield exactly one nonblank line'
    }
    $Manifest = $ManifestLines[0].Trim()
    if ($Manifest -cnotmatch '^[0-9a-f]{64}$') {
        throw '会话配置 manifest 不是 SHA-256 十六进制摘要 / Session configuration manifest is not a SHA-256 hex digest'
    }
    return $Manifest
}

function Assert-ContinuousWorktreeStatus {
    param([Parameter(Mandatory = $true)][string]$Context)

    $TrackedStatusLines = @(& git status --porcelain --untracked-files=no)
    if ($LASTEXITCODE -ne 0) {
        throw "$Context 期间无法确定 tracked worktree 状态 / Could not determine tracked worktree status during $Context"
    }
    if ($TrackedStatusLines.Count -ne 0) {
        throw "$Context 期间 tracked worktree 已改变 / Tracked worktree changed during $Context"
    }

    $UntrackedPaths = @(& git -c core.quotepath=false ls-files --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) {
        throw "$Context 期间无法确定 untracked worktree 状态 / Could not determine untracked worktree status during $Context"
    }
    foreach ($UntrackedPath in $UntrackedPaths) {
        $NormalizedPath = ([string]$UntrackedPath).Replace('\', '/')
        if ($NormalizedPath -cnotmatch '^(artifacts|outputs|state)/') {
            throw "$Context 期间发现未批准的 untracked 路径 $NormalizedPath / Unauthorized untracked path during ${Context}: $NormalizedPath"
        }
    }

    $IgnoredPaths = @(& git -c core.quotepath=false ls-files --others --ignored --exclude-standard)
    if ($LASTEXITCODE -ne 0) {
        throw "$Context 期间无法确定 ignored worktree 状态 / Could not determine ignored worktree status during $Context"
    }
    foreach ($IgnoredPath in $IgnoredPaths) {
        $NormalizedPath = ([string]$IgnoredPath).Replace('\', '/')
        $IsEvidence = $NormalizedPath -cmatch '^(artifacts|outputs|state)/'
        $IsSensitive = $NormalizedPath -match '\.(py|pyw|pth|ps1|psm1|psd1|yaml|yml|toml|json|ini|cfg|exe|dll|bat|cmd)$'
        if (-not $IsEvidence -and $IsSensitive) {
            throw "$Context 期间发现敏感 ignored 路径 $NormalizedPath / Sensitive ignored path during ${Context}: $NormalizedPath"
        }
    }
}

function Assert-ContinuousSessionScope {
    $CurrentBranchLines = @(& git branch --show-current)
    if ($LASTEXITCODE -ne 0 -or $CurrentBranchLines.Count -ne 1 -or
        [string]::IsNullOrWhiteSpace([string]$CurrentBranchLines[0])) {
        throw '范围验证期间无法确定当前 Git branch / Could not determine the current Git branch during scope verification'
    }
    if ($CurrentBranchLines[0].Trim() -cne $ExpectedBranch) {
        throw '会话范围冻结后 branch 已改变 / Branch changed after session scope was frozen'
    }

    $CurrentHeadLines = @(& git rev-parse HEAD)
    if ($LASTEXITCODE -ne 0 -or $CurrentHeadLines.Count -ne 1 -or
        [string]::IsNullOrWhiteSpace([string]$CurrentHeadLines[0])) {
        throw '范围验证期间无法确定 Git HEAD / Could not determine Git HEAD during scope verification'
    }
    if ($CurrentHeadLines[0].Trim() -cne $SessionHead) {
        throw '会话范围冻结后 Git HEAD 已改变 / Git HEAD changed after session scope was frozen'
    }

    Assert-ContinuousWorktreeStatus -Context '会话范围验证 / session scope verification'

    if (-not (Test-Path -LiteralPath $SessionLiveConfigPath -PathType Leaf)) {
        throw '冻结的本地会话配置已不存在 / Frozen local session config no longer exists'
    }
    $CurrentLiveConfigPath = (Resolve-Path -LiteralPath $SessionLiveConfigPath).Path
    if ($CurrentLiveConfigPath -cne $SessionLiveConfigPath) {
        throw '冻结的本地会话配置路径已改变 / Frozen local session config path changed'
    }
    $CurrentConfigManifest = Get-ContinuousConfigManifest -LiveConfigPath $SessionLiveConfigPath
    if ($CurrentConfigManifest -cne $SessionConfigManifest) {
        throw '范围冻结后会话配置 manifest 已改变 / Session configuration manifest changed after scope was frozen'
    }
}

function Invoke-ContinuousSessionCleanup {
    param(
        [AllowNull()][string]$SessionLiveConfigPath,
        [AllowNull()][System.Management.Automation.ErrorRecord]$PrimaryError
    )

    $TempCleanupError = $null
    $ApiKeyCleanupError = $null
    try {
        try {
            if (-not [string]::IsNullOrWhiteSpace($SessionLiveConfigPath) -and
                (Test-Path -LiteralPath $SessionLiveConfigPath)) {
                Remove-Item -LiteralPath $SessionLiveConfigPath -Force -ErrorAction Stop
            }
        } catch {
            $TempCleanupError = $_
        }
    } finally {
        try {
            Remove-Item Env:MOONSHOT_API_KEY -ErrorAction Stop
        } catch [System.Management.Automation.ItemNotFoundException] {
            # The key was never installed or is already absent.
        } catch {
            $ApiKeyCleanupError = $_
        }
    }

    if ($null -ne $PrimaryError) {
        if ($null -ne $TempCleanupError) {
            Write-Warning "临时 YAML 清理失败；保留原始主异常 / Temporary YAML cleanup failed; preserving the primary error: $($TempCleanupError.Exception.Message)"
        }
        if ($null -ne $ApiKeyCleanupError) {
            Write-Warning "API key 环境变量清理失败；保留原始主异常 / API-key cleanup failed; preserving the primary error: $($ApiKeyCleanupError.Exception.Message)"
        }
        return
    }
    if ($null -ne $TempCleanupError) {
        throw $TempCleanupError
    }
    if ($null -ne $ApiKeyCleanupError) {
        throw $ApiKeyCleanupError
    }
}

if ($LoadHelpersOnly) {
    return
}

Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = (Resolve-Path 'src').Path

$ExpectedBranch = 'codex/v2.1-continuous-eye-b0-70-22-fast'
$RequiredImplementationBaseline = 'ff5187be'
$CampaignPlaceholder = 'REPLACE_WITH_FRESH_CAMPAIGN_ID'
$TemplateConfigPath = Join-Path $PWD 'configs\ai_tuning\v2_1_continuous.live.yaml'
$LiveConfigPath = $null
$PrimaryError = $null
try {
    $BranchLines = @(& git branch --show-current)
    if ($LASTEXITCODE -ne 0 -or $BranchLines.Count -ne 1 -or
        [string]::IsNullOrWhiteSpace([string]$BranchLines[0])) {
        throw '无法确定当前 Git branch / Could not determine the current Git branch'
    }
    if ($BranchLines[0].Trim() -cne $ExpectedBranch) {
        throw "Git branch 错误：应为 $ExpectedBranch，实际为 $($BranchLines[0]) / Wrong Git branch: expected $ExpectedBranch, got $($BranchLines[0])"
    }

    $HeadLines = @(& git rev-parse HEAD)
    if ($LASTEXITCODE -ne 0 -or $HeadLines.Count -ne 1 -or
        [string]::IsNullOrWhiteSpace([string]$HeadLines[0])) {
        throw '无法确定 Git HEAD / Could not determine Git HEAD'
    }
    $Head = $HeadLines[0].Trim()
    & git merge-base --is-ancestor $RequiredImplementationBaseline $Head
    if ($LASTEXITCODE -ne 0) {
        throw "HEAD $Head 不包含所需实现基线 $RequiredImplementationBaseline / HEAD $Head does not contain required implementation baseline $RequiredImplementationBaseline"
    }

    Assert-ContinuousWorktreeStatus -Context '初始 PRE-FLIGHT 范围验证 / initial PRE-FLIGHT scope verification'

    if (-not (Test-Path -LiteralPath $TemplateConfigPath -PathType Leaf)) {
        throw '找不到签入的 LIVE 配置模板 / Checked-in LIVE config template was not found'
    }
    $TemplateText = Get-Content -LiteralPath $TemplateConfigPath -Raw -Encoding utf8
    $PlaceholderCount = ([regex]::Matches(
        $TemplateText,
        [regex]::Escape($CampaignPlaceholder)
    )).Count
    if ($PlaceholderCount -ne 1) {
        throw '签入模板必须且只能包含一个 campaign placeholder / Checked-in template must contain exactly one campaign placeholder'
    }

    $MaxRoundsText = Read-Host '请输入本次最大优化轮数 / Max optimization rounds'
    $ValidatedMaxRoundsText = ConvertTo-PositiveDecimalInteger -Text $MaxRoundsText
    $TargetTdecqText = Read-Host '请输入目标 TDECQ dB（严格小于） / Target TDECQ dB (strictly below)'
    $ValidatedTargetTdecq = ConvertTo-PositiveInvariantDouble -Text $TargetTdecqText
    Set-Variable -Name SessionMaxRounds -Value $ValidatedMaxRoundsText -Option ReadOnly
    Set-Variable -Name SessionTargetTdecqDb -Value $ValidatedTargetTdecq -Option ReadOnly

    $GeneratedCampaignId = New-ContinuousCampaignId
    Set-Variable -Name SessionCampaignId -Value $GeneratedCampaignId -Option ReadOnly
    $LiveConfigPath = Join-Path ([IO.Path]::GetTempPath()) `
        ("v2_1_continuous.live.$([guid]::NewGuid().ToString('N')).yaml")
    $SessionConfigText = $TemplateText.Replace($CampaignPlaceholder, $SessionCampaignId)
    Set-Content -LiteralPath $LiveConfigPath -Value $SessionConfigText -NoNewline -Encoding utf8

    $ResolvedLiveConfigPath = (Resolve-Path -LiteralPath $LiveConfigPath).Path
    Set-Variable -Name SessionLiveConfigPath -Value $ResolvedLiveConfigPath -Option ReadOnly
    Set-Variable -Name SessionHead -Value $Head -Option ReadOnly
    $InitialConfigManifest = Get-ContinuousConfigManifest -LiveConfigPath $SessionLiveConfigPath
    Set-Variable -Name SessionConfigManifest -Value $InitialConfigManifest -Option ReadOnly

    Write-Host "已自动生成 Campaign ID: $SessionCampaignId / Automatically generated Campaign ID: $SessionCampaignId"
    Write-Host "已验证 HEAD: $SessionHead / Verified HEAD: $SessionHead"

    $secureMoonshotKey = Read-Host '请输入 MOONSHOT_API_KEY / Enter MOONSHOT_API_KEY' -AsSecureString
    $MoonshotKey = [Net.NetworkCredential]::new('', $secureMoonshotKey).Password
    if ([string]::IsNullOrWhiteSpace($MoonshotKey)) {
        throw 'MOONSHOT_API_KEY 不能为空 / MOONSHOT_API_KEY must be nonblank'
    }
    $env:MOONSHOT_API_KEY = $MoonshotKey
    Remove-Variable secureMoonshotKey, MoonshotKey

    Assert-ContinuousSessionScope
    $PreflightResultLines = @(& py -3.13 -P scripts/run_continuous_eye.py `
        --config $SessionLiveConfigPath `
        --allow-device-contact `
        --max-rounds $SessionMaxRounds `
        --target-tdecq-db $SessionTargetTdecqDb `
        --live-operation PREFLIGHT)
    $PreflightExitCode = $LASTEXITCODE
    $PreflightResultLines | ForEach-Object { Write-Host $_ }
    if ($PreflightExitCode -ne 0) {
        throw "PRE-FLIGHT 失败（退出码 $PreflightExitCode）；不得运行 COMMISSION / PRE-FLIGHT failed (exit $PreflightExitCode); do not run COMMISSION"
    }
    $PreflightReport = ConvertFrom-ContinuousJsonResult -Lines $PreflightResultLines -Label 'PRE-FLIGHT'
    Assert-ContinuousPreflightReport -Report $PreflightReport

    [void](Read-Host 'PRE-FLIGHT 已通过；取得 COMMISSION 授权后按 Enter / PRE-FLIGHT passed; press Enter after COMMISSION approval')
    Assert-ContinuousSessionScope

    $CommissionResultLines = @(& py -3.13 -P scripts/run_continuous_eye.py `
        --config $SessionLiveConfigPath `
        --allow-device-contact `
        --max-rounds $SessionMaxRounds `
        --target-tdecq-db $SessionTargetTdecqDb `
        --live-operation COMMISSION)
    $CommissionExitCode = $LASTEXITCODE
    $CommissionResultLines | ForEach-Object { Write-Host $_ }
    if ($CommissionExitCode -ne 0) {
        throw "COMMISSION 失败（退出码 $CommissionExitCode）；保留上方输出并检查持久化恢复证据 / COMMISSION failed (exit $CommissionExitCode); preserve the output above and inspect persisted recovery evidence"
    }
    $CommissionResult = ConvertFrom-ContinuousJsonResult -Lines $CommissionResultLines -Label 'COMMISSION'
    if ($CommissionResult.state -cne 'COMPLETED') {
        throw "COMMISSION 未成功完成（终态 $($CommissionResult.state)）；检查持久化报告与恢复证据 / COMMISSION did not complete successfully (terminal state $($CommissionResult.state)); inspect persisted report and recovery evidence"
    }
} catch {
    $PrimaryError = $_
} finally {
    Invoke-ContinuousSessionCleanup `
        -SessionLiveConfigPath $LiveConfigPath `
        -PrimaryError $PrimaryError
}
if ($null -ne $PrimaryError) {
    throw $PrimaryError
}
