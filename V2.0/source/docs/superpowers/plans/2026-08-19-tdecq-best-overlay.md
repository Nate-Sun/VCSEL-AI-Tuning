# TDECQ Best-So-Far Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-install Windows overlay that observes a real V2.1 AI tuning campaign and displays the cumulative minimum TDECQ without changing hardware behavior.

**Architecture:** A standalone PowerShell script reads an explicitly selected `state.json` and its immutable sibling `campaign.json`, computes a best-so-far step series, and renders it in a native Windows Forms chart. The existing V2.1 controller remains the only component that selects parameters or writes hardware; the overlay has no AI tuning package imports and no write path into campaign artifacts.

**Tech Stack:** Windows PowerShell 5.1, .NET Windows Forms, `System.Windows.Forms.DataVisualization`, Python 3.12 `unittest` for subprocess-driven verification.

---

## File structure

- Create `scripts/tdecq_best_overlay.ps1`: command-line entry point, state parser, best-series calculation, file picker, polling timer, and native chart window.
- Create `scripts/start_tdecq_best_overlay.cmd`: double-click launcher that applies a process-local PowerShell execution-policy bypass.
- Create `tests_py/test_tdecq_best_overlay.py`: black-box tests that invoke the PowerShell script in non-UI JSON mode.
- Create `docs/tdecq-best-overlay-recording.md`: short Chinese operating guide for selecting the correct campaign and arranging the recording windows.
- Do not modify `src/vcsel_ai_tuning/**`, the V2.1 worktrees, controller policies, hardware adapters, or campaign artifacts.

### Task 1: Parse campaign state and calculate Best TDECQ

**Files:**
- Create: `scripts/tdecq_best_overlay.ps1`
- Create: `tests_py/test_tdecq_best_overlay.py`

- [ ] **Step 1: Write the failing black-box calculation test**

Create `tests_py/test_tdecq_best_overlay.py` with the loader and first behavior test:

```python
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tdecq_best_overlay.ps1"
POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")


def record(round_index, before, after, status="VALID"):
    return {
        "round_index": round_index,
        "measurement_status": status,
        "before_metrics": {"tdecq_db": before},
        "after_metrics": {"tdecq_db": after},
    }


def write_campaign(root, records, *, campaign_id="campaign-1", max_rounds=50):
    (root / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "max_rounds": max_rounds}),
        encoding="utf-8",
    )
    (root / "state.json").write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "execution_mode": "LIVE",
                "state": "RUNNING",
                "target_tdecq_db": 2.8,
                "completed_rounds": len(records),
                "round_records": records,
            }
        ),
        encoding="utf-8",
    )


def snapshot(root):
    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-StatePath",
            str(root / "state.json"),
            "-SnapshotJson",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return json.loads(completed.stdout)


class BestTdecqCalculationTests(unittest.TestCase):
    def test_best_series_uses_baseline_and_ignores_regressions_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_campaign(
                root,
                [
                    record(1, 5.21, 4.23),
                    record(2, 4.23, 4.29),
                    record(3, 4.29, 3.76),
                    record(4, 3.76, None, "INVALID"),
                ],
            )

            result = snapshot(root)

        self.assertEqual(result["campaign_id"], "campaign-1")
        self.assertEqual(result["execution_mode"], "LIVE")
        self.assertEqual(result["max_rounds"], 50)
        self.assertEqual(result["baseline_tdecq_db"], 5.21)
        self.assertEqual(result["best_tdecq_db"], 3.76)
        self.assertEqual(result["best_round"], 3)
        self.assertEqual(
            [point["best_tdecq_db"] for point in result["points"]],
            [5.21, 4.23, 4.23, 3.76, 3.76],
        )
        self.assertEqual(
            [point["changed"] for point in result["points"]],
            [True, True, False, True, False],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing script fails**

Run:

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay.BestTdecqCalculationTests.test_best_series_uses_baseline_and_ignores_regressions_and_invalid_values -v
```

Expected: `ERROR` or `FAIL` because `scripts/tdecq_best_overlay.ps1` does not exist.

- [ ] **Step 3: Implement the non-UI parser and calculation path**

Create `scripts/tdecq_best_overlay.ps1` with this initial implementation:

```powershell
[CmdletBinding()]
param(
    [string]$StatePath,
    [switch]$SnapshotJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-FiniteDouble {
    param([object]$Value)
    if ($null -eq $Value -or $Value -is [bool]) { return $null }
    try {
        $number = [Convert]::ToDouble(
            $Value,
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
    catch { return $null }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { return $null }
    return $number
}

function Read-JsonObject {
    param([Parameter(Mandatory)][string]$Path)
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-OverlaySnapshot {
    param(
        [Parameter(Mandatory)][object]$Campaign,
        [Parameter(Mandatory)][object]$State
    )
    $campaignId = [string]$Campaign.campaign_id
    if ([string]::IsNullOrWhiteSpace($campaignId)) { throw 'campaign.json has no campaign_id' }
    if ([string]$State.campaign_id -ne $campaignId) {
        throw 'campaign.json and state.json campaign_id mismatch'
    }
    $maxRounds = [int]$Campaign.max_rounds
    if ($maxRounds -le 0) { throw 'campaign.json max_rounds must be positive' }

    $records = @($State.round_records)
    $points = [System.Collections.Generic.List[object]]::new()
    $baseline = $null
    $best = $null
    $bestRound = $null

    if ($records.Count -gt 0) {
        $baseline = ConvertTo-FiniteDouble $records[0].before_metrics.tdecq_db
        if ($null -ne $baseline) {
            $best = $baseline
            $bestRound = 0
            $points.Add([ordered]@{
                round_index = 0
                best_tdecq_db = $best
                changed = $true
            })
        }
    }

    foreach ($record in $records) {
        $roundIndex = [int]$record.round_index
        $candidate = $null
        if ([string]$record.measurement_status -eq 'VALID') {
            $candidate = ConvertTo-FiniteDouble $record.after_metrics.tdecq_db
        }
        $changed = $false
        if ($null -ne $candidate -and ($null -eq $best -or $candidate -lt $best)) {
            $best = $candidate
            $bestRound = $roundIndex
            $changed = $true
        }
        if ($null -ne $best) {
            $points.Add([ordered]@{
                round_index = $roundIndex
                best_tdecq_db = $best
                changed = $changed
            })
        }
    }

    return [ordered]@{
        campaign_id = $campaignId
        execution_mode = [string]$State.execution_mode
        state = [string]$State.state
        target_tdecq_db = ConvertTo-FiniteDouble $State.target_tdecq_db
        max_rounds = $maxRounds
        completed_rounds = [int]$State.completed_rounds
        baseline_tdecq_db = $baseline
        best_tdecq_db = $best
        best_round = $bestRound
        waiting = ($records.Count -eq 0)
        points = @($points)
    }
}

if ($SnapshotJson) {
    if ([string]::IsNullOrWhiteSpace($StatePath)) { throw '-StatePath is required with -SnapshotJson' }
    $resolvedState = (Resolve-Path -LiteralPath $StatePath).Path
    $campaignPath = Join-Path (Split-Path -Parent $resolvedState) 'campaign.json'
    $snapshot = Get-OverlaySnapshot `
        -Campaign (Read-JsonObject $campaignPath) `
        -State (Read-JsonObject $resolvedState)
    $snapshot | ConvertTo-Json -Depth 10 -Compress
    exit 0
}
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run the same unittest command from Step 2.

Expected: one test reports `ok` and the process exits with code 0.

- [ ] **Step 5: Commit the calculation slice**

```powershell
git add -- scripts/tdecq_best_overlay.ps1 tests_py/test_tdecq_best_overlay.py
git commit -m "feat: calculate best TDECQ from campaign state"
```

### Task 2: Reject identity mismatches and preserve stable chart bounds

**Files:**
- Modify: `scripts/tdecq_best_overlay.ps1`
- Modify: `tests_py/test_tdecq_best_overlay.py`

- [ ] **Step 1: Add failing validation and axis tests**

Append these tests to `BestTdecqCalculationTests`:

```python
    def test_campaign_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_campaign(root, [record(1, 5.21, 4.23)])
            state_path = root / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["campaign_id"] = "different-campaign"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            completed = subprocess.run(
                [
                    str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(SCRIPT), "-StatePath", str(state_path), "-SnapshotJson",
                ],
                check=False, capture_output=True, text=True, encoding="utf-8",
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("campaign_id mismatch", completed.stderr)

    def test_axis_hints_are_fixed_from_baseline_target_and_max_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_campaign(root, [record(1, 5.21, 4.23)], max_rounds=50)
            result = snapshot(root)
        self.assertEqual(result["axis_x_min"], 0)
        self.assertEqual(result["axis_x_max"], 50)
        self.assertAlmostEqual(result["axis_y_max"], 5.5)
        self.assertAlmostEqual(result["axis_y_min"], 2.6)

    def test_empty_running_campaign_waits_without_inventing_a_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_campaign(root, [])
            result = snapshot(root)
        self.assertTrue(result["waiting"])
        self.assertIsNone(result["baseline_tdecq_db"])
        self.assertIsNone(result["best_tdecq_db"])
        self.assertEqual(result["points"], [])
```

- [ ] **Step 2: Run the two new tests and verify they fail**

Run:

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay.BestTdecqCalculationTests.test_campaign_identity_mismatch_is_rejected tests_py.test_tdecq_best_overlay.BestTdecqCalculationTests.test_axis_hints_are_fixed_from_baseline_target_and_max_rounds tests_py.test_tdecq_best_overlay.BestTdecqCalculationTests.test_empty_running_campaign_waits_without_inventing_a_baseline -v
```

Expected: mismatch assertion or missing axis keys causes failure.

- [ ] **Step 3: Add deterministic axis hints to `Get-OverlaySnapshot`**

Before the returned ordered dictionary, add:

```powershell
    $target = ConvertTo-FiniteDouble $State.target_tdecq_db
    $axisYMax = $null
    $axisYMin = $null
    if ($null -ne $baseline) {
        $axisYMax = [math]::Ceiling(($baseline + 0.2) * 10.0) / 10.0
        $floorSource = if ($null -ne $target) { [math]::Min($target, $baseline - 0.5) } else { $baseline - 0.5 }
        $axisYMin = [math]::Floor(($floorSource - 0.2) * 10.0) / 10.0
    }
```

Add these keys to the returned dictionary and reuse `$target`:

```powershell
        target_tdecq_db = $target
        axis_x_min = 0
        axis_x_max = $maxRounds
        axis_y_min = $axisYMin
        axis_y_max = $axisYMax
```

- [ ] **Step 4: Run the full calculation test class**

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay.BestTdecqCalculationTests -v
```

Expected: all calculation tests report `ok`.

- [ ] **Step 5: Commit validation and stable axes**

```powershell
git add -- scripts/tdecq_best_overlay.ps1 tests_py/test_tdecq_best_overlay.py
git commit -m "feat: validate overlay campaign identity and axes"
```

### Task 3: Render the native right-side chart and poll live state

**Files:**
- Modify: `scripts/tdecq_best_overlay.ps1`
- Modify: `tests_py/test_tdecq_best_overlay.py`

- [ ] **Step 1: Add a failing source-safety and non-UI contract test**

Append a new test class:

```python
class OverlaySafetyTests(unittest.TestCase):
    def test_overlay_has_no_hardware_or_ai_tuning_imports(self):
        source = SCRIPT.read_text(encoding="utf-8").lower()
        forbidden = [
            "vcsel_ai_tuning",
            "setfir",
            "set_fir",
            "write_field",
            "pyvisa",
            "serialport",
        ]
        self.assertEqual([term for term in forbidden if term in source], [])

    def test_source_defines_native_step_chart_and_quarter_second_polling(self):
        source = SCRIPT.read_text(encoding="utf-8")
        required = [
            "System.Windows.Forms.DataVisualization",
            "SeriesChartType]::StepLine",
            "$timer.Interval = 250",
            "function Select-StatePath",
            "ShowDialog()",
        ]
        self.assertEqual([term for term in required if term not in source], [])

    def test_snapshot_mode_does_not_modify_campaign_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_campaign(root, [record(1, 5.21, 4.23)])
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            snapshot(root)
            after = {path.name: path.read_bytes() for path in root.iterdir()}
        self.assertEqual(after, before)
```

- [ ] **Step 2: Run the safety tests and verify the source test fails before UI code exists**

Run:

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay.OverlaySafetyTests -v
```

Expected: `test_source_defines_native_step_chart_and_quarter_second_polling` fails because the native UI and polling loop do not exist yet; the read-only tests remain green.

- [ ] **Step 3: Add file selection, Windows Forms chart construction, and view update functions**

After `Get-OverlaySnapshot`, add functions with these exact responsibilities:

```powershell
function Select-StatePath {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = [System.Windows.Forms.OpenFileDialog]::new()
    $dialog.Title = '选择本次 AI tuning 的 state.json'
    $dialog.Filter = 'Campaign state (state.json)|state.json|JSON files (*.json)|*.json'
    $dialog.CheckFileExists = $true
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return $null }
    return $dialog.FileName
}

function New-OverlayView {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    Add-Type -AssemblyName System.Windows.Forms.DataVisualization

    $form = [System.Windows.Forms.Form]::new()
    $form.Text = 'Best TDECQ / 历史最低'
    $form.BackColor = [System.Drawing.Color]::FromArgb(5, 12, 11)
    $form.ForeColor = [System.Drawing.Color]::FromArgb(209, 250, 229)
    $form.Width = 640
    $form.Height = 1000
    $form.MinimumSize = [System.Drawing.Size]::new(480, 720)
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
    $working = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    $form.Location = [System.Drawing.Point]::new($working.Right - $form.Width, $working.Top)

    $header = [System.Windows.Forms.Panel]::new()
    $header.Dock = 'Top'
    $header.Height = 180
    $header.Padding = [System.Windows.Forms.Padding]::new(22, 18, 22, 8)

    $title = [System.Windows.Forms.Label]::new()
    $title.Text = 'BEST TDECQ / 历史最低'
    $title.ForeColor = [System.Drawing.Color]::FromArgb(155, 193, 181)
    $title.Font = [System.Drawing.Font]::new('Consolas', 13, [System.Drawing.FontStyle]::Bold)
    $title.AutoSize = $true
    $title.Location = [System.Drawing.Point]::new(22, 18)

    $bestLabel = [System.Windows.Forms.Label]::new()
    $bestLabel.Text = '-- dB'
    $bestLabel.ForeColor = [System.Drawing.Color]::FromArgb(110, 231, 183)
    $bestLabel.Font = [System.Drawing.Font]::new('Consolas', 38, [System.Drawing.FontStyle]::Bold)
    $bestLabel.AutoSize = $true
    $bestLabel.Location = [System.Drawing.Point]::new(18, 52)

    $detailLabel = [System.Windows.Forms.Label]::new()
    $detailLabel.Text = 'Waiting for Round 1'
    $detailLabel.ForeColor = [System.Drawing.Color]::FromArgb(126, 153, 144)
    $detailLabel.Font = [System.Drawing.Font]::new('Consolas', 11)
    $detailLabel.AutoSize = $true
    $detailLabel.Location = [System.Drawing.Point]::new(23, 120)

    $statusLabel = [System.Windows.Forms.Label]::new()
    $statusLabel.Text = 'Waiting for campaign state'
    $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(134, 239, 172)
    $statusLabel.Font = [System.Drawing.Font]::new('Consolas', 9)
    $statusLabel.AutoEllipsis = $true
    $statusLabel.Dock = 'Bottom'
    $statusLabel.Height = 26

    $header.Controls.AddRange(@($title, $bestLabel, $detailLabel))

    $chart = [System.Windows.Forms.DataVisualization.Charting.Chart]::new()
    $chart.Dock = 'Fill'
    $chart.BackColor = $form.BackColor
    $area = [System.Windows.Forms.DataVisualization.Charting.ChartArea]::new('BestArea')
    $area.BackColor = [System.Drawing.Color]::FromArgb(7, 16, 15)
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(32, 51, 45)
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(32, 51, 45)
    $area.AxisX.LabelStyle.ForeColor = [System.Drawing.Color]::FromArgb(97, 120, 111)
    $area.AxisY.LabelStyle.ForeColor = [System.Drawing.Color]::FromArgb(97, 120, 111)
    $area.AxisX.Title = 'ROUND'
    $area.AxisY.Title = 'TDECQ (dB)'
    $area.AxisX.TitleForeColor = $area.AxisX.LabelStyle.ForeColor
    $area.AxisY.TitleForeColor = $area.AxisY.LabelStyle.ForeColor
    $chart.ChartAreas.Add($area)

    $series = [System.Windows.Forms.DataVisualization.Charting.Series]::new('BestSoFar')
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::StepLine
    $series.BorderWidth = 4
    $series.Color = [System.Drawing.Color]::FromArgb(52, 211, 153)
    $series.MarkerStyle = [System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Circle
    $series.MarkerSize = 6
    $chart.Series.Add($series)

    $flashTimer = [System.Windows.Forms.Timer]::new()
    $flashTimer.Interval = 350
    $flashTimer.Add_Tick({
        $bestLabel.ForeColor = [System.Drawing.Color]::FromArgb(110, 231, 183)
        $flashTimer.Stop()
    })

    $form.Controls.Add($chart)
    $form.Controls.Add($header)
    $form.Controls.Add($statusLabel)
    return [pscustomobject]@{
        Form = $form; Chart = $chart; Area = $area; Series = $series
        BestLabel = $bestLabel; DetailLabel = $detailLabel; StatusLabel = $statusLabel
        FlashTimer = $flashTimer; AxisYMin = $null; AxisYMax = $null
        LastBest = $null; LastUpdateAt = Get-Date
    }
}

function Update-OverlayView {
    param([Parameter(Mandatory)][object]$View, [Parameter(Mandatory)][object]$Snapshot)
    $View.Series.Points.Clear()
    foreach ($point in @($Snapshot.points)) {
        [void]$View.Series.Points.AddXY([int]$point.round_index, [double]$point.best_tdecq_db)
    }
    $View.Area.AxisX.Minimum = [double]$Snapshot.axis_x_min
    $View.Area.AxisX.Maximum = [double]$Snapshot.axis_x_max
    if ($null -ne $Snapshot.axis_y_min -and $null -eq $View.AxisYMin) {
        $View.AxisYMin = [double]$Snapshot.axis_y_min
        $View.AxisYMax = [double]$Snapshot.axis_y_max
    }
    if ($null -ne $Snapshot.best_tdecq_db -and $null -ne $View.AxisYMin -and [double]$Snapshot.best_tdecq_db -lt $View.AxisYMin) {
        $View.AxisYMin = [math]::Floor(([double]$Snapshot.best_tdecq_db - 0.2) * 10.0) / 10.0
    }
    if ($null -ne $View.AxisYMin) {
        $View.Area.AxisY.Minimum = $View.AxisYMin
        $View.Area.AxisY.Maximum = $View.AxisYMax
    }
    if ($null -eq $Snapshot.best_tdecq_db) {
        $View.BestLabel.Text = '-- dB'
        $View.DetailLabel.Text = 'Waiting for Round 1'
    }
    else {
        $View.BestLabel.Text = ('{0:F2} dB' -f [double]$Snapshot.best_tdecq_db)
        $where = if ([int]$Snapshot.best_round -eq 0) { 'Baseline' } else { 'Round ' + [int]$Snapshot.best_round }
        $View.DetailLabel.Text = 'Best at ' + $where + '  |  Completed ' + [int]$Snapshot.completed_rounds + '/' + [int]$Snapshot.max_rounds
        if ($null -ne $View.LastBest -and [double]$Snapshot.best_tdecq_db -lt [double]$View.LastBest) {
            $View.BestLabel.ForeColor = [System.Drawing.Color]::FromArgb(209, 250, 229)
            $View.FlashTimer.Start()
        }
        $View.LastBest = [double]$Snapshot.best_tdecq_db
    }
    $View.LastUpdateAt = Get-Date
    $View.StatusLabel.Text = [string]$Snapshot.campaign_id + '  |  ' + [string]$Snapshot.execution_mode + '  |  ' + [string]$Snapshot.state
}
```

- [ ] **Step 4: Add the 250 ms polling loop and preserve the last valid chart**

Append the interactive entry path after the `-SnapshotJson` block:

```powershell
if ([string]::IsNullOrWhiteSpace($StatePath)) { $StatePath = Select-StatePath }
if ([string]::IsNullOrWhiteSpace($StatePath)) { exit 0 }
$resolvedState = (Resolve-Path -LiteralPath $StatePath).Path
$campaignPath = Join-Path (Split-Path -Parent $resolvedState) 'campaign.json'
$campaign = Read-JsonObject $campaignPath
$expectedCampaignId = [string]$campaign.campaign_id
$view = New-OverlayView
$script:lastCompleted = -1
$script:lastState = ''

$timer = [System.Windows.Forms.Timer]::new()
$timer.Interval = 250
$timer.Add_Tick({
    try {
        $state = Read-JsonObject $resolvedState
        if ([string]$state.campaign_id -ne $expectedCampaignId) {
            throw 'Campaign ID changed; restart the overlay with the intended state.json'
        }
        $snapshot = Get-OverlaySnapshot -Campaign $campaign -State $state
        if ([int]$snapshot.completed_rounds -ne $script:lastCompleted -or [string]$snapshot.state -ne $script:lastState) {
            Update-OverlayView -View $view -Snapshot $snapshot
            $script:lastCompleted = [int]$snapshot.completed_rounds
            $script:lastState = [string]$snapshot.state
        }
        elseif (((Get-Date) - $view.LastUpdateAt).TotalSeconds -ge 30) {
            $idleSeconds = [int](((Get-Date) - $view.LastUpdateAt).TotalSeconds)
            $view.StatusLabel.Text = [string]$snapshot.campaign_id + '  |  ' + [string]$snapshot.state + '  |  No new round for ' + $idleSeconds + ' s'
        }
    }
    catch {
        $view.StatusLabel.Text = 'Last valid chart retained  |  ' + $_.Exception.Message
        $view.StatusLabel.ForeColor = [System.Drawing.Color]::FromArgb(251, 191, 36)
    }
})
$view.Form.Add_Shown({ $timer.Start() })
$view.Form.Add_FormClosed({
    $timer.Stop(); $timer.Dispose()
    $view.FlashTimer.Stop(); $view.FlashTimer.Dispose()
})
[void]$view.Form.ShowDialog()
```

- [ ] **Step 5: Run calculation and safety tests after adding the UI**

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay -v
```

Expected: all tests report `ok`; no UI opens because tests use `-SnapshotJson`.

- [ ] **Step 6: Commit the live chart slice**

```powershell
git add -- scripts/tdecq_best_overlay.ps1 tests_py/test_tdecq_best_overlay.py
git commit -m "feat: show live best TDECQ recording overlay"
```

### Task 4: Verify the known 50-round campaign and terminal behavior

**Files:**
- Modify: `tests_py/test_tdecq_best_overlay.py`
- Modify: `scripts/tdecq_best_overlay.ps1`

- [ ] **Step 1: Add the historical 50-round regression test**

Append to `BestTdecqCalculationTests`:

```python
    def test_known_campaign_keeps_round_46_as_best_after_round_50(self):
        values = [
            4.23, 4.29, 3.76, None, 3.77, 4.01, 3.81, 4.38, 3.60, 4.12,
            3.54, 3.74, 3.49, 4.17, 3.66, 3.48, 3.61, 3.99, 3.64, 4.15,
            3.62, 3.43, 3.81, 3.40, 4.08, 3.43, 3.23, 4.58, 3.29, 3.37,
            3.24, 3.23, 4.10, 3.31, 4.45, 3.63, 3.31, 3.38, 3.57, 3.21,
            3.24, 5.29, 3.22, 3.64, 3.36, 3.16, 4.01, 3.17, 3.65, 3.25,
        ]
        records = []
        before = 5.21
        for index, value in enumerate(values, start=1):
            status = "INVALID" if value is None else "VALID"
            records.append(record(index, before, value, status))
            before = value
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_campaign(root, records, campaign_id="v21-b-20260817T125713321Z-b468337d")
            result = snapshot(root)
        self.assertEqual(result["completed_rounds"], 50)
        self.assertEqual(result["baseline_tdecq_db"], 5.21)
        self.assertEqual(result["best_tdecq_db"], 3.16)
        self.assertEqual(result["best_round"], 46)
        self.assertEqual(result["points"][-1]["best_tdecq_db"], 3.16)
```

- [ ] **Step 2: Run the historical test and verify it passes or exposes parser gaps**

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay.BestTdecqCalculationTests.test_known_campaign_keeps_round_46_as_best_after_round_50 -v
```

Expected: `ok`, with 3.16 dB attributed to Round 46.

- [ ] **Step 3: Add explicit terminal-state display tests**

Add a helper parameter `state="RUNNING"` to `write_campaign`, use it in the state payload, then add:

```python
    def test_terminal_state_is_returned_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_campaign(root, [record(1, 5.21, 4.23)], state="RECOVERY_REQUIRED")
            result = snapshot(root)
        self.assertEqual(result["state"], "RECOVERY_REQUIRED")
        self.assertEqual(result["best_tdecq_db"], 4.23)
```

- [ ] **Step 4: Run the full unittest module**

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay -v
```

Expected: every test reports `ok` with zero failures and zero errors.

- [ ] **Step 5: Commit the real-campaign regression coverage**

```powershell
git add -- tests_py/test_tdecq_best_overlay.py scripts/tdecq_best_overlay.ps1
git commit -m "test: verify 50-round best TDECQ overlay"
```

### Task 5: Add the double-click launcher and recording guide

**Files:**
- Create: `scripts/start_tdecq_best_overlay.cmd`
- Create: `docs/tdecq-best-overlay-recording.md`

- [ ] **Step 1: Create the double-click launcher**

Create `scripts/start_tdecq_best_overlay.cmd`:

```bat
@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tdecq_best_overlay.ps1" %*
if errorlevel 1 pause
endlocal
```

- [ ] **Step 2: Write the concise Chinese recording guide**

Create `docs/tdecq-best-overlay-recording.md` with:

```markdown
# Best TDECQ 录制窗口使用说明

1. 正常启动计划录制的真实 V2.1 AI tuning Campaign。
2. 等待 Campaign 目录中出现 `campaign.json` 和 `state.json`。
3. 双击 `scripts/start_tdecq_best_overlay.cmd`。
4. 在文件窗口中选择本次 Campaign 的 `state.json`，不要选择旧 Campaign。
5. 核对图表底部的 `Campaign ID`、`LIVE` 和运行状态。
6. 将 AI tuning/DCA 窗口放在左侧约 66%，将 Best TDECQ 窗口放在右侧约 34%。
7. 开始录屏。图表会在每轮结果落盘后自动更新。
8. 运行结束后，图表保持最终历史最低值；录制完成后直接关闭窗口。

右侧显示的是 `Best TDECQ / 历史最低`，不是当前轮测量值。脚本只读取实验文件，不连接或控制硬件。
```

- [ ] **Step 3: Verify launcher paths and documentation references**

Run:

```powershell
Test-Path -LiteralPath 'scripts\tdecq_best_overlay.ps1'
Test-Path -LiteralPath 'scripts\start_tdecq_best_overlay.cmd'
rg -n "state.json|Campaign ID|历史最低|只读取" 'docs\tdecq-best-overlay-recording.md'
```

Expected: both path checks print `True`; the guide search returns the selection, identity, label, and read-only statements.

- [ ] **Step 4: Commit launcher and guide**

```powershell
git add -- scripts/start_tdecq_best_overlay.cmd docs/tdecq-best-overlay-recording.md
git commit -m "docs: add Best TDECQ recording workflow"
```

### Task 6: Perform end-to-end verification without hardware

**Files:**
- Verify: `scripts/tdecq_best_overlay.ps1`
- Verify: `scripts/start_tdecq_best_overlay.cmd`
- Verify: `tests_py/test_tdecq_best_overlay.py`
- Verify: `docs/tdecq-best-overlay-recording.md`

- [ ] **Step 1: Run the complete overlay test module**

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests_py.test_tdecq_best_overlay -v
```

Expected: all overlay tests pass with zero failures and zero errors.

- [ ] **Step 2: Verify the historical campaign in non-UI read-only mode**

Run against a copied fixture or the existing persisted campaign path:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\tdecq_best_overlay.ps1 -StatePath 'C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-optimization\artifacts\v2.1\continuous\controller\v21-b-20260817T125713321Z-b468337d\state.json' -SnapshotJson
```

Expected JSON includes:

```json
{"campaign_id":"v21-b-20260817T125713321Z-b468337d","execution_mode":"LIVE","max_rounds":50,"completed_rounds":50,"baseline_tdecq_db":5.21,"best_tdecq_db":3.16,"best_round":46}
```

The actual JSON may contain additional keys and points; the listed values must match exactly.

- [ ] **Step 3: Run a UI smoke test using temporary copied artifacts**

Copy `campaign.json` and `state.json` from the historical campaign into a temporary directory, launch the overlay against the temporary `state.json`, and verify visually:

- the window opens on the right side;
- the chart background is dark and legible;
- the x-axis ends at Round 50;
- the large value is 3.16 dB;
- the detail says Round 46;
- the footer shows the exact Campaign ID, `LIVE`, and `COMPLETED`;
- replacing the temporary `state.json` briefly with invalid JSON retains the last valid chart and shows a warning, and restoring valid JSON resumes updates;
- closing the overlay does not affect the AI tuning application or campaign files.

- [ ] **Step 4: Run repository hygiene checks**

```powershell
git diff --check
git status --short
rg -n -i "vcsel_ai_tuning|setfir|set_fir|write_field|pyvisa|serialport" scripts\tdecq_best_overlay.ps1
```

Expected: `git diff --check` produces no errors; the forbidden-import search produces no matches; `git status --short` lists only the intended overlay changes, if any remain uncommitted.

- [ ] **Step 5: Commit any verification-only corrections**

If verification required a correction, stage only the overlay files and commit:

```powershell
git add -- scripts/tdecq_best_overlay.ps1 scripts/start_tdecq_best_overlay.cmd tests_py/test_tdecq_best_overlay.py docs/tdecq-best-overlay-recording.md
git commit -m "fix: finalize read-only Best TDECQ overlay"
```

If no correction was needed, do not create an empty commit.
