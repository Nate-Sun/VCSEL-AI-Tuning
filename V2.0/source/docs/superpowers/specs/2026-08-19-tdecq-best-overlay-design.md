# TDECQ Best-So-Far Recording Overlay Design

## Goal

Provide a simple right-side recording window for a real VCSEL V2.1 continuous AI tuning run. The existing AI tuning application and DCA eye display remain on the left. A standalone, read-only script displays the cumulative minimum TDECQ on the right and updates after each persisted round.

The overlay must never contact DCA, BERT, I2C, or any hardware-control API. It must not import VCSEL hardware modules or execute `setFir` or register writes.

## Confirmed data contract

The user selects the exact campaign `state.json` to monitor. This prevents the overlay from silently attaching to a different branch, worktree, or campaign. The script also reads the immutable sibling `campaign.json` to obtain `max_rounds` and verifies that both files contain the same `campaign_id`. The chart therefore uses a stable round axis from the start of recording.

For each update, the script reads:

- `campaign_id`
- `execution_mode`
- `state`
- `target_tdecq_db`
- `completed_rounds`
- `round_records[].round_index`
- `round_records[].measurement_status`
- `round_records[].before_metrics.tdecq_db`
- `round_records[].after_metrics.tdecq_db`

From the sibling `campaign.json`, the script reads:

- `campaign_id`
- `max_rounds`

The continuous controller saves `state.json` after appending each completed round. The store writes a temporary file, flushes it, and atomically replaces `state.json`. The overlay can therefore poll the file without coordinating with the tuning process and should normally see either the previous complete state or the next complete state.

The previously verified example campaign contains one baseline and 50 completed rounds. It is evidence for the schema and display behavior, not a hard-coded data source for future recordings.

## Recommended implementation

Use one Windows PowerShell script, `scripts/tdecq_best_overlay.ps1`, with a native Windows Forms chart. Keep it independent of the AI tuning application and avoid any Python, browser-server, or third-party runtime requirement for recording.

The script has five small responsibilities:

1. Resolve the exact `state.json` path supplied with `-StatePath`. If omitted, open a native file-selection dialog. Never auto-select the newest campaign.
2. Parse and validate `state.json` and its sibling `campaign.json` without importing the AI tuning package.
3. Open a dark, recording-ready Windows Forms chart window with no browser toolbar or local network service.
4. Poll the selected state file every 250 ms and redraw only when `completed_rounds` or terminal state changes.
5. Preserve the last valid chart if a transient read fails during file replacement, and keep the final chart visible until the user closes the window.

The PowerShell process does not modify `state.json`, the campaign directory, or any laboratory configuration.

## Best TDECQ calculation

The display series begins with the first round's valid `before_metrics.tdecq_db`, which represents the campaign baseline. Until the first round is persisted, the overlay displays `Waiting for Round 1`.

For each round in chronological order:

1. Accept the value only when `measurement_status == "VALID"` and `after_metrics.tdecq_db` is a finite number.
2. Ignore invalid, missing, NaN, and infinite values.
3. Calculate `best[round] = min(best[round - 1], current_valid_tdecq)`.
4. Extend the horizontal step line for every completed round.
5. Drop the line only when a new minimum is observed.

The large numeric callout shows the current best, not the most recent measurement. Labels must say `BEST TDECQ / 历史最低` and `Best so far` so viewers are not led to believe every measurement improved.

The page also shows the round that produced the best value. For example, if Round 50 measures 3.25 dB but Round 46 remains best at 3.16 dB, the overlay displays `3.16 dB — Round 46`.

## Visual behavior

The overlay is designed for the right 34% of a 1920 x 1080 recording:

- Dark, nearly black background.
- Green step line with a bright marker on the current best.
- Large best-value number at the top.
- Stable labels for start, current completed round, target, and best round.
- A brief glow animation only when a new minimum arrives.
- No raw per-round curve, parameter table, decorative particles, or unrelated controls.

The horizontal axis is fixed from Round 0 through `max_rounds`. The vertical axis is initialized from the baseline and stored campaign target so it does not rescale on every round. If a valid result exceeds the initial bounds, the axis expands once to keep the data visible.

## Recording workflow

1. Start the intended real AI tuning campaign normally.
2. After its campaign folder and `state.json` exist, start the PowerShell overlay script and choose that exact file.
3. Confirm the displayed `Campaign ID` and `Execution mode` match the run being recorded.
4. Place the AI tuning/DCA window on the left and the native overlay window on the right.
5. Start screen recording.
6. The chart advances from persisted results; no manual chart interaction is needed.
7. When the campaign reaches a terminal state, the overlay remains on the final best value until the user closes it.

## Error handling

- Missing file: show `Waiting for state.json` and keep retrying.
- Atomic replacement moment or temporary read failure: retain the last valid display and retry; do not clear the chart.
- Invalid JSON or schema mismatch: show a concise warning without crashing the tuning process.
- Campaign ID changes at the same path: freeze the chart and require explicit user confirmation or restart.
- File becomes stale: show the time since the last completed-round update.
- Terminal states such as `COMPLETED`, `FAILED`, or `RECOVERY_REQUIRED`: keep the final chart visible and show the exact status unchanged.

## Verification

Unit tests will cover:

- baseline extraction;
- cumulative minimum calculation;
- invalid and missing measurement filtering;
- best round attribution;
- empty and partially populated state files;
- campaign identity-change protection;
- mismatch protection between `campaign.json` and `state.json`;
- retry behavior across atomic file replacement;
- the known 50-round fixture, which must yield baseline 5.21 dB and best 3.16 dB at Round 46.

An integration test will drive the PowerShell script's non-UI snapshot mode against temporary `campaign.json` and `state.json` fixtures. A Windows UI smoke test will verify that the chart updates when a fixture grows from zero to multiple rounds and that the terminal state remains visible.

## Non-goals

- No hardware access or control.
- No changes to AI selection or stopping policy.
- No automatic discovery of the latest campaign.
- No video encoding or MP4 export.
- No modification of persisted experiment evidence.
