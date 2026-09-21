# V2.1 Continuous Eye Optimization B Windows Desktop Design

**Status:** Approved in conversation on 2026-08-18

**Desktop branch:** `codex/v2.1-continuous-eye-desktop-app`

**Validated engine tag:** `v2.1-continuous-eye-b-validated-20260817`

**Validated engine commit:** `4085b007546f820820dbdb7f7f0bba6ebc4b2049`

## 1. Purpose

Build the smallest safe Windows desktop application around the validated V2.1
Continuous Eye Optimization B engine. An operator supplies a Moonshot API key,
the nine B0 I2C values, a maximum round count, and a strict TDECQ target. The
application runs the existing PRE-FLIGHT and COMMISSION workflow, restores B0
on every supported exit path, presents the complete result after the run, and
exports one CSV row per measured point.

The desktop application is an operator and evidence layer. It is not a second
optimizer or hardware controller.

## 2. Scope Decisions

### Included

- Tkinter/ttk Windows GUI.
- PyInstaller `onedir` distribution behind a normal Windows installer and
  desktop shortcut.
- One settings page and one result page.
- Compact PRE-FLIGHT approval dialog instead of a separate page.
- Minimal run status containing only current round, maximum rounds, and a safe
  stop button.
- Nine editable B0 I2C integer fields, prefilled from the validated B0 on every
  new campaign.
- Editable DCA VISA resource, I2C COM port, and I2C device address.
- Local persistence of only those three non-secret bench endpoints.
- CSV export generated from immutable campaign artifacts.
- Offline/fake automated tests and packaged smoke tests.
- A minimal build-lineage and distribution-integrity gate before hardware is
  enabled.

### Excluded

- A live dashboard, live eye display, or live per-round result table.
- A standalone PRE-FLIGHT page.
- Native `.xlsx` export in the first release.
- BERT, FIR, NLC, pattern, power-supply, temperature-control, or optical-path
  controls.
- Changes to parameter ranges, `max_step` semantics, the objective, provider
  validation, controller behavior, DCA/I2C implementations, or B0 restore.
- Automatic PRE-FLIGHT-to-COMMISSION transition.
- Windows account/login features, licensing, auto-update, or certificate-based
  Authenticode signing in the first release.
- Real hardware use during development or automated testing.

## 3. Fixed Engine and Safety Baseline

Desktop development starts from the validated tag and commit above in an
independent worktree. The existing validated B worktree is never modified.

The following implementation remains the single source of optimization and
hardware behavior:

- `src/vcsel_ai_tuning/continuous_live.py`
- `src/vcsel_ai_tuning/v2/providers/kimi_client.py`
- `src/vcsel_ai_tuning/v2/providers/kimi_provider.py`
- `src/vcsel_ai_tuning/v2/continuous/controller.py`
- `src/vcsel_ai_tuning/v2/continuous/models.py`
- `src/vcsel_ai_tuning/v2/continuous/policy.py`
- `src/vcsel_ai_tuning/v2/continuous/store.py`
- the existing DCA, I2C, persistence, and recovery modules reached by those
  entry points

The signed PowerShell launcher remains unchanged. Because a packaged
application has no meaningful Git branch at runtime, the desktop entry point
uses the separate minimal integrity gate in section 8 rather than deleting or
weakening the launcher's branch check.

## 4. Operator Inputs

The settings page contains only the inputs required for a campaign.

### Required campaign inputs

- `MOONSHOT_API_KEY`, masked, nonblank, and never persisted.
- `max_rounds`, a positive decimal integer.
- `target_tdecq_db`, a positive finite number interpreted as strict
  `TDECQ < target`.
- Nine B0 integers:
  - `ibias_dac`
  - `vref`
  - `vbias_ref`
  - `vga_cur`
  - `res_gain`
  - `res_bw`
  - `gain_ctrl`
  - `efin_cur`
  - `eq`

Every campaign starts with the validated B0 values prefilled:

```yaml
ibias_dac: 60
vref: 15
vbias_ref: 15
vga_cur: 13
res_gain: 6
res_bw: 0
gain_ctrl: 2
efin_cur: 15
eq: 16
```

The operator may edit all nine values. The application validates integers,
completeness, encoding envelopes, and compatibility with the existing live
policy before PRE-FLIGHT.

### Fixed B0 fields

These are visible in a compact read-only summary and are never editable:

```yaml
pattern: SSPRQ
fir:
  taps: [0, 9, -28, 104, -8, 1, -3]
  upper_nlc: 6
  lower_nlc: 6
```

### Bench settings

An expandable bench section exposes only:

- DCA VISA resource
- I2C COM port
- I2C device address

The application stores these three non-secret values under the current user's
local application-data directory and preloads them on the next launch. Changes
must pass a new PRE-FLIGHT. Baud rate, DCA sources, expected instrument
identity, I2C channel, expected chip ID/revision, register mapping, and all
safety limits remain fixed and validated.

## 5. Minimal Operator Workflow

### 5.1 Settings page

The operator reviews the displayed tag/commit, enters the API key, edits B0 if
needed, supplies maximum rounds and target, and optionally edits the three bench
endpoints. `Run PRE-FLIGHT` remains disabled until local validation succeeds.

### 5.2 PRE-FLIGHT and approval dialog

PRE-FLIGHT calls the existing read-only live workflow. It may read I2C, acquire
DCA evidence, and contact Kimi. It must not write DUT parameters or contact
BERT.

There is no PRE-FLIGHT page. On success, a compact modal shows:

- DCA identity/configuration verified
- I2C identity and B0 readback verified
- Kimi connectivity and proposal validation verified
- `DUT write = false`
- `BERT contacted = false`

The dialog offers `View evidence` and `Approve and start optimization`. The
approve action is explicit and cannot be triggered automatically. A failed
check shows the reason and offers no COMMISSION action.

### 5.3 Minimal running state

After approval, the window shows only:

- `Optimizing: round N / maximum`
- `Safe stop`

It does not display live eyes, proposals, readbacks, Kimi logs, or measurements.
Safe stop calls the existing controller's `request_stop()` and never terminates
the worker process or thread forcibly.

### 5.4 Result page

The result page becomes available after completion, safe failure, operator
stop, or recovery-required termination. It shows:

- B0, best, and last measured-point comparison
- target and TDECQ improvement
- best parameters and best round
- every measured point with proposal, full readback, TDECQ, ER, and
  `VALID`/`INVALID`
- eye evidence for each measured point
- final successful Kimi attempt metadata associated with each selected point
- a read-only view of the complete original Kimi attempt log
- B0 restore state and evidence completeness
- `Open evidence folder`
- `Export CSV`

## 6. Desktop Architecture

### 6.1 GUI module

Tkinter owns the Windows main thread and renders the settings page, modal
approval, minimal running state, and result page. It never imports or directly
calls hardware adapters.

### 6.2 Desktop Session module

`DesktopSession` is the only interface between GUI code and the validated
engine. It is a deep module with a small operator-oriented interface:

- validate settings and create immutable session inputs
- run PRE-FLIGHT
- accept explicit COMMISSION approval
- request safe stop
- publish immutable status/result events to the GUI
- export CSV from persisted evidence

The engine runs in a single background worker so Tkinter remains responsive.
The GUI receives status through a thread-safe event queue. The session module
sets the API key only for the active engine session and removes it in `finally`
on every normal or exceptional exit.

### 6.3 Validated engine

The Desktop Session calls the existing `preflight_continuous_live`, continuous
activation/controller, `request_stop`, artifact stores, and recovery paths. It
does not reproduce proposal, policy, write, measurement, acceptance, or restore
logic.

## 7. Evidence and CSV

The existing JSON and JSONL files remain authoritative and are never rewritten
by the GUI. Desktop views and CSV export are projections of those files.

The installed application stores campaigns by default under:

```text
%USERPROFILE%\Documents\VCSEL Continuous Eye\campaigns\<campaign_id>\
```

Each CSV row represents one actual measured point. Row zero represents the B0
measurement. Optimization rows follow in round order. Required columns are:

- campaign/build identity: campaign ID, validated tag, validated commit,
  execution mode
- point identity: row type, round index, proposal attempt, timestamps
- nine requested fields, blank when unchanged/not requested
- nine complete verified readback fields
- TDECQ, outer ER, measurement status, and invalid reason when present
- best-so-far round and TDECQ
- evidence completeness and eye path
- final successful Kimi attempt finish reason, token counts, and parse status

Failed Kimi attempts remain in the original `kimi-response-attempts.jsonl` and
the result viewer. They are not flattened into extra measured-point rows.
Export is available for completed and partial campaigns whenever persisted
measured points exist.

## 8. Minimal Integrity Gate

The gate protects against accidentally building or distributing the wrong
same-named engine or safety configuration. It is not a login, license, or
adversarial tamper-proof signature system.

At build time:

1. Require the desktop worktree to descend from validated commit
   `4085b007546f820820dbdb7f7f0bba6ebc4b2049`.
2. Verify protected engine, launcher, and live configuration files against the
   validated tag unless an explicitly reviewed desktop seam requires a change.
3. Record the exact tag, base commit, desktop commit, protected-file hashes,
   build environment, and package contents in a build manifest.

At runtime:

1. Load and validate the packaged build manifest.
2. Verify the distributed protected configuration and package fingerprints.
3. Display the validated tag/commit.
4. Keep all real-hardware actions disabled if verification fails. Historical
   viewing and CSV export remain available.

The first release does not require a Windows certificate. This gate therefore
targets accidental lineage/configuration drift, not a malicious administrator
who can replace both the application and its manifest.

## 9. Failure and Close Behavior

- Closing during PRE-FLIGHT cancels the session, closes resources, and clears
  the API key. No B0 restore is required because PRE-FLIGHT performed no DUT
  write.
- Closing during COMMISSION is treated as safe stop. The window remains in a
  restoring state until the engine reaches a terminal result.
- Normal exceptions, provider failure, invalid measurements, and safe stop use
  the existing failure and B0 restore paths.
- Restore failure displays `RECOVERY_REQUIRED`, preserves all evidence, and
  blocks a new campaign.
- A power loss or forced OS process kill cannot guarantee immediate restore.
  On the next launch, unfinished durable state blocks new work and requires the
  existing recovery workflow before another COMMISSION.
- The GUI never deletes artifacts, outputs, state, or original provider logs.

## 10. Packaging and Target Computers

The build pipeline first produces and verifies a PyInstaller `onedir` package.
The release pipeline then wraps that exact directory in a normal Windows
installer that creates a desktop shortcut. The package bundles the application
Python runtime and Python dependencies, so the target PC does not need the
developer's Python or PowerShell 7 path.

Target PCs still require the correct VISA backend and the serial driver for the
I2C adapter. The application detects missing drivers during PRE-FLIGHT and does
not attempt COMMISSION.

## 11. Test Strategy

Implementation is test-driven and hardware-free by default.

- Unit tests cover input validation, default B0 behavior, settings persistence,
  secret scrubbing, CSV projection, integrity-gate decisions, and UI state
  transitions.
- Fake integration tests cover PRE-FLIGHT success/failure, the explicit human
  approval gate, commission completion, safe stop, provider failure, B0 restore,
  recovery-required results, and partial CSV export.
- GUI tests use a fake Desktop Session and prove that no GUI module imports or
  constructs hardware adapters.
- Packaged smoke tests launch the `.exe` in offline/fake mode and prove that
  real hardware remains unavailable.
- No automated or development test contacts real DCA, I2C, DUT, BERT, or Kimi.
- Any real hardware validation is described separately and requires explicit
  operator approval before execution.

## 12. Acceptance Criteria

The design is complete when an offline-tested Windows build can demonstrate:

1. Exact validated engine provenance is displayed and hardware is fail-closed
   on an integrity mismatch.
2. The operator can enter an API key, edit nine B0 integers, maximum rounds, and
   a strict target.
3. Only the three approved bench endpoints persist locally; the API key never
   appears in settings, logs, artifacts, or CSV.
4. PRE-FLIGHT cannot write DUT parameters or contact BERT.
5. COMMISSION cannot begin without the explicit approval dialog action.
6. Running UI exposes only round progress and safe stop.
7. Stop and supported failures use the existing B0 restore path.
8. Results display every measured point and its evidence after the run.
9. CSV export includes B0 plus every persisted measured point and opens in
   Excel.
10. Existing V2.1 continuous tests remain green, and all new desktop tests run
    offline.

There are no unresolved product choices in this design.
