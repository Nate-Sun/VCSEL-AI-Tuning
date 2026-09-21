# FlexDCA Single TDECQ Acquisition Design

## Context

The live FIR workflow can bind the existing TDECQ measurement to Function 1 and
read its value, status, details, and equalizer taps. It does not initiate a new
acquisition. The operator currently performs Stop, Clear, and Single manually.

During commissioning, Function 1 initially reported `INV` after Single and later
reported `CORR` with the same displayed TDECQ value. Therefore acquisition-stop
and TDECQ-computation completion are not safe to treat as the same event. A live
workflow must wait for a fresh, source-bound `CORR` result before using TDECQ.

## Goal

Add an explicitly authorized `SINGLE` acquisition workflow that:

1. shows and hashes the exact DCA mutations before execution;
2. reproduces Stop, Clear, and Single without changing BERT FIR;
3. waits for a fresh Function 1 TDECQ result to become `CORR`;
4. binds the accepted result to unchanged BERT FIR state; and
5. saves a complete machine-readable receipt for every attempted acquisition.

## Scope

This increment adds `measure-plan` and `measure` commands to
`scripts/run_live_fir.py`. It supports one acquisition mode, `single`.

The following are out of scope:

- continuous `:ACQuire:RUN` control;
- acquisition-limit tests;
- Autoscale or changes to eye, trigger, pattern, TDECQ, or equalizer settings;
- BERT FIR writes;
- automatic FIR proposal, acceptance, or rollback decisions based on the new
  measurement.

Continuous Run acquisition is the next separate increment after this Single
workflow is commissioned.

## Considered Approaches

### A. Direct Stop, Clear, Single with conditional TDECQ polling

This is the selected approach. It matches the operator's known-good manual flow,
changes the fewest DCA settings, and directly addresses transient `INV` results.

### B. Acquisition-limit test with Run and `*OPC?`

This offers an explicit acquisition count, but it modifies limit-test state and
uses Run rather than reproducing the current manual Single procedure. It belongs
in the later continuous-Run increment.

### C. Passive polling after a manual Single

This avoids DCA mutations but does not automate acquisition and cannot establish
the Clear boundary itself.

## Hardware Commands and Queries

The execution target is the configured DCA resource and Function 1 source. The
plan contains these mutations in order:

```text
:MEASure:EYE:TDEQ:SOURce FUNC1
:ACQuire:STOP
:ACQuire:CDISplay
:ACQuire:SINGle
```

On timeout, the workflow may send one additional safety command:

```text
:ACQuire:STOP
```

The workflow performs these queries as needed:

```text
*IDN?
:MEASure:EYE:TDEQ:SOURce?
:MEASure:EYE:TDEQ?
:MEASure:EYE:TDEQ:STATus?
:MEASure:EYE:TDEQ:STATus:DETails?
:SPRocess1:TEQualizer:TAPS?
:SPRocess1:TEQualizer:NPRecursors?
```

The Keysight acquisition documentation states that Single does not clear the
display and recommends Clear followed by Single for a new single waveform
acquisition:

- https://helpfiles.keysight.com/scopes/FlexDCA-UG/Content/Topics/Acquisition-Controls/acquisition_controls.htm
- https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/MEASure/aa_return_meas.htm

## Safety and Authorization

### Two-step authorization

`measure-plan` is non-mutating. It creates a canonical JSON plan and prints:

- DCA resource;
- measurement source;
- acquisition mode;
- timeout and poll interval;
- all planned mutation commands; and
- a SHA-256 `measurement_id`.

`measure` requires all of the following before any DCA mutation:

- `--execute` is present;
- the complete `--confirmation-id` exactly matches `measurement_id`;
- recalculating the canonical plan hash produces the stored ID;
- the plan endpoint matches the loaded live configuration;
- I2C ID and revision match configuration; and
- the BERT configuration can be read successfully.

Plan files are immutable inputs. Editing any hashed field invalidates the ID.

### Explicitly non-rollbackable Clear

`:ACQuire:CDISplay` removes acquired response data and cannot be rolled back.
This is why it is included in the displayed, hashed plan. If an error occurs
after Clear, the workflow records the failed stage and does not invent a
rollback. It never changes BERT while trying to recover an acquisition.

### BERT attribution guard

The workflow reads BERT port 1, channel 0 before DCA mutation and again after a
successful TDECQ read. The taps and both NLC values must match exactly. If they
do not, the measurement receipt is `INVALIDATED` and cannot feed FIR tuning.

## CLI Contract

### Create a plan

```powershell
python scripts\run_live_fir.py measure-plan `
  --mode single `
  --timeout 180 `
  --poll-interval 1 `
  --output artifacts\fir_runs\latest\measurement_plan.json
```

Defaults are 180 seconds for the total wait and 1 second between status polls.
Both values must be finite and positive. The poll interval must not exceed the
timeout.

### Execute the plan

```powershell
python scripts\run_live_fir.py measure `
  --plan artifacts\fir_runs\latest\measurement_plan.json `
  --execute `
  --confirmation-id <complete-measurement-id> `
  --receipt artifacts\fir_runs\latest\measurement_receipt.json
```

The default receipt path is `measurement_receipt.json` next to the plan.

Exit codes are:

- `0`: `MEASURED`;
- `3`: `TIMED_OUT` or `INVALIDATED`;
- `4`: `FAILED` after an attempted hardware operation;
- `1`: validation, configuration, or connection error before an acquisition
  receipt can be created.

## Acquisition State Machine

1. Validate the plan, confirmation, endpoint, I2C identity, and initial BERT
   configuration.
2. Open one DCA VISA session for the complete acquisition attempt.
3. Select `FUNC1` and require exact source readback before continuing.
4. Send Stop and Clear.
5. Query TDECQ status immediately after Clear and record a clear-boundary
   observation. If the result is still `CORR`, fail closed because the workflow
   cannot prove that a fresh measurement boundary was established.
6. Send Single and start a monotonic deadline.
7. Poll source, status, and details at the configured interval.
8. While status is not `CORR`, record status transitions and continue waiting.
   `INV` is never an immediate terminal failure.
9. At the deadline, send Stop once and return `TIMED_OUT` with the last status
   and details.
10. On `CORR`, query source, status, TDECQ, and details again as a consistency
    snapshot. Accept only `FUNC1`, `CORR`, and a finite TDECQ value.
11. Read Function 1 equalizer taps and precursor count with the same validation
    already used by the existing adapter.
12. Close the DCA session and read BERT again.
13. Return `MEASURED` only when BERT before and after match exactly; otherwise
    return `INVALIDATED`.

The polling implementation depends on injected monotonic-clock and wait
functions so unit tests do not sleep.

## Plan Document

The plan uses schema version `2.0` and contains:

```json
{
  "schema_version": "2.0",
  "measurement_id": "<sha256>",
  "created_at_utc": "<ISO-8601>",
  "target": {
    "resource": "TCPIP0::Nate-Rog::hislip0,4880::INSTR",
    "function": 1,
    "measurement_source": "FUNC1"
  },
  "acquisition": {
    "mode": "single",
    "timeout_s": 180.0,
    "poll_interval_s": 1.0,
    "commands": [
      ":MEASure:EYE:TDEQ:SOURce FUNC1",
      ":ACQuire:STOP",
      ":ACQuire:CDISplay",
      ":ACQuire:SINGle"
    ]
  }
}
```

`measurement_id` is computed from all fields except `measurement_id` itself.

## Receipt Document

Every attempted acquisition returns a schema-versioned receipt containing:

- `measurement_id` and final status;
- start and finish timestamps;
- DCA identity, target, and verified source;
- ordered command log with timestamps and outcomes;
- clear-boundary observation;
- poll count and status-transition history;
- final TDECQ status, value, details, taps, and precursor count when available;
- BERT configurations before and after;
- I2C identity;
- failed stage and error message when applicable; and
- whether the timeout Stop command succeeded.

Only a receipt satisfying all of these conditions is consumable by future FIR
logic:

```text
status == MEASURED
final_tdecq.status == CORR
verified_measurement_source == FUNC1
bert_before == bert_after
```

## Error Handling

- Source mismatch aborts before Stop, Clear, or Single.
- Failure before Clear preserves existing displayed acquisition data where
  possible.
- Failure after Clear produces `FAILED` with the exact failed stage.
- A timeout attempts one Stop and records its result.
- Invalid/non-finite TDECQ data cannot produce `MEASURED`.
- A changed BERT state produces `INVALIDATED`, not a numeric comparison.
- Receipt writes use the existing atomic temporary-file replacement pattern.

## Test Strategy

Unit tests use fake DCA, BERT, I2C, clock, and wait implementations to prove:

- plan hash stability and tamper rejection;
- exact confirmation and `--execute` gates;
- endpoint and I2C checks occur before DCA mutation;
- exact command ordering;
- source mismatch blocks acquisition;
- a clear boundary is required;
- `INV, INV, CORR` produces one accepted fresh result;
- perpetual `INV` times out and sends Stop;
- query/write failure records the failed stage;
- non-finite TDECQ is rejected;
- BERT changes invalidate an otherwise correct result;
- CLI exit codes and atomic receipt persistence; and
- existing probe, propose, apply, and restore behavior remains unchanged.

After all automated tests pass, commissioning performs one real Single
acquisition with the current FIR `[0, 1, -5, 110, 3, 1, 0]`. The operator sees
the complete plan and confirms its ID before execution. Success requires a
`MEASURED` receipt, Function 1 `CORR`, unchanged BERT readback, matching I2C
identity, and agreement with the DCA GUI.

## Next Increment

After this design is commissioned, the next design will add controlled Run
acquisition and repeated TDECQ sampling. It will reuse the plan/confirmation,
source verification, conditional measurement polling, BERT attribution, and
receipt primitives defined here rather than duplicating them.
