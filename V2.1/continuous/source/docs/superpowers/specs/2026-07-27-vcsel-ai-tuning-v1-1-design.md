# VCSEL AI Tuning V1.1 Backend Design

**Date:** 2026-07-27
**Status:** Approved requirements translated into implementation design
**Branch:** `codex/v1.1-ai-tuning`
**Product version:** `1.1.0` / `AI Tuning V1.1`

## 1. Purpose and compatibility boundary

V1.1 extends the existing FastAPI application with campaign orchestration,
dynamic baselines, deterministic hardware transactions, optimization and sweep
execution, persistence, recovery, exports, and SSE. It does not replace or
modify the frozen V1 API, V1 tag, or V1 test directory.

The existing `/api/campaigns` contract remains V1. V1.1 is exposed from the
same FastAPI application under `/api/v1.1`. The application continues to serve
`frontend/dist` through the existing static-hosting boundary.

V1.1 uses isolated defaults:

- SQLite: `state/v1.1/campaigns.sqlite3`
- Artifacts: `artifacts/v1.1/`

No V1 database or artifact is imported, rewritten, migrated, or displayed by
default. V1 and V1.1 must never own the live-hardware lease concurrently.

## 2. Architectural choice

The implementation uses a versioned vertical slice rather than copying all V1
routes or replacing V1 in place:

```text
FastAPI application
├── existing V1 router and service
├── /api/v1.1 router
│   ├── task lifecycle and single-task lock
│   ├── preflight, parameter mapping, and dynamic B0
│   ├── FULL_CAMPAIGN authorization
│   ├── AI optimization and sweep commands
│   ├── recovery and history apply
│   └── observations, exports, report, and SSE
├── V1.1 application service
├── V1.1 campaign engines
├── deterministic hardware transaction layer
├── V1.1 repository and artifact writer
└── existing frontend/dist hosting
```

Common infrastructure is reused only where its behavior is unchanged: FastAPI
application assembly, static frontend hosting, canonical JSON handling, SSE
frame formatting, and selected pure optimization/FIR interfaces. V1.1 domain
models, persistence schema, lifecycle rules, and API schemas remain versioned
because their semantics differ materially from V1.

## 3. Domain model

### 3.1 Task types

`campaign_type` is one of:

- `AI_OPTIMIZATION`
- `SWEEP`

Each task freezes exactly one I2C channel, one BERT port/channel, and one DCA
measurement mapping. The mapping is operator-confirmed and cannot be inferred
from software-visible device topology.

### 3.2 Lifecycle and phase

Lifecycle is separated from the type-specific execution phase.

Lifecycle states:

```text
DRAFT
PREFLIGHTED
B0_FROZEN
AUTHORIZED
RUNNING
PAUSED
ABORTING
ROLLING_BACK
COMPLETED
COMPLETED_NO_IMPROVEMENT
ABORTED
ABORTED_RECOVERED
FAILED
FAILED_INVALID_DCA_TAPS
RECOVERY_REQUIRED
```

Terminal states are `COMPLETED`, `COMPLETED_NO_IMPROVEMENT`, `ABORTED`,
`ABORTED_RECOVERED`, `FAILED`, and `FAILED_INVALID_DCA_TAPS`.
`RECOVERY_REQUIRED` remains hardware-owning until recovery is explicitly
resolved.

AI phases:

```text
PREFLIGHT
B0_MEASUREMENT
I2C_COARSE
I2C_FINE
FIR_FIRST_PASS
I2C_RETUNE
FIR_ITERATIVE
FINAL_VERIFICATION
RESTORE_B0
```

Sweep phases:

```text
PREFLIGHT
B0_MEASUREMENT
SWEEP_EXECUTION
RESTORE_B0
```

### 3.3 Single-task lock

SQLite owns the authoritative single-task lock. A partial unique index or an
equivalent transactionally checked lock row prevents more than one nonterminal
V1.1 task. Creation returns HTTP 409 with `ACTIVE_TASK_EXISTS` and the active
task ID. No queue record is created.

The process-level hardware lease remains a second line of defense. It cannot
replace the database invariant.

## 4. Preflight and dynamic B0

Preflight is read-only. It reads and records:

- DCA identity and critical configuration fingerprint;
- TDECQ source `FUNC1` and Outer ER source `CHAN1A` readbacks;
- I2C identity and current safe-list field/raw-register values;
- BERT identity, pattern, port/channel, seven FIR taps, and NLC;
- communication status for every configured adapter.

Preflight never initializes a device, runs an initialization script, changes a
channel-enable bit, writes FIR/NLC, or starts DCA acquisition.

The operator confirms the I2C channel, BERT port/channel, and DCA mapping.
Saving B0 then freezes a versioned immutable snapshot containing:

- all mapped safe-list I2C field values and raw registers;
- FIR in canonical `(Pre3, Pre2, Pre1, Main, Post1, Post2, Post3)` order;
- Upper/Lower NLC and pattern;
- device identities and operator-confirmed mapping;
- DCA configuration fingerprint;
- canonical JSON hash, creation time, and version.

B0 is permitted outside candidate ranges. The historical fixed FIR
`(0, 9, -28, 104, -8, 1, -3)` and NLC `6/6` are display examples only.

Before the first candidate, the engine measures B0 once. Invalid B0 TDECQ does
not block execution, and this observation does not consume candidate budget.

## 5. Script and RegMap parameter mapping

The Script parser supplies parameter name, current value, and first appearance
order. The RegMap parser supplies register address, bitfield, encoded width,
and hard encoding limits.

Only these names can become tunable:

```text
ibias_dac
vref
vbias_ref
vga_cur
res_gain
res_bw
gain_ctrl
efin_cur
eq
```

Channel control, test registers, `testch`, `power_up_done`, `disable`, `spare`,
and ambiguous mappings are excluded. A missing or conflicting mapping is
returned as `UNAVAILABLE` with a reason; it does not disable unrelated
parameters. A selected unavailable parameter prevents authorization/start.
The API never accepts a client-supplied register address or bit position.

Candidate values are raw integers generated strictly as `min + n * step` while
the value is at most `max`. A nonaligned `max` is not appended. Invalid bounds,
steps, encoded values, or selected unavailable parameters are rejected rather
than clipped.

## 6. FULL_CAMPAIGN authorization

V1.1 enables one `FULL_CAMPAIGN` authorization per task. The authorization is
immutable, hashed, and consumed only by the matching task. It freezes:

- B0 ID and hash;
- device identities and channel mapping;
- DCA fingerprint;
- selected parameters and discrete ranges;
- task type and AI depth or sweep configuration;
- candidate budget;
- sweep seed and required `max_points`;
- rollback target;
- authorization ID and timestamp.

No per-stage modal is required after authorization. Every candidate still
passes deterministic server-side validation against the frozen envelope before
any adapter write.

## 7. Hardware transaction layer

### 7.1 I2C

For a multi-parameter candidate, only changed safe-list parameters are written,
in Script appearance order. Each field uses read-modify-write with expected old
field and raw-register values. Each write is immediately followed by field and
full-register readback. The channel remains enabled and no disable/enable or
full initialization Script is executed.

Any stale state, write failure, or readback mismatch stops the candidate and
enters task rollback policy.

### 7.2 BERT

Each FIR candidate issues one complete `setFir`. Internal tap order is
`Pre3, Pre2, Pre1, Main, Post1, Post2, Post3`; vendor order is
`Pre1, Pre2, Pre3, Main, Post1, Post2, Post3, Upper, Lower`.

Before writing, the complete configuration is read. The write always includes
seven taps and B0 NLC. Afterwards `GetConfiguration` must exactly match taps,
NLC, pattern, port, and channel. Mismatch triggers rollback to the prewrite
configuration.

### 7.3 DCA

After I2C/BERT readback succeeds, the engine immediately performs clear/start,
polls status, and reads the new result. No guessed settling sleep is added.
Profile fields `i2c_settle_time` and `fir_settle_time` remain nullable.

Every Outer ER access selects and verifies `CHAN1A`; every TDECQ access selects
and verifies `FUNC1`. The critical DCA configuration fingerprint is verified
throughout a task. Drift pauses the task, restores B0, and records
`DCA_CONFIGURATION_DRIFT`.

On DCA timeout, the current candidate is not rewritten. The runtime reconnects
DCA, verifies identity and sources, and retries the acquisition once. A second
timeout pauses the task and restores B0.

## 8. AI optimization engine

The orchestrator consumes existing pure optimizer and FIR proposer interfaces;
it does not reimplement optimization mathematics.

Ranking is deterministic:

1. invalid TDECQ cannot win;
2. lower TDECQ wins;
3. only exactly equal TDECQ uses higher ER as tie-breaker;
4. no formal ER threshold or minimum TDECQ-improvement threshold is used.

Execution order is coarse I2C, fine I2C, first-pass FIR, I2C retune, iterative
FIR and I2C retune alternation, then final verification.

Coarse uses global sparse candidates. Fine uses the existing discrete Bayesian
interface. Retune uses local Bayesian candidates and trains only on observations
whose FIR exactly matches the current FIR.

The first FIR proposal consumes the latest valid DCA equalizer taps and the
existing first-pass formula. Later proposals use the existing iterative
formula. Targets are applied directly without 25/50/100 percent damping.
Pattern is SSPRQ, Main remains B0 Main, NLC remains B0 NLC, Main must be
positive, and FIR L1 must not exceed 170.

If the first FIR result does not beat B0, B0 FIR is restored and FIR/retune
stops. A first apparent improvement is measured two more times; the median of
three must strictly beat the historical best. Accepted improvement clears the
non-improvement count; rejection restores the best FIR and increments it.
Three consecutive non-improvements converge FIR.

Missing, wrong-count, NaN/Inf, or invalid-Main DCA equalizer taps fail the task
immediately as `FAILED_INVALID_DCA_TAPS` and restore B0 without retry.

Alternation stops when FIR does not improve, retune does not improve, or the
depth limit is reached: Fast 1, Standard 3, Deep 5, Advanced explicit.

Final verification applies the global best complete state, performs exact
readback, and measures three times. Median TDECQ is compared with the single B0
measurement. If final does not win, B0 is restored and the result is
`COMPLETED_NO_IMPROVEMENT`; otherwise final state is retained.

## 9. Sweep engine

Sweep is a separate task, with hardware type `I2C_SWEEP` or
`BERT_FIR_SWEEP`; cross-hardware combinations are rejected. Modes are
`SINGLE_PARAMETER` and `FULL_COMBINATION`.

I2C sweeps use mapped safe-list raw integers. FIR sweeps use seven taps, B0
NLC, positive Main, and L1 at most 170. Invalid FIR combinations are persisted
as `SKIPPED_CONSTRAINT` without hardware calls.

Single-parameter mode changes one parameter from B0 at a time and restores that
parameter before advancing. Full combination generates every point, uses the
frozen seed for deterministic execution order, and stores both original and
execution indexes. It never samples. Point count above required `max_points`
prevents authorization/start.

Each legal point is written, read back, measured once, and immediately committed
to SQLite and artifacts. Invalid TDECQ records `INVALID_TDECQ` and continues.
Points transition directly without intermediate B0 restore. Completion, pause,
abort, or failure restores B0; a successful sweep never leaves the best point
applied.

Resume re-runs preflight, verifies fingerprint and identities, restores B0 when
needed, verifies readback, and resumes from the first uncompleted execution
index using the original seed/order. Restore failure is `RECOVERY_REQUIRED`.

## 10. Rollback, recovery, and history apply

Write/readback failure, disconnect, DCA drift, and abort stop candidate
execution and attempt B0 restore. Rollback is recorded step by step. Failure is
`RECOVERY_REQUIRED`.

On application startup, unfinished hardware-owning tasks are marked
`RECOVERY_REQUIRED` without connecting to hardware. Operator-triggered recovery
performs preflight and B0 restore, then permits resume or terminal
`ABORTED_RECOVERED`.

History apply is a new independent transaction, not campaign resume. It first
requires no active task, performs read-only preflight, and captures the current
hardware state as a temporary rollback target. It applies I2C with per-field
readback, applies complete FIR with readback, and measures once. Success records
`HISTORY_RESULT_APPLIED`. Failure restores the temporary pre-click snapshot;
restore failure enters `RECOVERY_REQUIRED`.

## 11. Persistence and artifacts

V1.1 SQLite uses explicit numbered migrations recorded in `schema_migrations`.
The initial V1.1 migration creates:

- `campaigns` and the single-active-task invariant;
- `b0_snapshots`;
- `authorizations`;
- append-only `events`;
- `observations`;
- `sweep_points`;
- `hardware_transactions` and readbacks;
- `best_results`;
- `recovery_actions`;
- artifact metadata.

Migrations are transactional and refuse unknown/newer schema versions. Tests
copy fixture databases into temporary paths. V1 paths are never migration
inputs.

Each task artifact root contains canonical task JSON, append-safe event and
observation data, sweep JSON/CSV when applicable, raw hardware responses,
recovery evidence, and final Markdown report. API responses and artifacts carry
`product_version=1.1.0`, `schema_version`, `campaign_type`, `runtime_mode`,
`b0_hash`, and `configuration_fingerprint`.

## 12. FastAPI contract

All V1.1 routes use `/api/v1.1` and the existing application instance.

```text
GET    /runtime/health
GET    /runtime/capabilities
GET    /tasks/active
POST   /tasks
GET    /tasks/{task_id}
GET    /tasks/{task_id}/events
GET    /tasks/{task_id}/events/stream
GET    /tasks/{task_id}/observations
GET    /tasks/{task_id}/best-result
GET    /tasks/{task_id}/sweep-results
GET    /tasks/{task_id}/export.json
GET    /tasks/{task_id}/export.csv
GET    /tasks/{task_id}/report
POST   /tasks/{task_id}/preflight
GET    /tasks/{task_id}/hardware-snapshot
POST   /tasks/{task_id}/parameters/scan
POST   /tasks/{task_id}/parameters/validate
POST   /tasks/{task_id}/b0
POST   /tasks/{task_id}/sweep/count
POST   /tasks/{task_id}/authorize
POST   /tasks/{task_id}/start
POST   /tasks/{task_id}/pause
POST   /tasks/{task_id}/resume
POST   /tasks/{task_id}/abort
POST   /tasks/{task_id}/recover
POST   /history/{result_id}/apply
```

SSE uses persisted sequence IDs and supports `Last-Event-ID`. Pagination uses
bounded `limit` and nonnegative `offset`. Downloads stream already-persisted
content and never trigger hardware access.

Representative error codes include:

```text
ACTIVE_TASK_EXISTS
TASK_NOT_FOUND
INVALID_TASK_STATE
PREFLIGHT_REQUIRED
CHANNEL_MAPPING_UNCONFIRMED
PARAMETER_UNAVAILABLE
INVALID_DISCRETE_RANGE
SWEEP_MAX_POINTS_EXCEEDED
FULL_CAMPAIGN_AUTHORIZATION_REQUIRED
AUTHORIZATION_MISMATCH
STALE_HARDWARE_STATE
I2C_READBACK_MISMATCH
BERT_READBACK_MISMATCH
DCA_SOURCE_MISMATCH
DCA_CONFIGURATION_DRIFT
DCA_TIMEOUT_RETRY_EXHAUSTED
FAILED_INVALID_DCA_TAPS
ROLLBACK_FAILED
RECOVERY_REQUIRED
HISTORY_APPLY_FAILED
```

Conflict/state errors use 409, validation uses 422, missing resources use 404,
and hardware availability/failure uses 503 unless a persisted task state is the
primary response.

## 13. Runtime modes and safety

`FAKE` is the default development/test runtime and uses deterministic adapters
plus temporary V1.1 storage. `LIVE` requires explicit configuration and the
process-level exclusive hardware lease. App import/startup never connects to or
writes hardware. Hardware adapters are opened only by explicit preflight or
authorized actions.

Automated tests use FAKE/replay only. LIVE acceptance is separate and requires
operator authorization.

## 14. Verification strategy

Tests are written before production changes and cover:

- transactional single-task lock and no queue;
- dynamic B0 hashing/freezing;
- FULL_CAMPAIGN envelope and deterministic candidate validation;
- Script/RegMap mapping and exact discrete ranges;
- I2C ordered write/readback and BERT complete setFir/readback;
- DCA source binding, drift, timeout retry, and invalid taps;
- AI direct FIR targets, median confirmation, convergence, and depth limits;
- both sweep hardware types/modes, seed ordering, constraint skips, per-point
  persistence, pause/resume, and invalid-TDECQ continuation;
- rollback, crash recovery, temporary history rollback;
- migrations on copied/temporary databases;
- V1.1 path isolation;
- API conflicts, pagination, exports, SSE resume, and frontend hosting;
- proof that automated tests never construct LIVE hardware adapters.

Hardware-only acceptance remains required for physical channel mapping, Script
and RegMap samples, vendor BERT ordering/readback, I2C raw-register behavior,
DCA configuration fingerprint coverage, real timeout/reconnect behavior, and
measured settling profiles.
