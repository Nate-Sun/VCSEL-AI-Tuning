# AI Tuning V1.1 backend delivery contract

## Release and compatibility

- Product/package/OpenAPI version: `1.1.0`.
- SQLite schema version: `2`.
- The existing V1 `/api/campaigns` API and V1 execution path are unchanged.
  V1.1 is added to the same FastAPI application and OpenAPI document under
  `/api/v1.1`.
- When `frontend/dist/index.html` exists, the same FastAPI application continues
  to serve that build, its assets, and SPA fallbacks. `VCSEL_AI_FRONTEND_DIST`
  or the explicit application-factory argument may override the directory.
  API paths are never handled by the SPA fallback.

## Storage, schema migration, and data isolation

V1.1 owns separate persistence roots:

- SQLite: `state/v1.1/campaigns.sqlite3`
- Artifacts: `artifacts/v1.1/`

There is no fallback to the V1 `state/ai_tuning` or `artifacts/ai_tuning`
locations and no implicit import of V1 data. V1 and V1.1 share hardware
authority so that only one hardware-owning operation can run, but they do not
share campaign rows, events, B0 snapshots, authorizations, observations, sweep
points, best states, recovery records, or artifacts.

Schema initialization and upgrades are explicit and transactional:

1. A new empty database receives `schema_meta`, schema 1, then schema 2.
2. A valid schema-1 V1.1 database is migrated in place to schema 2. The
   persisted `schema_version` on tasks, events, B0, authorizations,
   observations, sweep points, best states, and recovery records is updated to
   `2` in the same migration.
3. Schema 2 adds the partial unique index
   `one_open_history_apply_recovery_per_task` on
   `(task_id, reason_code)` where `status = 'OPEN'` and
   `reason_code = 'HISTORY_APPLY_ROLLBACK_FAILED'`. This permits at most one
   unresolved History Apply rollback failure for a task.
4. Before creating the index, migration scans for legacy duplicate OPEN
   History Apply recovery rows. If any are present, it raises
   `DatabaseVersionError` and rolls back the whole migration; schema metadata
   and existing rows remain at schema 1. It never deletes or guesses which
   recovery is authoritative.
5. A nonempty database without `schema_meta`, an incompatible product version,
   malformed metadata, or a schema newer than `2` is rejected fail closed.

The schema contains `tasks`, `active_hardware_task`, `task_events`,
`b0_snapshots`, `task_authorizations`, `observations`, `sweep_points`,
`best_states`, and `recovery_records`. Persisted records carry version,
campaign/runtime, B0-hash, and configuration-fingerprint evidence as applicable.

## V1.1 HTTP API

Requests use strict Pydantic models: unknown fields and non-strict type
coercions are rejected. Parameter selection exposes only `name`, integer
`min`, integer `max`, and positive integer `step`; register addresses and bit
fields are not accepted over the wire.

| Method | Path | Input and semantics |
|---|---|---|
| GET | `/api/v1.1/runtime/health` | Reports runtime/gateway and worker health; an OPEN recovery overrides `status` to `RECOVERY_REQUIRED`. |
| GET | `/api/v1.1/runtime/capabilities` | Reports product/schema/runtime plus gateway and runner capabilities. |
| POST | `/api/v1.1/tasks` | Creates `AI_OPTIMIZATION` or `SWEEP`. Requires `task_id`, task type, operator-confirmed channel mapping, parameter ranges, depth, and candidate budget. `ADVANCED` requires `advanced_iteration_limit`; `SWEEP` additionally requires hardware, mode, seed, and `max_points`. |
| GET | `/api/v1.1/tasks/active` | Returns the single active V1.1 hardware task or `null`. |
| GET | `/api/v1.1/hardware/snapshot` | Performs an explicit read-only preflight capture and returns a short-lived snapshot token, B0 evidence, and mapping. |
| POST | `/api/v1.1/mapping/scan` | Explicitly scans Script/RegMap bindings and returns the discovered mapping. |
| POST | `/api/v1.1/parameters/validate` | Accepts an array of parameter selections and validates names/ranges against the scanned mapping. |
| POST | `/api/v1.1/sweeps/count` | Accepts hardware, mode, parameters, seed, and `max_points`; returns the deterministic point count and fails if it exceeds the explicit limit. It never samples automatically. |
| POST | `/api/v1.1/tasks/{task_id}/b0` | Body: `snapshot_token` and literal `operator_confirmed: true`. Verifies the task's exact channel mapping and freezes dynamic B0. |
| POST | `/api/v1.1/tasks/{task_id}/authorization` | Body: `authorization_id`, timezone-aware `expires_at`, and literal `operator_confirmed: true`. Freezes one immutable `FULL_CAMPAIGN` authorization. |
| POST | `/api/v1.1/tasks/{task_id}/start` | Re-attests persisted B0, identities, mapping, channel mapping, and DCA fingerprint, prepares the runner, durably enters `RUNNING`, then dispatches. |
| POST | `/api/v1.1/tasks/{task_id}/pause` | Cooperatively stops after verified B0 restoration, then persists `PAUSED`. |
| POST | `/api/v1.1/tasks/{task_id}/resume` | Re-preflights, restores/verifies B0, durably re-enters `RUNNING`, and resumes while skipping already persisted sweep points. |
| POST | `/api/v1.1/tasks/{task_id}/abort` | Body: nonblank `reason`. Before start this cancels without touching hardware; while running/paused it restores B0 before terminal `ABORTED`. |
| POST | `/api/v1.1/tasks/{task_id}/recovery` | Body: `action` (`RESUME` or `ABORT_RECOVERED`) and literal `operator_confirmed: true`. Recovery is explicit; resume re-attests/restores before execution, while abort-recovered restores B0 and terminates. |
| GET | `/api/v1.1/tasks/{task_id}` | Returns lifecycle state and frozen task metadata. |
| GET | `/api/v1.1/tasks/{task_id}/events?after_sequence=N` | Returns persisted audit events after a cursor. |
| GET | `/api/v1.1/tasks/{task_id}/events/stream` | SSE event stream; resumes from the `Last-Event-ID` header. |
| GET | `/api/v1.1/tasks/{task_id}/observations` | Returns persisted B0/candidate observations. |
| GET | `/api/v1.1/tasks/{task_id}/best` | Returns the persisted optimizer-owned complete best state, or `null`. |
| GET | `/api/v1.1/tasks/{task_id}/sweep-points?offset=N&limit=N` | Returns a paginated page; `limit` is 1–1000. |
| GET | `/api/v1.1/tasks/{task_id}/sweep-results.json` | Downloads the current complete JSON sweep export. |
| GET | `/api/v1.1/tasks/{task_id}/sweep-results.csv` | Downloads the current flattened CSV sweep export. |
| GET | `/api/v1.1/tasks/{task_id}/final-report` | Returns task metadata, observations, and event payloads. |
| POST | `/api/v1.1/history/apply` | Body: `source_task_id` and `observation_id`. Applies only a persisted complete best state, under exclusive authority, using click-time hardware state as the temporary rollback target. |
| POST | `/api/v1.1/recovery/history/{recovery_id}` | Body: literal `operator_confirmed: true`. Restores the persisted click-time snapshot for an OPEN History Apply rollback failure, resolves it atomically, then releases retained authority. Retrying an already resolved record is idempotent and does not touch hardware again. |

The channel mapping is an explicit, immutable part of task configuration, B0,
and authorization: `i2c_channel` is an integer; `bert_port` and
`bert_channel` are strings; DCA TDECQ is `FUNC1`; DCA Outer ER is `CHAN1A`.

## Frontend lifecycle, point statuses, errors, and events

The frontend must render persisted state rather than infer state from a button
click. Lifecycle states are:

- Setup: `DRAFT`, `PREFLIGHTED`, `B0_FROZEN`, `AUTHORIZED`.
- Active: `RUNNING`, `PAUSED`, `ABORTING`, `ROLLING_BACK`.
- Successful terminal: `COMPLETED`, `COMPLETED_NO_IMPROVEMENT`.
- Other terminal: `ABORTED`, `ABORTED_RECOVERED`, `FAILED`,
  `FAILED_INVALID_DCA_TAPS`.
- Fail-closed and operator-action state: `RECOVERY_REQUIRED`. It is
  intentionally nonterminal and retains hardware authority until recovery is
  resolved.

Sweep-point statuses are `PLANNED`, `COMPLETED`, `INVALID_TDECQ`,
`SKIPPED_CONSTRAINT`, and `FAILED`. A non-finite or missing TDECQ is persisted
as `INVALID_TDECQ`; it is never accepted as a valid measurement.

Structured HTTP error codes currently registered by the FastAPI application
are:

| HTTP | `detail.code` | Meaning |
|---|---|---|
| 409 | `ACTIVE_HARDWARE_TASK` | V1, V1.1, or retained recovery authority already owns hardware; response includes `active_task_id`. |
| 404 | `TASK_NOT_FOUND` | V1.1 task lookup failed. |
| 409 | `AUTHORIZATION_DRIFT` | Frozen authorization evidence differs from current attestation. |
| 409 | `TASK_STATE_CONFLICT` | A V1.1 repository or lifecycle operation is invalid for the persisted state. |
| 409 | `TASKCONCURRENCYERROR` | Persisted lifecycle compare-and-swap lost or state changed concurrently. |
| 409 | `HARDWARETRANSACTIONERROR` | Candidate/rollback hardware transaction failed. |
| 409 | `DCA_CONFIGURATION_DRIFT` | Frozen DCA fingerprint differs from current readback. |
| 422 | `PARAMETERSELECTIONERROR` | Parameter name/range is not valid for the scanned mapping. |
| 422 | `SWEEPLIMITEXCEEDED` | Deterministic sweep point count exceeds `max_points`. |
| 503 | `event_source_unavailable` | Event source was not injected. |
| 422 | `invalid_last_event_id` | SSE `Last-Event-ID` is invalid. |
| 404 | `task_not_found` | SSE task lookup failed. |
| 409 | `RUNTIME_ASSEMBLY_ERROR`, `CAPABILITY_UNAVAILABLE`, `READ_ONLY_MODE`, `RUNTIME_CLOSED`, or `HARDWARE_RESOURCE_BUSY` | Shared runtime assembly/capability errors. |

Ordinary strict request validation returns FastAPI/Pydantic HTTP 422 validation
details. Frontend code must not manufacture exception-class names that are not
registered as `detail.code` values.

Persisted V1.1 event types emitted by the current service, repository, and
persistent runner are:

- Setup/start: `TASK_CREATED`, `PREFLIGHT_CONFIRMED`, `B0_FROZEN`,
  `FULL_CAMPAIGN_AUTHORIZED`, `TASK_STARTED`, `TASK_DISPATCH_FAILED`.
- Pause/resume/cancel: `TASK_PAUSED_B0_RESTORED`,
  `TASK_RESUMED_AFTER_PREFLIGHT`, `TASK_RESUME_FAILED`,
  `TASK_CANCELLED_BEFORE_START`.
- Abort/recovery: `TASK_ABORT_REQUESTED`, `B0_ROLLBACK_STARTED`,
  `B0_ROLLBACK_FAILED`, `TASK_ABORTED_B0_RESTORED`,
  `RECOVERY_B0_RESTORED`, `RECOVERY_RESUME_FAILED`,
  `PROCESS_RESTART_DETECTED`.
- Runner terminal: `TASK_COMPLETED_B0_RESTORED`, `TASK_ROLLBACK_FAILED`,
  `TASK_EXECUTION_FAILED_B0_RESTORED`, `AI_EXECUTION_TERMINAL`,
  `AI_EXECUTION_FAILED_B0_RESTORED`, `AI_ROLLBACK_FAILED`.
- History Apply: `HISTORY_RESULT_APPLIED`,
  `HISTORY_RESULT_APPLY_FAILED_ROLLED_BACK`,
  `HISTORY_APPLY_ROLLBACK_FAILED`, `HISTORY_APPLY_RECOVERED`.

## Authorization, execution, and recovery semantics

`FULL_CAMPAIGN` is the only authorization mode. One operator confirmation
freezes task type, B0 hash, three hardware identities, mapping hash, exact
channel mapping, DCA configuration fingerprint, authorized parameter grids,
depth/iteration policy, candidate budget, seed, Sweep `max_points`, expiry,
and `RESTORE_B0` rollback policy. Start and recovery reconstruct this evidence
from SQLite and fail closed on drift or expiry.

The default FAKE runtime uses a persistent, single-worker, no-queue task runner:

- Start has two phases: synchronous preparation performs validation without
  mutation; dispatch occurs only after `RUNNING` is durably persisted.
- SWEEP planning is reconstructed from persisted task payload plus B0. Both
  `I2C_SWEEP`/`BERT_FIR_SWEEP` and
  `SINGLE_PARAMETER`/`FULL_COMBINATION` are deterministic for the frozen seed.
- Before measurement, the runner preflights, verifies/restores B0, verifies B0
  again, and persists the B0 observation once.
- Every completed, invalid, or constraint-skipped point is committed to SQLite
  first. `sweep_results.json` and `sweep_results.csv` are then atomically
  refreshed for that point, so progress survives pause or process loss.
- Pause and abort are cooperative. Both restore B0 before handing lifecycle
  control back to the service. Resume preserves the same seed and execution
  order and skips completed original indexes.
- Normal completion restores B0 and enters `COMPLETED`; it never automatically
  applies the best candidate. Ordinary execution failure with verified restore
  enters `FAILED`. An unverifiable restore enters `RECOVERY_REQUIRED` and
  retains authority.
- On startup, interrupted `RUNNING`, `PAUSED`, `ABORTING`, or `ROLLING_BACK`
  tasks are marked `RECOVERY_REQUIRED` using database access only. Startup does
  not call a gateway or mutate hardware.

Candidate transactions remain fail closed: safe-list I2C writes use ordered,
stale-state-guarded RMW plus field/raw-register readback; BERT uses one complete
seven-tap `setFir` with B0 NLC and verifies port/channel/pattern/taps/NLC; DCA is
measured only after write verification and current fingerprint/source
verification. Candidate failures attempt one configured B0 rollback. History
Apply instead rolls back to the complete click-time state captured immediately
before applying the persisted result.

## Runtime capabilities and optimizer integration boundary

For the assembled default FAKE runtime, `/api/v1.1/runtime/capabilities`
truthfully reports:

- `runtime_mode: "FAKE"`, `hardware_connected: false`,
  `hardware_mutation: false`, and `fake: true`;
- `sweep_execution_ready: true`;
- `ai_execution_ready: false`.

The backend provides an injected cooperative AI execution boundary with
`run(stop_event)` and `restore_b0()`. It does not copy or reimplement objective
comparison, Bayesian fitting, FIR formulas, medians, or alternation policy.
An injected implementation must observe `stop_event` and return. Runner
shutdown deliberately waits for the worker instead of timing out and falsely
claiming that execution stopped or B0 was restored; a non-cooperative AI
implementation can therefore block shutdown indefinitely. Production
integration must guarantee cooperative termination at every blocking boundary.
The separate optimizer worktree has not been integrated into this backend
worktree, so the default AI execution factory remains absent. Starting an
`AI_OPTIMIZATION` task therefore fails synchronously while the task remains
`AUTHORIZED`, before hardware mutation and without falling back to duplicate
math. Production AI readiness requires committing/integrating the optimizer
package and explicitly injecting its execution factory.

## Automated verification boundary and hardware acceptance

Backend automation uses fake/injected gateways and executors. It does not open
COM7, VISA/HiSLIP, or BERT UDP sockets; send SCPI acquisition commands; write
I2C; call real `setFir`; change FIR/NLC; initialize instruments; or run a real
AI tuning campaign. Passing automated tests is therefore software acceptance,
not hardware acceptance.

The following checks still require a separate, explicitly authorized hardware
session:

1. Read-only LIVE preflight identifies I2C COM7/115200/0xA6 and ID/REV,
   BERT `192.168.3.2:9000` with `port1`/channel `0`, and the configured FlexDCA
   VISA resource, without changing instrument setup.
2. Dynamic B0 captures exact I2C safe-list values and raw bytes, all seven FIR
   taps, upper/lower NLC, SSPRQ pattern, immutable channel mapping, hardware
   identities, and DCA fingerprint.
3. A single authorized safe I2C candidate verifies both field value and full
   raw register while preserving unrelated bits, then restores and verifies B0.
4. A single authorized BERT candidate sends exactly one complete seven-tap
   `setFir`, verifies port/channel/pattern/taps/NLC, then restores and verifies
   B0.
5. One DCA acquisition executes clear/single, explicitly selects and reads back
   Outer ER source `CHAN1A`, then explicitly selects and reads back TDECQ source
   `FUNC1`; status/details/value are accepted only after their matching source
   readback.
6. Operator-supervised negative cases cover disconnected I2C/BERT/DCA,
   write/readback mismatch, DCA fingerprint/source drift, acquisition timeout
   and one-reconnect retry, Pause, Abort, process restart, and failed rollback
   leading to retained `RECOVERY_REQUIRED` authority.
7. History Apply is verified with a persisted complete best state, click-time
   rollback, failure with successful rollback, failure with OPEN durable
   recovery, and explicit idempotent recovery.
8. Real per-candidate duration is measured before setting production defaults
   for I2C settle time, FIR settle time, authorization expiry, candidate budget,
   or Sweep `max_points`.

Until this checklist is completed, LIVE capability must not be inferred from
FAKE readiness, and the existing V1 backend and hosted frontend build remain
the production-compatible paths.
