# VCSEL AI Tuning V1.1 Backend Implementation Plan

> **For Codex:** Execute this plan task-by-task with test-driven development. Run only FAKE/replay tests during implementation; never contact real hardware.

**Goal:** Extend the existing VCSEL AI Tuning backend with an isolated V1.1 task engine for AI optimization and sweeps, full-campaign authorization, durable recovery, and `/api/v1.1` endpoints while preserving the existing V1 API and data.

**Architecture:** Keep the current FastAPI process and V1 modules intact. Add a versioned `vcsel_ai_tuning.v1_1` vertical slice with independent domain, persistence, orchestration, hardware transaction boundaries, and API routes. V1.1 uses `state/v1.1/` and `artifacts/v1.1/`; SQLite is authoritative for the global active-task lock, events, observations, sweep progress, authorization, and recovery state.

**Tech Stack:** Python 3.12+, stdlib dataclasses/enums/hashlib/sqlite3/csv/json, FastAPI, Pydantic, pytest, existing VCSEL hardware adapters and pure optimization modules.

**Safety:** Automated tests use FAKE/replay only. Do not initialize or acquire from DCA, write I2C, call BERT `setFir`, run AI tuning against live hardware, or install/upgrade dependencies.

---

## Task 1: Establish the V1.1 domain and isolated defaults

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/__init__.py`
- Create: `src/vcsel_ai_tuning/v1_1/constants.py`
- Create: `src/vcsel_ai_tuning/v1_1/domain.py`
- Create: `src/vcsel_ai_tuning/v1_1/config.py`
- Test: `tests_py/ai_tuning/v1_1/test_domain.py`
- Test: `tests_py/ai_tuning/v1_1/test_config.py`

1. Write failing tests for `product_version == "1.1.0"`, V1.1-only database/artifact defaults, task types, lifecycle states, runtime modes, and raw-integer discrete range generation.
2. Run the two tests and confirm failure because the V1.1 package does not yet exist.
3. Implement immutable enums/dataclasses, deterministic JSON/hash helpers, and range generation using `min + n * step` without appending a non-aligned maximum or clipping.
4. Re-run the focused tests and confirm they pass.

## Task 2: Model dynamic B0 snapshots and immutable authorization

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/baseline.py`
- Create: `src/vcsel_ai_tuning/v1_1/authorization.py`
- Test: `tests_py/ai_tuning/v1_1/test_baseline.py`
- Test: `tests_py/ai_tuning/v1_1/test_authorization.py`

1. Write failing tests for a B0 snapshot containing I2C fields/raw register evidence, seven FIR taps, NLC, DCA fingerprint, identities, version, timestamp, and deterministic hash.
2. Write failing tests showing authorization freezes B0 hash, identities, mapping, DCA configuration, parameter ranges, task/depth/budget/seed/max-points, rollback policy, authorization ID, and time.
3. Implement canonical serialization and hash verification. Reject mutation or start when the current inputs no longer match the frozen authorization.
4. Run focused tests.

## Task 3: Add explicit V1.1 SQLite migrations and the single-task lock

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/migrations.py`
- Create: `src/vcsel_ai_tuning/v1_1/repository.py`
- Test: `tests_py/ai_tuning/v1_1/test_migrations.py`
- Test: `tests_py/ai_tuning/v1_1/test_repository.py`

1. Write failing tests that initialize a fresh V1.1 database, upgrade a copied older schema explicitly, and reject unsupported newer schemas.
2. Write failing concurrency tests proving exactly one nonterminal hardware task can exist globally and a second create/start returns the active task ID.
3. Implement schema-version metadata, tasks, B0 snapshots, authorization, events, observations, sweep points, best states, recovery records, and a transactional singleton active-task row.
4. Ensure every persisted row carries `product_version`, `schema_version`, `campaign_type`, `runtime_mode`, B0 hash, and configuration fingerprint where applicable.
5. Run focused tests with temporary databases outside the worktree.

## Task 4: Implement Script/RegMap safe-field discovery

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/regmap.py`
- Test: `tests_py/ai_tuning/v1_1/test_regmap.py`

1. Write failing fixtures/tests for `ibias_dac`, `vref`, `vbias_ref`, `vga_cur`, `res_gain`, `res_bw`, `gain_ctrl`, `efin_cur`, and `eq`.
2. Test that test/control/disable/spare fields are excluded, ambiguous mappings are `UNAVAILABLE` with a reason, and selected unavailable fields prevent authorization.
3. Implement backend-owned mapping records with register address/bit position never accepted from the frontend.
4. Preserve Script write order and expose read-only scan/validation results.
5. Run focused tests.

## Task 5: Build atomic I2C and BERT transaction boundaries

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/hardware/__init__.py`
- Create: `src/vcsel_ai_tuning/v1_1/hardware/i2c.py`
- Create: `src/vcsel_ai_tuning/v1_1/hardware/bert.py`
- Test: `tests_py/ai_tuning/v1_1/hardware/test_i2c.py`
- Test: `tests_py/ai_tuning/v1_1/hardware/test_bert.py`

1. Write FAKE-adapter tests for stale guards, changed-safe-field-only RMW writes, immediate field/raw readback, Script order, and B0 rollback after mismatch.
2. Write tests for complete BERT `setFir` ordering: internal `Pre3,Pre2,Pre1,Main,Post1,Post2,Post3` to vendor `Pre1,Pre2,Pre3,Main,Post1,Post2,Post3,Upper,Lower`.
3. Require pre-read and full post-write verification of pattern, port/channel, taps, and B0 NLC. Roll back on mismatch.
4. Verify tests make no live I/O and run focused tests.

## Task 6: Build DCA fingerprinting and measurement transactions

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/hardware/dca.py`
- Test: `tests_py/ai_tuning/v1_1/hardware/test_dca.py`

1. Write FAKE tests for fingerprint capture, source verification, clear/start/poll/read ordering, nullable settle delays, invalid sentinel filtering, and measurement artifacts.
2. Test timeout behavior: reconnect, re-check identity/source, retry acquisition exactly once without rewriting the candidate; a second timeout pauses and requests B0 restore.
3. Test configuration drift detection pauses the task, restores B0, and records `DCA_CONFIGURATION_DRIFT`.
4. Implement the transaction with explicit requested/verified source evidence and fail-closed values.
5. Run focused tests.

## Task 7: Implement sweep planning, constraints, and durable resume

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/sweep.py`
- Test: `tests_py/ai_tuning/v1_1/test_sweep.py`
- Test: `tests_py/ai_tuning/v1_1/test_sweep_resume.py`

1. Write failing tests for hardware-exclusive `I2C_SWEEP` and `BERT_FIR_SWEEP`, `SINGLE_PARAMETER` and `FULL_COMBINATION`, deterministic seed/order, original/execution indexes, exact count, and mandatory `max_points` refusal.
2. Test FIR constraints (`Main > 0`, L1 <= 170, B0 NLC) and `SKIPPED_CONSTRAINT` without hardware calls.
3. Test immediate point persistence, `INVALID_TDECQ` continuation, B0 restoration on every terminal/pause path, and no retain-best behavior after successful sweep.
4. Test resume skips completed points and requires preflight/B0 readback; failure enters `RECOVERY_REQUIRED`.
5. Implement the pure planner and persisted executor hooks, then run focused tests.

## Task 8: Implement AI candidate ranking and alternating optimization

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/ranking.py`
- Create: `src/vcsel_ai_tuning/v1_1/ai_flow.py`
- Test: `tests_py/ai_tuning/v1_1/test_ranking.py`
- Test: `tests_py/ai_tuning/v1_1/test_ai_flow.py`
- Test: `tests_py/ai_tuning/v1_1/test_fir_policy.py`

1. Write tests for ranking: valid TDECQ required, lower TDECQ wins, exact TDECQ tie uses higher ER, and valid candidate beats invalid B0.
2. Write replay tests for I2C sparse coarse search, discrete Bayesian fine search, first FIR, same-FIR-history I2C retune, iterative alternation, and depth limits Fast=1/Standard=3/Deep=5/Advanced explicit.
3. Write FIR policy tests for the direct SSPRQ formula using B0 NLC/Main without damping, first-pass restore/exit, three-measurement median confirmation, historical-best restore, three consecutive no-improvement convergence, and invalid DCA taps terminal failure.
4. Implement deterministic state transitions and global-best complete-state tracking using existing pure optimization modules where their contracts match.
5. Run replay-only focused tests.

## Task 9: Implement final verification, abort, recovery, and history apply

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/recovery.py`
- Create: `src/vcsel_ai_tuning/v1_1/history.py`
- Test: `tests_py/ai_tuning/v1_1/test_recovery.py`
- Test: `tests_py/ai_tuning/v1_1/test_history.py`

1. Test final application of the global best, three DCA measurements, median ranking against single B0, winner retention, and `COMPLETED_NO_IMPROVEMENT` rollback.
2. Test abort/full rollback errors, restart detection to `RECOVERY_REQUIRED`, no startup hardware connection/write, and user-triggered recovery to resume or `ABORTED_RECOVERED`.
3. Test history apply captures a temporary rollback snapshot, performs preflight/identity/mapping checks, applies I2C then complete FIR with readbacks, measures DCA, and restores the temporary snapshot on failure.
4. Ensure history apply is not campaign resume and never uses an old campaign B0 as its rollback target.
5. Run focused tests.

## Task 10: Compose the V1.1 service and FAKE/LIVE runtime boundary

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/service.py`
- Create: `src/vcsel_ai_tuning/v1_1/runtime.py`
- Test: `tests_py/ai_tuning/v1_1/test_service.py`
- Test: `tests_py/ai_tuning/v1_1/test_runtime.py`

1. Write service tests for create, preflight, snapshot, scan, validate, B0 freeze, count, authorize, start, pause, resume, abort, recover, status, observations, best state, and paginated sweep results.
2. Test that runtime construction alone performs no hardware connection or write.
3. Compose FAKE/replay adapters by default for tests; keep LIVE construction explicit and lazy using the existing verified adapters.
4. Persist an event and state transition atomically around every task mutation.
5. Run focused tests.

## Task 11: Add `/api/v1.1` without changing V1 routes

**Files:**
- Create: `src/vcsel_ai_tuning/v1_1/api/__init__.py`
- Create: `src/vcsel_ai_tuning/v1_1/api/schemas.py`
- Create: `src/vcsel_ai_tuning/v1_1/api/routes.py`
- Create: `src/vcsel_ai_tuning/v1_1/api/events.py`
- Modify: `src/vcsel_ai_tuning/api/app.py`
- Test: `tests_py/ai_tuning/v1_1/api/test_routes.py`
- Test: `tests_py/ai_tuning/v1_1/api/test_events.py`
- Test: `tests_py/ai_tuning/v1_1/api/test_v1_compatibility.py`

1. Write API tests for runtime health/capabilities, active task conflict with ID, task lifecycle actions, read models, sweep pagination, JSON/CSV/report exports, and history apply.
2. Test SSE replay with `Last-Event-ID` and stable event ordering.
3. Mount the V1.1 router under `/api/v1.1` in the existing app. Keep `/api/campaigns`, current frontend/static hosting, and V1 response behavior unchanged.
4. Map domain failures to stable HTTP/error codes including 409 active-task conflict and recovery-required responses.
5. Run V1.1 API tests plus the existing V1 API compatibility tests.

## Task 12: Document migrations and verify the complete FAKE/replay backend

**Files:**
- Create: `docs/ai-tuning-v1.1-api.md`
- Create: `docs/ai-tuning-v1.1-migrations.md`
- Create: `docs/ai-tuning-v1.1-hardware-verification.md`
- Modify: `README.md`
- Test: `tests_py/ai_tuning/v1_1/test_end_to_end.py`

1. Add an end-to-end FAKE AI task, FAKE sweep/resume task, crash recovery, and history-apply scenario.
2. Document new OpenAPI routes, lifecycle/error states, explicit schema migrations, isolated data paths, frontend-visible states, and the separate hardware read-only/write verification checklist.
3. Run all V1.1 tests, then existing `tests_py/ai_tuning` regression tests when a compatible project Python is available.
4. Inspect the diff for accidental changes to V1 storage, V1 routes, real hardware defaults, I2C/BERT/DCA write paths, or unrelated worktree files.
5. Report exact tests run, any environment blocker, modified files, migration/API changes, hardware verification still required, branch, and commit hash. Do not merge to main.
