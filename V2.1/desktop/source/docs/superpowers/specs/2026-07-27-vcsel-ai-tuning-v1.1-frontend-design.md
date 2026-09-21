# VCSEL AI Tuning V1.1 Frontend Design

## Scope

Upgrade the V1 read-only Mission Control dashboard to a V1.1 FastAPI-controlled frontend covering AI Optimization, Sweep, Recovery, and historical result application.

The browser never connects to DCA, I2C, or BERT directly. It never emits raw SCPI, raw I2C, register addresses, or `setFir`; generates AI candidates; calculates hardware safety limits; accepts candidates; or decides rollback. Hardware state changes are shown only after a FastAPI response or validated SSE confirmation.

Work is restricted to `C:\Users\NateS\Documents\VCSEL\.worktrees\v1.1-frontend` on `codex/v1.1-frontend`. The V1 tag, V1 test directory, backend, hardware modules, other branches, and other Worktrees remain untouched.

## Product Structure

Use the approved Guided Operations structure while retaining Photon Lab Noir, Mission Control hierarchy, bilingual labels, IBM Plex Mono, Newsreader, responsive behavior, and keyboard accessibility.

- `/`: runtime, DCA/I2C/BERT, current metrics, active-task gate, history, Recovery, and task entry points.
- `/tasks/new`: eight-step task wizard.
- `/campaigns/:campaignId/optimization`: AI Optimization run.
- `/campaigns/:campaignId/sweep`: Sweep run and paginated points.
- `/campaigns/:campaignId/result`: type-specific results.
- `/recovery/:campaignId`: exclusive Recovery flow.
- `/history/:campaignId`: historical result and Apply flow.

## Runtime Gates

- `FAKE`: operations affect only fake FastAPI and remain visibly simulated.
- `LIVE_TUNING` wire mode is presented as `LIVE`; controls require explicit capabilities.
- `READ_ONLY`: all mutations are hidden or disabled.
- `RECOVERY_REQUIRED`: derived from canonical runtime/campaign state, globally blocks all mutations except Recovery.
- `RUNNING` or `PAUSED`: blocks creation of another Draft and shows `View current task`.

## State and Contracts

TanStack Query owns server state. A colocated reducer owns unfinished wizard input. Server responses always replace hardware-related local values; no optimistic hardware update is allowed.

Existing V1 endpoints are real. Missing V1.1 endpoints are development contracts implemented only in:

- `frontend/src/features/ai-tuning/api/client.ts`;
- `frontend/src/features/ai-tuning/schemas/contracts.ts`;
- `frontend/src/mocks/handlers.ts`.

Components consume parsed domain objects and contain no ad hoc wire fields. Missing production endpoints are listed explicitly in the final report.

SSE retains validation, ordering, deduplication, gap detection, `Last-Event-ID`, capped backoff, REST invalidation, stale/reconnecting states, and terminal close.

## Task Wizard

1. Task Type: AI Optimization or Sweep.
2. Hardware Mapping: I2C, BERT, DCA plus required same-physical-link confirmation.
3. Hardware Snapshot/B0: server snapshot, reference FIR, and `Save current state as B0`.
4. Script/RegMap: select/upload and server scan; only the nine safe I2C parameters are selectable.
5. Integer Ranges: enabled/min/max/step and server grid preview; `0..10 step 3` is `0,3,6,9`; B0 outside the range warns but does not fail.
6. Profile: Fast, Standard, Deep, Advanced with server budget and alternating rounds.
7. Sweep: I2C or BERT FIR, Single or Full, seed/max points/counts/constraint results; NLC locked to B0; no time estimate.
8. FULL_CAMPAIGN: one server authorization summary and one `Authorize and Start` action.

## Run, Recovery, and Results

Optimization displays server state, stages, rounds, first/iterative, requested/applied/readback, current and best metrics, FIR no-improvement count, DCA Equalizer validity, events, rollback, and convergence. It states that FIR uses the DCA formula, is not FIR Bayesian search, and has no damping. Controls are Pause, Resume, and Abort; no manual acceptance exists.

Sweep displays hardware/mode/seed, counts, original/execution indexes, parameters, readbacks, metrics, invalid/skipped reasons, device state, and SSE. Pause/Resume expose B0 restoration and Preflight. Results use server pagination/filter/sort/export, server-confirmed best highlighting, and explicit B0 restored state. Results are never auto-applied.

DCA drift pauses the task, shows configuration differences and B0 restore state, and blocks Resume until server Preflight allows it.

Recovery is the only mutation surface during `RECOVERY_REQUIRED`. Restore/verify must succeed before Continue or `ABORTED_RECOVERED` is enabled.

Historical Apply calls the API on the first click with no second modal, disables duplicate clicks, displays the eight server execution stages, and routes to Recovery on failure requiring it. It is never presented as continuation of the old campaign.

All Campaigns and Reports show `product_version`, `schema_version`, `runtime_mode`, and Campaign Type without relabeling V1 history.

## Verification

Update tests before implementation. Cover runtime modes and gates, the full wizard, allowlist, discrete grids, authorization, run states, controls, drift, Recovery, final verification, exports, history Apply, SSE reconnect, API errors, accessibility, and responsive layouts. Playwright uses only MSW FAKE runtime and rejects unexpected or raw-hardware requests.
