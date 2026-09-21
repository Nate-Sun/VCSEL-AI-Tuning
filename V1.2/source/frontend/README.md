# VCSEL AI Tuning Dashboard V1.1

React + TypeScript control surface for AI Optimization, Sweep, Recovery, and historical result Apply. The browser never imports hardware code and never creates low-level I2C, BERT, DCA, SCPI, or `setFir` commands.

## Safety model / 安全模型

- `FAKE`: simulated responses come either from MSW for isolated UI tests or from the real localhost FastAPI FAKE runtime for integration tests; neither path contacts physical instruments.
- `REPLAY`: persisted V1.1 evidence is parsed with the V1.1 runtime contract and remains non-hardware execution.
- `READ_ONLY`: monitoring is available; every mutation control is disabled.
- `LIVE`: controls call typed FastAPI endpoints only. The banner explicitly warns that authorized operations affect physical hardware.
- `RECOVERY_REQUIRED`: a global lock permits only the canonical Recovery workflow until B0 restoration is verified.
- Browser state is never treated as hardware truth. Requested, applied, readback, best-point, outcome, rollback, and Apply stages are displayed only after validated server responses.

## Routes / 页面

- `/` — V1.1 Overview, instrument state, current TDECQ/ER, active-task interlock, and history.
- `/tasks/new` — eight-step AI Optimization or Sweep wizard.
- `/runs/optimization/:taskId` — V1.1 task state, persisted observations, controls, B0 evidence, and task events.
- `/runs/sweep/:taskId` — V1.1 task state, exact server pagination, persisted point fields, and CSV/JSON exports.
- `/tasks/:taskId/result` — V1.1 final report, persisted observations, and Sweep exports.
- `/recovery/:taskId` — exclusive baseline restore and verification workflow.
- `/campaigns/:campaignId` — retained V1 monitoring detail.
- `/campaigns/:campaignId/result` — version-aware result, rollback evidence, artifacts, and server-staged historical Apply.

## API boundary

Stable V1 monitoring reads and finalized V1.1 task operations are centralized in `src/features/ai-tuning/api/client.ts`. There is no temporary endpoint registry or combined authorize/start route. UI components do not assemble API paths.

Canonical V1.1 groups:

| Group | Purpose |
| --- | --- |
| `/api/v1.1/runtime/capabilities`, `/tasks/active` | Authoritative V1.1 runtime gate, profile budgets, and active-task interlock |
| `/api/v1.1/hardware/snapshot`, `/mapping/scan`, `/parameters/validate`, `/sweeps/count` | Pre-create readback, mapping, parameter, and deterministic Sweep evidence |
| `/api/v1.1/tasks` and `/api/v1.1/tasks/{id}/*` | Complete Draft creation, B0 freeze, authorization, start, controls, recovery, results, and export |
| `/api/v1.1/history/apply` | Apply a persisted complete best observation using `source_task_id` and `observation_id` |
| `/api/v1.1/recovery/history/{recovery_id}` | Resolve an OPEN History Apply rollback failure |

The Wizard completes snapshot, mapping scan, parameter validation, and Sweep count before sending the complete `CreateTaskRequest`. It then freezes B0. A single final click sends authorization with a 30-minute TTL, verifies the returned state is `AUTHORIZED`, and only then sends start; the page uses the server-returned `RUNNING` task state.

The current Wizard exposes I2C Sweep only. BERT FIR Sweep remains fail-closed
until the backend mapping contract exposes FIR tap parameter names to the
browser. Run and result pages read task state, events, observations, points,
best state, and reports only from `/api/v1.1/tasks/{id}/*`; they never reinterpret
a V1.1 task ID as a legacy campaign ID. Sweep pages do not synthesize global
best/progress, device health, DCA drift, or B0-restore state from a single page.

## Local verification / 本地验证

Use the isolated frontend Worktree and FAKE runtime:

```powershell
cd C:\Users\NateS\Documents\VCSEL\frontend
pnpm test
pnpm typecheck
pnpm lint
pnpm build
pnpm e2e
pnpm e2e:real
```

`pnpm e2e` launches Vite in `test` mode using `.env.test`
(`VITE_ENABLE_MOCKS=true`). `pnpm e2e:real` disables MSW, starts the production
FastAPI application on localhost with a fresh temporary FAKE SQLite database,
drives the browser through create → B0 → authorization → start, and verifies
the V1.1 run page reached after Start. The Python HTTP smoke test separately
records request/response bodies. These tests block external VISA/serial driver
imports and do not select a LIVE runtime.

FastAPI V1.1 route functions currently return `Any`, so OpenAPI success
response schemas remain anonymous. Contract compatibility is therefore proven
by strict frontend Zod parsing of real FastAPI responses in `e2e:real`, not by
claiming that the generated OpenAPI document contains typed success models.

## Backend integration checklist

1. Confirm the final V1.1 endpoint paths and response schemas.
2. Preserve structured FastAPI errors: `detail.code`, `detail.message`, and optional `detail.fields`.
3. Supply server-generated discrete grids, budgets, best-point flags, outcomes, and Recovery/Apply stages.
4. Keep `FULL_CAMPAIGN` capability and runtime mode authoritative.
5. Run contract and browser tests in FAKE before any separately authorized LIVE lab trial.
