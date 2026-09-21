# VCSEL AI Tuning Dashboard V1.1

React + TypeScript control surface for AI Optimization, Sweep, Recovery, and historical result Apply. The browser never imports hardware code and never creates low-level I2C, BERT, DCA, SCPI, or `setFir` commands.

## Safety model / 安全模型

- `FAKE`: all device responses come from MSW and are labeled `SIMULATED`; no backend or physical instrument is contacted.
- `READ_ONLY`: monitoring is available; every mutation control is disabled.
- `LIVE`: controls call typed FastAPI endpoints only. The banner explicitly warns that authorized operations affect physical hardware.
- `RECOVERY_REQUIRED`: a global lock permits only the canonical Recovery workflow until B0 restoration is verified.
- Browser state is never treated as hardware truth. Requested, applied, readback, best-point, outcome, rollback, and Apply stages are displayed only after validated server responses.

## Routes / 页面

- `/` — V1.1 Overview, instrument state, current TDECQ/ER, active-task interlock, and history.
- `/tasks/new` — eight-step AI Optimization or Sweep wizard.
- `/runs/optimization/:campaignId` — Optimization phases, telemetry, FIR/DCA context, controls, B0, and events.
- `/runs/sweep/:campaignId` — Sweep telemetry, drift preflight, server pagination/filtering, and CSV/JSON exports.
- `/recovery/:campaignId` — exclusive baseline restore and verification workflow.
- `/campaigns/:campaignId` — retained V1 monitoring detail.
- `/campaigns/:campaignId/result` — version-aware result, rollback evidence, artifacts, and server-staged historical Apply.

## API boundary

Stable V1 reads and campaign actions are listed in `src/features/ai-tuning/api/client.ts`. V1.1 endpoints that still require backend confirmation are centralized in the exported `TEMPORARY_V11_ENDPOINTS` registry. UI components do not assemble API paths.

Temporary V1.1 groups:

| Group | Purpose |
| --- | --- |
| `/api/v1.1/tasks*` | Draft, snapshot, configuration scan, B0, range validation, Sweep count, authorization/start |
| `/api/v1.1/campaigns/{id}/sweep-*` | Server-owned Sweep pages and export |
| `/api/v1.1/campaigns/{id}/recovery*` | Recovery evidence, restore, continue, end |
| `/api/v1.1/campaigns/{id}/apply` | Eight-stage historical result Apply |

The backend may rename these temporary paths. Update only the registry/client, schemas, MSW handlers, and their tests when the final FastAPI contract is published.

## Local verification / 本地验证

Use the isolated frontend Worktree and FAKE runtime:

```powershell
cd C:\Users\NateS\Documents\VCSEL\.worktrees\v1.1-frontend\frontend
pnpm test
pnpm typecheck
pnpm lint
pnpm build
pnpm e2e
```

Playwright launches Vite in `test` mode using `.env.test` (`VITE_ENABLE_MOCKS=true`). This is the only approved browser-verification mode for this branch; it does not start FastAPI and cannot reach real hardware.

## Backend integration checklist

1. Confirm the final V1.1 endpoint paths and response schemas.
2. Preserve structured FastAPI errors: `detail.code`, `detail.message`, and optional `detail.fields`.
3. Supply server-generated discrete grids, budgets, best-point flags, outcomes, and Recovery/Apply stages.
4. Keep `FULL_CAMPAIGN` capability and runtime mode authoritative.
5. Run contract and browser tests in FAKE before any separately authorized LIVE lab trial.
