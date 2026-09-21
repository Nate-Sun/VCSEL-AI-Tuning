# VCSEL AI Tuning V1.1 Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete AI Tuning V1.1 FastAPI-controlled frontend with mock-backed missing contracts, runtime safety gates, guided task creation, Optimization/Sweep operations, Recovery, and history Apply.

**Architecture:** Keep all wire contracts and temporary endpoints in `contracts.ts`, `client.ts`, and `handlers.ts`. TanStack Query owns server state, a reducer owns unfinished wizard input, and validated SSE or HTTP responses are the only source of displayed hardware changes. Feature pages remain route-split and low-level components never call hardware or define wire fields.

**Tech Stack:** React 18, TypeScript strict, Vite, React Router, TanStack Query, Zod, Recharts, MSW, Vitest, React Testing Library, Playwright, ESLint.

---

## File Map

**Modify:**

- `frontend/package.json`: version `1.1.0` and verification scripts.
- `frontend/src/app/App.tsx`: V1.1 routes, dynamic banner, navigation, and route-level lazy loading.
- `frontend/src/features/ai-tuning/schemas/contracts.ts`: all V1/V1.1 runtime, task, Sweep, Recovery, Apply, and error schemas.
- `frontend/src/features/ai-tuning/api/client.ts`: generic GET/mutation transport and every real/temporary endpoint.
- `frontend/src/mocks/handlers.ts`: complete FAKE runtime scenarios.
- `frontend/src/mocks/browser.ts`: deterministic browser scenario selection.
- `frontend/src/styles/global.css`, `tokens.css`: approved high-fidelity Photon Lab Noir controls and responsive states.
- Existing V1 pages/tests: preserve read-only monitoring and version-aware history.

**Create:**

- `frontend/src/features/ai-tuning/store/wizardReducer.ts`: wizard-only state and transitions.
- `frontend/src/features/ai-tuning/hooks/useRuntimeGate.ts`: centralized mutation permission derivation.
- `frontend/src/features/ai-tuning/hooks/useV11Queries.ts`: query keys and typed query/mutation hooks.
- `frontend/src/features/ai-tuning/components/RuntimeModeBanner.tsx`.
- `frontend/src/features/ai-tuning/components/MutationButton.tsx`.
- `frontend/src/features/ai-tuning/components/wizard/*.tsx`: one focused component per step.
- `frontend/src/features/ai-tuning/components/run/*.tsx`: shared run metrics, controls, drift, and verification panels.
- `frontend/src/features/ai-tuning/components/SweepResultTable.tsx`.
- `frontend/src/features/ai-tuning/components/HistoryResultApplyButton.tsx`.
- `frontend/src/features/ai-tuning/pages/CreateTaskPage.tsx`.
- `frontend/src/features/ai-tuning/pages/OptimizationRunPage.tsx`.
- `frontend/src/features/ai-tuning/pages/SweepRunPage.tsx`.
- `frontend/src/features/ai-tuning/pages/RecoveryPage.tsx`.
- `frontend/src/features/ai-tuning/pages/HistoryResultPage.tsx`.
- Focused tests under `frontend/src/features/ai-tuning/tests/` and FAKE Playwright flows under `frontend/e2e/`.

### Task 1: Version, Runtime, and V1.1 Contract Foundation

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/features/ai-tuning/schemas/contracts.ts`
- Test: `frontend/src/features/ai-tuning/tests/contracts.test.ts`
- Create: `frontend/src/features/ai-tuning/tests/v11-contracts.test.ts`

- [ ] **Step 1: Write failing contract tests**

```ts
expect(runtimeModeSchema.parse("LIVE_TUNING")).toBe("LIVE");
expect(taskTypeSchema.parse("AI_OPTIMIZATION")).toBe("AI_OPTIMIZATION");
expect(taskTypeSchema.parse("SWEEP")).toBe("SWEEP");
expect(parameterNameSchema.safeParse("eq").success).toBe(true);
expect(parameterNameSchema.safeParse("testch").success).toBe(false);
expect(discreteGridSchema.parse({ values: [0, 3, 6, 9] }).values).not.toContain(10);
expect(apiErrorSchema.parse({ detail: { code: "campaign_conflict", message: "active" } }).detail.code).toBe("campaign_conflict");
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/v11-contracts.test.ts`

Expected: FAIL because the V1.1 schemas are not exported.

- [ ] **Step 3: Add strict schemas and version metadata**

Implement exact enums and objects in `contracts.ts`:

```ts
export const runtimeModeSchema = z
  .enum(["FAKE", "READ_ONLY", "LIVE_TUNING", "LIVE"])
  .transform((mode) => mode === "LIVE_TUNING" ? "LIVE" as const : mode);
export const taskTypeSchema = z.enum(["AI_OPTIMIZATION", "SWEEP"]);
export const campaignTypeSchema = taskTypeSchema;
export const safeParameterNames = [
  "ibias_dac", "vref", "vbias_ref", "vga_cur", "res_gain",
  "res_bw", "gain_ctrl", "efin_cur", "eq",
] as const;
export const parameterNameSchema = z.enum(safeParameterNames);
export const versionMetadataSchema = z.object({
  product_version: z.string().min(1),
  schema_version: z.string().min(1),
  runtime_mode: runtimeModeSchema,
  campaign_type: campaignTypeSchema,
});
export const apiErrorSchema = z.object({
  detail: z.object({ code: z.string(), message: z.string(), fields: z.record(z.string()).optional() }),
});
```

Add snapshot, scan result, range validation, profile, Sweep count/point/page, Recovery, history Apply stage, and V1.1 Campaign/Report schemas. Every numeric metric uses finite-number validation and every object uses `.strict()` unless the existing backend requires `.passthrough()`.

Set `frontend/package.json` version to `1.1.0`.

- [ ] **Step 4: Run contract tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/contracts.test.ts src/features/ai-tuning/tests/v11-contracts.test.ts`

Expected: both files pass; unsafe parameters and non-finite metrics are rejected.

### Task 2: Typed API Transport and Temporary Endpoint Registry

**Files:**
- Modify: `frontend/src/features/ai-tuning/api/client.ts`
- Create: `frontend/src/features/ai-tuning/tests/v11-api.test.ts`

- [ ] **Step 1: Write failing transport tests**

```ts
await api.pauseCampaign("campaign-1");
expect(requests.at(-1)).toMatchObject({ method: "POST", url: "/api/campaigns/campaign-1/pause" });
await expect(api.createTask({ task_type: "SWEEP" })).resolves.toMatchObject({ task_type: "SWEEP" });
expect(JSON.stringify(requests)).not.toMatch(/SCPI|register_address|setFir|raw_command/);
```

Also assert `409`, `422`, and `503` responses become `ApiResponseError` with `status`, `code`, `message`, and `fields`.

- [ ] **Step 2: Run the API test RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/v11-api.test.ts`

Expected: FAIL because mutations and normalized errors are absent.

- [ ] **Step 3: Implement one validated request helper**

```ts
async function requestJson<T>(
  method: "GET" | "POST",
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  body?: unknown,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { Accept: "application/json", ...(body ? { "Content-Type": "application/json" } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const payload: unknown = await response.json();
  if (!response.ok) throw ApiResponseError.fromResponse(response.status, payload);
  return schema.parse(payload);
}
```

Expose real V1 endpoints for health, capabilities, campaigns, observations, best, report, SSE, pause, resume, and abort. Register missing V1.1 paths in an exported `TEMPORARY_V11_ENDPOINTS` constant for create Draft, snapshot, scan, save B0, validate ranges, Sweep count, authorize/start, Recovery, Sweep page/export, and history Apply. Component code must never assemble paths.

- [ ] **Step 4: Run the API tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/api.test.ts src/features/ai-tuning/tests/v11-api.test.ts`

Expected: all requests use only typed GET/POST endpoints and errors retain backend codes.

### Task 3: FAKE Runtime Scenarios and Query Hooks

**Files:**
- Modify: `frontend/src/mocks/handlers.ts`
- Modify: `frontend/src/mocks/browser.ts`
- Create: `frontend/src/features/ai-tuning/hooks/useV11Queries.ts`
- Create: `frontend/src/features/ai-tuning/tests/v11-hooks.test.tsx`

- [ ] **Step 1: Write failing hook tests**

```tsx
const { result } = renderHook(() => usePauseCampaign("campaign-live"), { wrapper });
result.current.mutate();
expect(result.current.isPending).toBe(true);
await waitFor(() => expect(result.current.isSuccess).toBe(true));
expect(queryClient.getQueryState(["campaign", "campaign-live"])?.isInvalidated).toBe(true);
```

Add scenarios for FAKE Optimization, Sweep, READ_ONLY, LIVE capability denied, Recovery, drift, API unavailable, and history Apply rollback.

- [ ] **Step 2: Run hook tests RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/v11-hooks.test.tsx`

Expected: FAIL because hooks and MSW handlers are missing.

- [ ] **Step 3: Implement centralized query keys and mutation invalidation**

```ts
export const v11Keys = {
  active: ["v11", "active"] as const,
  task: (id: string) => ["v11", "task", id] as const,
  sweep: (id: string, query: SweepQuery) => ["v11", "sweep", id, query] as const,
  recovery: (id: string) => ["v11", "recovery", id] as const,
};
```

Mutations invalidate canonical queries only after success. Do not call `setQueryData` with assumed hardware values. MSW returns `simulated: true` for fake hardware snapshots and supplies every temporary contract from the three allowed contract files.

- [ ] **Step 4: Run hook tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/v11-hooks.test.tsx`

Expected: pending/success/error behavior and invalidation pass without optimistic state.

### Task 4: Dynamic Runtime Banner and Global Mutation Gate

**Files:**
- Create: `frontend/src/features/ai-tuning/hooks/useRuntimeGate.ts`
- Create: `frontend/src/features/ai-tuning/components/RuntimeModeBanner.tsx`
- Create: `frontend/src/features/ai-tuning/components/MutationButton.tsx`
- Modify: `frontend/src/app/App.tsx`
- Test: `frontend/src/features/ai-tuning/tests/runtime-gate.test.tsx`

- [ ] **Step 1: Write failing gate tests**

```ts
expect(deriveRuntimeGate(fakeCapabilities, null).canMutate).toBe(true);
expect(deriveRuntimeGate(readOnlyCapabilities, null).canMutate).toBe(false);
expect(deriveRuntimeGate(liveWithoutFullCampaign, null).canStart).toBe(false);
expect(deriveRuntimeGate(liveCapabilities, recoveryCampaign).recoveryOnly).toBe(true);
expect(deriveRuntimeGate(fakeCapabilities, runningCampaign).canCreateDraft).toBe(false);
```

- [ ] **Step 2: Run gate tests RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/runtime-gate.test.tsx`

Expected: FAIL because the gate and dynamic banner do not exist.

- [ ] **Step 3: Implement pure gate derivation and semantic controls**

```ts
export function deriveRuntimeGate(capability: Capability, active: Campaign | null): RuntimeGate {
  const recoveryOnly = active?.state === "RECOVERY_REQUIRED";
  const canMutate = capability.runtime_mode !== "READ_ONLY" && !recoveryOnly;
  return {
    canMutate,
    recoveryOnly,
    canCreateDraft: canMutate && active?.state !== "RUNNING" && active?.state !== "PAUSED",
    canStart: canMutate && capability.FULL_CAMPAIGN === true,
  };
}
```

`RuntimeModeBanner` renders FAKE simulation disclosure, LIVE real-hardware warning, READ_ONLY status, or global Recovery lock. `MutationButton` uses a native button with visible reason text and pending lock. Replace `ReadOnlyBanner` in `App.tsx` and add lazy routes for V1.1 pages.

- [ ] **Step 4: Run gate and shell tests GREEN**

Run: `cd frontend; pnpm test -- --run src/app/App.test.tsx src/features/ai-tuning/tests/runtime-gate.test.tsx`

Expected: runtime text, disabled reasons, Recovery navigation, and no READ_ONLY mutation controls pass.

### Task 5: V1.1 Overview and Eight-Step Wizard Core

**Files:**
- Modify: `frontend/src/features/ai-tuning/pages/OverviewPage.tsx`
- Create: `frontend/src/features/ai-tuning/store/wizardReducer.ts`
- Create: `frontend/src/features/ai-tuning/pages/CreateTaskPage.tsx`
- Create: `frontend/src/features/ai-tuning/components/wizard/TaskTypeSelector.tsx`
- Create: `frontend/src/features/ai-tuning/components/wizard/HardwareMappingStep.tsx`
- Create: `frontend/src/features/ai-tuning/tests/wizard-core.test.tsx`

- [ ] **Step 1: Write failing overview and reducer tests**

```ts
expect(wizardReducer(initialWizardState, { type: "SELECT_TASK_TYPE", value: "SWEEP" }).taskType).toBe("SWEEP");
expect(canContinueMapping({ i2c_channel: "2", bert_channel: "1", dca_source: "CH1A", same_link_confirmed: false })).toBe(false);
expect(screen.getByText("AI Tuning V1.1")).toBeVisible();
expect(screen.getByRole("link", { name: /View current task/i })).toBeVisible();
expect(screen.queryByRole("link", { name: /Create task/i })).not.toBeInTheDocument();
```

- [ ] **Step 2: Run wizard core tests RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/wizard-core.test.tsx`

Expected: FAIL because V1.1 overview and wizard reducer are missing.

- [ ] **Step 3: Implement Overview and steps 1-2**

The overview renders current TDECQ/ER only from the active snapshot, DCA/I2C/BERT state, runtime, Recovery, active/recent tasks, update time, and two equal launch cards. The reducer keeps the eight-step state locally and clears Sweep-only fields when task type changes to Optimization.

Hardware Mapping uses selects supplied by FastAPI and requires this checkbox before Continue:

```tsx
<label>
  <input type="checkbox" checked={mapping.same_link_confirmed} onChange={onConfirm} />
  以上三个通道属于同一条物理链路 / These three channels belong to the same physical link.
</label>
```

No automatic identify/change button exists.

- [ ] **Step 4: Run wizard core tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/wizard-core.test.tsx`

Expected: task type, active-task gate, mapping confirmation, and keyboard navigation pass.

### Task 6: Snapshot, B0, Script/RegMap, Ranges, Profiles, and Sweep Configuration

**Files:**
- Create: `frontend/src/features/ai-tuning/components/wizard/HardwareSnapshotStep.tsx`
- Create: `frontend/src/features/ai-tuning/components/wizard/BaselineStep.tsx`
- Create: `frontend/src/features/ai-tuning/components/wizard/ScriptRegMapStep.tsx`
- Create: `frontend/src/features/ai-tuning/components/wizard/ParameterRangeEditor.tsx`
- Create: `frontend/src/features/ai-tuning/components/wizard/ProfileSelector.tsx`
- Create: `frontend/src/features/ai-tuning/components/wizard/SweepConfiguration.tsx`
- Create: `frontend/src/features/ai-tuning/tests/wizard-configuration.test.tsx`

- [ ] **Step 1: Write failing configuration tests**

Cover snapshot fields, seven labeled FIR taps, locked NLC, save B0 pending state, safe allowlist, unavailable reason, absence of manual register input, grid `0,3,6,9`, B0-outside warning, server budgets, conditional Sweep step, mutually exclusive hardware, Single/Full, seed/counts, locked NLC, and absence of estimated time.

```tsx
expect(screen.getAllByRole("spinbutton")).toHaveLength(3);
expect(screen.getByText("0, 3, 6, 9")).toBeVisible();
expect(screen.getByText(/B0 is outside the selected range/i)).toHaveClass("notice-warning");
expect(screen.queryByLabelText(/register address/i)).not.toBeInTheDocument();
```

- [ ] **Step 2: Run configuration tests RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/wizard-configuration.test.tsx`

Expected: FAIL because steps 3-7 are absent.

- [ ] **Step 3: Implement steps 3-7 using server responses**

Snapshot tables render all required identities and configuration fields. The fixed historical FIR is labeled Reference, never B0. Save B0 waits for the server response.

Script/RegMap uses file/select controls and renders only parsed `parameterNameSchema` values. Unavailable entries remain visible but disabled with the server reason.

Ranges send integer min/max/step to `validateParameterRanges`; candidate values are rendered from its response. The browser does not calculate or append endpoints.

Profile cards render backend `budget` and `max_alternating_rounds`. Sweep hardware uses a radio group and Full mode renders seed, max points, original/legal/skipped counts and limit errors.

- [ ] **Step 4: Run configuration tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/wizard-configuration.test.tsx`

Expected: all step conditions, safety restrictions, and server-owned values pass.

### Task 7: FULL_CAMPAIGN Authorization and Start

**Files:**
- Create: `frontend/src/features/ai-tuning/components/wizard/AuthorizationSummary.tsx`
- Modify: `frontend/src/features/ai-tuning/pages/CreateTaskPage.tsx`
- Create: `frontend/src/features/ai-tuning/tests/authorization-start.test.tsx`

- [ ] **Step 1: Write failing authorization tests**

```tsx
await user.click(screen.getByRole("button", { name: /Authorize and Start/i }));
expect(authorizeRequests).toHaveLength(1);
expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
expect(screen.getByRole("button", { name: /Authorizing/i })).toBeDisabled();
```

Assert the summary contains Task Type, B0, mapping, DCA, ranges, steps, profile, budget, conditional seed/max points, rollback target, and authorization ID. READ_ONLY, active-task, missing capability, validation, and conflict paths must disable or retain the step.

- [ ] **Step 2: Run authorization tests RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/authorization-start.test.tsx`

Expected: FAIL because the final step is absent.

- [ ] **Step 3: Implement one-click authorization/start**

Use a single `MutationButton`; call `authorizeAndStart` once, wait for its response, then navigate by the server-returned Campaign Type. Do not infer hardware success or use optimistic query updates.

- [ ] **Step 4: Run authorization tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/authorization-start.test.tsx`

Expected: one request, pending lock, version metadata, error states, and type-specific navigation pass.

### Task 8: Optimization and Sweep Live Operations

**Files:**
- Create: `frontend/src/features/ai-tuning/components/run/RunControls.tsx`
- Create: `frontend/src/features/ai-tuning/components/run/FinalVerificationPanel.tsx`
- Create: `frontend/src/features/ai-tuning/components/run/DcaDriftPanel.tsx`
- Create: `frontend/src/features/ai-tuning/pages/OptimizationRunPage.tsx`
- Create: `frontend/src/features/ai-tuning/pages/SweepRunPage.tsx`
- Create: `frontend/src/features/ai-tuning/components/SweepResultTable.tsx`
- Create: `frontend/src/features/ai-tuning/tests/live-operations.test.tsx`

- [ ] **Step 1: Write failing live-operation tests**

Assert Optimization renders coarse/fine/FIR/retune, round/max, first/iterative, current/requested/applied/readback, B0 Main, locked NLC, metrics/best, FIR `0/3`, candidate budget, DCA taps, formula/no-Bayesian/no-damping disclosure, rollback, convergence, and no candidate accept button.

Assert Sweep renders I2C/FIR, Single/Full, seed/counts/indexes, current point, readback, `INVALID_TDECQ`, `SKIPPED_CONSTRAINT`, errors, device state, Pause B0 message, Resume Preflight/B0/remaining, pagination/filter/sort, exports, backend `is_best`, and restored B0.

- [ ] **Step 2: Run live tests RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/live-operations.test.tsx`

Expected: FAIL because operation pages are absent.

- [ ] **Step 3: Implement server-confirmed run pages**

`RunControls` derives enabled actions from runtime/campaign state, calls typed mutations, and displays pending/error without changing hardware values. `DcaDriftPanel` disables Resume until `preflight_allowed === true`. `SweepResultTable` passes page/filter/sort to the server and highlights only `point.is_best === true`.

Reuse validated `EventLog`, `ParameterDiffTable`, and event-stream hook. On event gaps, invalidate Campaign, observations, run summary, and Sweep page queries.

- [ ] **Step 4: Run live and SSE tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/live-operations.test.tsx src/features/ai-tuning/tests/eventStream.test.ts`

Expected: live state, controls, drift, results, and reconnect behavior pass.

### Task 9: Recovery, Results, and Historical Apply

**Files:**
- Create: `frontend/src/features/ai-tuning/pages/RecoveryPage.tsx`
- Create: `frontend/src/features/ai-tuning/pages/HistoryResultPage.tsx`
- Create: `frontend/src/features/ai-tuning/components/HistoryResultApplyButton.tsx`
- Modify: `frontend/src/features/ai-tuning/pages/CampaignResultPage.tsx`
- Create: `frontend/src/features/ai-tuning/tests/recovery-history-results.test.tsx`

- [ ] **Step 1: Write failing Recovery/Apply/result tests**

Assert Recovery evidence and exclusive mutation; successful restore enabling Continue/End; Optimization raw-three/median/final/outcomes; Sweep status/best/exports/restored; V1 version preserved; and one-click history Apply with eight server stages, no modal, pending lock, success, rollback failure, and Recovery routing.

```tsx
await user.click(screen.getByRole("button", { name: /Apply this result/i }));
expect(applyRequests).toBe(1);
expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
expect(await screen.findByText("HISTORY_RESULT_APPLIED")).toBeVisible();
```

- [ ] **Step 2: Run tests RED**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/recovery-history-results.test.tsx`

Expected: FAIL because Recovery and Apply pages are absent.

- [ ] **Step 3: Implement canonical Recovery and results**

Recovery reads all evidence from `recoverySchema`. Until `restore_verified`, hide or disable Continue/End and all other mutation components through the global gate. Historical Apply renders stages from the server response/event; on `RECOVERY_REQUIRED`, navigate to Recovery.

Result page switches on `campaign_type`, renders actual `product_version` and `schema_version`, and never relabels V1 history. `FAILED_INVALID_DCA_TAPS` never offers Apply.

- [ ] **Step 4: Run tests GREEN**

Run: `cd frontend; pnpm test -- --run src/features/ai-tuning/tests/recovery-history-results.test.tsx`

Expected: Recovery lock, final verification, version preservation, exports, and Apply flow pass.

### Task 10: Styling, Accessibility, Browser Flows, Documentation, and Final Commit

**Files:**
- Modify: `frontend/src/styles/tokens.css`
- Modify: `frontend/src/styles/global.css`
- Modify: `frontend/e2e/dashboard.spec.ts`
- Create: `frontend/e2e/v11-control.spec.ts`
- Modify: `frontend/README.md`
- Update: `frontend/artifacts/screenshots/*.png`

- [ ] **Step 1: Add failing Playwright expectations**

Cover FAKE overview → wizard → mapping confirmation → snapshot/B0 → allowlist/ranges → authorization → Optimization/Sweep SSE → pause/resume/abort → drift → Recovery → result/export → one-click history Apply. Track every request and fail if it targets a non-allowlisted endpoint, contains raw hardware terms, or uses a non-GET/POST method outside the typed registry.

- [ ] **Step 2: Run Playwright RED**

Run: `cd frontend; pnpm exec playwright test e2e/v11-control.spec.ts`

Expected: FAIL until routes and final responsive styles are complete.

- [ ] **Step 3: Apply approved high-fidelity responsive styling**

Use the approved Photon Lab Noir hierarchy: dynamic mode strip, instrument health row, large optical metrics, equal Optimization/Sweep launch cards, active-task interlock, eight-step wizard, compact run telemetry, and red Recovery lock. Preserve native semantic controls, visible focus, status text/icons, loading/empty/error/disabled/stale states, desktop and 360 px layouts.

Update README with V1.1 routes, FAKE-only browser verification, real/pending endpoint table, and explicit no-direct-hardware boundary.

- [ ] **Step 4: Run every acceptance command**

```powershell
cd frontend
pnpm test -- --run
pnpm run typecheck
pnpm run lint
pnpm run build
pnpm exec playwright test
```

Expected: all commands exit 0; browser console and Vite overlay contain no errors; desktop and 360 px screenshots exist.

- [ ] **Step 5: Verify scope and create frontend-only commits**

Run:

```powershell
git diff --check
git status --short
git diff --name-only -- src/vcsel_ai_tuning
```

Expected: no whitespace errors, no backend/hardware diff, and only approved frontend/design/plan files staged. Create focused frontend commits and report the final branch and commit hash; do not push, merge, modify V1 Tag, or touch the V1 test directory.
