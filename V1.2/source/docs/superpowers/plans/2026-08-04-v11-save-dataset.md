# V1.1 Optional Dataset Saving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, default-off `save_dataset` choice to V1.1 Sweep and AI Optimization creation and render the server-owned dataset outcome and files independently from measurement success.

**Architecture:** Extend the existing task document and request schemas, keep the choice in the wizard reducer, and serialize it through the existing request builder. A focused `DatasetStatusPanel` consumes only parsed task data and is shared by Sweep, Optimization, and final-report pages; MSW supplies fixed same-origin mock artifacts until the backend contract lands.

**Tech Stack:** React 18, TypeScript 5.7, Zod 3, TanStack Query, MSW, Vitest, Testing Library, Playwright, Vite.

---

### Task 1: Define and verify the dataset API contract

**Files:**
- Modify: `frontend/src/features/ai-tuning/schemas/contracts.ts`
- Modify: `frontend/src/features/ai-tuning/tests/v11-contracts.test.ts`
- Modify: `frontend/src/features/ai-tuning/tests/v11-api.test.ts`

- [ ] **Step 1: Write failing contract tests**

Add a task document fixture with `config.save_dataset`, `measurement_status`, `dataset_status`, `dataset_error`, and typed PNG/NPY/JSON files. Assert that Zod preserves all fields and rejects an unsafe external URL:

```ts
const parsed = v11TaskDocumentSchema.parse({
  ...taskDocument,
  config: { ...taskDocument.config, save_dataset: true },
  measurement_status: "SUCCESS",
  dataset_status: "PARTIAL",
  dataset_error: "equalized eye unavailable",
  dataset_files: [{
    file_id: "point-1-density",
    kind: "EYE_DENSITY_NPY",
    name: "density.npy",
    url: "/api/v1.1/tasks/task-1/datasets/point-1/density.npy",
    size_bytes: 2048,
    point_id: "point-1",
    observation_id: null,
  }],
});
expect(parsed.config.save_dataset).toBe(true);
expect(parsed.dataset_files[0].kind).toBe("EYE_DENSITY_NPY");
```

- [ ] **Step 2: Run the focused contract tests and verify RED**

Run: `npm.cmd test -- src/features/ai-tuning/tests/v11-contracts.test.ts src/features/ai-tuning/tests/v11-api.test.ts`

Expected: failure because dataset fields are stripped or absent from the inferred type.

- [ ] **Step 3: Add minimal schemas and request types**

Move the existing `safeArtifactUrlSchema` definition above the V1.1 task schemas, define exact enums and file metadata, then extend the task config/document:

```ts
export const v11MeasurementStatusSchema = z.enum(["PENDING", "RUNNING", "SUCCESS", "FAILED"]);
export const v11DatasetStatusSchema = z.enum(["DISABLED", "PENDING", "RUNNING", "SUCCESS", "PARTIAL", "FAILED"]);
export const v11DatasetFileSchema = z.object({
  file_id: z.string().min(1),
  kind: z.enum(["EYE_DENSITY_NPY", "RAW_EYE_PNG", "TDECQ_EQUALIZED_EYE_PNG", "MEASUREMENT_RECORD_JSON"]),
  name: z.string().min(1),
  url: safeArtifactUrlSchema,
  size_bytes: z.number().int().nonnegative().nullable().optional(),
  point_id: z.string().min(1).nullable().optional(),
  observation_id: z.string().min(1).nullable().optional(),
});
```

Add `save_dataset: z.boolean()` to `v11TaskConfigSchema`, task-level status/file fields to `v11TaskDocumentSchema`, `save_dataset: boolean` to `V11CreateTaskRequest`, and inferred exported types.

- [ ] **Step 4: Update API fixtures and prove transport preservation**

Add `save_dataset: false` to the shared `createRequest` in `v11-api.test.ts`, return task dataset fields from `taskDocument`, and assert the captured `/api/v1.1/tasks` body contains the boolean.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `npm.cmd test -- src/features/ai-tuning/tests/v11-contracts.test.ts src/features/ai-tuning/tests/v11-api.test.ts`

Expected: all focused contract/API tests pass.

### Task 2: Add the immutable creation-wizard choice

**Files:**
- Modify: `frontend/src/features/ai-tuning/store/wizardReducer.ts`
- Modify: `frontend/src/features/ai-tuning/store/wizardWorkflow.ts`
- Modify: `frontend/src/features/ai-tuning/pages/CreateTaskPage.tsx`
- Modify: `frontend/src/features/ai-tuning/tests/wizard-core.test.tsx`
- Modify: `frontend/src/styles/global.css`

- [ ] **Step 1: Write failing reducer and request-builder tests**

Assert the initial state is off and both task types serialize an explicit boolean:

```ts
expect(initialWizardState.saveDataset).toBe(false);
expect(buildV11CreateTaskRequest({ ...options, taskType: "AI_OPTIMIZATION", saveDataset: false })).toMatchObject({ save_dataset: false });
expect(buildV11CreateTaskRequest({ ...options, taskType: "SWEEP", saveDataset: true })).toMatchObject({ save_dataset: true });
```

- [ ] **Step 2: Write a failing wizard interaction test**

Render `/tasks/new?type=SWEEP`, verify the switch is off and the disabled explanation is visible, toggle it, advance through the existing MSW-backed workflow, and assert the POST body contains `save_dataset: true` and the review displays `ENABLED · LOCKED`.

- [ ] **Step 3: Run the wizard tests and verify RED**

Run: `npm.cmd test -- src/features/ai-tuning/tests/wizard-core.test.tsx`

Expected: failure because `saveDataset`, its action, copy, and request field do not exist.

- [ ] **Step 4: Implement reducer and workflow plumbing**

Add `saveDataset: boolean`, default it to `false`, add `SET_SAVE_DATASET`, accept `saveDataset` in `BuildTaskOptions`, and include `save_dataset: options.saveDataset` in the common task request.

- [ ] **Step 5: Implement the bilingual switch and locked review**

In the shared Profile step render a controlled checkbox:

```tsx
<label className={state.saveDataset ? "dataset-toggle enabled" : "dataset-toggle"}>
  <input
    type="checkbox"
    checked={state.saveDataset}
    disabled={Boolean(state.taskId)}
    onChange={(event) => dispatch({ type: "SET_SAVE_DATASET", value: event.target.checked })}
  />
  <span><b>保存实验数据 / Save dataset</b><small>{state.saveDataset ? enabledCopy : disabledCopy}</small></span>
</label>
```

Pass `state.saveDataset` into the request builder and add a review row that includes the server-created lock state.

- [ ] **Step 6: Add focused styles and verify GREEN**

Add industrial control-surface styling for `.dataset-toggle`, `.dataset-toggle.enabled`, and its disabled state without changing the existing visual language. Re-run the wizard test and expect it to pass.

### Task 3: Render measurement and dataset outcomes safely

**Files:**
- Create: `frontend/src/features/ai-tuning/components/DatasetStatusPanel.tsx`
- Modify: `frontend/src/features/ai-tuning/pages/OptimizationRunPage.tsx`
- Modify: `frontend/src/features/ai-tuning/pages/SweepRunPage.tsx`
- Modify: `frontend/src/features/ai-tuning/pages/V11TaskResultPage.tsx`
- Modify: `frontend/src/features/ai-tuning/tests/v11-run-pages.test.tsx`
- Modify: `frontend/src/styles/global.css`

- [ ] **Step 1: Write failing page tests for independent statuses**

Cover these cases with MSW task responses:

```ts
expect(await screen.findByText("Measurement: SUCCESS")).toBeInTheDocument();
expect(screen.getByText("Dataset saving: DISABLED")).toBeInTheDocument();
expect(screen.queryByAltText(/eye/i)).not.toBeInTheDocument();
```

For `PARTIAL`, keep the success measurement text and assert a separate warning plus `dataset_error`. For `SUCCESS`, assert raw/TDECQ PNG images have view links and NPY/JSON files have download links.

- [ ] **Step 2: Run run-page tests and verify RED**

Run: `npm.cmd test -- src/features/ai-tuning/tests/v11-run-pages.test.tsx`

Expected: failure because no dataset panel exists.

- [ ] **Step 3: Implement `DatasetStatusPanel`**

Accept only parsed task fields:

```ts
interface DatasetStatusPanelProps {
  saveDataset: boolean;
  measurementStatus: V11MeasurementStatus;
  datasetStatus: V11DatasetStatus;
  datasetError: string | null;
  files: V11DatasetFile[];
}
```

Render DISABLED as a neutral compact panel with no media region. Render PARTIAL/FAILED with `.dataset-warning` while leaving `.measurement-success` intact. Use `<img loading="lazy">` only for PNG kinds; add `target="_blank" rel="noreferrer"` view links and `download` for NPY/JSON.

- [ ] **Step 4: Integrate the panel on all V1.1 detail surfaces**

Pass `task.data.config.save_dataset` and task fields from `OptimizationRunPage` and `SweepRunPage`. Pass the same fields from `report.data` in `V11TaskResultPage`.

- [ ] **Step 5: Add responsive file-card styling and verify GREEN**

Add `.dataset-panel`, `.dataset-status-grid`, `.dataset-file-grid`, `.dataset-file`, `.dataset-thumbnail`, and warning/success modifiers. Re-run `v11-run-pages.test.tsx` and expect all tests to pass.

### Task 4: Align fixed mocks and preserve B0 truthfulness

**Files:**
- Modify: `frontend/src/mocks/handlers.ts`
- Create: `frontend/src/features/ai-tuning/fixtures/v11Dataset.ts`
- Modify: `frontend/src/features/ai-tuning/tests/wizard-core.test.tsx`
- Modify: `frontend/src/features/ai-tuning/tests/v11-run-pages.test.tsx`

- [ ] **Step 1: Add a fixed dataset fixture**

Create a fixture with FAKE-runtime PNG, NPY, and JSON same-origin URLs. Do not add actual eye data or claim hardware provenance.

- [ ] **Step 2: Make the mock task echo immutable configuration**

Set `defaultV11Config.save_dataset = false`; store the create request unchanged; have `v11Task()` return dataset SUCCESS/files only when the stored configuration enables saving, otherwise return DISABLED/empty files.

- [ ] **Step 3: Add a B0 regression assertion**

In the wizard page test, assert the rendered actual FIR values originate from `snapshot.b0.fir.taps` and that the reference line still contains `Reference only (not B0)`.

- [ ] **Step 4: Run the complete frontend unit suite**

Run: `npm.cmd test`

Expected: all frontend unit and component tests pass with no failed files.

### Task 5: Production and browser verification

**Files:**
- Modify only if a verification failure reveals a feature defect in the files listed above.

- [ ] **Step 1: Run static checks**

Run: `npm.cmd run lint`

Expected: zero warnings and zero errors.

Run: `npm.cmd run build`

Expected: TypeScript and Vite production build succeed.

- [ ] **Step 2: Run mock-only browser tests**

Run: `npm.cmd run e2e`

Expected: Playwright uses Vite/MSW only and all browser tests pass; do not run `e2e:real`.

- [ ] **Step 3: Check scope and safety**

Run: `git diff --check`

Run: `git diff --name-only -- src/vcsel_ai_tuning/hardware`

Expected: no whitespace errors and no hardware-module changes.

Review the staged diff and confirm it contains no FIR convolution, SHA-256 generation, DCA data processing, hardware commands, `setFir`, Kimi, or DINOv2 integration. The frontend may only send the approved high-level boolean and display parsed server evidence.

Run: `git -C C:\Users\NateS\Documents\VCSEL\.worktrees\v1.1-integration status --short`

Expected: the integration Worktree remains at its pre-existing status with only its untracked `artifacts/`.

- [ ] **Step 4: Commit the completed feature**

```powershell
git add -- frontend/src docs/superpowers/plans/2026-08-04-v11-save-dataset.md
git commit -m "feat(ai-tuning): add optional dataset saving UI"
```
