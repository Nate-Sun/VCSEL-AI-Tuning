# VCSEL AI Tuning V1.1 Optional Dataset Saving Design

## Goal

Add an optional, immutable-per-task dataset-saving choice to the V1.1 Sweep and AI Optimization creation flow, then display server-reported measurement and dataset outcomes without performing hardware control or data processing in the browser.

## Baseline and isolation

- Base commit: `e7ecc060403819c2fd7b10be742bf569e9ff8b18`.
- Feature branch: `codex/v1.1-save-dataset`.
- Feature worktree: `C:\Users\NateS\Documents\VCSEL\.worktrees\v1.1-save-dataset`.
- `C:\Users\NateS\Documents\VCSEL\.worktrees\v1.1-integration` remains untouched.
- No frontend code may import hardware modules, issue hardware commands, execute `setFir`, or implement DCA/data algorithms.

## Chosen contract approach

The V1.1 task document carries dataset state at the task level. This fits the existing task polling flow and avoids a second endpoint before the backend contract is available.

The create request always includes:

```json
{
  "save_dataset": false
}
```

The task configuration echoes the immutable choice as `config.save_dataset`.

The task document and final report include:

```json
{
  "measurement_status": "SUCCESS",
  "dataset_status": "DISABLED",
  "dataset_error": null,
  "dataset_files": []
}
```

Supported measurement states are `PENDING`, `RUNNING`, `SUCCESS`, and `FAILED`. Supported dataset states are `DISABLED`, `PENDING`, `RUNNING`, `SUCCESS`, `PARTIAL`, and `FAILED`.

Each dataset file has this shape:

```json
{
  "file_id": "point-0001-raw-eye",
  "kind": "RAW_EYE_PNG",
  "name": "raw_eye.png",
  "url": "/api/v1.1/tasks/task-1/datasets/point-0001/raw_eye.png",
  "size_bytes": 183204,
  "point_id": "point-0001",
  "observation_id": null
}
```

`kind` is one of `EYE_DENSITY_NPY`, `RAW_EYE_PNG`, `TDECQ_EQUALIZED_EYE_PNG`, and `MEASUREMENT_RECORD_JSON`. `size_bytes`, `point_id`, and `observation_id` may be null. URLs must pass the existing safe same-origin artifact URL rules before rendering.

Alternatives considered were a nested `dataset` object and a separate dataset endpoint. The nested form adds migration work without a current backend benefit; a separate endpoint adds polling and error-state complexity. The task-level form is the smallest compatible extension.

## Creation flow

The shared V1.1 creation wizard adds a bilingual `保存实验数据 / Save dataset` switch in the common Profile step, so it applies to both Sweep and AI Optimization.

The default is off. While off, the UI states that density matrices and PNGs are not saved, no extra DCA dataset acquisition is performed, and minimum safety/recovery records remain. While on, it states that raw density matrices, raw eye PNGs, TDECQ-equalized eye PNGs, and full measurement records are saved, with possible Round Time and disk-use impact.

The state reducer owns `saveDataset`, initialized to `false`. The request builder always serializes it as `save_dataset`. Once the server creates the task, the switch is disabled because the task configuration is immutable. The authorization review shows `ENABLED · LOCKED` or `DISABLED · LOCKED`.

## Dataset result presentation

A reusable `DatasetStatusPanel` is rendered on both live task pages and the V1.1 final report.

- Measurement and dataset states have separate visual rows.
- `measurement_status: SUCCESS` remains successful when dataset saving is `PARTIAL` or `FAILED`.
- `dataset_status: PARTIAL` and `FAILED` display a dataset-only warning with `dataset_error` when supplied.
- `dataset_status: DISABLED` displays `Dataset saving: DISABLED` with neutral styling, renders no missing-eye placeholder, and is never treated as a fault.
- Enabled tasks with no ready files show a compact pending/empty explanation rather than fabricated hardware evidence.
- PNG entries render a server-backed thumbnail plus a view link.
- NPY and `record.json` entries render download links.
- The browser does not decode NPY, calculate hashes, convolve FIR taps, or process DCA data.

## B0 integrity

The current B0 panel continues to render `snapshot.b0.fir.taps`, which is the BERT readback FIR returned by the V1.1 snapshot endpoint. `REFERENCE_FIR` remains text labeled `Reference only (not B0)` and is never copied into task state or request payloads.

## Mocking and truthfulness

Until the backend contract is merged, MSW fixtures provide fixed contract examples. Mock UI evidence remains visibly associated with the existing FAKE runtime and is not presented as real hardware state. Dataset links are inert same-origin mock artifact routes; no hardware service is started.

## Error handling

- Invalid or unsafe dataset file URLs are rejected by Zod parsing.
- Unknown required status values fail contract parsing rather than being silently reclassified.
- Dataset failures do not replace task, measurement, recovery, or rollback state.
- Missing optional file association fields do not prevent rendering.

## Test strategy

Implementation follows test-first development:

1. Reducer and workflow tests prove the default is false and both task types send an explicit boolean.
2. Wizard tests prove the copy changes with the switch, the review is locked, and the POST body contains the selected value.
3. Contract tests prove status and file metadata survive Zod parsing and unsafe URLs fail.
4. Run-page tests cover DISABLED, SUCCESS, PARTIAL, and FAILED independently from measurement SUCCESS.
5. Page tests prove DISABLED has no eye placeholder and enabled PNG/NPY/JSON files expose the correct view/download behavior.
6. Existing B0 tests continue to prove readback FIR is used and reference FIR remains reference-only.
7. Full frontend tests, lint, production build, and mock-only browser tests run before the final commit.
