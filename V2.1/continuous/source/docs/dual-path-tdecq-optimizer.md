# Dual-Path TDECQ Optimizer

## Supported modes

- `PATH_EXHAUSTIVE` measures every unique authorized endpoint and reports
  `GLOBAL_OPTIMUM_ON_ENUMERATED_GRID` only after complete enumeration.
- `PATH_TDECQ_BAYESIAN` minimizes valid TDECQ within a finite budget and reports
  `BEST_TDECQ_FOUND_WITHIN_BUDGET`; if it actually measures the full pool, it
  may report `GLOBAL_OPTIMUM_ON_ENUMERATED_GRID`.

## Objective

TDECQ is the only optimization objective. ER is retained in observations and
reports but cannot affect suggestions, stopping, or final ranking. Invalid
TDECQ remains `null` with status and reason. ER is report-only.

## Exhaustive effect memory

Path 1 produces response nodes, one-step parameter effects, two-parameter
interactions, invalid regions, and top anchors. A later physical device may use
compatible anchors as weak candidate-order hints, but historical values are not
inserted as current-device training observations. Historical anchors never train
a new-device model.

## Safety boundary

The optimizer never contacts hardware. The orchestrator must provide an
authorized finite endpoint space, validate every max-step transition, use actual
readback values, persist artifacts, and perform rollback. Unit tests and replay
are software evidence only; they do not make a live-hardware claim.

## Scope boundary

This package does not implement DriverProfile qualification, register writes,
DCA acquisition, authorization UI, artifact persistence, or rollback. Those
remain orchestration/backend work; passing optimizer tests does not complete the
full hardware-design acceptance suite. Driver qualification, live DCA, I2C, and
BERT orchestration, plus rollback, remain separate from this pure optimizer
package.

## Replay

Python 3.12 or newer is required by the project metadata. `.venv-ai-tuning`
already exists in this workspace; on a fresh checkout, set `$VcselPython` to an
available interpreter before creating it. Replace the placeholder path with any installed Python 3.12 or newer executable.

```powershell
$VcselPython = 'C:\path\to\python.exe'
& $VcselPython -m venv .venv-ai-tuning
& '.\.venv-ai-tuning\Scripts\python.exe' -m pip install -e '.[dev,optimizer]'
```

Run the practical offline replay verification with that environment:

```powershell
& '.\.venv-ai-tuning\Scripts\python.exe' -m pytest tests_py/ai_tuning/optimization/test_replay.py -q
```

Replay reports exhaustive grid-best TDECQ, Bayesian best TDECQ, measurement
count, invalid count, fallback count, stop reason, and exact grid regret. Replay
results do not prove real-hardware improvement. When no valid numeric TDECQ is
available, replay returns `NO_VALID_TDECQ`; grid-best TDECQ and exact grid regret
are both `null` on an all-invalid replay surface.
