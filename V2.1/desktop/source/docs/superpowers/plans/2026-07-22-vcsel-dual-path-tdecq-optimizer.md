# VCSEL Dual-Path TDECQ Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a hardware-isolated optimizer package with an exhaustive safe-grid path, a TDECQ-only Bayesian path, and reusable parameter-effect memory.

**Architecture:** Both paths consume the same canonical finite parameter space and immutable observations built from actual readback settings. `PATH_EXHAUSTIVE` measures every unique safe endpoint and can truthfully report a grid optimum only after full completion; `PATH_TDECQ_BAYESIAN` minimizes valid TDECQ with finite-pool expected improvement multiplied by TDECQ-validity probability, using deterministic coordinate descent as fallback. ER remains report-only and cannot affect suggestions, stopping, or final ranking.

**Tech Stack:** Python 3.12+, NumPy, SciPy, scikit-learn 1.9.x, Pydantic 2, pytest.

---

## Scope and non-negotiable rules

- Read `docs/superpowers/specs/2026-07-22-vcsel-dual-path-tdecq-tuning-design.md` completely before implementation.
- This plan supersedes `docs/superpowers/plans/2026-07-22-vcsel-ai-tuning-optimizer.md`; do not implement its ER-coarse, ER-constrained Bayesian, FIR-outer, or combined TDECQ/ER preference tasks.
- This plan implements the pure AI optimizer plus configuration, serialization, and replay support. New-driver hardware qualification remains a separate backend/orchestration implementation plan.
- No file under `src/vcsel_ai_tuning/optimization/` may import `vcsel_ai_tuning.hardware`, PyVISA, serial, socket, FastAPI, frontend modules, or persistence implementations.
- Missing or invalid TDECQ stays `None` plus status and reason. Never impute zero, a penalty, ER, or a preceding TDECQ value.
- Model features and measured-point keys always use `settings.readback`, never requested or merely applied settings.
- ER is stored for reporting only. Changing ER values while holding everything else fixed must not change a suggestion, stop decision, or best-point result.
- Maximum step constrains transitions, not membership in the exhaustive endpoint grid.
- All ordering and replay behavior is deterministic for a fixed seed.
- Repeated-measurement and noise-aware confirmation are deliberately outside this revision; stopping is based on pool exhaustion, budget, no safe candidate, or explicit fallback.
- This first optimizer slice persists response nodes, adjacent effect edges,
  pair interactions, invalid regions, and top anchors. Aggregate global/local
  sensitivity distributions, connected low-TDECQ regions, and robust-plateau
  summaries remain a follow-up derived-analysis layer over the same versioned
  nodes and edges; they are not silently claimed as delivered here.

## File map

**Create:**

- `src/vcsel_ai_tuning/optimization/__init__.py` — public optimizer exports without importing hardware.
- `src/vcsel_ai_tuning/optimization/protocol.py` — path enum, normalized observation, suggestions, snapshots, and optimizer protocol.
- `src/vcsel_ai_tuning/optimization/observations.py` — conversion from shared domain observations using readback and nullable TDECQ.
- `src/vcsel_ai_tuning/optimization/parameter_space.py` — finite endpoint enumeration, canonical keys, neighborhoods, and max-step transition planning.
- `src/vcsel_ai_tuning/optimization/exhaustive.py` — truthful exhaustive-path state machine and result.
- `src/vcsel_ai_tuning/optimization/effects.py` — response surface, one-step effects, two-parameter interactions, and anchor extraction.
- `src/vcsel_ai_tuning/optimization/coordinate.py` — deterministic TDECQ-only local fallback.
- `src/vcsel_ai_tuning/optimization/bayesian.py` — TDECQ GPR, guarded validity model, finite-pool EI, and JSON snapshot restore.
- `src/vcsel_ai_tuning/optimization/replay.py` — identical-surface exhaustive/Bayesian comparison.
- `tests_py/ai_tuning/optimization/conftest.py` — shared finite spaces and observations.
- `tests_py/ai_tuning/optimization/test_protocol.py`
- `tests_py/ai_tuning/optimization/test_observations.py`
- `tests_py/ai_tuning/optimization/test_parameter_space.py`
- `tests_py/ai_tuning/optimization/test_exhaustive.py`
- `tests_py/ai_tuning/optimization/test_effects.py`
- `tests_py/ai_tuning/optimization/test_coordinate.py`
- `tests_py/ai_tuning/optimization/test_bayesian.py`
- `tests_py/ai_tuning/optimization/test_replay.py`
- `tests_py/ai_tuning/optimization/test_import_isolation.py`
- `tests_py/ai_tuning/fixtures/dual_path_tdecq_surface.jsonl`
- `docs/dual-path-tdecq-optimizer.md`

**Modify:**

- `src/vcsel_ai_tuning/domain.py:93-187` — permit `None + status + reason` for invalid metrics while keeping valid metrics numeric.
- `src/vcsel_ai_tuning/config.py:27-83` — add the two optimizer modes, every-candidate standard measurement, and immutable tuning identity.
- `tests_py/ai_tuning/test_domain.py` — cover nullable TDECQ invariants.
- `tests_py/ai_tuning/test_config.py` — cover dual-path configuration invariants without breaking existing milestone-one configurations.
- `pyproject.toml:17-25` — add a dedicated optimizer dependency group.

## Environment setup

Commands below assume `python` resolves to Python 3.12 in an activated project environment. On the current Codex desktop, the fallback interpreter is:

```powershell
& 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -e '.[dev,optimizer]'
```

After Python is added to PATH, the portable equivalent is:

```powershell
python -m pip install -e '.[dev,optimizer]'
```

### Task 1: Represent invalid TDECQ without inventing a number

**Files:**

- Modify: `src/vcsel_ai_tuning/domain.py:93-113`
- Modify: `src/vcsel_ai_tuning/domain.py:176-187`
- Modify: `tests_py/ai_tuning/test_domain.py`

- [ ] **Step 1: Write failing tests for valid and invalid TDECQ metric states**

Append these tests to `tests_py/ai_tuning/test_domain.py`:

```python
def test_invalid_tdecq_preserves_none_status_and_reason():
    metric = MetricValue(
        value=None,
        unit="dB",
        fidelity=Fidelity.HARDWARE_STANDARD,
        method_version="flexdca-v1",
        valid=False,
        status="INV",
        invalid_reason="timeout waiting for CORR",
        measurement_definition_id="FUNC1:TDECQ",
    )

    assert metric.value is None
    assert metric.status == "INV"
    assert metric.invalid_reason == "timeout waiting for CORR"


def test_valid_metric_rejects_missing_numeric_value():
    with pytest.raises(ValueError, match="valid metric requires a numeric value"):
        MetricValue(
            value=None,
            unit="dB",
            fidelity=Fidelity.HARDWARE_STANDARD,
            method_version="flexdca-v1",
            valid=True,
            status="CORR",
            measurement_definition_id="FUNC1:TDECQ",
        )


def test_invalid_tdecq_rejects_numeric_substitution():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="invalid tdecq_db must keep value=None"):
        Observation(
            campaign_id="campaign-1",
            run_id="run-invalid-tdecq",
            candidate_id="candidate-invalid-tdecq",
            state=RunState.COMPLETE,
            settings=AppliedSettings(
                requested={"ibias_dac": 70.0},
                applied={"ibias_dac": 70.0},
                readback={"ibias_dac": 70.0},
            ),
            metrics={
                "tdecq_db": MetricValue(
                    value=0.0,
                    unit="dB",
                    fidelity=Fidelity.HARDWARE_STANDARD,
                    method_version="flexdca-v1",
                    valid=False,
                    status="INV",
                    invalid_reason="instrument returned INV",
                    measurement_definition_id="FUNC1:TDECQ",
                )
            },
            valid=True,
            started_at=now,
            finished_at=now,
        )


def test_legacy_valid_metric_without_status_remains_accepted():
    metric = MetricValue(
        value=0.03,
        unit="UI",
        fidelity=Fidelity.SIMULATION_PROXY,
        method_version="skew-v1",
    )

    assert metric.value == pytest.approx(0.03)
    assert metric.status is None
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/test_domain.py -q
```

Expected: collection or validation fails because `MetricValue.value` does not accept `None` and `status`/`invalid_reason` do not exist.

- [ ] **Step 3: Add nullable metric fields and exact invariants**

Change `MetricValue` in `src/vcsel_ai_tuning/domain.py` to preserve all existing fields and add the following definitions and validator:

```python
class MetricValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    value: float | None
    unit: str
    fidelity: Fidelity
    method_version: str
    uncertainty: float | None = Field(default=None, ge=0)
    valid: bool = True
    status: str | None = None
    invalid_reason: str | None = None
    measurement_definition_id: str | None = None

    @model_validator(mode="after")
    def validate_metric_state(self) -> "MetricValue":
        if self.valid and self.value is None:
            raise ValueError("valid metric requires a numeric value")
        if self.fidelity is Fidelity.HARDWARE_STANDARD and not self.measurement_definition_id:
            raise ValueError(
                "hardware_standard metric requires measurement_definition_id"
            )
        return self
```

Inside `Observation.validate_terminal_status`, keep the existing fidelity checks and add:

```python
            if name == "tdecq_db" and not metric.valid:
                if metric.value is not None:
                    raise ValueError("invalid tdecq_db must keep value=None")
                if not metric.status:
                    raise ValueError("invalid tdecq_db requires status")
                if not metric.invalid_reason:
                    raise ValueError("invalid tdecq_db requires invalid_reason")
```

- [ ] **Step 4: Run the domain tests and confirm GREEN**

Run:

```powershell
python -m pytest tests_py/ai_tuning/test_domain.py -q
```

Expected: all domain tests pass, including the three new TDECQ-state tests.

- [ ] **Step 5: Commit the nullable metric contract**

```powershell
git add src/vcsel_ai_tuning/domain.py tests_py/ai_tuning/test_domain.py
git commit -m "feat: preserve invalid TDECQ state"
```

### Task 2: Define dual-path protocol, observation normalization, and campaign configuration

**Files:**

- Create: `src/vcsel_ai_tuning/optimization/__init__.py`
- Create: `src/vcsel_ai_tuning/optimization/protocol.py`
- Create: `src/vcsel_ai_tuning/optimization/observations.py`
- Modify: `src/vcsel_ai_tuning/config.py:27-83`
- Create: `tests_py/ai_tuning/optimization/__init__.py`
- Create: `tests_py/ai_tuning/optimization/test_protocol.py`
- Create: `tests_py/ai_tuning/optimization/test_observations.py`
- Modify: `tests_py/ai_tuning/test_config.py`

- [ ] **Step 1: Write failing protocol and readback-normalization tests**

Create `tests_py/ai_tuning/optimization/test_protocol.py`:

```python
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    OptimizationContext,
    PathComplete,
    TuningPath,
)


def test_protocol_results_round_trip_as_json():
    context = OptimizationContext(
        driver_profile_hash="a" * 64,
        vcsel_model="APA4701",
        device_id="vcsel-001",
        test_context_id="53g-ssprq-25c",
    )
    suggestion = CandidateSuggestion(
        path=TuningPath.TDECQ_BAYESIAN,
        candidate_id="candidate-0001",
        requested={"ibias_dac": 70, "eq": 22},
        reason="highest finite-pool acquisition",
    )
    complete = PathComplete(
        path=TuningPath.EXHAUSTIVE,
        status="GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
        measured_count=4,
        candidate_count=4,
        best_observation_id="obs-3",
        best_tdecq_db=2.1,
    )

    assert context.model_copy() == context
    assert CandidateSuggestion.model_validate_json(suggestion.model_dump_json()) == suggestion
    assert PathComplete.model_validate_json(complete.model_dump_json()) == complete
```

Create `tests_py/ai_tuning/optimization/test_observations.py`:

```python
import pytest

from vcsel_ai_tuning.domain import RunState
from vcsel_ai_tuning.optimization.observations import normalize_observation


def test_normalization_uses_readback_not_requested(domain_observation, context):
    normalized = normalize_observation(domain_observation, context)

    assert normalized.settings_readback == {"ibias_dac": 72, "eq": 21}
    assert normalized.settings_requested == {"ibias_dac": 70, "eq": 22}
    assert normalized.tdecq_db == 2.4
    assert normalized.er_db == 3.2


def test_invalid_tdecq_stays_none_while_er_survives(invalid_tdecq_observation, context):
    normalized = normalize_observation(invalid_tdecq_observation, context)

    assert normalized.tdecq_db is None
    assert normalized.tdecq_status == "INV"
    assert normalized.tdecq_invalid_reason == "timeout waiting for CORR"
    assert normalized.er_db == 3.4
    assert normalized.er_valid is True


def test_application_failure_is_not_normalized_as_a_measurement(
    domain_observation,
    context,
):
    failed = domain_observation.model_copy(
        update={
            "state": RunState.ABORTED,
            "valid": False,
            "failure_reason": "write/readback mismatch",
        }
    )

    with pytest.raises(ValueError, match="completed valid measurement"):
        normalize_observation(failed, context)
```

Append configuration tests to `tests_py/ai_tuning/test_config.py`:

```python
def dual_path_campaign_dict(kind="PATH_EXHAUSTIVE"):
    data = valid_campaign_dict()
    data["optimizer"] = {"kind": kind, "budget": 12, "seed": 1124}
    data["parameters"] = [
        {
            "name": "eq",
            "unit": "code",
            "lower": 0,
            "upper": 2,
            "grid_values": [0, 1, 2],
            "max_step": 1,
        },
        {
            "name": "ibias_dac",
            "unit": "code",
            "lower": 0,
            "upper": 2,
            "grid_values": [0, 1, 2],
            "max_step": 1,
        },
    ]
    data["safe_baseline"] = {"eq": 0, "ibias_dac": 0}
    data["objective"] = {
        "primary_metric": "tdecq_db",
        "direction": "minimize",
        "metric_constraints": [],
        "tie_breakers": [],
    }
    data["measurement_policy"] = "standard_every_candidate"
    data["tuning_identity"] = {
        "driver_profile_hash": "a" * 64,
        "vcsel_model": "APA4701",
        "device_id": "vcsel-001",
        "test_context_id": "53g-ssprq-25c",
    }
    return data


def test_dual_path_requires_tdecq_only_objective():
    data = dual_path_campaign_dict("PATH_TDECQ_BAYESIAN")
    data["objective"]["tie_breakers"] = [
        {"metric": "er_db", "direction": "maximize"}
    ]

    with pytest.raises(ValidationError, match="TDECQ-only"):
        CampaignConfig.model_validate(data)


def test_dual_path_requires_standard_measurement_for_every_candidate():
    data = dual_path_campaign_dict()
    data["measurement_policy"] = "standard_best_only"

    with pytest.raises(ValidationError, match="standard_every_candidate"):
        CampaignConfig.model_validate(data)


def test_dual_path_requires_immutable_tuning_identity():
    data = dual_path_campaign_dict()
    data.pop("tuning_identity")

    with pytest.raises(ValidationError, match="tuning_identity"):
        CampaignConfig.model_validate(data)


def test_exhaustive_budget_cannot_silently_truncate_the_grid():
    data = dual_path_campaign_dict("PATH_EXHAUSTIVE")
    data["optimizer"]["budget"] = 8

    with pytest.raises(ValidationError, match="exhaustive budget"):
        CampaignConfig.model_validate(data)
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_protocol.py tests_py/ai_tuning/optimization/test_observations.py tests_py/ai_tuning/test_config.py -q
```

Expected: import failures for the new package and validation failures for unknown dual-path fields.

- [ ] **Step 3: Implement the protocol models**

Create `src/vcsel_ai_tuning/optimization/protocol.py`:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TuningPath(StrEnum):
    EXHAUSTIVE = "PATH_EXHAUSTIVE"
    TDECQ_BAYESIAN = "PATH_TDECQ_BAYESIAN"


class OptimizationContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    driver_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vcsel_model: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    test_context_id: str = Field(min_length=1)

    @property
    def compatibility_key(self) -> tuple[str, str, str]:
        return (
            self.driver_profile_hash,
            self.vcsel_model,
            self.test_context_id,
        )


class TuningObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    observation_id: str
    candidate_id: str
    context: OptimizationContext
    settings_requested: dict[str, int]
    settings_readback: dict[str, int]
    tdecq_db: float | None
    tdecq_status: str
    tdecq_invalid_reason: str | None = None
    er_db: float | None = None
    er_valid: bool = False
    er_status: str | None = None
    artifact_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_tdecq_state(self) -> "TuningObservation":
        if self.tdecq_status == "CORR" and self.tdecq_db is None:
            raise ValueError("CORR TDECQ requires a numeric value")
        if self.tdecq_status == "CORR" and self.tdecq_invalid_reason is not None:
            raise ValueError("CORR TDECQ cannot have an invalid reason")
        if self.tdecq_status != "CORR" and self.tdecq_db is not None:
            raise ValueError("non-CORR TDECQ must keep value=None")
        if self.tdecq_status != "CORR" and not self.tdecq_invalid_reason:
            raise ValueError("non-CORR TDECQ requires an invalid reason")
        if self.er_valid != (self.er_db is not None):
            raise ValueError("ER validity must match the presence of a numeric value")
        return self


class CandidateSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["CANDIDATE"] = "CANDIDATE"
    path: TuningPath
    candidate_id: str
    requested: dict[str, int]
    reason: str
    acquisition: float | None = None


class FallbackRequested(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["FALLBACK_REQUESTED"] = "FALLBACK_REQUESTED"
    path: TuningPath
    reason: str


class NoSafeCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["NO_SAFE_CANDIDATE"] = "NO_SAFE_CANDIDATE"
    path: TuningPath
    reason: str


class PathComplete(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["PATH_COMPLETE"] = "PATH_COMPLETE"
    path: TuningPath
    status: Literal[
        "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID",
        "EXHAUSTIVE_INCOMPLETE",
        "BEST_TDECQ_FOUND_WITHIN_BUDGET",
        "NO_VALID_TDECQ",
    ]
    measured_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    best_observation_id: str | None = None
    best_tdecq_db: float | None = None


Suggestion = CandidateSuggestion | FallbackRequested | NoSafeCandidate | PathComplete


class OptimizerProtocol(Protocol):
    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[tuple[int, ...]] = frozenset(),
    ) -> Suggestion:
        raise NotImplementedError
```

Create `tests_py/ai_tuning/optimization/conftest.py` with only the fixtures needed by
Task 2. Later tasks extend this file only after their production modules exist; this
keeps each RED/GREEN cycle independently runnable.

```python
from datetime import datetime, timezone

import pytest

from vcsel_ai_tuning.domain import (
    AppliedSettings,
    Fidelity,
    MetricValue,
    Observation,
    RunState,
)
from vcsel_ai_tuning.optimization.protocol import (
    OptimizationContext,
)


@pytest.fixture
def context():
    return OptimizationContext(
        driver_profile_hash="a" * 64,
        vcsel_model="APA4701",
        device_id="vcsel-001",
        test_context_id="53g-ssprq-25c",
    )


def _domain_observation(*, invalid_tdecq: bool) -> Observation:
    now = datetime.now(timezone.utc)
    tdecq = MetricValue(
        value=None if invalid_tdecq else 2.4,
        unit="dB",
        fidelity=Fidelity.HARDWARE_STANDARD,
        method_version="flexdca-v1",
        valid=not invalid_tdecq,
        status="INV" if invalid_tdecq else "CORR",
        invalid_reason="timeout waiting for CORR" if invalid_tdecq else None,
        measurement_definition_id="FUNC1:TDECQ",
    )
    return Observation(
        campaign_id="campaign-1",
        run_id="run-invalid" if invalid_tdecq else "run-valid",
        candidate_id="candidate-1",
        state=RunState.COMPLETE,
        settings=AppliedSettings(
            requested={"ibias_dac": 70.0, "eq": 22.0},
            applied={"ibias_dac": 72.0, "eq": 21.0},
            readback={"ibias_dac": 72.0, "eq": 21.0},
        ),
        metrics={
            "tdecq_db": tdecq,
            "er_db": MetricValue(
                value=3.4 if invalid_tdecq else 3.2,
                unit="dB",
                fidelity=Fidelity.HARDWARE_STANDARD,
                method_version="flexdca-v1",
                valid=True,
                status="CORR",
                measurement_definition_id="FUNC1:ER",
            ),
        },
        valid=True,
        started_at=now,
        finished_at=now,
    )


@pytest.fixture
def domain_observation():
    return _domain_observation(invalid_tdecq=False)


@pytest.fixture
def invalid_tdecq_observation():
    return _domain_observation(invalid_tdecq=True)
```

- [ ] **Step 4: Implement domain-observation normalization**

Create `src/vcsel_ai_tuning/optimization/observations.py`:

```python
from __future__ import annotations

from vcsel_ai_tuning.domain import Observation, RunState

from .protocol import OptimizationContext, TuningObservation


def _integer_settings(values: dict[str, float]) -> dict[str, int]:
    converted: dict[str, int] = {}
    for name, value in values.items():
        integer = int(value)
        if float(integer) != float(value):
            raise ValueError(f"optimizer parameter {name} must be integer-valued")
        converted[name] = integer
    return converted


def normalize_observation(
    observation: Observation,
    context: OptimizationContext,
) -> TuningObservation:
    if not observation.valid or observation.state is not RunState.COMPLETE:
        raise ValueError("optimizer requires a completed valid measurement")
    tdecq = observation.metrics.get("tdecq_db")
    er = observation.metrics.get("er_db")
    tdecq_valid = bool(tdecq is not None and tdecq.valid and tdecq.value is not None)
    return TuningObservation(
        observation_id=observation.run_id,
        candidate_id=observation.candidate_id,
        context=context,
        settings_requested=_integer_settings(observation.settings.requested),
        settings_readback=_integer_settings(observation.settings.readback),
        tdecq_db=float(tdecq.value) if tdecq_valid else None,
        tdecq_status=(tdecq.status or "CORR") if tdecq_valid else (
            tdecq.status if tdecq is not None and tdecq.status else "MISSING"
        ),
        tdecq_invalid_reason=(
            None if tdecq_valid else (
                tdecq.invalid_reason if tdecq is not None else "tdecq_db missing"
            )
        ),
        er_db=(
            float(er.value)
            if er is not None and er.valid and er.value is not None
            else None
        ),
        er_valid=bool(er is not None and er.valid and er.value is not None),
        er_status=(er.status if er is not None else "MISSING"),
        artifact_refs=tuple(
            f"{artifact.relative_path.as_posix()}#sha256={artifact.sha256}"
            for artifact in observation.artifacts
        ),
    )
```

- [ ] **Step 5: Extend campaign configuration without breaking legacy kinds**

Add the following model to `src/vcsel_ai_tuning/config.py` and fields to `CampaignConfig`:

```python
class TuningIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    driver_profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vcsel_model: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    test_context_id: str = Field(min_length=1)
```

Change `OptimizerConfig.kind` and add the campaign fields:

```python
class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal[
        "grid",
        "random",
        "sobol",
        "PATH_EXHAUSTIVE",
        "PATH_TDECQ_BAYESIAN",
    ]
    budget: int = Field(gt=0)
    seed: int = 0


class CampaignConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    campaign_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    backend: BackendConfig
    optimizer: OptimizerConfig
    parameters: tuple[ParameterSpec, ...] = Field(min_length=1)
    safe_baseline: dict[str, float]
    objective: ObjectiveConfig
    slow_measurement_for_best: bool = False
    measurement_policy: Literal[
        "fast_proxy",
        "standard_best_only",
        "standard_every_candidate",
    ] = "fast_proxy"
    tuning_identity: TuningIdentity | None = None
```

At the end of `validate_parameters_and_baseline`, add:

```python
        dual_path = self.optimizer.kind in {
            "PATH_EXHAUSTIVE",
            "PATH_TDECQ_BAYESIAN",
        }
        if dual_path:
            if self.tuning_identity is None:
                raise ValueError("dual-path optimizer requires tuning_identity")
            if self.measurement_policy != "standard_every_candidate":
                raise ValueError(
                    "dual-path optimizer requires standard_every_candidate"
                )
            if (
                self.objective.primary_metric != "tdecq_db"
                or self.objective.direction != "minimize"
                or self.objective.metric_constraints
                or self.objective.tie_breakers
            ):
                raise ValueError(
                    "dual-path optimizer requires a TDECQ-only minimize objective"
                )
            endpoint_count = 1
            for parameter in self.parameters:
                if not parameter.grid_values:
                    raise ValueError(
                        "dual-path optimizer requires finite grid_values"
                    )
                values: set[int] = set()
                for raw in parameter.grid_values:
                    prepared = parameter.prepare(raw)
                    integer = int(prepared)
                    if float(integer) != prepared:
                        raise ValueError(
                            "dual-path optimizer requires integer grid values"
                        )
                    values.add(integer)
                endpoint_count *= len(values)
            if (
                self.optimizer.kind == "PATH_EXHAUSTIVE"
                and self.optimizer.budget < endpoint_count
            ):
                raise ValueError(
                    "exhaustive budget must cover the full unfiltered grid"
                )
        return self
```

- [ ] **Step 6: Export only pure protocol symbols and run GREEN tests**

Create `src/vcsel_ai_tuning/optimization/__init__.py`:

```python
from .protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    OptimizationContext,
    OptimizerProtocol,
    PathComplete,
    Suggestion,
    TuningObservation,
    TuningPath,
)

__all__ = [
    "CandidateSuggestion",
    "FallbackRequested",
    "NoSafeCandidate",
    "OptimizationContext",
    "OptimizerProtocol",
    "PathComplete",
    "Suggestion",
    "TuningObservation",
    "TuningPath",
]
```

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_protocol.py tests_py/ai_tuning/optimization/test_observations.py tests_py/ai_tuning/test_config.py -q
```

Expected: all selected tests pass, including existing legacy `sobol` configuration tests.

- [ ] **Step 7: Commit protocol and configuration**

```powershell
git add src/vcsel_ai_tuning/optimization src/vcsel_ai_tuning/config.py tests_py/ai_tuning/optimization tests_py/ai_tuning/test_config.py
git commit -m "feat: define dual-path optimizer protocol"
```

### Task 3: Build the canonical finite endpoint space and safe transition planner

**Files:**

- Create: `src/vcsel_ai_tuning/optimization/parameter_space.py`
- Create: `tests_py/ai_tuning/optimization/conftest.py`
- Create: `tests_py/ai_tuning/optimization/test_parameter_space.py`

- [ ] **Step 1: Write failing tests for unique endpoints, keys, neighborhoods, and transitions**

Create `tests_py/ai_tuning/optimization/test_parameter_space.py`:

```python
import pytest

from vcsel_ai_tuning.domain import ParameterSpec
from vcsel_ai_tuning.optimization.parameter_space import (
    IntegerParameterSpace,
    NoTransitionPath,
)


def test_endpoint_grid_is_canonical_unique_and_ignores_max_step_membership():
    space = IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="ibias_dac",
                unit="code",
                lower=68,
                upper=72,
                quantization=2,
                max_step=2,
                grid_values=(72, 68, 70, 70),
            ),
            ParameterSpec(
                name="eq",
                unit="code",
                lower=21,
                upper=23,
                quantization=1,
                max_step=1,
                grid_values=(23, 21, 22),
            ),
        )
    )

    points = tuple(space.enumerate_points())

    assert len(points) == 9
    assert points[0] == {"eq": 21, "ibias_dac": 68}
    assert points[-1] == {"eq": 23, "ibias_dac": 72}
    assert {point["ibias_dac"] for point in points} == {68, 70, 72}


def test_cross_constraint_filters_endpoints_before_counting():
    space = IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="ibias_dac", unit="code", lower=70, upper=72,
                grid_values=(70, 72), max_step=2,
            ),
            ParameterSpec(
                name="eq", unit="code", lower=22, upper=23,
                grid_values=(22, 23), max_step=1,
            ),
        ),
        allowed=lambda point: not (
            point["ibias_dac"] == 72 and point["eq"] == 23
        ),
    )

    assert space.candidate_count == 3
    assert {space.key(point) for point in space.enumerate_points()} == {
        (22, 70),
        (22, 72),
        (23, 70),
    }

    with pytest.raises(ValueError, match="outside the safe endpoint grid"):
        space.key({"eq": 23, "ibias_dac": 72})


def test_transition_uses_only_one_parameter_and_respects_each_max_step():
    space = IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="ibias_dac", unit="code", lower=68, upper=72,
                grid_values=(68, 70, 72), max_step=2,
            ),
            ParameterSpec(
                name="eq", unit="code", lower=21, upper=23,
                grid_values=(21, 22, 23), max_step=1,
            ),
        )
    )

    path = space.shortest_transition(
        {"ibias_dac": 68, "eq": 21},
        {"ibias_dac": 72, "eq": 23},
    )

    assert path[-1] == {"eq": 23, "ibias_dac": 72}
    previous = {"eq": 21, "ibias_dac": 68}
    for step in path:
        changed = [name for name in step if step[name] != previous[name]]
        assert len(changed) == 1
        assert abs(step[changed[0]] - previous[changed[0]]) <= {
            "ibias_dac": 2,
            "eq": 1,
        }[changed[0]]
        previous = step


def test_transition_reports_disconnected_safe_regions():
    space = IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="eq", unit="code", lower=20, upper=22,
                grid_values=(20, 21, 22), max_step=1,
            ),
        ),
        allowed=lambda point: point["eq"] != 21,
    )

    with pytest.raises(NoTransitionPath, match="no safe transition"):
        space.shortest_transition({"eq": 20}, {"eq": 22})


def test_effect_adjacency_does_not_skip_an_intermediate_grid_value():
    space = IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="eq", unit="code", lower=0, upper=2,
                grid_values=(0, 1, 2), max_step=2,
            ),
        )
    )

    assert tuple(space.adjacent_neighbors({"eq": 0})) == ({"eq": 1},)


def test_space_rejects_authorization_that_removes_every_endpoint():
    with pytest.raises(ValueError, match="no safe endpoints"):
        IntegerParameterSpace(
            parameters=(
                ParameterSpec(
                    name="eq", unit="code", lower=0, upper=1,
                    grid_values=(0, 1), max_step=1,
                ),
            ),
            allowed=lambda point: False,
        )
```

Extend `tests_py/ai_tuning/optimization/conftest.py`. Add `ParameterSpec` to the
existing `vcsel_ai_tuning.domain` import and add this import after it:

```python
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
```

Then append:

```python
@pytest.fixture
def two_by_two_space():
    return IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="eq", unit="code", lower=0, upper=1,
                grid_values=(0, 1), max_step=1,
            ),
            ParameterSpec(
                name="ibias_dac", unit="code", lower=0, upper=1,
                grid_values=(0, 1), max_step=1,
            ),
        )
    )


@pytest.fixture
def three_by_three_space():
    return IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="eq", unit="code", lower=0, upper=2,
                grid_values=(0, 1, 2), max_step=1,
            ),
            ParameterSpec(
                name="ibias_dac", unit="code", lower=0, upper=2,
                grid_values=(0, 1, 2), max_step=1,
            ),
        )
    )
```

- [ ] **Step 2: Run the parameter-space tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_parameter_space.py -q
```

Expected: import fails because `parameter_space.py` does not exist.

- [ ] **Step 3: Implement canonical endpoint enumeration and deterministic BFS transitions**

Create `src/vcsel_ai_tuning/optimization/parameter_space.py`:

```python
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
from itertools import product
import json

from vcsel_ai_tuning.domain import ParameterSpec

from .protocol import OptimizationContext, TuningObservation

Point = dict[str, int]
PointKey = tuple[int, ...]


class NoTransitionPath(RuntimeError):
    pass


def validated_observation_map(
    space: "IntegerParameterSpace",
    context: OptimizationContext,
    observations: tuple[TuningObservation, ...],
) -> dict[PointKey, TuningObservation]:
    by_key: dict[PointKey, TuningObservation] = {}
    for observation in observations:
        if observation.context != context:
            raise ValueError("observation context does not match this campaign")
        key = space.key(observation.settings_readback)
        if key in by_key:
            raise ValueError("repeated endpoint observations are outside revision 1")
        by_key[key] = observation
    return by_key


class IntegerParameterSpace:
    def __init__(
        self,
        parameters: tuple[ParameterSpec, ...],
        allowed: Callable[[Mapping[str, int]], bool] | None = None,
    ):
        self.parameters = tuple(sorted(parameters, key=lambda item: item.name))
        self.names = tuple(parameter.name for parameter in self.parameters)
        if len(self.names) != len(set(self.names)):
            raise ValueError("parameter names must be unique")
        self._allowed = allowed or (lambda point: True)
        self._values = {
            parameter.name: self._canonical_values(parameter)
            for parameter in self.parameters
        }
        self._points = tuple(
            point for point in self._raw_points() if self._allowed(point)
        )
        if not self._points:
            raise ValueError("authorization produced no safe endpoints")
        self._keys = frozenset(self._unchecked_key(point) for point in self._points)
        fingerprint_payload = {
            "parameters": [
                {
                    "name": parameter.name,
                    "unit": parameter.unit,
                    "max_step": parameter.max_step,
                }
                for parameter in self.parameters
            ],
            "endpoint_keys": sorted(self._keys),
        }
        self.fingerprint = sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _canonical_values(parameter: ParameterSpec) -> tuple[int, ...]:
        if not parameter.grid_values:
            raise ValueError(f"{parameter.name} requires finite grid_values")
        values: set[int] = set()
        for raw in parameter.grid_values:
            prepared = parameter.prepare(raw)
            integer = int(prepared)
            if float(integer) != prepared:
                raise ValueError(f"{parameter.name} must use integer grid values")
            values.add(integer)
        if not values:
            raise ValueError(f"{parameter.name} has no grid values")
        return tuple(sorted(values))

    def _raw_points(self) -> Iterator[Point]:
        value_axes = tuple(self._values[name] for name in self.names)
        for values in product(*value_axes):
            yield dict(zip(self.names, values, strict=True))

    @property
    def candidate_count(self) -> int:
        return len(self._points)

    def enumerate_points(self) -> Iterator[Point]:
        for point in self._points:
            yield dict(point)

    def _unchecked_key(self, point: Mapping[str, int]) -> PointKey:
        if set(point) != set(self.names):
            raise ValueError("point must define every parameter exactly once")
        values: list[int] = []
        for name in self.names:
            integer = int(point[name])
            if float(integer) != float(point[name]):
                raise ValueError(f"{name} must be integer-valued")
            values.append(integer)
        return tuple(values)

    def key(self, point: Mapping[str, int]) -> PointKey:
        key = self._unchecked_key(point)
        if key not in self._keys:
            raise ValueError("point is outside the safe endpoint grid")
        return key

    def point(self, key: PointKey) -> Point:
        point = dict(zip(self.names, key, strict=True))
        if key not in self._keys:
            raise ValueError("point is outside the safe endpoint grid")
        return point

    def local_neighbors(self, point: Mapping[str, int]) -> Iterator[Point]:
        source_key = self.key(point)
        if source_key not in self._keys:
            raise ValueError("source is outside the safe endpoint grid")
        for index, parameter in enumerate(self.parameters):
            current = source_key[index]
            for value in self._values[parameter.name]:
                if value == current:
                    continue
                if parameter.max_step is not None:
                    if abs(value - current) > parameter.max_step:
                        continue
                candidate_key = list(source_key)
                candidate_key[index] = value
                normalized = tuple(candidate_key)
                if normalized in self._keys:
                    yield self.point(normalized)

    def adjacent_neighbors(self, point: Mapping[str, int]) -> Iterator[Point]:
        source_key = self.key(point)
        for index, parameter in enumerate(self.parameters):
            axis = self._values[parameter.name]
            position = axis.index(source_key[index])
            for neighbor_position in (position - 1, position + 1):
                if neighbor_position < 0 or neighbor_position >= len(axis):
                    continue
                value = axis[neighbor_position]
                if (
                    parameter.max_step is not None
                    and abs(value - source_key[index]) > parameter.max_step
                ):
                    continue
                candidate_key = list(source_key)
                candidate_key[index] = value
                normalized = tuple(candidate_key)
                if normalized in self._keys:
                    yield self.point(normalized)

    def shortest_transition(
        self,
        source: Mapping[str, int],
        target: Mapping[str, int],
    ) -> tuple[Point, ...]:
        source_key = self.key(source)
        target_key = self.key(target)
        if source_key not in self._keys or target_key not in self._keys:
            raise ValueError("transition endpoint is outside the safe grid")
        if source_key == target_key:
            return ()
        queue: deque[PointKey] = deque([source_key])
        parent: dict[PointKey, PointKey | None] = {source_key: None}
        while queue:
            current = queue.popleft()
            neighbors = sorted(
                (self.key(point) for point in self.local_neighbors(self.point(current)))
            )
            for neighbor in neighbors:
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                if neighbor == target_key:
                    path: list[PointKey] = [neighbor]
                    while parent[path[-1]] != source_key:
                        previous = parent[path[-1]]
                        if previous is None:
                            raise RuntimeError("transition parent chain is incomplete")
                        path.append(previous)
                    path.reverse()
                    return tuple(self.point(key) for key in path)
                queue.append(neighbor)
        raise NoTransitionPath(
            f"no safe transition from {source_key} to {target_key}"
        )
```

- [ ] **Step 4: Run parameter-space tests and confirm GREEN**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_parameter_space.py -q
```

Expected: all six parameter-space tests pass.

- [ ] **Step 5: Commit finite space and transition planning**

```powershell
git add src/vcsel_ai_tuning/optimization/parameter_space.py tests_py/ai_tuning/optimization/test_parameter_space.py tests_py/ai_tuning/optimization/conftest.py
git commit -m "feat: enumerate safe tuning endpoints"
```

### Task 4: Implement truthful exhaustive characterization

**Files:**

- Create: `src/vcsel_ai_tuning/optimization/exhaustive.py`
- Create: `tests_py/ai_tuning/optimization/test_exhaustive.py`

- [ ] **Step 1: Write failing exhaustive-path tests**

Create `tests_py/ai_tuning/optimization/test_exhaustive.py`:

```python
import pytest

from vcsel_ai_tuning.optimization.exhaustive import ExhaustiveOptimizer
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    OptimizationContext,
    PathComplete,
    TuningObservation,
)


def _context():
    return OptimizationContext(
        driver_profile_hash="a" * 64,
        vcsel_model="APA4701",
        device_id="vcsel-001",
        test_context_id="53g-ssprq-25c",
    )


def _observation(suggestion, tdecq_db, status="CORR"):
    return TuningObservation(
        observation_id=f"obs-{suggestion.candidate_id}",
        candidate_id=suggestion.candidate_id,
        context=_context(),
        settings_requested=suggestion.requested,
        settings_readback=suggestion.requested,
        tdecq_db=tdecq_db,
        tdecq_status=status,
        tdecq_invalid_reason=None if status == "CORR" else "instrument INV",
        er_db=9.9,
        er_valid=True,
        er_status="CORR",
    )


def test_exhaustive_suggests_every_endpoint_once_and_returns_true_grid_best(
    two_by_two_space,
):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context=_context())
    history = []
    values = [3.2, None, 2.4, 2.1]
    seen = set()

    for value in values:
        suggestion = optimizer.suggest(tuple(history))
        assert isinstance(suggestion, CandidateSuggestion)
        key = two_by_two_space.key(suggestion.requested)
        assert key not in seen
        seen.add(key)
        history.append(
            _observation(
                suggestion,
                value,
                status="CORR" if value is not None else "INV",
            )
        )

    result = optimizer.suggest(tuple(history))

    assert isinstance(result, PathComplete)
    assert result.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert result.measured_count == result.candidate_count == 4
    assert result.best_tdecq_db == 2.1


def test_partial_exhaustive_never_claims_global_optimum(two_by_two_space):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context=_context())
    first = optimizer.suggest(())
    history = (_observation(first, 2.5),)

    result = optimizer.finish(history, budget_exhausted=True)

    assert result.status == "EXHAUSTIVE_INCOMPLETE"
    assert result.best_tdecq_db == 2.5
    assert result.measured_count == 1
    assert result.candidate_count == 4


def test_complete_all_invalid_grid_reports_no_valid_tdecq(two_by_two_space):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context=_context())
    history = []
    for _ in range(two_by_two_space.candidate_count):
        suggestion = optimizer.suggest(tuple(history))
        history.append(_observation(suggestion, None, status="INV"))

    result = optimizer.suggest(tuple(history))

    assert result.status == "NO_VALID_TDECQ"
    assert result.best_observation_id is None
    assert result.best_tdecq_db is None


def test_exhaustive_rejects_observation_from_another_device(two_by_two_space):
    optimizer = ExhaustiveOptimizer(two_by_two_space, context=_context())
    suggestion = optimizer.suggest(())
    foreign = _observation(suggestion, 2.5).model_copy(
        update={
            "context": _context().model_copy(update={"device_id": "vcsel-foreign"})
        }
    )

    with pytest.raises(ValueError, match="context"):
        optimizer.suggest((foreign,))
```

- [ ] **Step 2: Run the exhaustive tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_exhaustive.py -q
```

Expected: import fails because `ExhaustiveOptimizer` does not exist.

- [ ] **Step 3: Implement deterministic enumeration and truthful finish states**

Create `src/vcsel_ai_tuning/optimization/exhaustive.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

from .parameter_space import (
    IntegerParameterSpace,
    PointKey,
    validated_observation_map,
)
from .protocol import (
    CandidateSuggestion,
    NoSafeCandidate,
    OptimizationContext,
    PathComplete,
    TuningObservation,
    TuningPath,
)


class ExhaustiveOptimizer:
    def __init__(
        self,
        space: IntegerParameterSpace,
        context: OptimizationContext,
    ):
        self.space = space
        self.context = context
        self._ordered_points = tuple(space.enumerate_points())
        self._candidate_ids = {
            space.key(point): f"exhaustive-{index:08d}"
            for index, point in enumerate(self._ordered_points)
        }

    def _measured_keys(
        self,
        observations: Sequence[TuningObservation],
    ) -> set[PointKey]:
        return set(
            validated_observation_map(
                self.space,
                self.context,
                tuple(observations),
            )
        )

    @staticmethod
    def _best(
        observations: Sequence[TuningObservation],
    ) -> TuningObservation | None:
        valid = [
            item
            for item in observations
            if item.tdecq_status == "CORR" and item.tdecq_db is not None
        ]
        return min(valid, key=lambda item: (item.tdecq_db, item.observation_id)) \
            if valid else None

    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey] = frozenset(),
    ) -> CandidateSuggestion | NoSafeCandidate | PathComplete:
        measured = self._measured_keys(observations)
        if len(measured) == self.space.candidate_count:
            return self.finish(observations, budget_exhausted=False)
        for point in self._ordered_points:
            key = self.space.key(point)
            if key in measured or key in pending_keys:
                continue
            return CandidateSuggestion(
                path=TuningPath.EXHAUSTIVE,
                candidate_id=self._candidate_ids[key],
                requested=point,
                reason="next unmeasured endpoint in canonical order",
            )
        return NoSafeCandidate(
            path=TuningPath.EXHAUSTIVE,
            reason="all remaining safe endpoints are pending",
        )

    def finish(
        self,
        observations: tuple[TuningObservation, ...],
        budget_exhausted: bool,
    ) -> PathComplete:
        measured = self._measured_keys(observations)
        best = self._best(observations)
        complete = len(measured) == self.space.candidate_count
        if complete and best is not None:
            status = "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
        elif complete:
            status = "NO_VALID_TDECQ"
        else:
            status = "EXHAUSTIVE_INCOMPLETE"
        if not complete and not budget_exhausted:
            raise ValueError("incomplete exhaustive result requires an explicit stop")
        return PathComplete(
            path=TuningPath.EXHAUSTIVE,
            status=status,
            measured_count=len(measured),
            candidate_count=self.space.candidate_count,
            best_observation_id=best.observation_id if best is not None else None,
            best_tdecq_db=best.tdecq_db if best is not None else None,
        )
```

- [ ] **Step 4: Run exhaustive and parameter-space tests and confirm GREEN**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_parameter_space.py tests_py/ai_tuning/optimization/test_exhaustive.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit exhaustive characterization**

```powershell
git add src/vcsel_ai_tuning/optimization/exhaustive.py tests_py/ai_tuning/optimization/test_exhaustive.py
git commit -m "feat: add exhaustive TDECQ path"
```

### Task 5: Derive response surfaces and parameter-effect memory

**Files:**

- Create: `src/vcsel_ai_tuning/optimization/effects.py`
- Create: `tests_py/ai_tuning/optimization/test_effects.py`

- [ ] **Step 1: Write failing tests for conditional edges, interactions, invalid regions, and anchors**

Create `tests_py/ai_tuning/optimization/test_effects.py`:

```python
import pytest

from vcsel_ai_tuning.optimization.effects import EffectMemory
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace


def test_effect_edges_are_one_parameter_adjacent_and_signed(
    two_by_two_space,
    complete_surface,
):
    memory = EffectMemory.build(two_by_two_space, complete_surface)

    edge = next(
        item
        for item in memory.edges
        if item.source_key == (0, 0) and item.target_key == (1, 0)
    )
    assert edge.changed_parameter == "eq"
    assert edge.step == 1
    assert edge.delta_tdecq_db == pytest.approx(-0.4)
    assert all(
        sum(left != right for left, right in zip(item.source_key, item.target_key)) == 1
        for item in memory.edges
    )


def test_effect_memory_recovers_known_two_parameter_interaction(
    two_by_two_space,
    complete_surface,
):
    memory = EffectMemory.build(two_by_two_space, complete_surface)

    interaction = memory.interactions[0]
    assert interaction.parameter_a == "eq"
    assert interaction.parameter_b == "ibias_dac"
    assert interaction.delta_delta_tdecq_db == pytest.approx(-0.3)


def test_invalid_points_are_remembered_but_not_ranked_as_anchors(
    two_by_two_space,
    surface_with_invalid_best,
):
    memory = EffectMemory.build(two_by_two_space, surface_with_invalid_best)

    assert (1, 1) in memory.invalid_keys
    assert (1, 1) not in memory.anchor_keys(limit=4)


def test_effect_memory_json_round_trip_is_exact(two_by_two_space, complete_surface):
    memory = EffectMemory.build(two_by_two_space, complete_surface)

    restored = EffectMemory.model_validate_json(memory.model_dump_json())

    assert restored == memory


def test_interaction_builder_skips_cross_constraint_holes(
    two_by_two_space,
    complete_surface,
):
    sparse = IntegerParameterSpace(
        parameters=two_by_two_space.parameters,
        allowed=lambda point: point != {"eq": 1, "ibias_dac": 1},
    )

    memory = EffectMemory.build(sparse, complete_surface[:3])

    assert memory.interactions == ()
```

Extend `tests_py/ai_tuning/optimization/conftest.py` by adding
`TuningObservation` to the existing protocol import, then append the concrete
surface factory and fixtures:

```python
def _tuning_observation(
    context: OptimizationContext,
    eq: int,
    ibias: int,
    tdecq: float | None,
    observation_id: str,
) -> TuningObservation:
    valid = tdecq is not None
    settings = {"eq": eq, "ibias_dac": ibias}
    return TuningObservation(
        observation_id=observation_id,
        candidate_id=f"candidate-{observation_id}",
        context=context,
        settings_requested=settings,
        settings_readback=settings,
        tdecq_db=tdecq,
        tdecq_status="CORR" if valid else "INV",
        tdecq_invalid_reason=None if valid else "instrument returned INV",
        er_db=float(10 - eq - ibias),
        er_valid=True,
        er_status="CORR",
    )


@pytest.fixture
def complete_surface(context):
    return (
        _tuning_observation(context, 0, 0, 3.2, "obs-00"),
        _tuning_observation(context, 0, 1, 3.0, "obs-01"),
        _tuning_observation(context, 1, 0, 2.8, "obs-10"),
        _tuning_observation(context, 1, 1, 2.3, "obs-11"),
    )


@pytest.fixture
def surface_with_invalid_best(context):
    return (
        _tuning_observation(context, 0, 0, 3.2, "obs-00"),
        _tuning_observation(context, 0, 1, 3.0, "obs-01"),
        _tuning_observation(context, 1, 0, 2.8, "obs-10"),
        _tuning_observation(context, 1, 1, None, "obs-11"),
    )
```

- [ ] **Step 2: Run the effect-memory tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_effects.py -q
```

Expected: import fails because `effects.py` does not exist.

- [ ] **Step 3: Implement versioned response nodes, edges, interactions, and anchors**

Create `src/vcsel_ai_tuning/optimization/effects.py` with these public models and builder:

```python
from __future__ import annotations

from itertools import combinations

from pydantic import BaseModel, ConfigDict

from .parameter_space import (
    IntegerParameterSpace,
    PointKey,
    validated_observation_map,
)
from .protocol import OptimizationContext, TuningObservation


class ResponseNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    key: PointKey
    observation_id: str
    tdecq_db: float | None
    tdecq_status: str
    tdecq_invalid_reason: str | None = None
    er_db: float | None = None
    er_valid: bool = False
    er_status: str | None = None
    artifact_refs: tuple[str, ...] = ()


class EffectEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source_key: PointKey
    target_key: PointKey
    changed_parameter: str
    step: int
    delta_tdecq_db: float


class PairInteraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    base_key: PointKey
    parameter_a: str
    parameter_b: str
    delta_delta_tdecq_db: float


class EffectMemory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: str = "effect-memory-v1"
    context: OptimizationContext
    space_fingerprint: str
    parameter_names: tuple[str, ...]
    nodes: tuple[ResponseNode, ...]
    edges: tuple[EffectEdge, ...]
    interactions: tuple[PairInteraction, ...]
    invalid_keys: frozenset[PointKey]

    @classmethod
    def build(
        cls,
        space: IntegerParameterSpace,
        observations: tuple[TuningObservation, ...],
    ) -> "EffectMemory":
        if not observations:
            raise ValueError("effect memory requires observations")
        context = observations[0].context
        by_key = validated_observation_map(space, context, observations)
        nodes = tuple(
            ResponseNode(
                key=key,
                observation_id=item.observation_id,
                tdecq_db=item.tdecq_db,
                tdecq_status=item.tdecq_status,
                tdecq_invalid_reason=item.tdecq_invalid_reason,
                er_db=item.er_db,
                er_valid=item.er_valid,
                er_status=item.er_status,
                artifact_refs=item.artifact_refs,
            )
            for key, item in sorted(by_key.items())
        )
        edges: list[EffectEdge] = []
        for source_key, source in sorted(by_key.items()):
            if source.tdecq_db is None:
                continue
            for target_point in space.adjacent_neighbors(space.point(source_key)):
                target_key = space.key(target_point)
                target = by_key.get(target_key)
                if target is None or target.tdecq_db is None or target_key <= source_key:
                    continue
                changed_index = next(
                    index
                    for index, values in enumerate(zip(source_key, target_key))
                    if values[0] != values[1]
                )
                edges.append(
                    EffectEdge(
                        source_key=source_key,
                        target_key=target_key,
                        changed_parameter=space.names[changed_index],
                        step=target_key[changed_index] - source_key[changed_index],
                        delta_tdecq_db=target.tdecq_db - source.tdecq_db,
                    )
                )
        interactions: list[PairInteraction] = []
        for base_key, base in sorted(by_key.items()):
            if base.tdecq_db is None:
                continue
            for index_a, index_b in combinations(range(len(space.names)), 2):
                point_a = space.point(base_key)
                point_b = space.point(base_key)
                values_a = sorted(
                    value for value in {item[index_a] for item in by_key}
                    if value > base_key[index_a]
                )
                values_b = sorted(
                    value for value in {item[index_b] for item in by_key}
                    if value > base_key[index_b]
                )
                if not values_a or not values_b:
                    continue
                point_a[space.names[index_a]] = values_a[0]
                point_b[space.names[index_b]] = values_b[0]
                point_ab = dict(point_a)
                point_ab[space.names[index_b]] = values_b[0]
                try:
                    key_a = space.key(point_a)
                    key_b = space.key(point_b)
                    key_ab = space.key(point_ab)
                except ValueError:
                    # A cross-parameter authorization rule removed a rectangle corner.
                    continue
                if not all(key in by_key for key in (key_a, key_b, key_ab)):
                    continue
                values = [
                    by_key[key_a].tdecq_db,
                    by_key[key_b].tdecq_db,
                    by_key[key_ab].tdecq_db,
                ]
                if any(value is None for value in values):
                    continue
                interactions.append(
                    PairInteraction(
                        base_key=base_key,
                        parameter_a=space.names[index_a],
                        parameter_b=space.names[index_b],
                        delta_delta_tdecq_db=(
                            by_key[key_ab].tdecq_db
                            - by_key[key_a].tdecq_db
                            - by_key[key_b].tdecq_db
                            + base.tdecq_db
                        ),
                    )
                )
        return cls(
            context=context,
            space_fingerprint=space.fingerprint,
            parameter_names=space.names,
            nodes=nodes,
            edges=tuple(edges),
            interactions=tuple(interactions),
            invalid_keys=frozenset(
                key for key, item in by_key.items() if item.tdecq_db is None
            ),
        )

    def anchor_keys(self, limit: int) -> tuple[PointKey, ...]:
        ranked = sorted(
            (
                (node.tdecq_db, node.key)
                for node in self.nodes
                if node.tdecq_db is not None
            ),
            key=lambda item: (item[0], item[1]),
        )
        return tuple(key for _, key in ranked[:limit])
```

- [ ] **Step 4: Correct the synthetic fixture's expected interaction and run GREEN**

In `tests_py/ai_tuning/optimization/conftest.py`, make the 2×2 valid surface values exactly:

```text
(eq=0, ibias=0) -> 3.2
(eq=1, ibias=0) -> 2.8
(eq=0, ibias=1) -> 3.0
(eq=1, ibias=1) -> 2.3
```

The interaction is `2.3 - 2.8 - 3.0 + 3.2 = -0.3`.

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_effects.py -q
```

Expected: all effect-memory tests pass.

- [ ] **Step 5: Commit parameter-effect memory**

```powershell
git add src/vcsel_ai_tuning/optimization/effects.py tests_py/ai_tuning/optimization/test_effects.py tests_py/ai_tuning/optimization/conftest.py
git commit -m "feat: remember TDECQ parameter effects"
```

### Task 6: Implement deterministic TDECQ-only coordinate fallback

**Files:**

- Create: `src/vcsel_ai_tuning/optimization/coordinate.py`
- Create: `tests_py/ai_tuning/optimization/test_coordinate.py`

- [ ] **Step 1: Write failing tests for one-coordinate moves, TDECQ ranking, ER invariance, and exhaustion**

Create `tests_py/ai_tuning/optimization/test_coordinate.py`:

```python
from vcsel_ai_tuning.optimization.coordinate import CoordinateDescent
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    NoSafeCandidate,
    PathComplete,
)


def test_coordinate_empty_history_starts_at_canonical_endpoint(
    two_by_two_space,
    context,
):
    optimizer = CoordinateDescent(two_by_two_space, context=context, seed=1124)

    suggestion = optimizer.suggest(())

    assert isinstance(suggestion, CandidateSuggestion)
    assert suggestion.requested == next(two_by_two_space.enumerate_points())


def test_coordinate_candidate_changes_exactly_one_parameter(
    two_by_two_space,
    complete_surface,
):
    optimizer = CoordinateDescent(
        two_by_two_space,
        context=complete_surface[0].context,
        seed=1124,
    )
    baseline_observation = complete_surface[0]

    suggestion = optimizer.suggest((baseline_observation,))

    assert isinstance(suggestion, CandidateSuggestion)
    changed = [
        name
        for name, value in suggestion.requested.items()
        if value != baseline_observation.settings_readback[name]
    ]
    assert len(changed) == 1


def test_coordinate_best_uses_only_lowest_valid_tdecq(
    two_by_two_space,
    complete_surface,
):
    optimizer = CoordinateDescent(
        two_by_two_space,
        context=complete_surface[0].context,
        seed=1124,
    )
    mutated_er = tuple(
        item.model_copy(update={"er_db": 1000.0 - index})
        for index, item in enumerate(complete_surface)
    )

    assert optimizer.best(complete_surface).observation_id == "obs-11"
    assert optimizer.best(mutated_er).observation_id == "obs-11"


def test_coordinate_returns_complete_when_no_unmeasured_neighbor_remains(
    two_by_two_space,
    complete_surface,
):
    optimizer = CoordinateDescent(
        two_by_two_space,
        context=complete_surface[0].context,
        seed=1124,
    )

    result = optimizer.suggest(complete_surface)

    assert isinstance(result, PathComplete)
    assert result.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
```

- [ ] **Step 2: Run coordinate tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_coordinate.py -q
```

Expected: import fails because `CoordinateDescent` does not exist.

- [ ] **Step 3: Implement seeded cyclic neighbor selection and TDECQ-only best selection**

Create `src/vcsel_ai_tuning/optimization/coordinate.py`:

```python
from __future__ import annotations

from random import Random

from .parameter_space import (
    IntegerParameterSpace,
    PointKey,
    validated_observation_map,
)
from .protocol import (
    CandidateSuggestion,
    NoSafeCandidate,
    OptimizationContext,
    PathComplete,
    TuningObservation,
    TuningPath,
)


class CoordinateDescent:
    def __init__(
        self,
        space: IntegerParameterSpace,
        context: OptimizationContext,
        seed: int,
    ):
        self.space = space
        self.context = context
        self.seed = seed

    def best(
        self,
        observations: tuple[TuningObservation, ...],
    ) -> TuningObservation | None:
        by_key = validated_observation_map(self.space, self.context, observations)
        valid = [item for item in by_key.values() if item.tdecq_db is not None]
        return min(valid, key=lambda item: (item.tdecq_db, item.observation_id)) \
            if valid else None

    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey] = frozenset(),
    ) -> CandidateSuggestion | NoSafeCandidate | PathComplete:
        measured = set(
            validated_observation_map(
                self.space,
                self.context,
                observations,
            )
        )
        if len(measured) == self.space.candidate_count:
            incumbent = self.best(observations)
            return PathComplete(
                path=TuningPath.TDECQ_BAYESIAN,
                status=(
                    "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
                    if incumbent is not None else "NO_VALID_TDECQ"
                ),
                measured_count=len(measured),
                candidate_count=self.space.candidate_count,
                best_observation_id=(
                    incumbent.observation_id if incumbent is not None else None
                ),
                best_tdecq_db=(
                    incumbent.tdecq_db if incumbent is not None else None
                ),
            )
        available = [
            point for point in self.space.enumerate_points()
            if self.space.key(point) not in measured
            and self.space.key(point) not in pending_keys
        ]
        if not available:
            return NoSafeCandidate(
                path=TuningPath.TDECQ_BAYESIAN,
                reason="all remaining safe endpoints are pending",
            )
        if not observations:
            chosen = available[0]
            return CandidateSuggestion(
                path=TuningPath.TDECQ_BAYESIAN,
                candidate_id="coordinate-00000000",
                requested=chosen,
                reason="deterministic canonical bootstrap",
            )
        incumbent = self.best(observations)
        if incumbent is not None:
            center = incumbent.settings_readback
        elif observations:
            center = observations[-1].settings_readback
        else:
            center = observations[-1].settings_readback
        candidates = [
            point for point in self.space.local_neighbors(center)
            if self.space.key(point) not in measured
            and self.space.key(point) not in pending_keys
        ]
        if not candidates:
            candidates = available
        ordered = sorted(candidates, key=self.space.key)
        random = Random(self.seed + len(observations))
        chosen = ordered[random.randrange(len(ordered))]
        return CandidateSuggestion(
            path=TuningPath.TDECQ_BAYESIAN,
            candidate_id=f"coordinate-{len(observations):08d}",
            requested=chosen,
            reason="deterministic TDECQ-only coordinate fallback",
        )
```

- [ ] **Step 4: Run coordinate tests and confirm GREEN**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_coordinate.py -q
```

Expected: all coordinate tests pass and ER mutation changes no result.

- [ ] **Step 5: Commit coordinate fallback**

```powershell
git add src/vcsel_ai_tuning/optimization/coordinate.py tests_py/ai_tuning/optimization/test_coordinate.py
git commit -m "feat: add TDECQ coordinate fallback"
```

### Task 7: Implement finite-pool TDECQ-only Bayesian optimization

**Files:**

- Modify: `pyproject.toml:17-25`
- Create: `src/vcsel_ai_tuning/optimization/bayesian.py`
- Create: `tests_py/ai_tuning/optimization/test_bayesian.py`

- [ ] **Step 1: Add the optimizer dependency group without overwriting existing user changes**

Append this sibling group under `[project.optional-dependencies]` in `pyproject.toml`; preserve the current `dev`, `hardware`, PySerial, and console-script entries exactly:

```toml
optimizer = [
  "scikit-learn>=1.9,<2",
]
```

Install the editable project:

```powershell
python -m pip install -e '.[dev,optimizer]'
```

Expected: installation succeeds and `python -c "import sklearn; print(sklearn.__version__)"` prints a `1.9.x` version.

- [ ] **Step 2: Write failing Bayesian tests**

Create `tests_py/ai_tuning/optimization/test_bayesian.py`:

```python
import numpy as np
import pytest

from vcsel_ai_tuning.optimization.bayesian import TdecqBayesianOptimizer
from vcsel_ai_tuning.optimization.protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    PathComplete,
)


def test_fit_uses_only_valid_numeric_tdecq_ids(
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )

    optimizer.fit(mixed_surface)

    assert optimizer.tdecq_training_ids == tuple(
        item.observation_id
        for item in mixed_surface
        if item.tdecq_status == "CORR" and item.tdecq_db is not None
    )


def test_acquisition_is_minimization_ei_times_valid_probability(
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )
    optimizer.fit(mixed_surface)

    predictions = optimizer.score_unmeasured(mixed_surface, frozenset())

    assert predictions
    for item in predictions:
        assert item.acquisition == pytest.approx(
            item.expected_improvement * item.valid_probability
        )


def test_er_only_mutation_cannot_change_bayesian_suggestion(
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )
    mutated = tuple(
        item.model_copy(update={"er_db": 1000.0 - index * 100.0})
        for index, item in enumerate(mixed_surface)
    )

    original = optimizer.suggest(mixed_surface)
    changed = optimizer.suggest(mutated)

    assert isinstance(original, CandidateSuggestion)
    assert changed.requested == original.requested
    assert changed.acquisition == pytest.approx(original.acquisition)


def test_pending_and_measured_points_are_never_suggested(
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )
    first = optimizer.suggest(mixed_surface)
    pending = frozenset({three_by_three_space.key(first.requested)})

    second = optimizer.suggest(mixed_surface, pending)

    measured = {
        three_by_three_space.key(item.settings_readback) for item in mixed_surface
    }
    assert isinstance(second, CandidateSuggestion)
    assert three_by_three_space.key(second.requested) not in measured | pending


def test_all_invalid_history_requests_coordinate_fallback(
    three_by_three_space,
    all_invalid_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=all_invalid_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )

    result = optimizer.suggest(all_invalid_surface)

    assert isinstance(result, FallbackRequested)
    assert "numeric TDECQ" in result.reason


def test_only_pending_remaining_points_report_no_safe_candidate(
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )
    measured = {
        three_by_three_space.key(item.settings_readback) for item in mixed_surface
    }
    pending = frozenset(
        three_by_three_space.key(point)
        for point in three_by_three_space.enumerate_points()
        if three_by_three_space.key(point) not in measured
    )

    result = optimizer.suggest(mixed_surface, pending)

    assert isinstance(result, NoSafeCandidate)


def test_bayesian_rejects_observation_from_another_device(
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )
    foreign = mixed_surface[0].model_copy(
        update={
            "context": mixed_surface[0].context.model_copy(
                update={"device_id": "vcsel-foreign"}
            )
        }
    )

    with pytest.raises(ValueError, match="context"):
        optimizer.fit((foreign,))


def test_degenerate_regressor_requests_coordinate_fallback(
    monkeypatch,
    three_by_three_space,
    mixed_surface,
):
    def fail_fit(model, features, targets):
        raise np.linalg.LinAlgError("synthetic singular covariance")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.fit",
        fail_fit,
    )
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )

    result = optimizer.suggest(mixed_surface)

    assert isinstance(result, FallbackRequested)
    assert "model fit failed" in result.reason


def test_full_pool_skips_model_fit_and_reports_grid_optimum(
    monkeypatch,
    two_by_two_space,
    complete_surface,
):
    def fail_fit(model, features, targets):
        raise np.linalg.LinAlgError("fit must not run for a complete pool")

    monkeypatch.setattr(
        "vcsel_ai_tuning.optimization.bayesian.GaussianProcessRegressor.fit",
        fail_fit,
    )
    optimizer = TdecqBayesianOptimizer(
        two_by_two_space,
        context=complete_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )

    result = optimizer.suggest(complete_surface)

    assert isinstance(result, PathComplete)
    assert result.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert result.best_tdecq_db == pytest.approx(2.3)
```

Append the Task 7-only fixtures to
`tests_py/ai_tuning/optimization/conftest.py`:

```python
@pytest.fixture
def mixed_surface(context):
    return (
        _tuning_observation(context, 0, 0, 3.4, "obs-00"),
        _tuning_observation(context, 0, 1, 3.0, "obs-01"),
        _tuning_observation(context, 1, 0, 3.1, "obs-10"),
        _tuning_observation(context, 1, 1, 2.4, "obs-11"),
        _tuning_observation(context, 1, 2, None, "obs-12"),
    )


@pytest.fixture
def all_invalid_surface(context):
    return (
        _tuning_observation(context, 0, 0, None, "obs-00"),
        _tuning_observation(context, 0, 1, None, "obs-01"),
        _tuning_observation(context, 1, 0, None, "obs-10"),
    )
```

- [ ] **Step 3: Run Bayesian tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_bayesian.py -q
```

Expected: import fails because `bayesian.py` does not exist.

- [ ] **Step 4: Implement normalized features, guarded models, and minimization EI**

Create `src/vcsel_ai_tuning/optimization/bayesian.py`:

```python
from __future__ import annotations

from hashlib import sha256

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)

from .parameter_space import (
    IntegerParameterSpace,
    Point,
    PointKey,
    validated_observation_map,
)
from .protocol import (
    CandidateSuggestion,
    FallbackRequested,
    NoSafeCandidate,
    OptimizationContext,
    PathComplete,
    TuningObservation,
    TuningPath,
)


class CandidatePrediction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    key: PointKey
    predicted_tdecq_db: float
    predicted_sigma_db: float = Field(ge=0)
    expected_improvement: float = Field(ge=0)
    valid_probability: float = Field(ge=0, le=1)
    acquisition: float = Field(ge=0)


class TdecqBayesianOptimizer:
    def __init__(
        self,
        space: IntegerParameterSpace,
        context: OptimizationContext,
        seed: int,
        min_valid_observations: int | None = None,
        xi: float = 0.01,
    ):
        self.space = space
        self.context = context
        self.seed = seed
        self.xi = xi
        self.min_valid_observations = (
            min_valid_observations
            if min_valid_observations is not None
            else max(3, len(space.names) + 1)
        )
        self.tdecq_training_ids: tuple[str, ...] = ()
        self._regressor: GaussianProcessRegressor | None = None
        self._classifier: GaussianProcessClassifier | None = None
        self._validity_mode: str = "unfitted"

    def _features(self, points: list[Point]) -> np.ndarray:
        rows: list[list[float]] = []
        all_points = tuple(self.space.enumerate_points())
        minima = {
            name: min(point[name] for point in all_points)
            for name in self.space.names
        }
        maxima = {
            name: max(point[name] for point in all_points)
            for name in self.space.names
        }
        for point in points:
            row: list[float] = []
            for name in self.space.names:
                span = maxima[name] - minima[name]
                row.append(0.0 if span == 0 else (point[name] - minima[name]) / span)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def fit(self, observations: tuple[TuningObservation, ...]) -> None:
        by_key = validated_observation_map(
            self.space,
            self.context,
            observations,
        )
        valid = [
            item
            for item in by_key.values()
            if item.tdecq_status == "CORR" and item.tdecq_db is not None
        ]
        self.tdecq_training_ids = tuple(item.observation_id for item in valid)
        self._regressor = None
        self._classifier = None
        if len(valid) >= self.min_valid_observations:
            x_valid = self._features(
                [dict(item.settings_readback) for item in valid]
            )
            y_valid = np.asarray([item.tdecq_db for item in valid], dtype=float)
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
            )
            self._regressor = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=self.seed,
                n_restarts_optimizer=0,
            )
            self._regressor.fit(x_valid, y_valid)
        labels = np.asarray(
            [
                int(item.tdecq_status == "CORR" and item.tdecq_db is not None)
                for item in by_key.values()
            ],
            dtype=int,
        )
        classes = set(labels.tolist())
        if classes == {1}:
            self._validity_mode = "all_valid"
        elif classes == {0} or not classes:
            self._validity_mode = "all_invalid"
        else:
            self._validity_mode = "classifier"
            self._classifier = GaussianProcessClassifier(
                kernel=1.0 * RBF(length_scale=1.0),
                random_state=self.seed,
                n_restarts_optimizer=0,
            )
            self._classifier.fit(
                self._features(
                    [dict(item.settings_readback) for item in by_key.values()]
                ),
                labels,
            )

    def _valid_probability(self, features: np.ndarray) -> np.ndarray:
        if self._validity_mode == "all_valid":
            return np.ones(features.shape[0], dtype=float)
        if self._validity_mode == "all_invalid":
            return np.zeros(features.shape[0], dtype=float)
        if self._classifier is None:
            raise RuntimeError("validity classifier is not fitted")
        return self._classifier.predict_proba(features)[:, 1]

    def score_unmeasured(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey],
    ) -> tuple[CandidatePrediction, ...]:
        if self._regressor is None:
            raise RuntimeError("TDECQ regressor is not fitted")
        measured = set(
            validated_observation_map(
                self.space,
                self.context,
                observations,
            )
        )
        points = [
            point for point in self.space.enumerate_points()
            if self.space.key(point) not in measured
            and self.space.key(point) not in pending_keys
        ]
        if not points:
            return ()
        features = self._features(points)
        mean, sigma = self._regressor.predict(features, return_std=True)
        valid_probability = self._valid_probability(features)
        best = min(
            item.tdecq_db
            for item in observations
            if item.tdecq_status == "CORR" and item.tdecq_db is not None
        )
        improvement = best - mean - self.xi
        expected_improvement = np.zeros_like(mean)
        positive_sigma = sigma > np.finfo(float).eps
        z = np.zeros_like(mean)
        z[positive_sigma] = improvement[positive_sigma] / sigma[positive_sigma]
        expected_improvement[positive_sigma] = (
            improvement[positive_sigma] * norm.cdf(z[positive_sigma])
            + sigma[positive_sigma] * norm.pdf(z[positive_sigma])
        )
        expected_improvement[~positive_sigma] = np.maximum(
            improvement[~positive_sigma], 0.0
        )
        expected_improvement = np.maximum(expected_improvement, 0.0)
        return tuple(
            CandidatePrediction(
                key=self.space.key(point),
                predicted_tdecq_db=float(mean[index]),
                predicted_sigma_db=float(sigma[index]),
                expected_improvement=float(expected_improvement[index]),
                valid_probability=float(valid_probability[index]),
                acquisition=float(
                    expected_improvement[index] * valid_probability[index]
                ),
            )
            for index, point in enumerate(points)
        )

    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey] = frozenset(),
    ) -> CandidateSuggestion | FallbackRequested | NoSafeCandidate | PathComplete:
        by_key = validated_observation_map(
            self.space,
            self.context,
            observations,
        )
        measured = set(by_key)
        remaining = {
            self.space.key(point) for point in self.space.enumerate_points()
        } - measured
        if not remaining:
            best = min(
                (item for item in observations if item.tdecq_db is not None),
                key=lambda item: (item.tdecq_db, item.observation_id),
                default=None,
            )
            return PathComplete(
                path=TuningPath.TDECQ_BAYESIAN,
                status=(
                    "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
                    if best is not None else "NO_VALID_TDECQ"
                ),
                measured_count=len(measured),
                candidate_count=self.space.candidate_count,
                best_observation_id=(
                    best.observation_id if best is not None else None
                ),
                best_tdecq_db=(best.tdecq_db if best is not None else None),
            )
        if remaining <= pending_keys:
            return NoSafeCandidate(
                path=TuningPath.TDECQ_BAYESIAN,
                reason="all remaining safe endpoints are pending",
            )
        try:
            self.fit(observations)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return FallbackRequested(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=f"Bayesian model fit failed: {type(error).__name__}",
            )
        if self._regressor is None:
            return FallbackRequested(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=(
                    "insufficient valid numeric TDECQ observations for Bayesian fit"
                ),
            )
        predictions = self.score_unmeasured(observations, pending_keys)
        if not predictions:
            raise RuntimeError("unmeasured non-pending pool produced no predictions")
        chosen = max(
            predictions,
            key=lambda item: (item.acquisition, tuple(-value for value in item.key)),
        )
        point = self.space.point(chosen.key)
        digest = sha256(repr(chosen.key).encode("ascii")).hexdigest()[:12]
        return CandidateSuggestion(
            path=TuningPath.TDECQ_BAYESIAN,
            candidate_id=f"bayesian-{digest}",
            requested=point,
            reason="maximum TDECQ EI multiplied by valid probability",
            acquisition=chosen.acquisition,
        )
```

- [ ] **Step 5: Run Bayesian and coordinate tests and confirm GREEN**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_bayesian.py tests_py/ai_tuning/optimization/test_coordinate.py -q
```

Expected: all selected tests pass; scikit-learn may emit convergence warnings, but no test fails.

- [ ] **Step 6: Commit TDECQ-only Bayesian optimization**

```powershell
git add pyproject.toml src/vcsel_ai_tuning/optimization/bayesian.py tests_py/ai_tuning/optimization/test_bayesian.py
git commit -m "feat: add TDECQ-only Bayesian path"
```

### Task 8: Add compatible effect-memory warm starts and serializable Bayesian snapshots

**Files:**

- Modify: `src/vcsel_ai_tuning/optimization/effects.py`
- Modify: `src/vcsel_ai_tuning/optimization/bayesian.py`
- Modify: `tests_py/ai_tuning/optimization/test_effects.py`
- Modify: `tests_py/ai_tuning/optimization/test_bayesian.py`

- [ ] **Step 1: Write failing compatibility, warm-start, and snapshot tests**

Append to `tests_py/ai_tuning/optimization/test_effects.py`:

```python
def test_new_device_same_profile_model_and_context_receives_weak_anchors(
    two_by_two_space,
    complete_surface,
):
    memory = EffectMemory.build(two_by_two_space, complete_surface)
    new_device_context = memory.context.model_copy(update={"device_id": "vcsel-002"})

    anchors = memory.compatible_anchor_points(
        context=new_device_context,
        space=two_by_two_space,
        limit=2,
    )

    assert anchors == (
        two_by_two_space.point((1, 1)),
        two_by_two_space.point((1, 0)),
    )


def test_different_driver_profile_receives_no_anchors(
    two_by_two_space,
    complete_surface,
):
    memory = EffectMemory.build(two_by_two_space, complete_surface)
    incompatible = memory.context.model_copy(
        update={"driver_profile_hash": "b" * 64}
    )

    assert memory.compatible_anchor_points(
        incompatible,
        two_by_two_space,
        limit=2,
    ) == ()
```

Append to `tests_py/ai_tuning/optimization/test_bayesian.py`:

```python
from vcsel_ai_tuning.optimization.effects import EffectMemory


def test_snapshot_round_trip_refits_to_same_suggestion(
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )
    first = optimizer.suggest(mixed_surface)
    assert isinstance(first, CandidateSuggestion)
    pending = frozenset({three_by_three_space.key(first.requested)})
    before = optimizer.suggest(mixed_surface, pending)
    snapshot = optimizer.snapshot(mixed_surface, pending_keys=pending)

    restored = TdecqBayesianOptimizer.from_snapshot(
        three_by_three_space,
        snapshot.model_dump_json(),
    )
    after = restored.suggest(snapshot.observations, snapshot.pending_keys)

    assert isinstance(before, CandidateSuggestion)
    assert isinstance(after, CandidateSuggestion)
    assert after.requested == before.requested
    assert after.acquisition == pytest.approx(before.acquisition)
    assert snapshot.context == mixed_surface[0].context
    assert snapshot.space_fingerprint == three_by_three_space.fingerprint


def test_snapshot_rejects_a_different_candidate_space(
    two_by_two_space,
    three_by_three_space,
    mixed_surface,
):
    optimizer = TdecqBayesianOptimizer(
        three_by_three_space,
        context=mixed_surface[0].context,
        seed=1124,
        min_valid_observations=3,
    )
    snapshot = optimizer.snapshot(mixed_surface, pending_keys=frozenset())

    with pytest.raises(ValueError, match="candidate space"):
        TdecqBayesianOptimizer.from_snapshot(
            two_by_two_space,
            snapshot.model_dump_json(),
        )


def test_historical_anchors_are_not_inserted_as_training_observations(
    two_by_two_space,
    complete_surface,
):
    effect_memory = EffectMemory.build(two_by_two_space, complete_surface)
    current_context = complete_surface[0].context.model_copy(
        update={"device_id": "vcsel-002"}
    )
    current_observations = tuple(
        item.model_copy(
            update={
                "observation_id": f"current-{item.observation_id}",
                "candidate_id": f"current-{item.candidate_id}",
                "context": current_context,
            }
        )
        for item in complete_surface[:3]
    )
    optimizer = TdecqBayesianOptimizer(
        two_by_two_space,
        context=current_context,
        seed=1124,
        min_valid_observations=3,
    )
    anchors = effect_memory.compatible_anchor_points(
        current_context,
        two_by_two_space,
        limit=2,
    )
    optimizer.fit(current_observations)

    assert optimizer.tdecq_training_ids == tuple(
        item.observation_id
        for item in current_observations
        if item.tdecq_db is not None
    )
    assert len(anchors) == 2


def test_compatible_memory_anchor_is_used_as_a_weak_initial_candidate(
    two_by_two_space,
    complete_surface,
):
    memory = EffectMemory.build(two_by_two_space, complete_surface)
    new_context = memory.context.model_copy(update={"device_id": "vcsel-002"})
    anchors = memory.compatible_anchor_points(
        new_context,
        two_by_two_space,
        limit=2,
    )
    optimizer = TdecqBayesianOptimizer(
        two_by_two_space,
        context=new_context,
        seed=1124,
        min_valid_observations=3,
        anchor_points=anchors,
    )

    result = optimizer.suggest(())

    assert isinstance(result, CandidateSuggestion)
    assert result.requested == anchors[0]
    assert result.reason == "compatible effect-memory anchor"
    assert optimizer.tdecq_training_ids == ()
```

- [ ] **Step 2: Run the added tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_effects.py tests_py/ai_tuning/optimization/test_bayesian.py -q
```

Expected: failures for missing compatibility and snapshot methods.

- [ ] **Step 3: Implement compatible weak-anchor extraction**

Add to `EffectMemory` in `effects.py`:

```python
    def compatible_anchor_points(
        self,
        context: OptimizationContext,
        space: IntegerParameterSpace,
        limit: int,
    ) -> tuple[dict[str, int], ...]:
        if context.compatibility_key != self.context.compatibility_key:
            return ()
        if space.fingerprint != self.space_fingerprint:
            return ()
        points: list[dict[str, int]] = []
        for key in self.anchor_keys(limit):
            try:
                points.append(space.point(key))
            except ValueError:
                continue
        return tuple(points)
```

These anchors are candidate-order hints only. Do not append historical TDECQ values to current-device training observations.

- [ ] **Step 4: Implement JSON snapshots that store observations and refit instead of serializing sklearn objects**

Add to `bayesian.py`:

```python
class BayesianSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: str = "tdecq-bayesian-v1"
    context: OptimizationContext
    space_fingerprint: str
    seed: int
    min_valid_observations: int
    xi: float
    observations: tuple[TuningObservation, ...]
    pending_keys: frozenset[PointKey]
    anchor_points: tuple[Point, ...]
```

Extend `TdecqBayesianOptimizer.__init__` with
`anchor_points: tuple[Point, ...] = ()`, validate each point through
`space.key(point)`, and save defensive copies on `self.anchor_points`. At the
inside the existing `if self._regressor is None:` branch, before requesting
coordinate fallback, return the first anchor whose canonical key is neither
measured nor pending:

```python
        for index, point in enumerate(self.anchor_points):
            key = self.space.key(point)
            if key in measured or key in pending_keys:
                continue
            return CandidateSuggestion(
                path=TuningPath.TDECQ_BAYESIAN,
                candidate_id=f"memory-anchor-{index:04d}",
                requested=dict(point),
                reason="compatible effect-memory anchor",
            )
```

This block is used only while `_regressor is None`; once enough current-device
TDECQ observations exist, ordinary Bayesian acquisition takes control. Extend
the snapshot methods as follows:

```python
    def snapshot(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey],
    ) -> BayesianSnapshot:
        validated_observation_map(self.space, self.context, observations)
        for key in pending_keys:
            self.space.point(key)
        return BayesianSnapshot(
            context=self.context,
            space_fingerprint=self.space.fingerprint,
            seed=self.seed,
            min_valid_observations=self.min_valid_observations,
            xi=self.xi,
            observations=observations,
            pending_keys=pending_keys,
            anchor_points=self.anchor_points,
        )

    @classmethod
    def from_snapshot(
        cls,
        space: IntegerParameterSpace,
        document: str,
    ) -> "TdecqBayesianOptimizer":
        snapshot = BayesianSnapshot.model_validate_json(document)
        if snapshot.space_fingerprint != space.fingerprint:
            raise ValueError("snapshot candidate space does not match")
        optimizer = cls(
            space=space,
            context=snapshot.context,
            seed=snapshot.seed,
            min_valid_observations=snapshot.min_valid_observations,
            xi=snapshot.xi,
            anchor_points=snapshot.anchor_points,
        )
        optimizer.fit(snapshot.observations)
        return optimizer
```

- [ ] **Step 5: Run effects and Bayesian tests and confirm GREEN**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_effects.py tests_py/ai_tuning/optimization/test_bayesian.py -q
```

Expected: all selected tests pass and restored suggestions match exactly.

- [ ] **Step 6: Commit warm starts and snapshots**

```powershell
git add src/vcsel_ai_tuning/optimization/effects.py src/vcsel_ai_tuning/optimization/bayesian.py tests_py/ai_tuning/optimization/test_effects.py tests_py/ai_tuning/optimization/test_bayesian.py
git commit -m "feat: add optimizer memory warm starts"
```

### Task 9: Build identical-surface replay comparison and regret reporting

**Files:**

- Create: `src/vcsel_ai_tuning/optimization/replay.py`
- Create: `tests_py/ai_tuning/fixtures/dual_path_tdecq_surface.jsonl`
- Create: `tests_py/ai_tuning/optimization/test_replay.py`

- [ ] **Step 1: Create a complete synthetic response-surface fixture**

Create `tests_py/ai_tuning/fixtures/dual_path_tdecq_surface.jsonl` with
exactly these nine lines:

```jsonl
{"observation_id":"surface-00","candidate_id":"fixture-candidate-00","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":0,"ibias_dac":0},"settings_readback":{"eq":0,"ibias_dac":0},"tdecq_db":3.4,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":2.5,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-01","candidate_id":"fixture-candidate-01","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":0,"ibias_dac":1},"settings_readback":{"eq":0,"ibias_dac":1},"tdecq_db":3.0,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":2.8,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-02","candidate_id":"fixture-candidate-02","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":0,"ibias_dac":2},"settings_readback":{"eq":0,"ibias_dac":2},"tdecq_db":2.9,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":3.0,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-10","candidate_id":"fixture-candidate-10","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":1,"ibias_dac":0},"settings_readback":{"eq":1,"ibias_dac":0},"tdecq_db":3.1,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":3.3,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-11","candidate_id":"fixture-candidate-11","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":1,"ibias_dac":1},"settings_readback":{"eq":1,"ibias_dac":1},"tdecq_db":2.4,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":2.7,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-12","candidate_id":"fixture-candidate-12","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":1,"ibias_dac":2},"settings_readback":{"eq":1,"ibias_dac":2},"tdecq_db":null,"tdecq_status":"INV","tdecq_invalid_reason":"instrument returned INV","er_db":4.2,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-20","candidate_id":"fixture-candidate-20","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":2,"ibias_dac":0},"settings_readback":{"eq":2,"ibias_dac":0},"tdecq_db":2.8,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":3.1,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-21","candidate_id":"fixture-candidate-21","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":2,"ibias_dac":1},"settings_readback":{"eq":2,"ibias_dac":1},"tdecq_db":2.1,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":2.6,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
{"observation_id":"surface-22","candidate_id":"fixture-candidate-22","context":{"driver_profile_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","vcsel_model":"APA4701","device_id":"fixture-vcsel","test_context_id":"53g-ssprq-25c"},"settings_requested":{"eq":2,"ibias_dac":2},"settings_readback":{"eq":2,"ibias_dac":2},"tdecq_db":2.3,"tdecq_status":"CORR","tdecq_invalid_reason":null,"er_db":4.5,"er_valid":true,"er_status":"CORR","artifact_refs":[]}
```

- [ ] **Step 2: Write failing replay tests**

Create `tests_py/ai_tuning/optimization/test_replay.py`:

```python
from pathlib import Path

import pytest

from vcsel_ai_tuning.domain import ParameterSpec
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import TuningPath
from vcsel_ai_tuning.optimization.replay import ReplaySurface, replay

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "dual_path_tdecq_surface.jsonl"
)


def _space():
    return IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="eq", unit="code", lower=0, upper=2,
                grid_values=(0, 1, 2), max_step=1,
            ),
            ParameterSpec(
                name="ibias_dac", unit="code", lower=0, upper=2,
                grid_values=(0, 1, 2), max_step=1,
            ),
        )
    )


def test_exhaustive_replay_returns_known_grid_optimum():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    report = replay(
        surface,
        path=TuningPath.EXHAUSTIVE,
        seed=1124,
        budget=9,
    )

    assert report.status == "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
    assert report.grid_best_tdecq_db == pytest.approx(2.1)
    assert report.best_tdecq_db == pytest.approx(2.1)
    assert report.measurement_count == 9
    assert report.regret_db == pytest.approx(0.0)
    assert report.stop_reason == "POOL_EXHAUSTED"


def test_bayesian_replay_is_seeded_and_reports_exact_regret():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())

    first = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )
    second = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )

    assert second == first
    assert first.measurement_count <= 7
    assert first.grid_best_tdecq_db == pytest.approx(2.1)
    assert first.regret_db == pytest.approx(first.best_tdecq_db - 2.1)
    assert first.status == "BEST_TDECQ_FOUND_WITHIN_BUDGET"
    assert first.stop_reason == "BUDGET_EXHAUSTED"


def test_replay_result_is_invariant_to_er_only_mutation():
    surface = ReplaySurface.from_jsonl(FIXTURE, _space())
    mutated = surface.with_er_values(
        {
            key: 1000.0 - index * 100.0
            for index, key in enumerate(surface.keys)
        }
    )

    original_report = replay(
        surface,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )
    mutated_report = replay(
        mutated,
        path=TuningPath.TDECQ_BAYESIAN,
        seed=1124,
        budget=7,
    )

    assert mutated_report == original_report
```

- [ ] **Step 3: Run replay tests and confirm RED**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_replay.py -q
```

Expected: import fails because `replay.py` does not exist.

- [ ] **Step 4: Implement a readback-keyed replay oracle and reports**

Create `src/vcsel_ai_tuning/optimization/replay.py`:

```python
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .bayesian import TdecqBayesianOptimizer
from .coordinate import CoordinateDescent
from .exhaustive import ExhaustiveOptimizer
from .parameter_space import (
    IntegerParameterSpace,
    PointKey,
    validated_observation_map,
)
from .protocol import (
    CandidateSuggestion,
    FallbackRequested,
    PathComplete,
    TuningObservation,
    TuningPath,
)


class ReplayReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    path: TuningPath
    status: str
    stop_reason: str
    seed: int
    budget: int = Field(gt=0)
    measurement_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    best_observation_id: str | None
    best_tdecq_db: float | None
    grid_best_tdecq_db: float
    regret_db: float | None
    reached_tdecq_below_3: bool


class ReplaySurface:
    def __init__(
        self,
        space: IntegerParameterSpace,
        observations: tuple[TuningObservation, ...],
    ):
        if not observations:
            raise ValueError("replay surface requires observations")
        self.space = space
        self.context = observations[0].context
        self._by_key = validated_observation_map(
            space,
            self.context,
            observations,
        )
        if len(self._by_key) != space.candidate_count:
            raise ValueError("replay surface must cover every safe endpoint exactly once")
        self.keys = tuple(sorted(self._by_key))

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        space: IntegerParameterSpace,
    ) -> "ReplaySurface":
        observations = tuple(
            TuningObservation.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return cls(space, observations)

    def observe(
        self,
        suggestion: CandidateSuggestion,
        sequence: int,
    ) -> TuningObservation:
        key = self.space.key(suggestion.requested)
        source = self._by_key[key]
        return source.model_copy(
            update={
                "observation_id": f"replay-{sequence:08d}",
                "candidate_id": suggestion.candidate_id,
                "settings_requested": dict(suggestion.requested),
            }
        )

    @property
    def grid_best_tdecq_db(self) -> float:
        values = [
            item.tdecq_db
            for item in self._by_key.values()
            if item.tdecq_db is not None
        ]
        if not values:
            raise ValueError("replay surface has no valid TDECQ")
        return min(values)

    def with_er_values(self, values: dict[PointKey, float]) -> "ReplaySurface":
        observations = tuple(
            item.model_copy(update={"er_db": values[key]})
            for key, item in sorted(self._by_key.items())
        )
        return ReplaySurface(self.space, observations)


def _best(
    observations: tuple[TuningObservation, ...],
) -> TuningObservation | None:
    valid = [item for item in observations if item.tdecq_db is not None]
    return min(valid, key=lambda item: (item.tdecq_db, item.observation_id)) \
        if valid else None


def replay(
    surface: ReplaySurface,
    path: TuningPath,
    seed: int,
    budget: int,
) -> ReplayReport:
    if budget <= 0:
        raise ValueError("budget must be positive")
    exhaustive = ExhaustiveOptimizer(surface.space, context=surface.context)
    bayesian = TdecqBayesianOptimizer(
        surface.space,
        context=surface.context,
        seed=seed,
    )
    coordinate = CoordinateDescent(
        surface.space,
        context=surface.context,
        seed=seed,
    )
    observations: tuple[TuningObservation, ...] = ()
    fallback_count = 0
    terminal: PathComplete | None = None
    stop_reason = "BUDGET_EXHAUSTED"
    while len(observations) < budget:
        if path is TuningPath.EXHAUSTIVE:
            suggestion = exhaustive.suggest(observations)
        else:
            suggestion = bayesian.suggest(observations)
            if isinstance(suggestion, FallbackRequested):
                fallback_count += 1
                suggestion = coordinate.suggest(observations)
        if isinstance(suggestion, PathComplete):
            terminal = suggestion
            stop_reason = "POOL_EXHAUSTED"
            break
        if not isinstance(suggestion, CandidateSuggestion):
            stop_reason = suggestion.kind
            break
        observation = surface.observe(suggestion, len(observations))
        observations = (*observations, observation)
    if len(observations) == surface.space.candidate_count:
        stop_reason = "POOL_EXHAUSTED"
    if terminal is None:
        if path is TuningPath.EXHAUSTIVE:
            terminal = exhaustive.finish(observations, budget_exhausted=True)
        else:
            best = _best(observations)
            terminal = PathComplete(
                path=path,
                status=(
                    "BEST_TDECQ_FOUND_WITHIN_BUDGET"
                    if best is not None else "NO_VALID_TDECQ"
                ),
                measured_count=len(observations),
                candidate_count=surface.space.candidate_count,
                best_observation_id=(
                    best.observation_id if best is not None else None
                ),
                best_tdecq_db=best.tdecq_db if best is not None else None,
            )
    best = _best(observations)
    grid_best = surface.grid_best_tdecq_db
    return ReplayReport(
        path=path,
        status=terminal.status,
        stop_reason=stop_reason,
        seed=seed,
        budget=budget,
        measurement_count=len(observations),
        candidate_count=surface.space.candidate_count,
        invalid_count=sum(item.tdecq_db is None for item in observations),
        fallback_count=fallback_count,
        best_observation_id=best.observation_id if best is not None else None,
        best_tdecq_db=best.tdecq_db if best is not None else None,
        grid_best_tdecq_db=grid_best,
        regret_db=(best.tdecq_db - grid_best if best is not None else None),
        reached_tdecq_below_3=bool(
            best is not None and best.tdecq_db is not None and best.tdecq_db < 3.0
        ),
    )
```

- [ ] **Step 5: Run replay tests and confirm GREEN**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_replay.py -q
```

Expected: all three replay tests pass. If the Bayesian budget does not reach the grid optimum, the reported regret remains exact and nonnegative.

- [ ] **Step 6: Commit replay comparison**

```powershell
git add src/vcsel_ai_tuning/optimization/replay.py tests_py/ai_tuning/optimization/test_replay.py tests_py/ai_tuning/fixtures/dual_path_tdecq_surface.jsonl
git commit -m "test: compare exhaustive and Bayesian TDECQ paths"
```

### Task 10: Enforce hardware isolation, document operation, and run full verification

**Files:**

- Create: `tests_py/ai_tuning/optimization/test_import_isolation.py`
- Modify: `src/vcsel_ai_tuning/optimization/__init__.py`
- Create: `docs/dual-path-tdecq-optimizer.md`

- [ ] **Step 1: Write the import-isolation test**

Create `tests_py/ai_tuning/optimization/test_import_isolation.py`:

```python
import ast
from pathlib import Path

OPTIMIZATION_ROOT = (
    Path(__file__).parents[3]
    / "src"
    / "vcsel_ai_tuning"
    / "optimization"
)
BANNED_ROOTS = {
    "fastapi",
    "pyvisa",
    "serial",
    "socket",
    "vcsel_ai_tuning.hardware",
    "vcsel_ai_tuning.persistence",
}


def test_optimizer_package_has_no_hardware_network_api_or_persistence_imports():
    violations = []
    for path in sorted(OPTIMIZATION_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(
                    name == root or name.startswith(f"{root}.")
                    for root in BANNED_ROOTS
                ):
                    violations.append(f"{path.name}: {name}")

    assert violations == []
```

- [ ] **Step 2: Export completed pure-package symbols**

Extend `src/vcsel_ai_tuning/optimization/__init__.py` so it exports these additional symbols:

```python
from .bayesian import BayesianSnapshot, CandidatePrediction, TdecqBayesianOptimizer
from .coordinate import CoordinateDescent
from .effects import EffectEdge, EffectMemory, PairInteraction, ResponseNode
from .exhaustive import ExhaustiveOptimizer
from .parameter_space import IntegerParameterSpace, NoTransitionPath
from .replay import ReplayReport, ReplaySurface, replay
```

Add every imported public name to `__all__`; do not import any hardware or orchestration module.

- [ ] **Step 3: Write operator and developer documentation**

Create `docs/dual-path-tdecq-optimizer.md` with these exact sections:

````markdown
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
TDECQ remains `null` with status and reason.

## Exhaustive effect memory

Path 1 produces response nodes, one-step parameter effects, two-parameter
interactions, invalid regions, and top anchors. A later physical device may use
compatible anchors as weak candidate-order hints, but historical values are not
inserted as current-device training observations.

## Safety boundary

The optimizer never contacts hardware. The orchestrator must provide an
authorized finite endpoint space, validate every max-step transition, use actual
readback values, persist artifacts, and perform rollback.

## Scope boundary

This package does not implement DriverProfile qualification, register writes,
DCA acquisition, authorization UI, artifact persistence, or rollback. Those
remain orchestration/backend work; passing optimizer tests does not complete the
full hardware-design acceptance suite.

## Replay

```powershell
python -m pytest tests_py/ai_tuning/optimization/test_replay.py -q
```

Replay reports exhaustive grid-best TDECQ, Bayesian best TDECQ, measurement
count, invalid count, fallback count, stop reason, and exact grid regret. Replay
results do not prove real-hardware improvement.
````

- [ ] **Step 4: Run optimizer tests**

Run:

```powershell
python -m pytest tests_py/ai_tuning/optimization -q -p no:cacheprovider
```

Expected: all collected optimizer tests pass without hardware or network access.

- [ ] **Step 5: Run the complete AI tuning suite**

Run:

```powershell
python -m pytest tests_py/ai_tuning -q -p no:cacheprovider
```

Expected: all existing and new tests pass; no legacy milestone-one configuration or replay test regresses.

- [ ] **Step 6: Run static delivery checks**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` prints nothing. Review `git status --short` and stage only files listed in this plan; preserve all unrelated user changes.

- [ ] **Step 7: Commit final isolation and documentation**

```powershell
git add src/vcsel_ai_tuning/optimization/__init__.py tests_py/ai_tuning/optimization/test_import_isolation.py docs/dual-path-tdecq-optimizer.md
git commit -m "docs: verify dual-path TDECQ optimizer"
```

## Delivery report

Report:

- every created and modified file;
- the exact installed optimizer dependency version;
- exact optimizer and full-suite test counts;
- the exhaustive candidate count and grid-best TDECQ from replay;
- the Bayesian budget, best TDECQ, exact regret, invalid count, and fallback count;
- the serialized effect-memory and Bayesian snapshot schema versions;
- confirmation that ER-only mutations do not change Bayesian output;
- confirmation that no optimizer module imports or contacts hardware;
- confirmation that driver qualification, live orchestration, and rollback are
  explicitly deferred to a separate implementation plan;
- limitations: grid-only global guarantee, finite-budget Bayesian result, no repeated-measurement/noise confirmation, no hysteresis/path-history model, and no live-hardware claim from replay.
- follow-up effect analysis: aggregate global/local sensitivity distributions,
  connected low-TDECQ regions, and robust-plateau summaries.
