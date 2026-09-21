from datetime import datetime, timezone
from pathlib import Path

import pytest

from vcsel_ai_tuning.domain import (
    AppliedSettings,
    ArtifactRef,
    Fidelity,
    MetricValue,
    Observation,
    ParameterSpec,
    RunState,
)
from vcsel_ai_tuning.optimization.parameter_space import IntegerParameterSpace
from vcsel_ai_tuning.optimization.protocol import (
    OptimizationContext,
    TuningObservation,
)


@pytest.fixture
def context():
    return OptimizationContext(
        driver_profile_hash="a" * 64,
        vcsel_model="APA4701",
        device_id="vcsel-001",
        test_context_id="53g-ssprq-25c",
    )


@pytest.fixture
def two_by_two_space():
    return IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="ibias_dac",
                unit="DAC",
                lower=0,
                upper=1,
                max_step=1,
                grid_values=(0, 1),
            ),
            ParameterSpec(
                name="eq",
                unit="tap",
                lower=0,
                upper=1,
                max_step=1,
                grid_values=(0, 1),
            ),
        )
    )


@pytest.fixture
def three_by_three_space():
    return IntegerParameterSpace(
        parameters=(
            ParameterSpec(
                name="ibias_dac",
                unit="DAC",
                lower=0,
                upper=2,
                max_step=1,
                grid_values=(0, 1, 2),
            ),
            ParameterSpec(
                name="eq",
                unit="tap",
                lower=0,
                upper=2,
                max_step=1,
                grid_values=(0, 1, 2),
            ),
        )
    )


def _tuning_observation(
    context: OptimizationContext,
    point: dict[str, int],
    *,
    observation_id: str,
    tdecq_db: float | None,
    tdecq_status: str | None = None,
    tdecq_invalid_reason: str | None = None,
    er_db: float | None = None,
    artifact_refs: tuple[str, ...] = (),
) -> TuningObservation:
    valid_tdecq = tdecq_db is not None
    if er_db is None:
        er_db = float(10 - sum(point.values()))
    return TuningObservation(
        observation_id=observation_id,
        candidate_id=f"candidate-{observation_id}",
        context=context,
        settings_requested=point,
        settings_readback=point,
        tdecq_db=tdecq_db,
        tdecq_status=tdecq_status or ("CORR" if valid_tdecq else "INV"),
        tdecq_invalid_reason=(
            None
            if valid_tdecq
            else tdecq_invalid_reason or "TDECQ measurement invalid"
        ),
        er_db=er_db,
        er_valid=True,
        er_status="CORR",
        artifact_refs=artifact_refs,
    )


@pytest.fixture
def tuning_observation():
    return _tuning_observation


@pytest.fixture
def complete_surface(context, two_by_two_space):
    values = {
        (0, 0): ("obs-00", 3.2),
        (0, 1): ("obs-01", 3.0),
        (1, 0): ("obs-10", 2.8),
        (1, 1): ("obs-11", 2.3),
    }
    return tuple(
        _tuning_observation(
            context,
            two_by_two_space.point(key),
            observation_id=observation_id,
            tdecq_db=tdecq_db,
        )
        for key, (observation_id, tdecq_db) in values.items()
    )


@pytest.fixture
def surface_with_invalid_best(context, two_by_two_space, complete_surface):
    valid_observations = complete_surface[:-1]
    invalid_best = _tuning_observation(
        context,
        two_by_two_space.point((1, 1)),
        observation_id="obs-11",
        tdecq_db=None,
        tdecq_status="INV",
        tdecq_invalid_reason="timeout waiting for CORR",
    )
    return valid_observations + (invalid_best,)


@pytest.fixture
def mixed_surface(context, three_by_three_space):
    values = {
        (0, 0): ("obs-00", 3.4),
        (0, 1): ("obs-01", 3.0),
        (1, 0): ("obs-10", 3.1),
        (1, 1): ("obs-11", 2.4),
        (1, 2): ("obs-12", None),
    }
    return tuple(
        _tuning_observation(
            context,
            three_by_three_space.point(key),
            observation_id=observation_id,
            tdecq_db=tdecq_db,
        )
        for key, (observation_id, tdecq_db) in values.items()
    )


@pytest.fixture
def all_invalid_surface(context, three_by_three_space):
    values = {
        (0, 0): "obs-00",
        (0, 1): "obs-01",
        (1, 0): "obs-10",
    }
    return tuple(
        _tuning_observation(
            context,
            three_by_three_space.point(key),
            observation_id=observation_id,
            tdecq_db=None,
        )
        for key, observation_id in values.items()
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
            applied={"ibias_dac": 71.0, "eq": 20.0},
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
        artifacts=[
            ArtifactRef(
                relative_path=Path("measurements") / "run-1.json",
                sha256="b" * 64,
                media_type="application/json",
            )
        ],
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
