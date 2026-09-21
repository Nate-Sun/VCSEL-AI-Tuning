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


def _canonical_tdecq_status(status: str | None) -> str | None:
    if status is None:
        return None
    canonical = status.strip().upper()
    return canonical or None


def normalize_observation(
    observation: Observation,
    context: OptimizationContext,
) -> TuningObservation:
    if (
        not isinstance(observation, Observation)
        or not observation.valid
        or observation.state is not RunState.COMPLETE
    ):
        raise ValueError("optimizer requires a completed valid measurement")

    tdecq = observation.metrics.get("tdecq_db")
    er = observation.metrics.get("er_db")
    tdecq_valid = bool(
        tdecq is not None and tdecq.valid and tdecq.value is not None
    )
    er_valid = bool(er is not None and er.valid and er.value is not None)
    tdecq_status = _canonical_tdecq_status(
        tdecq.status if tdecq is not None else None
    )
    if tdecq_valid:
        tdecq_status = tdecq_status or "CORR"
        if tdecq_status != "CORR":
            raise ValueError(
                "valid numeric TDECQ status must resolve to CORR"
            )
    else:
        tdecq_status = tdecq_status or "MISSING"

    return TuningObservation(
        observation_id=observation.run_id,
        candidate_id=observation.candidate_id,
        context=context,
        settings_requested=_integer_settings(observation.settings.requested),
        settings_readback=_integer_settings(observation.settings.readback),
        tdecq_db=float(tdecq.value) if tdecq_valid else None,
        tdecq_status=tdecq_status,
        tdecq_invalid_reason=(
            None
            if tdecq_valid
            else (
                tdecq.invalid_reason
                if tdecq is not None
                else "tdecq_db missing"
            )
        ),
        er_db=float(er.value) if er_valid else None,
        er_valid=er_valid,
        er_status=er.status if er is not None else "MISSING",
        artifact_refs=tuple(
            f"{artifact.relative_path.as_posix()}#sha256={artifact.sha256}"
            for artifact in observation.artifacts
        ),
    )
