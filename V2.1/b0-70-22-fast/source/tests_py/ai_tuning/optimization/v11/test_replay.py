from __future__ import annotations

import hashlib
import json

import pytest

import vcsel_ai_tuning.optimization.v11 as public_api
import vcsel_ai_tuning.optimization.v11.replay as replay_module
from vcsel_ai_tuning.optimization.v11.grid import TuningParameter
from vcsel_ai_tuning.optimization.v11.models import (
    ExperimentFingerprint,
    I2CObservation,
    MetricResult,
    OptimizerStatus,
)
from vcsel_ai_tuning.optimization.v11.replay import evaluate_replay


def _fingerprint() -> ExperimentFingerprint:
    return ExperimentFingerprint(
        fir_taps=(0, 6, -28, 104, -8, -10, 0),
        upper_nlc=6,
        lower_nlc=6,
        dca_config_fingerprint="f" * 64,
    )


def _observation(
    observation_id: str,
    *,
    candidate_id: str,
    eq: int,
    tdecq_db: float | None,
    er_db: float | None = None,
) -> I2CObservation:
    result = (
        MetricResult.valid(tdecq_db=tdecq_db, er_db=er_db)
        if tdecq_db is not None
        else MetricResult.invalid(
            status="INV",
            reason="offline replay preserved invalid DCA output",
        )
    )
    return I2CObservation(
        observation_id=observation_id,
        candidate_id=candidate_id,
        requested_i2c={"eq": eq},
        readback_i2c={"eq": eq},
        result=result,
        fingerprint=_fingerprint(),
    )


def test_replay_digest_and_report_are_independent_of_input_history_order():
    history = (
        _observation(
            "obs-b",
            candidate_id="candidate-b",
            eq=1,
            tdecq_db=2.4,
        ),
        _observation(
            "obs-a",
            candidate_id="candidate-a",
            eq=0,
            tdecq_db=2.6,
        ),
    )

    first = evaluate_replay(history)
    second = evaluate_replay(tuple(reversed(history)))

    assert first == second
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.digest == first.digest.lower()
    assert tuple(row.observation_id for row in first.observations) == (
        "obs-a",
        "obs-b",
    )


def test_replay_preserves_invalid_measurements_with_null_tdecq():
    invalid = _observation(
        "obs-invalid",
        candidate_id="candidate-invalid",
        eq=0,
        tdecq_db=None,
    )

    report = evaluate_replay((invalid,))

    assert report.status is OptimizerStatus.NO_VALID_TDECQ
    assert report.invalid_count == 1
    assert report.observations == (invalid,)
    assert report.observations[0].result.tdecq_db is None
    assert report.observations[0].result.tdecq_invalid_reason is not None
    assert report.best_observation is None


def test_replay_uses_strict_tdecq_then_exact_tie_er_objective():
    lower_tdecq = _observation(
        "obs-low",
        candidate_id="candidate-low",
        eq=0,
        tdecq_db=2.4,
        er_db=1.0,
    )
    worse_tdecq_huge_er = _observation(
        "obs-worse",
        candidate_id="candidate-worse",
        eq=1,
        tdecq_db=2.5,
        er_db=100.0,
    )
    tied_higher_er = _observation(
        "obs-tie",
        candidate_id="candidate-tie",
        eq=2,
        tdecq_db=2.4,
        er_db=2.0,
    )

    report = evaluate_replay(
        (worse_tdecq_huge_er, lower_tdecq, tied_higher_er)
    )

    assert report.status is OptimizerStatus.COMPLETE
    assert report.best_observation == tied_higher_er


def test_replay_candidate_order_is_canonical_and_deduplicated():
    history = (
        _observation(
            "obs-c-2",
            candidate_id="candidate-c",
            eq=2,
            tdecq_db=2.4,
        ),
        _observation(
            "obs-a",
            candidate_id="candidate-a",
            eq=0,
            tdecq_db=2.6,
        ),
        _observation(
            "obs-c-1",
            candidate_id="candidate-c",
            eq=2,
            tdecq_db=2.5,
        ),
    )

    report = evaluate_replay(history)

    assert report.candidate_order == ("candidate-a", "candidate-c")
    assert report.observation_count == 3


def test_empty_replay_is_explicitly_not_ready():
    report = evaluate_replay(())

    assert report.status is OptimizerStatus.NOT_READY
    assert report.observations == ()
    assert report.observation_count == 0
    assert report.invalid_count == 0
    assert report.best_observation is None


def test_replay_rejects_duplicate_observation_ids():
    first = _observation(
        "duplicate",
        candidate_id="candidate-a",
        eq=0,
        tdecq_db=2.4,
    )
    second = _observation(
        "duplicate",
        candidate_id="candidate-b",
        eq=1,
        tdecq_db=2.5,
    )

    with pytest.raises(ValueError, match="observation_id"):
        evaluate_replay((first, second))


def test_canonical_replay_report_rejects_digest_tampering():
    report = evaluate_replay(
        (
            _observation(
                "obs-a",
                candidate_id="candidate-a",
                eq=0,
                tdecq_db=2.4,
            ),
        )
    )
    report_type = type(report)
    assert report_type.model_validate_json(report.model_dump_json()) == report
    payload = report.model_dump(mode="json")
    payload["digest"] = "0" * 64

    with pytest.raises(ValueError, match="digest"):
        report_type.model_validate_json(json.dumps(payload))


def test_canonical_replay_report_rejects_noncanonical_candidate_order():
    report = evaluate_replay(
        (
            _observation(
                "obs-a",
                candidate_id="candidate-a",
                eq=0,
                tdecq_db=2.4,
            ),
        )
    )
    payload = report.model_dump(mode="json")
    payload["candidate_order"] = ["fabricated-candidate"]

    with pytest.raises(ValueError, match="candidate_order"):
        type(report).model_validate_json(json.dumps(payload))


def _runner_api(name: str):
    value = getattr(replay_module, name, None)
    assert value is not None, f"replay runner API {name} is missing"
    return value


def _runner_fingerprint(*, main_tap: int = 104) -> ExperimentFingerprint:
    return ExperimentFingerprint(
        fir_taps=(0, 6, -28, main_tap, -8, -10, 0),
        upper_nlc=6,
        lower_nlc=6,
        dca_config_fingerprint="a" * 64,
    )


def _oracle_outcome(
    *,
    readback_eq: int,
    tdecq_db: float | None,
    status: str = "INV",
    reason: str = "simulated invalid TDECQ",
):
    outcome_type = _runner_api("ReplayOracleOutcome")
    result = (
        MetricResult.valid(tdecq_db=tdecq_db, er_db=2.0)
        if tdecq_db is not None
        else MetricResult.invalid(status=status, reason=reason)
    )
    return outcome_type(readback_i2c={"eq": readback_eq}, result=result)


def _oracle_entry(requested_eq: int, *outcomes):
    entry_type = _runner_api("ReplayOracleEntry")
    return entry_type(
        requested_i2c={"eq": requested_eq},
        outcomes=tuple(outcomes),
    )


def _snapshot(
    *,
    stage,
    seed: int,
    coarse_budget: int,
    adaptive_budget: int,
    radius_steps: int,
    fingerprint: ExperimentFingerprint,
    entries=(),
    initial_history=(),
    current_readback=None,
    parameters=None,
):
    snapshot_type = _runner_api("ReplaySnapshot")
    oracle_type = _runner_api("ReplayOracle")
    return snapshot_type(
        stage=stage,
        parameters=(
            (
                TuningParameter(
                    name="eq",
                    tuning_min=0,
                    tuning_max=4,
                    tuning_step=1,
                ),
            )
            if parameters is None
            else tuple(parameters)
        ),
        seed=seed,
        coarse_budget=coarse_budget,
        adaptive_budget=adaptive_budget,
        radius_steps=radius_steps,
        fingerprint=fingerprint,
        oracle=oracle_type(entries=tuple(entries)),
        initial_history=tuple(initial_history),
        current_readback=current_readback,
    )


def test_algorithm_replay_coarse_then_local_fine_uses_global_readback():
    stage = _runner_api("ReplayStage")
    phase = _runner_api("ReplayPhase")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=3,
        adaptive_budget=1,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
        entries=(
            _oracle_entry(
                2,
                _oracle_outcome(readback_eq=2, tdecq_db=1.0),
            ),
            _oracle_entry(
                0,
                _oracle_outcome(readback_eq=4, tdecq_db=3.0),
            ),
            _oracle_entry(
                3,
                _oracle_outcome(readback_eq=3, tdecq_db=2.0),
            ),
            _oracle_entry(
                1,
                _oracle_outcome(readback_eq=1, tdecq_db=1.5),
            ),
        ),
    )

    report = _runner_api("run_replay")(snapshot)

    assert report.status is OptimizerStatus.COMPLETE
    assert tuple(step.phase for step in report.trace) == (
        phase.COARSE,
        phase.COARSE,
        phase.COARSE,
        phase.FINE_BOOTSTRAP,
        phase.FINE,
    )
    fine_step = report.trace[-1]
    assert fine_step.decision.candidate is not None
    assert fine_step.decision.candidate.requested_i2c == {"eq": 1}
    assert fine_step.decision.training_point_count == 3
    assert any(row.readback_i2c == {"eq": 4} for row in report.observations)
    assert report.candidate_order == tuple(
        step.decision.candidate.candidate_id
        for step in report.trace
        if step.decision.candidate is not None
    )


def test_algorithm_replay_same_snapshot_has_identical_trace_and_digest():
    stage = _runner_api("ReplayStage")
    snapshot_type = _runner_api("ReplaySnapshot")
    report_type = _runner_api("ReplayRunReport")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=3,
        adaptive_budget=1,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
        entries=tuple(
            _oracle_entry(
                eq,
                _oracle_outcome(
                    readback_eq=4 if eq == 0 else eq,
                    tdecq_db=float(abs(eq - 2) + 1),
                ),
            )
            for eq in range(5)
        ),
    )
    restored_snapshot = snapshot_type.model_validate_json(
        snapshot.model_dump_json()
    )

    first = _runner_api("run_replay")(snapshot)
    second = _runner_api("run_replay")(restored_snapshot)

    assert first.trace == second.trace
    assert first == second
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert report_type.model_validate_json(first.model_dump_json()) == first


def test_algorithm_replay_preserves_repeated_and_invalid_results_in_order():
    stage = _runner_api("ReplayStage")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=1,
        adaptive_budget=1,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
        entries=(
            _oracle_entry(
                2,
                _oracle_outcome(
                    readback_eq=2,
                    tdecq_db=None,
                    status="UNLOCKED",
                    reason="simulated DCA unlock",
                ),
                _oracle_outcome(readback_eq=2, tdecq_db=1.2),
            ),
        ),
    )

    report = _runner_api("run_replay")(snapshot)

    assert tuple(row.result.tdecq_db for row in report.observations) == (
        None,
        1.2,
    )
    assert report.observations[0].result.tdecq_status == "UNLOCKED"
    assert (
        report.observations[0].result.tdecq_invalid_reason
        == "simulated DCA unlock"
    )
    assert report.observations[0].readback_i2c == {"eq": 2}
    assert report.observations[1].readback_i2c == {"eq": 2}
    assert report.trace[0].observations == report.observations
    assert report.invalid_count == 1


def test_algorithm_replay_retune_isolates_fingerprint_and_remeasures_same_i2c():
    stage = _runner_api("ReplayStage")
    phase = _runner_api("ReplayPhase")
    old_fingerprint = _runner_fingerprint(main_tap=100)
    new_fingerprint = _runner_fingerprint(main_tap=104)
    old_observation = I2CObservation(
        observation_id="old-fir-observation",
        candidate_id="old-fir-candidate",
        requested_i2c={"eq": 4},
        readback_i2c={"eq": 4},
        result=MetricResult.valid(tdecq_db=1.0, er_db=2.0),
        fingerprint=old_fingerprint,
    )
    snapshot = _snapshot(
        stage=stage.RETUNE,
        seed=5,
        coarse_budget=1,
        adaptive_budget=2,
        radius_steps=2,
        fingerprint=new_fingerprint,
        initial_history=(old_observation,),
        current_readback={"eq": 2},
        entries=(
            _oracle_entry(
                4,
                _oracle_outcome(readback_eq=4, tdecq_db=1.4),
            ),
            _oracle_entry(
                0,
                _oracle_outcome(readback_eq=0, tdecq_db=1.6),
            ),
        ),
    )

    report = _runner_api("run_replay")(snapshot)

    assert report.status is OptimizerStatus.COMPLETE
    assert tuple(step.phase for step in report.trace) == (
        phase.RETUNE,
        phase.RETUNE,
    )
    assert report.trace[0].decision.all_history_count == 1
    assert report.trace[0].decision.selected_history_count == 0
    assert report.trace[0].decision.candidate is not None
    assert report.trace[0].decision.candidate.requested_i2c == {"eq": 4}
    assert report.observations[0].requested_i2c == old_observation.requested_i2c
    assert report.observations[0].fingerprint == new_fingerprint
    assert report.final_history == (old_observation,) + report.observations


def test_algorithm_replay_missing_oracle_is_explicitly_not_ready():
    stage = _runner_api("ReplayStage")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=1,
        adaptive_budget=2,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
    )

    report = _runner_api("run_replay")(snapshot)

    assert report.status is OptimizerStatus.NOT_READY
    assert "oracle" in report.reason.lower()
    assert len(report.trace) == 1
    assert report.trace[0].decision.candidate is not None
    assert report.trace[0].observations == ()
    assert report.observations == ()


def test_algorithm_replay_model_not_ready_has_no_fallback_candidate():
    stage = _runner_api("ReplayStage")
    phase = _runner_api("ReplayPhase")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=1,
        adaptive_budget=2,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
        entries=(
            _oracle_entry(
                2,
                _oracle_outcome(readback_eq=2, tdecq_db=1.0),
            ),
        ),
    )

    report = _runner_api("run_replay")(snapshot)

    assert report.status is OptimizerStatus.NOT_READY
    assert tuple(step.phase for step in report.trace) == (
        phase.COARSE,
        phase.FINE_BOOTSTRAP,
        phase.FINE,
    )
    assert report.trace[-1].decision.candidate is None
    assert report.candidate_order == (
        report.trace[0].decision.candidate.candidate_id,
    )


def test_coarse_fine_preserves_but_does_not_train_on_initial_history():
    stage = _runner_api("ReplayStage")
    fingerprint = _runner_fingerprint()
    previous = I2CObservation(
        observation_id="previous-observation",
        candidate_id="previous-candidate",
        requested_i2c={"eq": 4},
        readback_i2c={"eq": 4},
        result=MetricResult.valid(tdecq_db=0.5, er_db=2.0),
        fingerprint=fingerprint,
    )
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=1,
        adaptive_budget=1,
        radius_steps=1,
        fingerprint=fingerprint,
        initial_history=(previous,),
        entries=(
            _oracle_entry(
                2,
                _oracle_outcome(readback_eq=2, tdecq_db=None),
            ),
        ),
    )

    report = _runner_api("run_replay")(snapshot)

    assert report.status is OptimizerStatus.NO_VALID_TDECQ
    assert report.best_observation is None
    assert report.final_history == (previous,) + report.observations


def _redigest_run_report(payload: dict) -> None:
    canonical = {key: value for key, value in payload.items() if key != "digest"}
    document = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload["digest"] = hashlib.sha256(document).hexdigest()


def test_algorithm_replay_report_rejects_hash_consistent_fabricated_best():
    stage = _runner_api("ReplayStage")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=3,
        adaptive_budget=1,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
        entries=tuple(
            _oracle_entry(
                eq,
                _oracle_outcome(
                    readback_eq=4 if eq == 0 else eq,
                    tdecq_db=float(abs(eq - 2) + 1),
                ),
            )
            for eq in range(5)
        ),
    )
    report = _runner_api("run_replay")(snapshot)
    fabricated = I2CObservation(
        observation_id="fabricated-best",
        candidate_id="fabricated-candidate",
        requested_i2c={"eq": 4},
        readback_i2c={"eq": 4},
        result=MetricResult.valid(tdecq_db=0.1, er_db=9.0),
        fingerprint=snapshot.fingerprint,
    )
    payload = report.model_dump(mode="json")
    payload["best_observation"] = fabricated.model_dump(mode="json")
    _redigest_run_report(payload)

    with pytest.raises(ValueError, match="best_observation"):
        type(report).model_validate_json(json.dumps(payload))


def test_algorithm_replay_report_rejects_terminal_status_not_derived_from_trace():
    stage = _runner_api("ReplayStage")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=2,
        adaptive_budget=1,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
        entries=(
            _oracle_entry(
                2,
                _oracle_outcome(readback_eq=2, tdecq_db=1.0),
            ),
        ),
    )
    report = _runner_api("run_replay")(snapshot)
    assert report.status is OptimizerStatus.NOT_READY
    assert report.best_observation is not None
    payload = report.model_dump(mode="json")
    payload["status"] = OptimizerStatus.COMPLETE.value
    _redigest_run_report(payload)

    with pytest.raises(ValueError, match="terminal status"):
        type(report).model_validate_json(json.dumps(payload))


def test_algorithm_replay_report_rejects_hash_consistent_truncated_trace():
    stage = _runner_api("ReplayStage")
    snapshot = _snapshot(
        stage=stage.COARSE_FINE,
        seed=7,
        coarse_budget=3,
        adaptive_budget=1,
        radius_steps=1,
        fingerprint=_runner_fingerprint(),
        entries=tuple(
            _oracle_entry(
                eq,
                _oracle_outcome(
                    readback_eq=4 if eq == 0 else eq,
                    tdecq_db=float(abs(eq - 2) + 1),
                ),
            )
            for eq in range(5)
        ),
    )
    report = _runner_api("run_replay")(snapshot)
    payload = report.model_dump(mode="json")
    first_step = payload["trace"][0]
    first_observations = first_step["observations"]
    payload["trace"] = [first_step]
    payload["candidate_order"] = [
        first_step["decision"]["candidate"]["candidate_id"]
    ]
    payload["observations"] = first_observations
    payload["observation_count"] = len(first_observations)
    payload["invalid_count"] = 0
    payload["final_history"] = first_observations
    payload["best_observation"] = first_observations[0]
    payload["status"] = OptimizerStatus.COMPLETE.value
    payload["reason"] = "forged truncated trace"
    _redigest_run_report(payload)

    with pytest.raises(ValueError, match="incomplete"):
        type(report).model_validate_json(json.dumps(payload))


def _single_retune_report_for_tampering():
    stage = _runner_api("ReplayStage")
    snapshot = _snapshot(
        stage=stage.RETUNE,
        seed=5,
        coarse_budget=1,
        adaptive_budget=1,
        radius_steps=2,
        fingerprint=_runner_fingerprint(),
        current_readback={"eq": 2},
        entries=(
            _oracle_entry(
                4,
                _oracle_outcome(readback_eq=4, tdecq_db=1.4),
            ),
            _oracle_entry(
                3,
                _oracle_outcome(readback_eq=3, tdecq_db=1.5),
            ),
        ),
    )
    return _runner_api("run_replay")(snapshot)


def test_run_report_rejects_hash_consistent_oracle_observation_tampering():
    report = _single_retune_report_for_tampering()
    payload = report.model_dump(mode="json")
    tampered = payload["trace"][0]["observations"][0]
    tampered["readback_i2c"] = {"eq": 3}
    tampered["result"]["tdecq_db"] = 0.1
    payload["observations"] = [tampered]
    payload["final_history"] = [tampered]
    payload["best_observation"] = tampered
    _redigest_run_report(payload)

    with pytest.raises(ValueError, match="simulation"):
        type(report).model_validate_json(json.dumps(payload))


def test_run_report_rejects_hash_consistent_planner_candidate_tampering():
    report = _single_retune_report_for_tampering()
    payload = report.model_dump(mode="json")
    step = payload["trace"][0]
    step["decision"]["candidate"]["candidate_id"] = "forged-candidate"
    step["decision"]["candidate"]["requested_i2c"] = {"eq": 3}
    observation = step["observations"][0]
    observation["candidate_id"] = "forged-candidate"
    observation["requested_i2c"] = {"eq": 3}
    observation["readback_i2c"] = {"eq": 3}
    observation["result"]["tdecq_db"] = 1.5
    payload["candidate_order"] = ["forged-candidate"]
    payload["observations"] = [observation]
    payload["final_history"] = [observation]
    payload["best_observation"] = observation
    _redigest_run_report(payload)

    with pytest.raises(ValueError, match="simulation"):
        type(report).model_validate_json(json.dumps(payload))


def test_snapshot_rejects_parameter_grid_above_hard_point_limit():
    stage = _runner_api("ReplayStage")
    parameters = (
        TuningParameter(
            name="eq",
            tuning_min=0,
            tuning_max=4096,
            tuning_step=1,
        ),
    )

    with pytest.raises(ValueError, match="point_count"):
        _snapshot(
            stage=stage.COARSE_FINE,
            seed=1,
            coarse_budget=1,
            adaptive_budget=1,
            radius_steps=1,
            fingerprint=_runner_fingerprint(),
            parameters=parameters,
        )


def test_snapshot_rejects_combined_candidate_budget_above_hard_limit():
    stage = _runner_api("ReplayStage")

    with pytest.raises(ValueError, match="combined candidate"):
        _snapshot(
            stage=stage.COARSE_FINE,
            seed=1,
            coarse_budget=64,
            adaptive_budget=65,
            radius_steps=1,
            fingerprint=_runner_fingerprint(),
        )


def test_snapshot_rejects_radius_above_hard_limit():
    stage = _runner_api("ReplayStage")

    with pytest.raises(ValueError, match="radius_steps"):
        _snapshot(
            stage=stage.RETUNE,
            seed=1,
            coarse_budget=1,
            adaptive_budget=1,
            radius_steps=4097,
            fingerprint=_runner_fingerprint(),
            current_readback={"eq": 2},
        )


def test_oracle_rejects_more_than_eight_outcomes_for_one_request():
    with pytest.raises(ValueError, match="8 items"):
        _oracle_entry(
            2,
            *tuple(
                _oracle_outcome(readback_eq=2, tdecq_db=float(index + 1))
                for index in range(9)
            ),
        )


def test_oracle_rejects_more_than_128_requested_points():
    oracle_type = _runner_api("ReplayOracle")

    with pytest.raises(ValueError, match="128 items"):
        oracle_type(
            entries=tuple(
                _oracle_entry(
                    eq,
                    _oracle_outcome(readback_eq=eq, tdecq_db=1.0),
                )
                for eq in range(129)
            )
        )


def test_snapshot_rejects_more_than_128_initial_history_rows():
    stage = _runner_api("ReplayStage")
    fingerprint = _runner_fingerprint()
    history = tuple(
        I2CObservation(
            observation_id=f"history-{index:04d}",
            candidate_id=f"candidate-{index:04d}",
            requested_i2c={"eq": index % 5},
            readback_i2c={"eq": index % 5},
            result=MetricResult.valid(tdecq_db=1.0, er_db=2.0),
            fingerprint=fingerprint,
        )
        for index in range(129)
    )

    with pytest.raises(ValueError, match="128 items"):
        _snapshot(
            stage=stage.RETUNE,
            seed=1,
            coarse_budget=1,
            adaptive_budget=1,
            radius_steps=1,
            fingerprint=fingerprint,
            current_readback={"eq": 2},
            initial_history=history,
        )


def test_snapshot_rejects_more_than_64_potential_model_readbacks():
    stage = _runner_api("ReplayStage")
    parameters = (
        TuningParameter(
            name="eq",
            tuning_min=0,
            tuning_max=64,
            tuning_step=1,
        ),
    )
    entries = tuple(
        _oracle_entry(
            eq,
            _oracle_outcome(readback_eq=eq, tdecq_db=1.0),
        )
        for eq in range(65)
    )

    with pytest.raises(ValueError, match="unique model readbacks"):
        _snapshot(
            stage=stage.COARSE_FINE,
            seed=1,
            coarse_budget=1,
            adaptive_budget=1,
            radius_steps=1,
            fingerprint=_runner_fingerprint(),
            entries=entries,
            parameters=parameters,
        )


def test_algorithm_replay_api_is_publicly_exported():
    expected = {
        "ReplayDecision",
        "ReplayOracle",
        "ReplayOracleEntry",
        "ReplayOracleOutcome",
        "ReplayPhase",
        "ReplayRunReport",
        "ReplaySnapshot",
        "ReplayStage",
        "ReplayTraceStep",
        "run_replay",
    }

    assert expected <= set(public_api.__all__)
    for name in expected:
        assert getattr(public_api, name) is getattr(replay_module, name)
