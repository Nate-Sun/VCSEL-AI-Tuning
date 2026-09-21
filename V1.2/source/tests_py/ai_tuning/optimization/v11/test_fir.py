from __future__ import annotations

from math import inf, nan

import pytest

from vcsel_ai_tuning.hardware.fir import (
    calculate_first_pass,
    calculate_iterative_pass,
)
from vcsel_ai_tuning.optimization.v11.fir import (
    FirAuthorizationEnvelope,
    FirCandidate,
    FirDecisionController,
    FirDecisionStatus,
    FirTarget,
)
from vcsel_ai_tuning.optimization.v11.models import (
    AggregatedObservation,
    ExperimentFingerprint,
    I2CObservation,
    MetricResult,
)


DCA_TAPS = (
    0.0,
    0.0,
    -0.1,
    1.0,
    0.05,
    0.0,
    0.0,
    0.0,
    0.0,
)
ITERATIVE_DCA_TAPS = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DCA_CONFIG_FINGERPRINT = "a" * 64


def _baseline() -> FirTarget:
    return FirTarget(
        taps=(0, 6, -28, 104, -8, -10, 0),
        upper_nlc=6,
        lower_nlc=7,
    )


def _authorization(
    *,
    tap_ranges=None,
    max_step=None,
    max_total_change_l1=500,
) -> FirAuthorizationEnvelope:
    return FirAuthorizationEnvelope(
        tap_ranges=tap_ranges
        or (
            (-170, 170),
            (-170, 170),
            (-170, 170),
            (104, 104),
            (-170, 170),
            (-170, 170),
            (-170, 170),
        ),
        max_step=max_step or (200, 200, 200, 1, 200, 200, 200),
        max_total_change_l1=max_total_change_l1,
    )


def _controller(
    *,
    authorization=None,
    baseline_result=None,
    iterative_formula=calculate_iterative_pass,
    dca_config_fingerprint=DCA_CONFIG_FINGERPRINT,
    allow_unaccepted_iterative=False,
) -> FirDecisionController:
    return FirDecisionController(
        baseline_fir=_baseline(),
        baseline_result=baseline_result
        or MetricResult.valid(tdecq_db=3.0, er_db=3.0),
        authorization=authorization or _authorization(),
        expected_dca_tap_count=len(DCA_TAPS),
        calculate_iterative_pass=iterative_formula,
        dca_config_fingerprint=dca_config_fingerprint,
        allow_unaccepted_iterative=allow_unaccepted_iterative,
    )


def _valid(tdecq_db: float, er_db: float | None = 3.0) -> MetricResult:
    return MetricResult.valid(tdecq_db=tdecq_db, er_db=er_db)


def _confirm(
    controller: FirDecisionController,
    candidate: FirCandidate,
    measurements: tuple[MetricResult, MetricResult, MetricResult],
):
    first = controller.evaluate(candidate, (measurements[0],))
    assert first.status is FirDecisionStatus.NEEDS_CONFIRMATION
    return controller.evaluate(candidate, measurements)


def _retune_observation(
    controller: FirDecisionController,
    result: MetricResult,
    *,
    fingerprint: ExperimentFingerprint | None = None,
    aggregated: bool = False,
) -> I2CObservation | AggregatedObservation:
    context = fingerprint or controller.retune_fingerprint
    raw = I2CObservation(
        observation_id="retune-observation",
        candidate_id="retune-candidate",
        requested_i2c={"eq": 2},
        readback_i2c={"eq": 2},
        result=result,
        fingerprint=context,
    )
    if not aggregated:
        return raw
    return AggregatedObservation(
        readback_i2c=raw.readback_i2c,
        fingerprint=context,
        result=result,
        raw_observations=(raw,),
    )


def _accept_first(controller: FirDecisionController):
    proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert proposal.candidate is not None
    return _confirm(
        controller,
        proposal.candidate,
        (_valid(2.9), _valid(2.8), _valid(2.7)),
    )


def test_controller_requires_a_sha256_dca_config_fingerprint():
    with pytest.raises(ValueError, match="dca_config_fingerprint"):
        _controller(dca_config_fingerprint="not-a-sha256")


def test_current_and_retune_fingerprints_follow_the_accepted_fir_context():
    controller = _controller()

    assert controller.current_fingerprint == ExperimentFingerprint(
        fir_taps=_baseline().taps,
        upper_nlc=_baseline().upper_nlc,
        lower_nlc=_baseline().lower_nlc,
        dca_config_fingerprint=DCA_CONFIG_FINGERPRINT,
    )

    accepted = _accept_first(controller)
    assert accepted.recommended_fir is not None
    assert controller.retune_fingerprint == controller.current_fingerprint
    assert controller.retune_fingerprint == ExperimentFingerprint(
        fir_taps=accepted.recommended_fir.taps,
        upper_nlc=accepted.recommended_fir.upper_nlc,
        lower_nlc=accepted.recommended_fir.lower_nlc,
        dca_config_fingerprint=DCA_CONFIG_FINGERPRINT,
    )


def test_first_pass_convolves_b0_with_e0_and_never_uses_the_e0_only_formula():
    calls = []

    def iterative_formula(current_taps, dca_taps, dca_main_index, manual_main):
        calls.append(
            (tuple(current_taps), tuple(dca_taps), dca_main_index, manual_main)
        )
        return calculate_iterative_pass(
            current_taps,
            dca_taps,
            dca_main_index,
            manual_main,
        )

    controller = _controller(iterative_formula=iterative_formula)

    decision = controller.propose_first(DCA_TAPS, dca_main_index=3)
    expected = calculate_iterative_pass(
        _baseline().taps,
        DCA_TAPS,
        dca_main_index=3,
        manual_main=104,
    )
    legacy = calculate_first_pass(
        DCA_TAPS,
        dca_main_index=3,
        manual_main=104,
    )

    assert calls == [(_baseline().taps, DCA_TAPS, 3, 104)]
    assert decision.status is FirDecisionStatus.CANDIDATE
    assert decision.candidate is not None
    assert decision.candidate.candidate_id == "fir-formula-direct"
    assert decision.candidate.model_version == "fir-formula-direct-v1.1"
    assert decision.candidate.reason == "direct FIR formula proposal"
    assert decision.candidate.mode == "FIRST"
    assert decision.candidate.requested_fir.taps == expected.taps
    assert decision.candidate.requested_fir.taps != legacy.taps
    assert decision.candidate.requested_fir.upper_nlc == _baseline().upper_nlc
    assert decision.candidate.requested_fir.lower_nlc == _baseline().lower_nlc
    assert decision.candidate.requested_fir.taps[3] == _baseline().taps[3]


def test_iterative_uses_the_last_accepted_fir_and_never_restarts_first_pass():
    iterative_calls = []

    def iterative_formula(current_taps, dca_taps, dca_main_index, manual_main):
        iterative_calls.append(
            (tuple(current_taps), tuple(dca_taps), dca_main_index, manual_main)
        )
        return calculate_iterative_pass(
            current_taps,
            dca_taps,
            dca_main_index,
            manual_main,
        )

    controller = _controller(iterative_formula=iterative_formula)
    first = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert first.candidate is not None
    accepted = _confirm(
        controller,
        first.candidate,
        (_valid(2.7), _valid(2.5), _valid(2.6)),
    )
    assert accepted.status is FirDecisionStatus.ACCEPTED

    iterative = controller.propose_iterative(ITERATIVE_DCA_TAPS, dca_main_index=3)
    expected = calculate_iterative_pass(
        accepted.recommended_fir.taps,
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
        manual_main=104,
    )

    assert len(iterative_calls) == 2
    assert iterative_calls[0][0] == _baseline().taps
    assert iterative_calls[1][0] == accepted.recommended_fir.taps
    assert iterative.status is FirDecisionStatus.CANDIDATE
    assert iterative.candidate is not None
    assert iterative.candidate.mode == "ITERATIVE"
    assert iterative.candidate.requested_fir.taps == expected.taps


def test_explicit_fixed_schedule_can_iterate_from_best_after_first_rejection():
    calls = []

    def iterative_formula(current_taps, dca_taps, dca_main_index, manual_main):
        calls.append(tuple(current_taps))
        return calculate_iterative_pass(
            current_taps,
            dca_taps,
            dca_main_index,
            manual_main,
        )

    controller = _controller(
        iterative_formula=iterative_formula,
        allow_unaccepted_iterative=True,
    )
    first = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert first.candidate is not None
    rejected = controller.evaluate(first.candidate, (_valid(3.1),))
    assert rejected.status is FirDecisionStatus.REJECTED_FIRST_PASS

    iterative = controller.propose_iterative(
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
    )

    assert iterative.status is FirDecisionStatus.CANDIDATE
    assert iterative.candidate is not None
    assert iterative.candidate.mode == "ITERATIVE"
    assert calls == [_baseline().taps, _baseline().taps]


def test_rejected_candidate_never_becomes_the_next_convolution_base():
    calls = []

    def iterative_formula(current_taps, dca_taps, dca_main_index, manual_main):
        calls.append(tuple(current_taps))
        return calculate_iterative_pass(
            current_taps,
            dca_taps,
            dca_main_index,
            manual_main,
        )

    controller = _controller(iterative_formula=iterative_formula)
    first = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert first.candidate is not None
    accepted = _confirm(
        controller,
        first.candidate,
        (_valid(2.7), _valid(2.5), _valid(2.6)),
    )
    best_fir = accepted.recommended_fir
    assert best_fir is not None

    next_equalizer = (0.0, 0.0, 0.02, 1.0, -0.02, 0.0, 0.0, 0.0, 0.0)
    rejected = controller.propose_iterative(
        next_equalizer,
        dca_main_index=3,
    )
    assert rejected.candidate is not None
    assert rejected.candidate.requested_fir != best_fir
    controller.evaluate(rejected.candidate, (_valid(2.8),))

    following = controller.propose_iterative(
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
    )

    assert following.candidate is not None
    assert calls[-2:] == [best_fir.taps, best_fir.taps]
    assert rejected.candidate.requested_fir.taps not in calls


@pytest.mark.parametrize(
    ("authorization", "reason_fragment"),
    [
        (
            _authorization(
                tap_ranges=(
                    (0, 0),
                    (-170, 170),
                    (-170, 170),
                    (104, 104),
                    (-170, 170),
                    (-170, 170),
                    (-170, 170),
                )
            ),
            "range",
        ),
        (_authorization(max_step=(1, 1, 1, 1, 1, 1, 1)), "max_step"),
        (_authorization(max_total_change_l1=1), "total"),
    ],
)
def test_formula_target_still_obeys_authorized_ranges_and_change_limits(
    authorization,
    reason_fragment,
):
    decision = _controller(authorization=authorization).propose_first(
        DCA_TAPS,
        dca_main_index=3,
    )

    assert decision.status is FirDecisionStatus.REJECTED_CONSTRAINT
    assert decision.candidate is None
    assert reason_fragment in decision.reason


@pytest.mark.parametrize(
    ("dca_taps", "main_index"),
    [
        (None, 3),
        (DCA_TAPS[:-1], 3),
        ((*DCA_TAPS[:-1], "bad"), 3),
        ((*DCA_TAPS[:-1], nan), 3),
        ((*DCA_TAPS[:-1], inf), 3),
        (DCA_TAPS, -1),
        (DCA_TAPS, len(DCA_TAPS)),
        ((0.0,) * len(DCA_TAPS), 3),
    ],
)
def test_invalid_dca_taps_fail_without_candidate_or_retry(dca_taps, main_index):
    formula_calls = 0

    def iterative_formula(*args, **kwargs):
        nonlocal formula_calls
        formula_calls += 1
        return calculate_iterative_pass(*args, **kwargs)

    controller = _controller(iterative_formula=iterative_formula)

    decision = controller.propose_first(dca_taps, dca_main_index=main_index)

    assert decision.status is FirDecisionStatus.FAILED_INVALID_DCA_TAPS
    assert decision.candidate is None
    assert decision.recommended_fir == controller.best_fir
    assert formula_calls <= 1


def test_a_better_first_measurement_requests_exactly_two_confirmations():
    controller = _controller()
    proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert proposal.candidate is not None

    decision = controller.evaluate(proposal.candidate, (_valid(2.9),))

    assert decision.status is FirDecisionStatus.NEEDS_CONFIRMATION
    assert decision.confirmation_remaining == 2
    assert decision.accepted is False
    assert decision.should_retune is False


def test_three_measurements_use_tdecq_median_and_record_er_median():
    controller = _controller(
        baseline_result=MetricResult.valid(tdecq_db=3.0, er_db=3.0)
    )
    proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert proposal.candidate is not None

    decision = _confirm(
        controller,
        proposal.candidate,
        (
            _valid(2.9, 3.2),
            _valid(2.0, 3.1),
            _valid(4.0, 3.3),
        ),
    )

    assert decision.status is FirDecisionStatus.ACCEPTED
    assert decision.aggregate_result is not None
    assert decision.aggregate_result.tdecq_db == pytest.approx(2.9)
    assert decision.aggregate_result.er_db == pytest.approx(3.2)
    assert decision.accepted is True
    assert decision.should_retune is True


def test_three_measurements_cannot_bypass_the_first_look_decision():
    controller = _controller()
    proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert proposal.candidate is not None

    with pytest.raises(ValueError, match="first measurement"):
        controller.evaluate(
            proposal.candidate,
            (_valid(2.9), _valid(2.0), _valid(2.1)),
        )


def test_first_pass_rejections_use_the_same_three_miss_convergence_rule():
    controller = _controller()
    statuses = []

    for expected_count in (1, 2, 3):
        proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
        assert proposal.candidate is not None
        decision = controller.evaluate(
            proposal.candidate,
            (_valid(3.1, 100.0),),
        )
        statuses.append(decision.status)
        assert decision.recommended_fir == _baseline()
        assert decision.no_improvement_count == expected_count
        assert decision.accepted is False
        assert decision.should_retune is False

    assert statuses == [
        FirDecisionStatus.REJECTED_FIRST_PASS,
        FirDecisionStatus.REJECTED_FIRST_PASS,
        FirDecisionStatus.CONVERGED_NO_IMPROVEMENT,
    ]
    assert controller.first_pass_stopped is False


def test_iterative_failure_recommends_historical_best_and_converges_after_three():
    controller = _controller()
    first = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert first.candidate is not None
    first_accept = _confirm(
        controller,
        first.candidate,
        (_valid(2.6), _valid(2.5), _valid(2.4)),
    )
    best_fir = first_accept.recommended_fir
    assert best_fir is not None

    statuses = []
    for expected_count in (1, 2, 3):
        proposal = controller.propose_iterative(
            ITERATIVE_DCA_TAPS,
            dca_main_index=3,
        )
        assert proposal.candidate is not None
        decision = controller.evaluate(proposal.candidate, (_valid(2.6),))
        statuses.append(decision.status)
        assert decision.recommended_fir == best_fir
        assert decision.no_improvement_count == expected_count
        assert decision.should_retune is False

    assert statuses == [
        FirDecisionStatus.REJECTED_ITERATIVE,
        FirDecisionStatus.REJECTED_ITERATIVE,
        FirDecisionStatus.CONVERGED_NO_IMPROVEMENT,
    ]


def test_an_accepted_improvement_resets_the_no_improvement_counter():
    controller = _controller()
    first = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert first.candidate is not None
    _confirm(
        controller,
        first.candidate,
        (_valid(2.8), _valid(2.7), _valid(2.6)),
    )

    rejected = controller.propose_iterative(
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
    )
    assert rejected.candidate is not None
    controller.evaluate(rejected.candidate, (_valid(2.8),))
    assert controller.no_improvement_count == 1

    improved = controller.propose_iterative(
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
    )
    assert improved.candidate is not None
    decision = _confirm(
        controller,
        improved.candidate,
        (_valid(2.5), _valid(2.4), _valid(2.3)),
    )

    assert decision.status is FirDecisionStatus.ACCEPTED
    assert decision.no_improvement_count == 0
    assert controller.no_improvement_count == 0


def test_iterative_fir_compares_against_a_better_post_fir_retune_result():
    controller = _controller()
    first = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert first.candidate is not None
    accepted = _confirm(
        controller,
        first.candidate,
        (_valid(2.9), _valid(2.8), _valid(2.7)),
    )
    assert accepted.status is FirDecisionStatus.ACCEPTED
    assert controller.best_result.tdecq_db == pytest.approx(2.8)

    changed = controller.record_retune_result(
        _retune_observation(controller, _valid(2.4, 2.0))
    )
    proposal = controller.propose_iterative(
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
    )
    assert proposal.candidate is not None
    decision = controller.evaluate(proposal.candidate, (_valid(2.6, 100.0),))

    assert changed is True
    assert controller.best_result.tdecq_db == pytest.approx(2.4)
    assert decision.status is FirDecisionStatus.REJECTED_ITERATIVE
    assert decision.recommended_fir == accepted.recommended_fir


def test_record_retune_result_rejects_a_naked_metric_result():
    controller = _controller()
    _accept_first(controller)

    with pytest.raises(TypeError, match="I2CObservation|AggregatedObservation"):
        controller.record_retune_result(_valid(2.4))


@pytest.mark.parametrize("aggregated", [False, True])
def test_record_retune_result_accepts_only_the_same_context(aggregated):
    controller = _controller()
    _accept_first(controller)

    changed = controller.record_retune_result(
        _retune_observation(
            controller,
            _valid(2.4, 2.0),
            aggregated=aggregated,
        )
    )

    assert changed is True
    assert controller.best_result == _valid(2.4, 2.0)


@pytest.mark.parametrize(
    "difference",
    ["fir_taps", "upper_nlc", "lower_nlc", "dca_config_fingerprint"],
)
def test_record_retune_result_rejects_mismatched_experiment_context(difference):
    controller = _controller()
    _accept_first(controller)
    expected = controller.retune_fingerprint
    changes = {
        "fir_taps": (*expected.fir_taps[:-1], expected.fir_taps[-1] + 1),
        "upper_nlc": expected.upper_nlc + 1,
        "lower_nlc": expected.lower_nlc + 1,
        "dca_config_fingerprint": "b" * 64,
    }
    mismatched = expected.model_copy(update={difference: changes[difference]})
    previous = controller.best_result

    with pytest.raises(ValueError, match="fingerprint"):
        controller.record_retune_result(
            _retune_observation(
                controller,
                _valid(2.0, 100.0),
                fingerprint=mismatched,
            )
        )

    assert controller.best_result == previous


def test_record_retune_result_rejects_an_active_fir_candidate():
    controller = _controller()
    _accept_first(controller)
    proposal = controller.propose_iterative(
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
    )
    assert proposal.status is FirDecisionStatus.CANDIDATE

    with pytest.raises(ValueError, match="active|pending"):
        controller.record_retune_result(
            _retune_observation(controller, _valid(2.4))
        )


def test_record_retune_result_rejects_a_pending_fir_confirmation():
    controller = _controller()
    _accept_first(controller)
    proposal = controller.propose_iterative(
        ITERATIVE_DCA_TAPS,
        dca_main_index=3,
    )
    assert proposal.candidate is not None
    pending = controller.evaluate(proposal.candidate, (_valid(2.4),))
    assert pending.status is FirDecisionStatus.NEEDS_CONFIRMATION

    with pytest.raises(ValueError, match="active|pending"):
        controller.record_retune_result(
            _retune_observation(controller, _valid(2.3))
        )


def test_same_context_retune_uses_tdecq_only_objective():
    controller = _controller()
    _accept_first(controller)

    exact_tdecq_higher_er = _valid(2.8, 3.1)
    assert controller.record_retune_result(
        _retune_observation(controller, exact_tdecq_higher_er)
    ) is False

    lower_tdecq_lower_er = _valid(2.7, 1.0)
    assert controller.record_retune_result(
        _retune_observation(controller, lower_tdecq_lower_er)
    ) is True

    worse_tdecq_higher_er = _valid(2.9, 100.0)
    assert controller.record_retune_result(
        _retune_observation(controller, worse_tdecq_higher_er)
    ) is False
    assert controller.best_result == lower_tdecq_lower_er


def test_confirmation_rejects_if_any_tdecq_is_invalid():
    controller = _controller()
    proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert proposal.candidate is not None
    invalid = MetricResult.invalid(status="INV", reason="invalid DCA acquisition")

    decision = _confirm(
        controller,
        proposal.candidate,
        (_valid(2.5), invalid, _valid(2.4)),
    )

    assert decision.status is FirDecisionStatus.REJECTED_FIRST_PASS
    assert decision.aggregate_result is not None
    assert decision.aggregate_result.tdecq_db is None


def test_evaluation_accepts_only_one_or_three_measurements():
    controller = _controller()
    proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert proposal.candidate is not None

    with pytest.raises(ValueError, match="one or three"):
        controller.evaluate(proposal.candidate, (_valid(2.5), _valid(2.4)))


def test_evaluation_rejects_a_candidate_not_issued_by_this_controller():
    controller = _controller()
    proposal = controller.propose_first(DCA_TAPS, dca_main_index=3)
    assert proposal.candidate is not None
    tampered_target = proposal.candidate.requested_fir.model_copy(
        update={"upper_nlc": 9, "lower_nlc": 9}
    )
    tampered = proposal.candidate.model_copy(
        update={"requested_fir": tampered_target}
    )

    with pytest.raises(ValueError, match="issued"):
        controller.evaluate(tampered, (_valid(2.5),))


def test_repeated_proposal_request_returns_the_active_candidate_without_recalling_formula():
    calls = 0

    def iterative_formula(current_taps, dca_taps, dca_main_index, manual_main):
        nonlocal calls
        calls += 1
        return calculate_iterative_pass(
            current_taps,
            dca_taps,
            dca_main_index,
            manual_main,
        )

    controller = _controller(iterative_formula=iterative_formula)

    first = controller.propose_first(DCA_TAPS, dca_main_index=3)
    repeated = controller.propose_first(DCA_TAPS, dca_main_index=3)

    assert calls == 1
    assert repeated == first
