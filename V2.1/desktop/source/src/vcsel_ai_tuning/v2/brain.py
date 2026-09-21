"""Hardware-neutral, fail-closed boundary for V2 optimization providers."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from threading import RLock
from time import perf_counter
from typing import Callable, NamedTuple, Protocol, runtime_checkable
from types import MethodType
from unicodedata import category
from weakref import WeakKeyDictionary

from .contracts import (
    BrainCandidate,
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    EyeSample,
    MeasurementStatus,
    PerceptionResult,
    V2Observation,
)


_GUARD_PROVIDER_ID = "brain-guard"
_GUARD_PROVIDER_VERSION = "1"


@runtime_checkable
class OptimizationBrain(Protocol):
    """The sole read-only analysis boundary consumed by V2 orchestration."""

    @property
    def provider_key(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> BrainResult: ...


@runtime_checkable
class EyeScorer(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def score(self, sample: EyeSample) -> PerceptionResult: ...


@runtime_checkable
class DecisionProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def propose(
        self,
        sample: EyeSample,
        perception: PerceptionResult,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> DecisionProposal: ...


@runtime_checkable
class FusedBrainProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze_once(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> tuple[PerceptionResult, DecisionProposal]: ...


def _validate_minimum_confidence(value: float) -> float:
    if type(value) not in (int, float):
        raise TypeError("minimum_confidence must be a finite numeric probability")
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("minimum_confidence must be a finite numeric probability")
    return float(value)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _request_hash(
    sample: EyeSample,
    history: tuple[V2Observation, ...],
    safe_candidates: tuple[BrainCandidate, ...],
) -> str:
    return _canonical_hash(
        {
            "sample": sample.model_dump(mode="json"),
            "history": [item.model_dump(mode="json") for item in history],
            "safe_candidates": [item.model_dump(mode="json") for item in safe_candidates],
        }
    )


def _strict_clone(model_type, value):
    return model_type.model_validate(value.model_dump(round_trip=True, warnings=False), strict=True)


def _detached_inputs(
    sample: EyeSample,
    history: tuple[V2Observation, ...],
    safe_candidates: tuple[BrainCandidate, ...],
) -> tuple[EyeSample, tuple[V2Observation, ...], tuple[BrainCandidate, ...]]:
    """Return strict deep copies so providers cannot mutate gate inputs."""
    return (
        _strict_clone(EyeSample, sample),
        tuple(_strict_clone(V2Observation, item) for item in history),
        tuple(_strict_clone(BrainCandidate, item) for item in safe_candidates),
    )


def _response_hash(
    perception: PerceptionResult,
    decision: DecisionProposal,
    mode: BrainMode,
) -> str:
    return _canonical_hash(
        {
            "perception": perception.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "brain_mode": mode.value,
        }
    )


def _guard_perception(sample: EyeSample) -> PerceptionResult:
    return PerceptionResult(
        measurement_id=sample.measurement_id,
        quality_score=0.0,
        confidence=0.0,
        valid=False,
        ood=True,
        defects={},
        provider_id=_GUARD_PROVIDER_ID,
        provider_version=_GUARD_PROVIDER_VERSION,
        input_hash=sample.raw_eye_density_ref.sha256,
    )


def _guard_decision(sample: EyeSample, reason: str) -> DecisionProposal:
    return DecisionProposal(
        based_on_measurement_id=sample.measurement_id,
        action=DecisionAction.HOLD,
        reason_codes=(reason,),
        provider_id=_GUARD_PROVIDER_ID,
        provider_version=_GUARD_PROVIDER_VERSION,
    )


def _hold_result(
    sample: EyeSample,
    mode: BrainMode,
    request_hash: str,
    reason: str,
    started_at: float,
    perception: PerceptionResult | None = None,
) -> BrainResult:
    safe_perception = perception if perception is not None else _guard_perception(sample)
    decision = _guard_decision(sample, reason)
    return BrainResult(
        perception=safe_perception,
        decision=decision,
        brain_mode=mode,
        latency_ms=max(0.0, (perf_counter() - started_at) * 1000),
        request_hash=request_hash,
        response_hash=_response_hash(safe_perception, decision, mode),
    )


def _candidate_precheck(safe_candidates: tuple[BrainCandidate, ...]) -> str | None:
    if not safe_candidates:
        return "NO_SAFE_CANDIDATE"
    candidate_ids = tuple(candidate.candidate_id for candidate in safe_candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        return "DUPLICATE_CANDIDATE_ID"
    return None


def _canonical_perception(value: object) -> PerceptionResult | None:
    if not isinstance(value, PerceptionResult):
        return None
    try:
        return PerceptionResult.model_validate(
            value.model_dump(round_trip=True, warnings=False), strict=True
        )
    except Exception:
        return None


def _canonical_decision(value: object) -> DecisionProposal | None:
    if not isinstance(value, DecisionProposal):
        return None
    try:
        return DecisionProposal.model_validate(
            value.model_dump(round_trip=True, warnings=False), strict=True
        )
    except Exception:
        return None


def _trusted_perception(
    perception: object,
    sample: EyeSample,
    expected_provider_id: str | None = None,
    expected_provider_version: str | None = None,
) -> PerceptionResult | None:
    perception = _canonical_perception(perception)
    if perception is None:
        return None
    try:
        if (
            perception.measurement_id == sample.measurement_id
            and perception.input_hash == sample.raw_eye_density_ref.sha256
            and (expected_provider_id is None or perception.provider_id == expected_provider_id)
            and (
                expected_provider_version is None
                or perception.provider_version == expected_provider_version
            )
        ):
            return perception
    except Exception:
        return None
    return None


def _valid_identity(value: object) -> str | None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "|" in value
        or any(category(character) in {"Cc", "Cf", "Cs"} for character in value)
    ):
        return None
    return value


def _provider_snapshot(*providers: object) -> tuple[str, str, bool, tuple[str, ...], tuple[str, ...]]:
    """Read provider metadata once, then retain both public and component identities."""
    try:
        identities = tuple(_valid_identity(provider.provider_id) for provider in providers)
        versions = tuple(_valid_identity(provider.provider_version) for provider in providers)
    except Exception:
        return _GUARD_PROVIDER_ID, _GUARD_PROVIDER_VERSION, False, (), ()
    if any(value is None for value in identities) or any(value is None for value in versions):
        return _GUARD_PROVIDER_ID, _GUARD_PROVIDER_VERSION, False, (), ()
    # Keeping concrete tuples prevents later mutable provider metadata from
    # changing the contract.
    return "|".join(identities), "|".join(versions), True, identities, versions


def _perception_failure(
    perception: object,
    sample: EyeSample,
    minimum_confidence: float,
    expected_provider_id: str | None = None,
    expected_provider_version: str | None = None,
) -> str | None:
    perception = _canonical_perception(perception)
    if perception is None:
        return "INVALID_PERCEPTION_SCHEMA"
    try:
        if perception.measurement_id != sample.measurement_id:
            return "MEASUREMENT_ID_MISMATCH"
        if (
            (expected_provider_id is not None and perception.provider_id != expected_provider_id)
            or (
                expected_provider_version is not None
                and perception.provider_version != expected_provider_version
            )
        ):
            return "PROVIDER_IDENTITY_MISMATCH"
        if perception.input_hash != sample.raw_eye_density_ref.sha256:
            return "INPUT_HASH_MISMATCH"
        if not perception.valid:
            return "INVALID_EYE"
        if perception.ood:
            return "OOD"
        if perception.confidence < minimum_confidence:
            return "LOW_CONFIDENCE"
    except Exception:
        return "INVALID_PERCEPTION_SCHEMA"
    return None


def _proposal_matches_input(
    decision: object,
    safe_candidates: tuple[BrainCandidate, ...],
) -> bool:
    if not isinstance(decision, DecisionProposal):
        return False
    if decision.action is not DecisionAction.PROPOSE:
        return True
    candidate = next(
        (item for item in safe_candidates if item.candidate_id == decision.candidate_id),
        None,
    )
    return candidate is not None and _canonical_json(
        decision.model_dump(mode="json")["requested_params"]
    ) == _canonical_json(candidate.model_dump(mode="json")["requested_params"])


def _decision_failure(
    decision: object,
    sample: EyeSample,
    safe_candidates: tuple[BrainCandidate, ...],
    expected_provider_id: str | None = None,
    expected_provider_version: str | None = None,
) -> str | None:
    decision = _canonical_decision(decision)
    if decision is None:
        return "INVALID_DECISION_SCHEMA"
    try:
        if decision.based_on_measurement_id != sample.measurement_id:
            return "MEASUREMENT_ID_MISMATCH"
        if (
            (expected_provider_id is not None and decision.provider_id != expected_provider_id)
            or (
                expected_provider_version is not None
                and decision.provider_version != expected_provider_version
            )
        ):
            return "PROVIDER_IDENTITY_MISMATCH"
        if not _proposal_matches_input(decision, safe_candidates):
            return "CANDIDATE_NOT_IN_INPUT_SET"
    except Exception:
        return "INVALID_DECISION_SCHEMA"
    return None


def _finalize(
    sample: EyeSample,
    mode: BrainMode,
    request_hash: str,
    started_at: float,
    minimum_confidence: float,
    safe_candidates: tuple[BrainCandidate, ...],
    perception: object,
    decision: object,
    perception_provider_id: str | None = None,
    perception_provider_version: str | None = None,
    decision_provider_id: str | None = None,
    decision_provider_version: str | None = None,
) -> BrainResult:
    perception = _canonical_perception(perception)
    decision = _canonical_decision(decision)
    trusted_perception = _trusted_perception(
        perception, sample, perception_provider_id, perception_provider_version
    )
    perception_reason = _perception_failure(
        perception, sample, minimum_confidence, perception_provider_id, perception_provider_version
    )
    if perception_reason is not None:
        return _hold_result(
            sample,
            mode,
            request_hash,
            perception_reason,
            started_at,
            trusted_perception,
        )
    decision_reason = _decision_failure(
        decision, sample, safe_candidates, decision_provider_id, decision_provider_version
    )
    if decision_reason is not None:
        return _hold_result(
            sample,
            mode,
            request_hash,
            decision_reason,
            started_at,
            trusted_perception,
        )
    assert perception is not None
    assert decision is not None
    return BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=mode,
        latency_ms=max(0.0, (perf_counter() - started_at) * 1000),
        request_hash=request_hash,
        response_hash=_response_hash(perception, decision, mode),
    )


class ComposedBrain:
    """Local scorer and decision provider with a fail-closed hand-off."""

    def __init__(
        self,
        mode: BrainMode,
        scorer: EyeScorer,
        decision_provider: DecisionProvider,
        minimum_confidence: float,
    ) -> None:
        if not isinstance(mode, BrainMode):
            raise TypeError("mode must be a BrainMode")
        if mode is BrainMode.PURE_API:
            raise ValueError("ComposedBrain does not permit PURE_API mode")
        self._mode = mode
        self._scorer = scorer
        self._decision_provider = decision_provider
        self._minimum_confidence = _validate_minimum_confidence(minimum_confidence)
        (
            self._provider_key,
            self._provider_version,
            self._identity_valid,
            self._provider_ids,
            self._provider_versions,
        ) = _provider_snapshot(scorer, decision_provider)

    @property
    def provider_key(self) -> str:
        return self._provider_key

    @property
    def provider_version(self) -> str:
        return self._provider_version

    def analyze(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> BrainResult:
        started_at = perf_counter()
        gate_sample, gate_history, gate_candidates = _detached_inputs(
            sample, history, safe_candidates
        )
        request_hash = _request_hash(gate_sample, gate_history, gate_candidates)
        if not self._identity_valid:
            return _hold_result(gate_sample, self._mode, request_hash, "PROVIDER_ERROR", started_at)
        precheck_reason = _candidate_precheck(gate_candidates)
        if precheck_reason is not None:
            return _hold_result(gate_sample, self._mode, request_hash, precheck_reason, started_at)
        if gate_sample.measurement_status is not MeasurementStatus.VALID:
            return _hold_result(gate_sample, self._mode, request_hash, "INVALID_EYE", started_at)
        call_sample = _strict_clone(EyeSample, gate_sample)
        try:
            perception = _canonical_perception(self._scorer.score(call_sample))
        except Exception:
            return _hold_result(gate_sample, self._mode, request_hash, "PROVIDER_ERROR", started_at)
        perception_reason = _perception_failure(
            perception,
            gate_sample,
            self._minimum_confidence,
            self._provider_ids[0],
            self._provider_versions[0],
        )
        if perception_reason is not None:
            return _hold_result(
                gate_sample,
                self._mode,
                request_hash,
                perception_reason,
                started_at,
                _trusted_perception(
                    perception, gate_sample, self._provider_ids[0], self._provider_versions[0]
                ),
            )
        assert isinstance(perception, PerceptionResult)
        gate_perception = perception
        decision_perception = _canonical_perception(gate_perception)
        decision_sample, decision_history, decision_candidates = _detached_inputs(
            gate_sample, gate_history, gate_candidates
        )
        assert decision_perception is not None
        try:
            decision = self._decision_provider.propose(
                decision_sample, decision_perception, decision_history, decision_candidates
            )
        except Exception:
            return _hold_result(
                gate_sample,
                self._mode,
                request_hash,
                "PROVIDER_ERROR",
                started_at,
                gate_perception,
            )
        return _finalize(
            gate_sample,
            self._mode,
            request_hash,
            started_at,
            self._minimum_confidence,
            gate_candidates,
            gate_perception,
            decision,
            self._provider_ids[0],
            self._provider_versions[0],
            self._provider_ids[1],
            self._provider_versions[1],
        )


class MultimodalApiBrain:
    """A fused provider that is called once and shares final validation."""

    def __init__(self, provider: FusedBrainProvider, minimum_confidence: float) -> None:
        self._provider = provider
        self._minimum_confidence = _validate_minimum_confidence(minimum_confidence)
        (
            self._provider_key,
            self._provider_version,
            self._identity_valid,
            self._provider_ids,
            self._provider_versions,
        ) = _provider_snapshot(provider)

    @property
    def provider_key(self) -> str:
        return self._provider_key

    @property
    def provider_version(self) -> str:
        return self._provider_version

    def analyze(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> BrainResult:
        started_at = perf_counter()
        gate_sample, gate_history, gate_candidates = _detached_inputs(
            sample, history, safe_candidates
        )
        request_hash = _request_hash(gate_sample, gate_history, gate_candidates)
        if not self._identity_valid:
            return _hold_result(gate_sample, BrainMode.PURE_API, request_hash, "PROVIDER_ERROR", started_at)
        precheck_reason = _candidate_precheck(gate_candidates)
        if precheck_reason is not None:
            return _hold_result(gate_sample, BrainMode.PURE_API, request_hash, precheck_reason, started_at)
        if gate_sample.measurement_status is not MeasurementStatus.VALID:
            return _hold_result(gate_sample, BrainMode.PURE_API, request_hash, "INVALID_EYE", started_at)
        call_sample, call_history, call_candidates = _detached_inputs(
            gate_sample, gate_history, gate_candidates
        )
        try:
            output = self._provider.analyze_once(call_sample, call_history, call_candidates)
        except Exception:
            return _hold_result(gate_sample, BrainMode.PURE_API, request_hash, "PROVIDER_ERROR", started_at)
        if not isinstance(output, tuple) or len(output) != 2:
            return _hold_result(gate_sample, BrainMode.PURE_API, request_hash, "INVALID_PERCEPTION_SCHEMA", started_at)
        perception, decision = output
        return _finalize(
            gate_sample,
            BrainMode.PURE_API,
            request_hash,
            started_at,
            self._minimum_confidence,
            gate_candidates,
            perception,
            decision,
            self._provider_ids[0],
            self._provider_versions[0],
            self._provider_ids[0],
            self._provider_versions[0],
        )


def _trusted_builtin_profile(
    source: object,
    analyze_callable,
    provider_key: str,
    provider_version: str,
):
    """Return immutable component provenance only for an untouched built-in method."""
    identities = tuple(provider_key.split("|"))
    versions = tuple(provider_version.split("|"))
    if (
        _is_native_builtin_callable(source, analyze_callable)
        and type(source) is ComposedBrain
    ):
        if len(identities) == len(versions) == 2:
            return identities[0], versions[0], identities[1], versions[1]
    if (
        _is_native_builtin_callable(source, analyze_callable)
        and type(source) is MultimodalApiBrain
    ):
        if len(identities) == len(versions) == 1:
            return identities[0], versions[0], identities[0], versions[0]
    return None


def _is_native_builtin_callable(source: object, analyze_callable) -> bool:
    if type(analyze_callable) is not MethodType:
        return False
    expected = (
        ComposedBrain.analyze
        if type(source) is ComposedBrain
        else MultimodalApiBrain.analyze
        if type(source) is MultimodalApiBrain
        else None
    )
    return (
        expected is not None
        and analyze_callable.__self__ is source
        and analyze_callable.__func__ is expected
    )


def _is_guard_perception(perception: PerceptionResult, sample: EyeSample) -> bool:
    return perception == _guard_perception(sample)


def _is_guard_decision(decision: DecisionProposal, sample: EyeSample) -> bool:
    return (
        decision.based_on_measurement_id == sample.measurement_id
        and decision.action is DecisionAction.HOLD
        and decision.candidate_id is None
        and decision.requested_params is None
        and decision.provider_id == _GUARD_PROVIDER_ID
        and decision.provider_version == _GUARD_PROVIDER_VERSION
    )


def _trusted_builtin_hold_failure(sample: EyeSample, result: BrainResult, profile) -> str | None:
    perception_id, perception_version, decision_id, decision_version = profile
    perception = result.perception
    decision = result.decision
    if (
        perception.measurement_id != sample.measurement_id
        or decision.based_on_measurement_id != sample.measurement_id
    ):
        return "MEASUREMENT_ID_MISMATCH"
    if perception.input_hash != sample.raw_eye_density_ref.sha256:
        return "INPUT_HASH_MISMATCH"
    trusted_perception = (
        perception.provider_id == perception_id
        and perception.provider_version == perception_version
    )
    trusted_decision = (
        decision.provider_id == decision_id
        and decision.provider_version == decision_version
    )
    if not (trusted_perception or _is_guard_perception(perception, sample)):
        return "PROVIDER_IDENTITY_MISMATCH"
    if not (trusted_decision or _is_guard_decision(decision, sample)):
        return "PROVIDER_IDENTITY_MISMATCH"
    return None


class _FactoryGateState(NamedTuple):
    mode: BrainMode
    minimum_confidence: float
    provider_key: str
    provider_version: str
    analyze_callable: Callable[
        [EyeSample, tuple[V2Observation, ...], tuple[BrainCandidate, ...]],
        BrainResult,
    ]
    native_builtin: bool
    built_in_profile: tuple[str, str, str, str] | None


def _new_factory_gate_state_accessors():
    states = WeakKeyDictionary()
    lock = RLock()

    def register(brain: object, state: _FactoryGateState) -> None:
        if type(brain) is not _FactoryGatedBrain:
            raise TypeError("brain must be created by BrainFactory")
        with lock:
            if brain in states:
                raise RuntimeError("factory-gated brain state is already registered")
            states[brain] = state

    def get(brain: object) -> _FactoryGateState:
        if type(brain) is not _FactoryGatedBrain:
            raise TypeError("brain must be created by BrainFactory")
        with lock:
            try:
                return states[brain]
            except (KeyError, TypeError):
                raise TypeError("brain must be created by BrainFactory") from None

    return register, get


_register_factory_gate_state, _get_factory_gate_state = (
    _new_factory_gate_state_accessors()
)
del _new_factory_gate_state_accessors


def _new_factory_gate_enrollment():
    enrollment_token = object()

    def validate(candidate: object) -> bool:
        return candidate is enrollment_token

    def create(
        source: OptimizationBrain,
        mode: BrainMode,
        minimum_confidence: float,
        provider_key: str,
        provider_version: str,
        analyze_callable,
    ) -> "_FactoryGatedBrain":
        return _FactoryGatedBrain(
            source,
            mode,
            minimum_confidence,
            provider_key,
            provider_version,
            analyze_callable,
            _enrollment_token=enrollment_token,
        )

    return validate, create


_validate_factory_gate_enrollment, _create_factory_gated_brain = (
    _new_factory_gate_enrollment()
)
del _new_factory_gate_enrollment


def _claim_factory_gate_creator():
    """Transfer the one-time enrollment creator to the local factory module."""
    creator = _create_factory_gated_brain
    del globals()["_create_factory_gated_brain"]
    del globals()["_claim_factory_gate_creator"]
    return creator


class _FactoryGatedBrain:
    """Factory-only outer validation for arbitrary OptimizationBrain builders."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        source: OptimizationBrain,
        mode: BrainMode,
        minimum_confidence: float,
        provider_key: str,
        provider_version: str,
        analyze_callable,
        *,
        _enrollment_token: object | None = None,
    ) -> None:
        if not _validate_factory_gate_enrollment(_enrollment_token):
            raise TypeError("factory-gated brain enrollment capability is required")
        _register_factory_gate_state(
            self,
            _FactoryGateState(
                mode=mode,
                minimum_confidence=minimum_confidence,
                provider_key=provider_key,
                provider_version=provider_version,
                analyze_callable=analyze_callable,
                native_builtin=_is_native_builtin_callable(source, analyze_callable),
                built_in_profile=_trusted_builtin_profile(
                    source, analyze_callable, provider_key, provider_version
                ),
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("factory-gated brain is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("factory-gated brain is immutable")

    @property
    def provider_key(self) -> str:
        return _get_factory_gate_state(self).provider_key

    @property
    def provider_version(self) -> str:
        return _get_factory_gate_state(self).provider_version

    def analyze(
        self,
        sample: EyeSample,
        history: tuple[V2Observation, ...],
        safe_candidates: tuple[BrainCandidate, ...],
    ) -> BrainResult:
        state = _get_factory_gate_state(self)
        started_at = perf_counter()
        gate_sample, gate_history, gate_candidates = _detached_inputs(
            sample, history, safe_candidates
        )
        request_hash = _request_hash(gate_sample, gate_history, gate_candidates)
        precheck_reason = _candidate_precheck(gate_candidates)
        if precheck_reason is not None:
            return _hold_result(gate_sample, state.mode, request_hash, precheck_reason, started_at)
        if gate_sample.measurement_status is not MeasurementStatus.VALID:
            return _hold_result(gate_sample, state.mode, request_hash, "INVALID_EYE", started_at)
        call_sample, call_history, call_candidates = _detached_inputs(
            gate_sample, gate_history, gate_candidates
        )
        try:
            raw_result = state.analyze_callable(call_sample, call_history, call_candidates)
        except Exception:
            return _hold_result(gate_sample, state.mode, request_hash, "PROVIDER_ERROR", started_at)
        if not isinstance(raw_result, BrainResult):
            return _hold_result(gate_sample, state.mode, request_hash, "INVALID_BRAIN_RESULT_SCHEMA", started_at)
        try:
            result = BrainResult.model_validate(
                raw_result.model_dump(round_trip=True, warnings=False), strict=True
            )
        except Exception:
            return _hold_result(gate_sample, state.mode, request_hash, "INVALID_BRAIN_RESULT_SCHEMA", started_at)
        if result.brain_mode is not state.mode:
            return _hold_result(gate_sample, state.mode, request_hash, "BRAIN_MODE_MISMATCH", started_at)
        if result.request_hash != request_hash:
            return _hold_result(gate_sample, state.mode, request_hash, "REQUEST_HASH_MISMATCH", started_at)
        if result.response_hash != _response_hash(
            result.perception, result.decision, state.mode
        ):
            return _hold_result(gate_sample, state.mode, request_hash, "RESPONSE_HASH_MISMATCH", started_at)

        profile = state.built_in_profile
        if state.native_builtin and profile is None:
            if result.decision.action is DecisionAction.HOLD and _is_guard_perception(
                result.perception, gate_sample
            ) and _is_guard_decision(result.decision, gate_sample):
                return BrainResult(
                    perception=result.perception,
                    decision=result.decision,
                    brain_mode=state.mode,
                    latency_ms=max(0.0, (perf_counter() - started_at) * 1000),
                    request_hash=request_hash,
                    response_hash=_response_hash(result.perception, result.decision, state.mode),
                )
            return _hold_result(
                gate_sample,
                state.mode,
                request_hash,
                "PROVIDER_IDENTITY_MISMATCH",
                started_at,
            )
        if profile is not None and result.decision.action is DecisionAction.HOLD:
            hold_failure = _trusted_builtin_hold_failure(gate_sample, result, profile)
            if hold_failure is not None:
                return _hold_result(
                    gate_sample,
                    state.mode,
                    request_hash,
                    hold_failure,
                    started_at,
                )
            return BrainResult(
                perception=result.perception,
                decision=result.decision,
                brain_mode=state.mode,
                latency_ms=max(0.0, (perf_counter() - started_at) * 1000),
                request_hash=request_hash,
                response_hash=_response_hash(result.perception, result.decision, state.mode),
            )
        if profile is None:
            perception_id = decision_id = state.provider_key
            perception_version = decision_version = state.provider_version
        else:
            perception_id, perception_version, decision_id, decision_version = profile
        return _finalize(
            gate_sample,
            state.mode,
            request_hash,
            started_at,
            state.minimum_confidence,
            gate_candidates,
            result.perception,
            result.decision,
            perception_id,
            perception_version,
            decision_id,
            decision_version,
        )


def is_factory_gated_brain(value: object) -> bool:
    """Return whether value is the exact non-spoofable BrainFactory wrapper."""
    if type(value) is not _FactoryGatedBrain:
        return False
    try:
        _get_factory_gate_state(value)
    except TypeError:
        return False
    return True


def analyze_factory_gated(
    brain: object,
    sample: EyeSample,
    history: tuple[V2Observation, ...],
    safe_candidates: tuple[BrainCandidate, ...],
) -> BrainResult:
    """Invoke the exact factory gate without overridable instance dispatch."""
    _get_factory_gate_state(brain)
    return _FactoryGatedBrain.analyze(brain, sample, history, safe_candidates)
