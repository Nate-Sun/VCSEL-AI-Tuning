from __future__ import annotations

from hashlib import sha256
import json
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
    """Stateless deterministic TDECQ-only coordinate fallback."""

    def __init__(
        self,
        space: IntegerParameterSpace,
        context: OptimizationContext,
        seed: int,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("coordinate seed must be an integer, not bool")

        self._space = space
        self._context = context
        self._seed = seed
        self._points = tuple(space.enumerate_points())
        self._keys = tuple(space.key(point) for point in self._points)
        self._index_by_key = {
            key: index for index, key in enumerate(self._keys)
        }

    def best(
        self,
        observations: tuple[TuningObservation, ...],
    ) -> TuningObservation | None:
        measured = validated_observation_map(
            self._space,
            self._context,
            observations,
        )
        return self._best_observation(measured)

    def suggest(
        self,
        observations: tuple[TuningObservation, ...],
        pending_keys: frozenset[PointKey] = frozenset(),
    ) -> CandidateSuggestion | NoSafeCandidate | PathComplete:
        measured = validated_observation_map(
            self._space,
            self._context,
            observations,
        )
        pending = self._validated_pending_keys(pending_keys)
        history_digest = self._history_digest(measured)
        measured_keys = frozenset(measured)
        available_keys = tuple(
            key
            for key in self._keys
            if key not in measured_keys and key not in pending
        )

        if len(measured) == self._space.candidate_count:
            return self._complete(measured)

        if not available_keys:
            remaining_count = self._space.candidate_count - len(measured)
            return NoSafeCandidate(
                path=TuningPath.TDECQ_BAYESIAN,
                reason=(
                    f"All {remaining_count} unmeasured safe endpoints are pending; "
                    "no endpoint can be suggested until pending work resolves."
                ),
            )

        if not observations:
            return self._suggestion(
                available_keys[0],
                history_length=0,
                history_digest=history_digest,
                reason="deterministic canonical bootstrap",
            )

        incumbent = self._best_observation(measured)
        if incumbent is None:
            center = observations[-1]
            center_reason = "chronological invalid-TDECQ fallback readback"
        else:
            center = incumbent
            center_reason = "best valid-TDECQ readback"

        available_set = frozenset(available_keys)
        local_keys = tuple(
            sorted(
                self._space.key(point)
                for point in self._space.local_neighbors(
                    center.settings_readback
                )
                if self._space.key(point) in available_set
            )
        )
        if local_keys:
            selected_key = Random(self._seed + len(observations)).choice(
                local_keys
            )
            return self._suggestion(
                selected_key,
                history_length=len(observations),
                history_digest=history_digest,
                reason=(
                    "deterministic seeded local coordinate step from "
                    f"{center_reason}"
                ),
            )

        return self._suggestion(
            available_keys[0],
            history_length=len(observations),
            history_digest=history_digest,
            reason=(
                "deterministic global restart from the first canonical "
                "available endpoint after local coordinate exhaustion"
            ),
        )

    def _complete(
        self,
        measured: dict[PointKey, TuningObservation],
    ) -> PathComplete:
        incumbent = self._best_observation(measured)
        best_fields = (
            {}
            if incumbent is None
            else {
                "best_observation_id": incumbent.observation_id,
                "best_tdecq_db": incumbent.tdecq_db,
            }
        )
        return PathComplete(
            path=TuningPath.TDECQ_BAYESIAN,
            status=(
                "NO_VALID_TDECQ"
                if incumbent is None
                else "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
            ),
            measured_count=len(measured),
            candidate_count=self._space.candidate_count,
            **best_fields,
        )

    def _suggestion(
        self,
        key: PointKey,
        *,
        history_length: int,
        history_digest: str,
        reason: str,
    ) -> CandidateSuggestion:
        index = self._index_by_key[key]
        return CandidateSuggestion(
            path=TuningPath.TDECQ_BAYESIAN,
            candidate_id=(
                f"coordinate-{history_length:08d}-endpoint-{index:08d}-"
                f"history-{history_digest}"
            ),
            requested=self._space.point(key),
            reason=reason,
        )

    def _validated_pending_keys(
        self,
        pending_keys: frozenset[PointKey],
    ) -> frozenset[PointKey]:
        validated: set[PointKey] = set()
        for key in pending_keys:
            try:
                point = self._space.point(key)
            except ValueError as error:
                raise ValueError(
                    f"pending key {key!r} is not a valid safe key: {error}"
                ) from error
            validated.add(self._space.key(point))
        return frozenset(validated)

    @staticmethod
    def _history_digest(
        measured: dict[PointKey, TuningObservation],
    ) -> str:
        authoritative_history = [
            {
                "key": list(key),
                "observation_id": observation.observation_id,
                "candidate_id": observation.candidate_id,
                "tdecq_status": observation.tdecq_status,
                "tdecq_db": observation.tdecq_db,
            }
            for key, observation in sorted(measured.items())
        ]
        canonical = json.dumps(
            authoritative_history,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(canonical).hexdigest()[:16]

    @staticmethod
    def _best_observation(
        measured: dict[PointKey, TuningObservation],
    ) -> TuningObservation | None:
        eligible = (
            (observation, key)
            for key, observation in measured.items()
            if observation.tdecq_status == "CORR"
            and observation.tdecq_db is not None
        )
        ranked = sorted(
            eligible,
            key=lambda item: (
                item[0].tdecq_db,
                item[0].observation_id,
                item[1],
            ),
        )
        return None if not ranked else ranked[0][0]
