from __future__ import annotations

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
    """Stateless exhaustive search over a canonical finite endpoint grid."""

    def __init__(
        self,
        space: IntegerParameterSpace,
        context: OptimizationContext,
    ) -> None:
        self._space = space
        self._context = context
        self._points = tuple(space.enumerate_points())
        self._keys = tuple(space.key(point) for point in self._points)

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

        for index, (point, key) in enumerate(zip(self._points, self._keys, strict=True)):
            if key in measured or key in pending:
                continue
            return CandidateSuggestion(
                path=TuningPath.EXHAUSTIVE,
                candidate_id=f"exhaustive-{index:08d}",
                requested=point,
                reason=(
                    "First canonical endpoint that is neither measured by actual "
                    f"readback nor pending ({index + 1} of {self._space.candidate_count})."
                ),
            )

        if len(measured) == self._space.candidate_count:
            return self.finish(observations, budget_exhausted=False)

        remaining_count = self._space.candidate_count - len(measured)
        return NoSafeCandidate(
            path=TuningPath.EXHAUSTIVE,
            reason=(
                f"All {remaining_count} unmeasured safe endpoints are pending; "
                "no endpoint can be suggested until pending work resolves."
            ),
        )

    def finish(
        self,
        observations: tuple[TuningObservation, ...],
        budget_exhausted: bool,
    ) -> PathComplete:
        measured = validated_observation_map(
            self._space,
            self._context,
            observations,
        )
        measured_count = len(measured)
        candidate_count = self._space.candidate_count
        best = self._best_observation(measured)
        best_fields = (
            {}
            if best is None
            else {
                "best_observation_id": best.observation_id,
                "best_tdecq_db": best.tdecq_db,
            }
        )

        if measured_count == candidate_count:
            status = (
                "NO_VALID_TDECQ"
                if best is None
                else "GLOBAL_OPTIMUM_ON_ENUMERATED_GRID"
            )
        elif budget_exhausted:
            status = "EXHAUSTIVE_INCOMPLETE"
        else:
            raise ValueError(
                "incomplete exhaustive result requires explicit budget exhaustion"
            )

        return PathComplete(
            path=TuningPath.EXHAUSTIVE,
            status=status,
            measured_count=measured_count,
            candidate_count=candidate_count,
            **best_fields,
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
