from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .domain import Candidate, ConstraintResult, ParameterSpec


class SafetyViolation(RuntimeError):
    pass


class SafetyDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    passed: bool
    applied: dict[str, float] = Field(default_factory=dict)
    constraints: tuple[ConstraintResult, ...] = ()

    def require(self, stage: str) -> None:
        if not self.passed:
            failed = ", ".join(
                result.name for result in self.constraints if not result.passed
            )
            raise SafetyViolation(f"{stage} safety check failed: {failed}")


class SafetyPolicy:
    def __init__(self, parameters: tuple[ParameterSpec, ...]):
        self._specs = {parameter.name: parameter for parameter in parameters}
        if len(self._specs) != len(parameters):
            raise ValueError("parameter names must be unique")

    def evaluate_candidate(
        self,
        candidate: Candidate,
        reference: dict[str, float],
    ) -> SafetyDecision:
        expected = set(self._specs)
        actual = set(candidate.requested)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            return SafetyDecision(
                passed=False,
                constraints=(
                    ConstraintResult(
                        name="candidate.parameter_set",
                        passed=False,
                        reason=f"missing={missing}, extra={extra}",
                    ),
                ),
            )

        constraints: list[ConstraintResult] = []
        applied: dict[str, float] = {}
        for name, spec in self._specs.items():
            try:
                prepared = spec.prepare(candidate.requested[name])
            except ValueError as error:
                constraints.append(
                    ConstraintResult(
                        name=f"{name}.bounds",
                        passed=False,
                        value=float(candidate.requested[name]),
                        limit=f"[{spec.lower}, {spec.upper}] {spec.unit}",
                        reason=str(error),
                    )
                )
                continue

            applied[name] = prepared
            constraints.append(
                ConstraintResult(
                    name=f"{name}.bounds",
                    passed=True,
                    value=prepared,
                    limit=f"[{spec.lower}, {spec.upper}] {spec.unit}",
                )
            )
            if spec.max_step is not None:
                if name not in reference:
                    constraints.append(
                        ConstraintResult(
                            name=f"{name}.max_step",
                            passed=False,
                            reason="reference value missing",
                        )
                    )
                else:
                    delta = abs(prepared - float(reference[name]))
                    constraints.append(
                        ConstraintResult(
                            name=f"{name}.max_step",
                            passed=delta <= spec.max_step,
                            value=delta,
                            limit=f"<= {spec.max_step} {spec.unit}",
                            reason=None if delta <= spec.max_step else "step too large",
                        )
                    )

        tap_names = sorted(name for name in applied if name.startswith("ffe_tap_"))
        if tap_names:
            norm = sum(abs(applied[name]) for name in tap_names)
            if norm == 0.0:
                constraints.append(
                    ConstraintResult(
                        name="ffe.l1_norm",
                        passed=False,
                        reason="FFE tap L1 norm is zero",
                    )
                )
            else:
                normalized = {name: applied[name] / norm for name in tap_names}
                normalized_in_bounds = all(
                    self._specs[name].lower <= value <= self._specs[name].upper
                    for name, value in normalized.items()
                )
                constraints.append(
                    ConstraintResult(
                        name="ffe.l1_norm",
                        passed=normalized_in_bounds,
                        value=sum(abs(value) for value in normalized.values()),
                        limit="L1 norm = 1 and every tap in bounds",
                        reason=None
                        if normalized_in_bounds
                        else "normalized FFE tap outside bounds",
                    )
                )
                if normalized_in_bounds:
                    applied.update(normalized)

        passed = len(applied) == len(self._specs) and all(
            result.passed for result in constraints
        )
        return SafetyDecision(
            passed=passed,
            applied=applied if passed else {},
            constraints=tuple(constraints),
        )

    def evaluate_readback(
        self,
        applied: dict[str, float],
        readback: dict[str, float],
    ) -> SafetyDecision:
        constraints: list[ConstraintResult] = []
        for name, spec in self._specs.items():
            if name not in applied or name not in readback:
                constraints.append(
                    ConstraintResult(
                        name=f"{name}.readback",
                        passed=False,
                        reason="applied or read-back value missing",
                    )
                )
                continue
            delta = abs(float(readback[name]) - float(applied[name]))
            constraints.append(
                ConstraintResult(
                    name=f"{name}.readback",
                    passed=delta <= spec.readback_tolerance,
                    value=delta,
                    limit=f"<= {spec.readback_tolerance} {spec.unit}",
                    reason=None
                    if delta <= spec.readback_tolerance
                    else "read-back mismatch",
                )
            )
        passed = len(constraints) == len(self._specs) and all(
            result.passed for result in constraints
        )
        return SafetyDecision(
            passed=passed,
            applied=dict(applied) if passed else {},
            constraints=tuple(constraints),
        )
