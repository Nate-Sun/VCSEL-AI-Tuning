"""Versioned deterministic I2C policy for continuous optimization."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from math import ceil
from types import MappingProxyType
from typing import Annotated, Mapping

from pydantic import Field, StrictInt, model_validator

from vcsel_ai_tuning.campaign.models import (
    Candidate,
    ParameterSpec,
    StageAuthorization,
)

from .models import FrozenModel, NonBlankText


class ProposalErrorCode(StrEnum):
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    NON_INTEGER_VALUE = "NON_INTEGER_VALUE"
    OUTSIDE_ENCODING_RANGE = "OUTSIDE_ENCODING_RANGE"
    OUTSIDE_TUNING_RANGE = "OUTSIDE_TUNING_RANGE"
    EXCEEDS_MAX_STEP = "EXCEEDS_MAX_STEP"
    TOO_MANY_CHANGED_PARAMETERS = "TOO_MANY_CHANGED_PARAMETERS"
    NO_EFFECTIVE_CHANGE = "NO_EFFECTIVE_CHANGE"
    MISSING_READBACK = "MISSING_READBACK"


class ProposalValidationError(ValueError):
    def __init__(self, code: ProposalErrorCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


class ParameterRule(FrozenModel):
    name: NonBlankText
    encoding_min: StrictInt
    encoding_max: StrictInt
    tuning_min: StrictInt
    tuning_max: StrictInt
    max_step: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "ParameterRule":
        if self.encoding_min >= self.encoding_max:
            raise ValueError("encoding_min must be less than encoding_max")
        if self.tuning_min > self.tuning_max:
            raise ValueError("tuning_min must not exceed tuning_max")
        if not (
            self.encoding_min <= self.tuning_min <= self.encoding_max
            and self.encoding_min <= self.tuning_max <= self.encoding_max
        ):
            raise ValueError("tuning range must be inside encoding range")
        if self.max_step > self.encoding_max - self.encoding_min:
            raise ValueError("max_step must not exceed encoding width")
        return self

    def authorization_spec(self) -> ParameterSpec:
        return ParameterSpec(
            name=self.name,
            encoding_min=self.encoding_min,
            encoding_max=self.encoding_max,
            tuning_min=self.tuning_min,
            tuning_max=self.tuning_max,
            max_step=self.max_step,
            enabled=True,
        )


class I2cParameterPolicy(FrozenModel):
    version: NonBlankText
    rules: tuple[ParameterRule, ...] = Field(min_length=1)
    max_changed_parameters: StrictInt = Field(default=3, gt=0)

    @model_validator(mode="after")
    def validate_unique_names(self) -> "I2cParameterPolicy":
        names = tuple(rule.name for rule in self.rules)
        if len(names) != len(set(names)):
            raise ValueError("parameter policy names must be unique")
        return self

    @property
    def by_name(self) -> Mapping[str, ParameterRule]:
        return MappingProxyType({rule.name: rule for rule in self.rules})

    def authorization_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(rule.authorization_spec() for rule in self.rules)

    def validate_proposal(
        self,
        proposal: Mapping[str, object],
        current_readback: Mapping[str, object],
    ) -> dict[str, int]:
        if not isinstance(proposal, Mapping):
            raise TypeError("proposal must be a mapping")
        if not isinstance(current_readback, Mapping):
            raise TypeError("current_readback must be a mapping")
        rules = self.by_name
        sparse: dict[str, int] = {}
        for name, value in proposal.items():
            rule = rules.get(name)
            if rule is None:
                raise ProposalValidationError(
                    ProposalErrorCode.UNKNOWN_PARAMETER,
                    str(name),
                )
            if type(value) is not int:
                raise ProposalValidationError(
                    ProposalErrorCode.NON_INTEGER_VALUE,
                    name,
                )
            if not rule.encoding_min <= value <= rule.encoding_max:
                raise ProposalValidationError(
                    ProposalErrorCode.OUTSIDE_ENCODING_RANGE,
                    name,
                )
            if not rule.tuning_min <= value <= rule.tuning_max:
                raise ProposalValidationError(
                    ProposalErrorCode.OUTSIDE_TUNING_RANGE,
                    name,
                )
            if name not in current_readback or type(current_readback[name]) is not int:
                raise ProposalValidationError(
                    ProposalErrorCode.MISSING_READBACK,
                    name,
                )
            current = current_readback[name]
            if value == current:
                continue
            if abs(value - current) > rule.max_step:
                raise ProposalValidationError(
                    ProposalErrorCode.EXCEEDS_MAX_STEP,
                    name,
                )
            sparse[name] = value
        if not sparse:
            raise ProposalValidationError(
                ProposalErrorCode.NO_EFFECTIVE_CHANGE,
                "proposal changes no parameter",
            )
        if len(sparse) > self.max_changed_parameters:
            raise ProposalValidationError(
                ProposalErrorCode.TOO_MANY_CHANGED_PARAMETERS,
                str(len(sparse)),
            )
        return sparse


def _rule(name: str, maximum: int) -> ParameterRule:
    return ParameterRule(
        name=name,
        encoding_min=0,
        encoding_max=maximum,
        tuning_min=0,
        tuning_max=maximum,
        max_step=ceil(0.15 * maximum),
    )


DEFAULT_I2C_POLICY = I2cParameterPolicy(
    version="i2c-encoding-v1",
    rules=(
        _rule("ibias_dac", 127),
        _rule("gain_ctrl", 63),
        _rule("eq", 31),
        _rule("efin_cur", 15),
        _rule("vga_cur", 15),
        _rule("res_gain", 15),
        _rule("res_bw", 15),
        _rule("vbias_ref", 15),
        _rule("vref", 15),
    ),
)


def build_restricted_policy(
    config: Mapping[str, object],
    *,
    authorization_specs: tuple[ParameterSpec, ...],
    baseline_i2c: Mapping[str, int],
) -> I2cParameterPolicy:
    if not isinstance(config, Mapping):
        raise TypeError("parameter_policy must be a mapping")
    allowed_policy_keys = {"version", "max_changed_parameters", "rules"}
    if set(config) != allowed_policy_keys:
        raise ValueError(
            "parameter_policy must contain exactly version, "
            "max_changed_parameters, and rules"
        )
    raw_rules = config["rules"]
    if not isinstance(raw_rules, Mapping):
        raise TypeError("parameter_policy rules must be a mapping")

    specs = {spec.name: spec for spec in authorization_specs if spec.enabled}
    if set(raw_rules) != set(specs):
        raise ValueError(
            "parameter_policy rules must exactly match enabled authorization parameters"
        )

    rules: list[ParameterRule] = []
    for name in (spec.name for spec in authorization_specs if spec.enabled):
        raw_rule = raw_rules[name]
        if not isinstance(raw_rule, Mapping):
            raise TypeError(f"parameter_policy rule {name} must be a mapping")
        if set(raw_rule) != {"minimum", "maximum", "max_step"}:
            raise ValueError(f"parameter_policy rule {name} has invalid fields")
        minimum = raw_rule["minimum"]
        maximum = raw_rule["maximum"]
        max_step = raw_rule["max_step"]
        if any(type(value) is not int for value in (minimum, maximum, max_step)):
            raise TypeError(f"parameter_policy rule {name} values must be integers")

        spec = specs[name]
        if minimum > maximum:
            raise ValueError(f"parameter_policy rule {name} minimum exceeds maximum")
        if not spec.tuning_min <= minimum <= maximum <= spec.tuning_max:
            raise ValueError(
                f"parameter_policy rule {name} exceeds authorization tuning range"
            )
        if max_step <= 0 or max_step > spec.max_step:
            raise ValueError(
                f"parameter_policy rule {name} max_step exceeds authorization"
            )
        baseline_value = baseline_i2c.get(name)
        if type(baseline_value) is not int or not minimum <= baseline_value <= maximum:
            raise ValueError(f"B0 value for {name} is outside campaign tuning range")
        rules.append(
            ParameterRule(
                name=name,
                encoding_min=spec.encoding_min,
                encoding_max=spec.encoding_max,
                tuning_min=minimum,
                tuning_max=maximum,
                max_step=max_step,
            )
        )

    return I2cParameterPolicy(
        version=config["version"],
        max_changed_parameters=config["max_changed_parameters"],
        rules=tuple(rules),
    )


def materialize_candidate(
    *,
    campaign_id: str,
    round_index: int,
    attempt: int,
    requested: Mapping[str, object],
    current_readback: Mapping[str, object],
    authorization: StageAuthorization,
    policy: I2cParameterPolicy,
    provider_version: str,
    reason: str,
) -> Candidate:
    if authorization.campaign_id != campaign_id:
        raise ValueError("authorization campaign does not match candidate campaign")
    if type(round_index) is not int or round_index <= 0:
        raise ValueError("round_index must be a positive integer")
    if type(attempt) is not int or attempt <= 0:
        raise ValueError("attempt must be a positive integer")
    sparse = policy.validate_proposal(requested, current_readback)
    canonical = json.dumps(
        sparse,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    candidate_id = "-".join(
        (
            campaign_id,
            f"r{round_index:04d}",
            f"a{attempt}",
            sha256(canonical.encode("utf-8")).hexdigest()[:16],
        )
    )
    return Candidate(
        candidate_id=candidate_id,
        stage=authorization.stage,
        requested_i2c=sparse,
        requested_fir=None,
        predicted_metrics={},
        model_version=provider_version,
        reason=reason,
    )
