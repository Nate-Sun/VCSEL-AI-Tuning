"""Small OpenAI-compatible client for Kimi K3 vision decisions."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ..contracts import (
    BrainCandidate,
    EyeMetrics,
    EyeSample,
    Identifier,
    ImmutableFloatMapping,
    NonBlankText,
    Probability,
)
from ..continuous.models import ProviderActionKind, ReasonCode
from ..continuous.provider import ProviderContext


MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_KIMI_MODEL = "kimi-k3"
MAX_KIMI_CONTEXT_BYTES = 64 * 1024
MAX_KIMI_REQUEST_BYTES = 4 * 1024 * 1024
MAX_KIMI_IMAGE_BYTES = 1024 * 1024
DEFAULT_CONTINUOUS_MAX_COMPLETION_TOKENS = 4096
TRUNCATION_RETRY_MAX_COMPLETION_TOKENS = 8192
CONTINUOUS_MAX_ATTEMPTS = 2
_TRUNCATION_FINISH_REASONS = frozenset(
    {
        "length",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "token_limit",
    }
)
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)
KIMI_REASON_CODE_PATTERN = r"^[A-Z][A-Z0-9_]{0,63}$"
KimiReasonCode = Annotated[
    str,
    Field(pattern=KIMI_REASON_CODE_PATTERN, max_length=64),
]

KIMI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "vcsel_v21_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "valid": {"type": "boolean"},
                "ood": {"type": "boolean"},
                "action": {
                    "type": "string",
                    "enum": ["PROPOSE", "HOLD", "STOP", "REQUEST_HUMAN"],
                },
                "candidate_id": {"type": ["string", "null"]},
                "better_than_reference": {"type": ["boolean", "null"]},
                "reason_codes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "pattern": KIMI_REASON_CODE_PATTERN,
                    },
                    "minItems": 1,
                    "maxItems": 16,
                },
            },
            "required": [
                "quality_score",
                "confidence",
                "valid",
                "ood",
                "action",
                "candidate_id",
                "better_than_reference",
                "reason_codes",
            ],
            "additionalProperties": False,
        },
    },
}

_CONTINUOUS_PARAMETER_PROPERTIES = {
    "ibias_dac": {"type": "integer", "minimum": 0, "maximum": 127},
    "gain_ctrl": {"type": "integer", "minimum": 0, "maximum": 63},
    "eq": {"type": "integer", "minimum": 0, "maximum": 31},
    "efin_cur": {"type": "integer", "minimum": 0, "maximum": 15},
    "vga_cur": {"type": "integer", "minimum": 0, "maximum": 15},
    "res_gain": {"type": "integer", "minimum": 0, "maximum": 15},
    "res_bw": {"type": "integer", "minimum": 0, "maximum": 15},
    "vbias_ref": {"type": "integer", "minimum": 0, "maximum": 15},
    "vref": {"type": "integer", "minimum": 0, "maximum": 15},
}

KIMI_CONTINUOUS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "vcsel_v21_continuous_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "valid": {"type": "boolean"},
                "ood": {"type": "boolean"},
                "action": {
                    "type": "string",
                    "enum": [item.value for item in ProviderActionKind],
                },
                "requested_params": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": _CONTINUOUS_PARAMETER_PROPERTIES,
                            "minProperties": 1,
                            "maxProperties": 3,
                            "additionalProperties": False,
                        },
                        {"type": "null"},
                    ]
                },
                "better_than_reference": {"type": ["boolean", "null"]},
                "reason_codes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [item.value for item in ReasonCode],
                    },
                    "minItems": 1,
                    "maxItems": 16,
                },
                "summary": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": [
                "quality_score",
                "confidence",
                "valid",
                "ood",
                "action",
                "requested_params",
                "better_than_reference",
                "reason_codes",
                "summary",
            ],
            "additionalProperties": False,
        },
    },
}


class KimiRecentSummary(BaseModel):
    """Strict, compact form emitted by KimiCampaignMemory.recent_summaries."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    measurement_id: Identifier
    quality_score: Probability
    confidence: Probability
    valid: StrictBool
    ood: StrictBool
    reference_measurement_id: Identifier | None
    better_than_reference: StrictBool | None
    action: Literal["PROPOSE", "HOLD", "STOP", "REQUEST_HUMAN"]
    candidate_id: Identifier | None
    reason_codes: tuple[NonBlankText, ...] = Field(min_length=1)
    readback_params: ImmutableFloatMapping
    metrics: EyeMetrics

    @model_validator(mode="after")
    def validate_comparison_provenance(self) -> "KimiRecentSummary":
        if (self.reference_measurement_id is None) != (
            self.better_than_reference is None
        ):
            raise ValueError(
                "reference_measurement_id and better_than_reference must be supplied together"
            )
        if (self.action == "PROPOSE") != (self.candidate_id is not None):
            raise ValueError(
                "PROPOSE requires candidate_id and non-PROPOSE forbids candidate_id"
            )
        return self


class KimiResponse(BaseModel):
    """Strictly validate every field advertised by KIMI_RESPONSE_FORMAT."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    quality_score: Probability
    confidence: Probability
    valid: StrictBool
    ood: StrictBool
    action: Literal["PROPOSE", "HOLD", "STOP", "REQUEST_HUMAN"]
    candidate_id: Identifier | None
    better_than_reference: StrictBool | None
    reason_codes: tuple[KimiReasonCode, ...] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_action_candidate(self) -> "KimiResponse":
        if (self.action == "PROPOSE") != (self.candidate_id is not None):
            raise ValueError(
                "PROPOSE requires candidate_id and non-PROPOSE forbids candidate_id"
            )
        return self


class KimiContinuousResponse(BaseModel):
    """Strict continuous response; candidate identity remains locally owned."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    quality_score: Probability
    confidence: Probability
    valid: StrictBool
    ood: StrictBool
    action: ProviderActionKind
    requested_params: dict[str, int] | None
    better_than_reference: StrictBool | None
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1, max_length=16)
    summary: NonBlankText = Field(max_length=512)

    @field_validator("action", mode="before")
    @classmethod
    def normalize_proposed_action_alias(cls, value: object) -> object:
        if value == "PROPOSED":
            return ProviderActionKind.PROPOSE
        return value

    @model_validator(mode="after")
    def validate_action_params(self) -> "KimiContinuousResponse":
        if self.action is ProviderActionKind.PROPOSE:
            if not self.requested_params:
                raise ValueError("PROPOSE requires requested_params")
            if len(self.requested_params) > 3:
                raise ValueError("PROPOSE permits at most three requested parameters")
        elif self.requested_params is not None:
            raise ValueError(f"{self.action.value} forbids requested_params")
        if self.requested_params is not None:
            for name, value in self.requested_params.items():
                rule = _CONTINUOUS_PARAMETER_PROPERTIES.get(name)
                if rule is None or type(value) is not int:
                    raise ValueError("requested_params must use strict I2C integers")
                if not rule["minimum"] <= value <= rule["maximum"]:
                    raise ValueError("requested parameter is outside encoding range")
        return self


class KimiApiClient:
    """Call Kimi without leaking API details into the V2 brain adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_KIMI_MODEL,
        sdk_client: Any | None = None,
        response_log_path: str | Path | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not resolved_key:
            raise ValueError("MOONSHOT_API_KEY is not set")
        if not model.strip():
            raise ValueError("model must not be blank")
        self.model = model
        self._response_log_path = (
            None if response_log_path is None else Path(response_log_path)
        )
        if sdk_client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "Kimi support is not installed; install vcsel-ai-tuning[kimi]"
                ) from error
            sdk_client = OpenAI(
                api_key=resolved_key,
                base_url=MOONSHOT_BASE_URL,
                timeout=60.0,
            )
        self._sdk_client = sdk_client

    def analyze(
        self,
        *,
        png_bytes: bytes,
        instruction: str,
        sample: EyeSample,
        safe_candidates: tuple[BrainCandidate, ...],
        reference_png_bytes: bytes | None = None,
        reference_measurement_id: str | None = None,
        recent_summaries: tuple[Mapping[str, object], ...] = (),
    ) -> Mapping[str, object]:
        png_bytes = _validate_png_bytes(png_bytes, field="png_bytes")
        has_reference = reference_png_bytes is not None
        if has_reference != (reference_measurement_id is not None):
            raise ValueError(
                "reference_png_bytes and reference_measurement_id must be supplied together"
            )
        if has_reference:
            reference_png_bytes = _validate_png_bytes(
                reference_png_bytes,
                field="reference_png_bytes",
            )
            reference_measurement_id = _validate_identifier(
                reference_measurement_id,
                field="reference_measurement_id",
            )
        if type(recent_summaries) is not tuple:
            raise ValueError("recent_summaries must be a tuple")
        if len(recent_summaries) > 5:
            raise ValueError("recent_summaries may contain at most 5 assessments")
        summaries = tuple(
            _validate_recent_summary(summary) for summary in recent_summaries
        )

        image_data = base64.b64encode(png_bytes).decode("ascii")
        reference_image_data = (
            base64.b64encode(reference_png_bytes).decode("ascii")
            if has_reference
            else None
        )
        candidates = [
            candidate.model_dump(mode="json") for candidate in safe_candidates
        ]
        context = {
            "instruction": instruction,
            "measurement_id": sample.measurement_id,
            "current_readback_params": dict(sample.readback_params),
            "metrics": sample.metrics.model_dump(mode="json"),
            "safe_candidates": candidates,
            "reference_measurement_id": reference_measurement_id,
            "recent_assessments": [summary.model_dump(mode="json") for summary in summaries],
        }
        context_json = _serialize_json(context, label="context")
        if len(context_json.encode("utf-8")) > MAX_KIMI_CONTEXT_BYTES:
            raise ValueError("Kimi context exceeds maximum serialized byte size")

        content: list[dict[str, object]] = [
            {"type": "text", "text": "CURRENT_EYE"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
        if has_reference:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"BEST_REFERENCE_EYE:{reference_measurement_id}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{reference_image_data}"
                        },
                    },
                ]
            )
        content.append({"type": "text", "text": context_json})
        request_kwargs = {
            "model": self.model,
            "reasoning_effort": "low",
            "max_completion_tokens": 1024,
            "response_format": KIMI_RESPONSE_FORMAT,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You assess a PAM4 eye diagram and select at most one candidate. "
                        "For PROPOSE, candidate_id must exactly match one supplied safe "
                        "candidate. Never invent or modify parameter values. "
                        "Set better_than_reference to null when no best-reference eye was "
                        "supplied, and to a boolean when one was supplied."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        }
        request_json = _serialize_json(request_kwargs, label="request")
        if len(request_json.encode("utf-8")) > MAX_KIMI_REQUEST_BYTES:
            raise ValueError("Kimi request exceeds maximum serialized byte size")
        try:
            completion = self._sdk_client.chat.completions.create(**request_kwargs)
        except Exception:
            completion = None
        if completion is None:
            raise RuntimeError("Kimi API request failed") from None
        try:
            response_content = completion.choices[0].message.content
            response = KimiResponse.model_validate_json(
                response_content,
                strict=True,
            )
        except Exception:
            response = None
        if response is None:
            raise ValueError("Kimi returned an invalid response") from None
        payload = response.model_dump(mode="json")
        comparison = payload["better_than_reference"]
        if not has_reference and comparison is not None:
            raise ValueError("Kimi better_than_reference must be null without reference")
        if has_reference and type(comparison) is not bool:
            raise ValueError("Kimi better_than_reference must be boolean with reference")
        return payload

    def analyze_continuous(
        self,
        *,
        png_bytes: bytes,
        instruction: str,
        context: ProviderContext,
        reference_png_bytes: bytes | None = None,
    ) -> Mapping[str, object]:
        """Request one sparse absolute proposal without exposing executable objects."""

        png_bytes = _validate_png_bytes(png_bytes, field="png_bytes")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must not be blank")
        if not isinstance(context, ProviderContext):
            raise TypeError("context must be a ProviderContext")
        has_reference = reference_png_bytes is not None
        if has_reference != (context.best_reference is not None):
            raise ValueError(
                "reference image and context best_reference must be supplied together"
            )
        if has_reference:
            reference_png_bytes = _validate_png_bytes(
                reference_png_bytes,
                field="reference_png_bytes",
            )
        image_data = base64.b64encode(png_bytes).decode("ascii")
        content: list[dict[str, object]] = [
            {"type": "text", "text": "CURRENT_EYE"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
        if has_reference:
            reference_data = base64.b64encode(reference_png_bytes).decode("ascii")
            content.extend(
                (
                    {
                        "type": "text",
                        "text": (
                            "BEST_REFERENCE_EYE:"
                            f"{context.best_reference.measurement_id}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{reference_data}"},
                    },
                )
            )
        prompt = {
            "measurement_id": context.sample.measurement_id,
            "measurement_status": context.sample.measurement_status.value,
            "metrics": context.sample.metrics.model_dump(mode="json"),
            "verified_readback": dict(context.verified_readback),
            "parameter_policy": {
                "version": context.policy.version,
                "maximum_changed_parameters": context.policy.max_changed_parameters,
                "rules": {
                    rule.name: {
                        "encoding_min": rule.encoding_min,
                        "encoding_max": rule.encoding_max,
                        "tuning_min": rule.tuning_min,
                        "tuning_max": rule.tuning_max,
                        "max_step": rule.max_step,
                        "integer_only": True,
                    }
                    for rule in context.policy.rules
                },
            },
            "round": {
                "current": context.hardware_round,
                "maximum": context.max_rounds,
                "remaining": context.remaining_rounds,
            },
            "target": {"tdecq_db_lt": context.target_tdecq_db},
            "recent_valid_summaries": [
                dict(summary) for summary in context.recent_valid_summaries
            ],
            "best_reference": (
                None
                if context.best_reference is None
                else {
                    "measurement_id": context.best_reference.measurement_id,
                    "tdecq_db": context.best_reference.metrics.tdecq_db,
                }
            ),
            "allowed_output_schema": dict(context.allowed_output_schema),
        }
        if context.sample.measurement_status.value == "INVALID":
            prompt["metric_status"] = {
                "er_status": context.sample.er_status,
                "tdecq_status": context.sample.tdecq_status,
                "tdecq_invalid_reason": context.sample.tdecq_invalid_reason,
            }
        prompt_json = _serialize_json(prompt, label="continuous context")
        if len(prompt_json.encode("utf-8")) > MAX_KIMI_CONTEXT_BYTES:
            raise ValueError("Kimi context exceeds maximum serialized byte size")
        content.append({"type": "text", "text": prompt_json})
        request_kwargs = {
            "model": self.model,
            "reasoning_effort": "low",
            "max_completion_tokens": DEFAULT_CONTINUOUS_MAX_COMPLETION_TOKENS,
            "response_format": KIMI_CONTINUOUS_RESPONSE_FORMAT,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{instruction.strip()} "
                        "Assess the PAM4 eye and return one structured action. "
                        + (
                            "The INVALID current eye is failed-candidate evidence, not "
                            "an acceptable result; use it with the best valid reference "
                            "to propose different sparse values. "
                            if context.sample.measurement_status.value == "INVALID"
                            else ""
                        )
                        +
                        "For PROPOSE, provide sparse absolute integer I2C values within "
                        "each supplied tuning_min and tuning_max and within max_step from "
                        "the latest verified readback. For HOLD, STOP, or REQUEST_HUMAN, set "
                        "requested_params to null. Set better_than_reference to null when no "
                        "best-reference eye was supplied, and to a boolean when one was "
                        "supplied. You cannot authorize or execute hardware. "
                        "Return ONLY one valid JSON object. No Markdown. No ```json code "
                        "fences. No text before or after the JSON. Do not provide long "
                        "reasoning. Keep summary and reason_codes concise."
                    ),
                },
                {"role": "user", "content": content},
            ],
        }
        request_json = _serialize_json(request_kwargs, label="continuous request")
        if len(request_json.encode("utf-8")) > MAX_KIMI_REQUEST_BYTES:
            raise ValueError("Kimi request exceeds maximum serialized byte size")
        validation_error: Exception | None = None
        max_completion_tokens = DEFAULT_CONTINUOUS_MAX_COMPLETION_TOKENS
        for attempt in range(1, CONTINUOUS_MAX_ATTEMPTS + 1):
            request_kwargs["max_completion_tokens"] = max_completion_tokens
            try:
                completion = self._sdk_client.chat.completions.create(**request_kwargs)
            except Exception:
                raise RuntimeError("Kimi API request failed") from None
            try:
                choice = completion.choices[0]
                response_content = choice.message.content
                finish_reason = _normalize_finish_reason(
                    getattr(choice, "finish_reason", None)
                )
            except Exception:
                response_content = None
                finish_reason = None
            self._append_response_event(
                {
                    "event": "RESPONSE_RECEIVED",
                    "measurement_id": context.sample.measurement_id,
                    "attempt": attempt,
                    "max_completion_tokens": max_completion_tokens,
                    "finish_reason": finish_reason,
                    "raw_response_content": response_content,
                    "token_usage": _serialize_token_usage(
                        getattr(completion, "usage", None)
                    ),
                }
            )
            if _is_truncation_finish_reason(finish_reason):
                validation_error = ValueError(
                    f"response truncated: finish_reason={finish_reason}"
                )
                self._append_response_event(
                    {
                        "event": "PARSE_RESULT",
                        "measurement_id": context.sample.measurement_id,
                        "attempt": attempt,
                        "parse_success": False,
                        "parse_error": str(validation_error),
                    }
                )
                max_completion_tokens = TRUNCATION_RETRY_MAX_COMPLETION_TOKENS
                continue
            try:
                response = KimiContinuousResponse.model_validate_json(
                    response_content,
                    strict=True,
                )
                payload = response.model_dump(mode="json")
                comparison = payload["better_than_reference"]
                if not has_reference and comparison is not None:
                    raise ValueError(
                        "Kimi better_than_reference must be null without reference"
                    )
                if has_reference and type(comparison) is not bool:
                    raise ValueError(
                        "Kimi better_than_reference must be boolean with reference"
                    )
                self._append_response_event(
                    {
                        "event": "PARSE_RESULT",
                        "measurement_id": context.sample.measurement_id,
                        "attempt": attempt,
                        "parse_success": True,
                    }
                )
                return payload
            except Exception as error:
                validation_error = error
                self._append_response_event(
                    {
                        "event": "PARSE_RESULT",
                        "measurement_id": context.sample.measurement_id,
                        "attempt": attempt,
                        "parse_success": False,
                        "parse_error": f"{type(error).__name__}: {error}",
                    }
                )
        raise ValueError(
            f"Kimi returned an invalid response: {validation_error}"
        ) from None

    def _append_response_event(self, event: Mapping[str, object]) -> None:
        if self._response_log_path is None:
            return
        payload = _serialize_json(event, label="response attempt event")
        self._response_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._response_log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def _normalize_finish_reason(value: object) -> str | None:
    if value is None:
        return None
    normalized = getattr(value, "value", value)
    return str(normalized).strip() or None


def _is_truncation_finish_reason(value: str | None) -> bool:
    return value is not None and value.casefold() in _TRUNCATION_FINISH_REASONS


def _serialize_token_usage(value: object) -> object:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    except TypeError:
        return str(value)


def _serialize_json(value: object, *, label: str) -> str:
    """Serialize exactly the value measured at the Kimi SDK boundary."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        pass
    raise ValueError(f"Kimi {label} is not JSON-serializable") from None


def _validate_png_bytes(value: object, *, field: str) -> bytes:
    if type(value) is not bytes or not value:
        raise ValueError(f"{field} must be non-empty bytes")
    if len(value) > MAX_KIMI_IMAGE_BYTES:
        raise ValueError(f"{field} exceeds maximum byte size")
    return value


def _validate_identifier(value: object, *, field: str) -> str:
    try:
        return _IDENTIFIER_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        raise ValueError(f"{field} must be a valid identifier") from None


def _validate_recent_summary(value: object) -> KimiRecentSummary:
    if not isinstance(value, Mapping):
        raise ValueError("recent_summaries entries must be mappings")
    try:
        return KimiRecentSummary.model_validate(value, strict=True)
    except ValidationError:
        raise ValueError("recent_summaries entries must use the strict Kimi schema") from None
