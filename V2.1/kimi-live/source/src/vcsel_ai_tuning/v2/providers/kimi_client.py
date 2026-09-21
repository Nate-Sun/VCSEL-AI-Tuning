"""Small OpenAI-compatible client for Kimi K3 vision decisions."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    ValidationError,
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


MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_KIMI_MODEL = "kimi-k3"
MAX_KIMI_CONTEXT_BYTES = 64 * 1024
MAX_KIMI_REQUEST_BYTES = 4 * 1024 * 1024
MAX_KIMI_IMAGE_BYTES = 1024 * 1024
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


class KimiApiClient:
    """Call Kimi without leaking API details into the V2 brain adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_KIMI_MODEL,
        sdk_client: Any | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not resolved_key:
            raise ValueError("MOONSHOT_API_KEY is not set")
        if not model.strip():
            raise ValueError("model must not be blank")
        self.model = model
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
