from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from vcsel_ai_tuning.v2.contracts import MeasurementStatus
from vcsel_ai_tuning.v2.continuous.models import ProviderActionKind, ReasonCode
from vcsel_ai_tuning.v2.continuous.policy import (
    DEFAULT_I2C_POLICY,
    I2cParameterPolicy,
    ParameterRule,
)
from vcsel_ai_tuning.v2.continuous.provider import ProviderContext
from vcsel_ai_tuning.v2.kimi_memory import KimiCampaignMemory
from vcsel_ai_tuning.v2.providers.kimi_client import (
    KIMI_CONTINUOUS_RESPONSE_FORMAT,
    KimiApiClient,
)
from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider


class RecordingSdk:
    def __init__(self, payload: dict[str, object]) -> None:
        self.request: dict[str, object] | None = None

        def create(**request: object) -> SimpleNamespace:
            self.request = request
            return SimpleNamespace(
                choices=(
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload)),
                    ),
                ),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class SequencedSdk:
    def __init__(self, *payloads: dict[str, object]) -> None:
        self.requests: list[dict[str, object]] = []
        self._payloads = iter(payloads)

        def create(**request: object) -> SimpleNamespace:
            self.requests.append(request)
            return SimpleNamespace(
                choices=(
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(next(self._payloads))),
                    ),
                ),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class RawSequencedSdk:
    def __init__(
        self,
        *responses: tuple[str, str | None, object | None],
    ) -> None:
        self.requests: list[dict[str, object]] = []
        self._responses = iter(responses)

        def create(**request: object) -> SimpleNamespace:
            self.requests.append(request)
            content, finish_reason, usage = next(self._responses)
            return SimpleNamespace(
                choices=(
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        message=SimpleNamespace(content=content),
                    ),
                ),
                usage=usage,
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _payload(**updates: object) -> dict[str, object]:
    payload = {
        "quality_score": 0.72,
        "confidence": 0.86,
        "valid": True,
        "ood": False,
        "action": "PROPOSE",
        "requested_params": {"eq": 23, "gain_ctrl": 6},
        "better_than_reference": None,
        "reason_codes": ["EXPECTED_IMPROVEMENT"],
        "summary": "Try two bounded absolute I2C values.",
    }
    payload.update(updates)
    return payload


def _saved_sample(tmp_path, eye_sample):
    relative_path = "eyes/current.npy"
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    np.save(path, np.ones((16, 24), dtype=np.uint32))
    digest = sha256(path.read_bytes()).hexdigest()
    reference = eye_sample.raw_eye_density_ref.model_copy(
        update={"relative_path": relative_path, "sha256": digest},
    )
    return eye_sample.model_copy(
        update={
            "measurement_id": "measurement-current",
            "raw_eye_density_ref": reference,
            "artifact_hashes": {relative_path: digest},
            "readback_params": {"eq": 22.0, "gain_ctrl": 5.0},
        }
    )


def _context(
    sample,
    *,
    best_reference=None,
    hardware_round: int = 1,
    max_rounds: int = 3,
    target_tdecq_db: float = 2.7,
    recent_valid_summaries=(),
    policy=DEFAULT_I2C_POLICY,
) -> ProviderContext:
    return ProviderContext(
        sample=sample,
        verified_readback={"eq": 22, "gain_ctrl": 5},
        policy=policy,
        hardware_round=hardware_round,
        max_rounds=max_rounds,
        target_tdecq_db=target_tdecq_db,
        recent_valid_summaries=recent_valid_summaries,
        best_reference=best_reference,
        allowed_output_schema={"version": "provider-action-v1"},
    )


def test_continuous_client_sends_restricted_tuning_bounds(tmp_path, eye_sample) -> None:
    policy = I2cParameterPolicy(
        version="i2c-restricted-v1",
        max_changed_parameters=1,
        rules=(
            ParameterRule(
                name="eq",
                encoding_min=0,
                encoding_max=31,
                tuning_min=16,
                tuning_max=25,
                max_step=3,
            ),
        ),
    )
    sample = _saved_sample(tmp_path, eye_sample)
    sdk = RecordingSdk(_payload(requested_params={"eq": 19}))
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)

    client.analyze_continuous(
        png_bytes=b"current-eye",
        instruction="Improve TDECQ within the supplied policy.",
        context=_context(sample, policy=policy),
    )

    text = sdk.request["messages"][1]["content"][-1]["text"]
    prompt = json.loads(text)
    assert prompt["parameter_policy"] == {
        "version": "i2c-restricted-v1",
        "maximum_changed_parameters": 1,
        "rules": {
            rule.name: {
                "encoding_min": rule.encoding_min,
                "encoding_max": rule.encoding_max,
                "tuning_min": rule.tuning_min,
                "tuning_max": rule.tuning_max,
                "max_step": rule.max_step,
                "integer_only": True,
            }
            for rule in policy.rules
        },
    }


def test_continuous_schema_accepts_only_sparse_absolute_i2c_values() -> None:
    schema = KIMI_CONTINUOUS_RESPONSE_FORMAT["json_schema"]["schema"]
    requested = schema["properties"]["requested_params"]
    object_schema = requested["anyOf"][0]
    assert object_schema["minProperties"] == 1
    assert object_schema["maxProperties"] == 3
    assert object_schema["additionalProperties"] is False
    assert set(object_schema["properties"]) == set(DEFAULT_I2C_POLICY.by_name)
    assert all(item["type"] == "integer" for item in object_schema["properties"].values())
    assert "candidate_id" not in schema["properties"]


def test_continuous_client_sends_policy_readback_round_and_target_without_menu(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    sdk = RecordingSdk(_payload())
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)

    result = client.analyze_continuous(
        png_bytes=b"current-eye",
        instruction="Improve TDECQ within the supplied policy.",
        context=_context(sample, max_rounds=20, target_tdecq_db=2.8),
    )

    assert result["requested_params"] == {"eq": 23, "gain_ctrl": 6}
    assert sdk.request is not None
    text = sdk.request["messages"][1]["content"][-1]["text"]
    prompt = json.loads(text)
    assert prompt["verified_readback"] == {"eq": 22, "gain_ctrl": 5}
    assert prompt["round"] == {"current": 1, "maximum": 20, "remaining": 20}
    assert prompt["target"] == {"tdecq_db_lt": 2.8}
    assert prompt["measurement_status"] == "VALID"
    assert "metric_status" not in prompt
    assert set(prompt["parameter_policy"]["rules"]) == set(DEFAULT_I2C_POLICY.by_name)
    assert "safe_candidates" not in prompt
    assert "candidate_id" not in prompt
    assert (
        "For HOLD, STOP, or REQUEST_HUMAN, set requested_params to null."
        in sdk.request["messages"][0]["content"]
    )


def test_continuous_client_sends_bounded_read_only_prior_campaign_summary(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    sdk = RecordingSdk(_payload())
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)

    client.analyze_continuous(
        png_bytes=b"current-eye",
        instruction="Improve TDECQ within the supplied policy.",
        context=_context(sample),
    )

    text = sdk.request["messages"][1]["content"][-1]["text"]
    prompt = json.loads(text)
    summary = prompt["prior_campaign_summary"]
    assert summary["schema_version"] == "prior-campaign-summary-v1"
    assert summary["role"] == "READ_ONLY_ADVISORY"
    assert summary["source_campaigns"] == [
        "v21-b-20260817T125713321Z-b468337d",
        "v21-b-20260818T060928264Z-6ee6ae55",
        "v21-b-20260818T083709035Z-74272a5f",
    ]
    assert summary["completed_rounds"] == 150
    assert summary["finite_valid_tdecq_rounds"] == 149
    assert summary["objective"] == {
        "metric": "tdecq_db",
        "direction": "minimize",
        "historical_target_lt_db": 2.8,
        "target_reached": False,
    }
    assert summary["aggregate_tdecq_db"] == {
        "best": 2.87,
        "median": 3.43,
        "worst": 5.29,
    }
    assert summary["best_observations"][0] == {
        "campaign_id": "v21-b-20260818T083709035Z-74272a5f",
        "round": 46,
        "tdecq_db": 2.87,
        "params": {
            "ibias_dac": 68,
            "efin_cur": 15,
            "vga_cur": 11,
            "eq": 22,
            "res_gain": 7,
            "res_bw": 1,
            "gain_ctrl": 7,
            "vbias_ref": 15,
            "vref": 15,
        },
    }
    assert summary["top_20_observed_region"] == {
        "ibias_dac": [68, 68],
        "efin_cur": [15, 15],
        "vga_cur": [11, 12],
        "eq": [21, 22],
        "res_gain": [7, 9],
        "res_bw": [1, 1],
        "gain_ctrl": [6, 7],
        "vbias_ref": [15, 15],
        "vref": [15, 15],
    }
    assert summary["repeatability"]["maximum_exact_point_spread_db"] == 0.83
    assert len(summary["best_observations"]) <= 5
    assert len(summary["observed_poor_examples"]) <= 5
    assert len(json.dumps(summary, separators=(",", ":")).encode("utf-8")) <= 6000
    system_prompt = sdk.request["messages"][0]["content"]
    assert "prior_campaign_summary is read-only advisory evidence" in system_prompt
    assert "current eye, verified readback, and parameter_policy take priority" in system_prompt


def test_continuous_client_retries_one_invalid_response_with_identical_request(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    sdk = SequencedSdk(
        _payload(action="STOP", requested_params={"eq": 23}),
        _payload(),
    )
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)

    result = client.analyze_continuous(
        png_bytes=b"current-eye",
        instruction="Improve TDECQ within the supplied policy.",
        context=_context(sample),
    )

    assert result["action"] == "PROPOSE"
    assert len(sdk.requests) == 2
    assert sdk.requests[0] == sdk.requests[1]


def test_continuous_client_logs_raw_valid_response_before_parse_result(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    raw_content = json.dumps(_payload())
    usage = SimpleNamespace(
        prompt_tokens=321,
        completion_tokens=87,
        total_tokens=408,
    )
    sdk = RawSequencedSdk((raw_content, "stop", usage))
    response_log = tmp_path / "kimi-response-attempts.jsonl"
    client = KimiApiClient(
        api_key="test-key",
        sdk_client=sdk,
        response_log_path=response_log,
    )

    result = client.analyze_continuous(
        png_bytes=b"current-eye",
        instruction="Improve TDECQ within the supplied policy.",
        context=_context(sample),
    )

    assert result["action"] == "PROPOSE"
    assert sdk.requests[0]["max_completion_tokens"] == 4096
    assert "Return ONLY one valid JSON object." in sdk.requests[0]["messages"][0]["content"]
    records = [json.loads(line) for line in response_log.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "RESPONSE_RECEIVED",
        "PARSE_RESULT",
    ]
    assert records[0] == {
        "event": "RESPONSE_RECEIVED",
        "measurement_id": sample.measurement_id,
        "attempt": 1,
        "max_completion_tokens": 4096,
        "finish_reason": "stop",
        "raw_response_content": raw_content,
        "token_usage": {
            "prompt_tokens": 321,
            "completion_tokens": 87,
            "total_tokens": 408,
        },
    }
    assert records[1]["event"] == "PARSE_RESULT"
    assert records[1]["attempt"] == 1
    assert records[1]["parse_success"] is True
    assert "parse_error" not in records[1]


def test_continuous_client_retries_truncated_response_with_larger_token_limit(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    truncated = '{"quality_score":0.7,"summary":"unterminated'
    sdk = RawSequencedSdk(
        (truncated, "length", {"completion_tokens": 4096}),
        (json.dumps(_payload()), "stop", {"completion_tokens": 91}),
    )
    response_log = tmp_path / "kimi-response-attempts.jsonl"
    client = KimiApiClient(
        api_key="test-key",
        sdk_client=sdk,
        response_log_path=response_log,
    )

    result = client.analyze_continuous(
        png_bytes=b"current-eye",
        instruction="Improve TDECQ within the supplied policy.",
        context=_context(sample),
    )

    assert result["summary"] == _payload()["summary"]
    assert [request["max_completion_tokens"] for request in sdk.requests] == [4096, 8192]
    records = [json.loads(line) for line in response_log.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "RESPONSE_RECEIVED",
        "PARSE_RESULT",
        "RESPONSE_RECEIVED",
        "PARSE_RESULT",
    ]
    assert records[0]["raw_response_content"] == truncated
    assert records[0]["finish_reason"] == "length"
    assert records[1]["parse_success"] is False
    assert records[2]["attempt"] == 2
    assert records[2]["max_completion_tokens"] == 8192
    assert records[3]["parse_success"] is True


def test_continuous_client_normalizes_exact_proposed_action_alias(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    client = KimiApiClient(
        api_key="test-key",
        sdk_client=RecordingSdk(_payload(action="PROPOSED")),
    )

    result = client.analyze_continuous(
        png_bytes=b"current-eye",
        instruction="Improve TDECQ within the supplied policy.",
        context=_context(sample),
    )

    assert result["action"] == "PROPOSE"
    assert result["requested_params"] == {"eq": 23, "gain_ctrl": 6}


def test_continuous_client_reports_second_response_validation_reason(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    invalid = _payload(action="STOP", requested_params={"eq": 23})
    sdk = SequencedSdk(invalid, invalid)
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)

    with pytest.raises(ValueError, match="STOP forbids requested_params"):
        client.analyze_continuous(
            png_bytes=b"current-eye",
            instruction="Improve TDECQ within the supplied policy.",
            context=_context(sample),
        )

    assert len(sdk.requests) == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"requested_params": {"eq": 23, "gain_ctrl": 6, "vref": 14, "res_bw": 1}},
        {"requested_params": {"eq": 23.0}},
        {"requested_params": {"unknown": 1}},
        {"action": "STOP", "requested_params": {"eq": 23}},
        {"action": "PROPOSE", "requested_params": None},
        {"action": "proposed"},
        {"reason_codes": ["INVENTED"]},
        {"summary": ""},
    ],
)
def test_continuous_client_rejects_invalid_structured_response(
    tmp_path,
    eye_sample,
    updates: dict[str, object],
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    client = KimiApiClient(
        api_key="test-key",
        sdk_client=RecordingSdk(_payload(**updates)),
    )
    with pytest.raises(ValueError, match="invalid response"):
        client.analyze_continuous(
            png_bytes=b"current-eye",
            instruction="Improve TDECQ within the supplied policy.",
            context=_context(sample),
        )


def test_kimi_provider_implements_continuous_decision_without_candidate_menu(
    tmp_path,
    eye_sample,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    sdk = RecordingSdk(_payload())
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="Improve TDECQ within the supplied policy.",
        memory=KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=sample.campaign_id),
    )

    action = provider.decide(_context(sample))

    assert action.action is ProviderActionKind.PROPOSE
    assert action.requested_params == {"eq": 23, "gain_ctrl": 6}
    assert action.reason_codes == (ReasonCode.EXPECTED_IMPROVEMENT,)
    assert action.confidence == 0.86
    assert action.summary == "Try two bounded absolute I2C values."


def test_kimi_provider_can_propose_recovery_from_invalid_current_eye(
    tmp_path,
    eye_sample,
) -> None:
    current = _saved_sample(tmp_path, eye_sample).model_copy(
        update={
            "measurement_status": MeasurementStatus.INVALID,
            "metrics": eye_sample.metrics.model_copy(update={"tdecq_db": None}),
            "er_status": "CORR",
            "tdecq_status": "INV",
            "tdecq_invalid_reason": "The target SER is lower than actual SER.",
        }
    )
    reference = current.model_copy(
        update={
            "measurement_id": "best-valid",
            "measurement_status": MeasurementStatus.VALID,
            "metrics": eye_sample.metrics,
            "tdecq_status": "CORR",
            "tdecq_invalid_reason": None,
        }
    )
    sdk = RecordingSdk(_payload(valid=False, better_than_reference=False))
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="Improve TDECQ within the supplied policy.",
        memory=KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=current.campaign_id),
    )

    action = provider.decide(_context(current, best_reference=reference, hardware_round=2))

    assert action.action is ProviderActionKind.PROPOSE
    content = sdk.request["messages"][1]["content"]
    prompt = json.loads(content[-1]["text"])
    assert prompt["measurement_status"] == "INVALID"
    assert prompt["metric_status"] == {
        "er_status": "CORR",
        "tdecq_status": "INV",
        "tdecq_invalid_reason": "The target SER is lower than actual SER.",
    }
    assert [item["text"] for item in content if item["type"] == "text"][:2] == [
        "CURRENT_EYE",
        "BEST_REFERENCE_EYE:best-valid",
    ]


def test_kimi_continuous_sends_exact_indexed_provider_png_without_reprocessing(
    tmp_path,
    eye_sample,
) -> None:
    from io import BytesIO
    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (4, 3), (1, 2, 3)).save(stream, format="PNG")
    png = stream.getvalue()
    relative = "eyes/current.png"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(png)
    digest = sha256(png).hexdigest()
    sample = eye_sample.model_copy(
        update={
            "measurement_id": "screen-current",
            "raw_eye_density_ref": eye_sample.raw_eye_density_ref.model_copy(
                update={
                    "relative_path": relative,
                    "sha256": digest,
                    "media_type": "image/png",
                }
            ),
            "artifact_hashes": {relative: digest},
            "readback_params": {"eq": 22.0, "gain_ctrl": 5.0},
        }
    )
    sdk = RecordingSdk(_payload())
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="Improve TDECQ within the supplied policy.",
        memory=KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=sample.campaign_id),
    )

    provider.decide(_context(sample))

    sent_url = sdk.request["messages"][1]["content"][1]["image_url"]["url"]
    import base64

    assert base64.b64decode(sent_url.rsplit(",", 1)[1]) == png


def test_kimi_continuous_sends_controller_reference_and_configured_instruction(
    tmp_path,
    eye_sample,
) -> None:
    import base64
    from io import BytesIO
    from PIL import Image

    def saved_png_sample(name: str, color: tuple[int, int, int]):
        stream = BytesIO()
        Image.new("RGB", (4, 3), color).save(stream, format="PNG")
        png = stream.getvalue()
        relative = f"eyes/{name}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        digest = sha256(png).hexdigest()
        sample = eye_sample.model_copy(
            update={
                "measurement_id": name,
                "raw_eye_density_ref": eye_sample.raw_eye_density_ref.model_copy(
                    update={
                        "relative_path": relative,
                        "sha256": digest,
                        "media_type": "image/png",
                    }
                ),
                "artifact_hashes": {relative: digest},
                "readback_params": {"eq": 22.0, "gain_ctrl": 5.0},
            }
        )
        return sample, png

    current, current_png = saved_png_sample("round-1-worse", (3, 2, 1))
    reference, reference_png = saved_png_sample("baseline-best", (1, 2, 3))
    sdk = RecordingSdk(_payload(better_than_reference=False))
    instruction = "Treat the left raw eye as primary and the right TDECQ view as reference."
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction=instruction,
        memory=KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=current.campaign_id),
    )

    action = provider.decide(
        _context(
            current,
            best_reference=reference,
            hardware_round=2,
            recent_valid_summaries=(
                {"round_index": 1, "metrics": {"tdecq_db": 3.36}},
            ),
        )
    )

    assert action.action is ProviderActionKind.PROPOSE
    assert sdk.request is not None
    content = sdk.request["messages"][1]["content"]
    assert [item["text"] for item in content if item["type"] == "text"][:2] == [
        "CURRENT_EYE",
        "BEST_REFERENCE_EYE:baseline-best",
    ]
    assert base64.b64decode(content[1]["image_url"]["url"].rsplit(",", 1)[1]) == current_png
    assert base64.b64decode(content[3]["image_url"]["url"].rsplit(",", 1)[1]) == reference_png
    assert instruction in sdk.request["messages"][0]["content"]


def test_kimi_continuous_rejects_reference_metadata_without_matching_image(
    tmp_path,
    eye_sample,
) -> None:
    current = _saved_sample(tmp_path, eye_sample)
    reference = current.model_copy(update={"measurement_id": "baseline-best"})
    client = KimiApiClient(api_key="test-key", sdk_client=RecordingSdk(_payload()))

    with pytest.raises(ValueError, match="reference image and context"):
        client.analyze_continuous(
            png_bytes=b"current-eye",
            instruction="Improve TDECQ within the supplied policy.",
            context=_context(current, best_reference=reference),
        )


def test_kimi_continuous_requires_comparison_when_reference_is_supplied(
    tmp_path,
    eye_sample,
) -> None:
    current = _saved_sample(tmp_path, eye_sample)
    reference = current.model_copy(update={"measurement_id": "baseline-best"})
    client = KimiApiClient(api_key="test-key", sdk_client=RecordingSdk(_payload()))

    with pytest.raises(ValueError, match="must be boolean with reference"):
        client.analyze_continuous(
            png_bytes=b"current-eye",
            instruction="Improve TDECQ within the supplied policy.",
            context=_context(current, best_reference=reference),
            reference_png_bytes=b"reference-eye",
        )


@pytest.mark.parametrize(
    "valid, ood, expected_reason",
    [
        (False, False, ReasonCode.INVALID_OBSERVATION),
        (True, True, ReasonCode.OOD),
    ],
)
def test_kimi_provider_maps_invalid_or_ood_to_non_executable_hold(
    tmp_path,
    eye_sample,
    valid,
    ood,
    expected_reason,
) -> None:
    sample = _saved_sample(tmp_path, eye_sample)
    response = _payload() | {"valid": valid, "ood": ood}
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=RecordingSdk(response)),
        artifact_root=tmp_path,
        instruction="Improve TDECQ within the supplied policy.",
        memory=KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=sample.campaign_id),
    )

    action = provider.decide(_context(sample))

    assert action.action is ProviderActionKind.HOLD
    assert action.requested_params is None
    assert action.reason_codes == (expected_reason,)
