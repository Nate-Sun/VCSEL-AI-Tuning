from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vcsel_ai_tuning.v2 import (
    BrainFactory,
    BrainFactoryConfig,
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    MultimodalApiBrain,
    PerceptionResult,
)
from vcsel_ai_tuning.v2.kimi_memory import (
    KIMI_PROVIDER_ID,
    KimiCampaignMemory,
    KimiMemoryRevisionConflict,
)
from vcsel_ai_tuning.v2.providers import kimi_client
from vcsel_ai_tuning.v2.providers.kimi_client import KIMI_RESPONSE_FORMAT, KimiApiClient
from vcsel_ai_tuning.v2.providers.kimi_provider import KimiProvider


class RecordingSdk:
    def __init__(self, payload: dict[str, object], *, on_create=None) -> None:
        self.request: dict[str, object] | None = None
        self._on_create = on_create

        def create(**request: object) -> SimpleNamespace:
            self.request = request
            if self._on_create is not None:
                self._on_create()
            message = SimpleNamespace(content=json.dumps(payload))
            return SimpleNamespace(
                choices=(SimpleNamespace(message=message),),
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )


def _payload(*, better_than_reference: object = None) -> dict[str, object]:
    return {
        "quality_score": 0.73,
        "confidence": 0.86,
        "valid": True,
        "ood": False,
        "action": "PROPOSE",
        "candidate_id": "candidate-002",
        "reason_codes": ["IMPROVE_MIDDLE_EYE"],
        "better_than_reference": better_than_reference,
    }


def _recent_summary(
    *,
    reference_measurement_id: str | None = None,
    better_than_reference: bool | None = None,
) -> dict[str, object]:
    return {
        "measurement_id": "measurement-old",
        "quality_score": 0.73,
        "confidence": 0.86,
        "valid": True,
        "ood": False,
        "reference_measurement_id": reference_measurement_id,
        "better_than_reference": better_than_reference,
        "action": "PROPOSE",
        "candidate_id": "candidate-old",
        "reason_codes": ("IMPROVE_MIDDLE_EYE",),
        "readback_params": {"bias_ma": 6.01},
        "metrics": {
            "tdecq_db": 1.25,
            "outer_er_chan1a_db": 4.1,
            "rlm": 0.98,
            "outer_oma": 0.72,
            "other_metrics": {"eye_height": 0.31},
        },
    }


def _analyze(
    sdk: RecordingSdk,
    valid_eye_sample,
    brain_candidates,
    **kwargs: object,
) -> dict[str, object]:
    request = {
        "png_bytes": b"current-eye",
        "instruction": "prefer balanced eyes",
        "sample": valid_eye_sample,
        "safe_candidates": brain_candidates,
        **kwargs,
    }
    return dict(
        KimiApiClient(api_key="test-key", sdk_client=sdk).analyze(
            **request,
        )
    )


@pytest.fixture
def valid_eye_sample(tmp_path, eye_sample):
    return _saved_sample(eye_sample, tmp_path, "measurement-current")


def _saved_sample(eye_sample, artifact_root, measurement_id: str, *, value: int = 1):
    relative_path = f"eyes/{measurement_id}.npy"
    density_path = artifact_root / relative_path
    density_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(density_path, np.full((16, 24), value, dtype=np.uint32))
    digest = sha256(density_path.read_bytes()).hexdigest()
    reference = eye_sample.raw_eye_density_ref.model_copy(
        update={"relative_path": relative_path, "sha256": digest},
    )
    return eye_sample.model_copy(
        update={
            "measurement_id": measurement_id,
            "raw_eye_density_ref": reference,
            "artifact_hashes": {relative_path: digest},
        },
    )


def _append_assessment(
    memory: KimiCampaignMemory,
    sample,
    *,
    reference_measurement_id: str | None = None,
    better_than_reference: bool | None = None,
) -> None:
    perception = PerceptionResult(
        measurement_id=sample.measurement_id,
        quality_score=0.73,
        confidence=0.86,
        valid=True,
        ood=False,
        defects={},
        provider_id=KIMI_PROVIDER_ID,
        provider_version="test-kimi",
        input_hash=sample.raw_eye_density_ref.sha256,
        memory_revision=memory.read_snapshot().revision,
        memory_id=memory.memory_id,
        reference_measurement_id=reference_measurement_id,
        better_than_reference=better_than_reference,
    )
    decision = DecisionProposal(
        based_on_measurement_id=sample.measurement_id,
        action=DecisionAction.PROPOSE,
        candidate_id="candidate-002",
        requested_params={"bias_ma": 6.01},
        reason_codes=("IMPROVE_MIDDLE_EYE",),
        provider_id=KIMI_PROVIDER_ID,
        provider_version="test-kimi",
    )
    memory.append(
        sample,
        BrainResult(
            perception=perception,
            decision=decision,
            brain_mode=BrainMode.PURE_API,
            latency_ms=1.0,
            request_hash="a" * 64,
            response_hash="b" * 64,
        ),
    )


def _provider(sdk: RecordingSdk, tmp_path, memory: KimiCampaignMemory) -> KimiProvider:
    return KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="prefer balanced eyes",
        memory=memory,
    )


def test_kimi_provider_sends_png_as_image_part_and_maps_local_candidate(
    tmp_path,
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(_payload())
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=valid_eye_sample.campaign_id)
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="优先改善三只眼的均衡程度",
        memory=memory,
    )

    perception, decision = provider.analyze_once(
        valid_eye_sample,
        (),
        brain_candidates,
    )

    assert sdk.request is not None
    messages = sdk.request["messages"]
    assert isinstance(messages, list)
    content = messages[1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "CURRENT_EYE"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    response_format = sdk.request["response_format"]
    assert response_format["json_schema"]["strict"] is True
    assert sdk.request["model"] == "kimi-k3"
    assert sdk.request["reasoning_effort"] == "low"
    assert perception.measurement_id == valid_eye_sample.measurement_id
    assert perception.reference_measurement_id is None
    assert perception.better_than_reference is None
    assert perception.memory_id == memory.memory_id
    assert decision.candidate_id == "candidate-002"
    assert decision.requested_params == brain_candidates[1].requested_params


@pytest.mark.parametrize("memory", [None, object()])
def test_kimi_provider_requires_campaign_memory_at_construction(tmp_path, memory):
    with pytest.raises(TypeError, match="KimiCampaignMemory"):
        KimiProvider(
            KimiApiClient(api_key="test-key", sdk_client=RecordingSdk(_payload())),
            artifact_root=tmp_path,
            instruction="prefer balanced eyes",
            memory=memory,
        )


def test_kimi_provider_rejects_campaign_memory_subclass(tmp_path):
    class MemorySubclass(KimiCampaignMemory):
        pass

    memory = MemorySubclass(tmp_path / "kimi.jsonl", campaign_id="campaign-001")

    with pytest.raises(TypeError, match="KimiCampaignMemory"):
        _provider(RecordingSdk(_payload()), tmp_path, memory)


def test_kimi_provider_compares_current_eye_to_campaign_best_without_mutating_journal(
    tmp_path,
    eye_sample,
    brain_candidates,
):
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    best = _saved_sample(eye_sample, tmp_path, "measurement-best", value=2)
    current = _saved_sample(eye_sample, tmp_path, "measurement-current", value=3)
    _append_assessment(memory, best)
    journal_before = (tmp_path / "kimi.jsonl").read_bytes()
    expected_revision = memory.read_snapshot().revision
    sdk = RecordingSdk(_payload(better_than_reference=True))

    perception, _decision = _provider(sdk, tmp_path, memory).analyze_once(current, (), brain_candidates)

    assert (tmp_path / "kimi.jsonl").read_bytes() == journal_before
    assert sdk.request is not None
    content = sdk.request["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text", "image_url", "text"]
    assert content[2]["text"] == "BEST_REFERENCE_EYE:measurement-best"
    context = json.loads(content[-1]["text"])
    assert context["reference_measurement_id"] == "measurement-best"
    assert len(context["recent_assessments"]) <= 5
    assert perception.reference_measurement_id == "measurement-best"
    assert perception.better_than_reference is True
    assert perception.memory_revision == expected_revision
    assert perception.memory_id == memory.memory_id


def test_kimi_provider_result_cannot_append_after_sdk_interleaving_changes_snapshot(
    tmp_path,
    eye_sample,
    brain_candidates,
):
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    best = _saved_sample(eye_sample, tmp_path, "measurement-best", value=2)
    current = _saved_sample(eye_sample, tmp_path, "measurement-current", value=3)
    interleaving = _saved_sample(eye_sample, tmp_path, "measurement-interleaving", value=4)
    _append_assessment(memory, best)
    snapshot_revision = memory.read_snapshot().revision
    sdk = RecordingSdk(
        _payload(better_than_reference=False),
        on_create=lambda: _append_assessment(
            memory,
            interleaving,
            reference_measurement_id=best.measurement_id,
            better_than_reference=False,
        ),
    )

    perception, decision = _provider(sdk, tmp_path, memory).analyze_once(current, (), brain_candidates)

    assert perception.memory_revision == snapshot_revision
    result = BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=BrainMode.PURE_API,
        latency_ms=1.0,
        request_hash="c" * 64,
        response_hash="d" * 64,
    )
    with pytest.raises(KimiMemoryRevisionConflict):
        memory.append(current, result)

    assert [entry.sample.measurement_id for entry in memory.load()] == [
        "measurement-best",
        "measurement-interleaving",
    ]


def test_kimi_provider_uses_latest_derived_best_and_only_five_recent_summaries(
    tmp_path,
    eye_sample,
    brain_candidates,
):
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    first = _saved_sample(eye_sample, tmp_path, "measurement-001", value=1)
    _append_assessment(memory, first)
    best = first
    for index in range(2, 8):
        sample = _saved_sample(eye_sample, tmp_path, f"measurement-{index:03d}", value=index)
        is_better = index == 4
        _append_assessment(
            memory,
            sample,
            reference_measurement_id=best.measurement_id,
            better_than_reference=is_better,
        )
        if is_better:
            best = sample
    current = _saved_sample(eye_sample, tmp_path, "measurement-current", value=8)
    sdk = RecordingSdk(_payload(better_than_reference=False))

    perception, _decision = _provider(sdk, tmp_path, memory).analyze_once(current, (), brain_candidates)

    assert sdk.request is not None
    content = sdk.request["messages"][1]["content"]
    assert content[2]["text"] == "BEST_REFERENCE_EYE:measurement-004"
    context = json.loads(content[-1]["text"])
    assert [summary["measurement_id"] for summary in context["recent_assessments"]] == [
        "measurement-003", "measurement-004", "measurement-005", "measurement-006", "measurement-007"
    ]
    assert perception.reference_measurement_id == "measurement-004"
    assert perception.better_than_reference is False


@pytest.mark.parametrize("damage", ["delete", "corrupt"], ids=["missing", "hash-mismatch"])
def test_kimi_provider_rejects_unusable_best_before_sdk_call(
    tmp_path,
    eye_sample,
    brain_candidates,
    damage,
):
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    best = _saved_sample(eye_sample, tmp_path, "measurement-best", value=2)
    current = _saved_sample(eye_sample, tmp_path, "measurement-current", value=3)
    _append_assessment(memory, best)
    best_path = tmp_path / best.raw_eye_density_ref.relative_path
    if damage == "delete":
        best_path.unlink()
    else:
        best_path.write_bytes(b"corrupt")
    sdk = RecordingSdk(_payload(better_than_reference=True))

    with pytest.raises(ValueError):
        _provider(sdk, tmp_path, memory).analyze_once(current, (), brain_candidates)

    assert sdk.request is None


def test_kimi_provider_rejects_campaign_mismatch_corrupt_journal_and_duplicate_before_sdk(
    tmp_path,
    eye_sample,
    brain_candidates,
):
    memory = KimiCampaignMemory(tmp_path / "kimi.jsonl", campaign_id=eye_sample.campaign_id)
    current = _saved_sample(eye_sample, tmp_path, "measurement-current")
    sdk = RecordingSdk(_payload())
    foreign = current.model_copy(update={"campaign_id": "campaign-foreign"})
    with pytest.raises(ValueError, match="campaign"):
        _provider(sdk, tmp_path, memory).analyze_once(foreign, (), brain_candidates)
    assert sdk.request is None

    (tmp_path / "kimi.jsonl").write_bytes(b"{corrupt}\\n")
    with pytest.raises(ValueError, match="LF|JSON"):
        _provider(sdk, tmp_path, memory).analyze_once(current, (), brain_candidates)
    assert sdk.request is None

    memory = KimiCampaignMemory(tmp_path / "duplicate.jsonl", campaign_id=eye_sample.campaign_id)
    _append_assessment(memory, current)
    with pytest.raises(ValueError, match="duplicate"):
        _provider(sdk, tmp_path, memory).analyze_once(current, (), brain_candidates)
    assert sdk.request is None


def test_kimi_client_current_only_sends_one_image_and_null_comparison(
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(_payload())

    payload = _analyze(sdk, valid_eye_sample, brain_candidates)

    assert payload["better_than_reference"] is None
    assert sdk.request is not None
    content = sdk.request["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[0]["text"] == "CURRENT_EYE"
    context = json.loads(content[-1]["text"])
    assert context["reference_measurement_id"] is None
    assert context["recent_assessments"] == []


def test_kimi_client_sends_reference_and_five_recent_assessments(
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(_payload(better_than_reference=True))
    summaries = tuple(
        _recent_summary(
            reference_measurement_id="measurement-best",
            better_than_reference=False,
        )
        | {"measurement_id": f"old-{index}"}
        for index in range(5)
    )

    payload = _analyze(
        sdk,
        valid_eye_sample,
        brain_candidates,
        reference_png_bytes=b"best-eye",
        reference_measurement_id="measurement-best",
        recent_summaries=summaries,
    )

    assert payload["better_than_reference"] is True
    assert sdk.request is not None
    content = sdk.request["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text", "image_url", "text"]
    assert content[2]["text"] == "BEST_REFERENCE_EYE:measurement-best"
    context = json.loads(content[-1]["text"])
    assert context["reference_measurement_id"] == "measurement-best"
    assert [summary["measurement_id"] for summary in context["recent_assessments"]] == [
        f"old-{index}" for index in range(5)
    ]
    assert context["recent_assessments"][0]["reason_codes"] == [
        "IMPROVE_MIDDLE_EYE"
    ]


@pytest.mark.parametrize(
    "reference_png_bytes,reference_measurement_id",
    [(b"best-eye", None), (None, "measurement-best")],
)
def test_kimi_client_rejects_unpaired_reference_before_sdk_call(
    valid_eye_sample,
    brain_candidates,
    reference_png_bytes,
    reference_measurement_id,
):
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="supplied together"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            reference_png_bytes=reference_png_bytes,
            reference_measurement_id=reference_measurement_id,
        )

    assert sdk.request is None


def test_kimi_client_rejects_more_than_five_or_invalid_recent_assessments_before_sdk_call(
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="at most 5"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            recent_summaries=tuple(_recent_summary() for _ in range(6)),
        )
    with pytest.raises(ValueError, match="must be mappings"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            recent_summaries=(_recent_summary(), "not-a-mapping"),
        )

    assert sdk.request is None


@pytest.mark.parametrize("comparison", [True, False])
def test_kimi_client_rejects_boolean_comparison_without_reference(
    valid_eye_sample,
    brain_candidates,
    comparison,
):
    sdk = RecordingSdk(_payload(better_than_reference=comparison))

    with pytest.raises(ValueError, match="better_than_reference"):
        _analyze(sdk, valid_eye_sample, brain_candidates)


@pytest.mark.parametrize("comparison", [None, pytest.param("missing", id="missing")])
def test_kimi_client_requires_boolean_comparison_with_reference(
    valid_eye_sample,
    brain_candidates,
    comparison,
):
    payload = _payload()
    if comparison == "missing":
        del payload["better_than_reference"]
    else:
        payload["better_than_reference"] = comparison
    sdk = RecordingSdk(payload)

    expected_error = "invalid response" if comparison == "missing" else "better_than_reference"
    with pytest.raises(ValueError, match=expected_error):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            reference_png_bytes=b"best-eye",
            reference_measurement_id="measurement-best",
        )


def test_kimi_client_requires_comparison_key_even_without_reference(
    valid_eye_sample,
    brain_candidates,
):
    payload = _payload()
    del payload["better_than_reference"]
    sdk = RecordingSdk(payload)

    with pytest.raises(ValueError, match="invalid response"):
        _analyze(sdk, valid_eye_sample, brain_candidates)


@pytest.mark.parametrize("png_bytes", [b"", bytearray(b"eye"), memoryview(b"eye")])
def test_kimi_client_rejects_invalid_current_image_before_sdk_call(
    valid_eye_sample,
    brain_candidates,
    png_bytes,
):
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="png_bytes"):
        _analyze(sdk, valid_eye_sample, brain_candidates, png_bytes=png_bytes)

    assert sdk.request is None


@pytest.mark.parametrize("reference_png_bytes", [b"", bytearray(b"eye"), memoryview(b"eye")])
def test_kimi_client_rejects_invalid_reference_image_before_sdk_call(
    valid_eye_sample,
    brain_candidates,
    reference_png_bytes,
):
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="reference_png_bytes"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            reference_png_bytes=reference_png_bytes,
            reference_measurement_id="measurement-best",
        )

    assert sdk.request is None


def test_kimi_client_rejects_oversized_images_before_base64_or_sdk_call(
    monkeypatch,
    valid_eye_sample,
    brain_candidates,
):
    monkeypatch.setattr(kimi_client, "MAX_KIMI_IMAGE_BYTES", 2)
    accepted_sdk = RecordingSdk(_payload())
    _analyze(accepted_sdk, valid_eye_sample, brain_candidates, png_bytes=b"ok")
    assert accepted_sdk.request is not None

    def forbidden_base64(value: bytes) -> bytes:
        raise AssertionError(f"base64 encoding reached for {len(value)} bytes")

    monkeypatch.setattr(kimi_client.base64, "b64encode", forbidden_base64)
    sdk = RecordingSdk(_payload())
    with pytest.raises(ValueError, match="png_bytes"):
        _analyze(sdk, valid_eye_sample, brain_candidates, png_bytes=b"bad")
    with pytest.raises(ValueError, match="reference_png_bytes"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            png_bytes=b"ok",
            reference_png_bytes=b"bad",
            reference_measurement_id="measurement-best",
        )
    assert sdk.request is None


@pytest.mark.parametrize("reference_measurement_id", ["", "has space", "bad\x00id", True, 1])
def test_kimi_client_rejects_invalid_reference_identifier_before_sdk_call(
    valid_eye_sample,
    brain_candidates,
    reference_measurement_id,
):
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="reference_measurement_id"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            reference_png_bytes=b"best-eye",
            reference_measurement_id=reference_measurement_id,
        )

    assert sdk.request is None


@pytest.mark.parametrize(
    "recent_summaries",
    [
        [_recent_summary()],
        (_recent_summary() | {"extra": True},),
        ({key: value for key, value in _recent_summary().items() if key != "metrics"},),
        (_recent_summary() | {"quality_score": True},),
        (_recent_summary(better_than_reference=True),),
        (_recent_summary() | {"reason_codes": ()},),
        (_recent_summary() | {"reason_codes": (" ",)},),
        (_recent_summary() | {"candidate_id": None},),
        (_recent_summary() | {"action": "HOLD"},),
    ],
    ids=[
        "list",
        "extra",
        "missing",
        "wrong-type",
        "unpaired-comparison",
        "empty-reasons",
        "blank-reason",
        "propose-without-candidate",
        "hold-with-candidate",
    ],
)
def test_kimi_client_rejects_noncanonical_or_invalid_recent_assessments_before_sdk_call(
    valid_eye_sample,
    brain_candidates,
    recent_summaries,
):
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="recent_summaries"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            recent_summaries=recent_summaries,
        )

    assert sdk.request is None


def test_kimi_client_hides_sdk_exception_details(
    valid_eye_sample,
    brain_candidates,
):
    secret = "sk-test-secret"
    data_url = "data:image/png;base64,secret-data"

    class FailingSdk:
        def __init__(self) -> None:
            def create(**request: object) -> None:
                del request
                raise RuntimeError(f"{secret} {data_url}")

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    with pytest.raises(RuntimeError, match="Kimi API request failed") as raised:
        _analyze(FailingSdk(), valid_eye_sample, brain_candidates)

    assert secret not in str(raised.value)
    assert data_url not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_kimi_client_hides_response_parse_exception_context(
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(_payload())
    sdk.chat.completions.create = lambda **request: SimpleNamespace(
        choices=(SimpleNamespace(message=SimpleNamespace(content="not-json")),)
    )

    with pytest.raises(ValueError, match="invalid response") as raised:
        _analyze(sdk, valid_eye_sample, brain_candidates)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_kimi_client_returns_only_strictly_validated_response_data(
    valid_eye_sample,
    brain_candidates,
):
    expected = _payload()

    assert _analyze(
        RecordingSdk(expected),
        valid_eye_sample,
        brain_candidates,
    ) == expected


@pytest.mark.parametrize(
    "damage",
    [
        {"reason_codes": "BAD"},
        {"extra": "sk-test-secret data:image/png;base64,secret-data"},
        {"candidate_id": None},
        {"action": "HOLD"},
        {"quality_score": "0.73"},
        {"valid": 1},
        {"ood": "false"},
        {"candidate_id": True},
        {"reason_codes": [1]},
        {"reason_codes": ["lowercase"]},
        {"reason_codes": ["FREE TEXT"]},
        {"reason_codes": ["BAD\nCODE"]},
        {"reason_codes": ["A" * 65]},
        {"reason_codes": ["A"] * 17},
    ],
    ids=[
        "reason-string",
        "extra-property",
        "propose-null-candidate",
        "hold-nonnull-candidate",
        "string-number",
        "integer-bool",
        "string-bool",
        "boolean-identifier",
        "integer-reason",
        "lowercase-reason",
        "free-text-reason",
        "control-reason",
        "long-reason",
        "too-many-reasons",
    ],
)
def test_kimi_client_rejects_schema_invalid_response_without_error_chain(
    valid_eye_sample,
    brain_candidates,
    damage,
):
    sdk = RecordingSdk(_payload() | damage)

    with pytest.raises(ValueError, match="invalid response") as raised:
        _analyze(sdk, valid_eye_sample, brain_candidates)

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "sk-test-secret" not in str(raised.value)
    assert "data:image/png;base64" not in str(raised.value)


@pytest.mark.parametrize(
    "reason_code",
    ["IMPROVE_MIDDLE_EYE", "BASELINE", "BETTER", "CHECK", "TRY"],
)
def test_kimi_client_accepts_categorical_reason_codes(
    valid_eye_sample,
    brain_candidates,
    reason_code,
):
    payload = _payload() | {"reason_codes": [reason_code]}

    assert _analyze(
        RecordingSdk(payload),
        valid_eye_sample,
        brain_candidates,
    )["reason_codes"] == [reason_code]


def test_schema_invalid_kimi_response_fails_closed_through_multimodal_brain(
    tmp_path,
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(_payload() | {"reason_codes": "BAD"})
    memory = KimiCampaignMemory(
        tmp_path / "kimi.jsonl",
        campaign_id=valid_eye_sample.campaign_id,
    )
    brain = MultimodalApiBrain(_provider(sdk, tmp_path, memory), 0.8)

    result = brain.analyze(valid_eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert result.decision.candidate_id is None
    assert result.decision.requested_params is None


@pytest.mark.parametrize(
    "reason_code",
    ["sk-test-secret", "data:image/png;base64,secret-data"],
    ids=["secret-like", "data-url"],
)
def test_factory_brain_rejects_reason_code_without_journal_or_result_leakage(
    tmp_path,
    valid_eye_sample,
    brain_candidates,
    reason_code,
):
    memory_path = tmp_path / "kimi.jsonl"
    memory = KimiCampaignMemory(
        memory_path,
        campaign_id=valid_eye_sample.campaign_id,
    )
    provider = _provider(
        RecordingSdk(_payload() | {"reason_codes": [reason_code]}),
        tmp_path,
        memory,
    )
    factory = BrainFactory()
    factory.register(
        BrainMode.PURE_API,
        lambda config: MultimodalApiBrain(provider, config.minimum_confidence),
    )
    brain = factory.create(
        BrainFactoryConfig(
            brain_mode=BrainMode.PURE_API,
            enabled_brain_modes=(BrainMode.PURE_API,),
            minimum_confidence=0.8,
        )
    )

    result = brain.analyze(valid_eye_sample, (), brain_candidates)

    journal = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
    assert reason_code not in journal
    assert reason_code not in result.model_dump_json()
    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)
    assert result.decision.candidate_id is None
    assert result.decision.requested_params is None


def test_kimi_client_hides_context_serialization_exception_context(
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="context is not JSON-serializable") as raised:
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            instruction=object(),
        )

    assert sdk.request is None
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_kimi_client_rejects_oversized_context_before_sdk_call(
    monkeypatch,
    valid_eye_sample,
    brain_candidates,
):
    monkeypatch.setattr(kimi_client, "MAX_KIMI_CONTEXT_BYTES", 32)
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="context exceeds"):
        _analyze(
            sdk,
            valid_eye_sample,
            brain_candidates,
            instruction="x" * 64,
        )

    assert sdk.request is None


def test_kimi_client_counts_multibyte_context_by_utf8_bytes(
    monkeypatch,
    valid_eye_sample,
    brain_candidates,
):
    baseline_sdk = RecordingSdk(_payload())
    _analyze(
        baseline_sdk,
        valid_eye_sample,
        brain_candidates,
        instruction="é",
    )
    assert baseline_sdk.request is not None
    context = baseline_sdk.request["messages"][1]["content"][-1]["text"]
    context_bytes = len(context.encode("utf-8"))

    monkeypatch.setattr(kimi_client, "MAX_KIMI_CONTEXT_BYTES", context_bytes)
    exact_sdk = RecordingSdk(_payload())
    _analyze(exact_sdk, valid_eye_sample, brain_candidates, instruction="é")
    assert exact_sdk.request is not None

    monkeypatch.setattr(kimi_client, "MAX_KIMI_CONTEXT_BYTES", context_bytes - 1)
    rejected_sdk = RecordingSdk(_payload())
    with pytest.raises(ValueError, match="context exceeds"):
        _analyze(rejected_sdk, valid_eye_sample, brain_candidates, instruction="é")
    assert rejected_sdk.request is None


def test_kimi_client_rejects_oversized_request_before_sdk_call(
    monkeypatch,
    valid_eye_sample,
    brain_candidates,
):
    monkeypatch.setattr(kimi_client, "MAX_KIMI_REQUEST_BYTES", 128)
    sdk = RecordingSdk(_payload())

    with pytest.raises(ValueError, match="request exceeds"):
        _analyze(sdk, valid_eye_sample, brain_candidates)

    assert sdk.request is None


def test_kimi_response_schema_requires_strict_reference_comparison():
    schema = KIMI_RESPONSE_FORMAT["json_schema"]

    assert schema["strict"] is True
    assert schema["schema"]["properties"]["better_than_reference"] == {
        "type": ["boolean", "null"]
    }
    assert "better_than_reference" in schema["schema"]["required"]
    reason_codes = schema["schema"]["properties"]["reason_codes"]
    assert reason_codes["maxItems"] == 16
    assert reason_codes["items"]["pattern"] == r"^[A-Z][A-Z0-9_]{0,63}$"
    assert reason_codes["items"]["maxLength"] == 64
