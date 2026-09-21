# V2.1 Kimi Compact Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded, campaign-scoped memory so each Kimi request can compare the current eye with the best prior eye without resending the complete sweep history.

**Architecture:** A strict JSONL journal stores each accepted Kimi assessment together with its exact EyeSample. KimiProvider reads that journal to send the current image, at most one best-reference image, and five compact summaries; execute_kimi_step appends the assessment before the existing V2.0 candidate runner can contact hardware. The V2.0 safety, readback, measurement, rollback, and recovery path stays unchanged.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, Pillow, OpenAI-compatible Moonshot client, pytest

---

## File structure

- Create **src/vcsel_ai_tuning/v2/kimi_memory.py**: strict JSONL models, load validation, best-eye derivation, compact summaries, and durable append.
- Modify **src/vcsel_ai_tuning/v2/contracts.py**: add explicit comparison provenance to PerceptionResult.
- Modify **src/vcsel_ai_tuning/v2/providers/kimi_client.py**: send optional reference image and bounded recent summaries; require a comparison result in strict JSON.
- Modify **src/vcsel_ai_tuning/v2/providers/kimi_provider.py**: load memory, render the best reference, and map comparison output into the contract.
- Modify **src/vcsel_ai_tuning/v2/kimi_step.py**: persist a genuine Kimi assessment before candidate execution.
- Create **tests_py/ai_tuning/v2/test_kimi_memory.py**: journal and deterministic best-selection tests.
- Modify **tests_py/ai_tuning/v2/test_contracts.py**: comparison field validation tests.
- Modify **tests_py/ai_tuning/v2/test_kimi_provider.py**: one-image/two-image payload and bounded-history tests.
- Modify **tests_py/ai_tuning/v2/test_kimi_step.py**: write ordering and write-failure fail-closed tests.

### Task 1: Add comparison provenance to the V2 perception contract

**Files:**
- Modify: **src/vcsel_ai_tuning/v2/contracts.py:257**
- Modify: **tests_py/ai_tuning/v2/test_contracts.py**

- [ ] **Step 1: Write failing contract tests**

Append these tests:

~~~python
def test_perception_allows_reference_comparison(
    perception_result: PerceptionResult,
) -> None:
    perception = perception_result.model_copy(
        update={
            "reference_measurement_id": "measurement-best",
            "better_than_reference": True,
        }
    )

    validated = PerceptionResult.model_validate(
        perception.model_dump(round_trip=True, warnings=False),
        strict=True,
    )

    assert validated.reference_measurement_id == "measurement-best"
    assert validated.better_than_reference is True


@pytest.mark.parametrize(
    ("reference_measurement_id", "better_than_reference"),
    [
        ("measurement-best", None),
        (None, True),
    ],
)
def test_perception_requires_complete_reference_comparison_pair(
    perception_result: PerceptionResult,
    reference_measurement_id: str | None,
    better_than_reference: bool | None,
) -> None:
    payload = perception_result.model_dump()
    payload.update(
        {
            "reference_measurement_id": reference_measurement_id,
            "better_than_reference": better_than_reference,
        }
    )

    with pytest.raises(ValidationError, match="reference"):
        PerceptionResult(**payload)
~~~

- [ ] **Step 2: Run the tests and verify RED**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_contracts.py -k "reference_comparison" -v
~~~

Expected: the valid-pair test fails because PerceptionResult forbids the two new fields.

- [ ] **Step 3: Add the two optional fields and pair validator**

Extend PerceptionResult with:

~~~python
class PerceptionResult(FrozenModel):
    measurement_id: Identifier
    quality_score: Probability
    confidence: Probability
    valid: StrictBool
    ood: StrictBool
    defects: ImmutableProbabilityMapping
    feature_vector: tuple[FiniteFloat, ...] | None = None
    reference_measurement_id: Identifier | None = None
    better_than_reference: StrictBool | None = None
    provider_id: NonBlankText
    provider_version: NonBlankText
    input_hash: Sha256

    @model_validator(mode="after")
    def validate_reference_comparison(self) -> "PerceptionResult":
        has_reference = self.reference_measurement_id is not None
        has_comparison = self.better_than_reference is not None
        if has_reference != has_comparison:
            raise ValueError(
                "reference_measurement_id and better_than_reference must be supplied together"
            )
        return self
~~~

- [ ] **Step 4: Run focused and V2 contract tests**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_contracts.py -v
~~~

Expected: all contract tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add src/vcsel_ai_tuning/v2/contracts.py tests_py/ai_tuning/v2/test_contracts.py
git commit -m "feat: add eye comparison provenance"
~~~

### Task 2: Implement the strict campaign JSONL memory

**Files:**
- Create: **src/vcsel_ai_tuning/v2/kimi_memory.py**
- Create: **tests_py/ai_tuning/v2/test_kimi_memory.py**

- [ ] **Step 1: Write failing journal tests**

Create test helpers that make a complete Kimi result without network or hardware:

~~~python
from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from vcsel_ai_tuning.v2.contracts import (
    BrainMode,
    BrainResult,
    DecisionAction,
    DecisionProposal,
    PerceptionResult,
)
from vcsel_ai_tuning.v2.kimi_memory import KimiCampaignMemory


def kimi_result(
    sample,
    *,
    score: float,
    reference_id: str | None = None,
    better: bool | None = None,
) -> BrainResult:
    perception = PerceptionResult(
        measurement_id=sample.measurement_id,
        quality_score=score,
        confidence=0.9,
        valid=True,
        ood=False,
        defects={},
        reference_measurement_id=reference_id,
        better_than_reference=better,
        provider_id="moonshot-kimi",
        provider_version="kimi-k3",
        input_hash=sample.raw_eye_density_ref.sha256,
    )
    decision = DecisionProposal(
        based_on_measurement_id=sample.measurement_id,
        action=DecisionAction.HOLD,
        reason_codes=("OBSERVE",),
        provider_id="moonshot-kimi",
        provider_version="kimi-k3",
    )
    return BrainResult(
        perception=perception,
        decision=decision,
        brain_mode=BrainMode.PURE_API,
        latency_ms=1.0,
        request_hash="b" * 64,
        response_hash="c" * 64,
    )


def next_sample(eye_sample, number: int):
    return eye_sample.model_copy(
        update={
            "measurement_id": f"measurement-{number:03d}",
            "sweep_point_id": f"point-{number:03d}",
        }
    )


def test_empty_memory_has_no_best(tmp_path, eye_sample):
    memory = KimiCampaignMemory(
        tmp_path / "kimi_history.jsonl",
        campaign_id=eye_sample.campaign_id,
    )

    assert memory.load() == ()
    assert memory.best_entry(()) is None
    assert memory.recent_summaries((), limit=5) == ()


def test_better_comparison_becomes_next_best(tmp_path, eye_sample):
    memory = KimiCampaignMemory(
        tmp_path / "kimi_history.jsonl",
        campaign_id=eye_sample.campaign_id,
    )
    first = next_sample(eye_sample, 1)
    second = next_sample(eye_sample, 2)
    memory.append(first, kimi_result(first, score=0.6))
    memory.append(
        second,
        kimi_result(
            second,
            score=0.7,
            reference_id=first.measurement_id,
            better=True,
        ),
    )

    entries = memory.load()

    assert memory.best_entry(entries).sample.measurement_id == second.measurement_id


def test_recent_summaries_are_limited_to_five(tmp_path, eye_sample):
    memory = KimiCampaignMemory(
        tmp_path / "kimi_history.jsonl",
        campaign_id=eye_sample.campaign_id,
    )
    best_id = None
    for number in range(1, 8):
        sample = next_sample(eye_sample, number)
        memory.append(
            sample,
            kimi_result(
                sample,
                score=number / 10,
                reference_id=best_id,
                better=None if best_id is None else True,
            ),
        )
        best_id = sample.measurement_id

    summaries = memory.recent_summaries(memory.load(), limit=5)

    assert len(summaries) == 5
    assert summaries[0]["measurement_id"] == "measurement-003"
    assert summaries[-1]["measurement_id"] == "measurement-007"


def test_corrupt_or_cross_campaign_history_fails_closed(tmp_path, eye_sample):
    path = tmp_path / "kimi_history.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    memory = KimiCampaignMemory(path, campaign_id=eye_sample.campaign_id)

    with pytest.raises(ValueError, match="invalid Kimi memory"):
        memory.load()

    other = eye_sample.model_copy(update={"campaign_id": "campaign-other"})
    path.write_text(
        json.dumps(
            {
                "sample": other.model_dump(mode="json"),
                "perception": kimi_result(other, score=0.5).perception.model_dump(mode="json"),
                "decision": kimi_result(other, score=0.5).decision.model_dump(mode="json"),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="campaign"):
        memory.load()


def test_stale_reference_and_duplicate_measurement_fail_closed(tmp_path, eye_sample):
    memory = KimiCampaignMemory(
        tmp_path / "kimi_history.jsonl",
        campaign_id=eye_sample.campaign_id,
    )
    first = next_sample(eye_sample, 1)
    second = next_sample(eye_sample, 2)
    memory.append(first, kimi_result(first, score=0.6))

    with pytest.raises(ValueError, match="reference"):
        memory.append(
            second,
            kimi_result(
                second,
                score=0.7,
                reference_id="measurement-stale",
                better=True,
            ),
        )

    with pytest.raises(ValueError, match="duplicate"):
        memory.append(first, kimi_result(first, score=0.6))
~~~

- [ ] **Step 2: Run the new tests and verify RED**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_memory.py -v
~~~

Expected: collection fails because vcsel_ai_tuning.v2.kimi_memory does not exist.

- [ ] **Step 3: Implement strict records, sequential best derivation, summaries, and durable append**

Create **src/vcsel_ai_tuning/v2/kimi_memory.py** with these public types and behaviors:

~~~python
"""Bounded, campaign-scoped memory for stateless Kimi optimization calls."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import os
from pathlib import Path

from pydantic import AwareDatetime, model_validator

from .contracts import BrainResult, DecisionProposal, EyeSample, FrozenModel, PerceptionResult


KIMI_PROVIDER_ID = "moonshot-kimi"


class KimiMemoryEntry(FrozenModel):
    sample: EyeSample
    perception: PerceptionResult
    decision: DecisionProposal
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def validate_measurement_ids(self) -> "KimiMemoryEntry":
        measurement_id = self.sample.measurement_id
        if self.perception.measurement_id != measurement_id:
            raise ValueError("memory perception measurement mismatch")
        if self.decision.based_on_measurement_id != measurement_id:
            raise ValueError("memory decision measurement mismatch")
        if self.perception.input_hash != self.sample.raw_eye_density_ref.sha256:
            raise ValueError("memory eye input hash mismatch")
        return self


class KimiCampaignMemory:
    def __init__(self, path: str | Path, *, campaign_id: str) -> None:
        if not campaign_id.strip():
            raise ValueError("campaign_id must not be blank")
        self.path = Path(path)
        self.campaign_id = campaign_id

    def load(self) -> tuple[KimiMemoryEntry, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise ValueError("invalid Kimi memory path")
        entries: list[KimiMemoryEntry] = []
        seen: set[str] = set()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line:
                    raise ValueError("empty Kimi memory line")
                entry = KimiMemoryEntry.model_validate_json(line, strict=True)
                measurement_id = entry.sample.measurement_id
                if entry.sample.campaign_id != self.campaign_id:
                    raise ValueError("Kimi memory campaign mismatch")
                if measurement_id in seen:
                    raise ValueError("duplicate Kimi memory measurement")
                seen.add(measurement_id)
                entries.append(entry)
            self.best_entry(tuple(entries))
        except Exception as error:
            if isinstance(error, ValueError) and "campaign" in str(error):
                raise
            raise ValueError("invalid Kimi memory") from error
        return tuple(entries)

    @staticmethod
    def _eligible(entry: KimiMemoryEntry) -> bool:
        return (
            entry.perception.provider_id == KIMI_PROVIDER_ID
            and entry.decision.provider_id == KIMI_PROVIDER_ID
            and entry.perception.valid
            and not entry.perception.ood
        )

    def best_entry(
        self,
        entries: tuple[KimiMemoryEntry, ...],
    ) -> KimiMemoryEntry | None:
        best: KimiMemoryEntry | None = None
        for entry in entries:
            expected_reference = None if best is None else best.sample.measurement_id
            if entry.perception.reference_measurement_id != expected_reference:
                raise ValueError("Kimi memory reference mismatch")
            if not self._eligible(entry):
                continue
            if best is None or entry.perception.better_than_reference is True:
                best = entry
        return best

    def recent_summaries(
        self,
        entries: tuple[KimiMemoryEntry, ...],
        *,
        limit: int,
    ) -> tuple[Mapping[str, object], ...]:
        if type(limit) is not int or not 0 <= limit <= 5:
            raise ValueError("summary limit must be between zero and five")
        if limit == 0:
            return ()
        return tuple(
            {
                "measurement_id": entry.sample.measurement_id,
                "quality_score": entry.perception.quality_score,
                "confidence": entry.perception.confidence,
                "valid": entry.perception.valid,
                "ood": entry.perception.ood,
                "reference_measurement_id": entry.perception.reference_measurement_id,
                "better_than_reference": entry.perception.better_than_reference,
                "action": entry.decision.action.value,
                "candidate_id": entry.decision.candidate_id,
                "reason_codes": list(entry.decision.reason_codes),
                "readback_params": dict(entry.sample.readback_params),
                "metrics": entry.sample.metrics.model_dump(mode="json"),
            }
            for entry in entries[-limit:]
        )

    def append(self, sample: EyeSample, result: BrainResult) -> KimiMemoryEntry:
        if sample.campaign_id != self.campaign_id:
            raise ValueError("Kimi memory campaign mismatch")
        if result.perception.provider_id != KIMI_PROVIDER_ID:
            raise ValueError("brain result is not a Kimi assessment")
        entries = self.load()
        if any(item.sample.measurement_id == sample.measurement_id for item in entries):
            raise ValueError("duplicate Kimi memory measurement")
        entry = KimiMemoryEntry(
            sample=sample,
            perception=result.perception,
            decision=result.decision,
            recorded_at=datetime.now(timezone.utc),
        )
        self.best_entry(entries + (entry,))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(entry.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return entry
~~~

During implementation, preserve exact error categories used by the tests: malformed content, duplicate measurement ID, cross-campaign content, and stale reference must all raise before a new line is written.

- [ ] **Step 4: Run memory and contract tests**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_memory.py tests_py/ai_tuning/v2/test_contracts.py -v
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add src/vcsel_ai_tuning/v2/kimi_memory.py tests_py/ai_tuning/v2/test_kimi_memory.py
git commit -m "feat: add compact Kimi campaign memory"
~~~

### Task 3: Send the best reference and bounded summaries to Kimi

**Files:**
- Modify: **src/vcsel_ai_tuning/v2/providers/kimi_client.py:19-135**
- Modify: **tests_py/ai_tuning/v2/test_kimi_provider.py**

- [ ] **Step 1: Extend the recording test to require comparison payload rules**

Change RecordingSdk to retain every request, then add direct client tests:

~~~python
def _user_content(sdk: RecordingSdk) -> list[dict[str, object]]:
    assert sdk.request is not None
    messages = sdk.request["messages"]
    assert isinstance(messages, list)
    content = messages[1]["content"]
    assert isinstance(content, list)
    return content


def test_kimi_client_sends_current_only_without_reference(
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(
        {
            "quality_score": 0.6,
            "confidence": 0.9,
            "valid": True,
            "ood": False,
            "better_than_reference": None,
            "action": "HOLD",
            "candidate_id": None,
            "reason_codes": ["BASELINE"],
        }
    )
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)

    payload = client.analyze(
        png_bytes=b"current",
        reference_png_bytes=None,
        reference_measurement_id=None,
        recent_summaries=(),
        instruction="improve the eye",
        sample=valid_eye_sample,
        safe_candidates=brain_candidates,
    )

    content = _user_content(sdk)
    assert sum(part["type"] == "image_url" for part in content) == 1
    assert payload["better_than_reference"] is None


def test_kimi_client_sends_current_reference_and_five_summaries(
    valid_eye_sample,
    brain_candidates,
):
    sdk = RecordingSdk(
        {
            "quality_score": 0.8,
            "confidence": 0.9,
            "valid": True,
            "ood": False,
            "better_than_reference": True,
            "action": "HOLD",
            "candidate_id": None,
            "reason_codes": ["BETTER"],
        }
    )
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)
    summaries = tuple({"measurement_id": f"m-{index}"} for index in range(5))

    client.analyze(
        png_bytes=b"current",
        reference_png_bytes=b"best",
        reference_measurement_id="measurement-best",
        recent_summaries=summaries,
        instruction="improve the eye",
        sample=valid_eye_sample,
        safe_candidates=brain_candidates,
    )

    content = _user_content(sdk)
    assert sum(part["type"] == "image_url" for part in content) == 2
    context = json.loads(content[-1]["text"])
    assert context["reference_measurement_id"] == "measurement-best"
    assert len(context["recent_assessments"]) == 5


@pytest.mark.parametrize(
    ("reference_png_bytes", "reference_measurement_id", "returned_comparison"),
    [
        (None, None, True),
        (b"best", "measurement-best", None),
    ],
)
def test_kimi_client_rejects_inconsistent_comparison(
    valid_eye_sample,
    brain_candidates,
    reference_png_bytes,
    reference_measurement_id,
    returned_comparison,
):
    sdk = RecordingSdk(
        {
            "quality_score": 0.8,
            "confidence": 0.9,
            "valid": True,
            "ood": False,
            "better_than_reference": returned_comparison,
            "action": "HOLD",
            "candidate_id": None,
            "reason_codes": ["CHECK"],
        }
    )
    client = KimiApiClient(api_key="test-key", sdk_client=sdk)

    with pytest.raises(ValueError, match="comparison"):
        client.analyze(
            png_bytes=b"current",
            reference_png_bytes=reference_png_bytes,
            reference_measurement_id=reference_measurement_id,
            recent_summaries=(),
            instruction="improve the eye",
            sample=valid_eye_sample,
            safe_candidates=brain_candidates,
        )
~~~

- [ ] **Step 2: Run client tests and verify RED**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py -k "client" -v
~~~

Expected: calls fail because analyze does not accept reference or summary arguments.

- [ ] **Step 3: Extend the strict schema and request construction**

Add better_than_reference to KIMI_RESPONSE_FORMAT:

~~~python
"better_than_reference": {"type": ["boolean", "null"]},
~~~

Add it to the required list. Change analyze to accept:

~~~python
def analyze(
    self,
    *,
    png_bytes: bytes,
    reference_png_bytes: bytes | None,
    reference_measurement_id: str | None,
    recent_summaries: tuple[Mapping[str, object], ...],
    instruction: str,
    sample: EyeSample,
    safe_candidates: tuple[BrainCandidate, ...],
) -> Mapping[str, object]:
~~~

Reject mismatched reference inputs before the SDK call:

~~~python
if (reference_png_bytes is None) != (reference_measurement_id is None):
    raise ValueError("reference image and measurement ID must be supplied together")
if len(recent_summaries) > 5:
    raise ValueError("at most five recent summaries are allowed")
~~~

Build the user content in this fixed order:

~~~python
content: list[dict[str, object]] = [
    {"type": "text", "text": "CURRENT_EYE"},
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_data}"},
    },
]
if reference_png_bytes is not None:
    reference_data = base64.b64encode(reference_png_bytes).decode("ascii")
    content.extend(
        [
            {
                "type": "text",
                "text": f"BEST_REFERENCE_EYE:{reference_measurement_id}",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{reference_data}"
                },
            },
        ]
    )
context["reference_measurement_id"] = reference_measurement_id
context["recent_assessments"] = list(recent_summaries)
content.append(
    {
        "type": "text",
        "text": json.dumps(
            context,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }
)
~~~

Use content as messages[1]["content"]. Update the system instruction to say better_than_reference must be null without a reference image and a boolean with one. After JSON parsing, enforce:

~~~python
comparison = payload.get("better_than_reference")
if reference_measurement_id is None:
    if comparison is not None:
        raise ValueError("Kimi returned an inconsistent comparison")
elif type(comparison) is not bool:
    raise ValueError("Kimi returned an inconsistent comparison")
~~~

- [ ] **Step 4: Run the client tests**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py -k "client" -v
~~~

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add src/vcsel_ai_tuning/v2/providers/kimi_client.py tests_py/ai_tuning/v2/test_kimi_provider.py
git commit -m "feat: send bounded eye history to Kimi"
~~~

### Task 4: Make KimiProvider select and render the best prior eye

**Files:**
- Modify: **src/vcsel_ai_tuning/v2/providers/kimi_provider.py:30-93**
- Modify: **tests_py/ai_tuning/v2/test_kimi_provider.py**

- [ ] **Step 1: Write provider integration tests**

Add a fixture helper that saves a separate verified NPY for each copied sample. Add these cases:

~~~python
def test_kimi_provider_uses_memory_best_as_reference(
    tmp_path,
    valid_eye_sample,
    brain_candidates,
):
    history_path = tmp_path / "kimi_history.jsonl"
    memory = KimiCampaignMemory(
        history_path,
        campaign_id=valid_eye_sample.campaign_id,
    )
    first = valid_eye_sample
    memory.append(first, kimi_result(first, score=0.6))
    current = save_density_copy(tmp_path, first, number=2, fill=2)
    sdk = RecordingSdk(
        {
            "quality_score": 0.8,
            "confidence": 0.9,
            "valid": True,
            "ood": False,
            "better_than_reference": True,
            "action": "HOLD",
            "candidate_id": None,
            "reason_codes": ["BETTER"],
        }
    )
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="improve the eye",
        memory=memory,
    )

    perception, _decision = provider.analyze_once(
        current,
        (),
        brain_candidates,
    )

    content = _user_content(sdk)
    assert sum(part["type"] == "image_url" for part in content) == 2
    assert perception.reference_measurement_id == first.measurement_id
    assert perception.better_than_reference is True


def test_kimi_provider_fails_before_api_when_best_artifact_is_missing(
    tmp_path,
    valid_eye_sample,
    brain_candidates,
):
    memory = KimiCampaignMemory(
        tmp_path / "kimi_history.jsonl",
        campaign_id=valid_eye_sample.campaign_id,
    )
    memory.append(valid_eye_sample, kimi_result(valid_eye_sample, score=0.6))
    current = save_density_copy(
        tmp_path,
        valid_eye_sample,
        number=2,
        fill=2,
    )
    (tmp_path / valid_eye_sample.raw_eye_density_ref.relative_path).unlink()
    sdk = RecordingSdk({})
    provider = KimiProvider(
        KimiApiClient(api_key="test-key", sdk_client=sdk),
        artifact_root=tmp_path,
        instruction="improve the eye",
        memory=memory,
    )

    with pytest.raises(ValueError):
        provider.analyze_once(current, (), brain_candidates)

    assert sdk.request is None
~~~

The save_density_copy helper must create a unique relative NPY path, calculate SHA-256 from the saved bytes, and update raw_eye_density_ref plus artifact_hashes so preprocessing verifies the artifact.

- [ ] **Step 2: Run provider tests and verify RED**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py -k "provider" -v
~~~

Expected: KimiProvider rejects the memory keyword or fails to send a reference.

- [ ] **Step 3: Inject memory and prepare bounded request context**

Import KIMI_PROVIDER_ID and KimiCampaignMemory. Replace the provider ID literal and constructor with:

~~~python
class KimiProvider:
    provider_id = KIMI_PROVIDER_ID

    def __init__(
        self,
        client: KimiApiClient,
        *,
        artifact_root: str | Path,
        instruction: str,
        memory: KimiCampaignMemory,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be blank")
        self._client = client
        self._artifact_root = Path(artifact_root)
        self._instruction = instruction
        self._memory = memory
        self.provider_version = client.model
~~~

At the start of analyze_once, replace del history with deterministic local context:

~~~python
del history
if sample.campaign_id != self._memory.campaign_id:
    raise ValueError("sample campaign does not match Kimi memory")
entries = self._memory.load()
if any(entry.sample.measurement_id == sample.measurement_id for entry in entries):
    raise ValueError("duplicate Kimi memory measurement")
best = self._memory.best_entry(entries)
recent_summaries = self._memory.recent_summaries(entries, limit=5)
png = _png_bytes(preprocess_eye_density(sample, self._artifact_root))
reference_png = (
    None
    if best is None
    else _png_bytes(preprocess_eye_density(best.sample, self._artifact_root))
)
reference_measurement_id = (
    None if best is None else best.sample.measurement_id
)
~~~

Pass reference_png_bytes, reference_measurement_id, and recent_summaries into client.analyze. Populate the new PerceptionResult fields from the exact reference shown and the validated response:

~~~python
reference_measurement_id=reference_measurement_id,
better_than_reference=payload["better_than_reference"],
~~~

Update the original single-image provider test to create an empty
KimiCampaignMemory and pass memory=memory. Add better_than_reference=None to
its synthetic SDK response so every test response satisfies the new strict
schema.

- [ ] **Step 4: Run all provider tests**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_provider.py -v
~~~

Expected: all provider and client tests pass without network access.

- [ ] **Step 5: Commit**

~~~powershell
git add src/vcsel_ai_tuning/v2/providers/kimi_provider.py tests_py/ai_tuning/v2/test_kimi_provider.py
git commit -m "feat: compare Kimi eye against campaign best"
~~~

### Task 5: Persist Kimi assessment before the hardware runner

**Files:**
- Modify: **src/vcsel_ai_tuning/v2/kimi_step.py:68-152**
- Modify: **tests_py/ai_tuning/v2/test_kimi_step.py**

- [ ] **Step 1: Write ordering and failure tests**

Change the test brain provider IDs to moonshot-kimi and add a recording memory:

~~~python
class RecordingMemory:
    def __init__(self, events, *, error=None):
        self.events = events
        self.error = error

    def append(self, sample, result):
        self.events.append(("memory", sample.measurement_id))
        if self.error is not None:
            raise self.error


class OrderedOrchestrator(RecordingOrchestrator):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def run_one_candidate(self, authorization, candidate):
        self.events.append(("hardware", candidate.candidate_id))
        return super().run_one_candidate(authorization, candidate)


def test_kimi_step_records_memory_before_candidate_execution(
    tmp_path,
    eye_sample,
):
    events = []
    campaign_candidates = two_candidates()
    brain = StaticBrain(
        brain_result(
            eye_sample,
            action=DecisionAction.PROPOSE,
            candidate=campaign_candidates[0],
        )
    )

    execute_kimi_step(
        brain=brain,
        orchestrator=OrderedOrchestrator(events),
        authorization=object(),
        sample=eye_sample,
        candidates=campaign_candidates,
        instruction="improve the eye",
        output_path=tmp_path / "v21_result.json",
        memory=RecordingMemory(events),
    )

    assert events == [
        ("memory", eye_sample.measurement_id),
        ("hardware", campaign_candidates[0].candidate_id),
    ]


def test_kimi_step_memory_failure_blocks_candidate_execution(
    tmp_path,
    eye_sample,
):
    events = []
    campaign_candidates = two_candidates()
    brain = StaticBrain(
        brain_result(
            eye_sample,
            action=DecisionAction.PROPOSE,
            candidate=campaign_candidates[0],
        )
    )
    orchestrator = OrderedOrchestrator(events)

    with pytest.raises(OSError, match="disk full"):
        execute_kimi_step(
            brain=brain,
            orchestrator=orchestrator,
            authorization=object(),
            sample=eye_sample,
            candidates=campaign_candidates,
            instruction="improve the eye",
            output_path=tmp_path / "v21_result.json",
            memory=RecordingMemory(events, error=OSError("disk full")),
        )

    assert orchestrator.calls == 0
    assert events == [("memory", eye_sample.measurement_id)]


def test_kimi_guard_hold_is_not_written_as_kimi_memory(
    tmp_path,
    eye_sample,
):
    result = brain_result(eye_sample, action=DecisionAction.HOLD)
    guarded = result.model_copy(
        update={
            "perception": result.perception.model_copy(
                update={"provider_id": "brain-guard", "provider_version": "1"}
            ),
            "decision": result.decision.model_copy(
                update={"provider_id": "brain-guard", "provider_version": "1"}
            ),
        }
    )
    events = []
    output = execute_kimi_step(
        brain=StaticBrain(guarded),
        orchestrator=OrderedOrchestrator(events),
        authorization=object(),
        sample=eye_sample,
        candidates=two_candidates(),
        instruction="improve the eye",
        output_path=tmp_path / "v21_result.json",
        memory=RecordingMemory(events),
    )

    assert output["execution_status"] == "HOLD"
    assert events == []
~~~

Update every existing execute_kimi_step call to pass a RecordingMemory. For HOLD and STOP returned directly by Kimi, assert memory is still written.

- [ ] **Step 2: Run step tests and verify RED**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_step.py -v
~~~

Expected: execute_kimi_step does not accept memory, so the new tests fail before hardware code runs.

- [ ] **Step 3: Add the memory boundary to execute and submit**

Import KIMI_PROVIDER_ID and KimiCampaignMemory. Extend execute_kimi_step:

~~~python
def execute_kimi_step(
    *,
    brain: OptimizationBrain,
    orchestrator: CandidateRunner,
    authorization: StageAuthorization,
    sample: EyeSample,
    candidates: tuple[Candidate, ...],
    instruction: str,
    output_path: Path,
    memory: KimiCampaignMemory,
) -> dict[str, object]:
~~~

Immediately after brain.analyze and before candidate selection, add:

~~~python
brain_result = brain.analyze(sample, (), safe_candidates)
if brain_result.perception.provider_id == KIMI_PROVIDER_ID:
    memory.append(sample, brain_result)
elif brain_result.decision.action is DecisionAction.PROPOSE:
    raise ValueError("non-Kimi brain result cannot execute a Kimi step")
~~~

This preserves fail-closed guard HOLD results without mislabeling them as Kimi history, while guaranteeing any executable Kimi proposal is durable before run_one_candidate.

Extend submit_kimi_step with the same required memory parameter and forward it inside the runtime lambda:

~~~python
def submit_kimi_step(
    runtime: CampaignRuntime,
    *,
    campaign_id: str,
    brain: OptimizationBrain,
    authorization: StageAuthorization,
    sample: EyeSample,
    candidates: tuple[Candidate, ...],
    instruction: str,
    output_path: Path,
    memory: KimiCampaignMemory,
) -> Future[dict[str, object]]:
~~~

- [ ] **Step 4: Run Kimi step and provider tests**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2/test_kimi_step.py tests_py/ai_tuning/v2/test_kimi_provider.py tests_py/ai_tuning/v2/test_kimi_memory.py -v
~~~

Expected: all selected tests pass; no network or hardware is called.

- [ ] **Step 5: Commit**

~~~powershell
git add src/vcsel_ai_tuning/v2/kimi_step.py tests_py/ai_tuning/v2/test_kimi_step.py
git commit -m "feat: persist Kimi decision before execution"
~~~

### Task 6: Run regression and safety-boundary verification

**Files:**
- Verify only; do not edit hardware safety ranges, hardware adapters, FIR code, or Dashboard code.

- [ ] **Step 1: Verify V2 cannot import hardware implementation modules**

Run:

~~~powershell
python -m pytest tests_py/ai_tuning/v2 -v
~~~

Expected: all V2 tests pass, including existing import-isolation tests.

- [ ] **Step 2: Run the complete Python suite**

Run:

~~~powershell
python -m pytest
~~~

Expected: the pre-existing suite remains green. The last verified baseline before this feature was 1936 passed, 3 skipped.

- [ ] **Step 3: Check repository cleanliness and accidental secret logging**

Run:

~~~powershell
git diff --check
rg -n "MOONSHOT_API_KEY|sk-[A-Za-z0-9]" src tests_py docs --glob "!docs/superpowers/plans/2026-08-09-v2-1-kimi-live-minimal.md"
git status --short
~~~

Expected: git diff --check is silent; the search finds only the environment-variable name and synthetic test keys, never a real key; status contains only intended feature files if the last test task required a fix.

- [ ] **Step 4: Commit any test-only correction, otherwise leave HEAD unchanged**

If regression testing required a correction:

~~~powershell
git add src/vcsel_ai_tuning/v2 tests_py/ai_tuning/v2
git commit -m "test: complete Kimi memory regression coverage"
~~~

If no correction was required, do not create an empty commit.
