from __future__ import annotations

import inspect
from math import inf, nan
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from pydantic import ValidationError

import vcsel_ai_tuning.v2 as v2
import vcsel_ai_tuning.v2.brain as brain_module
import vcsel_ai_tuning.v2.factory as factory_module
from vcsel_ai_tuning.v2.brain import ComposedBrain, OptimizationBrain, _request_hash, _response_hash
from vcsel_ai_tuning.v2.contracts import BrainCandidate, BrainMode, BrainResult, DecisionAction, DecisionProposal, PerceptionResult
from vcsel_ai_tuning.v2.factory import (
    BrainCapability,
    BrainFactory,
    BrainFactoryConfig,
    BrainUnavailableError,
    V2ConfigDocument,
    load_v2_config,
)
from vcsel_ai_tuning.v2.providers.fake import (
    FakeEyeScorer,
    FirstCandidateDecisionProvider,
)
from vcsel_ai_tuning.v2.replay import ReplayCase, replay


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class SpyBrain:
    provider_key = "spy"
    provider_version = "1"

    def analyze(self, sample, history, safe_candidates):
        raise AssertionError("SpyBrain.analyze must not be called in factory tests")


class RuntimeProtocolFalsePositive:
    provider_key = "provider"
    provider_version = "1"
    analyze = 42


class IdentityBrain:
    def __init__(self, provider_key, provider_version):
        self.provider_key = provider_key
        self.provider_version = provider_version

    def analyze(self, sample, history, safe_candidates):
        raise AssertionError("IdentityBrain.analyze must not be called in factory tests")


class SecretIdentityBrain:
    @property
    def provider_key(self):
        raise RuntimeError("secret-provider-identity")

    @property
    def provider_version(self):
        return "1"

    def analyze(self, sample, history, safe_candidates):
        raise AssertionError("SecretIdentityBrain.analyze must not be called")


class AdversarialBrain:
    """Generic builder result used to exercise the factory's outer gate."""

    def __init__(self, attack: str | None = None) -> None:
        self.attack = attack
        self.identity_reads = 0

    @property
    def provider_key(self):
        self.identity_reads += 1
        return "generic"

    @property
    def provider_version(self):
        self.identity_reads += 1
        return "1"

    def analyze(self, sample, history, safe_candidates):
        if self.attack == "raise":
            raise RuntimeError("untrusted detail")
        perception = PerceptionResult(
            measurement_id="foreign-measurement" if self.attack == "foreign" else sample.measurement_id,
            quality_score=0.9,
            confidence=0.9 if self.attack != "low" else 0.1,
            valid=self.attack != "invalid",
            ood=self.attack == "ood",
            defects={},
            provider_id="generic" if self.attack != "perception_identity" else "foreign",
            provider_version="1",
            input_hash=sample.raw_eye_density_ref.sha256,
        )
        decision = DecisionProposal(
            based_on_measurement_id="foreign-measurement" if self.attack == "foreign" else sample.measurement_id,
            action=DecisionAction.PROPOSE,
            candidate_id=("foreign" if self.attack == "outside" else safe_candidates[0].candidate_id),
            requested_params=(
                {"bias_ma": 999.0}
                if self.attack == "params"
                else safe_candidates[0].requested_params
            ),
            reason_codes=("UNTRUSTED",),
            provider_id="generic" if self.attack != "decision_identity" else "foreign",
            provider_version="1",
        )
        result = BrainResult(
            perception=perception,
            decision=decision,
            brain_mode=BrainMode.PURE_API if self.attack == "mode" else BrainMode.LOCAL_VISION_API,
            latency_ms=999.0,
            request_hash=("b" * 64 if self.attack == "request" else _request_hash(sample, history, safe_candidates)),
            response_hash="c" * 64,
        )
        if self.attack == "corrupt":
            return BrainResult.model_construct(
                perception=object(), decision=decision, brain_mode=BrainMode.LOCAL_VISION_API,
                latency_ms=0.0, request_hash="a" * 64, response_hash="b" * 64,
            )
        return result.model_copy(update={
            "response_hash": _response_hash(perception, decision, result.brain_mode)
            if self.attack != "response" else "c" * 64
        })


class AnalyzeDescriptorBrain(AdversarialBrain):
    """The second descriptor read is hostile; a factory must freeze the first."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze_reads = 0

    @property
    def analyze(self):
        self.analyze_reads += 1
        if self.analyze_reads > 1:
            raise AssertionError("analyze descriptor read more than once")
        return AdversarialBrain.analyze.__get__(self, type(self))


class SpoofedNativeCallable:
    """Callable object that imitates a bound method's public attributes."""

    def __init__(self, source) -> None:
        self._source = source
        self.__self__ = source
        self.__func__ = ComposedBrain.analyze

    def __call__(self, sample, history, safe_candidates):
        return ComposedBrain.analyze(self._source, sample, history, safe_candidates)


class AnyStr(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class InvalidMetadataScorer:
    provider_id = " "
    provider_version = "1"

    def score(self, sample):
        raise AssertionError("invalid metadata must prevent score")


class MutatingInputBrain:
    provider_key = "generic"
    provider_version = "1"

    def analyze(self, sample, history, safe_candidates):
        request_hash = _request_hash(sample, history, safe_candidates)
        object.__setattr__(safe_candidates[0], "candidate_id", "outside")
        object.__setattr__(safe_candidates[0], "requested_params", {"bias_ma": 999.0})
        perception = PerceptionResult(
            measurement_id=sample.measurement_id,
            quality_score=0.9,
            confidence=0.9,
            valid=True,
            ood=False,
            defects={},
            provider_id="generic",
            provider_version="1",
            input_hash=sample.raw_eye_density_ref.sha256,
        )
        decision = DecisionProposal(
            based_on_measurement_id=sample.measurement_id,
            action=DecisionAction.PROPOSE,
            candidate_id=safe_candidates[0].candidate_id,
            requested_params=safe_candidates[0].requested_params,
            reason_codes=("MUTATED",),
            provider_id="generic",
            provider_version="1",
        )
        return BrainResult(
            perception=perception,
            decision=decision,
            brain_mode=BrainMode.LOCAL_VISION_API,
            latency_ms=0.0,
            request_hash=request_hash,
            response_hash=_response_hash(perception, decision, BrainMode.LOCAL_VISION_API),
        )


def config(
    mode: BrainMode = BrainMode.LOCAL_VISION_API,
    enabled: tuple[BrainMode, ...] = (BrainMode.LOCAL_VISION_API,),
    confidence: float = 0.70,
) -> BrainFactoryConfig:
    return BrainFactoryConfig(
        brain_mode=mode,
        enabled_brain_modes=enabled,
        minimum_confidence=confidence,
    )


def test_package_exports_factory_api_without_internal_or_optional_modules():
    expected = {
        "BrainCapability": BrainCapability,
        "BrainFactory": BrainFactory,
        "BrainFactoryConfig": BrainFactoryConfig,
        "BrainUnavailableError": BrainUnavailableError,
        "load_v2_config": load_v2_config,
        "is_factory_gated_brain": brain_module.is_factory_gated_brain,
    }

    assert {name: getattr(v2, name) for name in expected} == expected
    assert set(expected).issubset(v2.__all__)
    internal_names = (
        "V2ConfigDocument",
        "FakeEyeScorer",
        "FirstCandidateDecisionProvider",
        "ReplayCase",
        "replay",
        "preprocessing",
        "evaluation",
        "providers",
    )
    assert not set(internal_names).intersection(v2.__all__)

    source_root = REPOSITORY_ROOT / "src"
    probe = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(source_root)!r})",
            "import vcsel_ai_tuning.v2 as v2",
            f"internal_names = {internal_names!r}",
            "violations = [name for name in internal_names if hasattr(v2, name)]",
            "raise SystemExit(1 if violations else 0)",
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_factory_gate_identity_helper_rejects_direct_and_accepts_exact_wrapper():
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: SpyBrain())

    assert brain_module.is_factory_gated_brain(SpyBrain()) is False
    assert brain_module.is_factory_gated_brain(
        factory.create(config())
    ) is True

    with pytest.raises(TypeError, match="BrainFactory"):
        brain_module.analyze_factory_gated(SpyBrain(), object(), (), ())

    forged = object.__new__(brain_module._FactoryGatedBrain)
    assert brain_module.is_factory_gated_brain(forged) is False
    with pytest.raises(TypeError, match="BrainFactory"):
        brain_module.analyze_factory_gated(forged, object(), (), ())


def test_factory_gated_brain_direct_construction_requires_enrollment_capability():
    source = SpyBrain()

    with pytest.raises(TypeError, match="enrollment"):
        brain_module._FactoryGatedBrain(
            source,
            BrainMode.LOCAL_VISION_API,
            0.70,
            source.provider_key,
            source.provider_version,
            source.analyze,
        )

    unenrolled = object.__new__(brain_module._FactoryGatedBrain)
    with pytest.raises(TypeError, match="enrollment"):
        brain_module._FactoryGatedBrain.__init__(
            unenrolled,
            source,
            BrainMode.LOCAL_VISION_API,
            0.70,
            source.provider_key,
            source.provider_version,
            source.analyze,
        )
    assert brain_module.is_factory_gated_brain(unenrolled) is False


def test_factory_enrollment_creator_is_not_available_from_normal_module_apis():
    creator_names = (
        "_create_factory_gated_brain",
        "_claim_factory_gate_creator",
    )

    assert not any(hasattr(brain_module, name) for name in creator_names)
    assert not any(hasattr(factory_module, name) for name in creator_names)


def test_factory_create_does_not_expose_an_enrollment_creator_parameter():
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: SpyBrain())

    assert tuple(inspect.signature(BrainFactory.create).parameters) == ("self", "config")
    assert BrainFactory.create.__defaults__ is None
    assert BrainFactory.create.__kwdefaults__ is None
    with pytest.raises(TypeError):
        factory.create(config(), lambda *_: object())
    assert brain_module.is_factory_gated_brain(factory.create(config())) is True


def test_factory_gated_brain_is_slotted_and_immutable_after_construction():
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: SpyBrain())
    brain = factory.create(config())
    state = brain_module._get_factory_gate_state(brain)

    assert not hasattr(brain, "__dict__")
    assert isinstance(state, tuple)
    for attribute, replacement in (
        ("analyze", lambda *args: None),
        ("provider_key", "spoofed-provider"),
        ("_analyze_callable", lambda *args: None),
        ("_minimum_confidence", 0.0),
        ("_provider_key", "spoofed-provider"),
        ("_provider_version", "spoofed-version"),
        ("_mode", BrainMode.PURE_API),
        ("_native_builtin", False),
        ("_built_in_profile", None),
    ):
        with pytest.raises(AttributeError):
            setattr(brain, attribute, replacement)

    for attribute, replacement in (
        ("analyze", lambda *args: None),
        ("_analyze_callable", lambda *args: None),
        ("_minimum_confidence", 0.0),
        ("_provider_key", "spoofed-provider"),
        ("_provider_version", "spoofed-version"),
        ("_mode", BrainMode.PURE_API),
        ("_built_in_profile", None),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(brain, attribute, replacement)

    for attribute, replacement in (
        ("mode", BrainMode.PURE_API),
        ("minimum_confidence", 0.0),
        ("provider_key", "spoofed-provider"),
        ("provider_version", "spoofed-version"),
        ("analyze_callable", lambda *args: None),
        ("native_builtin", False),
        ("built_in_profile", None),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(state, attribute, replacement)

    assert brain.provider_key == "spy"
    assert brain.provider_version == "1"

    with pytest.raises(AttributeError):
        del brain._analyze_callable


def test_config_converts_only_yaml_wire_types_to_brain_mode_and_tuple():
    parsed = BrainFactoryConfig.model_validate(
        {
            "brain_mode": "LOCAL_VISION_API",
            "enabled_brain_modes": ["LOCAL_VISION_API"],
            "minimum_confidence": 0.70,
        }
    )

    assert parsed.brain_mode is BrainMode.LOCAL_VISION_API
    assert parsed.enabled_brain_modes == (BrainMode.LOCAL_VISION_API,)
    assert isinstance(parsed.enabled_brain_modes, tuple)


def test_config_uses_and_validates_default_minimum_confidence():
    parsed = BrainFactoryConfig.model_validate(
        {
            "brain_mode": "LOCAL_VISION_API",
            "enabled_brain_modes": ["LOCAL_VISION_API"],
        }
    )

    assert parsed.minimum_confidence == 0.7


def test_config_rejects_a_custom_string_subclass_as_a_wire_brain_mode():
    class CustomString(str):
        pass

    with pytest.raises(ValidationError):
        BrainFactoryConfig.model_validate(
            {
                "brain_mode": CustomString("LOCAL_VISION_API"),
                "enabled_brain_modes": ["LOCAL_VISION_API"],
                "minimum_confidence": 0.7,
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (),
            "minimum_confidence": 0.7,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (
                BrainMode.LOCAL_VISION_API,
                BrainMode.LOCAL_VISION_API,
            ),
            "minimum_confidence": 0.7,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (BrainMode.PURE_API,),
            "minimum_confidence": 0.7,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (BrainMode.LOCAL_VISION_API,),
            "minimum_confidence": True,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (BrainMode.LOCAL_VISION_API,),
            "minimum_confidence": "0.7",
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (BrainMode.LOCAL_VISION_API,),
            "minimum_confidence": nan,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (BrainMode.LOCAL_VISION_API,),
            "minimum_confidence": inf,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (BrainMode.LOCAL_VISION_API,),
            "minimum_confidence": -0.01,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": (BrainMode.LOCAL_VISION_API,),
            "minimum_confidence": 1.01,
        },
        {
            "brain_mode": BrainMode.LOCAL_VISION_API,
            "enabled_brain_modes": {BrainMode.LOCAL_VISION_API},
            "minimum_confidence": 0.7,
        },
        {
            "brain_mode": 1,
            "enabled_brain_modes": (BrainMode.LOCAL_VISION_API,),
            "minimum_confidence": 0.7,
        },
    ],
)
def test_config_rejects_inconsistent_or_non_strict_values(payload):
    with pytest.raises(ValidationError):
        BrainFactoryConfig.model_validate(payload)


def test_config_is_frozen_and_forbids_extra_fields():
    configured = config()
    with pytest.raises(ValidationError):
        configured.minimum_confidence = 0.8
    with pytest.raises(ValidationError):
        BrainFactoryConfig.model_validate(
            {**configured.model_dump(), "unexpected": "value"}
        )


@pytest.mark.parametrize(
    ("installed", "enabled", "reason"),
    [
        (False, True, "NOT_INSTALLED"),
        (False, False, "NOT_INSTALLED"),
        (True, False, "NOT_ENABLED"),
        (True, True, None),
    ],
)
def test_capability_reports_consistent_state(installed, enabled, reason):
    capability = BrainCapability(
        brain_mode=BrainMode.LOCAL_VISION_API,
        installed=installed,
        enabled=enabled,
        available=installed and enabled,
        reason=reason,
    )
    assert capability.reason == reason


def test_available_capability_defaults_reason_to_none_without_bypassing_validation():
    capability = BrainCapability(
        brain_mode=BrainMode.LOCAL_VISION_API,
        installed=True,
        enabled=True,
        available=True,
    )
    assert capability.reason is None

    with pytest.raises(ValidationError):
        BrainCapability(
            brain_mode=BrainMode.LOCAL_VISION_API,
            installed=False,
            enabled=True,
            available=False,
        )


@pytest.mark.parametrize(
    ("available", "reason"),
    [(True, "NOT_INSTALLED"), (False, None), (True, "NOT_ENABLED")],
)
def test_capability_rejects_contradictory_manual_construction(available, reason):
    with pytest.raises(ValidationError):
        BrainCapability(
            brain_mode=BrainMode.LOCAL_VISION_API,
            installed=True,
            enabled=True,
            available=available,
            reason=reason,
        )


def test_document_has_only_v2_top_level_field():
    document = V2ConfigDocument.model_validate({"v2": config().model_dump()})
    assert document.v2 == config()
    with pytest.raises(ValidationError):
        V2ConfigDocument.model_validate({"v2": config().model_dump(), "other": {}})


def test_register_rejects_invalid_and_duplicate_registration():
    factory = BrainFactory()
    with pytest.raises(TypeError, match="BrainMode"):
        factory.register("LOCAL_VISION_API", lambda _: SpyBrain())
    with pytest.raises(TypeError, match="callable"):
        factory.register(BrainMode.LOCAL_VISION_API, SpyBrain())
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: SpyBrain())
    with pytest.raises(ValueError, match="already registered"):
        factory.register(BrainMode.LOCAL_VISION_API, lambda _: SpyBrain())


def test_factories_have_private_registries():
    left, right = BrainFactory(), BrainFactory()
    left.register(BrainMode.LOCAL_VISION_API, lambda _: SpyBrain())
    assert left.capabilities(config())[1].installed is True
    assert right.capabilities(config())[1].installed is False


def test_capabilities_are_ordered_complete_and_never_invoke_builders():
    calls = 0

    def builder(_: BrainFactoryConfig) -> SpyBrain:
        nonlocal calls
        calls += 1
        return SpyBrain()

    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, builder)
    capabilities = factory.capabilities(config())

    assert [item.brain_mode for item in capabilities] == list(BrainMode)
    assert len(capabilities) == 4
    assert capabilities[1].available is True
    assert capabilities[0].installed is False
    assert capabilities[0].available is False
    assert calls == 0


def test_capabilities_distinguish_disabled_from_uninstalled():
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: SpyBrain())
    configured = config(BrainMode.PURE_API, (BrainMode.PURE_API,))
    capabilities = {item.brain_mode: item for item in factory.capabilities(configured)}

    assert capabilities[BrainMode.LOCAL_VISION_API].reason == "NOT_ENABLED"
    assert capabilities[BrainMode.PURE_API].reason == "NOT_INSTALLED"


def test_factory_methods_require_brain_factory_config():
    factory = BrainFactory()
    with pytest.raises(TypeError, match="BrainFactoryConfig"):
        factory.capabilities(object())
    with pytest.raises(TypeError, match="BrainFactoryConfig"):
        factory.create(object())


def test_create_rejects_enabled_but_uninstalled_mode():
    with pytest.raises(BrainUnavailableError, match="not installed"):
        BrainFactory().create(config())


def test_create_invokes_only_selected_builder_once():
    selected_calls = 0
    unselected_calls = 0

    def selected(_: BrainFactoryConfig) -> SpyBrain:
        nonlocal selected_calls
        selected_calls += 1
        return SpyBrain()

    def unselected(_: BrainFactoryConfig) -> SpyBrain:
        nonlocal unselected_calls
        unselected_calls += 1
        return SpyBrain()

    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, selected)
    factory.register(BrainMode.PURE_API, unselected)
    result = factory.create(config())

    assert isinstance(result, OptimizationBrain)
    assert selected_calls == 1
    assert unselected_calls == 0


def test_create_propagates_builder_error_without_fallback():
    fallback_calls = 0

    def failing(_: BrainFactoryConfig):
        raise RuntimeError("builder failed")

    def fallback(_: BrainFactoryConfig) -> SpyBrain:
        nonlocal fallback_calls
        fallback_calls += 1
        return SpyBrain()

    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, failing)
    factory.register(BrainMode.PURE_API, fallback)

    with pytest.raises(RuntimeError, match="builder failed"):
        factory.create(config())
    assert fallback_calls == 0


def test_create_rejects_incompatible_builder_result():
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: object())
    with pytest.raises(TypeError, match="OptimizationBrain"):
        factory.create(config())


@pytest.mark.parametrize(
    ("brain", "secret"),
    [
        (RuntimeProtocolFalsePositive(), False),
        (IdentityBrain(42, "1"), False),
        (IdentityBrain("", "1"), False),
        (IdentityBrain("provider", "   "), False),
        (SecretIdentityBrain(), True),
    ],
)
def test_create_rejects_protocol_shaped_brains_with_unsafe_identity_or_analyze(
    brain,
    secret,
):
    selected_calls = 0
    fallback_calls = 0

    def selected(_: BrainFactoryConfig):
        nonlocal selected_calls
        selected_calls += 1
        return brain

    def fallback(_: BrainFactoryConfig) -> SpyBrain:
        nonlocal fallback_calls
        fallback_calls += 1
        return SpyBrain()

    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, selected)
    factory.register(BrainMode.PURE_API, fallback)

    with pytest.raises(TypeError) as error:
        factory.create(config())

    assert selected_calls == 1
    assert fallback_calls == 0
    if secret:
        assert "secret-provider-identity" not in str(error.value)


def test_load_v2_config_reads_only_strict_yaml_document(tmp_path: Path):
    source = tmp_path / "v2.yaml"
    source.write_text(
        "v2:\n"
        "  brain_mode: LOCAL_VISION_API\n"
        "  enabled_brain_modes:\n"
        "    - LOCAL_VISION_API\n"
        "  minimum_confidence: 0.70\n",
        encoding="utf-8",
    )
    assert load_v2_config(source) == config()


def test_load_v2_config_uses_default_minimum_confidence(tmp_path: Path):
    source = tmp_path / "v2-default.yaml"
    source.write_text(
        "v2:\n"
        "  brain_mode: LOCAL_VISION_API\n"
        "  enabled_brain_modes:\n"
        "    - LOCAL_VISION_API\n",
        encoding="utf-8",
    )

    assert load_v2_config(source).minimum_confidence == 0.7


@pytest.mark.parametrize(
    "contents",
    [
        "v2:\n"
        "  brain_mode: LOCAL_VISION_API\n"
        "  enabled_brain_modes: [LOCAL_VISION_API]\n"
        "v2:\n"
        "  brain_mode: LOCAL_VISION_API\n"
        "  enabled_brain_modes: [LOCAL_VISION_API]\n",
        "v2:\n"
        "  brain_mode: LOCAL_VISION_API\n"
        "  brain_mode: PURE_API\n"
        "  enabled_brain_modes: [LOCAL_VISION_API]\n",
        "v2:\n"
        "  brain_mode: LOCAL_VISION_API\n"
        "  enabled_brain_modes: [LOCAL_VISION_API]\n"
        "  enabled_brain_modes: [PURE_API]\n",
        "v2:\n"
        "  brain_mode: LOCAL_VISION_API\n"
        "  enabled_brain_modes: [LOCAL_VISION_API]\n"
        "  minimum_confidence: 0.7\n"
        "  minimum_confidence: 0.8\n",
    ],
)
def test_load_v2_config_rejects_duplicate_yaml_keys(tmp_path: Path, contents: str):
    source = tmp_path / "duplicate.yaml"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_v2_config(source)


def test_load_v2_config_allows_unambiguous_safe_anchor_and_alias(tmp_path: Path):
    source = tmp_path / "alias.yaml"
    source.write_text(
        "v2:\n"
        "  brain_mode: &selected LOCAL_VISION_API\n"
        "  enabled_brain_modes: [*selected]\n"
        "  minimum_confidence: 0.7\n",
        encoding="utf-8",
    )

    assert load_v2_config(source) == config()


@pytest.mark.parametrize("contents", ["", "- v2\n", "0.7\n", "plain scalar\n"])
def test_load_v2_config_rejects_every_non_mapping_root_with_value_error(
    tmp_path: Path,
    contents: str,
):
    source = tmp_path / "not-a-mapping.yaml"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_v2_config(source)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "- v2\n",
        "unknown: value\n",
        "v2:\n  brain_mode: NOT_A_MODE\n  enabled_brain_modes: [NOT_A_MODE]\n  minimum_confidence: 0.7\n",
        "v2:\n  brain_mode: LOCAL_VISION_API\n  enabled_brain_modes: [LOCAL_VISION_API, LOCAL_VISION_API]\n  minimum_confidence: 0.7\n",
        "v2:\n  brain_mode: LOCAL_VISION_API\n  enabled_brain_modes: [PURE_API]\n  minimum_confidence: 0.7\n",
        "v2:\n  brain_mode: LOCAL_VISION_API\n  enabled_brain_modes: [LOCAL_VISION_API]\n  minimum_confidence: true\n",
        "v2:\n  brain_mode: LOCAL_VISION_API\n  enabled_brain_modes: [LOCAL_VISION_API]\n  minimum_confidence: 0.7\n  api_key: secret\n",
    ],
)
def test_load_v2_config_rejects_malformed_or_unknown_yaml(tmp_path: Path, contents: str):
    source = tmp_path / "invalid.yaml"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises((ValueError, ValidationError)):
        load_v2_config(source)


def test_example_yaml_is_safe_and_enables_only_local_vision_api():
    example = REPOSITORY_ROOT / "configs" / "ai_tuning" / "v2.example.yaml"
    contents = example.read_text(encoding="utf-8")
    loaded = load_v2_config(example)

    assert loaded.enabled_brain_modes == (BrainMode.LOCAL_VISION_API,)
    assert "api_key" not in contents.lower()
    assert "model_path" not in contents.lower()


def test_factory_replays_example_config_with_local_fake_brain(
    eye_sample,
    brain_candidates,
):
    configured = load_v2_config(
        REPOSITORY_ROOT / "configs" / "ai_tuning" / "v2.example.yaml"
    )
    factory = BrainFactory()
    factory.register(
        BrainMode.LOCAL_VISION_API,
        lambda selected: ComposedBrain(
            BrainMode.LOCAL_VISION_API,
            FakeEyeScorer(quality_score=0.8, confidence=0.95),
            FirstCandidateDecisionProvider(),
            selected.minimum_confidence,
        ),
    )
    second_sample = eye_sample.model_copy(
        update={
            "measurement_id": "measurement-002",
            "sweep_point_id": "point-002",
        }
    )

    records = replay(
        factory.create(configured),
        (
            ReplayCase(sample=eye_sample, safe_candidates=brain_candidates),
            ReplayCase(sample=second_sample, safe_candidates=brain_candidates),
        ),
    )
    capabilities = factory.capabilities(configured)

    assert len(records) == 2
    assert [record.measurement_id for record in records] == [
        "measurement-001",
        "measurement-002",
    ]
    assert [record.result.decision.action for record in records] == [
        DecisionAction.PROPOSE,
        DecisionAction.PROPOSE,
    ]
    assert [record.result.decision.candidate_id for record in records] == [
        brain_candidates[0].candidate_id,
        brain_candidates[0].candidate_id,
    ]
    assert [record.result.decision.requested_params for record in records] == [
        brain_candidates[0].requested_params,
        brain_candidates[0].requested_params,
    ]
    assert [capability.brain_mode for capability in capabilities] == list(BrainMode)
    assert [capability.available for capability in capabilities] == [
        False,
        True,
        False,
        False,
    ]
    assert [capability.installed for capability in capabilities] == [
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("attack", "reason"),
    [
        ("outside", "CANDIDATE_NOT_IN_INPUT_SET"),
        ("params", "CANDIDATE_NOT_IN_INPUT_SET"),
        ("invalid", "INVALID_EYE"),
        ("ood", "OOD"),
        ("low", "LOW_CONFIDENCE"),
        ("foreign", "MEASUREMENT_ID_MISMATCH"),
        ("mode", "BRAIN_MODE_MISMATCH"),
        ("request", "REQUEST_HASH_MISMATCH"),
        ("response", "RESPONSE_HASH_MISMATCH"),
        ("perception_identity", "PROVIDER_IDENTITY_MISMATCH"),
        ("decision_identity", "PROVIDER_IDENTITY_MISMATCH"),
        ("corrupt", "INVALID_BRAIN_RESULT_SCHEMA"),
        ("raise", "PROVIDER_ERROR"),
    ],
)
def test_factory_outer_gate_turns_generic_adversarial_results_into_guard_holds(
    attack, reason, eye_sample, brain_candidates
):
    source = AdversarialBrain(attack)
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)

    result = factory.create(config()).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == (reason,)
    assert result.decision.candidate_id is result.decision.requested_params is None
    assert "untrusted detail" not in result.model_dump_json()


def test_factory_outer_gate_snapshots_generic_identity_and_canonicalizes_legal_result(
    eye_sample, brain_candidates
):
    source = AdversarialBrain()
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)
    brain = factory.create(config())

    result = brain.analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.PROPOSE
    assert result.request_hash == _request_hash(eye_sample, (), brain_candidates)
    assert result.response_hash == _response_hash(result.perception, result.decision, BrainMode.LOCAL_VISION_API)
    assert source.identity_reads == 2


def test_factory_freezes_the_validated_analyze_descriptor_once(eye_sample, brain_candidates):
    source = AnalyzeDescriptorBrain()
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)
    brain = factory.create(config())

    result = brain.analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.PROPOSE
    assert source.analyze_reads == 1


@pytest.mark.parametrize("proposal", [True, False])
def test_instance_overridden_exact_composed_analyze_cannot_claim_builtin_trust(
    proposal, eye_sample, brain_candidates
):
    source = ComposedBrain(
        BrainMode.LOCAL_VISION_API, FakeEyeScorer(), FirstCandidateDecisionProvider(), 0.7
    )
    native = ComposedBrain(
        BrainMode.LOCAL_VISION_API,
        FakeEyeScorer(confidence=0.9 if proposal else 0.1),
        FirstCandidateDecisionProvider(),
        0.7,
    )
    source.analyze = native.analyze
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)

    result = factory.create(config()).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_IDENTITY_MISMATCH",)


def test_spoofed_callable_attributes_cannot_claim_builtin_trust(eye_sample, brain_candidates):
    source = ComposedBrain(
        BrainMode.LOCAL_VISION_API, FakeEyeScorer(), FirstCandidateDecisionProvider(), 0.7
    )
    source.analyze = SpoofedNativeCallable(source)
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)

    result = factory.create(config()).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_IDENTITY_MISMATCH",)


def test_native_composed_low_confidence_hold_keeps_its_reason_through_factory(
    eye_sample, brain_candidates
):
    source = ComposedBrain(
        BrainMode.LOCAL_VISION_API,
        FakeEyeScorer(confidence=0.1),
        FirstCandidateDecisionProvider(),
        0.7,
    )
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)

    result = factory.create(config()).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("LOW_CONFIDENCE",)


def test_factory_builtin_profile_is_bound_to_frozen_public_identity(
    eye_sample, brain_candidates
):
    scorer = FakeEyeScorer()
    decision = FirstCandidateDecisionProvider()
    source = ComposedBrain(BrainMode.LOCAL_VISION_API, scorer, decision, 0.7)
    # Simulate a hostile in-process mutation after the built-in public identity
    # was frozen, but before Factory snapshots that public identity.
    scorer.provider_id, scorer.provider_version = "spoof-score", "2"
    decision.provider_id, decision.provider_version = "spoof-decision", "2"
    source._provider_ids = ("spoof-score", "spoof-decision")
    source._provider_versions = ("2", "2")
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)

    result = factory.create(config()).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_IDENTITY_MISMATCH",)


def test_factory_rejects_string_subclass_provider_identity():
    factory = BrainFactory()
    factory.register(
        BrainMode.LOCAL_VISION_API,
        lambda _: IdentityBrain(AnyStr("claimed"), "1"),
    )

    with pytest.raises(TypeError, match="OptimizationBrain"):
        factory.create(config())


def test_factory_preserves_native_invalid_metadata_provider_error_hold(
    eye_sample, brain_candidates
):
    source = ComposedBrain(
        BrainMode.LOCAL_VISION_API,
        InvalidMetadataScorer(),
        FirstCandidateDecisionProvider(),
        0.7,
    )
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)

    result = factory.create(config()).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("PROVIDER_ERROR",)


@pytest.mark.parametrize(
    "bad_config",
    [
        lambda: config().model_copy(update={"minimum_confidence": -1.0}),
        lambda: config().model_copy(update={"minimum_confidence": nan}),
        lambda: config().model_copy(update={"brain_mode": "INVALID"}),
        lambda: BrainFactoryConfig.model_construct(
            brain_mode=BrainMode.LOCAL_VISION_API,
            enabled_brain_modes=(BrainMode.LOCAL_VISION_API,),
            minimum_confidence=-1.0,
        ),
    ],
)
def test_factory_revalidates_config_escape_hatches_before_builder_or_capabilities(bad_config):
    builder_calls = 0

    def builder(_):
        nonlocal builder_calls
        builder_calls += 1
        return SpyBrain()

    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, builder)

    with pytest.raises((TypeError, ValueError, ValidationError)):
        factory.create(bad_config())
    with pytest.raises((TypeError, ValueError, ValidationError)):
        factory.capabilities(bad_config())
    assert builder_calls == 0


def test_factory_gate_uses_unexposed_input_snapshot_for_hash_and_candidate_validation(
    eye_sample, brain_candidates
):
    original_candidate = brain_candidates[0].model_dump(round_trip=True)
    source = MutatingInputBrain()
    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: source)

    result = factory.create(config()).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert brain_candidates[0].model_dump(round_trip=True) == original_candidate


def test_builder_cannot_mutate_factory_config_snapshot_to_lower_the_gate(
    eye_sample, brain_candidates
):
    supplied = config(confidence=0.8)

    def builder(builder_config):
        object.__setattr__(builder_config, "minimum_confidence", -1.0)
        return AdversarialBrain("low")

    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, builder)

    result = factory.create(supplied).analyze(eye_sample, (), brain_candidates)

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("LOW_CONFIDENCE",)
    assert supplied.minimum_confidence == 0.8


@pytest.mark.parametrize(("candidate_value", "decision_value"), [(1, 1.0), (1.0, 1)])
def test_factory_generic_requires_json_exact_candidate_parameters(
    candidate_value, decision_value, eye_sample
):
    candidate = BrainCandidate(candidate_id="numeric", requested_params={"bias_ma": candidate_value})

    class NumericMismatchBrain:
        provider_key = "numeric"
        provider_version = "1"

        def analyze(self, sample, history, safe_candidates):
            perception = PerceptionResult(
                measurement_id=sample.measurement_id, quality_score=0.9, confidence=0.9,
                valid=True, ood=False, defects={}, provider_id="numeric", provider_version="1",
                input_hash=sample.raw_eye_density_ref.sha256,
            )
            decision = DecisionProposal(
                based_on_measurement_id=sample.measurement_id, action=DecisionAction.PROPOSE,
                candidate_id="numeric", requested_params={"bias_ma": decision_value},
                reason_codes=("NUMERIC",), provider_id="numeric", provider_version="1",
            )
            return BrainResult(
                perception=perception, decision=decision, brain_mode=BrainMode.LOCAL_VISION_API,
                latency_ms=0.0, request_hash=_request_hash(sample, history, safe_candidates),
                response_hash=_response_hash(perception, decision, BrainMode.LOCAL_VISION_API),
            )

    factory = BrainFactory()
    factory.register(BrainMode.LOCAL_VISION_API, lambda _: NumericMismatchBrain())
    result = factory.create(config()).analyze(eye_sample, (), (candidate,))

    assert result.decision.action is DecisionAction.HOLD
    assert result.decision.reason_codes == ("CANDIDATE_NOT_IN_INPUT_SET",)


def test_factory_import_does_not_load_hardware_network_or_optional_provider_sdks():
    source_root = Path(__file__).resolve().parents[3] / "src"
    probe = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(source_root)!r})",
            "import vcsel_ai_tuning.v2.factory",
            "forbidden_prefixes = ('pyvisa', 'requests', 'httpx', 'openai', 'torch', 'cv2')",
            "violations = sorted("
            "name for name in sys.modules "
            "if any(name == prefix or name.startswith(f'{prefix}.') "
            "for prefix in forbidden_prefixes)"
            ")",
            "if violations:",
            "    print(', '.join(violations), file=sys.stderr)",
            "    raise SystemExit(1)",
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
