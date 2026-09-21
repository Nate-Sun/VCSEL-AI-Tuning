# V2.1 Restricted B0 Parameter Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the continuous LIVE B0 with the approved version 2 values and enforce the approved campaign-specific I2C ranges in both Kimi context and local pre-execution validation.

**Architecture:** Extend the existing continuous `ParameterRule` to distinguish hardware encoding bounds from narrower campaign tuning bounds, then build the effective policy from the LIVE campaign YAML and the already-approved hardware parameter specifications. Pass that one policy object through preflight and commission so Kimi sees exactly the limits that local candidate materialization enforces; leave the hardware profile and low-level adapters unchanged as an independent outer boundary.

**Tech Stack:** Python 3.12, Pydantic 2, PyYAML, pytest, existing V2.1 continuous controller and campaign authorization chain.

---

## File map

- Modify `src/vcsel_ai_tuning/v2/continuous/policy.py`: represent tuning bounds, parse restricted campaign policy, and reject proposals outside it.
- Modify `src/vcsel_ai_tuning/v2/providers/kimi_client.py`: serialize tuning bounds in Kimi's structured context.
- Modify `src/vcsel_ai_tuning/continuous_live.py`: require, validate, and reuse one explicit policy for PRE-FLIGHT and COMMISSION.
- Add `configs/ai_tuning/b0.v2_1.continuous.yaml`: install the approved B0 version 2 I2C values without changing the shared legacy B0.
- Modify `configs/ai_tuning/runtime.v2_1.continuous.live.yaml`: point only the V2.1 continuous runtime at the campaign-specific B0.
- Modify `configs/ai_tuning/v2_1_continuous.live.yaml`: install the approved restricted policy and fresh campaign ID.
- Modify `tests_py/ai_tuning/v2/continuous/test_policy.py`: cover policy construction and local enforcement.
- Modify `tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py`: prove Kimi receives the restricted bounds.
- Modify `tests_py/ai_tuning/v2/continuous/test_controller.py`: prove the effective policy is persisted in campaign evidence.
- Modify `tests_py/ai_tuning/test_continuous_live.py`: prove LIVE composition fails closed and uses the configured policy and B0.

No hardware profile, register map, I2C adapter, DCA adapter, BERT adapter, FIR/NLC path, or `setFir` path is modified.

Use this Python executable for every test command:

```powershell
$VcselPython = 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

### Task 1: Add restricted tuning bounds to the deterministic policy

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/continuous/policy.py:23-163`
- Test: `tests_py/ai_tuning/v2/continuous/test_policy.py`

- [ ] **Step 1: Add approved policy fixtures and failing construction tests**

Add these fixtures below `NOW` in `test_policy.py`:

```python
APPROVED_B0 = {
    "ibias_dac": 60,
    "efin_cur": 15,
    "vga_cur": 13,
    "eq": 16,
    "res_gain": 6,
    "res_bw": 0,
    "gain_ctrl": 2,
    "vbias_ref": 15,
    "vref": 15,
}

APPROVED_POLICY_CONFIG = {
    "version": "i2c-restricted-v1",
    "max_changed_parameters": 3,
    "rules": {
        "ibias_dac": {"minimum": 60, "maximum": 80, "max_step": 10},
        "efin_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
        "vga_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
        "eq": {"minimum": 16, "maximum": 25, "max_step": 3},
        "res_gain": {"minimum": 0, "maximum": 10, "max_step": 3},
        "res_bw": {"minimum": 0, "maximum": 5, "max_step": 1},
        "gain_ctrl": {"minimum": 0, "maximum": 10, "max_step": 3},
        "vbias_ref": {"minimum": 10, "maximum": 15, "max_step": 3},
        "vref": {"minimum": 10, "maximum": 15, "max_step": 3},
    },
}
```

Import `deepcopy` and the new `build_restricted_policy`, then add:

```python
def _restricted_policy():
    return build_restricted_policy(
        APPROVED_POLICY_CONFIG,
        authorization_specs=DEFAULT_I2C_POLICY.authorization_specs(),
        baseline_i2c=APPROVED_B0,
    )


def test_build_restricted_policy_preserves_encoding_and_applies_tuning_bounds() -> None:
    policy = _restricted_policy()

    assert policy.version == "i2c-restricted-v1"
    assert policy.max_changed_parameters == 3
    assert {
        rule.name: (
            rule.encoding_min,
            rule.encoding_max,
            rule.tuning_min,
            rule.tuning_max,
            rule.max_step,
        )
        for rule in policy.rules
    } == {
        "ibias_dac": (0, 127, 60, 80, 10),
        "gain_ctrl": (0, 63, 0, 10, 3),
        "eq": (0, 31, 16, 25, 3),
        "efin_cur": (0, 15, 10, 15, 3),
        "vga_cur": (0, 15, 10, 15, 3),
        "res_gain": (0, 15, 0, 10, 3),
        "res_bw": (0, 15, 0, 5, 1),
        "vbias_ref": (0, 15, 10, 15, 3),
        "vref": (0, 15, 10, 15, 3),
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload["rules"].pop("eq"), "exactly"),
        (
            lambda payload: payload["rules"]["eq"].update(minimum=26),
            "minimum",
        ),
        (
            lambda payload: payload["rules"]["eq"].update(maximum=32),
            "authorization",
        ),
        (
            lambda payload: payload["rules"]["res_bw"].update(max_step=4),
            "max_step",
        ),
    ],
)
def test_build_restricted_policy_rejects_invalid_campaign_limits(
    mutator,
    message: str,
) -> None:
    payload = deepcopy(APPROVED_POLICY_CONFIG)
    mutator(payload)

    with pytest.raises(ValueError, match=message):
        build_restricted_policy(
            payload,
            authorization_specs=DEFAULT_I2C_POLICY.authorization_specs(),
            baseline_i2c=APPROVED_B0,
        )


def test_build_restricted_policy_rejects_b0_outside_tuning_range() -> None:
    baseline = dict(APPROVED_B0, eq=15)

    with pytest.raises(ValueError, match="B0.*eq"):
        build_restricted_policy(
            APPROVED_POLICY_CONFIG,
            authorization_specs=DEFAULT_I2C_POLICY.authorization_specs(),
            baseline_i2c=baseline,
        )
```

- [ ] **Step 2: Run the new policy tests and verify they fail**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_policy.py -q
```

Expected: FAIL during import because `build_restricted_policy` and tuning-bound fields do not exist.

- [ ] **Step 3: Extend `ParameterRule` and implement strict config construction**

In `policy.py`, add `OUTSIDE_TUNING_RANGE` to `ProposalErrorCode`, replace `ParameterRule`, update `_rule`, and add the builder below `DEFAULT_I2C_POLICY`:

```python
class ProposalErrorCode(StrEnum):
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    NON_INTEGER_VALUE = "NON_INTEGER_VALUE"
    OUTSIDE_ENCODING_RANGE = "OUTSIDE_ENCODING_RANGE"
    OUTSIDE_TUNING_RANGE = "OUTSIDE_TUNING_RANGE"
    EXCEEDS_MAX_STEP = "EXCEEDS_MAX_STEP"
    TOO_MANY_CHANGED_PARAMETERS = "TOO_MANY_CHANGED_PARAMETERS"
    NO_EFFECTIVE_CHANGE = "NO_EFFECTIVE_CHANGE"
    MISSING_READBACK = "MISSING_READBACK"


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
```

Update `_rule` so the default policy keeps its existing full range:

```python
def _rule(name: str, maximum: int) -> ParameterRule:
    return ParameterRule(
        name=name,
        encoding_min=0,
        encoding_max=maximum,
        tuning_min=0,
        tuning_max=maximum,
        max_step=ceil(0.15 * maximum),
    )
```

Add this builder after `DEFAULT_I2C_POLICY`:

```python
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
        raise ValueError("parameter_policy must contain exactly version, max_changed_parameters, and rules")
    raw_rules = config["rules"]
    if not isinstance(raw_rules, Mapping):
        raise TypeError("parameter_policy rules must be a mapping")

    specs = {spec.name: spec for spec in authorization_specs if spec.enabled}
    if set(raw_rules) != set(specs):
        raise ValueError("parameter_policy rules must exactly match enabled authorization parameters")

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
            raise ValueError(f"parameter_policy rule {name} exceeds authorization tuning range")
        if max_step <= 0 or max_step > spec.max_step:
            raise ValueError(f"parameter_policy rule {name} max_step exceeds authorization")
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
```

- [ ] **Step 4: Enforce tuning bounds before the step check**

In `I2cParameterPolicy.validate_proposal`, immediately after the encoding-range check, add:

```python
            if not rule.tuning_min <= value <= rule.tuning_max:
                raise ProposalValidationError(
                    ProposalErrorCode.OUTSIDE_TUNING_RANGE,
                    name,
                )
```

Add these tests:

```python
def test_restricted_policy_rejects_value_inside_encoding_but_outside_tuning() -> None:
    with pytest.raises(ProposalValidationError) as captured:
        _restricted_policy().validate_proposal(
            {"eq": 26},
            APPROVED_B0,
        )

    assert captured.value.code is ProposalErrorCode.OUTSIDE_TUNING_RANGE


def test_restricted_policy_treats_max_step_as_delta_not_grid_quantum() -> None:
    sparse = _restricted_policy().validate_proposal(
        {"ibias_dac": 67, "res_bw": 1},
        APPROVED_B0,
    )

    assert sparse == {"ibias_dac": 67, "res_bw": 1}
```

- [ ] **Step 5: Run policy tests and verify they pass**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_policy.py -q
```

Expected: all tests in `test_policy.py` PASS.

- [ ] **Step 6: Commit the policy model**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/policy.py tests_py/ai_tuning/v2/continuous/test_policy.py
git commit -m "feat(v2.1): add restricted continuous I2C policy"
```

### Task 2: Send restricted tuning bounds to Kimi

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/providers/kimi_client.py:481-493`
- Test: `tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py:90-154`

- [ ] **Step 1: Write a failing structured-context test**

Import `I2cParameterPolicy` and `ParameterRule`. Allow `_context` to accept a `policy` argument defaulting to `DEFAULT_I2C_POLICY` and pass that argument into `ProviderContext`:

```python
def _context(
    sample,
    *,
    best_reference=None,
    hardware_round: int = 1,
    recent_valid_summaries=(),
    policy=DEFAULT_I2C_POLICY,
) -> ProviderContext:
    return ProviderContext(
        sample=sample,
        verified_readback={"eq": 22, "gain_ctrl": 5},
        policy=policy,
        hardware_round=hardware_round,
        max_rounds=3,
        target_tdecq_db=2.7,
        recent_valid_summaries=recent_valid_summaries,
        best_reference=best_reference,
        allowed_output_schema={"version": "provider-action-v1"},
    )
```

Add this focused serialization test:

```python
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
```

- [ ] **Step 2: Run the Kimi test and verify it fails**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py::test_continuous_client_sends_restricted_tuning_bounds -q
```

Expected: FAIL because `tuning_min` and `tuning_max` are missing from the serialized context.

- [ ] **Step 3: Serialize tuning bounds and make the system instruction explicit**

Change each rule document in `kimi_client.py` to:

```python
                    rule.name: {
                        "encoding_min": rule.encoding_min,
                        "encoding_max": rule.encoding_max,
                        "tuning_min": rule.tuning_min,
                        "tuning_max": rule.tuning_max,
                        "max_step": rule.max_step,
                        "integer_only": True,
                    }
```

Change the PROPOSE instruction to:

```python
                        "For PROPOSE, provide sparse absolute integer I2C values within "
                        "each supplied tuning_min and tuning_max and within max_step from "
                        "the latest verified readback. "
```

- [ ] **Step 4: Run the focused and complete Kimi continuous tests**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py -q
```

Expected: all tests in `test_kimi_continuous.py` PASS.

- [ ] **Step 5: Commit the Kimi boundary change**

```powershell
git add src/vcsel_ai_tuning/v2/providers/kimi_client.py tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py
git commit -m "feat(v2.1): send restricted I2C bounds to Kimi"
```

### Task 3: Require and reuse the LIVE campaign policy

**Files:**
- Modify: `src/vcsel_ai_tuning/continuous_live.py:11-258`
- Test: `tests_py/ai_tuning/test_continuous_live.py`
- Test: `tests_py/ai_tuning/v2/continuous/test_controller.py`

- [ ] **Step 1: Add a reusable test policy configuration**

Add this helper near the top of `test_continuous_live.py`:

```python
def _policy_config() -> dict[str, object]:
    return {
        "version": "i2c-restricted-v1",
        "max_changed_parameters": 3,
        "rules": {
            "ibias_dac": {"minimum": 60, "maximum": 80, "max_step": 10},
            "efin_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
            "vga_cur": {"minimum": 10, "maximum": 15, "max_step": 3},
            "eq": {"minimum": 16, "maximum": 25, "max_step": 3},
            "res_gain": {"minimum": 0, "maximum": 10, "max_step": 3},
            "res_bw": {"minimum": 0, "maximum": 5, "max_step": 1},
            "gain_ctrl": {"minimum": 0, "maximum": 10, "max_step": 3},
            "vbias_ref": {"minimum": 10, "maximum": 15, "max_step": 3},
            "vref": {"minimum": 10, "maximum": 15, "max_step": 3},
        },
    }
```

Add `"parameter_policy": _policy_config()` to every direct PRE-FLIGHT and COMMISSION config dictionary in this test file, except the new missing-policy case below.

- [ ] **Step 2: Write a failing fail-closed test**

Add:

```python
def test_live_preflight_requires_explicit_parameter_policy_before_contact(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.providers.kimi_client.KimiApiClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Kimi contacted")),
    )
    monkeypatch.setattr(
        "vcsel_ai_tuning.hardware.live_runtime_factory.create_live_api_runtime_from_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hardware runtime opened")
        ),
    )

    with pytest.raises(RuntimeError, match="parameter_policy"):
        preflight_continuous_live(
            {"runtime_config": "runtime.yaml", "campaign_id": "campaign-1"},
            tmp_path,
        )
```

- [ ] **Step 3: Run the fail-closed test and verify it fails**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py::test_live_preflight_requires_explicit_parameter_policy_before_contact -q
```

Expected: FAIL because the current code proceeds to construct `KimiApiClient` without requiring the policy.

- [ ] **Step 4: Add one policy assembly helper and use it in both paths**

At module level in `continuous_live.py`, add:

```python
def _require_policy_config(config: Mapping[str, object]) -> Mapping[str, object]:
    policy = config.get("parameter_policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("continuous LIVE parameter_policy is required")
    return policy


def _build_live_policy(
    raw_policy: Mapping[str, object],
    *,
    baseline: Any,
    parameter_specs: tuple[Any, ...],
) -> Any:
    from vcsel_ai_tuning.v2.continuous.policy import build_restricted_policy

    return build_restricted_policy(
        raw_policy,
        authorization_specs=parameter_specs,
        baseline_i2c=baseline.i2c.model_dump(),
    )
```

In both `preflight_continuous_live` and `activate_continuous_live`, call `_require_policy_config(config)` before constructing Kimi or the runtime.

Import and call `build_continuous_live_tuning_profile` after loading B0 and before `runtime.entry(...)`:

```python
        tuning_profile = build_continuous_live_tuning_profile(
            runtime.config.hardware.driver_profile,
            baseline,
        )
        policy = _build_live_policy(
            raw_policy,
            baseline=baseline,
            parameter_specs=tuning_profile.parameter_specs,
        )
```

In commission, explicitly load the baseline from `runtime.config.b0_path` before building the policy. Replace both uses of `DEFAULT_I2C_POLICY.model_dump(mode="json")` with:

```python
            policy=policy.model_dump(mode="json"),
```

Remove the now-unused `DEFAULT_I2C_POLICY` imports from both functions.

- [ ] **Step 5: Prove preflight and commission carry the same restricted policy**

In the existing preflight bootstrap test, add `captured_policies = []`, append `context.policy` inside `Provider.decide`, and assert:

```python
    assert captured_policies[0].by_name["eq"].tuning_min == 16
    assert captured_policies[0].by_name["eq"].tuning_max == 25
    assert captured_policies[0].by_name["res_bw"].max_step == 1
```

In an existing commission assembly test, assert the returned request contains:

```python
    effective = I2cParameterPolicy.model_validate(dependencies.request.policy)
    assert effective.version == "i2c-restricted-v1"
    assert effective.by_name["ibias_dac"].tuning_min == 60
    assert effective.by_name["ibias_dac"].tuning_max == 80
    assert effective.by_name["gain_ctrl"].max_step == 3
```

In `test_controller_completes_max_rounds_only_after_durable_valid_evidence`, import `json`, create a restricted request before `controller.start`, and assert the exact request policy is written to `campaign.json`:

```python
    policy_payload = DEFAULT_I2C_POLICY.model_dump(mode="json")
    policy_payload["version"] = "i2c-restricted-v1"
    ibias_rule = next(
        rule for rule in policy_payload["rules"] if rule["name"] == "ibias_dac"
    )
    ibias_rule.update(tuning_min=60, tuning_max=80, max_step=10)
    request = _request(tmp_path, initial, max_rounds=2).model_copy(
        update={"policy": policy_payload}
    )

    result = controller.start(request)

    campaign = json.loads(
        (tmp_path / "results" / "campaign-1" / "campaign.json").read_text(
            encoding="utf-8"
        )
    )
    assert campaign["policy"] == policy_payload
```

- [ ] **Step 6: Run LIVE composition tests**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/v2/continuous/test_controller.py -q
```

Expected: both test modules PASS, including the no-contact missing-policy test and exact policy persistence assertion.

- [ ] **Step 7: Commit LIVE policy composition**

```powershell
git add src/vcsel_ai_tuning/continuous_live.py tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/v2/continuous/test_controller.py
git commit -m "feat(v2.1): load restricted policy for live campaigns"
```

### Task 4: Install the approved B0 and checked-in LIVE policy

**Files:**
- Add: `configs/ai_tuning/b0.v2_1.continuous.yaml`
- Modify: `configs/ai_tuning/runtime.v2_1.continuous.live.yaml:8`
- Modify: `configs/ai_tuning/v2_1_continuous.live.yaml:1-17`
- Test: `tests_py/ai_tuning/test_continuous_live.py`

- [ ] **Step 1: Write a failing checked-in configuration test**

Add imports for `build_continuous_live_tuning_profile` and `build_restricted_policy`, then add:

```python
def test_checked_in_v21_live_config_has_approved_b0_and_policy() -> None:
    document = yaml.safe_load(
        Path("configs/ai_tuning/v2_1_continuous.live.yaml").read_text(
            encoding="utf-8"
        )
    )
    live = document["live"]
    runtime_document = yaml.safe_load(
        Path(live["runtime_config"]).read_text(encoding="utf-8")
    )
    baseline_path = (
        Path(live["runtime_config"]).parent / runtime_document["b0_path"]
    ).resolve()
    baseline = load_b0(baseline_path)
    profile_path = (
        Path(live["runtime_config"]).parent
        / runtime_document["hardware"]["driver_profile"]
    ).resolve()
    profile = build_continuous_live_tuning_profile(profile_path, baseline)
    policy = build_restricted_policy(
        live["parameter_policy"],
        authorization_specs=profile.parameter_specs,
        baseline_i2c=baseline.i2c.model_dump(),
    )

    assert baseline.version == 2
    assert baseline.i2c.model_dump() == {
        "ibias_dac": 60,
        "vref": 15,
        "vbias_ref": 15,
        "vga_cur": 13,
        "res_gain": 6,
        "res_bw": 0,
        "gain_ctrl": 2,
        "efin_cur": 15,
        "eq": 16,
    }
    assert live["campaign_id"] == "v21-pf-20260815-01"
    assert policy.version == "i2c-restricted-v1"
    assert policy.max_changed_parameters == 3
```

- [ ] **Step 2: Run the checked-in configuration test and verify it fails**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py::test_checked_in_v21_live_config_has_approved_b0_and_policy -q
```

Expected: FAIL because B0 is version 1, the old values remain, and the LIVE YAML has no `parameter_policy`.

- [ ] **Step 3: Add `b0.v2_1.continuous.yaml` exactly and reference it from the V2.1 continuous runtime**

Keep the existing FIR/NLC block and replace the version and I2C block with:

```yaml
version: 2
i2c:
  ibias_dac: 60
  vref: 15
  vbias_ref: 15
  vga_cur: 13
  res_gain: 6
  res_bw: 0
  gain_ctrl: 2
  efin_cur: 15
  eq: 16
```

- [ ] **Step 4: Update the LIVE YAML exactly**

Change the used campaign ID to the verified-unused ID `v21-pf-20260815-01`. Under `live`, after `kimi_model`, add:

```yaml
  parameter_policy:
    version: i2c-restricted-v1
    max_changed_parameters: 3
    rules:
      ibias_dac: {minimum: 60, maximum: 80, max_step: 10}
      efin_cur: {minimum: 10, maximum: 15, max_step: 3}
      vga_cur: {minimum: 10, maximum: 15, max_step: 3}
      eq: {minimum: 16, maximum: 25, max_step: 3}
      res_gain: {minimum: 0, maximum: 10, max_step: 3}
      res_bw: {minimum: 0, maximum: 5, max_step: 1}
      gain_ctrl: {minimum: 0, maximum: 10, max_step: 3}
      vbias_ref: {minimum: 10, maximum: 15, max_step: 3}
      vref: {minimum: 10, maximum: 15, max_step: 3}
```

Do not change the instruction, round count, duration, Kimi model, runtime path, DCA resource, I2C endpoint, hardware profile, FIR/NLC data, or schema-required BERT endpoint metadata.

- [ ] **Step 5: Run the configuration and baseline tests**

Run:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py::test_checked_in_v21_live_config_has_approved_b0_and_policy tests_py/ai_tuning/campaign/test_models.py -q
```

Expected: PASS. No test may contact DCA, I2C, BERT, Kimi, or another external service.

- [ ] **Step 6: Commit the approved configuration**

```powershell
git add configs/ai_tuning/b0.v2_1.continuous.yaml configs/ai_tuning/runtime.v2_1.continuous.live.yaml configs/ai_tuning/v2_1_continuous.live.yaml tests_py/ai_tuning/test_continuous_live.py
git commit -m "chore(v2.1): install restricted B0 campaign"
```

### Task 5: Verify safety boundaries and the complete regression surface

**Files:**
- Verify only; change production files only if a test exposes a defect in the approved scope.

- [ ] **Step 1: Run focused restricted-policy tests**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_policy.py tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py tests_py/ai_tuning/test_continuous_live.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete continuous controller suite**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous -q
```

Expected: PASS, including controller rollback, B0 recovery, evidence persistence, fake end-to-end, and no-FIR/BERT/NLC tests.

- [ ] **Step 3: Run the hardware configuration tests without hardware contact**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/hardware/test_live_config.py tests_py/ai_tuning/hardware/test_live_runtime_factory.py tests_py/ai_tuning/campaign/test_authorization.py -q
```

Expected: PASS. These tests use fakes and configuration parsing only.

- [ ] **Step 4: Run the full Python suite**

```powershell
& $VcselPython -m pytest -q
```

Expected: PASS with no external hardware or Kimi contact.

- [ ] **Step 5: Verify repository cleanliness and scope**

```powershell
git diff --check
git status --short
git log --oneline -5
```

Expected:

- `git diff --check` prints nothing and exits 0.
- `git status --short` lists only the pre-existing untracked `artifacts/`, `outputs/`, and `state/` directories.
- recent commits show the policy model, Kimi boundary, LIVE composition, and approved configuration commits.

- [ ] **Step 6: Produce a read-only implementation handoff**

Record in the final handoff:

- branch, worktree, and final HEAD;
- B0 version 2 and exact I2C values;
- effective `i2c-restricted-v1` ranges and steps;
- reserved campaign ID `v21-pf-20260815-01`;
- exact test commands and results;
- confirmation that no PRE-FLIGHT, COMMISSION, I2C write, DCA acquisition, BERT access, FIR/NLC write, or `setFir` action was performed.

Do not run PRE-FLIGHT or COMMISSION as part of this implementation plan. A later live run requires the user's separate device-contact authorization, the new campaign-bound authorization token, and local `MOONSHOT_API_KEY` input outside chat.
