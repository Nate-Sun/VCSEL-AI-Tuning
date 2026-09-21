# V2.1 Operator Runtime Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require the operator to supply a positive optimization-round budget and positive target TDECQ for every LIVE PRE-FLIGHT and COMMISSION, remove the continuous campaign's aggregate wall-clock limit, and preserve all per-operation timeouts and existing hardware-safety boundaries.

**Architecture:** Parse the two operator values at the CLI boundary before any LIVE factory or hardware path can run, then inject those exact values into one immutable LIVE activation configuration used by preflight, commission, Kimi context, controller stopping, and the commission token. Represent the no-total-time behavior with an explicit `CONTINUOUS_UNBOUNDED` authorization time policy that is legal only for the continuous I2C-only RESTORE_B0 and I2C_SCREENING stages; keep the existing bounded policy as the default for every general campaign. Remove the controller's commission deadline while retaining the finite candidate budget, human-wait lease timeout, Kimi/DCA/I2C call timeouts, stop handling, and B0 recovery.

**Tech Stack:** Python 3.12, argparse, Pydantic 2, PyYAML, SQLite campaign store, pytest, PowerShell.

---

## File map

- Modify `src/vcsel_ai_tuning/v2/continuous/models.py`: remove the ten-round cap, require a positive target value, and remove `commission_deadline` from continuous requests.
- Modify `src/vcsel_ai_tuning/v2/continuous/fake.py`: let FAKE callers use defaults or explicitly override the same round and target values.
- Modify `src/vcsel_ai_tuning/v2/continuous/factory.py`: validate and carry the operator values, and bind both values into the commission token without a duration.
- Modify `src/vcsel_ai_tuning/v2/continuous/cli.py`: add `--max-rounds` and `--target-tdecq-db`; require them for LIVE before lazy activation is built.
- Modify `src/vcsel_ai_tuning/continuous_live.py`: use the injected values in PRE-FLIGHT and COMMISSION, remove duration input, and create no commission deadline.
- Modify `src/vcsel_ai_tuning/campaign/models.py`: add an explicit bounded versus continuous-unbounded authorization time policy.
- Modify `src/vcsel_ai_tuning/campaign/authorization.py`: hash and validate the time policy, and skip only the elapsed-time budget for authorized continuous-unbounded stages.
- Modify `src/vcsel_ai_tuning/campaign/store.py`: retain atomic candidate-count enforcement while skipping elapsed-time exhaustion only for continuous-unbounded authorizations.
- Modify `src/vcsel_ai_tuning/campaign/orchestrator.py`: create continuous-only unbounded RESTORE_B0 and I2C_SCREENING authorizations and remove the 10-round and deadline logic.
- Modify `src/vcsel_ai_tuning/campaign/service.py`: keep generic stage authorization reuse bounded and null-safe.
- Modify `src/vcsel_ai_tuning/v2/continuous/controller.py`: remove aggregate commission-timeout checks; continue to stop on target, rounds, user request, or operation failure.
- Modify `configs/ai_tuning/v2_1_continuous.live.yaml`: remove fixed rounds and duration, and remove the hard-coded target from Kimi's instruction.
- Modify `docs/v2-1-continuous-eye-testbench.md`: document one PowerShell prompt feeding the same values to PRE-FLIGHT and COMMISSION.
- Modify focused tests under `tests_py/ai_tuning/v2/continuous/`, `tests_py/ai_tuning/campaign/`, and `tests_py/ai_tuning/test_continuous_live.py`.

Do not modify the B0 file, restricted parameter policy, register map, I2C/DCA adapters, BERT/FIR/NLC paths, Kimi 60-second request timeout, DCA 180-second acquisition timeout, or I2C serial timeouts.

Use this Python executable for every test command:

```powershell
$VcselPython = 'C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

### Task 1: Generalize the continuous request models without weakening validation

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/continuous/models.py:136-194`
- Modify: `src/vcsel_ai_tuning/v2/continuous/fake.py:146-190`
- Test: `tests_py/ai_tuning/v2/continuous/test_models.py:105-135`
- Test: `tests_py/ai_tuning/v2/continuous/test_fake_e2e.py`

- [ ] **Step 1: Write failing model tests for an uncapped positive round count and positive target**

Replace the existing ten-round boundary tests with:

```python
@pytest.mark.parametrize("bad", [True, 0, -1, 3.0])
def test_request_rejects_non_positive_or_non_strict_round_counts(bad: object) -> None:
    with pytest.raises(ValidationError):
        _request(max_rounds=bad)


@pytest.mark.parametrize("rounds", [1, 20, 10_000])
def test_request_accepts_any_positive_integer_round_budget(rounds: int) -> None:
    assert _request(max_rounds=rounds).max_rounds == rounds


@pytest.mark.parametrize("bad", [0.0, -0.1, float("inf"), float("nan")])
def test_stopping_policy_requires_positive_finite_target(bad: float) -> None:
    with pytest.raises(ValidationError):
        StoppingPolicy(target_tdecq_db=bad)


def test_continuous_request_no_longer_accepts_campaign_deadline() -> None:
    with pytest.raises(ValidationError, match="extra"):
        _request(commission_deadline=NOW)
```

Add a FAKE construction test proving an explicit target is propagated while the existing no-argument FAKE path retains `2.0` as a test default.

- [ ] **Step 2: Run the model tests and verify they fail**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_models.py tests_py/ai_tuning/v2/continuous/test_fake_e2e.py -q
```

Expected: FAIL because `max_rounds` still has `le=10`, non-positive targets are accepted, `commission_deadline` exists, and the fake builders do not accept a target.

- [ ] **Step 3: Make the model constraints exact**

In both `ContinuousPreflightRequest` and `ContinuousRequest`, use:

```python
max_rounds: StrictInt = Field(gt=0)
```

Change the target field to:

```python
target_tdecq_db: FiniteFloat = Field(default=2.0, gt=0)
```

Delete this field from `ContinuousRequest`:

```python
commission_deadline: AwareDatetime | None = None
```

Remove the now-unused `AwareDatetime` import if no other model uses it.

- [ ] **Step 4: Thread an optional target through the FAKE builders**

Add `target_tdecq_db: float = 2.0` to `fake_request()` and `fake_scenario()`, pass it through, and construct:

```python
stopping_policy=StoppingPolicy(target_tdecq_db=target_tdecq_db),
```

This is a test/simulation convenience only; LIVE will be required to supply a value in Task 2.

- [ ] **Step 5: Run the focused model and FAKE tests**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_models.py tests_py/ai_tuning/v2/continuous/test_fake_e2e.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the continuous model change**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/models.py src/vcsel_ai_tuning/v2/continuous/fake.py tests_py/ai_tuning/v2/continuous/test_models.py tests_py/ai_tuning/v2/continuous/test_fake_e2e.py
git commit -m "feat(v2.1): accept operator continuous targets"
```

### Task 2: Require operator values at the LIVE CLI boundary and bind the token

**Files:**
- Modify: `src/vcsel_ai_tuning/v2/continuous/cli.py:21-104`
- Modify: `src/vcsel_ai_tuning/v2/continuous/factory.py:20-145`
- Test: `tests_py/ai_tuning/v2/continuous/test_factory_cli.py`

- [ ] **Step 1: Write failing CLI no-contact tests**

Add a small LIVE YAML fixture containing `runtime_config` and `campaign_id`, monkeypatch `build_continuous_dependencies` to raise `AssertionError("factory contacted")`, and parameterize these argument omissions:

```python
@pytest.mark.parametrize(
    "operator_args",
    [
        [],
        ["--max-rounds", "20"],
        ["--target-tdecq-db", "2.8"],
    ],
)
def test_live_requires_both_operator_values_before_factory_contact(
    tmp_path, monkeypatch, capsys, operator_args
) -> None:
    config = _write_live_config(tmp_path)
    monkeypatch.setattr(
        "vcsel_ai_tuning.v2.continuous.cli.build_continuous_dependencies",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("factory contacted")),
    )

    exit_code = main(["--config", str(config), *operator_args])

    assert exit_code == 2
    assert "max-rounds" in capsys.readouterr().err or "target-tdecq-db" in capsys.readouterr().err
```

Avoid reading `capsys` twice in the final test implementation: store `captured = capsys.readouterr()` first.

Add parser validation cases for `--max-rounds 0`, `--max-rounds -1`, `--target-tdecq-db 0`, `nan`, and `inf`. Add a factory capture test asserting LIVE receives exactly `max_rounds=20` and `target_tdecq_db=2.8`, regardless of stale YAML values.

- [ ] **Step 2: Write failing token-scope tests**

Replace duration-based token calls with:

```python
token = commission_authorization_token_for(
    "campaign-1",
    max_rounds=20,
    target_tdecq_db=2.8,
)
assert token == (
    "operator-approved-v21-commission:campaign=campaign-1:"
    "max_rounds=20:target_tdecq_db=2.8:"
    "provider=kimi:i2c_only=true:restore_b0=true"
)
```

Parameterize mismatches for campaign ID, rounds, and target and prove `LazyLiveActivation.activate()` rejects each before importing `vcsel_ai_tuning.continuous_live`. Also prove `1`, `20`, and `10_000` rounds are accepted while 0, negatives, floats, and booleans are rejected.

- [ ] **Step 3: Run the CLI/factory tests and verify they fail**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_factory_cli.py -q
```

Expected: FAIL because the CLI has no operator options, LIVE reads YAML rounds, and the token still binds duration instead of target.

- [ ] **Step 4: Add strict CLI converters and LIVE pre-factory validation**

In `cli.py`, add:

```python
import math


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed
```

Register:

```python
parser.add_argument("--max-rounds", type=_positive_int)
parser.add_argument("--target-tdecq-db", type=_positive_finite_float)
```

After loading YAML and resolving `mode`, but before `build_continuous_dependencies()`, fail with code 2 when LIVE lacks either value. Do not prompt on Python stdin and do not fall back to YAML for LIVE. For FAKE only, retain YAML/default fallback:

```python
if mode is ExecutionMode.LIVE:
    if args.max_rounds is None or args.target_tdecq_db is None:
        print(
            "LIVE requires --max-rounds and --target-tdecq-db",
            file=sys.stderr,
        )
        return 2
    max_rounds = args.max_rounds
    target_tdecq_db = args.target_tdecq_db
else:
    max_rounds = args.max_rounds or int(payload.get("max_rounds", 3))
    target_tdecq_db = (
        args.target_tdecq_db
        if args.target_tdecq_db is not None
        else float(payload.get("target_tdecq_db", 2.0))
    )
```

- [ ] **Step 5: Change the token and immutable LIVE activation scope**

Change `commission_authorization_token_for()` to accept `target_tdecq_db` and validate both inputs without an upper round limit:

```python
def commission_authorization_token_for(
    campaign_id: str,
    max_rounds: int,
    target_tdecq_db: float,
) -> str:
    if type(max_rounds) is not int or max_rounds <= 0:
        raise ValueError("continuous commission rounds must be a positive integer")
    if (
        isinstance(target_tdecq_db, bool)
        or not isinstance(target_tdecq_db, (int, float))
        or not math.isfinite(float(target_tdecq_db))
        or float(target_tdecq_db) <= 0
    ):
        raise ValueError("continuous target TDECQ must be positive and finite")
    target = format(float(target_tdecq_db), ".15g")
    return (
        "operator-approved-v21-commission:"
        f"campaign={campaign_id}:"
        f"max_rounds={max_rounds}:"
        f"target_tdecq_db={target}:"
        "provider=kimi:i2c_only=true:restore_b0=true"
    )
```

Add `target_tdecq_db: float = 2.0` to `build_continuous_dependencies()`. For LIVE, inject both parsed values into `LazyLiveActivation.config` with `MappingProxyType`; for FAKE, pass both to `fake_scenario()`. Update both CLI token calculation and `LazyLiveActivation.activate()` to use campaign, rounds, and target only.

- [ ] **Step 6: Run the complete CLI/factory test module**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_factory_cli.py -q
```

Expected: PASS. The missing/invalid-value tests must prove no factory import, provider construction, secret lookup, or hardware path occurs.

- [ ] **Step 7: Commit the operator boundary**

```powershell
git add src/vcsel_ai_tuning/v2/continuous/cli.py src/vcsel_ai_tuning/v2/continuous/factory.py tests_py/ai_tuning/v2/continuous/test_factory_cli.py
git commit -m "feat(v2.1): require live rounds and target"
```

### Task 3: Add a narrowly scoped continuous-unbounded authorization policy

**Files:**
- Modify: `src/vcsel_ai_tuning/campaign/models.py:45-54,219-241`
- Modify: `src/vcsel_ai_tuning/campaign/authorization.py:82-99,155-205,258-281`
- Test: `tests_py/ai_tuning/campaign/test_models.py`
- Test: `tests_py/ai_tuning/campaign/test_authorization.py`

- [ ] **Step 1: Write failing model tests for explicit bounded and unbounded policies**

Add the enum import and tests equivalent to:

```python
def test_bounded_authorization_still_requires_duration_and_expiry() -> None:
    with pytest.raises(ValidationError, match="bounded"):
        _authorization(max_duration_s=None, expires_at=None)


@pytest.mark.parametrize(
    "stage",
    [CampaignState.RESTORE_B0, CampaignState.I2C_SCREENING],
)
def test_continuous_unbounded_authorization_allows_only_continuous_i2c_stages(
    stage: CampaignState,
) -> None:
    authorization = _authorization(
        stage=stage,
        time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
        max_duration_s=None,
        expires_at=None,
    )
    assert authorization.max_duration_s is None
    assert authorization.expires_at is None


@pytest.mark.parametrize(
    "stage",
    [CampaignState.I2C_FINE, CampaignState.FIR_APPLY, CampaignState.FINAL_VERIFICATION],
)
def test_continuous_unbounded_authorization_rejects_other_stages(stage) -> None:
    with pytest.raises(ValidationError, match="continuous I2C-only"):
        _authorization(
            stage=stage,
            time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
            max_duration_s=None,
            expires_at=None,
        )
```

Also test that a continuous-unbounded authorization rejects a non-null duration or expiry, and that bounded remains the default for every existing fixture.

- [ ] **Step 2: Write failing authorization validation and budget tests**

Create continuous-unbounded authorizations with a correct deterministic authorization ID and assert:

```python
validated = validate_authorization(
    authorization,
    now=authorization.issued_at + timedelta(days=30),
    current_campaign_id=authorization.campaign_id,
    current_baseline_hash=authorization.baseline_hash,
)
assert validated.time_policy is AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED
```

Then call `validate_candidate()` with `elapsed_s=10_000_000.0` and an in-budget candidate index; it must pass. A candidate at `candidate_index == max_candidates` must still fail. Retain the existing bounded expiry and duration-exhaustion tests unchanged.

- [ ] **Step 3: Run the focused campaign model/authorization tests and verify they fail**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/campaign/test_models.py tests_py/ai_tuning/campaign/test_authorization.py -q
```

Expected: FAIL because the enum and null-duration semantics do not exist.

- [ ] **Step 4: Implement an explicit time policy with a fail-closed model validator**

Add:

```python
class AuthorizationTimePolicy(StrEnum):
    BOUNDED = "BOUNDED"
    CONTINUOUS_UNBOUNDED = "CONTINUOUS_UNBOUNDED"
```

Change `StageAuthorization` fields and validator:

```python
time_policy: AuthorizationTimePolicy = AuthorizationTimePolicy.BOUNDED
max_duration_s: FiniteFloat | None
issued_at: AwareDatetime
expires_at: AwareDatetime | None

@model_validator(mode="after")
def validate_time_policy(self) -> "StageAuthorization":
    if self.time_policy is AuthorizationTimePolicy.BOUNDED:
        if self.max_duration_s is None or self.max_duration_s <= 0:
            raise ValueError("bounded authorization requires positive max_duration_s")
        if self.expires_at is None or self.expires_at <= self.issued_at:
            raise ValueError("bounded authorization requires expires_at after issued_at")
        return self
    if self.stage not in {CampaignState.RESTORE_B0, CampaignState.I2C_SCREENING}:
        raise ValueError("continuous I2C-only time policy has invalid stage")
    if self.mode is not AuthorizationMode.PER_STAGE:
        raise ValueError("continuous I2C-only time policy requires PER_STAGE")
    if self.max_duration_s is not None or self.expires_at is not None:
        raise ValueError("continuous unbounded authorization forbids duration and expiry")
    return self
```

Do not add a permissive `None` default. Callers must continue to state their duration policy explicitly through the existing required fields.

- [ ] **Step 5: Hash and validate the new policy**

Add `time_policy` to `_authorization_payload()` so changing the policy invalidates the authorization ID. In `validate_authorization()`:

- always validate `issued_at` and `now`;
- preserve the not-yet-valid check;
- for `BOUNDED`, require and validate `expires_at` exactly as today;
- for `CONTINUOUS_UNBOUNDED`, require `expires_at is None` and skip only expiration;
- preserve all mode, stage, campaign, baseline, rollback, parameter, FIR, and hash checks.

In `_validate_budget()`, keep candidate and elapsed input validation, then use:

```python
if (
    authorization.time_policy is AuthorizationTimePolicy.BOUNDED
    and authorization.max_duration_s is not None
    and elapsed_s >= authorization.max_duration_s
):
    raise AuthorizationError("elapsed_s reaches or exceeds max_duration_s")
```

- [ ] **Step 6: Run the campaign model and authorization tests**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/campaign/test_models.py tests_py/ai_tuning/campaign/test_authorization.py -q
```

Expected: PASS, including all pre-existing bounded authorization tests.

- [ ] **Step 7: Commit the authorization model**

```powershell
git add src/vcsel_ai_tuning/campaign/models.py src/vcsel_ai_tuning/campaign/authorization.py tests_py/ai_tuning/campaign/test_models.py tests_py/ai_tuning/campaign/test_authorization.py
git commit -m "feat(v2.1): scope unbounded continuous authorization"
```

### Task 4: Remove duration from continuous orchestration while retaining candidate budgets

**Files:**
- Modify: `src/vcsel_ai_tuning/campaign/orchestrator.py:120-132,275-345,464-526`
- Modify: `src/vcsel_ai_tuning/campaign/store.py:1065-1090`
- Modify: `src/vcsel_ai_tuning/campaign/service.py:142-165`
- Test: `tests_py/ai_tuning/campaign/test_orchestrator.py:830-950`
- Test: `tests_py/ai_tuning/campaign/test_store.py:630-730`
- Test: `tests_py/ai_tuning/campaign/test_service.py`

- [ ] **Step 1: Write failing continuous orchestration tests**

Change continuous preparation calls to:

```python
prepared = harness.orchestrator.prepare_continuous_commission(
    max_candidates=20,
)
```

Assert:

```python
assert prepared.authorization.time_policy is AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED
assert prepared.authorization.max_candidates == 20
assert prepared.authorization.max_duration_s is None
assert prepared.authorization.expires_at is None
assert not hasattr(prepared, "deadline")
```

Retain existing assertions that B0 is restored, baseline DCA evidence is persisted, I2C_SCREENING is entered, and exceptions recover to B0. Add a non-continuous orchestrator test showing the same internal unbounded helper is rejected when `continuous_i2c_only` is false.

- [ ] **Step 2: Write failing durable-store tests**

Persist and activate a continuous-unbounded I2C_SCREENING authorization. Advance the test clock far beyond any former 900-second duration and reserve candidates until `max_candidates`; all in-budget reservations pass, while the next reservation fails with `max_candidates budget exhausted`. Keep the existing bounded `max_duration budget exhausted` test unchanged.

- [ ] **Step 3: Run focused orchestration/store tests and verify they fail**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/campaign/test_orchestrator.py tests_py/ai_tuning/campaign/test_store.py tests_py/ai_tuning/campaign/test_service.py -q
```

Expected: FAIL because continuous preparation still requires duration, caps candidates at 10, returns a deadline, and the store compares elapsed time against `None`.

- [ ] **Step 4: Add a continuous-only authorization constructor**

Keep public `plan_stage(..., max_duration_s=...)` unchanged and bounded. Add a private helper in `CampaignOrchestrator` whose first guard is:

```python
if not self.continuous_i2c_only:
    raise OrchestratorError(
        "unbounded authorization requires an I2C-only continuous orchestrator"
    )
if stage not in {CampaignState.RESTORE_B0, CampaignState.I2C_SCREENING}:
    raise OrchestratorError("unbounded continuous stage is not permitted")
```

Build the authorization with:

```python
time_policy=AuthorizationTimePolicy.CONTINUOUS_UNBOUNDED,
max_duration_s=None,
expires_at=None,
```

Reuse the same snapshot, rollback hash, parameter specs, deterministic ID, and `STAGE_PLANNED` event logic as bounded `plan_stage()`. Extract a shared internal constructor if that avoids duplicating the safety-critical fields; do not change the external behavior of generic `plan_stage()`.

- [ ] **Step 5: Simplify continuous commission preparation**

Change the signature to:

```python
def prepare_continuous_commission(
    self,
    *,
    max_candidates: int,
) -> ContinuousCommissionPreparation:
```

Reject non-positive integers, but remove the `> 10` rejection. Use continuous-unbounded authorizations for RESTORE_B0 and I2C_SCREENING. Delete `started_at`, `deadline`, `remaining_s`, and the `continuous commission duration expired` branch. Change the preparation dataclass to contain only `observation` and `authorization`.

- [ ] **Step 6: Make durable candidate reservation policy-aware**

In `CampaignStore.reserve_candidate()`, preserve the atomic candidate-count check and use:

```python
if (
    authorization.time_policy is AuthorizationTimePolicy.BOUNDED
    and authorization.max_duration_s is not None
    and elapsed_s >= authorization.max_duration_s
):
    raise StoreError("authorization max_duration budget exhausted")
```

Update `CampaignService` authorization reuse so it only compares `expires_at > now` for bounded authorizations. The service's normal stage-planning request remains duration-required and must never synthesize a continuous-unbounded authorization.

- [ ] **Step 7: Run focused campaign tests**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/campaign/test_orchestrator.py tests_py/ai_tuning/campaign/test_store.py tests_py/ai_tuning/campaign/test_service.py -q
```

Expected: PASS. Existing generic duration and expiry failures must still pass unchanged.

- [ ] **Step 8: Commit continuous orchestration changes**

```powershell
git add src/vcsel_ai_tuning/campaign/orchestrator.py src/vcsel_ai_tuning/campaign/store.py src/vcsel_ai_tuning/campaign/service.py tests_py/ai_tuning/campaign/test_orchestrator.py tests_py/ai_tuning/campaign/test_store.py tests_py/ai_tuning/campaign/test_service.py
git commit -m "feat(v2.1): remove continuous aggregate duration"
```

### Task 5: Use the operator target and remove the controller deadline

**Files:**
- Modify: `src/vcsel_ai_tuning/continuous_live.py:25-315`
- Modify: `src/vcsel_ai_tuning/v2/continuous/controller.py:145-225,755-830`
- Test: `tests_py/ai_tuning/test_continuous_live.py`
- Test: `tests_py/ai_tuning/v2/continuous/test_controller.py`
- Test: `tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py`

- [ ] **Step 1: Write failing LIVE composition tests for exact propagation**

Update PRE-FLIGHT and COMMISSION fixtures to carry:

```python
"max_rounds": 20,
"target_tdecq_db": 2.8,
```

Remove every `commission_max_duration_s`. Capture the generated request and assert in both paths:

```python
assert request.max_rounds == 20
assert request.stopping_policy.target_tdecq_db == pytest.approx(2.8)
```

For COMMISSION, assert `prepare_continuous_commission(max_candidates=20)` is called without a duration and the returned request has no `commission_deadline`. Update token calls to bind `(campaign_id, 20, 2.8)`. Add mismatch cases showing changed target or rounds fail before `KimiApiClient` and before `create_live_api_runtime_from_path`.

- [ ] **Step 2: Write failing controller target-boundary tests**

Parameterize three one-round outcomes using a `StoppingPolicy(target_tdecq_db=2.8)` request:

```python
@pytest.mark.parametrize(
    ("tdecq_db", "status", "target_reached"),
    [
        (2.79, MeasurementStatus.VALID, True),
        (2.80, MeasurementStatus.VALID, False),
        (2.79, MeasurementStatus.INVALID, False),
    ],
)
def test_operator_target_is_strict_and_requires_valid_measurement(...):
    ...
```

Replace `test_candidate_crossing_commission_deadline_restores_b0_and_times_out` with a test that advances the fake clock beyond the former deadline during provider/execution calls and proves the controller continues until target or `MAX_ROUNDS_REACHED`; it must never emit `COMMISSION_TIMEOUT`.

- [ ] **Step 3: Run focused LIVE/controller tests and verify they fail**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/v2/continuous/test_controller.py tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py -q
```

Expected: FAIL because LIVE still reads duration/default target and the controller still checks a commission deadline.

- [ ] **Step 4: Make LIVE use the injected values as its sole source**

At the top of both LIVE functions, read the already-validated immutable values:

```python
max_rounds = int(config["max_rounds"])
target_tdecq_db = float(config["target_tdecq_db"])
stopping_policy = StoppingPolicy(target_tdecq_db=target_tdecq_db)
```

Use that `stopping_policy` in `ContinuousPreflightRequest` and `ContinuousRequest`. Remove all `config.get("max_rounds", 3)`, `StoppingPolicy()` defaults in LIVE, `commission_max_duration_s`, and `prepared.deadline`. Call:

```python
prepared = entry.orchestrator.prepare_continuous_commission(
    max_candidates=max_rounds,
)
```

The existing `ProviderContext` construction already reads `request.max_rounds` and `request.stopping_policy.target_tdecq_db`; preserve that path so Kimi receives exactly:

```json
{"round": {"maximum": 20}, "target": {"tdecq_db_lt": 2.8}}
```

- [ ] **Step 5: Remove aggregate deadline checks from the controller**

Delete:

- the pre-round and post-provider `_commission_expired()` branches;
- `_commission_expired()` itself;
- `COMMISSION_TIMEOUT` as a controller stop reason;
- the `min(human_timeout, commission_deadline)` logic in `_await_human()`.

Keep `human_timeout_s` and its lease expiry as an independent bounded human-wait interval:

```python
expires_at = issued_at + timedelta(seconds=request.human_timeout_s)
```

Do not change provider exception handling, execution exception handling, stop requests, evidence persistence, target validation, acceptance measurement, B0 restore, or shutdown cleanup.

- [ ] **Step 6: Prove the Kimi structured context uses the operator values**

In `test_kimi_continuous.py`, add or update a serialization assertion with `max_rounds=20` and `target_tdecq_db=2.8`:

```python
assert prompt["round"]["maximum"] == 20
assert prompt["target"] == {"tdecq_db_lt": 2.8}
```

This test validates the structured prompt only; the YAML instruction must not repeat a numeric target.

- [ ] **Step 7: Run LIVE/controller/Kimi tests**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/v2/continuous/test_controller.py tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py -q
```

Expected: PASS, including strict `2.79 < 2.8`, equality not passing, INVALID not passing, no aggregate timeout, and B0 recovery tests.

- [ ] **Step 8: Commit LIVE and controller composition**

```powershell
git add src/vcsel_ai_tuning/continuous_live.py src/vcsel_ai_tuning/v2/continuous/controller.py tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/v2/continuous/test_controller.py tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py
git commit -m "feat(v2.1): run continuous control without deadline"
```

### Task 6: Remove fixed YAML controls and document human input

**Files:**
- Modify: `configs/ai_tuning/v2_1_continuous.live.yaml:1-31`
- Modify: `docs/v2-1-continuous-eye-testbench.md`
- Test: `tests_py/ai_tuning/test_continuous_live.py`

- [ ] **Step 1: Write a failing checked-in configuration test**

Extend the checked-in V2.1 configuration test:

```python
def test_checked_in_v21_live_config_requires_operator_runtime_controls() -> None:
    document = yaml.safe_load(
        Path("configs/ai_tuning/v2_1_continuous.live.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "max_rounds" not in document
    assert "target_tdecq_db" not in document
    assert "commission_max_duration_s" not in document["live"]
    instruction = document["live"]["instruction"]
    assert "2.0" not in instruction
    assert "2.8" not in instruction
    assert "target.tdecq_db_lt" in instruction
```

- [ ] **Step 2: Run the checked-in configuration test and verify it fails**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py::test_checked_in_v21_live_config_requires_operator_runtime_controls -q
```

Expected: FAIL because rounds, duration, and `2.0 dB` are still in YAML.

- [ ] **Step 3: Remove fixed runtime values from YAML**

Delete:

```yaml
max_rounds: 10
commission_max_duration_s: 900.0
```

Keep the approved B0, campaign ID, Kimi model, and restricted parameter policy unchanged. Replace the numeric instruction sentence with:

```yaml
    Use the structured target.tdecq_db_lt supplied for this run as the strict
    TDECQ goal, and use the structured round.maximum as the total proposal
    budget.
```

- [ ] **Step 4: Add the PowerShell operator workflow to the testbench**

Document one prompt before PRE-FLIGHT:

```powershell
$MaxRounds = [int](Read-Host '请输入本次最大优化轮数 / Max optimization rounds')
$TargetTdecqDb = [double](Read-Host '请输入目标 TDECQ dB（严格小于） / Target TDECQ dB (strictly below)')

if ($MaxRounds -le 0) { throw 'MaxRounds must be a positive integer' }
if ($TargetTdecqDb -le 0 -or [double]::IsNaN($TargetTdecqDb) -or [double]::IsInfinity($TargetTdecqDb)) {
    throw 'TargetTdecqDb must be a positive finite number'
}
```

Pass the same values to both commands:

```powershell
--max-rounds $MaxRounds `
--target-tdecq-db $TargetTdecqDb
```

State explicitly that changing either value after PRE-FLIGHT invalidates the commission token and requires a new authorization. Use `2.8` as the example input, not as a configuration default. Preserve all existing separate PRE-FLIGHT/COMMISSION, device-contact, and authorization-token instructions.

- [ ] **Step 5: Run config and CLI documentation-adjacent tests**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/v2/continuous/test_factory_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit configuration and runbook updates**

```powershell
git add configs/ai_tuning/v2_1_continuous.live.yaml docs/v2-1-continuous-eye-testbench.md tests_py/ai_tuning/test_continuous_live.py
git commit -m "docs(v2.1): prompt for continuous run controls"
```

### Task 7: Verify safety boundaries and full regression behavior

**Files:**
- Verify only; change production files only if a test exposes a defect inside the approved scope.

- [ ] **Step 1: Run the complete continuous suite**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous tests_py/ai_tuning/test_continuous_live.py -q
```

Expected: PASS. This includes Kimi prompt serialization, local proposal policy, evidence persistence, manual stop, acceptance, B0 recovery, fake end-to-end, and no BERT/FIR/NLC behavior.

- [ ] **Step 2: Run the campaign authorization/orchestration/store suite**

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/campaign -q
```

Expected: PASS. Continuous-unbounded tests skip only cumulative time; general campaigns still fail on expired authorization or exhausted duration.

- [ ] **Step 3: Verify the retained single-operation timeout configuration and tests**

Run the relevant provider and hardware configuration tests:

```powershell
& $VcselPython -m pytest tests_py/ai_tuning/v2/continuous/test_kimi_continuous.py tests_py/ai_tuning/hardware/test_live_config.py tests_py/ai_tuning/hardware/test_live_runtime_factory.py tests_py/ai_tuning/hardware/test_vd1124.py -q
```

Expected: PASS. Confirm by read-only inspection that:

- `KimiApiClient` still constructs the SDK with `timeout=60.0`;
- `runtime.v2_1.continuous.live.yaml` still has `measurement_timeout_s: 180.0`;
- VD1124 serial construction still sets finite `timeout` and `write_timeout`;
- no BERT, FIR, NLC, or `setFir` call was added.

- [ ] **Step 4: Run the full Python suite**

```powershell
& $VcselPython -m pytest -q
```

Expected: PASS with no external Kimi or hardware contact.

- [ ] **Step 5: Check formatting, repository scope, and exact lineage**

```powershell
git diff --check
git status --short --branch
git log --oneline -8
git rev-parse HEAD
git branch --show-current
```

Expected:

- `git diff --check` prints nothing and exits 0;
- the branch remains `codex/v2.1-continuous-eye-optimization`;
- only the pre-existing untracked `artifacts/`, `outputs/`, and `state/` directories remain outside commits;
- no hardware-generated artifact is staged or committed.

- [ ] **Step 6: Produce the implementation handoff without running hardware**

Report:

- worktree, branch, and final HEAD;
- CLI inputs and the exact example `20` rounds / `2.8 dB` target;
- strict stop condition `valid TDECQ < target`;
- commission token scope with no duration;
- confirmation that continuous has no aggregate duration but still has a finite round budget;
- retained Kimi, DCA, I2C, human-wait, manual-stop, B0 restore, policy, identity, and readback safeguards;
- test commands and results;
- explicit confirmation that implementation verification did not perform PRE-FLIGHT, COMMISSION, I2C writes, DCA acquisition, BERT access, FIR/NLC writes, or `setFir`.

Do not run a real PRE-FLIGHT or COMMISSION during implementation. A real run remains a separate user-authorized hardware operation and must use a fresh campaign ID and a commission token computed from the exact operator-entered rounds and target.
