# FlexDCA Single TDECQ Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a confirmation-gated Stop/Clear/Single workflow that accepts only a fresh Function 1 `CORR` TDECQ result and records a complete receipt without changing BERT FIR.

**Architecture:** Keep acquisition planning and execution in a focused `dca_acquisition.py` module. Refactor the FlexDCA adapter so a final snapshot can be read from an already verified source without silently selecting it again. Add thin `measure-plan` and `measure` CLI handlers that validate live configuration and persist artifacts atomically.

**Tech Stack:** Python 3.11+, argparse, PyVISA session protocol, UDP BERT reader, pytest, JSON/SHA-256 artifacts.

---

## File Structure

- Create `src/vcsel_ai_tuning/hardware/dca_acquisition.py`: plan hashing, plan persistence, Single acquisition state machine, receipt construction.
- Modify `src/vcsel_ai_tuning/hardware/flexdca.py`: expose a no-selection source-verified snapshot reader used after acquisition.
- Modify `src/vcsel_ai_tuning/hardware/live_cli.py`: add `measure-plan` and `measure` commands and live dependency wiring.
- Create `tests_py/ai_tuning/hardware/test_dca_acquisition.py`: deterministic plan and state-machine tests with fake clock/session/BERT.
- Modify `tests_py/ai_tuning/hardware/test_flexdca.py`: test final snapshot reads without an extra source-selection write.
- Modify `tests_py/ai_tuning/hardware/test_live_cli.py`: parser, safety order, receipt path, and exit-code tests.

### Task 1: Read a snapshot from an already selected and verified source

**Files:**
- Modify: `src/vcsel_ai_tuning/hardware/flexdca.py`
- Test: `tests_py/ai_tuning/hardware/test_flexdca.py`

- [ ] **Step 1: Write the failing no-reselection test**

Add a test that calls a wished-for `read_verified_tdecq()` API and proves that
it queries `:MEASure:EYE:TDEQ:SOURce?` first, returns the complete snapshot, and
never calls `write()`:

```python
def test_read_verified_tdecq_does_not_reselect_source():
    session = FakeQuerySession(function=1, status="CORR")

    snapshot = read_verified_tdecq(
        session, function=1, measurement_source="FUNC1"
    )

    assert snapshot.verified_measurement_source == "FUNC1"
    assert snapshot.status == "CORR"
    assert session.writes == []
    assert session.commands[0] == ":MEASure:EYE:TDEQ:SOURce?"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv-ai-tuning\Scripts\python.exe -m pytest `
  tests_py\ai_tuning\hardware\test_flexdca.py::test_read_verified_tdecq_does_not_reselect_source -q
```

Expected: collection fails because `read_verified_tdecq` is not defined.

- [ ] **Step 3: Extract the verified-reader implementation**

In `flexdca.py`, split source verification from selection and make the existing
reader delegate to the new API:

```python
def _verify_measurement_source(
    session: QuerySession, measurement_source: str
) -> tuple[str, dict[str, str]]:
    source_command = ":MEASure:EYE:TDEQ:SOURce?"
    raw_responses = _query_all(session, (source_command,))
    raw_source = raw_responses[source_command].strip().upper()
    source_match = re.fullmatch(r"FUNC(?:TION)?([0-9]+)", raw_source)
    verified_source = (
        f"FUNC{int(source_match.group(1), 10)}" if source_match else raw_source
    )
    if verified_source != measurement_source:
        raise DcaReadError(
            f"TDECQ source readback {raw_source or '<empty>'} does not match "
            f"configured source {measurement_source}"
        )
    return verified_source, raw_responses


def read_verified_tdecq(
    session: QuerySession,
    function: int = 1,
    *,
    measurement_source: str,
) -> DcaSnapshot:
    canonical_source = _normalize_function_source(measurement_source, function)
    verified_source, source_responses = _verify_measurement_source(
        session, canonical_source
    )
    return _read_tdecq_snapshot(
        session,
        function=function,
        canonical_source=canonical_source,
        verified_source=verified_source,
        source_responses=source_responses,
    )
```

Keep `read_existing_tdecq()` behavior unchanged: it selects the configured
source once, then delegates to the same snapshot parser.

- [ ] **Step 4: Run the FlexDCA tests and verify GREEN**

Run:

```powershell
.\.venv-ai-tuning\Scripts\python.exe -m pytest `
  tests_py\ai_tuning\hardware\test_flexdca.py -q
```

Expected: all FlexDCA tests pass, including exact existing write ordering.

### Task 2: Add immutable measurement plan artifacts

**Files:**
- Create: `src/vcsel_ai_tuning/hardware/dca_acquisition.py`
- Create: `tests_py/ai_tuning/hardware/test_dca_acquisition.py`

- [ ] **Step 1: Write failing plan round-trip and validation tests**

Create tests covering stable SHA-256 IDs, atomic save/load, tamper rejection,
and timing validation:

```python
def test_measurement_plan_round_trip_and_hash_rejects_tampering(tmp_path):
    plan = create_measurement_plan(
        resource="TCPIP0::Nate-Rog::hislip0,4880::INSTR",
        function=1,
        measurement_source="FUNC1",
        timeout_s=180,
        poll_interval_s=1,
        created_at_utc=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )
    path = save_measurement_plan(plan, tmp_path / "measurement_plan.json")
    assert load_measurement_plan(path) == plan
    assert len(plan["measurement_id"]) == 64
    assert plan["acquisition"]["commands"] == [
        ":MEASure:EYE:TDEQ:SOURce FUNC1",
        ":ACQuire:STOP",
        ":ACQuire:CDISplay",
        ":ACQuire:SINGle",
    ]

    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["acquisition"]["timeout_s"] = 5
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(AcquisitionWorkflowError, match="hash"):
        load_measurement_plan(path)
```

Use parameterized tests to reject non-finite/non-positive timeout and poll
values, a poll interval greater than timeout, unsupported mode, invalid
Function source, and malformed schema/ID.

- [ ] **Step 2: Run plan tests and verify RED**

Run:

```powershell
.\.venv-ai-tuning\Scripts\python.exe -m pytest `
  tests_py\ai_tuning\hardware\test_dca_acquisition.py -q
```

Expected: import fails because the new module does not exist.

- [ ] **Step 3: Implement the plan API**

Create `dca_acquisition.py` with these public functions:

```python
SCHEMA_VERSION = "2.0"
SINGLE_COMMANDS = (
    ":MEASure:EYE:TDEQ:SOURce FUNC1",
    ":ACQuire:STOP",
    ":ACQuire:CDISplay",
    ":ACQuire:SINGle",
)


class AcquisitionWorkflowError(RuntimeError):
    pass


def _canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(
        document, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def _measurement_hash(document: Mapping[str, object]) -> str:
    payload = {
        key: value for key, value in document.items()
        if key != "measurement_id"
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_measurement_plan(
    document: Mapping[str, object]
) -> Mapping[str, object]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise AcquisitionWorkflowError("measurement plan schema does not match")
    measurement_id = document.get("measurement_id")
    if not isinstance(measurement_id, str) or len(measurement_id) != 64:
        raise AcquisitionWorkflowError("measurement ID is missing or invalid")
    if not hmac.compare_digest(measurement_id, _measurement_hash(document)):
        raise AcquisitionWorkflowError("measurement plan hash does not match")
    target = document.get("target")
    acquisition = document.get("acquisition")
    if not isinstance(target, Mapping) or not isinstance(acquisition, Mapping):
        raise AcquisitionWorkflowError("measurement plan sections are invalid")
    if acquisition.get("mode") != "single":
        raise AcquisitionWorkflowError("measurement mode must be single")
    if acquisition.get("commands") != list(SINGLE_COMMANDS):
        raise AcquisitionWorkflowError("measurement commands do not match")
    return document


def create_measurement_plan(
    *,
    resource: str,
    function: int,
    measurement_source: str,
    timeout_s: float,
    poll_interval_s: float,
    created_at_utc: datetime | None = None,
) -> dict[str, object]:
    timeout = float(timeout_s)
    interval = float(poll_interval_s)
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_s must be positive and finite")
    if not isfinite(interval) or interval <= 0 or interval > timeout:
        raise ValueError("poll_interval_s must be positive and no greater than timeout")
    if function != 1 or measurement_source.upper() not in {"FUNC1", "FUNCTION1"}:
        raise ValueError("Single acquisition requires Function 1")
    created = created_at_utc or datetime.now(timezone.utc)
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created.isoformat(),
        "target": {
            "resource": resource,
            "function": 1,
            "measurement_source": "FUNC1",
        },
        "acquisition": {
            "mode": "single",
            "timeout_s": timeout,
            "poll_interval_s": interval,
            "commands": list(SINGLE_COMMANDS),
        },
    }
    document["measurement_id"] = _measurement_hash(document)
    return document


def save_measurement_plan(
    document: Mapping[str, object], path: str | Path
) -> Path:
    _validate_measurement_plan(document)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_measurement_plan(path: str | Path) -> dict[str, object]:
    plan_path = Path(path)
    try:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionWorkflowError(f"cannot load plan {plan_path}: {error}") from error
    if not isinstance(value, dict):
        raise AcquisitionWorkflowError("measurement plan root must be an object")
    _validate_measurement_plan(value)
    return value
```

Normalize numeric timing fields to `float`, canonicalize `FUNCtion1` to
`FUNC1`, compute the ID from all fields except `measurement_id`, compare hashes
with `hmac.compare_digest`, and use the existing `.tmp` plus `replace()` atomic
write pattern.

- [ ] **Step 4: Run plan tests and verify GREEN**

Run the test command from Step 2. Expected: all plan tests pass.

### Task 3: Implement the Single acquisition state machine

**Files:**
- Modify: `src/vcsel_ai_tuning/hardware/dca_acquisition.py`
- Modify: `tests_py/ai_tuning/hardware/test_dca_acquisition.py`

- [ ] **Step 1: Write the failing successful-transition test**

Use a scripted fake DCA and fake monotonic clock. The fake returns `INV` at the
clear boundary, then `INV`, `INV`, and `CORR` while polling, followed by the
final snapshot responses:

```python
def test_single_measurement_waits_through_inv_and_accepts_fresh_corr():
    dca = ScriptedDca(inv_before_corr=2)
    bert = ScriptedBert([BERT_BASELINE, BERT_BASELINE])
    clock = FakeClock()

    receipt = execute_single_measurement(
        _plan(),
        dca,
        bert,
        _matched_i2c(),
        bert_port="port1",
        bert_channel="0",
        allow_execute=True,
        confirmation_id=_plan()["measurement_id"],
        monotonic=clock.monotonic,
        wait=clock.wait,
        now_utc=clock.now_utc,
    )

    assert receipt["status"] == "MEASURED"
    assert receipt["final_tdecq"]["status"] == "CORR"
    assert receipt["final_tdecq"]["tdecq_db"] == pytest.approx(4.84)
    assert dca.writes == list(SINGLE_COMMANDS)
    assert receipt["bert_before"]["taps"] == list(BASELINE_TAPS)
    assert receipt["bert_after"]["taps"] == list(BASELINE_TAPS)
```

- [ ] **Step 2: Run the successful-transition test and verify RED**

Run only this test. Expected: failure because
`execute_single_measurement()` is missing.

- [ ] **Step 3: Implement gates and the happy-path state machine**

Add protocols for a DCA write/query session and BERT configuration reader.
Implement `execute_single_measurement()` with injected clock functions. Before
the first DCA write it must validate `--execute`, exact ID, plan hash, matched
I2C identity, and initial BERT readback. Then:

```python
session.write(source_command)
verified_source = session.query(source_query).strip().upper()
session.write(":ACQuire:STOP")
session.write(":ACQuire:CDISplay")
clear_status = session.query(status_query).strip().upper()
if clear_status == "CORR":
    return failed_receipt("clear_boundary", "Clear did not invalidate TDECQ")
session.write(":ACQuire:SINGle")

while monotonic() < deadline:
    source = session.query(source_query).strip().upper()
    status = session.query(status_query).strip().upper()
    details = session.query(details_query).strip()
    record_transition(source, status, details)
    if source != expected_source:
        return failed_receipt("poll_source", "source changed")
    if status == "CORR":
        snapshot = read_verified_tdecq(
            session,
            function=function,
            measurement_source=expected_source,
        )
        bert_after = bert_client.get_configuration(bert_port, bert_channel)
        return measured_or_invalidated_receipt(snapshot, bert_after)
    wait(poll_interval_s)
```

Record command outcomes and only status changes, plus the total poll count.
Compare BERT port, channel, taps, upper NLC, and lower NLC exactly.

- [ ] **Step 4: Run the successful-transition test and verify GREEN**

Run only the test from Step 1. Expected: pass.

- [ ] **Step 5: Write failing safety and failure-path tests**

Add independent tests proving:

```python
@pytest.mark.parametrize(
    ("allow_execute", "confirmation_id", "message"),
    [(False, None, "--execute"), (True, "wrong", "measurement ID")],
)
def test_execution_gates_prevent_all_dca_writes(
    allow_execute, confirmation_id, message
):
    plan = _plan()
    dca = ScriptedDca(inv_before_corr=0)
    bert = ScriptedBert([BERT_BASELINE])
    with pytest.raises(AcquisitionWorkflowError, match=message):
        execute_single_measurement(
            plan, dca, bert, _matched_i2c(),
            bert_port="port1", bert_channel="0",
            allow_execute=allow_execute, confirmation_id=confirmation_id,
        )
    assert dca.writes == []


def test_source_mismatch_stops_before_stop_clear_single():
    receipt, dca = _execute_with_dca(ScriptedDca(source="CHAN1A"))
    assert receipt["status"] == "FAILED"
    assert dca.writes == [":MEASure:EYE:TDEQ:SOURce FUNC1"]


def test_corr_at_clear_boundary_fails_before_single():
    receipt, dca = _execute_with_dca(ScriptedDca(clear_status="CORR"))
    assert receipt["failed_stage"] == "clear_boundary"
    assert ":ACQuire:SINGle" not in dca.writes


def test_permanent_inv_times_out_and_stops_once():
    receipt, dca = _execute_with_dca(ScriptedDca(inv_before_corr=None))
    assert receipt["status"] == "TIMED_OUT"
    assert dca.writes[-1] == ":ACQuire:STOP"
    assert dca.writes.count(":ACQuire:STOP") == 2


def test_changed_bert_invalidates_measurement():
    changed = replace(BERT_BASELINE, taps=(0, 1, -4, 110, 3, 1, 0))
    receipt, _ = _execute_with_bert([BERT_BASELINE, changed])
    assert receipt["status"] == "INVALIDATED"
```

Assertions must check exact DCA writes, receipt status, failed stage, last
details, timeout Stop result, and that `set_fir` is never available or called.

- [ ] **Step 6: Run the new failure-path tests and verify RED**

Expected: each new test fails on the missing behavior, not fixture errors.

- [ ] **Step 7: Implement minimal error and timeout handling**

Wrap all operations after the first DCA mutation so failures return a `FAILED`
receipt with `failed_stage` and `error`. On timeout send exactly one additional
`:ACQuire:STOP`, record its success/failure, and return `TIMED_OUT`. Reject
non-finite final TDECQ and return `INVALIDATED` when BERT readback differs.

- [ ] **Step 8: Run all acquisition tests and verify GREEN**

Run:

```powershell
.\.venv-ai-tuning\Scripts\python.exe -m pytest `
  tests_py\ai_tuning\hardware\test_dca_acquisition.py -q
```

Expected: all plan and state-machine tests pass with no real sleeps.

### Task 4: Add CLI commands and atomic receipt persistence

**Files:**
- Modify: `src/vcsel_ai_tuning/hardware/live_cli.py`
- Modify: `tests_py/ai_tuning/hardware/test_live_cli.py`

- [ ] **Step 1: Write failing parser and measure-plan tests**

Add parser assertions for `measure-plan` and `measure`, then invoke
`measure-plan` with monkeypatched configuration:

```python
def test_measure_plan_saves_hashed_non_mutating_plan(tmp_path, monkeypatch):
    output = tmp_path / "measurement_plan.json"
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda _: _config())

    exit_code = live_cli.main([
        "measure-plan", "--mode", "single", "--timeout", "180",
        "--poll-interval", "1", "--output", str(output),
    ])

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(plan["measurement_id"]) == 64
    assert plan["acquisition"]["commands"][2] == ":ACQuire:CDISplay"
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run `test_live_cli.py`. Expected: parser rejects the two new commands.

- [ ] **Step 3: Add parsers and `measure-plan` handler**

Add defaults under `artifacts/fir_runs/latest` and CLI options exactly matching
the design. `_run_measure_plan()` calls `create_measurement_plan()`, saves it,
and prints the resource, source, timing, ordered commands, ID, and output path.
It must not create DCA, BERT, or I2C clients.

- [ ] **Step 4: Run the measure-plan CLI test and verify GREEN**

Expected: plan test passes and existing CLI tests remain green.

- [ ] **Step 5: Write failing execution wiring tests**

Monkeypatch dependencies and prove order and exit mapping:

```python
def test_measure_checks_i2c_before_opening_dca_and_saves_receipt(
    tmp_path, monkeypatch
):
    plan_path = tmp_path / "measurement_plan.json"
    receipt_path = tmp_path / "measurement_receipt.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = {"measurement_id": "a" * 64, "target": {"resource": "DCA"}}
    monkeypatch.setattr(live_cli, "load_measurement_plan", lambda _: plan)
    monkeypatch.setattr(live_cli, "load_live_hardware_config", lambda _: _config())
    monkeypatch.setattr(live_cli, "_probe_i2c_only", lambda _: _matched_i2c())
    monkeypatch.setattr(live_cli, "open_visa_session", lambda *a, **k: FakeContext())
    monkeypatch.setattr(
        live_cli, "execute_single_measurement",
        lambda *a, **k: {"status": "MEASURED", "measurement_id": "a" * 64},
    )
    assert live_cli.main([
        "measure", "--plan", str(plan_path), "--execute",
        "--confirmation-id", "a" * 64, "--receipt", str(receipt_path),
    ]) == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "MEASURED"


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("MEASURED", 0), ("TIMED_OUT", 3),
     ("INVALIDATED", 3), ("FAILED", 4)],
)
def test_measure_exit_codes(status, exit_code, tmp_path, monkeypatch):
    _install_measure_fakes(monkeypatch, status=status)
    assert live_cli.main(_measure_argv(tmp_path)) == exit_code


def test_measure_i2c_mismatch_never_opens_dca(tmp_path, monkeypatch):
    opened = []
    _install_measure_fakes(monkeypatch, i2c_matched=False)
    monkeypatch.setattr(live_cli, "open_visa_session", lambda *a, **k: opened.append(a))
    assert live_cli.main(_measure_argv(tmp_path)) == 1
    assert opened == []
```

- [ ] **Step 6: Run execution wiring tests and verify RED**

Expected: failures because `_run_measure()` and dispatch are missing.

- [ ] **Step 7: Implement live dependency wiring**

`_run_measure()` loads the hashed plan, compares its DCA target with live
configuration, probes I2C, creates the read-only BERT client, opens one VISA
session, calls `execute_single_measurement()`, atomically writes the receipt,
prints status/TDECQ/details, and maps status to the documented exit codes.

- [ ] **Step 8: Run all CLI tests and verify GREEN**

Run:

```powershell
.\.venv-ai-tuning\Scripts\python.exe -m pytest `
  tests_py\ai_tuning\hardware\test_live_cli.py -q
```

Expected: all commands, including probe/propose/apply/restore, pass.

### Task 5: Full verification and real-hardware plan generation

**Files:**
- Verify: all changed source and test files
- Create at runtime: `artifacts/fir_runs/2026-07-21-single-baseline/measurement_plan.json`

- [ ] **Step 1: Run focused tests**

```powershell
.\.venv-ai-tuning\Scripts\python.exe -m pytest `
  tests_py\ai_tuning\hardware\test_flexdca.py `
  tests_py\ai_tuning\hardware\test_dca_acquisition.py `
  tests_py\ai_tuning\hardware\test_live_cli.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run the complete test suite**

```powershell
.\.venv-ai-tuning\Scripts\python.exe -m pytest -q
```

Expected: zero failures.

- [ ] **Step 3: Generate but do not execute the real plan**

```powershell
.\.venv-ai-tuning\Scripts\python.exe scripts\run_live_fir.py measure-plan `
  --mode single `
  --timeout 180 `
  --poll-interval 1 `
  --output artifacts\fir_runs\2026-07-21-single-baseline\measurement_plan.json
```

Expected: the command prints the four exact DCA mutations and a complete
`measurement_id`. It performs no hardware access.

- [ ] **Step 4: Stop for explicit real-hardware authorization**

Show the user:

- current BERT FIR expected for attribution: `[0, 1, -5, 110, 3, 1, 0]`;
- DCA resource and Function 1 source;
- Stop/Clear/Single commands;
- Clear's non-rollbackable effect;
- timeout and receipt path; and
- complete `measurement_id`.

Do not execute `measure` until the user returns that exact ID.
