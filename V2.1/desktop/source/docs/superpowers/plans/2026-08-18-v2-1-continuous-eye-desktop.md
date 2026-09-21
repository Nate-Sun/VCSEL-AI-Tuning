# V2.1 Continuous Eye Desktop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an offline-verified Windows `.exe` that wraps the validated V2.1 B continuous eye engine with the approved minimal Tkinter workflow and CSV result export.

**Architecture:** A small `desktop` package owns immutable operator inputs, session-local config generation, build lineage verification, artifact projection, and a background engine session. Tkinter calls only the `DesktopSession` interface; the existing continuous engine remains the sole hardware and optimization implementation.

**Tech Stack:** Python 3.13, Tkinter/ttk, Pydantic v2, PyYAML, Pillow, existing VCSEL continuous engine, pytest, PyInstaller `onedir`, PowerShell build script.

**Worktree import rule:** The system Python may contain an editable installation that points at the protected validated worktree. Before every Python or pytest command in this plan, set `$env:PYTHONPATH = (Resolve-Path .\src)` in the same PowerShell process. Each test must confirm `vcsel_ai_tuning.__file__` resolves inside `v2-1-continuous-eye-desktop-app`; the build script must likewise invoke PyInstaller with this worktree's `src` first. Never change or reinstall the protected worktree to solve imports.

---

## File Structure

- `src/vcsel_ai_tuning/desktop/models.py`: immutable desktop input, status, and result contracts.
- `src/vcsel_ai_tuning/desktop/settings.py`: atomic persistence of the three allowed bench endpoints.
- `src/vcsel_ai_tuning/desktop/session_config.py`: session-local B0, hardware profile, runtime, and live YAML generation.
- `src/vcsel_ai_tuning/desktop/integrity.py`: minimal build manifest and protected-file fingerprint checks.
- `src/vcsel_ai_tuning/desktop/artifacts.py`: read-only campaign/result projection and eye lookup.
- `src/vcsel_ai_tuning/desktop/csv_export.py`: one-row-per-measured-point CSV export.
- `src/vcsel_ai_tuning/desktop/session.py`: background PRE-FLIGHT/COMMISSION orchestration and safe stop.
- `src/vcsel_ai_tuning/desktop/app.py`: approved minimal Tkinter UI.
- `src/vcsel_ai_tuning/desktop/main.py`: desktop entry point and application paths.
- `src/vcsel_ai_tuning/desktop/__main__.py`: `python -m` entry point.
- `scripts/build_v2_1_desktop.ps1`: clean-lineage checks, manifest generation, PyInstaller `onedir` build, offline smoke test.
- `packaging/v2_1_desktop.spec`: deterministic PyInstaller inputs and data files.
- `tests_py/ai_tuning/desktop/`: focused offline tests for every desktop module.

### Task 1: Desktop input contracts and validation

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/__init__.py`
- Create: `src/vcsel_ai_tuning/desktop/models.py`
- Create: `tests_py/ai_tuning/desktop/test_desktop_models.py`

- [ ] **Step 1: Write failing contract tests**

Define `DEFAULT_B0` from the nine approved validated values and a local `_inputs(**overrides)` helper in the test module; neither helper reads hardware or environment secrets.

```python
def test_run_inputs_accept_exact_operator_surface():
    value = RunInputs(
        api_key="secret",
        max_rounds=50,
        target_tdecq_db=2.8,
        b0=DEFAULT_B0,
        bench=BenchSettings(
            dca_visa_resource="TCPIP0::scope::hislip0,4880::INSTR",
            i2c_com_port="COM7",
            i2c_address=0xA6,
        ),
    )
    assert value.max_rounds == 50
    assert value.api_key.get_secret_value() == "secret"

def test_run_inputs_reject_bool_rounds_and_nonfinite_target():
    with pytest.raises(ValidationError):
        _inputs(max_rounds=True)
    with pytest.raises(ValidationError):
        _inputs(target_tdecq_db=float("nan"))

def test_b0_requires_all_nine_strict_integers():
    raw = dict(DEFAULT_B0.model_dump())
    del raw["eq"]
    with pytest.raises(ValidationError):
        B0Input.model_validate(raw)
```

- [ ] **Step 2: Run the tests and confirm missing-module failure**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_desktop_models.py -q`

Expected: collection fails because `vcsel_ai_tuning.desktop.models` does not exist.

- [ ] **Step 3: Implement strict immutable models**

```python
class B0Input(FrozenModel):
    ibias_dac: StrictInt = Field(ge=0)
    vref: StrictInt = Field(ge=0)
    vbias_ref: StrictInt = Field(ge=0)
    vga_cur: StrictInt = Field(ge=0)
    res_gain: StrictInt = Field(ge=0)
    res_bw: StrictInt = Field(ge=0)
    gain_ctrl: StrictInt = Field(ge=0)
    efin_cur: StrictInt = Field(ge=0)
    eq: StrictInt = Field(ge=0)

class BenchSettings(FrozenModel):
    dca_visa_resource: NonBlank
    i2c_com_port: NonBlank
    i2c_address: StrictInt = Field(ge=0, le=255)

class RunInputs(FrozenModel):
    api_key: SecretStr
    max_rounds: StrictInt = Field(gt=0)
    target_tdecq_db: FiniteFloat = Field(gt=0)
    b0: B0Input
    bench: BenchSettings
```

Also define `DesktopPhase`, `PreflightSummary`, `ProgressSnapshot`, and `DesktopResult` without hardware types.

- [ ] **Step 4: Run model tests**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_desktop_models.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/vcsel_ai_tuning/desktop tests_py/ai_tuning/desktop/test_desktop_models.py
git commit -m "feat(desktop): add strict operator input contracts"
```

### Task 2: Safe local settings persistence

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/settings.py`
- Create: `tests_py/ai_tuning/desktop/test_settings.py`

- [ ] **Step 1: Write failing settings tests**

Define `BENCH` in the test module as the approved three-field `BenchSettings` sample. It must not contain an API key or any other runtime value.

```python
def test_settings_round_trip_contains_only_bench_endpoints(tmp_path):
    store = BenchSettingsStore(tmp_path / "settings.json")
    store.save(BENCH)
    assert store.load() == BENCH
    text = (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert "api" not in text.lower()
    assert "moonshot" not in text.lower()

def test_corrupt_settings_fail_closed(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(DesktopSettingsError):
        BenchSettingsStore(path).load()
```

- [ ] **Step 2: Verify failure**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_settings.py -q`

Expected: import failure for `BenchSettingsStore`.

- [ ] **Step 3: Implement atomic JSON persistence**

```python
class BenchSettingsStore:
    def load(self) -> BenchSettings | None:
        if not self.path.exists():
            return None
        return BenchSettings.model_validate_json(self.path.read_text("utf-8"))

    def save(self, value: BenchSettings) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(value.model_dump_json(), encoding="utf-8")
        os.replace(temporary, self.path)
```

Create parent directories, flush and `fsync`, restrict the serialized schema to `BenchSettings`, and raise `DesktopSettingsError` for corrupt data.

- [ ] **Step 4: Run tests and commit**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_settings.py -q`

Expected: all tests pass.

```powershell
git add src/vcsel_ai_tuning/desktop/settings.py tests_py/ai_tuning/desktop/test_settings.py
git commit -m "feat(desktop): persist only approved bench settings"
```

### Task 3: Session-local configuration generation

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/session_config.py`
- Create: `tests_py/ai_tuning/desktop/test_session_config.py`

- [ ] **Step 1: Write failing config-generation tests**

Define `TEMPLATE_ROOT` as the repository `configs/ai_tuning` directory, load `VALIDATED_RUNTIME` from the checked-in runtime template, and construct `INPUTS` entirely in the test module. These names are test data, not production globals.

```python
def test_builder_changes_only_approved_b0_and_endpoint_fields(tmp_path):
    paths = SessionConfigBuilder(TEMPLATE_ROOT).build(tmp_path, INPUTS, "desktop-test")
    b0 = yaml.safe_load(paths.b0.read_text("utf-8"))
    hardware_profile = yaml.safe_load(paths.hardware.read_text("utf-8"))
    runtime = yaml.safe_load(paths.runtime.read_text("utf-8"))
    assert b0["i2c"] == INPUTS.b0.model_dump()
    assert b0["fir"]["taps"] == [0, 9, -28, 104, -8, 1, -3]
    assert runtime["hardware"]["dca_visa_resource"] == INPUTS.bench.dca_visa_resource
    assert runtime["hardware"]["i2c_com_port"] == INPUTS.bench.i2c_com_port
    assert runtime["hardware"]["i2c_address"] == INPUTS.bench.i2c_address
    assert runtime["hardware"]["bert_host"] == VALIDATED_RUNTIME["hardware"]["bert_host"]
    assert runtime["hardware"]["bert_port"] == VALIDATED_RUNTIME["hardware"]["bert_port"]
    assert hardware_profile["dca"]["resource"] == INPUTS.bench.dca_visa_resource
    assert hardware_profile["i2c"]["port"] == INPUTS.bench.i2c_com_port
    assert hardware_profile["i2c"]["device_address"] == INPUTS.bench.i2c_address
    assert hardware_profile["bert"] == VALIDATED_HARDWARE["bert"]

def test_generated_documents_never_contain_api_key(tmp_path):
    paths = SessionConfigBuilder(TEMPLATE_ROOT).build(tmp_path, INPUTS, "desktop-test")
    assert "secret" not in "".join(path.read_text("utf-8") for path in paths.config_files)
```

- [ ] **Step 2: Verify failure**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_session_config.py -q`

Expected: import failure for `SessionConfigBuilder`.

- [ ] **Step 3: Implement deterministic generation**

Load the four validated templates (`b0.v2_1.continuous.yaml`, `../hardware/vcsel_lab_v2_1_continuous.yaml`, `runtime.v2_1.continuous.live.yaml`, and `v2_1_continuous.live.yaml`), deep-copy them, replace only approved fields, and write `b0.yaml`, `hardware.yaml`, `runtime.yaml`, and `live.yaml` under the campaign directory. The session-local hardware profile changes only the DCA resource, I2C port, and I2C address because the existing runtime rejects endpoint disagreement between runtime and driver profile. Preserve all BERT and measurement metadata unchanged. Return:

```python
@dataclass(frozen=True, slots=True)
class SessionPaths:
    campaign_root: Path
    controller_root: Path
    engine_artifact_root: Path
    b0: Path
    hardware: Path
    runtime: Path
    live: Path
```

Use atomic YAML writes. Validate generated B0 through `load_b0`, the session hardware profile through `load_live_hardware_config`, and the generated runtime through `load_runtime_config`. Confirm the loaded runtime endpoints equal the loaded profile endpoints before returning.

- [ ] **Step 4: Run tests and commit**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_session_config.py -q`

Expected: all tests pass.

```powershell
git add src/vcsel_ai_tuning/desktop/session_config.py tests_py/ai_tuning/desktop/test_session_config.py
git commit -m "feat(desktop): generate isolated campaign configs"
```

### Task 4: Minimal lineage and distribution integrity gate

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/integrity.py`
- Create: `tests_py/ai_tuning/desktop/test_integrity.py`

- [ ] **Step 1: Write failing integrity tests**

```python
def test_manifest_accepts_exact_protected_files(tmp_path):
    protected = tmp_path / "engine.py"
    protected.write_text("validated", encoding="utf-8")
    manifest = build_manifest(tmp_path, [Path("engine.py")], desktop_commit="abc")
    assert verify_manifest(tmp_path, manifest).verified is True

def test_manifest_rejects_modified_protected_file(tmp_path):
    protected = tmp_path / "engine.py"
    protected.write_text("validated", encoding="utf-8")
    manifest = build_manifest(tmp_path, [Path("engine.py")], desktop_commit="abc")
    protected.write_text("modified", encoding="utf-8")
    result = verify_manifest(tmp_path, manifest)
    assert result.verified is False
    assert "engine.py" in result.errors[0]
```

- [ ] **Step 2: Verify failure**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_integrity.py -q`

Expected: import failure for manifest functions.

- [ ] **Step 3: Implement SHA-256 manifest checks**

```python
VALIDATED_TAG = "v2.1-continuous-eye-b-validated-20260817"
VALIDATED_COMMIT = "4085b007546f820820dbdb7f7f0bba6ebc4b2049"

class BuildManifest(FrozenModel):
    schema_version: Literal[1]
    validated_tag: Literal[VALIDATED_TAG]
    validated_commit: Literal[VALIDATED_COMMIT]
    desktop_commit: str
    files: dict[str, str]
```

Reject missing, additional protected-path mismatches, invalid hashes, and malformed manifests. Keep historical viewing/export available but return `verified=False` for hardware enablement.

- [ ] **Step 4: Run tests and commit**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_integrity.py -q`

Expected: all tests pass.

```powershell
git add src/vcsel_ai_tuning/desktop/integrity.py tests_py/ai_tuning/desktop/test_integrity.py
git commit -m "feat(desktop): add minimal validated-build gate"
```

### Task 5: Read-only result projection and CSV export

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/artifacts.py`
- Create: `src/vcsel_ai_tuning/desktop/csv_export.py`
- Create: `tests_py/ai_tuning/desktop/conftest.py`
- Create: `tests_py/ai_tuning/desktop/test_artifacts.py`
- Create: `tests_py/ai_tuning/desktop/test_csv_export.py`

- [ ] **Step 1: Define the persisted-evidence fixture and write failing artifact tests**

In `conftest.py`, define a `CampaignFixture` helper with `root: Path` and `round(index) -> Path`. Its fixture writes a minimal real-layout controller tree (`state.json`, `rounds/<index>/round.json`, `kimi-response-attempts.jsonl`) plus engine observation and eye-evidence files. Round 0 is B0; rounds 1 and 2 are measured proposals; round 2 is best. Keep all JSON keys aligned with the checked-in continuous `RoundRecord` and observation schemas.

```python
def test_reader_orders_rounds_and_resolves_eye_evidence(campaign_fixture):
    result = CampaignArtifactReader(campaign_fixture.root).read()
    assert [point.round_index for point in result.points] == [0, 1, 2]
    assert result.points[1].eye_path.name == "eye-1.png"
    assert result.best_point.round_index == 2

def test_corrupt_round_is_reported_not_silently_skipped(campaign_fixture):
    campaign_fixture.round(2).write_text("bad-json", encoding="utf-8")
    with pytest.raises(DesktopArtifactError, match="round 2"):
        CampaignArtifactReader(campaign_fixture.root).read()
```

- [ ] **Step 2: Write failing CSV tests**

```python
def test_csv_contains_b0_and_one_row_per_measured_round(campaign_fixture, tmp_path):
    result = CampaignArtifactReader(campaign_fixture.root).read()
    output = export_campaign_csv(result, tmp_path / "results.csv")
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8-sig")))
    assert [row["row_type"] for row in rows] == ["B0", "ROUND", "ROUND"]
    assert rows[2]["readback_eq"] == "23"
    assert rows[2]["tdecq_db"] == "3.16"
```

- [ ] **Step 3: Implement strict read-only projections**

Parse controller `state.json`, sorted `rounds/*/round.json`, Kimi attempts, and engine observation JSON. Resolve eye paths only from persisted evidence references; never guess missing evidence.

- [ ] **Step 4: Implement atomic UTF-8-BOM CSV export**

Use `csv.DictWriter`, a fixed field order, a temporary file plus `os.replace`, and `utf-8-sig` so Excel opens Chinese text correctly. Include the approved identity, requested, full readback, metric, status, best-so-far, eye, and Kimi fields.

- [ ] **Step 5: Run tests and commit**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_artifacts.py tests_py/ai_tuning/desktop/test_csv_export.py -q`

Expected: all tests pass.

```powershell
git add src/vcsel_ai_tuning/desktop/artifacts.py src/vcsel_ai_tuning/desktop/csv_export.py tests_py/ai_tuning/desktop
git commit -m "feat(desktop): project evidence and export CSV"
```

### Task 6: Desktop Session orchestration and safe stop

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/session.py`
- Create: `tests_py/ai_tuning/desktop/test_session.py`

- [ ] **Step 1: Write failing orchestration tests**

Define local `inputs` and deterministic `fake_engine` pytest fixtures plus `approved_running_session(engine, inputs)`, which runs PRE-FLIGHT and then calls `approve_and_start()`. The fake adapter records preflight, commission, shutdown, and stop calls and writes only temporary artifact data.

```python
def test_commission_cannot_start_before_preflight_approval(fake_engine, inputs):
    session = DesktopSession(fake_engine)
    with pytest.raises(DesktopSessionError, match="approval"):
        session.approve_and_start()

def test_preflight_summary_proves_no_write_and_no_bert(fake_engine, inputs):
    summary = DesktopSession(fake_engine).run_preflight(inputs)
    assert summary.dut_write_performed is False
    assert summary.bert_contacted is False

def test_safe_stop_delegates_to_existing_controller(fake_engine, inputs):
    session = approved_running_session(fake_engine, inputs)
    session.request_stop()
    assert fake_engine.stop_requests == 1

def test_api_key_is_removed_on_success_and_failure(fake_engine, inputs, monkeypatch):
    session = DesktopSession(fake_engine)
    session.run_preflight(inputs)
    assert "MOONSHOT_API_KEY" not in os.environ
    fake_engine.fail_commission = True
    session.approve_and_start()
    with pytest.raises(RuntimeError):
        session.wait(timeout=2)
    assert "MOONSHOT_API_KEY" not in os.environ
```

- [ ] **Step 2: Verify failure**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_session.py -q`

Expected: import failure for `DesktopSession`.

- [ ] **Step 3: Implement the session state machine**

Define an injected `EngineAdapter` protocol and a production `ContinuousLiveEngineAdapter`. `run_preflight(inputs)` stores the immutable pending inputs and builds a fresh `LazyLiveActivation` for `.preflight()`. `approve_and_start()` is legal only after successful PRE-FLIGHT and is itself the operator's explicit approval action; it builds another fresh activation from the stored inputs, stores the active controller, and calls `controller.start(request)` in the worker. `request_stop()` delegates to the active controller. All engine shutdown and environment-secret cleanup lives in `finally`.

- [ ] **Step 4: Implement progress without live experiment data**

Expose only `ProgressSnapshot(current_round, max_rounds, phase)` by counting validated persisted round records. Do not expose metrics, parameters, eyes, or provider content until the terminal result projection.

- [ ] **Step 5: Run tests and commit**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_session.py -q`

Expected: all tests pass.

```powershell
git add src/vcsel_ai_tuning/desktop/session.py tests_py/ai_tuning/desktop/test_session.py
git commit -m "feat(desktop): orchestrate approved continuous sessions"
```

### Task 7: Minimal Tkinter UI

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/app.py`
- Create: `tests_py/ai_tuning/desktop/test_app.py`

- [ ] **Step 1: Define a headless-safe harness and write failing UI state tests against a fake session**

In `test_app.py`, define `AppHarness` around a real withdrawn `tk.Tk()` root and an injected synchronous fake `DesktopSession`; skip only if constructing the root raises `tk.TclError`. The harness locates widgets by the explicit `name=` passed at construction, drains queued callbacks with `update()`, and tracks the approval `Toplevel`. Do not mock Tk widget behavior.

```python
def test_successful_preflight_requires_explicit_approval(app_harness):
    app_harness.click_preflight()
    app_harness.drain_events()
    assert app_harness.session.commission_calls == 0
    assert app_harness.approval_dialog.visible is True

def test_running_view_contains_only_progress_and_safe_stop(app_harness):
    app_harness.approve()
    names = app_harness.visible_widget_names()
    assert {"round_progress", "safe_stop"} <= names
    assert not {"tdecq", "eye", "readback", "kimi"} & names

def test_window_close_while_running_requests_safe_stop(app_harness):
    app_harness.approve()
    app_harness.close_window()
    assert app_harness.session.stop_requests == 1
```

- [ ] **Step 2: Verify failure**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_app.py -q`

Expected: import failure for `DesktopApp`.

- [ ] **Step 3: Implement settings and approval UI**

Build the nine B0 inputs from one ordered field tuple, mask the API key, validate locally, show build identity, and keep bench endpoints in a collapsed ttk frame. PRE-FLIGHT work runs outside the Tk main thread and returns through `root.after` polling of a queue.

- [ ] **Step 4: Implement minimal running and result UI**

The running frame contains only progress text and safe stop. The result frame contains terminal summary, a scrollable measured-point table, selected-point eye preview, read-only Kimi attempt text, restore/evidence status, open-folder action, and CSV export.

- [ ] **Step 5: Run headless-safe UI tests and commit**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_app.py -q`

Expected: all tests pass or skip only when Tk cannot create a test root in a headless environment.

```powershell
git add src/vcsel_ai_tuning/desktop/app.py tests_py/ai_tuning/desktop/test_app.py
git commit -m "feat(desktop): add minimal Tkinter operator flow"
```

### Task 8: Entry point, build manifest, and PyInstaller package

**Files:**
- Create: `src/vcsel_ai_tuning/desktop/main.py`
- Create: `src/vcsel_ai_tuning/desktop/__main__.py`
- Create: `packaging/v2_1_desktop.spec`
- Create: `scripts/build_v2_1_desktop.ps1`
- Modify: `pyproject.toml`
- Create: `tests_py/ai_tuning/desktop/test_entrypoint.py`
- Create: `tests_py/ai_tuning/desktop/test_build_script.py`

- [ ] **Step 1: Write failing entry/build tests**

```python
def test_entrypoint_offline_flag_never_builds_live_adapter(monkeypatch):
    monkeypatch.setattr(main, "build_live_session", forbidden)
    assert main.run(["--offline-smoke-test"]) == 0

def test_build_script_requires_validated_base_and_protected_diff_check():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "4085b007546f820820dbdb7f7f0bba6ebc4b2049" in text
    assert "git diff" in text
    assert "--onedir" in text or "v2_1_desktop.spec" in text
```

- [ ] **Step 2: Verify failure**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_entrypoint.py tests_py/ai_tuning/desktop/test_build_script.py -q`

Expected: imports/files missing.

- [ ] **Step 3: Implement entry point and package configuration**

Add `vcsel-continuous-eye-desktop = "vcsel_ai_tuning.desktop.main:main"` and a `desktop-build` optional dependency containing PyInstaller. Resolve templates and build manifest relative to `sys._MEIPASS` when frozen and the repository/package root in development. `--offline-smoke-test` must load integrity, settings, fake session, and result projections without creating Tk or hardware adapters.

- [ ] **Step 4: Implement fail-closed PowerShell build**

The script must:

1. require PowerShell 7 and a clean desktop worktree,
2. verify the validated commit is an ancestor,
3. verify protected engine/config/launcher paths match the validated tag,
4. generate `build-manifest.json`,
5. call the current interpreter's PyInstaller with the checked-in spec,
6. run `VCSELContinuousEye.exe --offline-smoke-test`, and
7. emit the absolute distribution path and hashes.

- [ ] **Step 5: Run tests and commit**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop/test_entrypoint.py tests_py/ai_tuning/desktop/test_build_script.py -q`

Expected: all tests pass.

```powershell
git add src/vcsel_ai_tuning/desktop packaging scripts/build_v2_1_desktop.ps1 pyproject.toml tests_py/ai_tuning/desktop
git commit -m "build(desktop): add verified Windows onedir package"
```

### Task 9: End-to-end fake workflow and packaged EXE verification

**Files:**
- Create: `tests_py/ai_tuning/desktop/test_fake_e2e.py`
- Modify only as required by test failures: desktop files created above

- [ ] **Step 1: Write the full fake workflow test**

```python
def test_fake_desktop_workflow_exports_complete_csv(tmp_path):
    session = fake_desktop_session(tmp_path, rounds=3)
    preflight = session.run_preflight(INPUTS)
    assert preflight.approvable is True
    session.approve_and_start()
    result = session.wait(timeout=10)
    assert result.b0_restored is True
    rows = list(csv.DictReader(session.export_csv(tmp_path / "result.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 4
    assert rows[0]["row_type"] == "B0"
```

Define `fake_desktop_session` in the test module. It composes the real `DesktopSession`, `CampaignArtifactReader`, and CSV exporter with a deterministic fake `EngineAdapter` that writes the same persisted fixture schema used in Task 5; it must not import or construct live hardware/provider adapters.

- [ ] **Step 2: Run the focused desktop suite**

Run: `py -3.13 -m pytest tests_py/ai_tuning/desktop -q`

Expected: all desktop tests pass.

- [ ] **Step 3: Run existing continuous regression tests**

Run: `py -3.13 -m pytest tests_py/ai_tuning/test_continuous_live.py tests_py/ai_tuning/v2/continuous -q`

Expected: all tests pass.

- [ ] **Step 4: Install build dependency in an isolated build environment and build**

```powershell
py -3.13 -m venv --system-site-packages .venv-desktop-build
.\.venv-desktop-build\Scripts\python.exe -m pip install "pyinstaller>=6.16,<7"
& "C:\Users\NateS\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe" -NoProfile -File .\scripts\build_v2_1_desktop.ps1 -Python .\.venv-desktop-build\Scripts\python.exe
```

Expected: the script prints a verified `dist\VCSELContinuousEye\VCSELContinuousEye.exe` path and the offline smoke test exits 0.

- [ ] **Step 5: Run full offline regression**

Run: `py -3.13 -m pytest tests_py/ai_tuning -q`

Expected: 0 failures; Windows symlink tests may retain the baseline skips.

- [ ] **Step 6: Commit verification assets and documentation only**

Do not commit `build`, `dist`, virtual environments, user settings, artifacts, or secrets.

```powershell
git add tests_py/ai_tuning/desktop docs
git commit -m "test(desktop): prove packaged offline workflow"
```

## Final Verification Checklist

- [ ] `git diff v2.1-continuous-eye-b-validated-20260817 --` shows only approved desktop, packaging, test, project metadata, and documentation changes.
- [ ] Protected engine, launcher, DCA/I2C, provider, policy, and restore files match the validated tag.
- [ ] `MOONSHOT_API_KEY` does not appear in tracked files or generated offline evidence.
- [ ] Desktop tests pass.
- [ ] Existing continuous regression tests pass.
- [ ] Full AI-tuning test suite has zero failures.
- [ ] Packaged `.exe --offline-smoke-test` exits 0.
- [ ] No real hardware or live Kimi request was attempted.
