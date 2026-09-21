from __future__ import annotations

from pathlib import Path
from threading import Event
import tkinter as tk

import pytest

from vcsel_ai_tuning.desktop.app import DesktopApp
from vcsel_ai_tuning.desktop.integrity import IntegrityResult
from vcsel_ai_tuning.desktop.models import (
    DesktopPhase,
    DesktopResult,
    PreflightSummary,
    ProgressSnapshot,
)
from vcsel_ai_tuning.desktop.settings import BenchSettingsStore


class FakeSession:
    def __init__(self) -> None:
        self.phase = DesktopPhase.IDLE
        self.preflight_calls = 0
        self.commission_calls = 0
        self.stop_requests = 0
        self.preflight_entered = Event()

    def run_preflight(self, inputs):
        self.preflight_calls += 1
        self.preflight_entered.set()
        self.phase = DesktopPhase.AWAITING_APPROVAL
        return PreflightSummary(
            campaign_id="desktop-test",
            dca_verified=True,
            i2c_readback_verified=True,
            kimi_proposal_valid=True,
            dut_write_performed=False,
            bert_contacted=False,
            details="ready",
        )

    def approve_and_start(self):
        self.commission_calls += 1
        self.phase = DesktopPhase.RUNNING

    def cancel_approval(self):
        self.phase = DesktopPhase.IDLE

    def request_stop(self):
        self.stop_requests += 1
        self.phase = DesktopPhase.STOPPING

    def progress(self):
        return ProgressSnapshot(
            phase=self.phase,
            current_round=1,
            max_rounds=5,
        )

    def wait(self, timeout=None):
        raise TimeoutError("still running")


class AppHarness:
    def __init__(self, root: tk.Tk, app: DesktopApp, session: FakeSession) -> None:
        self.root = root
        self.app = app
        self.session = session
        api_entry = self.find("api_key")
        api_entry.insert(0, "test-key")

    def find(self, name: str):
        pending = [self.root]
        while pending:
            parent = pending.pop()
            for child in parent.winfo_children():
                if child.winfo_name() == name:
                    return child
                pending.append(child)
        raise LookupError(f"widget {name!r} was not found")

    def click_preflight(self) -> None:
        root_button = self.find("preflight")
        root_button.invoke()

    def drain_events(self, predicate, attempts: int = 100) -> None:
        pause = Event()
        for _ in range(attempts):
            self.root.update()
            if predicate():
                return
            pause.wait(0.01)
        raise AssertionError("Tk event condition was not reached")

    def approval_exists(self) -> bool:
        return any(child.winfo_name() == "approval" for child in self.root.winfo_children())

    def approve(self) -> None:
        self.drain_events(self.approval_exists)
        button = self.find("approve_start")
        button.invoke()
        self.root.update()

    def current_widget_names(self) -> set[str]:
        names: set[str] = set()

        def collect(widget) -> None:
            for child in widget.winfo_children():
                names.add(child.winfo_name())
                collect(child)

        collect(self.app.current_frame)
        return names


@pytest.fixture(scope="module")
def tk_parent():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk root is unavailable: {error}")
    root.withdraw()
    yield root
    if root.winfo_exists():
        root.destroy()


@pytest.fixture
def app_harness(tmp_path, tk_parent):
    root = tk.Toplevel(tk_parent)
    root.withdraw()
    session = FakeSession()
    app = DesktopApp(
        root,
        session,
        BenchSettingsStore(tmp_path / "settings.json"),
        IntegrityResult(verified=True),
    )
    harness = AppHarness(root, app, session)
    yield harness
    if root.winfo_exists():
        root.destroy()


def test_successful_preflight_requires_explicit_approval(app_harness) -> None:
    app_harness.click_preflight()
    app_harness.drain_events(app_harness.approval_exists)

    assert app_harness.session.preflight_calls == 1
    assert app_harness.session.commission_calls == 0
    assert app_harness.approval_exists() is True


def test_running_view_contains_only_progress_and_safe_stop(app_harness) -> None:
    app_harness.click_preflight()
    app_harness.approve()

    names = app_harness.current_widget_names()

    assert {"round_progress", "safe_stop"} <= names
    assert not {"tdecq", "eye", "readback", "kimi"} & names
    assert app_harness.session.commission_calls == 1


def test_window_close_while_running_requests_safe_stop(app_harness) -> None:
    app_harness.click_preflight()
    app_harness.approve()

    app_harness.app.request_close()

    assert app_harness.session.stop_requests == 1
    assert app_harness.root.winfo_exists() == 1


def test_integrity_failure_disables_real_preflight(tmp_path, tk_parent) -> None:
    root = tk.Toplevel(tk_parent)
    root.withdraw()
    try:
        app = DesktopApp(
            root,
            FakeSession(),
            BenchSettingsStore(tmp_path / "settings.json"),
            IntegrityResult(verified=False, errors=("engine hash mismatch",)),
        )
        harness = AppHarness.__new__(AppHarness)
        harness.root = root
        button = AppHarness.find(harness, "preflight")
        history = AppHarness.find(harness, "open_history")

        assert str(button.cget("state")) == "disabled"
        assert str(history.cget("state")) != "disabled"
        assert "engine hash mismatch" in app.status_text.get()
    finally:
        root.destroy()


def test_form_uses_exact_default_b0_and_three_bench_endpoints(app_harness) -> None:
    inputs = app_harness.app.collect_inputs()

    assert inputs.b0.model_dump() == {
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
    assert set(inputs.bench.model_dump()) == {
        "dca_visa_resource",
        "i2c_com_port",
        "i2c_address",
    }


def test_terminal_result_renders_all_points_and_selected_evidence(
    app_harness, campaign_fixture
) -> None:
    app_harness.app._show_result(
        DesktopResult(
            campaign_id=campaign_fixture.campaign_id,
            phase=DesktopPhase.COMPLETED,
            campaign_root=campaign_fixture.root,
            b0_restored=True,
            message="MAX_ROUNDS_REACHED",
        )
    )

    tree = app_harness.find("points")

    assert len(tree.get_children()) == 3
    assert app_harness.app.current_frame.winfo_name() == "result"
    assert "kimi" in app_harness.current_widget_names()
