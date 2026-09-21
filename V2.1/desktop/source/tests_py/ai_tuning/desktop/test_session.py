from __future__ import annotations

import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from vcsel_ai_tuning.desktop.models import (
    DEFAULT_B0,
    BenchSettings,
    DesktopPhase,
    DesktopResult,
    PreflightSummary,
    RunInputs,
)
import vcsel_ai_tuning.desktop.session as session_module
from vcsel_ai_tuning.desktop.session import (
    ContinuousLiveEngineAdapter,
    DesktopSession,
    DesktopSessionError,
)
from vcsel_ai_tuning.desktop.session_config import SessionConfigBuilder
from vcsel_ai_tuning.v2.continuous.models import ContinuousState, ExecutionMode


@pytest.fixture
def inputs() -> RunInputs:
    return RunInputs(
        api_key="temporary-session-key",
        max_rounds=5,
        target_tdecq_db=2.8,
        b0=DEFAULT_B0,
        bench=BenchSettings(
            dca_visa_resource="TCPIP0::scope::hislip0,4880::INSTR",
            i2c_com_port="COM7",
            i2c_address=0xA6,
        ),
    )


class FakeEngine:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "desktop-test"
        self.root.mkdir()
        self.preflight_calls = 0
        self.commission_calls = 0
        self.stop_requests = 0
        self.completed = 0
        self.fail_preflight = False
        self.fail_commission = False
        self.commission_entered = Event()
        self.release_commission = Event()
        self.keys_seen: list[str | None] = []

    def preflight(self, inputs: RunInputs) -> PreflightSummary:
        self.preflight_calls += 1
        self.keys_seen.append(os.environ.get("MOONSHOT_API_KEY"))
        if self.fail_preflight:
            raise RuntimeError("preflight failed")
        return PreflightSummary(
            campaign_id="desktop-test",
            dca_verified=True,
            i2c_readback_verified=True,
            kimi_proposal_valid=True,
            dut_write_performed=False,
            bert_contacted=False,
            details="ready",
        )

    def commission(self, inputs: RunInputs) -> DesktopResult:
        self.commission_calls += 1
        self.keys_seen.append(os.environ.get("MOONSHOT_API_KEY"))
        self.commission_entered.set()
        if self.fail_commission:
            raise RuntimeError("commission failed")
        if not self.release_commission.wait(timeout=5):
            raise RuntimeError("fake commission timed out")
        self.completed = 2
        return DesktopResult(
            campaign_id="desktop-test",
            phase=DesktopPhase.COMPLETED,
            campaign_root=self.root,
            b0_restored=True,
            message="MAX_ROUNDS_REACHED",
        )

    def request_stop(self) -> None:
        self.stop_requests += 1
        self.release_commission.set()

    def completed_rounds(self) -> int:
        return self.completed


@pytest.fixture
def fake_engine(tmp_path) -> FakeEngine:
    return FakeEngine(tmp_path)


def test_commission_cannot_start_before_preflight_approval(fake_engine) -> None:
    session = DesktopSession(fake_engine)

    with pytest.raises(DesktopSessionError, match="approval"):
        session.approve_and_start()

    assert fake_engine.commission_calls == 0


def test_preflight_summary_proves_no_write_and_no_bert(
    fake_engine, inputs
) -> None:
    summary = DesktopSession(fake_engine).run_preflight(inputs)

    assert summary.approvable is True
    assert summary.dut_write_performed is False
    assert summary.bert_contacted is False
    assert fake_engine.keys_seen == ["temporary-session-key"]
    assert "MOONSHOT_API_KEY" not in os.environ


def test_safe_stop_delegates_to_engine_and_waits_for_terminal_restore(
    fake_engine, inputs
) -> None:
    session = DesktopSession(fake_engine)
    session.run_preflight(inputs)
    session.approve_and_start()
    assert fake_engine.commission_entered.wait(timeout=2)

    session.request_stop()
    result = session.wait(timeout=2)

    assert fake_engine.stop_requests == 1
    assert result.b0_restored is True
    assert session.phase is DesktopPhase.COMPLETED


def test_api_key_is_removed_after_async_commission_failure(
    fake_engine, inputs
) -> None:
    session = DesktopSession(fake_engine)
    session.run_preflight(inputs)
    fake_engine.fail_commission = True

    session.approve_and_start()
    with pytest.raises(RuntimeError, match="commission failed"):
        session.wait(timeout=2)

    assert "MOONSHOT_API_KEY" not in os.environ
    assert session.phase is DesktopPhase.FAILED


def test_preflight_can_retry_after_previous_preflight_failure(
    fake_engine, inputs
) -> None:
    session = DesktopSession(fake_engine)
    fake_engine.fail_preflight = True

    with pytest.raises(RuntimeError, match="preflight failed"):
        session.run_preflight(inputs)
    assert session.phase is DesktopPhase.FAILED

    fake_engine.fail_preflight = False
    summary = session.run_preflight(inputs)

    assert summary.approvable is True
    assert session.phase is DesktopPhase.AWAITING_APPROVAL
    assert fake_engine.preflight_calls == 2


def test_existing_environment_key_is_restored_not_overwritten(
    fake_engine, inputs, monkeypatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "pre-existing")
    session = DesktopSession(fake_engine)

    session.run_preflight(inputs)

    assert fake_engine.keys_seen == ["temporary-session-key"]
    assert os.environ["MOONSHOT_API_KEY"] == "pre-existing"


def test_progress_exposes_only_round_count_maximum_and_phase(
    fake_engine, inputs
) -> None:
    session = DesktopSession(fake_engine)
    session.run_preflight(inputs)
    session.approve_and_start()
    assert fake_engine.commission_entered.wait(timeout=2)

    progress = session.progress()

    assert progress.model_dump() == {
        "phase": DesktopPhase.RUNNING,
        "current_round": 0,
        "max_rounds": 5,
    }
    session.request_stop()
    session.wait(timeout=2)


def test_rejected_preflight_cannot_be_approved(fake_engine, inputs) -> None:
    def rejected(_: RunInputs) -> PreflightSummary:
        return PreflightSummary(
            campaign_id="desktop-test",
            dca_verified=True,
            i2c_readback_verified=False,
            kimi_proposal_valid=True,
            dut_write_performed=False,
            bert_contacted=False,
            details="B0 mismatch",
        )

    fake_engine.preflight = rejected
    session = DesktopSession(fake_engine)

    summary = session.run_preflight(inputs)
    with pytest.raises(DesktopSessionError, match="approval"):
        session.approve_and_start()

    assert summary.approvable is False
    assert session.phase is DesktopPhase.FAILED


def test_cancel_approval_discards_pending_secret_and_returns_idle(
    fake_engine, inputs
) -> None:
    session = DesktopSession(fake_engine)
    session.run_preflight(inputs)

    session.cancel_approval()

    assert session.phase is DesktopPhase.IDLE
    with pytest.raises(DesktopSessionError, match="approval"):
        session.approve_and_start()
    assert fake_engine.commission_calls == 0


def test_wait_timeout_does_not_abandon_running_worker(fake_engine, inputs) -> None:
    session = DesktopSession(fake_engine)
    session.run_preflight(inputs)
    session.approve_and_start()
    assert fake_engine.commission_entered.wait(timeout=2)

    with pytest.raises(TimeoutError, match="still running"):
        session.wait(timeout=0.01)

    session.request_stop()
    assert session.wait(timeout=2).phase is DesktopPhase.COMPLETED


def _live_report(inputs: RunInputs) -> dict[str, object]:
    return {
        "dut_write_performed": False,
        "hardware_write_performed": False,
        "bert_contacted": False,
        "kimi_connectivity_verified": True,
        "provider_action": "PROPOSE",
        "i2c_readback": inputs.b0.model_dump(),
        "adapter_identities": {
            "dca": {"identity_verified": True, "configuration_verified": True},
            "i2c": {"identity_verified": True, "configuration_verified": True},
        },
    }


def test_live_adapter_uses_fresh_preflight_and_commission_activations(
    tmp_path, inputs, monkeypatch
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    adapter = ContinuousLiveEngineAdapter(
        SessionConfigBuilder(repository_root / "configs" / "ai_tuning"),
        tmp_path,
        campaign_id_factory=lambda: "desktop-live-test",
    )
    calls: list[dict[str, object]] = []
    shutdown_calls: list[bool] = []

    class PreflightActivation:
        def preflight(self) -> dict[str, object]:
            return _live_report(inputs)

    result = SimpleNamespace(
        campaign_id="desktop-live-test",
        state=ContinuousState.COMPLETED,
        b0_restored=True,
        stop_reason="MAX_ROUNDS_REACHED",
    )
    controller = SimpleNamespace(start=lambda request: result)
    request = SimpleNamespace(campaign_id="desktop-live-test")
    activated = SimpleNamespace(
        controller=controller,
        request=request,
        shutdown=lambda: shutdown_calls.append(True),
    )

    class CommissionActivation:
        def activate(self):
            return activated

    activations = [PreflightActivation(), CommissionActivation()]

    def fake_build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(live_activation=activations.pop(0))

    monkeypatch.setattr(session_module, "build_continuous_dependencies", fake_build)

    summary = adapter.preflight(inputs)
    desktop_result = adapter.commission(inputs)

    assert summary.approvable is True
    assert desktop_result.phase is DesktopPhase.COMPLETED
    assert desktop_result.b0_restored is True
    assert shutdown_calls == [True]
    assert len(calls) == 2
    assert all(call["mode"] is ExecutionMode.LIVE for call in calls)
    assert all(call["allow_device_contact"] is True for call in calls)


def test_live_adapter_rejects_commission_input_changed_after_preflight(
    tmp_path, inputs, monkeypatch
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    adapter = ContinuousLiveEngineAdapter(
        SessionConfigBuilder(repository_root / "configs" / "ai_tuning"),
        tmp_path,
        campaign_id_factory=lambda: "desktop-live-test",
    )
    activation = SimpleNamespace(preflight=lambda: _live_report(inputs))
    monkeypatch.setattr(
        session_module,
        "build_continuous_dependencies",
        lambda **kwargs: SimpleNamespace(live_activation=activation),
    )
    adapter.preflight(inputs)
    changed = inputs.model_copy(update={"max_rounds": inputs.max_rounds + 1})

    with pytest.raises(DesktopSessionError, match="do not match"):
        adapter.commission(changed)


def test_live_adapter_delivers_safe_stop_to_active_validated_controller(
    tmp_path, inputs, monkeypatch
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    adapter = ContinuousLiveEngineAdapter(
        SessionConfigBuilder(repository_root / "configs" / "ai_tuning"),
        tmp_path,
        campaign_id_factory=lambda: "desktop-live-stop",
    )
    entered = Event()
    released = Event()
    stop_calls: list[str] = []
    terminal = SimpleNamespace(
        campaign_id="desktop-live-stop",
        state=ContinuousState.COMPLETED,
        b0_restored=True,
        stop_reason="USER_STOP_ACCEPTED",
    )

    class Controller:
        def start(self, request):
            entered.set()
            if not released.wait(timeout=5):
                raise RuntimeError("stop was not delivered")
            return terminal

        def get_status(self, campaign_id):
            return SimpleNamespace(state=ContinuousState.RUNNING)

        def request_stop(self, campaign_id):
            stop_calls.append(campaign_id)
            released.set()

    activated = SimpleNamespace(
        controller=Controller(),
        request=SimpleNamespace(campaign_id="desktop-live-stop"),
        shutdown=lambda: None,
    )
    activations = [
        SimpleNamespace(preflight=lambda: _live_report(inputs)),
        SimpleNamespace(activate=lambda: activated),
    ]
    monkeypatch.setattr(
        session_module,
        "build_continuous_dependencies",
        lambda **kwargs: SimpleNamespace(live_activation=activations.pop(0)),
    )
    session = DesktopSession(adapter)
    session.run_preflight(inputs)
    session.approve_and_start()
    assert entered.wait(timeout=2)

    session.request_stop()
    result = session.wait(timeout=2)

    assert stop_calls == ["desktop-live-stop"]
    assert result.message == "USER_STOP_ACCEPTED"
    assert result.b0_restored is True
