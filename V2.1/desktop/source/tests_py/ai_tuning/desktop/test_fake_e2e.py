import csv
import os

from vcsel_ai_tuning.desktop.models import (
    DEFAULT_B0,
    BenchSettings,
    DesktopPhase,
    DesktopResult,
    PreflightSummary,
    RunInputs,
)
from vcsel_ai_tuning.desktop.session import DesktopSession


INPUTS = RunInputs(
    api_key="fake-e2e-key",
    max_rounds=2,
    target_tdecq_db=2.8,
    b0=DEFAULT_B0,
    bench=BenchSettings(
        dca_visa_resource="TCPIP0::fake::hislip0,4880::INSTR",
        i2c_com_port="COM99",
        i2c_address=0xA6,
    ),
)


class PersistedFakeEngine:
    def __init__(self, fixture) -> None:
        self.fixture = fixture
        self.stop_requests = 0

    def preflight(self, inputs: RunInputs) -> PreflightSummary:
        assert os.environ["MOONSHOT_API_KEY"] == "fake-e2e-key"
        return PreflightSummary(
            campaign_id=self.fixture.campaign_id,
            dca_verified=True,
            i2c_readback_verified=True,
            kimi_proposal_valid=True,
            dut_write_performed=False,
            bert_contacted=False,
            details="fake offline preflight",
        )

    def commission(self, inputs: RunInputs) -> DesktopResult:
        assert os.environ["MOONSHOT_API_KEY"] == "fake-e2e-key"
        return DesktopResult(
            campaign_id=self.fixture.campaign_id,
            phase=DesktopPhase.COMPLETED,
            campaign_root=self.fixture.root,
            b0_restored=True,
            message="MAX_ROUNDS_REACHED",
        )

    def request_stop(self) -> None:
        self.stop_requests += 1

    def completed_rounds(self) -> int:
        return 2


def test_fake_desktop_workflow_exports_complete_csv(
    campaign_fixture, tmp_path
) -> None:
    session = DesktopSession(PersistedFakeEngine(campaign_fixture))

    preflight = session.run_preflight(INPUTS)
    assert preflight.approvable is True
    session.approve_and_start()
    result = session.wait(timeout=5)
    output = session.export_csv(tmp_path / "result.csv")
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig", newline="")))

    assert result.b0_restored is True
    assert len(rows) == 3
    assert rows[0]["row_type"] == "B0"
    assert rows[-1]["round_index"] == "2"
    assert rows[-1]["readback_eq"] == "23"
    assert "MOONSHOT_API_KEY" not in os.environ
