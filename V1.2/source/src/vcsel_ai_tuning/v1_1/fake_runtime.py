"""Deterministic, connection-free V1.1 gateway and control runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .baseline import B0Snapshot, DcaFingerprint, HardwareIdentity
from .domain import RuntimeMode, TaskType
from .history_apply import HistoryHardwareState, HistoryResultTarget
from .regmap import RegMapField, ScriptEntry, scan_parameter_mapping
from .service import GatewayAttestation, PreflightCapture, PreflightEvidence


class FakeHistoryHardware:
    """Deterministic in-memory History Apply hardware for the FAKE runtime."""

    fake = True

    def __init__(
        self,
        b0: B0Snapshot,
        *,
        fail_at: str | None = None,
        restore_fails: bool = False,
    ) -> None:
        if not isinstance(b0, B0Snapshot):
            raise TypeError("b0 must be a B0Snapshot")
        if fail_at not in {None, "i2c", "fir", "measure"}:
            raise ValueError("unsupported FAKE History Apply failure stage")
        self._baseline = HistoryHardwareState(
            i2c=dict(b0.i2c_fields),
            fir=b0.fir_taps,
            upper_nlc=b0.upper_nlc,
            lower_nlc=b0.lower_nlc,
            pattern=b0.bert_pattern,
            bert_port=b0.bert_port,
            bert_channel=b0.bert_channel,
            dca_fingerprint=b0.dca_fingerprint.content_hash,
        )
        self._state = self._baseline
        self._fail_at = fail_at
        self._restore_fails = restore_fails
        self.calls: list[tuple[object, ...]] = []

    @property
    def state(self) -> HistoryHardwareState:
        return self._state

    def preflight_for_history(self, target: HistoryResultTarget) -> None:
        self.calls.append(("preflight", target.observation_id))
        if (
            set(target.i2c) != set(self._baseline.i2c)
            or target.pattern != self._baseline.pattern
            or target.bert_port != self._baseline.bert_port
            or target.bert_channel != self._baseline.bert_channel
            or target.dca_fingerprint != self._baseline.dca_fingerprint
        ):
            raise ValueError("FAKE History Apply target does not match B0 hardware")

    def capture_current_state(self) -> HistoryHardwareState:
        self.calls.append(("capture",))
        return self._state

    def apply_i2c(
        self,
        requested: Mapping[str, int],
        *,
        rollback_values: Mapping[str, int],
    ) -> Mapping[str, int]:
        self.calls.append(("i2c", dict(requested), dict(rollback_values)))
        if dict(rollback_values) != dict(self._state.i2c):
            raise ValueError("FAKE History Apply I2C rollback snapshot drifted")
        if self._fail_at == "i2c":
            raise RuntimeError("fake I2C apply failed")
        self._state = replace(self._state, i2c=dict(requested))
        return dict(self._state.i2c)

    def apply_fir(
        self,
        target: HistoryResultTarget,
        *,
        rollback_state: HistoryHardwareState,
    ) -> tuple[int, ...]:
        self.calls.append(("fir", tuple(target.fir), rollback_state.fir))
        if self._fail_at == "fir":
            raise RuntimeError("fake FIR apply failed")
        self._state = replace(
            self._state,
            fir=target.fir,
            upper_nlc=target.upper_nlc,
            lower_nlc=target.lower_nlc,
        )
        return self._state.fir

    def measure_once(self) -> Mapping[str, Any]:
        self.calls.append(("measure",))
        if self._fail_at == "measure":
            raise RuntimeError("fake measurement failed")
        i2c_delta = sum(
            (self._state.i2c[name] - baseline) ** 2
            for name, baseline in self._baseline.i2c.items()
        )
        fir_delta = sum(
            abs(value - baseline)
            for value, baseline in zip(
                self._state.fir,
                self._baseline.fir,
                strict=True,
            )
        )
        return {
            "status": "CORR",
            "tdecq": round(2.8 - 0.02 * i2c_delta + 0.002 * fir_delta, 6),
            "er": round(5.4 + 0.01 * i2c_delta - 0.001 * fir_delta, 6),
            "fake": True,
        }

    def restore_state_once(self, snapshot: HistoryHardwareState) -> None:
        self.calls.append(("restore", snapshot))
        if self._restore_fails:
            raise RuntimeError("fake restore failed")
        self._state = snapshot


class FakeV11Gateway:
    """Static readback model.  It never imports or constructs hardware drivers."""

    def __init__(self) -> None:
        entries = (
            ScriptEntry("ibias_dac", 64, 0),
            ScriptEntry("vref", 8, 1),
            ScriptEntry("vbias_ref", 8, 2),
            ScriptEntry("vga_cur", 16, 3),
            ScriptEntry("res_gain", 12, 4),
            ScriptEntry("res_bw", 10, 5),
            ScriptEntry("gain_ctrl", 16, 6),
            ScriptEntry("efin_cur", 12, 7),
            ScriptEntry("eq", 20, 8),
        )
        fields = tuple(
            RegMapField(
                name=entry.name,
                channel0_address=0x2000 + index,
                channel_stride=0x100,
                bit_start=0,
                bits=8,
                encoding_min=0,
                encoding_max=255,
            )
            for index, entry in enumerate(entries)
        )
        self._mapping = scan_parameter_mapping(entries, fields)
        i2c = {entry.name: entry.current_value for entry in entries}
        self._snapshot = B0Snapshot.capture(
            b0_id="fake-live-b0",
            version=1,
            captured_at=datetime.now(timezone.utc),
            i2c_fields=i2c,
            i2c_raw_registers={
                field.channel0_address: i2c[field.name] for field in fields
            },
            fir_taps=(0, 9, -28, 104, -8, 1, -3),
            upper_nlc=6,
            lower_nlc=6,
            bert_pattern="SSPRQ",
            i2c_channel=0,
            bert_port="port1",
            bert_channel="0",
            dca_tdecq_source="FUNC1",
            dca_tdecq_equalizer_function=1,
            dca_outer_er_source="CHAN1A",
            dca_fingerprint=DcaFingerprint.from_mappings(
                identity="FAKE,FlexDCA,1.1",
                measurement_sources={
                    "outer_er": "CHAN1A",
                    "tdecq": "FUNC1",
                },
                configuration={
                    "baud_rate": "53.125 GBd",
                    "pattern": "SSPRQ",
                    "tdecq_equalizer_function": "FUNC1",
                    "tdecq_input_channel": "CHAN1A",
                },
            ),
            hardware_identities={
                "i2c": HardwareIdentity("i2c", "FAKE-COM", "0x52:0x00"),
                "bert": HardwareIdentity("bert", "FAKE-UDP", "FAKE-VIT"),
                "dca": HardwareIdentity("dca", "FAKE-VISA", "FAKE-FlexDCA"),
            },
        )
        self._capture_count = 0

    @property
    def runtime_mode(self) -> RuntimeMode:
        return RuntimeMode.FAKE

    @property
    def b0(self) -> B0Snapshot:
        return self._snapshot

    def health(self):
        return {"status": "ok", "hardware_connected": False}

    def capabilities(self):
        return {
            "campaign_types": ["AI_OPTIMIZATION", "SWEEP"],
            "dynamic_b0": True,
            "full_campaign_authorization": True,
            "hardware_connected": False,
            "hardware_mutation": False,
            "fake": True,
        }

    def preflight_capture(self) -> PreflightCapture:
        self._capture_count += 1
        return PreflightCapture(
            snapshot_token=f"fake-snapshot-{self._capture_count:08d}",
            b0=self._snapshot,
            mapping=self._mapping,
            evidence=self._evidence(),
        )

    def current_attestation(self) -> GatewayAttestation:
        return GatewayAttestation.from_capture(
            PreflightCapture(
                snapshot_token="fake-current",
                b0=self._snapshot,
                mapping=self._mapping,
                evidence=self._evidence(),
            )
        )

    @staticmethod
    def _evidence() -> PreflightEvidence:
        return PreflightEvidence.from_payload(
            {
                "fake": True,
                "targets": {
                    "bert": "FAKE-UDP",
                    "dca": "FAKE-VISA",
                    "i2c": "FAKE-COM",
                },
                "mutation_counts": {
                    "bert_writes": 0,
                    "dca_writes": 0,
                    "i2c_page_select_writes": 0,
                    "i2c_tunable_register_writes": 0,
                },
            }
        )


class FakeV11TaskRunner:
    """Control-signal recorder used by API/UI tests; no hardware side effects."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def start(self, task_id: str, task_type: TaskType) -> None:
        self.calls.append(("start", task_id, task_type.value))

    def pause_and_restore_b0(self, task_id: str) -> None:
        self.calls.append(("pause_restore_b0", task_id))

    def resume_from_b0(self, task_id: str) -> None:
        self.calls.append(("resume_from_b0", task_id))

    def abort_and_restore_b0(self, task_id: str) -> None:
        self.calls.append(("abort_restore_b0", task_id))

    def recover_b0(self, task_id: str) -> None:
        self.calls.append(("recover_b0", task_id))
