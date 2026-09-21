"""Deterministic, connection-free V1.1 gateway and control runner."""

from __future__ import annotations

from datetime import datetime, timezone

from .baseline import B0Snapshot, DcaFingerprint, HardwareIdentity
from .domain import RuntimeMode, TaskType
from .regmap import RegMapField, ScriptEntry, scan_parameter_mapping
from .service import GatewayAttestation, PreflightCapture


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

    def health(self):
        return {"status": "ok", "hardware_connected": False}

    def capabilities(self):
        return {
            "campaign_types": ["AI_OPTIMIZATION", "SWEEP"],
            "dynamic_b0": True,
            "full_campaign_authorization": True,
            "hardware_mutation": False,
            "fake": True,
        }

    def preflight_capture(self) -> PreflightCapture:
        self._capture_count += 1
        return PreflightCapture(
            snapshot_token=f"fake-snapshot-{self._capture_count:08d}",
            b0=self._snapshot,
            mapping=self._mapping,
        )

    def current_attestation(self) -> GatewayAttestation:
        return GatewayAttestation.from_capture(
            PreflightCapture(
                snapshot_token="fake-current",
                b0=self._snapshot,
                mapping=self._mapping,
            )
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
