from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml


I2C_NAMES = (
    "ibias_dac",
    "vref",
    "vbias_ref",
    "vga_cur",
    "res_gain",
    "res_bw",
    "gain_ctrl",
    "efin_cur",
    "eq",
)
BASELINE = {
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


@dataclass(frozen=True, slots=True)
class CampaignFixture:
    root: Path
    campaign_id: str

    @property
    def controller(self) -> Path:
        return self.root / "controller" / self.campaign_id

    @property
    def engine(self) -> Path:
        return self.root / "engine" / self.campaign_id

    def round(self, index: int) -> Path:
        return self.controller / "rounds" / f"{index:04d}" / "round.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _round(
    campaign_id: str,
    index: int,
    before_id: str,
    after_id: str,
    before_tdecq: float,
    after_tdecq: float,
    eq: int,
) -> dict[str, object]:
    readback = BASELINE | {"eq": eq}
    return {
        "campaign_id": campaign_id,
        "round_index": index,
        "proposal_attempt": 1,
        "before_observation_id": before_id,
        "after_observation_id": after_id,
        "requested_params": {"eq": eq},
        "applied_params": {"eq": eq},
        "readback_params": readback,
        "before_metrics": {"outer_er_db": 3.0, "tdecq_db": before_tdecq},
        "after_metrics": {"outer_er_db": 3.0 + index / 10, "tdecq_db": after_tdecq},
        "measurement_status": "VALID",
        "started_at": f"2026-08-18T00:0{index}:00Z",
        "finished_at": f"2026-08-18T00:0{index}:10Z",
        "evidence_persisted": True,
        "stage_timings_s": {"total": 10.0},
    }


def _observation(
    fixture: CampaignFixture,
    observation_id: str,
    sequence: int,
    tdecq: float,
    eq: int,
) -> dict[str, object]:
    eye_relative = Path("eye_artifacts") / f"eye-{sequence}.png"
    eye_path = fixture.engine / eye_relative
    eye_path.parent.mkdir(parents=True, exist_ok=True)
    eye_bytes = b"\x89PNG\r\n\x1a\n" + bytes([sequence])
    eye_path.write_bytes(eye_bytes)
    return {
        "artifacts": [],
        "campaign_id": fixture.campaign_id,
        "candidate_id": f"candidate-{sequence}",
        "elapsed_s": 1.0,
        "evidence": {
            "png_path": eye_relative.as_posix(),
            "png_sha256": sha256(eye_bytes).hexdigest(),
            "provenance": "SCREEN_IMAGE_DERIVED",
        },
        "feasible": True,
        "measured_at": f"2026-08-18T00:0{sequence}:10Z",
        "metric_status": {"er_status": "CORR", "tdecq_status": "CORR"},
        "metrics": {"er_db": 3.0 + sequence / 10, "tdecq_db": tdecq},
        "observation_id": observation_id,
        "raw_responses": {},
        "settings": {
            "applied_fir": None,
            "applied_i2c": {"eq": eq},
            "quantized_i2c": {"eq": eq},
            "readback_fir": None,
            "readback_i2c": BASELINE | {"eq": eq},
            "requested_fir": None,
            "requested_i2c": {"eq": eq},
        },
        "stage": "I2C_SCREENING" if sequence else "BASELINE_MEASUREMENT",
        "valid": True,
    }


@pytest.fixture
def campaign_fixture(tmp_path) -> CampaignFixture:
    campaign_id = "desktop-test"
    fixture = CampaignFixture(root=tmp_path / campaign_id, campaign_id=campaign_id)
    fixture.controller.mkdir(parents=True)
    fixture.engine.mkdir(parents=True)
    config = fixture.root / "config"
    config.mkdir(parents=True)
    (config / "b0.yaml").write_text(
        yaml.safe_dump(
            {
                "baseline_id": "B0",
                "version": 2,
                "pattern": "SSPRQ",
                "fir": {
                    "taps": [0, 9, -28, 104, -8, 1, -3],
                    "upper_nlc": 6,
                    "lower_nlc": 6,
                },
                "i2c": BASELINE,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rounds = [
        _round(campaign_id, 1, "baseline-0", "round-1", 4.2, 3.6, 20),
        _round(campaign_id, 2, "round-1", "round-2", 3.6, 3.16, 23),
    ]
    for record in rounds:
        _write_json(fixture.round(int(record["round_index"])), record)

    for observation in (
        _observation(fixture, "baseline-0", 0, 4.2, 16),
        _observation(fixture, "round-1", 1, 3.6, 20),
        _observation(fixture, "round-2", 2, 3.16, 23),
    ):
        _write_json(
            fixture.engine / "observations" / f"{observation['observation_id']}.json",
            observation,
        )

    campaign = {
        "campaign_id": campaign_id,
        "execution_mode": "LIVE",
        "max_rounds": 2,
        "policy": {"version": "i2c-restricted-v1"},
        "provider": {"provider_id": "moonshot-kimi", "provider_version": "kimi-k3"},
        "target_tdecq_db": 2.8,
    }
    _write_json(fixture.controller / "campaign.json", campaign)
    state = {
        "campaign_id": campaign_id,
        "state": "COMPLETED",
        "execution_mode": "LIVE",
        "completed_rounds": 2,
        "target_tdecq_db": 2.8,
        "target_reached": False,
        "acceptance_passed": False,
        "closed_loop_proven": True,
        "stop_reason": "MAX_ROUNDS_REACHED",
        "provider_error_type": None,
        "provider_error_message": None,
        "started_at": "2026-08-18T00:00:00Z",
        "finished_at": "2026-08-18T00:03:00Z",
        "adapter_identities": {"dca": {"identity_verified": True}, "i2c": {"identity_verified": True}},
        "b0_restored": True,
        "evidence_complete": True,
        "round_records": rounds,
    }
    _write_json(fixture.controller / "state.json", state)

    events = [
        {
            "attempt": 1,
            "event": "RESPONSE_RECEIVED",
            "measurement_id": "baseline-0",
            "finish_reason": "stop",
            "raw_response_content": '{"action":"PROPOSE","requested_params":{"eq":20}}',
            "token_usage": {"total_tokens": 100},
        },
        {
            "attempt": 1,
            "event": "PARSE_RESULT",
            "measurement_id": "baseline-0",
            "parse_success": True,
        },
        {
            "attempt": 1,
            "event": "RESPONSE_RECEIVED",
            "measurement_id": "round-1",
            "finish_reason": "stop",
            "raw_response_content": '{"action":"PROPOSE","requested_params":{"eq":23}}',
            "token_usage": {"total_tokens": 110},
        },
    ]
    (fixture.controller / "kimi-response-attempts.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return fixture
