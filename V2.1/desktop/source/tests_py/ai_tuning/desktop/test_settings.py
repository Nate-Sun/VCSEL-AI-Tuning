import json

import pytest

from vcsel_ai_tuning.desktop.models import BenchSettings
from vcsel_ai_tuning.desktop.settings import (
    BenchSettingsStore,
    DesktopSettingsError,
)


BENCH = BenchSettings(
    dca_visa_resource="TCPIP0::scope::hislip0,4880::INSTR",
    i2c_com_port="COM7",
    i2c_address=0xA6,
)


def test_missing_settings_returns_none(tmp_path) -> None:
    assert BenchSettingsStore(tmp_path / "settings.json").load() is None


def test_settings_round_trip_contains_only_bench_endpoints(tmp_path) -> None:
    path = tmp_path / "nested" / "settings.json"
    store = BenchSettingsStore(path)

    store.save(BENCH)

    assert store.load() == BENCH
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == BENCH.model_dump(mode="json")
    text = path.read_text(encoding="utf-8")
    assert "api" not in text.lower()
    assert "moonshot" not in text.lower()


@pytest.mark.parametrize("content", ["not-json", "[]", '{"i2c_com_port":"COM7"}'])
def test_corrupt_or_incomplete_settings_fail_closed(tmp_path, content: str) -> None:
    path = tmp_path / "settings.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DesktopSettingsError, match="settings"):
        BenchSettingsStore(path).load()


def test_save_rejects_anything_except_bench_settings(tmp_path) -> None:
    store = BenchSettingsStore(tmp_path / "settings.json")

    with pytest.raises(TypeError, match="BenchSettings"):
        store.save({"api_key": "secret"})  # type: ignore[arg-type]

    assert not (tmp_path / "settings.json").exists()
    assert not (tmp_path / "settings.tmp").exists()
