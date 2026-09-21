import csv
import json

from vcsel_ai_tuning.desktop.artifacts import CampaignArtifactReader
from vcsel_ai_tuning.desktop.csv_export import export_campaign_csv


def test_csv_contains_b0_and_one_row_per_measured_round(
    campaign_fixture, tmp_path
) -> None:
    result = CampaignArtifactReader(campaign_fixture.root).read()
    output = export_campaign_csv(result, tmp_path / "nested" / "results.csv")

    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8-sig")))

    assert [row["row_type"] for row in rows] == ["B0", "ROUND", "ROUND"]
    assert rows[0]["readback_eq"] == "16"
    assert rows[2]["requested_eq"] == "23"
    assert rows[2]["readback_eq"] == "23"
    assert rows[2]["tdecq_db"] == "3.16"
    assert rows[2]["is_best"] == "true"
    assert json.loads(rows[2]["kimi_events_json"])[0]["measurement_id"] == "round-1"


def test_csv_is_excel_friendly_atomic_and_contains_lineage(
    campaign_fixture, tmp_path
) -> None:
    result = CampaignArtifactReader(campaign_fixture.root).read()
    output = export_campaign_csv(result, tmp_path / "结果.csv")

    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    text = output.read_text(encoding="utf-8-sig")
    assert "4085b007546f820820dbdb7f7f0bba6ebc4b2049" in text
    assert "v2.1-continuous-eye-b-validated-20260817" in text
    assert not output.with_suffix(".tmp").exists()
