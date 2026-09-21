import json

import pytest

from vcsel_ai_tuning.desktop.artifacts import (
    CampaignArtifactReader,
    DesktopArtifactError,
)


def test_reader_orders_b0_and_rounds_and_resolves_eye_evidence(
    campaign_fixture,
) -> None:
    result = CampaignArtifactReader(campaign_fixture.root).read()

    assert [point.round_index for point in result.points] == [0, 1, 2]
    assert [point.row_type for point in result.points] == ["B0", "ROUND", "ROUND"]
    assert result.points[1].eye_path.name == "eye-1.png"
    assert result.best_point.round_index == 2
    assert result.last_point.round_index == 2
    assert result.b0_point.readback_params["eq"] == 16
    assert result.b0_restored is True


def test_reader_associates_kimi_events_with_resulting_round(campaign_fixture) -> None:
    result = CampaignArtifactReader(campaign_fixture.root).read()

    assert len(result.points[1].kimi_events) == 2
    assert result.points[1].kimi_events[0]["measurement_id"] == "baseline-0"
    assert result.points[2].kimi_events[0]["measurement_id"] == "round-1"


def test_corrupt_round_is_reported_not_silently_skipped(campaign_fixture) -> None:
    campaign_fixture.round(2).write_text("bad-json", encoding="utf-8")

    with pytest.raises(DesktopArtifactError, match="round 2"):
        CampaignArtifactReader(campaign_fixture.root).read()


def test_round_file_must_match_strict_state_record(campaign_fixture) -> None:
    payload = json.loads(campaign_fixture.round(2).read_text(encoding="utf-8"))
    payload["after_metrics"]["tdecq_db"] = 9.9
    campaign_fixture.round(2).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DesktopArtifactError, match="state"):
        CampaignArtifactReader(campaign_fixture.root).read()


def test_evidence_path_must_be_persisted_safe_and_hash_verified(
    campaign_fixture,
) -> None:
    observation = campaign_fixture.engine / "observations" / "round-1.json"
    payload = json.loads(observation.read_text(encoding="utf-8"))
    payload["evidence"]["png_path"] = "../invented.png"
    observation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DesktopArtifactError, match="eye evidence"):
        CampaignArtifactReader(campaign_fixture.root).read()


def test_missing_kimi_log_is_allowed_but_corrupt_log_is_not(campaign_fixture) -> None:
    log = campaign_fixture.controller / "kimi-response-attempts.jsonl"
    log.unlink()
    result = CampaignArtifactReader(campaign_fixture.root).read()
    assert all(not point.kimi_events for point in result.points)

    log.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(DesktopArtifactError, match="Kimi"):
        CampaignArtifactReader(campaign_fixture.root).read()


def test_unreadable_kimi_log_reports_artifact_error(campaign_fixture) -> None:
    log = campaign_fixture.controller / "kimi-response-attempts.jsonl"
    log.unlink()
    log.mkdir()

    with pytest.raises(DesktopArtifactError, match="Kimi"):
        CampaignArtifactReader(campaign_fixture.root).read()
