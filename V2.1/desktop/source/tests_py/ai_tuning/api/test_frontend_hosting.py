from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient

from vcsel_ai_tuning.api.app import create_app
from vcsel_ai_tuning.api.routes import CampaignService


class _StatusProvider:
    def health(self) -> dict[str, object]:
        return {"status": "ok", "runtime_mode": "FAKE"}

    def capabilities_payload(self) -> dict[str, object]:
        return {"runtime_mode": "FAKE", "read_only": False}


def _write_frontend_build(root: Path) -> Path:
    frontend_dist = root / "dist"
    assets = frontend_dist / "assets"
    assets.mkdir(parents=True)
    (frontend_dist / "index.html").write_text(
        "<!doctype html><title>VCSEL Demo</title><div id='root'></div>",
        encoding="utf-8",
    )
    (assets / "app.js").write_text(
        "window.__VCSEL_DEMO__ = true;",
        encoding="utf-8",
    )
    return frontend_dist


def test_fastapi_hosts_frontend_and_preserves_api_routes(tmp_path: Path) -> None:
    frontend_dist = _write_frontend_build(tmp_path)
    service = cast(CampaignService, cast(Any, object()))
    app = create_app(
        service,
        status_provider=_StatusProvider(),
        frontend_dist=frontend_dist,
    )

    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.headers["content-type"].startswith("text/html")
        assert "VCSEL Demo" in root.text

        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert asset.headers["content-type"].startswith("text/javascript")
        assert "__VCSEL_DEMO__" in asset.text

        browser_route = client.get("/campaigns/demo")
        assert browser_route.status_code == 200
        assert "VCSEL Demo" in browser_route.text

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "runtime_mode": "FAKE"}

        unknown_api = client.get("/api/not-a-route")
        assert unknown_api.status_code == 404
        assert unknown_api.headers["content-type"].startswith("application/json")
