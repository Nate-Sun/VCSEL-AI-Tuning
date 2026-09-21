"""FastAPI application factory for injected campaign services."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
from typing import Any, NoReturn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from vcsel_ai_tuning.campaign.authorization import AuthorizationError
from vcsel_ai_tuning.campaign.orchestrator import OrchestratorError
from vcsel_ai_tuning.campaign.state_machine import IllegalTransition
from vcsel_ai_tuning.campaign.store import UnknownCampaignError
from vcsel_ai_tuning.runtime import RuntimeAssemblyError
from vcsel_ai_tuning.v1_1.authorization import AuthorizationDriftError
from vcsel_ai_tuning.v1_1.hardware_control import (
    DcaConfigurationDrift,
    HardwareTransactionError,
)
from vcsel_ai_tuning.v1_1.regmap import ParameterSelectionError
from vcsel_ai_tuning.v1_1.repository import (
    ActiveTaskConflict,
    TaskConcurrencyError,
    TaskStoreError,
    UnknownTaskError,
)
from vcsel_ai_tuning.v1_1.sweep import SweepLimitExceeded

from .routes import CampaignEventSource, CampaignService, create_router
from .health import RuntimeStatusProvider, create_health_router
from .v1_1_routes import (
    V11CampaignService,
    V11TaskEventSource,
    create_v11_router,
)


@dataclass(frozen=True, slots=True)
class ApiRuntime:
    service: CampaignService
    event_source: CampaignEventSource
    status_provider: RuntimeStatusProvider | None = None
    v11_service: V11CampaignService | None = None
    v11_event_source: V11TaskEventSource | None = None
    shutdown: Callable[[], None] | None = None


def _resolve_frontend_dist(configured: str | Path | None) -> Path:
    if configured is not None:
        return Path(configured).expanduser().resolve()
    override = os.environ.get("VCSEL_AI_FRONTEND_DIST")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _include_frontend_routes(
    app: FastAPI,
    configured_dist: str | Path | None,
) -> None:
    frontend_dist = _resolve_frontend_dist(configured_dist)
    index = frontend_dist / "index.html"
    if not index.is_file():
        return
    frontend_root = frontend_dist.resolve()

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str) -> FileResponse:
        if frontend_path == "api" or frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        requested = (frontend_root / frontend_path).resolve()
        try:
            requested.relative_to(frontend_root)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Not Found") from error
        if frontend_path and requested.is_file():
            return FileResponse(requested)
        if frontend_path.startswith("assets/"):
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(index)


def _domain_error(
    status_code: int,
    code: str,
) -> Callable[[Request, Exception], Any]:
    async def handle(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": {
                    "code": code,
                    "message": str(error),
                }
            },
        )

    return handle


def create_app(
    service: CampaignService,
    event_source: CampaignEventSource | None = None,
    *,
    status_provider: RuntimeStatusProvider | None = None,
    v11_service: V11CampaignService | None = None,
    v11_event_source: V11TaskEventSource | None = None,
    shutdown: Callable[[], None] | None = None,
    frontend_dist: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if shutdown is not None:
                shutdown()

    app = FastAPI(
        title="VCSEL AI Tuning API",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.add_exception_handler(
        UnknownCampaignError,
        _domain_error(404, "campaign_not_found"),
    )
    app.add_exception_handler(
        AuthorizationError,
        _domain_error(422, "authorization_error"),
    )
    app.add_exception_handler(
        OrchestratorError,
        _domain_error(409, "campaign_conflict"),
    )
    app.add_exception_handler(
        IllegalTransition,
        _domain_error(409, "campaign_conflict"),
    )

    async def runtime_error(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": getattr(error, "code", "RUNTIME_ASSEMBLY_ERROR"),
                    "message": str(error),
                }
            },
        )

    app.add_exception_handler(RuntimeAssemblyError, runtime_error)

    async def active_task_conflict(
        _request: Request, error: ActiveTaskConflict
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": "ACTIVE_HARDWARE_TASK",
                    "message": str(error),
                    "active_task_id": error.active_task_id,
                }
            },
        )

    app.add_exception_handler(ActiveTaskConflict, active_task_conflict)
    app.add_exception_handler(
        UnknownTaskError,
        _domain_error(404, "TASK_NOT_FOUND"),
    )
    app.add_exception_handler(
        AuthorizationDriftError,
        _domain_error(409, "AUTHORIZATION_DRIFT"),
    )
    app.add_exception_handler(
        TaskStoreError,
        _domain_error(409, "TASK_STATE_CONFLICT"),
    )
    app.add_exception_handler(
        DcaConfigurationDrift,
        _domain_error(409, "DCA_CONFIGURATION_DRIFT"),
    )
    for error_type in (
        TaskConcurrencyError,
        HardwareTransactionError,
    ):
        app.add_exception_handler(
            error_type,
            _domain_error(409, error_type.__name__.upper()),
        )
    for error_type in (ParameterSelectionError, SweepLimitExceeded):
        app.add_exception_handler(
            error_type,
            _domain_error(422, error_type.__name__.upper()),
        )
    app.include_router(create_router(service, event_source))
    app.include_router(create_health_router(status_provider))
    if v11_service is not None:
        app.include_router(create_v11_router(v11_service, v11_event_source))
    _include_frontend_routes(app, frontend_dist)
    return app


def create_app_from_runtime_factory(
    factory: Callable[[], ApiRuntime],
) -> FastAPI:
    if not callable(factory):
        raise TypeError("API runtime factory must be callable")
    runtime = factory()
    if not isinstance(runtime, ApiRuntime):
        raise TypeError("API runtime factory must return ApiRuntime")
    if not isinstance(runtime.service, CampaignService):
        raise TypeError("ApiRuntime.service must implement CampaignService")
    if not isinstance(runtime.event_source, CampaignEventSource):
        raise TypeError("ApiRuntime.event_source must implement CampaignEventSource")
    return create_app(
        runtime.service,
        event_source=runtime.event_source,
        status_provider=runtime.status_provider,
        v11_service=runtime.v11_service,
        v11_event_source=runtime.v11_event_source,
        shutdown=runtime.shutdown,
    )


def _load_factory(path: str) -> Callable[[], ApiRuntime]:
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "VCSEL_AI_CAMPAIGN_SERVICE_FACTORY must use module:callable syntax"
        )
    factory = getattr(import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError("configured campaign service factory is not callable")
    return factory


def main() -> NoReturn:
    factory_path = os.environ.get("VCSEL_AI_CAMPAIGN_SERVICE_FACTORY")
    runtime_config_path = os.environ.get("VCSEL_AI_RUNTIME_CONFIG")
    host = os.environ.get("VCSEL_AI_API_HOST", "127.0.0.1")
    port = int(os.environ.get("VCSEL_AI_API_PORT", "8000"))
    if factory_path:
        app = create_app_from_runtime_factory(_load_factory(factory_path))
    elif runtime_config_path:
        from vcsel_ai_tuning.runtime import create_runtime
        from vcsel_ai_tuning.runtime_config import load_runtime_config

        config = load_runtime_config(runtime_config_path)
        api_runtime = create_runtime(config)
        app = create_app_from_runtime_factory(lambda: api_runtime)
        host = config.api.host
        port = config.api.port
    else:
        from vcsel_ai_tuning.runtime import create_fake_runtime

        api_runtime = create_fake_runtime()
        app = create_app_from_runtime_factory(lambda: api_runtime)
    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
    )
    raise SystemExit(0)
