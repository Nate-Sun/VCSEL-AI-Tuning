"""Thin V1.1 HTTP routes over an injected campaign application service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from vcsel_ai_tuning.v1_1.domain import LifecycleState
from vcsel_ai_tuning.v1_1.events import (
    InvalidLastEventId,
    parse_last_event_id,
    stream_task_events,
)
from vcsel_ai_tuning.v1_1.repository import TaskEvent

from .v1_1_schemas import (
    AbortRequest,
    ApplyHistoryResultRequest,
    AuthorizeTaskRequest,
    CreateTaskRequest,
    HistoryRecoveryRequest,
    ParameterSelectionTuple,
    RecoveryRequest,
    SaveB0Request,
    SweepCountRequest,
)


@runtime_checkable
class V11CampaignService(Protocol):
    def runtime_health(self) -> Any: ...

    def runtime_capabilities(self) -> Any: ...

    def create_task(self, request: CreateTaskRequest) -> Any: ...

    def active_task(self) -> Any: ...

    def hardware_snapshot(self) -> Any: ...

    def scan_parameter_mapping(self) -> Any: ...

    def validate_parameters(
        self, parameters: ParameterSelectionTuple
    ) -> Any: ...

    def save_b0(self, task_id: str, request: SaveB0Request) -> Any: ...

    def count_sweep(self, request: SweepCountRequest) -> Any: ...

    def authorize_task(
        self, task_id: str, request: AuthorizeTaskRequest
    ) -> Any: ...

    def start_task(self, task_id: str) -> Any: ...

    def pause_task(self, task_id: str) -> Any: ...

    def resume_task(self, task_id: str) -> Any: ...

    def abort_task(self, task_id: str, request: AbortRequest) -> Any: ...

    def recover_task(self, task_id: str, request: RecoveryRequest) -> Any: ...

    def get_task(self, task_id: str) -> Any: ...

    def observations(self, task_id: str) -> Any: ...

    def best_result(self, task_id: str) -> Any: ...

    def sweep_points(self, task_id: str, *, offset: int, limit: int) -> Any: ...

    def final_report(self, task_id: str) -> Any: ...

    def sweep_json_path(self, task_id: str) -> Path: ...

    def sweep_csv_path(self, task_id: str) -> Path: ...

    def apply_history_result(
        self, request: ApplyHistoryResultRequest
    ) -> Any: ...

    def recover_history_result(
        self, recovery_id: str, request: HistoryRecoveryRequest
    ) -> Any: ...


@runtime_checkable
class V11TaskEventSource(Protocol):
    def events(
        self, task_id: str, *, after_sequence: int = 0
    ) -> Sequence[TaskEvent]: ...

    def current_state(self, task_id: str) -> LifecycleState | None: ...


def create_v11_router(
    service: V11CampaignService,
    event_source: V11TaskEventSource | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1.1", tags=["AI Tuning V1.1"])

    @router.get("/runtime/health")
    def runtime_health() -> Any:
        return service.runtime_health()

    @router.get("/runtime/capabilities")
    def runtime_capabilities() -> Any:
        return service.runtime_capabilities()

    @router.post("/tasks", status_code=status.HTTP_201_CREATED)
    def create_task(request: CreateTaskRequest) -> Any:
        return service.create_task(request)

    @router.get("/tasks/active")
    def active_task() -> Any:
        return service.active_task()

    @router.get("/hardware/snapshot")
    def hardware_snapshot() -> Any:
        return service.hardware_snapshot()

    @router.post("/mapping/scan")
    def scan_parameter_mapping() -> Any:
        return service.scan_parameter_mapping()

    @router.post("/parameters/validate")
    def validate_parameters(
        parameters: ParameterSelectionTuple,
    ) -> Any:
        return service.validate_parameters(parameters)

    @router.post("/sweeps/count")
    def count_sweep(request: SweepCountRequest) -> Any:
        return service.count_sweep(request)

    @router.post("/tasks/{task_id}/b0")
    def save_b0(task_id: str, request: SaveB0Request) -> Any:
        return service.save_b0(task_id, request)

    @router.post("/tasks/{task_id}/authorization")
    def authorize_task(
        task_id: str, request: AuthorizeTaskRequest
    ) -> Any:
        return service.authorize_task(task_id, request)

    @router.post("/tasks/{task_id}/start")
    def start_task(task_id: str) -> Any:
        return service.start_task(task_id)

    @router.post("/tasks/{task_id}/pause")
    def pause_task(task_id: str) -> Any:
        return service.pause_task(task_id)

    @router.post("/tasks/{task_id}/resume")
    def resume_task(task_id: str) -> Any:
        return service.resume_task(task_id)

    @router.post("/tasks/{task_id}/abort")
    def abort_task(task_id: str, request: AbortRequest) -> Any:
        return service.abort_task(task_id, request)

    @router.post("/tasks/{task_id}/recovery")
    def recover_task(task_id: str, request: RecoveryRequest) -> Any:
        return service.recover_task(task_id, request)

    @router.get("/tasks/{task_id}")
    def get_task(task_id: str) -> Any:
        return service.get_task(task_id)

    @router.get("/tasks/{task_id}/events")
    def events(
        task_id: str,
        after_sequence: Annotated[int, Query(ge=0)] = 0,
    ) -> Any:
        if event_source is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "event_source_unavailable"},
            )
        return event_source.events(task_id, after_sequence=after_sequence)

    @router.get("/tasks/{task_id}/events/stream")
    def event_stream(
        task_id: str,
        last_event_id: Annotated[
            str | None, Header(alias="Last-Event-ID")
        ] = None,
    ) -> StreamingResponse:
        if event_source is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "event_source_unavailable"},
            )
        try:
            parsed = parse_last_event_id(last_event_id)
        except InvalidLastEventId as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_last_event_id",
                    "message": str(error),
                },
            ) from error
        if event_source.current_state(task_id) is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "task_not_found", "task_id": task_id},
            )
        frames = stream_task_events(
            event_source,
            task_id,
            last_event_id=str(parsed),
        )
        return StreamingResponse(frames, media_type="text/event-stream")

    @router.get("/tasks/{task_id}/observations")
    def observations(task_id: str) -> Any:
        return service.observations(task_id)

    @router.get("/tasks/{task_id}/best")
    def best_result(task_id: str) -> Any:
        return service.best_result(task_id)

    @router.get("/tasks/{task_id}/sweep-points")
    def sweep_points(
        task_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> Any:
        return service.sweep_points(task_id, offset=offset, limit=limit)

    @router.get("/tasks/{task_id}/sweep-results.json")
    def sweep_json(task_id: str) -> FileResponse:
        return FileResponse(
            service.sweep_json_path(task_id),
            media_type="application/json",
            filename=f"{task_id}-sweep-results.json",
        )

    @router.get("/tasks/{task_id}/sweep-results.csv")
    def sweep_csv(task_id: str) -> FileResponse:
        return FileResponse(
            service.sweep_csv_path(task_id),
            media_type="text/csv",
            filename=f"{task_id}-sweep-results.csv",
        )

    @router.get("/tasks/{task_id}/final-report")
    def final_report(task_id: str) -> Any:
        return service.final_report(task_id)

    @router.post("/history/apply")
    def apply_history_result(request: ApplyHistoryResultRequest) -> Any:
        return service.apply_history_result(request)

    @router.post("/recovery/history/{recovery_id}")
    def recover_history_result(
        recovery_id: str, request: HistoryRecoveryRequest
    ) -> Any:
        return service.recover_history_result(recovery_id, request)

    return router
