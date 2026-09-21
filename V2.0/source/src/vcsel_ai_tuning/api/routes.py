"""Thin HTTP routes over an injected campaign application service."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import APIRouter, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from vcsel_ai_tuning.campaign.models import (
    AuthorizationMode,
    CampaignState,
    StageAuthorization,
)
from vcsel_ai_tuning.campaign.store import UnknownCampaignError

from .schemas import (
    CreateCampaignRequest,
    StageApprovalRequest,
    StagePlanRequest,
)


@runtime_checkable
class CampaignService(Protocol):
    def create_campaign(self, request: CreateCampaignRequest) -> Any: ...

    def get_campaign(self, campaign_id: str) -> Any: ...

    def preflight(self, campaign_id: str) -> Any: ...

    def plan_stage(self, campaign_id: str, request: StagePlanRequest) -> Any: ...

    def approve_stage(
        self,
        campaign_id: str,
        authorization: StageAuthorization,
    ) -> Any: ...

    def pause_campaign(self, campaign_id: str) -> Any: ...

    def resume_campaign(self, campaign_id: str) -> Any: ...

    def abort_campaign(self, campaign_id: str) -> Any: ...

    def rollback_campaign(self, campaign_id: str) -> Any: ...

    def list_observations(self, campaign_id: str) -> Any: ...

    def get_best_observation(self, campaign_id: str) -> Any: ...

    def get_report(self, campaign_id: str) -> Any: ...


@runtime_checkable
class CampaignEventSource(Protocol):
    def events(
        self,
        campaign_id: str,
        *,
        after_sequence: int = 0,
    ) -> Iterable[Any]: ...

    def current_state(self, campaign_id: str) -> CampaignState | None: ...


def create_router(
    service: CampaignService,
    event_source: CampaignEventSource | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])

    @router.post("", status_code=status.HTTP_201_CREATED)
    def create_campaign(request: CreateCampaignRequest) -> Any:
        return service.create_campaign(request)

    @router.get("")
    def list_campaigns(
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Any:
        list_method = getattr(service, "list_campaigns", None)
        if not callable(list_method):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "campaign_list_unavailable",
                    "message": "campaign listing is not configured",
                },
            )
        return list_method(limit=limit, offset=offset)

    @router.get("/{campaign_id}")
    def get_campaign(campaign_id: str) -> Any:
        return service.get_campaign(campaign_id)

    @router.post("/{campaign_id}/preflight")
    def preflight(campaign_id: str) -> Any:
        return service.preflight(campaign_id)

    @router.post("/{campaign_id}/stages/plan")
    def plan_stage(campaign_id: str, request: StagePlanRequest) -> Any:
        return service.plan_stage(campaign_id, request)

    @router.post("/{campaign_id}/stages/approve")
    def approve_stage(campaign_id: str, request: StageApprovalRequest) -> Any:
        authorization = request.authorization
        if authorization.mode != AuthorizationMode.PER_STAGE:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "authorization_error",
                    "message": (
                        f"{authorization.mode.value} authorization is disabled; "
                        "V1 requires PER_STAGE"
                    ),
                },
            )
        return service.approve_stage(campaign_id, authorization)

    @router.post("/{campaign_id}/pause")
    def pause_campaign(campaign_id: str) -> Any:
        return service.pause_campaign(campaign_id)

    @router.post("/{campaign_id}/resume")
    def resume_campaign(campaign_id: str) -> Any:
        return service.resume_campaign(campaign_id)

    @router.post("/{campaign_id}/abort")
    def abort_campaign(campaign_id: str) -> Any:
        return service.abort_campaign(campaign_id)

    @router.post("/{campaign_id}/rollback")
    def rollback_campaign(campaign_id: str) -> Any:
        return service.rollback_campaign(campaign_id)

    @router.get("/{campaign_id}/observations")
    def list_observations(campaign_id: str) -> Any:
        return service.list_observations(campaign_id)

    @router.get("/{campaign_id}/best")
    def get_best_observation(campaign_id: str) -> Any:
        return service.get_best_observation(campaign_id)

    @router.get("/{campaign_id}/report")
    def get_report(campaign_id: str) -> Any:
        return service.get_report(campaign_id)

    @router.get("/{campaign_id}/events/stream")
    def stream_events(
        campaign_id: str,
        last_event_id: Annotated[
            str | None,
            Header(alias="Last-Event-ID"),
        ] = None,
    ) -> StreamingResponse:
        if event_source is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "event_stream_unavailable",
                    "message": "campaign event source is not configured",
                },
            )
        from .events import (
            InvalidLastEventId,
            parse_last_event_id,
            stream_campaign_events,
        )

        try:
            parsed_last_event_id = parse_last_event_id(last_event_id)
        except InvalidLastEventId as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_last_event_id",
                    "message": str(error),
                },
            ) from error
        if event_source.current_state(campaign_id) is None:
            raise UnknownCampaignError(f"unknown campaign: {campaign_id}")
        frames = stream_campaign_events(
            event_source,
            campaign_id,
            last_event_id=str(parsed_last_event_id),
        )
        return StreamingResponse(frames, media_type="text/event-stream")

    return router
