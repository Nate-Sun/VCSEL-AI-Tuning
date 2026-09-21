"""Read-only runtime health and capability endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from fastapi import APIRouter, HTTPException, status


@runtime_checkable
class RuntimeStatusProvider(Protocol):
    def health(self) -> Mapping[str, Any]: ...

    def capabilities_payload(self) -> Mapping[str, Any]: ...


def create_health_router(
    provider: RuntimeStatusProvider | None,
) -> APIRouter:
    router = APIRouter(tags=["runtime"])

    def require_provider() -> RuntimeStatusProvider:
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "runtime_status_unavailable",
                    "message": "runtime status provider is not configured",
                },
            )
        return provider

    @router.get("/health")
    def health() -> Mapping[str, Any]:
        return require_provider().health()

    @router.get("/capabilities")
    def capabilities() -> Mapping[str, Any]:
        return require_provider().capabilities_payload()

    return router
