from fastapi import APIRouter

from app.schemas.health import HealthResponse, ReadinessResponse
from app.services.platform_service import platform_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    return await platform_service.get_health()


@router.get("/system/ready", response_model=ReadinessResponse)
async def get_readiness() -> ReadinessResponse:
    return await platform_service.get_readiness()
