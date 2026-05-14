from fastapi import APIRouter, Depends

from app.dependencies import get_timeseries_service
from app.schemas.timeseries import (
    AnomalyEventResponse,
    ExperimentSummaryResponse,
    MetricWindowResponse,
)
from app.services.timeseries_service import TimeseriesService

router = APIRouter()


@router.get("/metrics", response_model=MetricWindowResponse)
async def get_metrics(
    limit: int = 240,
    sensor_id: str | None = None,
    service: TimeseriesService = Depends(get_timeseries_service),
) -> MetricWindowResponse:
    return await service.get_metrics(limit=limit, sensor_id=sensor_id)


@router.get("/anomalies", response_model=list[AnomalyEventResponse])
async def get_anomalies(
    limit: int = 100,
    sensor_id: str | None = None,
    service: TimeseriesService = Depends(get_timeseries_service),
) -> list[AnomalyEventResponse]:
    return await service.get_anomalies(limit=limit, sensor_id=sensor_id)


@router.get("/experiments", response_model=list[ExperimentSummaryResponse])
async def get_experiments(
    limit: int = 50,
    service: TimeseriesService = Depends(get_timeseries_service),
) -> list[ExperimentSummaryResponse]:
    return await service.get_experiments(limit=limit)
