from fastapi import APIRouter, Depends

from app.dependencies import get_timeseries_service
from app.schemas.timeseries import StreamSnapshotResponse
from app.services.timeseries_service import TimeseriesService

router = APIRouter()


@router.get("/stream/live", response_model=StreamSnapshotResponse)
async def get_live_stream(
    limit: int = 120,
    service: TimeseriesService = Depends(get_timeseries_service),
) -> StreamSnapshotResponse:
    return await service.get_stream_snapshot(limit=limit)
