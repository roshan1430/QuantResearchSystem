from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AnomalyEvent, ExperimentRun, SensorMetric
from app.db.session import SessionLocal
from app.schemas.timeseries import (
    AnomalyEventResponse,
    ExperimentSummaryResponse,
    MetricPointResponse,
    MetricWindowResponse,
    StreamSnapshotResponse,
)


class TimeseriesService:
    async def get_metrics(self, limit: int, sensor_id: str | None = None) -> MetricWindowResponse:
        async with SessionLocal() as session:
            statement = select(SensorMetric).order_by(desc(SensorMetric.timestamp)).limit(limit)
            if sensor_id:
                statement = statement.where(SensorMetric.sensor_id == sensor_id)
            result = await session.execute(statement)
            rows = list(reversed(result.scalars().all()))
        points = [MetricPointResponse.model_validate(row, from_attributes=True) for row in rows]
        return MetricWindowResponse(points=points, count=len(points))

    async def get_anomalies(self, limit: int, sensor_id: str | None = None) -> list[AnomalyEventResponse]:
        async with SessionLocal() as session:
            statement = select(AnomalyEvent).order_by(desc(AnomalyEvent.timestamp)).limit(limit)
            if sensor_id:
                statement = statement.where(AnomalyEvent.sensor_id == sensor_id)
            result = await session.execute(statement)
            rows = result.scalars().all()
        return [AnomalyEventResponse.model_validate(row, from_attributes=True) for row in rows]

    async def get_experiments(self, limit: int) -> list[ExperimentSummaryResponse]:
        async with SessionLocal() as session:
            statement = select(ExperimentRun).order_by(desc(ExperimentRun.created_at)).limit(limit)
            result = await session.execute(statement)
            rows = result.scalars().all()
        return [ExperimentSummaryResponse.model_validate(row, from_attributes=True) for row in rows]

    async def get_stream_snapshot(self, limit: int) -> StreamSnapshotResponse:
        metric_window = await self.get_metrics(limit=limit)
        anomaly_window = await self.get_anomalies(limit=max(20, limit // 4))
        return StreamSnapshotResponse(metrics=metric_window.points, anomalies=anomaly_window)


timeseries_service = TimeseriesService()
