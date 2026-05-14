import asyncio
import json
import logging
from datetime import datetime

from app.core.config import settings
from app.db.models import AnomalyEvent, SensorMetric
from app.db.session import SessionLocal
from app.websocket.manager import websocket_manager

logger = logging.getLogger(__name__)


class StreamService:
    def __init__(self) -> None:
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        if not settings.enable_kafka_consumers:
            logger.info("Kafka consumers disabled for local runtime.")
            return
        self._tasks = [
            asyncio.create_task(self._consume_metrics()),
            asyncio.create_task(self._consume_anomalies()),
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _consume_metrics(self) -> None:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            settings.kafka_metric_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id="backend-metric-consumer",
            auto_offset_reset="latest",
        )
        await consumer.start()
        try:
            async for msg in consumer:
                payload = json.loads(msg.value.decode("utf-8"))
                await self._persist_metric(payload)
                await websocket_manager.broadcast({"type": "metric", "data": payload})
        finally:
            await consumer.stop()

    async def _consume_anomalies(self) -> None:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            settings.kafka_anomaly_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id="backend-anomaly-consumer",
            auto_offset_reset="latest",
        )
        await consumer.start()
        try:
            async for msg in consumer:
                payload = json.loads(msg.value.decode("utf-8"))
                await self._persist_anomaly(payload)
                await websocket_manager.broadcast({"type": "anomaly", "data": payload})
        finally:
            await consumer.stop()

    async def _persist_metric(self, payload: dict) -> None:
        async with SessionLocal() as session:
            session.add(
                SensorMetric(
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    sensor_id=payload["sensor_id"],
                    temperature_k=payload["temperature_k"],
                    pressure_atm=payload["pressure_atm"],
                    vibration_mms=payload["vibration_mms"],
                    rolling_mean_16=payload.get("rolling_mean_16"),
                    rolling_std_16=payload.get("rolling_std_16"),
                    z_score_16=payload.get("z_score_16"),
                    momentum_8=payload.get("momentum_8"),
                    lag_1=payload.get("lag_1"),
                    volatility_16=payload.get("volatility_16"),
                    ema_12=payload.get("ema_12"),
                    event_source=payload.get("event_source", "stream"),
                )
            )
            await session.commit()

    async def _persist_anomaly(self, payload: dict) -> None:
        async with SessionLocal() as session:
            session.add(
                AnomalyEvent(
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    sensor_id=payload["sensor_id"],
                    model_name=payload["model_name"],
                    anomaly_score=payload["anomaly_score"],
                    threshold=payload.get("threshold"),
                    is_anomaly=payload["is_anomaly"],
                    explanation=payload.get("explanation"),
                )
            )
            await session.commit()


stream_service = StreamService()
