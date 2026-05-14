import asyncio
import math
import random
from collections import deque
from datetime import datetime, timezone

from app.core.config import settings
from app.db.models import AnomalyEvent, SensorMetric
from app.db.session import SessionLocal
from app.websocket.manager import websocket_manager


class DemoRuntimeService:
    """Generates local live telemetry when Kafka is unavailable."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._history: deque[dict] = deque(maxlen=64)
        self._step = 0

    async def start(self) -> None:
        if settings.enable_kafka_consumers or not settings.enable_demo_stream:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while True:
            metric = self._next_metric()
            await self._persist_metric(metric)
            await websocket_manager.broadcast({"type": "metric", "data": metric})

            anomaly = self._maybe_create_anomaly(metric)
            if anomaly is not None:
                await self._persist_anomaly(anomaly)
                await websocket_manager.broadcast({"type": "anomaly", "data": anomaly})

            await asyncio.sleep(settings.demo_tick_seconds)

    def _next_metric(self) -> dict:
        self._step += 1
        timestamp = datetime.now(timezone.utc)

        temperature = 4.2 + math.sin(self._step / 5.0) * 0.08 + random.gauss(0.0, 0.015)
        pressure = 1.0 + math.cos(self._step / 7.0) * 0.02 + random.gauss(0.0, 0.004)
        vibration = 0.052 + math.sin(self._step / 3.5) * 0.01 + random.gauss(0.0, 0.002)

        if self._step % 15 == 0:
            temperature += 0.22
            vibration += 0.03

        record = {
            "timestamp": timestamp.isoformat(),
            "sensor_id": settings.demo_sensor_id,
            "temperature_k": round(temperature, 5),
            "pressure_atm": round(pressure, 5),
            "vibration_mms": round(vibration, 5),
        }
        self._history.append(record)
        return record | self._engineer_features()

    def _engineer_features(self) -> dict:
        temps = [item["temperature_k"] for item in self._history]
        vibs = [item["vibration_mms"] for item in self._history]
        latest = temps[-1]

        window = temps[-16:]
        vib_window = vibs[-16:]
        rolling_mean = sum(window) / len(window)
        rolling_std = (sum((value - rolling_mean) ** 2 for value in window) / max(len(window), 1)) ** 0.5
        z_score = (latest - rolling_mean) / rolling_std if rolling_std > 1e-8 else 0.0
        momentum = latest - temps[-8] if len(temps) >= 8 else 0.0
        lag_1 = temps[-2] if len(temps) > 1 else latest
        vib_mean = sum(vib_window) / len(vib_window)
        volatility = (sum((value - vib_mean) ** 2 for value in vib_window) / max(len(vib_window), 1)) ** 0.5

        alpha = 2 / 13
        ema = temps[0]
        for value in temps[1:]:
            ema = alpha * value + (1 - alpha) * ema

        return {
            "rolling_mean_16": round(rolling_mean, 5),
            "rolling_std_16": round(rolling_std, 5),
            "z_score_16": round(z_score, 5),
            "momentum_8": round(momentum, 5),
            "lag_1": round(lag_1, 5),
            "volatility_16": round(volatility, 5),
            "ema_12": round(ema, 5),
            "event_source": "local-demo",
        }

    def _maybe_create_anomaly(self, metric: dict) -> dict | None:
        score = abs(metric["z_score_16"]) + (metric["volatility_16"] * 10.0)
        if score <= 2.4 and self._step % 15 != 0:
            return None
        return {
            "timestamp": metric["timestamp"],
            "sensor_id": metric["sensor_id"],
            "model_name": "local_demo_detector",
            "anomaly_score": round(score, 5),
            "threshold": 2.4,
            "is_anomaly": True,
            "explanation": {
                "z_score_16": metric["z_score_16"],
                "volatility_16": metric["volatility_16"],
            },
        }

    async def _persist_metric(self, metric: dict) -> None:
        async with SessionLocal() as session:
            session.add(
                SensorMetric(
                    timestamp=datetime.fromisoformat(metric["timestamp"]),
                    sensor_id=metric["sensor_id"],
                    temperature_k=metric["temperature_k"],
                    pressure_atm=metric["pressure_atm"],
                    vibration_mms=metric["vibration_mms"],
                    rolling_mean_16=metric["rolling_mean_16"],
                    rolling_std_16=metric["rolling_std_16"],
                    z_score_16=metric["z_score_16"],
                    momentum_8=metric["momentum_8"],
                    lag_1=metric["lag_1"],
                    volatility_16=metric["volatility_16"],
                    ema_12=metric["ema_12"],
                    event_source=metric["event_source"],
                )
            )
            await session.commit()

    async def _persist_anomaly(self, anomaly: dict) -> None:
        async with SessionLocal() as session:
            session.add(
                AnomalyEvent(
                    timestamp=datetime.fromisoformat(anomaly["timestamp"]),
                    sensor_id=anomaly["sensor_id"],
                    model_name=anomaly["model_name"],
                    anomaly_score=anomaly["anomaly_score"],
                    threshold=anomaly["threshold"],
                    is_anomaly=anomaly["is_anomaly"],
                    explanation=anomaly["explanation"],
                )
            )
            await session.commit()


demo_runtime_service = DemoRuntimeService()
