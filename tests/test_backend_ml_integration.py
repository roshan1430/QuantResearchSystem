from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml-engine"))

from app.api.router import api_router  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_db_session  # noqa: E402
from app.dependencies import get_ml_service  # noqa: E402
from app.schemas.ml import TrainRequest  # noqa: E402
from app.services.ml_service import MLService  # noqa: E402
import app.services.timeseries_service as timeseries_service_module  # noqa: E402
from engine.main import app as ml_app  # noqa: E402


class LocalMLService(MLService):
    async def _post(self, path: str, payload: dict) -> dict:
        transport = httpx.ASGITransport(app=ml_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ml-engine.test") as client:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return response.json()

    async def _get(self, path: str) -> dict:
        transport = httpx.ASGITransport(app=ml_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ml-engine.test") as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.json()


class BackendMLIntegrationTests(unittest.TestCase):
    def test_backend_training_creates_experiment_and_trained_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset = temp_root / "sample_timeseries.csv"
            shutil.copyfile(ROOT / "datasets" / "sample_timeseries.csv", dataset)

            database_path = temp_root / "integration.db"
            engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}", future=True, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

            async def override_get_db_session():
                async with session_factory() as session:
                    yield session

            async def initialize_temp_db() -> None:
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)

            original_session_local = timeseries_service_module.SessionLocal
            timeseries_service_module.SessionLocal = session_factory

            test_backend = FastAPI()
            test_backend.include_router(api_router)
            test_backend.dependency_overrides[get_ml_service] = lambda: LocalMLService()
            test_backend.dependency_overrides[get_db_session] = override_get_db_session

            previous_cwd = Path.cwd()
            os.chdir(temp_root)
            try:
                asyncio.run(initialize_temp_db())
                with TestClient(test_backend) as client:
                    train_response = client.post(
                        "/train",
                        json=TrainRequest(
                            task_type="forecasting",
                            model_name="trained_forecast_regressor",
                            dataset_path=str(dataset),
                            target_column="target_temperature",
                            parameters={"source": "integration-test"},
                        ).model_dump(mode="json"),
                    )
                    self.assertEqual(train_response.status_code, 200, msg=train_response.text)
                    train_payload = train_response.json()
                    self.assertTrue(train_payload["accepted"])
                    self.assertTrue(train_payload["artifact_uri"])

                    models_response = client.get("/models")
                    self.assertEqual(models_response.status_code, 200)
                    model_payload = models_response.json()
                    trained_forecaster = next(
                        item for item in model_payload["models"] if item["model_name"] == "trained_forecast_regressor"
                    )
                    self.assertEqual(trained_forecaster["status"], "trained")

                    base_time = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
                    context = []
                    for index in range(18):
                        temperature = 4.2 + index * 0.01
                        context.append(
                            {
                                "timestamp": (base_time + timedelta(seconds=index)).isoformat(),
                                "sensor_id": "cryo_pump_A1",
                                "temperature_k": temperature,
                                "pressure_atm": 1.01,
                                "vibration_mms": 0.054,
                                "rolling_mean_16": temperature - 0.02,
                                "rolling_std_16": 0.01,
                                "z_score_16": 1.2,
                                "momentum_8": 0.03,
                                "lag_1": temperature - 0.01,
                                "volatility_16": 0.004,
                                "ema_12": temperature - 0.015,
                            }
                        )

                    forecast_response = client.post(
                        "/forecast",
                        json={"sensor_id": "cryo_pump_A1", "horizon": 4, "target": "temperature_k", "context": context},
                    )
                    self.assertEqual(forecast_response.status_code, 200, msg=forecast_response.text)
                    forecast_payload = forecast_response.json()
                    self.assertEqual(forecast_payload["model_name"], "trained_forecast_regressor")
                    self.assertEqual(len(forecast_payload["forecast"]), 4)

                    experiments_response = client.get("/experiments?limit=5")
                    self.assertEqual(experiments_response.status_code, 200)
                    experiments = experiments_response.json()
                    self.assertEqual(len(experiments), 1)
                    self.assertEqual(experiments[0]["model_name"], "trained_forecast_regressor")
            finally:
                os.chdir(previous_cwd)
                timeseries_service_module.SessionLocal = original_session_local
                asyncio.run(engine.dispose())


if __name__ == "__main__":
    unittest.main()
