from datetime import datetime, timezone

import httpx

from app.core.config import settings
from app.schemas.ml import (
    AnomalyRequest,
    AnomalyResponse,
    ForecastRequest,
    ForecastResponse,
    ModelCatalogResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)


class MLService:
    async def _post(self, path: str, payload: dict) -> dict:
        async with httpx.AsyncClient(base_url=settings.ml_engine_url, timeout=30.0) as client:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return response.json()

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(base_url=settings.ml_engine_url, timeout=15.0) as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.json()

    async def list_models(self) -> ModelCatalogResponse:
        return ModelCatalogResponse(**await self._get("/models"))

    async def predict(self, payload: PredictRequest) -> PredictResponse:
        return PredictResponse(**await self._post("/predict", payload.model_dump(mode="json")))

    async def detect_anomaly(self, payload: AnomalyRequest) -> AnomalyResponse:
        return AnomalyResponse(**await self._post("/anomaly", payload.model_dump(mode="json")))

    async def forecast(self, payload: ForecastRequest) -> ForecastResponse:
        return ForecastResponse(**await self._post("/forecast", payload.model_dump(mode="json")))

    async def train(self, payload: TrainRequest) -> TrainResponse:
        response = await self._post("/train", payload.model_dump(mode="json"))
        return TrainResponse(**response)


ml_service = MLService()
