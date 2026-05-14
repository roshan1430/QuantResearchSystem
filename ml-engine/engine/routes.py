from fastapi import APIRouter

from engine.schemas import (
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
from engine.services.registry import registry
from engine.services.training_service import training_service

router = APIRouter()


@router.get("/models", response_model=ModelCatalogResponse)
async def list_models() -> ModelCatalogResponse:
    return registry.describe()


@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    return registry.predict(payload)


@router.post("/anomaly", response_model=AnomalyResponse)
async def anomaly(payload: AnomalyRequest) -> AnomalyResponse:
    return registry.detect_anomaly(payload)


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(payload: ForecastRequest) -> ForecastResponse:
    return registry.forecast(payload)


@router.post("/train", response_model=TrainResponse)
async def train(payload: TrainRequest) -> TrainResponse:
    return training_service.start_training(payload)
