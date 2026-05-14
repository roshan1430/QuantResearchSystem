from engine.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    ForecastRequest,
    ForecastResponse,
    ModelCatalogResponse,
    PredictRequest,
    PredictResponse,
)
from engine.services.forecasting import forecasting_service
from engine.services.models import BaselineAnomalyService, BaselinePredictorService, describe_models


class ModelRegistry:
    def __init__(self) -> None:
        self.anomaly_service = BaselineAnomalyService()
        self.predictor = BaselinePredictorService()
        self.forecaster = forecasting_service

    def describe(self) -> ModelCatalogResponse:
        return describe_models()

    def predict(self, payload: PredictRequest) -> PredictResponse:
        return self.predictor.predict(payload)

    def detect_anomaly(self, payload: AnomalyRequest) -> AnomalyResponse:
        return self.anomaly_service.detect(payload)

    def forecast(self, payload: ForecastRequest) -> ForecastResponse:
        return self.forecaster.forecast(payload)


registry = ModelRegistry()
