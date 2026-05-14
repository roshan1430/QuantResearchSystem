from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    timestamp: datetime
    sensor_id: str
    temperature_k: float
    pressure_atm: float
    vibration_mms: float
    rolling_mean_16: float | None = None
    rolling_std_16: float | None = None
    z_score_16: float | None = None
    momentum_8: float | None = None
    lag_1: float | None = None
    volatility_16: float | None = None
    ema_12: float | None = None


class PredictRequest(BaseModel):
    model_name: str = "random_forest"
    records: list[FeatureVector]


class PredictResponse(BaseModel):
    model_name: str
    predictions: list[float]
    generated_at: datetime


class AnomalyRequest(BaseModel):
    model_name: str = "isolation_forest"
    records: list[FeatureVector]


class AnomalyItem(BaseModel):
    sensor_id: str
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    threshold: float | None = None
    explanation: dict[str, float] | None = None


class AnomalyResponse(BaseModel):
    model_name: str
    anomalies: list[AnomalyItem]
    generated_at: datetime


class ForecastRequest(BaseModel):
    sensor_id: str
    horizon: int = Field(default=12, ge=1, le=240)
    target: str = "temperature_k"
    context: list[FeatureVector]


class ForecastPoint(BaseModel):
    step: int
    timestamp: datetime
    value: float


class ForecastResponse(BaseModel):
    model_name: str
    sensor_id: str
    target: str
    horizon: int
    forecast: list[ForecastPoint]
    generated_at: datetime


class TrainRequest(BaseModel):
    task_type: str = "anomaly_detection"
    model_name: str
    dataset_path: str
    target_column: str | None = None
    timestamp_column: str = "timestamp"
    sensor_id_column: str = "sensor_id"
    parameters: dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    accepted: bool
    run_name: str
    model_name: str
    task_type: str
    detail: str
    metrics: dict[str, float] | None = None
    artifact_uri: str | None = None


class ModelDescriptor(BaseModel):
    model_name: str
    task_type: str
    framework: str
    status: str


class ModelCatalogResponse(BaseModel):
    models: list[ModelDescriptor]
