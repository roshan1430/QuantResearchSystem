from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MetricPointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class MetricWindowResponse(BaseModel):
    points: list[MetricPointResponse]
    count: int


class AnomalyEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    sensor_id: str
    model_name: str
    anomaly_score: float
    threshold: float | None = None
    is_anomaly: bool
    explanation: dict | None = None


class ExperimentSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_name: str
    task_type: str
    model_name: str
    status: str
    metrics: dict | None = None
    created_at: datetime


class StreamSnapshotResponse(BaseModel):
    metrics: list[MetricPointResponse]
    anomalies: list[AnomalyEventResponse]
