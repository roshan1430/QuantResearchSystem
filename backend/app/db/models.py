from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SensorMetric(Base):
    __tablename__ = "sensor_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    sensor_id: Mapped[str] = mapped_column(String(128), index=True)
    temperature_k: Mapped[float] = mapped_column(Float)
    pressure_atm: Mapped[float] = mapped_column(Float)
    vibration_mms: Mapped[float] = mapped_column(Float)
    rolling_mean_16: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_std_16: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_score_16: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_8: Mapped[float | None] = mapped_column(Float, nullable=True)
    lag_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_16: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_12: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_source: Mapped[str] = mapped_column(String(64), default="stream")


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    sensor_id: Mapped[str] = mapped_column(String(128), index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    anomaly_score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, index=True)
    explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    sensor_id: Mapped[str] = mapped_column(String(128), index=True)
    target_name: Mapped[str] = mapped_column(String(128))
    predicted_value: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_name: Mapped[str] = mapped_column(String(255), index=True)
    task_type: Mapped[str] = mapped_column(String(128), index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(64), index=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class ExternalDataEvent(Base):
    __tablename__ = "external_data_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=datetime.utcnow)
    source_name: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
