from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchSettings(BaseSettings):
    """Central settings for experiments and reproducible research runs."""

    model_config = SettingsConfigDict(
        env_prefix="QRS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "QuantResearchSystem"
    environment: str = "development"
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")
    log_level: str = "INFO"
    transaction_cost_bps: float = Field(default=10.0, ge=0.0)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    target_column: str = "target_return"
    regime_column: str = "regime"
    random_state: int = 42
    forecast_horizon: int = Field(default=5, ge=1)
    sequence_length: int = Field(default=30, ge=5)
    train_split: float = Field(default=0.7, gt=0.0, lt=1.0)
    validation_split: float = Field(default=0.15, gt=0.0, lt=1.0)
    benchmark_symbol: str = "^GSPC"
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "market-regime-research"


@lru_cache(maxsize=1)
def get_settings() -> ResearchSettings:
    return ResearchSettings()
