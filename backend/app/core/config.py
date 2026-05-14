from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "Real-Time Time-Series Intelligence Platform"
    app_version: str = "0.2.0"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    database_url: str = "sqlite+aiosqlite:///./local_platform.db"
    ml_engine_url: str = "http://127.0.0.1:8001"
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_metric_topic: str = "processed_features"
    kafka_anomaly_topic: str = "anomaly_events"
    kafka_alert_topic: str = "system_alerts"
    enable_kafka_consumers: bool = False
    enable_demo_stream: bool = True
    demo_sensor_id: str = "cryo_pump_A1"
    demo_tick_seconds: float = 1.5
    nasa_api_key: str = "DEMO_KEY"
    nasa_live_url: str = "https://api.nasa.gov/planetary/apod"
    cern_live_url: str = ""
    external_timeout_seconds: float = 15.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
