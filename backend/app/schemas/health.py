from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    kafka: str
    ml_engine: str


class ReadinessResponse(BaseModel):
    ready: bool
    database_ok: bool
    ml_engine_ok: bool
    details: dict[str, str]
