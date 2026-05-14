from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.db.models import AnomalyEvent, PredictionRecord
from app.db.session import get_db_session
from app.dependencies import get_ml_service
from app.schemas.ml import (
    AnomalyRequest,
    AnomalyResponse,
    ForecastRequest,
    ForecastResponse,
    ModelCatalogResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.ml_service import MLService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models", response_model=ModelCatalogResponse)
async def list_models(ml_service: MLService = Depends(get_ml_service)) -> ModelCatalogResponse:
    return await ml_service.list_models()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    ml_service: MLService = Depends(get_ml_service),
    db: AsyncSession = Depends(get_db_session),
) -> PredictResponse:
    result = await ml_service.predict(payload)
    try:
        for record, prediction in zip(payload.records, result.predictions):
            db.add(
                PredictionRecord(
                    timestamp=record.timestamp,
                    sensor_id=record.sensor_id,
                    target_name="temperature_k",
                    predicted_value=prediction,
                    model_name=result.model_name,
                    metadata_json={"source": "api", "generated_at": result.generated_at.isoformat()},
                )
            )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("Failed to persist prediction records: %s", exc)
    return result


@router.post("/anomaly", response_model=AnomalyResponse)
async def detect_anomaly(
    payload: AnomalyRequest,
    ml_service: MLService = Depends(get_ml_service),
    db: AsyncSession = Depends(get_db_session),
) -> AnomalyResponse:
    result = await ml_service.detect_anomaly(payload)
    try:
        for item in result.anomalies:
            db.add(
                AnomalyEvent(
                    timestamp=item.timestamp,
                    sensor_id=item.sensor_id,
                    model_name=result.model_name,
                    anomaly_score=item.anomaly_score,
                    threshold=item.threshold,
                    is_anomaly=item.is_anomaly,
                    explanation=item.explanation,
                )
            )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("Failed to persist anomaly records: %s", exc)
    return result


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(payload: ForecastRequest, ml_service: MLService = Depends(get_ml_service)) -> ForecastResponse:
    return await ml_service.forecast(payload)
