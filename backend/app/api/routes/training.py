from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ExperimentRun
from app.db.session import get_db_session
from app.dependencies import get_ml_service
from app.schemas.ml import TrainRequest, TrainResponse
from app.services.ml_service import MLService

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
async def train_model(
    payload: TrainRequest,
    ml_service: MLService = Depends(get_ml_service),
    db: AsyncSession = Depends(get_db_session),
) -> TrainResponse:
    result = await ml_service.train(payload)
    run = ExperimentRun(
        run_name=result.run_name,
        task_type=payload.task_type,
        model_name=result.model_name,
        status="accepted" if result.accepted else "failed",
        metrics=result.metrics,
        parameters=payload.parameters,
        artifact_uri=result.artifact_uri,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.commit()
    return result
