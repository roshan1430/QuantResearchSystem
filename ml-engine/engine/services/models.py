from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import OneClassSVM

from engine.schemas import (
    AnomalyItem,
    AnomalyRequest,
    AnomalyResponse,
    ModelCatalogResponse,
    ModelDescriptor,
    PredictRequest,
    PredictResponse,
)
from engine.services.forecasting import forecasting_service
from engine.services.features import FEATURE_COLUMNS, records_to_frame


@dataclass
class SupportedModel:
    model_name: str
    task_type: str
    framework: str
    status: str


class BaselineAnomalyService:
    def __init__(self) -> None:
        self.models = {
            "isolation_forest": IsolationForest(n_estimators=200, contamination=0.05, random_state=42),
            "one_class_svm": OneClassSVM(kernel="rbf", nu=0.05),
        }
        self._warm_models()

    def _warm_models(self) -> None:
        sample = np.random.default_rng(42).normal(size=(256, len(FEATURE_COLUMNS)))
        sample_frame = records_to_frame(
            [dict(zip(FEATURE_COLUMNS, row.tolist())) for row in sample]
        )
        for model in self.models.values():
            model.fit(sample_frame[FEATURE_COLUMNS])

    def detect(self, payload: AnomalyRequest) -> AnomalyResponse:
        frame = records_to_frame([record.model_dump(mode="json") for record in payload.records])
        model = self.models[payload.model_name]
        scores = model.decision_function(frame[FEATURE_COLUMNS])
        predictions = model.predict(frame[FEATURE_COLUMNS])
        threshold = float(np.percentile(scores, 5))
        anomalies = []
        for idx, record in enumerate(payload.records):
            importance = {
                "z_score_16": float(frame.iloc[idx]["z_score_16"]),
                "volatility_16": float(frame.iloc[idx]["volatility_16"]),
            }
            anomalies.append(
                AnomalyItem(
                    sensor_id=record.sensor_id,
                    timestamp=record.timestamp,
                    anomaly_score=float(scores[idx]),
                    is_anomaly=bool(predictions[idx] == -1),
                    threshold=threshold,
                    explanation=importance,
                )
            )
        return AnomalyResponse(
            model_name=payload.model_name,
            anomalies=anomalies,
            generated_at=datetime.now(timezone.utc),
        )


class BaselinePredictorService:
    def __init__(self) -> None:
        self.model = RandomForestRegressor(n_estimators=150, random_state=42)
        self._warm_model()

    def _warm_model(self) -> None:
        rng = np.random.default_rng(7)
        sample = rng.normal(size=(512, len(FEATURE_COLUMNS)))
        target = sample[:, 0] * 0.65 + sample[:, 6] * 0.2 + rng.normal(scale=0.05, size=512)
        sample_frame = records_to_frame(
            [dict(zip(FEATURE_COLUMNS, row.tolist())) for row in sample]
        )
        self.model.fit(sample_frame[FEATURE_COLUMNS], target)

    def predict(self, payload: PredictRequest) -> PredictResponse:
        frame = records_to_frame([record.model_dump(mode="json") for record in payload.records])
        predictions = self.model.predict(frame[FEATURE_COLUMNS]).tolist()
        return PredictResponse(
            model_name=payload.model_name,
            predictions=[float(item) for item in predictions],
            generated_at=datetime.now(timezone.utc),
        )


SUPPORTED_MODELS = [
    SupportedModel("isolation_forest", "anomaly_detection", "scikit-learn", "online"),
    SupportedModel("one_class_svm", "anomaly_detection", "scikit-learn", "online"),
    SupportedModel("random_forest", "regression", "scikit-learn", "online"),
    SupportedModel("xgboost", "regression", "xgboost", "roadmap"),
    SupportedModel("lstm_autoencoder", "anomaly_detection", "pytorch", "planned"),
]


def describe_models() -> ModelCatalogResponse:
    forecasting_status = "trained" if forecasting_service.has_trained_model() else "context-aware-baseline"
    return ModelCatalogResponse(
        models=[
            ModelDescriptor(
                model_name=item.model_name,
                task_type=item.task_type,
                framework=item.framework,
                status=item.status,
            )
            for item in SUPPORTED_MODELS
        ]
        + [
            ModelDescriptor(
                model_name=forecasting_service.trained_model_name,
                task_type="forecasting",
                framework="scikit-learn",
                status=forecasting_status,
            )
        ]
    )
