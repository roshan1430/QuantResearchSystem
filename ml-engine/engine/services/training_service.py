from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from engine.schemas import TrainRequest, TrainResponse
from engine.services.datasets import DatasetValidationError, load_training_frame, write_dataset_manifest
from engine.services.features import FEATURE_COLUMNS
from engine.services.forecasting import forecasting_service


class TrainingService:
    def start_training(self, payload: TrainRequest) -> TrainResponse:
        dataset = Path(payload.dataset_path)
        run_name = f"{payload.model_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        if not dataset.exists():
            return TrainResponse(
                accepted=False,
                run_name=run_name,
                model_name=payload.model_name,
                task_type=payload.task_type,
                detail=f"Dataset not found: {dataset}",
            )

        try:
            import mlflow  # type: ignore
        except ImportError:
            mlflow = None

        try:
            frame, summary = load_training_frame(
                dataset,
                timestamp_column=payload.timestamp_column,
                sensor_id_column=payload.sensor_id_column,
                target_column=payload.target_column,
            )
        except DatasetValidationError as exc:
            return TrainResponse(
                accepted=False,
                run_name=run_name,
                model_name=payload.model_name,
                task_type=payload.task_type,
                detail=str(exc),
            )

        manifest_path = write_dataset_manifest(dataset, summary, output_dir=Path("experiments") / "manifests")
        metrics: dict[str, float]
        artifact_uri: str | None = None
        if payload.task_type == "forecasting":
            if summary.target_column is None:
                return TrainResponse(
                    accepted=False,
                    run_name=run_name,
                    model_name=payload.model_name,
                    task_type=payload.task_type,
                    detail="Forecasting training requires a target column in the dataset or request.",
                )
            metrics, artifact_uri = self._train_regressor(frame, summary.target_column, run_name=run_name, dataset_path=str(dataset))
            detail = (
                f"Walk-forward MAE: {metrics['mae_walk_forward']:.4f}; "
                f"baseline MAE: {metrics['baseline_mae']:.4f}; "
                f"dataset rows: {summary.rows}; "
                f"artifact: {artifact_uri}"
            )
        else:
            metrics = self._train_anomaly(frame)
            detail = (
                f"Isolation forest mean decision score: {metrics['train_score_mean']:.4f}; "
                f"anomaly rate: {metrics['anomaly_rate']:.4f}; "
                f"dataset rows: {summary.rows}"
            )

        self._write_run_manifest(
            run_name=run_name,
            payload=payload,
            dataset=dataset,
            dataset_manifest=manifest_path,
            metrics=metrics,
        )
        if mlflow is not None:
            mlflow.set_experiment("time_series_intelligence")
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(payload.parameters | {"model_name": payload.model_name, "task_type": payload.task_type})
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
        return TrainResponse(
            accepted=True,
            run_name=run_name,
            model_name=payload.model_name,
            task_type=payload.task_type,
            detail=detail,
            metrics=metrics,
            artifact_uri=artifact_uri,
        )

    def _train_regressor(
        self,
        frame: pd.DataFrame,
        target_column: str,
        *,
        run_name: str,
        dataset_path: str,
    ) -> tuple[dict[str, float], str]:
        splits = TimeSeriesSplit(n_splits=4)
        X = frame[FEATURE_COLUMNS].fillna(0.0)
        y = frame[target_column]
        errors = []
        baseline_errors = []
        trained_model = None
        for train_index, test_index in splits.split(X):
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            predictions = model.predict(X.iloc[test_index])
            errors.append(mean_absolute_error(y.iloc[test_index], predictions))
            baseline = [float(y.iloc[train_index].iloc[-1])] * len(test_index)
            baseline_errors.append(mean_absolute_error(y.iloc[test_index], baseline))
            trained_model = model
        if trained_model is None:
            raise ValueError("Unable to train forecasting model from the provided dataset")
        metrics = {
            "mae_walk_forward": float(sum(errors) / len(errors)),
            "baseline_mae": float(sum(baseline_errors) / len(baseline_errors)),
        }
        artifact_uri = forecasting_service.save_trained_model(
            trained_model,
            run_name=run_name,
            dataset_path=dataset_path,
            target_column=target_column,
            metrics=metrics,
        )
        return metrics, artifact_uri

    def _train_anomaly(self, frame: pd.DataFrame) -> dict[str, float]:
        model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        X = frame[FEATURE_COLUMNS].fillna(0.0)
        model.fit(X)
        scores = model.decision_function(X)
        predictions = model.predict(X)
        anomaly_rate = float((predictions == -1).sum() / len(predictions))
        return {
            "train_score_mean": float(scores.mean()),
            "anomaly_rate": anomaly_rate,
        }

    def _write_run_manifest(
        self,
        *,
        run_name: str,
        payload: TrainRequest,
        dataset: Path,
        dataset_manifest: Path,
        metrics: dict[str, float],
    ) -> None:
        manifests_dir = Path("experiments") / "runs"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        payload_json = {
            "run_name": run_name,
            "dataset_path": str(dataset),
            "task_type": payload.task_type,
            "model_name": payload.model_name,
            "target_column": payload.target_column,
            "parameters": payload.parameters,
            "dataset_manifest": str(dataset_manifest),
            "metrics": metrics,
        }
        (manifests_dir / f"{run_name}.json").write_text(
            json.dumps(payload_json, indent=2),
            encoding="utf-8",
        )


training_service = TrainingService()
