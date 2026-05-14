from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from engine.schemas import FeatureVector, ForecastPoint, ForecastRequest, ForecastResponse
from engine.services.features import FEATURE_COLUMNS, records_to_frame


class TransformerForecaster:
    """Uses a persisted forecasting regressor when available and falls back to a stable baseline."""

    baseline_model_name = "transformer_forecaster"
    trained_model_name = "trained_forecast_regressor"

    def __init__(self) -> None:
        self._artifact_dir = Path("experiments") / "models"
        self._artifact_path = self._artifact_dir / "forecast_regressor.joblib"
        self._metadata_path = self._artifact_dir / "forecast_regressor.metadata.json"

    def forecast(self, payload: ForecastRequest) -> ForecastResponse:
        model = self.load_trained_model()
        if model is None:
            return self._baseline_forecast(payload)
        return self._trained_forecast(payload, model)

    def load_trained_model(self) -> RandomForestRegressor | None:
        if not self._artifact_path.exists():
            return None
        return joblib.load(self._artifact_path)

    def has_trained_model(self) -> bool:
        return self._artifact_path.exists() and self._metadata_path.exists()

    def artifact_path(self) -> Path:
        return self._artifact_path

    def training_metadata(self) -> dict | None:
        if not self._metadata_path.exists():
            return None
        return json.loads(self._metadata_path.read_text(encoding="utf-8"))

    def save_trained_model(
        self,
        model: RandomForestRegressor,
        *,
        run_name: str,
        dataset_path: str,
        target_column: str,
        metrics: dict[str, float],
    ) -> str:
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self._artifact_path)
        self._metadata_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "model_name": self.trained_model_name,
                    "dataset_path": dataset_path,
                    "target_column": target_column,
                    "metrics": metrics,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(self._artifact_path)

    def _trained_forecast(self, payload: ForecastRequest, model: RandomForestRegressor) -> ForecastResponse:
        frame = records_to_frame([record.model_dump(mode="json") for record in payload.context])
        history = [record.model_dump(mode="json") for record in payload.context]
        start_time = payload.context[-1].timestamp if payload.context else datetime.now(timezone.utc)
        forecast = []
        for step in range(1, payload.horizon + 1):
            current = frame.iloc[[-1]][FEATURE_COLUMNS]
            predicted_value = float(model.predict(current)[0])
            next_row = self._build_recursive_row(history=history, predicted_value=predicted_value, step=step, start_time=start_time)
            history.append(next_row)
            frame = records_to_frame(history)
            forecast.append(
                ForecastPoint(
                    step=step,
                    timestamp=datetime.fromisoformat(next_row["timestamp"]),
                    value=predicted_value,
                )
            )

        return ForecastResponse(
            model_name=self.trained_model_name,
            sensor_id=payload.sensor_id,
            target=payload.target,
            horizon=payload.horizon,
            forecast=forecast,
            generated_at=datetime.now(timezone.utc),
        )

    def _build_recursive_row(self, *, history: list[dict], predicted_value: float, step: int, start_time: datetime) -> dict:
        last_row = history[-1]
        temperature_history = [float(item["temperature_k"]) for item in history] + [predicted_value]
        vibration_history = [float(item["vibration_mms"]) for item in history] + [float(last_row["vibration_mms"])]

        rolling_window = temperature_history[-16:]
        vibration_window = vibration_history[-16:]
        rolling_mean = float(np.mean(rolling_window))
        rolling_std = float(np.std(rolling_window))
        z_score = (predicted_value - rolling_mean) / rolling_std if rolling_std > 1e-9 else 0.0
        momentum = predicted_value - temperature_history[-9] if len(temperature_history) >= 9 else 0.0
        lag_1 = temperature_history[-2]
        volatility = float(np.std(vibration_window))
        ema = float(pd.Series(temperature_history).ewm(span=12, adjust=False).mean().iloc[-1])

        return {
            "timestamp": (start_time + timedelta(seconds=step)).isoformat(),
            "sensor_id": last_row["sensor_id"],
            "temperature_k": predicted_value,
            "pressure_atm": float(last_row["pressure_atm"]),
            "vibration_mms": float(last_row["vibration_mms"]),
            "rolling_mean_16": rolling_mean,
            "rolling_std_16": rolling_std,
            "z_score_16": z_score,
            "momentum_8": momentum,
            "lag_1": lag_1,
            "volatility_16": volatility,
            "ema_12": ema,
        }

    def _baseline_forecast(self, payload: ForecastRequest) -> ForecastResponse:
        history = np.array([getattr(point, payload.target) for point in payload.context], dtype=float)
        if history.size == 0:
            history = np.array([0.0], dtype=float)

        ema_window = history[-12:]
        smoothed = self._ema(ema_window, alpha=0.32)

        slope_window = history[-8:] if history.size >= 8 else history
        steps = np.arange(slope_window.size, dtype=float)
        slope = self._linear_slope(steps, slope_window)

        recent_diff = np.diff(history[-6:]) if history.size >= 3 else np.array([], dtype=float)
        acceleration = float(recent_diff[-1] - recent_diff.mean()) if recent_diff.size >= 2 else 0.0
        acceleration = float(np.clip(acceleration, -0.015, 0.015))

        start_time = payload.context[-1].timestamp if payload.context else datetime.now(timezone.utc)
        forecast = []
        anchor = max(float(smoothed), float(history[-1])) if slope >= 0 else min(float(smoothed), float(history[-1]))
        value = anchor
        for step in range(1, payload.horizon + 1):
            dampening = max(0.25, 1.0 - (step - 1) * 0.08)
            value = value + slope + (acceleration * dampening)
            forecast.append(
                ForecastPoint(
                    step=step,
                    timestamp=start_time + timedelta(seconds=step),
                    value=float(value),
                )
            )

        return ForecastResponse(
            model_name=self.baseline_model_name,
            sensor_id=payload.sensor_id,
            target=payload.target,
            horizon=payload.horizon,
            forecast=forecast,
            generated_at=datetime.now(timezone.utc),
        )

    def _ema(self, values: np.ndarray, *, alpha: float) -> float:
        result = float(values[0])
        for value in values[1:]:
            result = alpha * float(value) + (1.0 - alpha) * result
        return result

    def _linear_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        if y.size <= 1:
            return 0.0
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 1e-12:
            return 0.0
        numer = float(((x - x_mean) * (y - y_mean)).sum())
        return numer / denom


forecasting_service = TransformerForecaster()
