from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from engine.services.features import FEATURE_COLUMNS


TIMESTAMP_ALIASES = ("timestamp", "Britimestamp")
OPTIONAL_TARGET_COLUMNS = ("target_temperature", "forecast_target_temperature")


@dataclass
class DatasetSummary:
    rows: int
    sensors: int
    feature_columns: list[str]
    target_column: str | None


class DatasetValidationError(ValueError):
    pass


def load_training_frame(
    dataset_path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    sensor_id_column: str = "sensor_id",
    target_column: str | None = None,
) -> tuple[pd.DataFrame, DatasetSummary]:
    path = Path(dataset_path)
    frame = pd.read_csv(path)
    frame = _normalize_columns(frame, timestamp_column=timestamp_column)

    required_columns = {sensor_id_column, *FEATURE_COLUMNS}
    missing_columns = sorted(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    if target_column is None:
        target_column = _detect_target_column(frame)
    if target_column and target_column not in frame.columns:
        raise DatasetValidationError(f"Target column not found: {target_column}")

    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    if frame[timestamp_column].isna().any():
        raise DatasetValidationError(f"Column '{timestamp_column}' contains invalid timestamps")

    frame = frame.sort_values([sensor_id_column, timestamp_column]).reset_index(drop=True)
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].ffill().fillna(0.0)

    summary = DatasetSummary(
        rows=len(frame),
        sensors=int(frame[sensor_id_column].nunique()),
        feature_columns=list(FEATURE_COLUMNS),
        target_column=target_column,
    )
    return frame, summary


def write_dataset_manifest(
    dataset_path: str | Path,
    summary: DatasetSummary,
    *,
    output_dir: str | Path,
) -> Path:
    dataset = Path(dataset_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{dataset.stem}.manifest.json"
    payload = {
        "dataset": dataset.name,
        "rows": summary.rows,
        "sensors": summary.sensors,
        "feature_columns": summary.feature_columns,
        "target_column": summary.target_column,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _normalize_columns(frame: pd.DataFrame, *, timestamp_column: str) -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        for alias in TIMESTAMP_ALIASES:
            if alias in frame.columns:
                frame = frame.rename(columns={alias: timestamp_column})
                break
    return frame


def _detect_target_column(frame: pd.DataFrame) -> str | None:
    for column in OPTIONAL_TARGET_COLUMNS:
        if column in frame.columns:
            return column
    return None
