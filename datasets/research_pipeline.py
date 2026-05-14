from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_PARAMETERS = ("T2M", "PS", "WS10M")
RAW_COLUMNS = ("YEAR", "MO", "DY", "HR", *NASA_PARAMETERS)

SITE_PRESETS = {
    "cern": {"latitude": 46.233, "longitude": 6.055, "sensor_id": "cern_research_station"},
    "kennedy_space_center": {"latitude": 28.5729, "longitude": -80.649, "sensor_id": "nasa_ksc_station"},
}


@dataclass
class ResearchDatasetMetadata:
    source: str
    sensor_id: str
    rows: int
    latitude: float
    longitude: float
    start: str
    end: str
    raw_parameters: list[str]
    notes: list[str]


def build_nasa_power_hourly_url(
    *,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    community: str = "RE",
) -> str:
    params = ",".join(NASA_PARAMETERS)
    return (
        f"{NASA_POWER_BASE_URL}?parameters={params}"
        f"&community={community}"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={start}"
        f"&end={end}"
        "&format=csv"
        "&header=false"
        "&time-standard=UTC"
    )


def load_nasa_power_csv(source: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"NASA POWER file is missing required columns: {', '.join(missing)}")
    return frame


def transform_nasa_power_to_training_frame(raw_frame: pd.DataFrame, *, sensor_id: str) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame["timestamp"] = pd.to_datetime(
        frame[["YEAR", "MO", "DY", "HR"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
        ),
        utc=True,
    )

    temperature_k = pd.to_numeric(frame["T2M"], errors="coerce") + 273.15
    pressure_atm = pd.to_numeric(frame["PS"], errors="coerce") / 101.325
    wind_speed = pd.to_numeric(frame["WS10M"], errors="coerce")
    vibration_proxy = _scale_to_vibration_proxy(wind_speed)

    dataset = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "sensor_id": sensor_id,
            "temperature_k": temperature_k.round(5),
            "pressure_atm": pressure_atm.round(5),
            "vibration_mms": vibration_proxy.round(5),
        }
    ).sort_values("timestamp")

    dataset["rolling_mean_16"] = dataset["temperature_k"].rolling(window=16, min_periods=2).mean()
    dataset["rolling_std_16"] = dataset["temperature_k"].rolling(window=16, min_periods=2).std(ddof=0)
    dataset["z_score_16"] = (
        (dataset["temperature_k"] - dataset["rolling_mean_16"])
        / dataset["rolling_std_16"].where(dataset["rolling_std_16"] > 1e-9)
    )
    dataset["momentum_8"] = dataset["temperature_k"] - dataset["temperature_k"].shift(8)
    dataset["lag_1"] = dataset["temperature_k"].shift(1)
    dataset["volatility_16"] = dataset["vibration_mms"].rolling(window=16, min_periods=2).std(ddof=0)
    dataset["ema_12"] = dataset["temperature_k"].ewm(span=12, adjust=False).mean()
    dataset["target_temperature"] = dataset["temperature_k"].shift(-1)
    dataset = dataset.ffill().bfill()
    dataset = dataset.iloc[:-1].reset_index(drop=True)
    return dataset.round(5)


def write_dataset_manifest(output_csv: str | Path, metadata: ResearchDatasetMetadata) -> Path:
    output_path = Path(output_csv)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source": metadata.source,
                "sensor_id": metadata.sensor_id,
                "rows": metadata.rows,
                "latitude": metadata.latitude,
                "longitude": metadata.longitude,
                "start": metadata.start,
                "end": metadata.end,
                "raw_parameters": metadata.raw_parameters,
                "notes": metadata.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _scale_to_vibration_proxy(wind_speed: pd.Series) -> pd.Series:
    min_speed = float(wind_speed.min())
    max_speed = float(wind_speed.max())
    speed_range = max(max_speed - min_speed, 1e-9)
    normalized = (wind_speed - min_speed) / speed_range
    return 0.02 + normalized * 0.08
