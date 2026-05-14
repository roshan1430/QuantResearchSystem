from pathlib import Path

import pandas as pd
import json


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"
DATASET = DATASET_DIR / "sample_timeseries.csv"
GENERATED_DATASET = DATASET_DIR / "generated" / "nasa_power_training.csv"
NASA_SAMPLE = DATASET_DIR / "nasa_apod_sample.json"
CERN_SAMPLE = DATASET_DIR / "cern_live_sample.json"


def main() -> None:
    selected_dataset = GENERATED_DATASET if GENERATED_DATASET.exists() else DATASET
    frame = pd.read_csv(selected_dataset, parse_dates=["timestamp"])
    nasa_event = json.loads(NASA_SAMPLE.read_text(encoding="utf-8"))
    cern_event = json.loads(CERN_SAMPLE.read_text(encoding="utf-8"))
    print("Dataset:", selected_dataset.name)
    print("Rows:", len(frame))
    print("Sensors:", frame["sensor_id"].nunique())
    print()

    print("External science context")
    print("NASA sample title:", nasa_event["title"])
    print("CERN sample title:", cern_event["title"])
    print()

    print("Per-sensor summary")
    summary = (
        frame.groupby("sensor_id")[["temperature_k", "pressure_atm", "vibration_mms", "target_temperature"]]
        .agg(["mean", "min", "max"])
        .round(4)
    )
    print(summary)
    print()

    print("Correlation with target_temperature")
    correlations = frame.drop(columns=["timestamp", "sensor_id"]).corr(numeric_only=True)["target_temperature"]
    print(correlations.sort_values(ascending=False).round(4))
    print()

    print("Recent rows")
    print(frame.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
