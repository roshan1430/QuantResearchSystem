from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research_pipeline import (
    SITE_PRESETS,
    ResearchDatasetMetadata,
    build_nasa_power_hourly_url,
    load_nasa_power_csv,
    transform_nasa_power_to_training_frame,
    write_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training-ready research dataset from actual NASA POWER hourly data."
    )
    parser.add_argument("--site", choices=sorted(SITE_PRESETS), default="cern")
    parser.add_argument("--start", required=True, help="Start date in YYYYMMDD format.")
    parser.add_argument("--end", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument(
        "--source-csv",
        help="Optional local NASA POWER CSV file. If omitted, the script reads directly from the official NASA URL.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "generated" / "nasa_power_training.csv"),
        help="Output CSV path for the transformed training dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    site = SITE_PRESETS[args.site]
    source = args.source_csv or build_nasa_power_hourly_url(
        latitude=site["latitude"],
        longitude=site["longitude"],
        start=args.start,
        end=args.end,
    )
    raw = load_nasa_power_csv(source)
    dataset = transform_nasa_power_to_training_frame(raw, sensor_id=site["sensor_id"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    metadata = ResearchDatasetMetadata(
        source="nasa-power-hourly",
        sensor_id=site["sensor_id"],
        rows=len(dataset),
        latitude=site["latitude"],
        longitude=site["longitude"],
        start=args.start,
        end=args.end,
        raw_parameters=["T2M", "PS", "WS10M"],
        notes=[
            "Data fetched from the official NASA POWER hourly point API.",
            "temperature_k is derived from T2M by converting Celsius to Kelvin.",
            "pressure_atm is derived from PS by converting kilopascals to atmospheres.",
            "vibration_mms is a scaled wind-speed proxy used to fit the existing platform schema.",
            "CERN remains an external event context source through the backend external fetch flow.",
        ],
    )
    manifest_path = write_dataset_manifest(output_path, metadata)

    print(f"[ok] wrote training dataset: {output_path}")
    print(f"[ok] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
