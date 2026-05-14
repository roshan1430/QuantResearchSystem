from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datasets"))

from research_pipeline import (  # noqa: E402
    ResearchDatasetMetadata,
    build_nasa_power_hourly_url,
    transform_nasa_power_to_training_frame,
    write_dataset_manifest,
)


class ResearchPipelineTests(unittest.TestCase):
    def test_build_nasa_power_url_targets_official_hourly_endpoint(self) -> None:
        url = build_nasa_power_hourly_url(
            latitude=46.233,
            longitude=6.055,
            start="20250101",
            end="20250107",
        )

        self.assertIn("power.larc.nasa.gov/api/temporal/hourly/point", url)
        self.assertIn("parameters=T2M,PS,WS10M", url)
        self.assertIn("time-standard=UTC", url)

    def test_transform_nasa_power_to_training_frame_generates_platform_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "YEAR": [2025, 2025, 2025, 2025],
                "MO": [1, 1, 1, 1],
                "DY": [1, 1, 1, 1],
                "HR": [0, 1, 2, 3],
                "T2M": [10.0, 11.0, 12.0, 13.0],
                "PS": [101.325, 101.225, 101.125, 101.025],
                "WS10M": [2.0, 3.0, 4.0, 5.0],
            }
        )

        dataset = transform_nasa_power_to_training_frame(raw, sensor_id="cern_research_station")

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.loc[0, "sensor_id"], "cern_research_station")
        self.assertAlmostEqual(dataset.loc[0, "temperature_k"], 283.15, places=2)
        self.assertAlmostEqual(dataset.loc[0, "pressure_atm"], 1.0, places=5)
        self.assertIn("target_temperature", dataset.columns)

    def test_write_dataset_manifest_persists_source_metadata(self) -> None:
        metadata = ResearchDatasetMetadata(
            source="nasa-power-hourly",
            sensor_id="cern_research_station",
            rows=24,
            latitude=46.233,
            longitude=6.055,
            start="20250101",
            end="20250107",
            raw_parameters=["T2M", "PS", "WS10M"],
            notes=["official NASA POWER source"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv = Path(temp_dir) / "research.csv"
            output_csv.write_text("timestamp\n", encoding="utf-8")
            manifest = write_dataset_manifest(output_csv, metadata)
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(payload["source"], "nasa-power-hourly")
        self.assertEqual(payload["sensor_id"], "cern_research_station")
        self.assertEqual(payload["rows"], 24)


if __name__ == "__main__":
    unittest.main()
