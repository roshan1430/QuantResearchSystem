from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml-engine"))

from engine.services.datasets import load_training_frame, write_dataset_manifest  # noqa: E402


class DatasetServiceTests(unittest.TestCase):
    def test_load_training_frame_normalizes_timestamp_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "alias_dataset.csv"
            dataset.write_text(
                "\n".join(
                    [
                        "Britimestamp,sensor_id,temperature_k,pressure_atm,vibration_mms,rolling_mean_16,rolling_std_16,z_score_16,momentum_8,lag_1,volatility_16,ema_12,target_temperature",
                        "2026-05-08T10:00:00Z,sensor_A,4.2,1.0,0.05,4.1,0.01,1.0,0.02,4.18,0.004,4.19,4.25",
                        "2026-05-08T10:00:01Z,sensor_A,4.3,1.0,0.05,4.2,0.01,1.1,0.03,4.20,0.004,4.22,4.28",
                    ]
                ),
                encoding="utf-8",
            )

            frame, summary = load_training_frame(dataset)

            self.assertIn("timestamp", frame.columns)
            self.assertEqual(summary.rows, 2)
            self.assertEqual(summary.sensors, 1)
            self.assertEqual(summary.target_column, "target_temperature")

    def test_write_dataset_manifest_creates_json_summary(self) -> None:
        source = ROOT / "datasets" / "sample_timeseries.csv"
        frame, summary = load_training_frame(source)

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = write_dataset_manifest(source, summary, output_dir=temp_dir)
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(payload["dataset"], "sample_timeseries.csv")
        self.assertEqual(payload["rows"], len(frame))
        self.assertEqual(payload["target_column"], "target_temperature")


if __name__ == "__main__":
    unittest.main()
