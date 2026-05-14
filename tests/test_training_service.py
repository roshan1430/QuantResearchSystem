from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml-engine"))

from engine.schemas import TrainRequest  # noqa: E402
from engine.services.training_service import training_service  # noqa: E402


class TrainingServiceTests(unittest.TestCase):
    def test_forecasting_training_writes_manifests(self) -> None:
        source_dataset = ROOT / "datasets" / "sample_timeseries.csv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset = temp_root / "sample_timeseries.csv"
            shutil.copyfile(source_dataset, dataset)

            previous_cwd = Path.cwd()
            os.chdir(temp_root)
            try:
                response = training_service.start_training(
                    TrainRequest(
                        task_type="forecasting",
                        model_name="random_forest",
                        dataset_path=str(dataset),
                        target_column="target_temperature",
                        parameters={"source": "unit-test"},
                    )
                )
            finally:
                os.chdir(previous_cwd)

            self.assertTrue(response.accepted, msg=response.detail)
            self.assertTrue(response.artifact_uri)
            manifest_dir = temp_root / "experiments" / "manifests"
            run_dir = temp_root / "experiments" / "runs"
            model_dir = temp_root / "experiments" / "models"
            self.assertTrue(manifest_dir.exists())
            self.assertTrue(run_dir.exists())
            self.assertTrue(model_dir.exists())

            run_files = list(run_dir.glob("*.json"))
            self.assertEqual(len(run_files), 1)
            payload = json.loads(run_files[0].read_text(encoding="utf-8"))
            self.assertIn("mae_walk_forward", payload["metrics"])
            self.assertIn("baseline_mae", payload["metrics"])


if __name__ == "__main__":
    unittest.main()
