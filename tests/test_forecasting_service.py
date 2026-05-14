from __future__ import annotations

import sys
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml-engine"))

from engine.schemas import FeatureVector, ForecastRequest  # noqa: E402
from engine.services.forecasting import TransformerForecaster  # noqa: E402


class ForecastingServiceTests(unittest.TestCase):
    def test_forecast_is_timestamp_ordered_and_respects_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                base_time = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
                context = []
                for index in range(12):
                    value = 4.2 + index * 0.01
                    context.append(
                        FeatureVector(
                            timestamp=base_time + timedelta(seconds=index),
                            sensor_id="sensor_A",
                            temperature_k=value,
                            pressure_atm=1.0,
                            vibration_mms=0.05,
                            rolling_mean_16=value - 0.01,
                            rolling_std_16=0.01,
                            z_score_16=1.0,
                            momentum_8=0.02,
                            lag_1=value - 0.01,
                            volatility_16=0.004,
                            ema_12=value - 0.005,
                        )
                    )

                payload = ForecastRequest(sensor_id="sensor_A", horizon=6, target="temperature_k", context=context)
                result = TransformerForecaster().forecast(payload)

                self.assertEqual(len(result.forecast), 6)
                self.assertEqual(result.forecast[0].timestamp, context[-1].timestamp + timedelta(seconds=1))
                self.assertTrue(all(item.value > context[-1].temperature_k for item in result.forecast))
                self.assertTrue(all(left.timestamp < right.timestamp for left, right in zip(result.forecast, result.forecast[1:])))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
