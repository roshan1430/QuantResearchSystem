from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class FeatureConfig:
    rolling_window: int = 16
    momentum_window: int = 8
    ema_span: int = 12
    history_limit: int = 256


class FeatureEngineeringEngine:
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.buffers: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=self.config.history_limit))

    def ingest(self, payload: dict) -> dict | None:
        sensor_id = payload["sensor_id"]
        self.buffers[sensor_id].append(payload)
        if len(self.buffers[sensor_id]) < self.config.rolling_window:
            return None

        frame = pl.DataFrame(list(self.buffers[sensor_id])).sort("timestamp")
        frame = frame.with_columns(
            [
                pl.col("temperature_k").rolling_mean(self.config.rolling_window).alias("rolling_mean_16"),
                pl.col("temperature_k").rolling_std(self.config.rolling_window).alias("rolling_std_16"),
                (
                    (pl.col("temperature_k") - pl.col("temperature_k").rolling_mean(self.config.rolling_window))
                    / (pl.col("temperature_k").rolling_std(self.config.rolling_window) + 1e-6)
                ).alias("z_score_16"),
                (pl.col("temperature_k") - pl.col("temperature_k").shift(self.config.momentum_window)).alias("momentum_8"),
                pl.col("temperature_k").shift(1).alias("lag_1"),
                pl.col("vibration_mms").rolling_std(self.config.rolling_window).alias("volatility_16"),
                pl.col("temperature_k").ewm_mean(span=self.config.ema_span).alias("ema_12"),
            ]
        )
        latest = frame.tail(1).to_dicts()[0]
        if latest["rolling_mean_16"] is None:
            return None
        latest["event_source"] = "streaming-engine"
        return latest
