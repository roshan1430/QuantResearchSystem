from __future__ import annotations

import numpy as np
import pandas as pd

from quantresearch.data.preprocessing import preprocess_ohlcv_frame
from quantresearch.features import engineer_features


def make_frame(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=rows, freq="D")
    base = np.linspace(100, 130, rows)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.5,
            "low": base - 1.0,
            "close": base + np.sin(np.linspace(0, 10, rows)),
            "volume": np.linspace(1_000_000, 1_500_000, rows),
        },
        index=index,
    )


def test_feature_engineering_generates_expected_columns() -> None:
    frame = engineer_features(preprocess_ohlcv_frame(make_frame()))
    expected = {
        "rsi_14",
        "macd",
        "bb_upper",
        "volatility_21",
        "momentum_21",
        "trend_gap",
        "realized_volatility_5",
        "drawdown_21",
        "return_lag_5",
        "regime",
    }
    assert expected.issubset(frame.columns)
    assert (frame["regime"] == "range").all()
