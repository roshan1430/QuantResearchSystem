from __future__ import annotations

import numpy as np
import pandas as pd

from workflows.compare_models import compare_models


def test_compare_models_returns_ranked_frame() -> None:
    index = pd.date_range("2022-01-01", periods=180, freq="D")
    base = np.linspace(100, 145, 180)
    raw = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + np.sin(np.linspace(0, 16, 180)),
            "volume": np.linspace(900_000, 1_400_000, 180),
        },
        index=index,
    )

    comparison = compare_models(raw)

    assert {"model", "mae", "rmse", "sharpe_ratio", "directional_accuracy"}.issubset(comparison.columns)
    assert set(comparison["model"]) == {"random_forest", "xgboost", "lstm", "transformer"}
