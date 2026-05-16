from __future__ import annotations

import numpy as np
import pandas as pd

from quantresearch.pipelines import RegimeResearchPipeline


def test_regime_pipeline_runs_end_to_end() -> None:
    index = pd.date_range("2020-01-01", periods=180, freq="D")
    base = np.linspace(100, 150, 180)
    raw = pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + np.sin(np.linspace(0, 12, 180)),
            "volume": np.linspace(100_000, 200_000, 180),
        },
        index=index,
    )

    result = RegimeResearchPipeline().run(raw)

    assert not result.dataset.empty
    assert not result.walk_forward.predictions.empty
    assert "equity_curve" in result.backtest_report.columns
    assert not result.regime_performance.empty
    assert "directional_accuracy" in result.cross_validation.summary_frame().columns
    assert not result.explainability.empty
