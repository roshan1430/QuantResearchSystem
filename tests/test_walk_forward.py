from __future__ import annotations

import numpy as np
import pandas as pd

from quantresearch.evaluation.walk_forward import walk_forward_validation
from quantresearch.models.traditional import RandomForestRegimeModel


def test_walk_forward_validation_produces_predictions() -> None:
    index = pd.date_range("2021-01-01", periods=100, freq="D")
    features = pd.DataFrame(
        {
            "f1": np.linspace(0, 1, 100),
            "f2": np.cos(np.linspace(0, 5, 100)),
        },
        index=index,
    )
    target = pd.Series(features["f1"] * 0.5 + 0.1, index=index)

    result = walk_forward_validation(
        model=RandomForestRegimeModel(n_estimators=50),
        features=features,
        target=target,
        train_size=60,
        test_size=10,
    )

    assert len(result.predictions) == 40
    assert result.mae >= 0
    assert result.rmse >= 0
    assert 0 <= result.directional_accuracy <= 1
