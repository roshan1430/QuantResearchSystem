from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TrainableModel(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> object: ...

    def predict(self, features: pd.DataFrame) -> pd.Series: ...


@dataclass(slots=True)
class WalkForwardResult:
    predictions: pd.Series
    mae: float
    rmse: float


def walk_forward_validation(
    model: TrainableModel,
    features: pd.DataFrame,
    target: pd.Series,
    train_size: int,
    test_size: int,
) -> WalkForwardResult:
    predictions: list[pd.Series] = []
    start = train_size
    while start < len(features):
        stop = min(start + test_size, len(features))
        train_x = features.iloc[:start]
        train_y = target.iloc[:start]
        test_x = features.iloc[start:stop]
        model.fit(train_x, train_y)
        predictions.append(model.predict(test_x))
        start = stop

    combined = pd.concat(predictions).sort_index()
    actual = target.loc[combined.index]
    mae = float(mean_absolute_error(actual, combined))
    rmse = float(mean_squared_error(actual, combined) ** 0.5)
    return WalkForwardResult(predictions=combined, mae=mae, rmse=rmse)
