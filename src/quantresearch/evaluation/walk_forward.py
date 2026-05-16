from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
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
    directional_accuracy: float


@dataclass(slots=True)
class CrossValidationFoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    mae: float
    rmse: float
    directional_accuracy: float


@dataclass(slots=True)
class CrossValidationResult:
    folds: list[CrossValidationFoldResult]

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": fold.fold,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "mae": fold.mae,
                    "rmse": fold.rmse,
                    "directional_accuracy": fold.directional_accuracy,
                }
                for fold in self.folds
            ]
        )

    @property
    def mean_mae(self) -> float:
        return float(np.mean([fold.mae for fold in self.folds])) if self.folds else 0.0

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([fold.rmse for fold in self.folds])) if self.folds else 0.0

    @property
    def mean_directional_accuracy(self) -> float:
        return float(np.mean([fold.directional_accuracy for fold in self.folds])) if self.folds else 0.0


def _directional_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")
    return float((np.sign(aligned_actual) == np.sign(aligned_predicted)).mean()) if len(aligned_actual) else 0.0


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
    return WalkForwardResult(
        predictions=combined,
        mae=mae,
        rmse=rmse,
        directional_accuracy=_directional_accuracy(actual, combined),
    )


def time_series_cross_validation(
    model_factory: object,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_splits: int = 4,
    min_train_size: int = 30,
) -> CrossValidationResult:
    total = len(features)
    if total <= min_train_size + n_splits:
        raise ValueError("Not enough observations for time-series cross-validation.")

    split_size = max((total - min_train_size) // n_splits, 1)
    folds: list[CrossValidationFoldResult] = []
    train_end = min_train_size

    for fold_number in range(1, n_splits + 1):
        test_end = min(train_end + split_size, total)
        if test_end <= train_end:
            break
        model = model_factory()
        train_x = features.iloc[:train_end]
        train_y = target.iloc[:train_end]
        test_x = features.iloc[train_end:test_end]
        test_y = target.iloc[train_end:test_end]
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        folds.append(
            CrossValidationFoldResult(
                fold=fold_number,
                train_start=str(train_x.index[0].date()),
                train_end=str(train_x.index[-1].date()),
                test_start=str(test_x.index[0].date()),
                test_end=str(test_x.index[-1].date()),
                mae=float(mean_absolute_error(test_y, predicted)),
                rmse=float(mean_squared_error(test_y, predicted) ** 0.5),
                directional_accuracy=_directional_accuracy(test_y, predicted),
            )
        )
        train_end = test_end

    return CrossValidationResult(folds=folds)
