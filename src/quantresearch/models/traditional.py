from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quantresearch.models.base import ModelArtifact


@dataclass
class RandomForestRegimeModel:
    n_estimators: int = 300
    random_state: int = 42
    model: RandomForestRegressor = field(init=False)

    def __post_init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            min_samples_leaf=5,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact:
        self.model.fit(features, target)
        return ModelArtifact(
            name="random_forest",
            task="return_forecasting",
            metadata={"n_estimators": self.n_estimators},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(features), index=features.index, name="prediction")


@dataclass
class XGBoostRegimeModel:
    params: dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
        }
    )
    model: object | None = field(init=False, default=None)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install the 'tree' extra to use XGBoost models.") from exc
        self.model = XGBRegressor(**self.params)
        self.model.fit(features, target)
        return ModelArtifact(name="xgboost", task="return_forecasting", metadata=self.params)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return pd.Series(self.model.predict(features), index=features.index, name="prediction")
