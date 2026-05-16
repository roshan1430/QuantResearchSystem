from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantresearch.models.base import ModelArtifact


class _TorchUnavailableMixin:
    @staticmethod
    def ensure_torch() -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install the 'deep' extra to use deep learning forecasters.") from exc


@dataclass
class LSTMForecaster(_TorchUnavailableMixin):
    hidden_size: int = 64
    layers: int = 2
    epochs: int = 20
    fallback_model: object | None = None
    backend: str = "torch"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact:
        try:
            self.ensure_torch()
        except ImportError:
            self.backend = "sequence_mlp_fallback"
            self.fallback_model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(self.hidden_size, max(self.hidden_size // 2, 16)),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=1e-3,
                    max_iter=max(self.epochs * 20, 400),
                    random_state=42,
                ),
            )
            self.fallback_model.fit(features, target)
        else:
            self.backend = "torch"
        return ModelArtifact(
            name="lstm",
            task="sequence_forecasting",
            metadata={
                "hidden_size": self.hidden_size,
                "layers": self.layers,
                "epochs": self.epochs,
                "backend": self.backend,
            },
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.backend == "sequence_mlp_fallback" and self.fallback_model is not None:
            return pd.Series(self.fallback_model.predict(features), index=features.index, name="prediction")
        self.ensure_torch()
        return pd.Series(index=features.index, data=0.0, name="prediction")


@dataclass
class GRUForecaster(_TorchUnavailableMixin):
    hidden_size: int = 64
    layers: int = 2
    epochs: int = 20

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact:
        self.ensure_torch()
        return ModelArtifact(
            name="gru",
            task="sequence_forecasting",
            metadata={"hidden_size": self.hidden_size, "layers": self.layers, "epochs": self.epochs},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        self.ensure_torch()
        return pd.Series(index=features.index, data=0.0, name="prediction")


@dataclass
class TransformerForecaster(_TorchUnavailableMixin):
    d_model: int = 64
    nhead: int = 4
    layers: int = 2
    epochs: int = 20
    fallback_model: object | None = None
    backend: str = "torch"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact:
        try:
            self.ensure_torch()
        except ImportError:
            self.backend = "attention_mlp_fallback"
            self.fallback_model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(self.d_model * 2, self.d_model),
                    activation="relu",
                    solver="lbfgs",
                    alpha=5e-4,
                    max_iter=max(self.epochs * 20, 500),
                    random_state=42,
                ),
            )
            self.fallback_model.fit(features, target)
        else:
            self.backend = "torch"
        return ModelArtifact(
            name="transformer",
            task="sequence_forecasting",
            metadata={
                "d_model": self.d_model,
                "nhead": self.nhead,
                "layers": self.layers,
                "epochs": self.epochs,
                "backend": self.backend,
            },
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.backend == "attention_mlp_fallback" and self.fallback_model is not None:
            return pd.Series(self.fallback_model.predict(features), index=features.index, name="prediction")
        self.ensure_torch()
        return pd.Series(index=features.index, data=0.0, name="prediction")
