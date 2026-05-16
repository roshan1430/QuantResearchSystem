from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact:
        self.ensure_torch()
        return ModelArtifact(
            name="lstm",
            task="sequence_forecasting",
            metadata={"hidden_size": self.hidden_size, "layers": self.layers, "epochs": self.epochs},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
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

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact:
        self.ensure_torch()
        return ModelArtifact(
            name="transformer",
            task="sequence_forecasting",
            metadata={"d_model": self.d_model, "nhead": self.nhead, "layers": self.layers, "epochs": self.epochs},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        self.ensure_torch()
        return pd.Series(index=features.index, data=0.0, name="prediction")
