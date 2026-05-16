from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(slots=True)
class ModelArtifact:
    name: str
    task: str
    metadata: dict[str, object]


class ForecastModel(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> ModelArtifact: ...

    def predict(self, features: pd.DataFrame) -> pd.Series: ...
