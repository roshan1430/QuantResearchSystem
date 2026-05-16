from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantresearch.models.base import ModelArtifact


@dataclass
class RLPortfolioAgent:
    algorithm: str = "PPO"
    total_timesteps: int = 20_000

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> ModelArtifact:
        try:
            import gymnasium  # noqa: F401
            import stable_baselines3  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install the 'rl' extra to use reinforcement learning agents.") from exc
        return ModelArtifact(
            name=self.algorithm.lower(),
            task="portfolio_optimization",
            metadata={"total_timesteps": self.total_timesteps},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(index=features.index, data=1.0 / max(len(features.columns), 1), name="weight")
