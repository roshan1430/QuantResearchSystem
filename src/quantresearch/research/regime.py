from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class RegimeDetectionResult:
    regime_series: pd.Series
    cluster_assignments: pd.Series
    regime_summary: pd.DataFrame
    feature_columns: list[str]


class ClusterRegimeDetector:
    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_regimes, n_init=20, random_state=random_state)

    def _feature_columns(self, frame: pd.DataFrame) -> list[str]:
        preferred = [
            "volatility_21",
            "momentum_5",
            "momentum_21",
            "macd",
            "macd_signal",
            "range_ratio",
            "volume_zscore_21",
            "trend_gap",
        ]
        return [column for column in preferred if column in frame.columns]

    def detect(self, frame: pd.DataFrame) -> RegimeDetectionResult:
        feature_columns = self._feature_columns(frame)
        if not feature_columns:
            raise ValueError("Regime detection requires engineered feature columns.")

        feature_frame = frame[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(feature_frame) < self.n_regimes * 3:
            raise ValueError("Not enough observations to detect market regimes.")

        scaled = self.scaler.fit_transform(feature_frame)
        clusters = pd.Series(self.model.fit_predict(scaled), index=feature_frame.index, name="cluster")
        summary = feature_frame.groupby(clusters).mean()

        labels = self._label_clusters(summary)
        regime_series = clusters.map(labels).rename("regime_detected")
        regime_summary = (
            feature_frame.assign(cluster=clusters, regime=regime_series)
            .groupby("regime")
            .agg(
                observations=("cluster", "size"),
                avg_volatility=("volatility_21", "mean"),
                avg_momentum=("momentum_21", "mean"),
                avg_range_ratio=("range_ratio", "mean"),
            )
            .sort_values("avg_volatility", ascending=False)
        )

        return RegimeDetectionResult(
            regime_series=regime_series,
            cluster_assignments=clusters,
            regime_summary=regime_summary,
            feature_columns=feature_columns,
        )

    def _label_clusters(self, cluster_summary: pd.DataFrame) -> dict[int, str]:
        stress_cluster = int(cluster_summary["volatility_21"].idxmax())
        remaining = [cluster for cluster in cluster_summary.index if cluster != stress_cluster]

        trend_cluster = max(
            remaining,
            key=lambda cluster: (
                float(cluster_summary.loc[cluster, "momentum_21"]),
                float(cluster_summary.loc[cluster, "macd"]),
            ),
        )
        labels = {stress_cluster: "stress", trend_cluster: "trend"}
        for cluster in cluster_summary.index:
            labels.setdefault(int(cluster), "range")
        return labels
