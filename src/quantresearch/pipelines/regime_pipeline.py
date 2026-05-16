from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import pandas as pd

from quantresearch.backtesting import BacktestEngine
from quantresearch.data.preprocessing import preprocess_ohlcv_frame
from quantresearch.evaluation.metrics import BacktestSummary, compute_backtest_summary
from quantresearch.evaluation.walk_forward import (
    CrossValidationResult,
    WalkForwardResult,
    time_series_cross_validation,
    walk_forward_validation,
)
from quantresearch.explainability import compute_shap_summary
from quantresearch.features import engineer_features
from quantresearch.models.deep_learning import LSTMForecaster, TransformerForecaster
from quantresearch.models.traditional import RandomForestRegimeModel, XGBoostRegimeModel
from quantresearch.research import ClusterRegimeDetector
from quantresearch.research.regime import RegimeDetectionResult
from quantresearch.research.statistics import StatisticalResearchReport, run_statistical_research


@dataclass(slots=True)
class RegimeResearchResult:
    dataset: pd.DataFrame
    feature_columns: list[str]
    model_name: str
    walk_forward: WalkForwardResult
    cross_validation: CrossValidationResult
    stats: StatisticalResearchReport
    regime_detection: RegimeDetectionResult
    backtest_report: pd.DataFrame
    backtest_summary: BacktestSummary
    baseline_backtest_summary: BacktestSummary
    regime_performance: pd.DataFrame
    strategy_comparison: pd.DataFrame
    explainability: pd.DataFrame


class RegimeResearchPipeline:
    def __init__(
        self,
        model_factory: Callable[[], object] | None = None,
        backtest_engine: BacktestEngine | None = None,
        regime_detector: ClusterRegimeDetector | None = None,
    ) -> None:
        self.model_factory = model_factory or RandomForestRegimeModel
        self.backtest_engine = backtest_engine or BacktestEngine()
        self.regime_detector = regime_detector or ClusterRegimeDetector()

    @staticmethod
    def available_models() -> dict[str, Callable[[], object]]:
        return {
            "random_forest": RandomForestRegimeModel,
            "xgboost": XGBoostRegimeModel,
            "lstm": LSTMForecaster,
            "transformer": TransformerForecaster,
        }

    def run(self, raw_frame: pd.DataFrame) -> RegimeResearchResult:
        dataset = engineer_features(preprocess_ohlcv_frame(raw_frame))
        regime_detection = self.regime_detector.detect(dataset)
        dataset["regime"] = regime_detection.regime_series.reindex(dataset.index).ffill().bfill()
        dataset["regime_code"] = dataset["regime"].map({"stress": -1.0, "range": 0.0, "trend": 1.0}).fillna(0.0)

        feature_columns = [
            column
            for column in dataset.columns
            if column
            not in {"target_return", "regime", "open", "high", "low", "close", "volume", "log_return", "return"}
        ]
        features = dataset[feature_columns]
        target = dataset["target_return"]

        walk_forward_model = self.model_factory()
        walk_forward = walk_forward_validation(
            model=walk_forward_model,
            features=features,
            target=target,
            train_size=max(int(len(features) * 0.7), 30),
            test_size=max(int(len(features) * 0.1), 5),
        )
        cross_validation = time_series_cross_validation(
            self.model_factory,
            features,
            target,
            n_splits=min(4, max(len(features) // 10, 2)),
            min_train_size=max(int(len(features) * 0.45), 24),
        )

        adaptive_signals = self._adaptive_signals(
            predictions=walk_forward.predictions,
            regimes=dataset.loc[walk_forward.predictions.index, "regime"],
        )
        backtest_report, backtest_summary = self.backtest_engine.run(
            prices=dataset.loc[walk_forward.predictions.index, "close"],
            signals=adaptive_signals,
        )

        baseline_signals = walk_forward.predictions.apply(lambda value: 1.0 if value > 0 else -1.0)
        _, baseline_summary = self.backtest_engine.run(
            prices=dataset.loc[walk_forward.predictions.index, "close"],
            signals=baseline_signals,
        )

        stats = run_statistical_research(dataset)
        full_model = self.model_factory()
        full_model.fit(features, target)
        underlying_model = getattr(full_model, "model", None) or getattr(full_model, "fallback_model", None) or full_model
        explainability = compute_shap_summary(underlying_model, features, target)
        diagnostic_predictions = full_model.predict(features)
        diagnostic_signals = self._adaptive_signals(diagnostic_predictions, dataset["regime"])
        diagnostic_report, diagnostic_summary = self.backtest_engine.run(dataset["close"], diagnostic_signals)
        regime_performance = self._summarize_regime_performance(
            diagnostic_report.join(dataset[["regime"]], how="left")
        )
        strategy_comparison = pd.DataFrame(
            [
                {"strategy": "adaptive_regime_strategy", **asdict(backtest_summary)},
                {"strategy": "directional_baseline", **asdict(baseline_summary)},
                {"strategy": "full_sample_regime_diagnostic", **asdict(diagnostic_summary)},
            ]
        ).sort_values("sharpe_ratio", ascending=False)

        return RegimeResearchResult(
            dataset=dataset,
            feature_columns=feature_columns,
            model_name=type(walk_forward_model).__name__,
            walk_forward=walk_forward,
            cross_validation=cross_validation,
            stats=stats,
            regime_detection=regime_detection,
            backtest_report=backtest_report,
            backtest_summary=backtest_summary,
            baseline_backtest_summary=baseline_summary,
            regime_performance=regime_performance,
            strategy_comparison=strategy_comparison,
            explainability=explainability,
        )

    def _adaptive_signals(self, predictions: pd.Series, regimes: pd.Series) -> pd.Series:
        aligned_regimes = regimes.reindex(predictions.index).ffill().bfill()
        directional = predictions.apply(lambda value: 1.0 if value > 0 else -1.0)
        adaptive = pd.Series(index=predictions.index, dtype=float)
        adaptive.loc[aligned_regimes == "trend"] = directional.loc[aligned_regimes == "trend"]
        adaptive.loc[aligned_regimes == "range"] = -0.5 * directional.loc[aligned_regimes == "range"]
        adaptive.loc[aligned_regimes == "stress"] = 0.0
        return adaptive.fillna(0.0)

    def _summarize_regime_performance(self, report: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for regime, frame in report.groupby("regime"):
            if frame.empty:
                continue
            summary = compute_backtest_summary(frame["strategy_returns"], frame["turnover"])
            rows.append({"regime": regime, **asdict(summary), "observations": float(len(frame))})
        return pd.DataFrame(rows).set_index("regime").sort_values("sharpe_ratio", ascending=False)
