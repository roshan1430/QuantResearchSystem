from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantresearch.backtesting import BacktestEngine
from quantresearch.data.preprocessing import preprocess_ohlcv_frame
from quantresearch.evaluation.walk_forward import WalkForwardResult, walk_forward_validation
from quantresearch.features import engineer_features
from quantresearch.models.traditional import RandomForestRegimeModel
from quantresearch.research.statistics import StatisticalResearchReport, run_statistical_research


@dataclass(slots=True)
class RegimeResearchResult:
    dataset: pd.DataFrame
    walk_forward: WalkForwardResult
    stats: StatisticalResearchReport
    backtest_report: pd.DataFrame
    backtest_summary: object


class RegimeResearchPipeline:
    def __init__(self, model: RandomForestRegimeModel | None = None, backtest_engine: BacktestEngine | None = None) -> None:
        self.model = model or RandomForestRegimeModel()
        self.backtest_engine = backtest_engine or BacktestEngine()

    def run(self, raw_frame: pd.DataFrame) -> RegimeResearchResult:
        dataset = engineer_features(preprocess_ohlcv_frame(raw_frame))
        feature_columns = [
            column
            for column in dataset.columns
            if column
            not in {"target_return", "regime", "open", "high", "low", "close", "volume", "log_return", "return"}
        ]
        features = dataset[feature_columns]
        target = dataset["target_return"]
        walk_forward = walk_forward_validation(
            model=self.model,
            features=features,
            target=target,
            train_size=max(int(len(features) * 0.7), 30),
            test_size=max(int(len(features) * 0.1), 5),
        )
        signals = walk_forward.predictions.apply(lambda value: 1.0 if value > 0 else -1.0)
        backtest_report, backtest_summary = self.backtest_engine.run(
            prices=dataset.loc[walk_forward.predictions.index, "close"],
            signals=signals,
        )
        stats = run_statistical_research(dataset)
        return RegimeResearchResult(
            dataset=dataset,
            walk_forward=walk_forward,
            stats=stats,
            backtest_report=backtest_report,
            backtest_summary=backtest_summary,
        )
