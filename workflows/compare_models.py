from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantresearch.backtesting import BacktestEngine
from quantresearch.data.preprocessing import preprocess_ohlcv_frame
from quantresearch.evaluation.walk_forward import walk_forward_validation
from quantresearch.features import engineer_features
from quantresearch.models.traditional import RandomForestRegimeModel, XGBoostRegimeModel


def compare_models(raw_frame: pd.DataFrame) -> pd.DataFrame:
    dataset = engineer_features(preprocess_ohlcv_frame(raw_frame))
    excluded = {"target_return", "regime", "open", "high", "low", "close", "volume", "log_return", "return"}
    feature_columns = [column for column in dataset.columns if column not in excluded]
    features = dataset[feature_columns]
    target = dataset["target_return"]
    backtest_engine = BacktestEngine()

    candidates = {"random_forest": RandomForestRegimeModel()}
    try:
        candidates["xgboost"] = XGBoostRegimeModel()
    except Exception:
        pass

    rows: list[dict[str, float | str]] = []
    for name, model in candidates.items():
        walk_forward = walk_forward_validation(
            model=model,
            features=features,
            target=target,
            train_size=max(int(len(features) * 0.7), 30),
            test_size=max(int(len(features) * 0.1), 5),
        )
        signals = walk_forward.predictions.apply(lambda value: 1.0 if value > 0 else -1.0)
        _, summary = backtest_engine.run(dataset.loc[walk_forward.predictions.index, "close"], signals)
        rows.append(
            {
                "model": name,
                "mae": walk_forward.mae,
                "rmse": walk_forward.rmse,
                **asdict(summary),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
