from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestSummary:
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    cagr: float
    turnover: float


def compute_backtest_summary(strategy_returns: pd.Series, turnover: pd.Series) -> BacktestSummary:
    clean_returns = strategy_returns.fillna(0.0)
    cumulative_curve = (1 + clean_returns).cumprod()
    years = max(len(clean_returns) / 252, 1 / 252)
    cagr = float(cumulative_curve.iloc[-1] ** (1 / years) - 1)
    sharpe = float(np.sqrt(252) * clean_returns.mean() / clean_returns.std()) if clean_returns.std() else 0.0
    running_max = cumulative_curve.cummax()
    drawdown = cumulative_curve / running_max - 1
    return BacktestSummary(
        cumulative_return=float(cumulative_curve.iloc[-1] - 1),
        sharpe_ratio=sharpe,
        max_drawdown=float(drawdown.min()),
        cagr=cagr,
        turnover=float(turnover.fillna(0.0).mean()),
    )
