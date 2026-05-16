from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestSummary:
    cumulative_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    cagr: float
    volatility: float
    win_rate: float
    turnover: float


def compute_backtest_summary(strategy_returns: pd.Series, turnover: pd.Series) -> BacktestSummary:
    clean_returns = strategy_returns.fillna(0.0)
    cumulative_curve = (1 + clean_returns).cumprod()
    years = max(len(clean_returns) / 252, 1 / 252)
    cagr = float(cumulative_curve.iloc[-1] ** (1 / years) - 1)
    volatility = float(clean_returns.std() * np.sqrt(252))
    sharpe = float(np.sqrt(252) * clean_returns.mean() / clean_returns.std()) if clean_returns.std() else 0.0
    downside = clean_returns.clip(upper=0)
    downside_std = float(downside.std())
    sortino = float(np.sqrt(252) * clean_returns.mean() / downside_std) if downside_std else 0.0
    running_max = cumulative_curve.cummax()
    drawdown = cumulative_curve / running_max - 1
    return BacktestSummary(
        cumulative_return=float(cumulative_curve.iloc[-1] - 1),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=float(drawdown.min()),
        cagr=cagr,
        volatility=volatility,
        win_rate=float((clean_returns > 0).mean()),
        turnover=float(turnover.fillna(0.0).mean()),
    )
