from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantresearch.evaluation.metrics import BacktestSummary, compute_backtest_summary


@dataclass(slots=True)
class BacktestEngine:
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0

    def run(self, prices: pd.Series, signals: pd.Series) -> tuple[pd.DataFrame, BacktestSummary]:
        returns = prices.pct_change().fillna(0.0)
        aligned_signals = signals.reindex(prices.index).ffill().fillna(0.0).clip(-1.0, 1.0)
        turnover = aligned_signals.diff().abs().fillna(aligned_signals.abs())
        cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10_000
        strategy_returns = aligned_signals.shift(1).fillna(0.0) * returns - turnover * cost_rate
        equity_curve = (1 + strategy_returns).cumprod()
        benchmark_curve = (1 + returns).cumprod()
        drawdown = equity_curve / equity_curve.cummax() - 1
        report = pd.DataFrame(
            {
                "price": prices,
                "returns": returns,
                "signal": aligned_signals,
                "turnover": turnover,
                "strategy_returns": strategy_returns,
                "equity_curve": equity_curve,
                "benchmark_curve": benchmark_curve,
                "drawdown": drawdown,
            }
        )
        return report, compute_backtest_summary(strategy_returns, turnover)
