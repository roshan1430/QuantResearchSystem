from __future__ import annotations

import pandas as pd

from quantresearch.backtesting import BacktestEngine


def test_backtest_engine_returns_report_and_summary() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.Series([100, 101, 100.5, 102, 103, 104], index=index)
    signals = pd.Series([0, 1, 1, -1, 1, 1], index=index)

    report, summary = BacktestEngine(transaction_cost_bps=10, slippage_bps=5).run(prices, signals)

    assert list(report.columns) == [
        "price",
        "returns",
        "signal",
        "turnover",
        "strategy_returns",
        "equity_curve",
        "benchmark_curve",
        "drawdown",
    ]
    assert isinstance(summary.sharpe_ratio, float)
    assert isinstance(summary.sortino_ratio, float)
    assert isinstance(summary.win_rate, float)
    assert report["equity_curve"].iloc[-1] > 0
