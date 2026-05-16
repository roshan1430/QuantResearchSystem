from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from quantresearch.evaluation.metrics import BacktestSummary
from quantresearch.evaluation.walk_forward import CrossValidationResult, WalkForwardResult
from quantresearch.research.statistics import StatisticalResearchReport


def _table(frame: pd.DataFrame, index: bool = True) -> str:
    rounded = frame.round(4)
    try:
        return rounded.to_markdown(index=index)
    except ImportError:
        return "```text\n" + rounded.to_string(index=index) + "\n```"


def write_experiment_report(
    path: str | Path,
    *,
    experiment_name: str,
    model_name: str,
    walk_forward: WalkForwardResult,
    cross_validation: CrossValidationResult,
    summary: BacktestSummary,
    regime_summary: pd.DataFrame,
    regime_performance: pd.DataFrame,
    stats: StatisticalResearchReport,
    explainability: pd.DataFrame,
) -> Path:
    backtest_metrics = _table(pd.DataFrame([asdict(summary)]), index=False)
    diagnostics = _table(pd.DataFrame([asdict(stats)]), index=False)
    payload = f"""# Experiment Report: {experiment_name}

## Overview

- Model: `{model_name}`
- Walk-forward MAE: `{walk_forward.mae:.6f}`
- Walk-forward RMSE: `{walk_forward.rmse:.6f}`
- Directional accuracy: `{walk_forward.directional_accuracy:.2%}`
- Sharpe ratio: `{summary.sharpe_ratio:.3f}`
- Sortino ratio: `{summary.sortino_ratio:.3f}`
- CAGR: `{summary.cagr:.2%}`

## Cross-Validation Summary

{_table(cross_validation.summary_frame(), index=False)}

## Backtest Metrics

{backtest_metrics}

## Regime Detection Summary

{_table(regime_summary)}

## Performance By Regime

{_table(regime_performance)}

## Statistical Diagnostics

{diagnostics}

## Explainability Snapshot

{_table(explainability.head(10), index=False)}
"""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload, encoding="utf-8")
    return report_path


def write_model_comparison_report(path: str | Path, comparison: pd.DataFrame) -> Path:
    report = f"""# Model Comparison Report

## Ranking

{_table(comparison, index=False)}

## Takeaways

- Best Sharpe ratio: `{comparison.iloc[0]['model']}`.
- Best directional accuracy: `{comparison.sort_values('directional_accuracy', ascending=False).iloc[0]['model']}`.
- Lowest RMSE: `{comparison.sort_values('rmse', ascending=True).iloc[0]['model']}`.
"""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def write_backtesting_report(
    path: str | Path,
    *,
    summary: BacktestSummary,
    regime_performance: pd.DataFrame,
    strategy_comparison: pd.DataFrame,
) -> Path:
    strategy_summary = _table(pd.DataFrame([asdict(summary)]), index=False)
    report = f"""# Backtesting Report

## Strategy Summary

{strategy_summary}

## Strategy Comparison

{_table(strategy_comparison, index=False)}

## Regime-Level Performance

{_table(regime_performance)}
"""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path
