from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class StatisticalResearchReport:
    adf_pvalue: float
    adf_statistic: float
    close_return_correlation: float
    realized_volatility: float
    cointegration_pvalue: float | None


def run_statistical_research(frame: pd.DataFrame, benchmark: pd.Series | None = None) -> StatisticalResearchReport:
    close = frame["close"].dropna()
    returns = frame["return"].dropna()
    try:
        from statsmodels.tsa.stattools import adfuller, coint
    except ImportError:
        adf_statistic = float(returns.autocorr(lag=1) if len(returns) > 1 else 0.0)
        adf_pvalue = 1.0
        coint = None
    else:
        adf_statistic, adf_pvalue, *_ = adfuller(returns)
    close_return_correlation = close.pct_change().corr(returns)
    realized_volatility = float(returns.std() * np.sqrt(252))

    cointegration_pvalue: float | None = None
    if benchmark is not None and coint is not None:
        aligned = pd.concat([close, benchmark], axis=1).dropna()
        if len(aligned) > 20:
            _, cointegration_pvalue, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])

    return StatisticalResearchReport(
        adf_pvalue=float(adf_pvalue),
        adf_statistic=float(adf_statistic),
        close_return_correlation=float(close_return_correlation),
        realized_volatility=realized_volatility,
        cointegration_pvalue=cointegration_pvalue,
    )
