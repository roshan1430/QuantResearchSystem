from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_regime_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame.index, y=frame["close"], mode="lines", name="Close"))
    if "regime" in frame.columns:
        regime_map = {"trend": "#146c43", "range": "#c27c0e", "stress": "#b42318"}
        for regime, color in regime_map.items():
            mask = frame["regime"] == regime
            if mask.any():
                figure.add_trace(
                    go.Scatter(
                        x=frame.index[mask],
                        y=frame.loc[mask, "close"],
                        mode="markers",
                        marker={"size": 6, "color": color},
                        name=f"Regime: {regime}",
                    )
                )
    figure.update_layout(template="plotly_white", title="Market Regime Detection")
    return figure


def create_backtest_figure(report: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=report.index, y=report["equity_curve"], mode="lines", name="Strategy"))
    figure.update_layout(template="plotly_white", title="Backtest Equity Curve")
    return figure
