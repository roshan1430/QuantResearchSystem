from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REGIME_COLORS = {"trend": "#146c43", "range": "#c27c0e", "stress": "#b42318"}


def save_equity_curve(report: pd.DataFrame, path: str | Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(report.index, report["equity_curve"], label="Adaptive strategy", color="#0f766e", linewidth=2)
    if "benchmark_curve" in report.columns:
        axis.plot(report.index, report["benchmark_curve"], label="Buy and hold", color="#6b7280", linewidth=1.5)
    axis.set_title("Equity Curve")
    axis.set_ylabel("Growth of 1.0")
    axis.legend()
    axis.grid(alpha=0.2)
    return _save_figure(figure, path)


def save_prediction_vs_actual(actual: pd.Series, predicted: pd.Series, path: str | Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(actual.index, actual, label="Actual return", color="#111827", linewidth=1.8)
    axis.plot(predicted.index, predicted, label="Predicted return", color="#2563eb", linewidth=1.4)
    axis.set_title("Prediction vs Actual")
    axis.legend()
    axis.grid(alpha=0.2)
    return _save_figure(figure, path)


def save_correlation_heatmap(frame: pd.DataFrame, path: str | Path) -> Path:
    correlation = frame.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(correlation.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=90, fontsize=8)
    axis.set_yticks(range(len(correlation.index)))
    axis.set_yticklabels(correlation.index, fontsize=8)
    axis.set_title("Feature Correlation Heatmap")
    figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    figure.tight_layout()
    return _save_figure(figure, path)


def save_feature_importance(importance: pd.DataFrame, path: str | Path) -> Path:
    top = importance.head(12).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top["feature"], top["mean_abs_shap"], color="#7c3aed")
    axis.set_title("Feature Importance")
    axis.set_xlabel("Importance")
    axis.grid(axis="x", alpha=0.2)
    return _save_figure(figure, path)


def save_portfolio_allocation(report: pd.DataFrame, path: str | Path) -> Path:
    exposure = report["signal"].value_counts(normalize=True).rename(index={-1.0: "Short", 0.0: "Cash", 1.0: "Long"})
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.pie(
        exposure.values,
        labels=exposure.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ef4444", "#9ca3af", "#10b981"],
    )
    axis.set_title("Portfolio Allocation")
    return _save_figure(figure, path)


def save_regime_visualization(frame: pd.DataFrame, path: str | Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(frame.index, frame["close"], color="#111827", linewidth=1.5, label="Close")
    for regime, color in REGIME_COLORS.items():
        mask = frame["regime"] == regime
        if mask.any():
            axis.scatter(frame.index[mask], frame.loc[mask, "close"], s=18, color=color, label=regime.title())
    axis.set_title("Detected Market Regimes")
    axis.legend()
    axis.grid(alpha=0.2)
    return _save_figure(figure, path)


def save_regime_allocation(report: pd.DataFrame, path: str | Path) -> Path:
    counts = report["regime"].value_counts()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(counts.index, counts.values, color=[REGIME_COLORS.get(label, "#6b7280") for label in counts.index])
    axis.set_title("Market Regime Allocation")
    axis.set_ylabel("Observations")
    return _save_figure(figure, path)


def _save_figure(figure: plt.Figure, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
