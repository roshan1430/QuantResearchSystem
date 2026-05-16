from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0)


def engineer_features(frame: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    features = frame.copy()
    close = features["close"]
    returns = features["return"]

    features["rsi_14"] = _rsi(close, period=14)
    features["ema_12"] = _ema(close, span=12)
    features["ema_26"] = _ema(close, span=26)
    features["trend_gap"] = (features["ema_12"] - features["ema_26"]) / close.replace(0, np.nan)
    features["macd"] = features["ema_12"] - features["ema_26"]
    features["macd_signal"] = _ema(features["macd"], span=9)
    features["bb_mid"] = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    features["bb_upper"] = features["bb_mid"] + 2 * rolling_std
    features["bb_lower"] = features["bb_mid"] - 2 * rolling_std
    features["volatility_21"] = returns.rolling(window=21).std() * np.sqrt(252)
    features["momentum_5"] = close.pct_change(periods=5)
    features["momentum_21"] = close.pct_change(periods=21)
    features["range_ratio"] = (features["high"] - features["low"]) / features["close"]
    features["realized_volatility_5"] = returns.rolling(window=5).std() * np.sqrt(252)
    features["drawdown_21"] = close / close.rolling(window=21).max() - 1
    features["volume_zscore_21"] = (
        (features["volume"] - features["volume"].rolling(21).mean())
        / features["volume"].rolling(21).std()
    )

    for lag in range(1, lags + 1):
        features[f"return_lag_{lag}"] = returns.shift(lag)
        features[f"volume_lag_{lag}"] = features["volume"].shift(lag)

    features["regime"] = "range"
    return features.dropna()
