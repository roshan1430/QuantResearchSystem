# Methodology

## Research Objective

The platform studies whether regime-aware signals can improve adaptive return forecasting and downstream trading performance under realistic execution assumptions.

## Workflow

1. Ingest market data from Yahoo Finance or curated CSV files.
2. Normalize and preprocess OHLCV time series.
3. Engineer technical, volatility, momentum, and lag features.
4. Label coarse market regimes such as `trend`, `range`, and `stress`.
5. Run statistical diagnostics to validate stationarity and dependence structure.
6. Train forecasting models under walk-forward validation.
7. Convert forecasts into trading signals and evaluate with transaction costs and slippage.
8. Track experiments, compare models, and report risk-adjusted metrics.

## Metrics

- Forecast quality: MAE, RMSE
- Strategy quality: Sharpe ratio, max drawdown, CAGR, turnover
- Research diagnostics: ADF p-value, realized volatility, correlation, cointegration p-value
