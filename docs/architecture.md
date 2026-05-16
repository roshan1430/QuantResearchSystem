# Architecture

## Research Layers

1. `data/`
   Handles Yahoo Finance ingestion, CSV import, normalization, and OHLCV preprocessing.
2. `features/`
   Produces technical, lagged, momentum, and volatility-driven covariates.
3. `research/`
   Runs stationarity, correlation, cointegration, and realized volatility diagnostics.
4. `models/`
   Hosts tree-based, deep learning, transformer, and reinforcement learning interfaces.
5. `evaluation/` and `backtesting/`
   Executes walk-forward validation and realistic trading simulation with costs and slippage.
6. `tracking/`, `visualization/`, and `workflows/`
   Support experiment logging, reporting, and reproducible research runs.

## Design Principles

- Reproducibility first: configuration-driven experiments and explicit artifacts.
- Modularity: each research concern lives in a focused package.
- Graceful optionality: heavyweight stacks are exposed through extras instead of hard dependencies.
- Internship-grade presentation: clear boundaries between data engineering, research, modeling, and evaluation.
