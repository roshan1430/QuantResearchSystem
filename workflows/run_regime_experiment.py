from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantresearch.config import get_settings
from quantresearch.data.ingestion import load_market_data
from quantresearch.logging_utils import configure_logging, get_logger
from quantresearch.pipelines import RegimeResearchPipeline
from quantresearch.research import (
    write_backtesting_report,
    write_experiment_report,
)
from quantresearch.tracking import MLflowTracker
from quantresearch.visualization import (
    save_correlation_heatmap,
    save_equity_curve,
    save_feature_importance,
    save_portfolio_allocation,
    save_prediction_vs_actual,
    save_regime_allocation,
    save_regime_visualization,
)

from compare_models import compare_models, write_comparison_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a market regime detection experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = config["data"]
    model_cfg = config["model"]
    evaluation_cfg = config.get("evaluation", {})
    model_name = model_cfg.get("name", "random_forest")
    model_factory = RegimeResearchPipeline.available_models().get(
        model_name,
        RegimeResearchPipeline.available_models()["random_forest"],
    )

    bundle = load_market_data(
        symbol=data_cfg["symbol"],
        start=data_cfg.get("start"),
        end=data_cfg.get("end"),
        csv_path=data_cfg.get("csv_path"),
        interval=data_cfg.get("interval", "1d"),
    )
    result = RegimeResearchPipeline(model_factory=model_factory).run(bundle.frame)

    tracker = MLflowTracker(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )
    with tracker.start_run(config["experiment"]["name"]):
        tracker.log_params({"symbol": data_cfg["symbol"], "model": model_name})
        tracker.log_metrics(
            {
                "mae": result.walk_forward.mae,
                "rmse": result.walk_forward.rmse,
                "directional_accuracy": result.walk_forward.directional_accuracy,
                "sharpe_ratio": result.backtest_summary.sharpe_ratio,
                "sortino_ratio": result.backtest_summary.sortino_ratio,
                "max_drawdown": result.backtest_summary.max_drawdown,
                "cagr": result.backtest_summary.cagr,
                "volatility": result.backtest_summary.volatility,
                "win_rate": result.backtest_summary.win_rate,
            }
        )

    output_dir = Path(config["experiment"].get("artifact_dir", settings.artifact_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    result.dataset.to_csv(output_dir / "engineered_dataset.csv")
    result.backtest_report.to_csv(output_dir / "backtest_report.csv")
    pd.DataFrame({"prediction": result.walk_forward.predictions}).to_csv(output_dir / "predictions.csv")
    result.regime_performance.to_csv(output_dir / "regime_performance.csv")
    result.explainability.to_csv(output_dir / "feature_importance.csv", index=False)
    result.cross_validation.summary_frame().to_csv(output_dir / "cross_validation.csv", index=False)
    result.strategy_comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)

    summary_payload = {
        "experiment": config["experiment"]["name"],
        "model": model_name,
        "walk_forward": {
            "mae": result.walk_forward.mae,
            "rmse": result.walk_forward.rmse,
            "directional_accuracy": result.walk_forward.directional_accuracy,
        },
        "cross_validation": {
            "mean_mae": result.cross_validation.mean_mae,
            "mean_rmse": result.cross_validation.mean_rmse,
            "mean_directional_accuracy": result.cross_validation.mean_directional_accuracy,
        },
        "backtest_summary": asdict(result.backtest_summary),
        "baseline_backtest_summary": asdict(result.baseline_backtest_summary),
        "evaluation_metrics_requested": evaluation_cfg.get("metrics", []),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    actual = result.dataset.loc[result.walk_forward.predictions.index, "target_return"]
    save_equity_curve(result.backtest_report, output_dir / "equity_curve.png")
    save_prediction_vs_actual(actual, result.walk_forward.predictions, output_dir / "prediction_vs_actual.png")
    save_correlation_heatmap(result.dataset[result.feature_columns], output_dir / "correlation_heatmap.png")
    save_feature_importance(result.explainability, output_dir / "feature_importance.png")
    save_portfolio_allocation(result.backtest_report, output_dir / "portfolio_allocation.png")
    save_regime_visualization(result.dataset[["close", "regime"]], output_dir / "market_regimes.png")
    save_regime_allocation(
        result.backtest_report.join(result.dataset[["regime"]], how="left"),
        output_dir / "regime_allocation.png",
    )

    write_experiment_report(
        output_dir / "experiment_report.md",
        experiment_name=config["experiment"]["name"],
        model_name=model_name,
        walk_forward=result.walk_forward,
        cross_validation=result.cross_validation,
        summary=result.backtest_summary,
        regime_summary=result.regime_detection.regime_summary,
        regime_performance=result.regime_performance,
        stats=result.stats,
        explainability=result.explainability,
    )
    write_backtesting_report(
        output_dir / "backtesting_report.md",
        summary=result.backtest_summary,
        regime_performance=result.regime_performance,
        strategy_comparison=result.strategy_comparison,
    )
    comparison = compare_models(bundle.frame)
    write_comparison_artifacts(comparison, output_dir)
    logger.info("Experiment completed and saved to %s", output_dir)


if __name__ == "__main__":
    main()
