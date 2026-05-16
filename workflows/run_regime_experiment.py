from __future__ import annotations

import argparse
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
from quantresearch.tracking import MLflowTracker


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

    bundle = load_market_data(
        symbol=data_cfg["symbol"],
        start=data_cfg.get("start"),
        end=data_cfg.get("end"),
        csv_path=data_cfg.get("csv_path"),
        interval=data_cfg.get("interval", "1d"),
    )
    pipeline = RegimeResearchPipeline()
    result = pipeline.run(bundle.frame)

    tracker = MLflowTracker(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )
    with tracker.start_run(config["experiment"]["name"]):
        tracker.log_params({"symbol": data_cfg["symbol"], "model": model_cfg["name"]})
        tracker.log_metrics(
            {
                "mae": result.walk_forward.mae,
                "rmse": result.walk_forward.rmse,
                "sharpe_ratio": result.backtest_summary.sharpe_ratio,
                "max_drawdown": result.backtest_summary.max_drawdown,
                "cagr": result.backtest_summary.cagr,
            }
        )

    output_dir = Path(config["experiment"].get("artifact_dir", settings.artifact_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    result.dataset.to_csv(output_dir / "engineered_dataset.csv")
    result.backtest_report.to_csv(output_dir / "backtest_report.csv")
    pd.DataFrame({"prediction": result.walk_forward.predictions}).to_csv(output_dir / "predictions.csv")
    logger.info("Experiment completed and saved to %s", output_dir)


if __name__ == "__main__":
    main()
