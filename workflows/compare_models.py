from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantresearch.pipelines import RegimeResearchPipeline
from quantresearch.research import write_model_comparison_report


def compare_models(raw_frame: pd.DataFrame) -> pd.DataFrame:
    candidates = RegimeResearchPipeline.available_models()
    rows: list[dict[str, float | str]] = []
    for name, factory in candidates.items():
        result = RegimeResearchPipeline(model_factory=factory).run(raw_frame)
        rows.append(
            {
                "model": name,
                "mae": result.walk_forward.mae,
                "rmse": result.walk_forward.rmse,
                "directional_accuracy": result.walk_forward.directional_accuracy,
                "cv_mae": result.cross_validation.mean_mae,
                "cv_rmse": result.cross_validation.mean_rmse,
                "cv_directional_accuracy": result.cross_validation.mean_directional_accuracy,
                **asdict(result.backtest_summary),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def write_comparison_artifacts(comparison: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    write_model_comparison_report(output_path / "model_comparison_report.md", comparison)
