from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(ROOT / "ml-engine"))

from build_research_dataset import main as build_dataset_main  # noqa: E402
from engine.schemas import TrainRequest  # noqa: E402
from engine.services.training_service import training_service  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real NASA POWER dataset and train the forecasting model in one command."
    )
    parser.add_argument("--site", default="cern", choices=["cern", "kennedy_space_center"])
    parser.add_argument("--start", required=True, help="Start date in YYYYMMDD format.")
    parser.add_argument("--end", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument(
        "--service-url",
        default="http://127.0.0.1:8001/train",
        help="Optional ML engine training endpoint. Falls back to direct in-process training if unreachable.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "datasets" / "generated" / "nasa_power_training.csv"),
        help="Output CSV path for the generated research dataset.",
    )
    parser.add_argument(
        "--source-csv",
        help="Optional local NASA POWER CSV file for offline transformation and training.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Optional prebuilt training dataset. If supplied, the build step is skipped and the file is trained directly.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path or args.output
    if not args.dataset_path:
        build_args = [
            "build_research_dataset.py",
            "--site",
            args.site,
            "--start",
            args.start,
            "--end",
            args.end,
            "--output",
            args.output,
        ]
        if args.source_csv:
            build_args.extend(["--source-csv", args.source_csv])

        previous_argv = sys.argv[:]
        sys.argv = build_args
        try:
            build_dataset_main()
        finally:
            sys.argv = previous_argv

    payload = TrainRequest(
        task_type="forecasting",
        model_name="trained_forecast_regressor",
        dataset_path=dataset_path,
        target_column="target_temperature",
        parameters={
            "source": "deployment/train_research_pipeline.py",
            "site": args.site,
            "start": args.start,
            "end": args.end,
            "used_prebuilt_dataset": bool(args.dataset_path),
        },
    )
    result = _train(payload, service_url=args.service_url)
    print(f"[ok] training accepted: {result['accepted']}")
    print(f"[ok] run name: {result['run_name']}")
    print(f"[ok] detail: {result['detail']}")
    if result.get("artifact_uri"):
        print(f"[ok] artifact: {result['artifact_uri']}")
    return 0


def _train(payload: TrainRequest, *, service_url: str) -> dict:
    request = urllib.request.Request(
        service_url,
        data=json.dumps(payload.model_dump(mode="json")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError):
        response = training_service.start_training(payload)
        return response.model_dump(mode="json")


if __name__ == "__main__":
    raise SystemExit(main())
