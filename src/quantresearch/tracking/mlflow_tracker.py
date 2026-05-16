from __future__ import annotations

from contextlib import AbstractContextManager


class _NullRun(AbstractContextManager["_NullRun"]):
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def start_run(self, run_name: str) -> AbstractContextManager[object]:
        try:
            import mlflow
        except ImportError:
            return _NullRun()

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, object]) -> None:
        try:
            import mlflow
        except ImportError:
            return
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        try:
            import mlflow
        except ImportError:
            return
        mlflow.log_metrics(metrics)
