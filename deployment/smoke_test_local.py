import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


BACKEND = "http://127.0.0.1:8000"
ML_ENGINE = "http://127.0.0.1:8001"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_json(url: str, retries: int = 8, sleep_seconds: float = 1.5) -> dict:
    last_error = None
    for _ in range(retries):
        try:
            return get_json(url)
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds)
    raise RuntimeError(f"service not reachable: {url} ({last_error})")


def main() -> int:
    print("[smoke] checking backend health")
    health = wait_for_json(f"{BACKEND}/health")
    print(f"[ok] /health -> {health.get('status')}")

    print("[smoke] checking system readiness")
    ready = wait_for_json(f"{BACKEND}/system/ready")
    print(f"[ok] /system/ready -> ready={ready.get('ready')}")

    print("[smoke] checking ml-engine direct")
    models = wait_for_json(f"{ML_ENGINE}/models")
    print(f"[ok] ml-engine models -> {len(models.get('models', []))} entries")

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "model_name": "random_forest",
        "records": [
            {
                "timestamp": now,
                "sensor_id": "cryo_pump_A1",
                "temperature_k": 4.22,
                "pressure_atm": 1.01,
                "vibration_mms": 0.054,
                "rolling_mean_16": 4.20,
                "rolling_std_16": 0.02,
                "z_score_16": 1.0,
                "momentum_8": 0.01,
                "lag_1": 4.21,
                "volatility_16": 0.004,
                "ema_12": 4.19,
            }
        ],
    }
    print("[smoke] checking /predict")
    prediction = post_json(f"{BACKEND}/predict", payload)
    print(f"[ok] /predict -> {len(prediction.get('predictions', []))} predictions")

    anomaly_payload = {
        "model_name": "isolation_forest",
        "records": payload["records"],
    }
    print("[smoke] checking /anomaly")
    anomaly = post_json(f"{BACKEND}/anomaly", anomaly_payload)
    print(f"[ok] /anomaly -> {len(anomaly.get('anomalies', []))} anomalies")

    print("[smoke] checking /experiments")
    experiments = get_json(f"{BACKEND}/experiments?limit=5")
    print(f"[ok] /experiments -> {len(experiments)} rows")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"[error] network failure: {exc}")
        print("[hint] Ensure backend (:8000) and ml-engine (:8001) are both running.")
        raise SystemExit(1)
    except Exception as exc:
        print(f"[error] smoke test failed: {exc}")
        print("[hint] Start ml-engine first, then backend, then rerun smoke test.")
        raise SystemExit(1)
