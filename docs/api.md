# API Notes

## Key endpoints
- `GET /health`
- `GET /metrics`
- `GET /anomalies`
- `GET /stream/live`
- `GET /models`
- `POST /predict`
- `POST /anomaly`
- `POST /forecast`
- `POST /train`
- `GET /docs`

## Example anomaly request

```json
{
  "model_name": "isolation_forest",
  "records": [
    {
      "timestamp": "2026-05-08T10:00:00Z",
      "sensor_id": "cryo_pump_A1",
      "temperature_k": 4.32,
      "pressure_atm": 1.01,
      "vibration_mms": 0.051,
      "rolling_mean_16": 4.29,
      "rolling_std_16": 0.03,
      "z_score_16": 1.1,
      "momentum_8": 0.04,
      "lag_1": 4.30,
      "volatility_16": 0.01,
      "ema_12": 4.28
    }
  ]
}
```
