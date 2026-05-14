# Developer Guide

## Services
- `frontend`: React + Vite scientific dashboard.
- `backend`: FastAPI gateway, persistence, websocket fan-out.
- `ml-engine`: model serving and experiment entrypoint.
- `streaming-processor`: Kafka consumer that engineers features.
- `streaming-producer`: deterministic synthetic signal generator.
- `postgres`: PostgreSQL / Timescale-compatible storage.
- `mlflow`: experiment tracking server.

## Local workflow
1. Copy `backend/.env.example` into `backend/.env` if you want local overrides.
2. Start the platform with `docker-compose up --build`.
3. Visit `http://localhost:5173` for the dashboard and `http://localhost:8000/docs` for OpenAPI.

### Fast local (no Docker)
1. Start local processes:
   - `python deployment/run_local.py`
2. Run smoke test:
   - `python deployment/smoke_test_local.py`
3. Check readiness endpoint:
   - `GET http://127.0.0.1:8000/system/ready`

### Build a stronger training dataset
1. Generate a real hourly dataset from the official NASA POWER API:
   - `python datasets/build_research_dataset.py --site cern --start 20250101 --end 20250107`
2. Train the ML engine against the generated file in `datasets/generated/`.
3. Keep CERN external fetch enabled for research-event context and dashboard realism.

### One-command real-data training
1. Build the NASA POWER dataset and train forecasting in one command:
   - `python deployment/train_research_pipeline.py --site cern --start 20250101 --end 20250107`
2. If the ML engine is running on `:8001`, the script will train through the service endpoint.
3. If the service is not running, the script falls back to direct local training and still writes the forecasting artifact.

### External live data sources
- List source adapters:
  - `GET http://127.0.0.1:8000/external/sources`
- Pull NASA live event now:
  - `GET http://127.0.0.1:8000/external/fetch/nasa`
  - `POST http://127.0.0.1:8000/external/fetch/nasa`
- Pull CERN live event now:
  - Set `CERN_LIVE_URL` in `backend/.env` to a JSON endpoint
  - `GET http://127.0.0.1:8000/external/fetch/cern`
  - `POST http://127.0.0.1:8000/external/fetch/cern`
  - If `CERN_LIVE_URL` is unset, the backend will fall back to `datasets/cern_live_sample.json` for offline demos
- Query stored external events:
  - `GET http://127.0.0.1:8000/external/events?limit=20`
- Use cached research examples when live endpoints are unavailable:
  - `datasets/nasa_apod_sample.json`
  - `datasets/cern_live_sample.json`

## Research extensions
- Promote DuckDB for offline studies in notebooks while keeping PostgreSQL as the serving store.
- Improve the persisted forecasting artifact with deeper sequence models once the current real-data workflow is stable.
- Extend backend persistence with Alembic migrations and Timescale hypertable creation.
