# Real-Time Time-Series Intelligence Platform

Research-grade monorepo for time-series ingestion, feature engineering, anomaly detection, forecasting, experiment tracking, and scientific dashboards, framed around NASA and CERN-style research telemetry.

## Platform goals
- ingest raw time-series measurements into a streaming backbone
- engineer reproducible statistical features with vectorized pipelines
- serve anomaly detection and forecasting behind a FastAPI gateway
- persist metrics, anomalies, predictions, and experiment metadata
- expose a live Plotly dashboard over REST and WebSockets
- stay modular enough for research iteration and production hardening

## Monorepo structure

```text
cern-quant-ml/
|-- backend/        # FastAPI API gateway, persistence, websockets
|-- frontend/       # React + Vite + Tailwind scientific dashboard
|-- ml-engine/      # model serving and training endpoints
|-- streaming/      # Kafka producer and feature engineering processor
|-- datasets/       # sample training and validation datasets
|-- notebooks/      # offline exploratory analysis
|-- experiments/    # tracked artifacts and experiment manifests
|-- docs/           # architecture, API, deployment notes
|-- deployment/     # startup scripts and developer guides
`-- docker/         # reserved shared container assets
```

## Architecture

```text
Frontend (React + Plotly)
    ->
FastAPI API Gateway
    ->
Kafka-backed streaming layer
    ->
Feature engineering engine
    ->
ML inference / forecasting engine
    ->
PostgreSQL / Timescale-compatible store
```

More detail is in [docs/architecture.md](/d:/Desktop/QuantResearchSystem/docs/architecture.md:1).

## Implemented
- modular FastAPI backend with routers, schemas, service layer, websocket endpoint, async SQLAlchemy session, and environment-driven config
- ML engine service exposing `GET /models`, `POST /predict`, `POST /anomaly`, `POST /forecast`, and `POST /train`
- Kafka streaming producer and feature engineering processor with rolling mean, rolling std, z-score, momentum, lag, volatility, and EMA features
- multi-page React dashboard for dashboard, live stream monitor, anomaly detection, forecasting, model metrics, and system health
- Dockerfiles for each service plus a `docker-compose.yml` stack with PostgreSQL, Kafka, MLflow, backend, ML engine, streaming workers, and frontend
- local demo runtime for live metrics and anomaly events when Docker/Kafka are unavailable
- NASA and CERN-style external source support, a real NASA POWER dataset builder for stronger training data, sample research datasets, deployment notes, API examples, CI workflow, and developer documentation

## API surface
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

## Run locally today

### Install dependencies

```bash
python -m pip install -r backend/requirements.txt
python -m pip install -r ml-engine/requirements.txt
cd frontend
npm install
cd ..
```

### One-command local startup

```bash
python deployment/run_local.py
```

This launches:
- frontend at `http://127.0.0.1:5173`
- backend docs at `http://127.0.0.1:8000/docs`
- ML engine model catalog at `http://127.0.0.1:8001/models`

### Local runtime behavior
- backend defaults to SQLite for local development
- Kafka consumers are disabled locally unless explicitly enabled
- a demo telemetry generator streams live sensor points and anomaly events to the dashboard without Docker

## Docker path

When Docker becomes available:

```bash
docker compose up --build
```

## Research notes
- training uses time-aware validation concepts rather than random train-test splitting
- NASA public metadata feeds and CERN-style JSON events are used to give the platform a more authentic live-research feel
- actual training datasets can now be built from the official NASA POWER hourly API using `datasets/build_research_dataset.py`
- forecasting can now be upgraded into a persisted trained regressor artifact via `deployment/train_research_pipeline.py`
- the anomaly serving stack includes baseline scikit-learn models now, with PyTorch LSTM autoencoder left as the next extension
- PostgreSQL is configured for platform storage, and the compose image is Timescale-compatible for later hypertable promotion

## Validation status
- Python modules in `backend/app`, `ml-engine/engine`, and `streaming/app` compile successfully.
- backend and ML engine imports now work on the local Python 3.14 environment.
- frontend production build passes after the Tailwind/PostCSS fix.

## Next engineering steps
1. add Alembic migrations and explicit Timescale hypertable setup
2. keep iterating on the trained forecasting artifact with larger real-data windows and richer features
3. extend integration coverage from backend-to-ML into Kafka-to-API data flow
4. strengthen anomaly training with a persisted deep sequence model
5. restore optional MLflow tracking for local runs once Docker or pyarrow tooling is available
