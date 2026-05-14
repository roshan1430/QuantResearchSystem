# Architecture

```mermaid
flowchart LR
    A[React Dashboard] --> B[FastAPI API Gateway]
    B --> C[WebSocket Broadcast Manager]
    B --> D[PostgreSQL / Timescale Store]
    E[Streaming Producer] --> F[Kafka Raw Topic]
    F --> G[Feature Engineering Engine]
    G --> H[Processed Feature Topic]
    H --> I[ML Inference / Forecasting Engine]
    I --> J[Anomaly Topic]
    H --> B
    J --> B
    I --> K[MLflow Tracking]
```

## Design notes
- The backend is the control plane and observability surface.
- Streaming and ML components are independently deployable workers.
- Data validation is enforced at API boundaries with Pydantic schemas.
- Feature engineering uses vectorized operations to preserve scientific reproducibility.
