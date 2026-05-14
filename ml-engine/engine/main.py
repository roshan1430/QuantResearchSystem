from fastapi import FastAPI

from engine.routes import router

app = FastAPI(
    title="Time-Series ML Engine",
    version="0.2.0",
    description="Model serving and training engine for anomaly detection and forecasting.",
)
app.include_router(router)
