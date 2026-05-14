from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import initialize_database
from app.services.demo_runtime_service import demo_runtime_service
from app.services.stream_service import stream_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    await initialize_database()
    await stream_service.start()
    await demo_runtime_service.start()
    yield
    await demo_runtime_service.stop()
    await stream_service.stop()


app = FastAPI(
    title="Real-Time Time-Series Intelligence Platform",
    description="Scientific ML systems backend for streaming analytics, anomaly detection, and forecasting.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
