from fastapi import APIRouter

from app.api.routes import external, health, models, stream, timeseries, training, websocket

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(timeseries.router, tags=["timeseries"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(training.router, tags=["training"])
api_router.include_router(stream.router, tags=["stream"])
api_router.include_router(websocket.router, tags=["websocket"])
api_router.include_router(external.router, tags=["external"])
