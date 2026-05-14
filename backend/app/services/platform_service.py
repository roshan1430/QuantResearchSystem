import httpx
from sqlalchemy import text

from app.core.config import settings
from app.db.session import SessionLocal
from app.schemas.health import HealthResponse, ReadinessResponse


class PlatformService:
    async def get_health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=settings.app_version,
            database="connected",
            kafka="configured",
            ml_engine="reachable-via-service",
        )

    async def get_readiness(self) -> ReadinessResponse:
        database_ok = await self._check_database()
        ml_engine_ok = await self._check_ml_engine()
        ready = database_ok and ml_engine_ok
        return ReadinessResponse(
            ready=ready,
            database_ok=database_ok,
            ml_engine_ok=ml_engine_ok,
            details={
                "database": "ok" if database_ok else "unreachable",
                "ml_engine": "ok" if ml_engine_ok else "unreachable",
                "mode": "local-demo" if settings.enable_demo_stream else "streaming",
            },
        )

    async def _check_database(self) -> bool:
        try:
            async with SessionLocal() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def _check_ml_engine(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=settings.ml_engine_url, timeout=5.0) as client:
                response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False


platform_service = PlatformService()
