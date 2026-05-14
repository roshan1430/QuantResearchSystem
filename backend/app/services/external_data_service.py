from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import ExternalDataEvent
from app.db.session import SessionLocal
from app.schemas.external import ExternalEventResponse, ExternalFetchResponse, ExternalSourceDescriptor


class ExternalDataService:
    def __init__(self) -> None:
        self._datasets_dir = Path(__file__).resolve().parents[3] / "datasets"
        self.sources = {
            "nasa": ExternalSourceDescriptor(
                source_name="nasa",
                status="active",
                description="NASA public live metadata feed used as a real-world scientific context source.",
            ),
            "cern": ExternalSourceDescriptor(
                source_name="cern",
                status="live-or-cached",
                description="CERN JSON endpoint adapter with cached fallback for detector-style or accelerator telemetry experiments.",
            ),
        }

    async def list_sources(self) -> list[ExternalSourceDescriptor]:
        return list(self.sources.values())

    async def fetch_now(self, source_name: str) -> ExternalFetchResponse:
        if source_name == "nasa":
            event = await self._fetch_nasa()
        elif source_name == "cern":
            event = await self._fetch_cern()
        else:
            raise ValueError(f"Unsupported source: {source_name}")

        await self._persist_event(event)
        return event

    async def list_events(self, source_name: str | None = None, limit: int = 50) -> list[ExternalEventResponse]:
        async with SessionLocal() as session:
            stmt = select(ExternalDataEvent).order_by(desc(ExternalDataEvent.timestamp)).limit(limit)
            if source_name:
                stmt = stmt.where(ExternalDataEvent.source_name == source_name)
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [
            ExternalEventResponse(
                id=row.id,
                source_name=row.source_name,
                event_type=row.event_type,
                title=row.title,
                timestamp=row.timestamp,
                payload=row.payload,
            )
            for row in rows
        ]

    async def _fetch_nasa(self) -> ExternalFetchResponse:
        params = {"api_key": settings.nasa_api_key}
        async with httpx.AsyncClient(timeout=settings.external_timeout_seconds) as client:
            response = await client.get(settings.nasa_live_url, params=params)
            response.raise_for_status()
            payload = response.json()
        return ExternalFetchResponse(
            source_name="nasa",
            event_type="apod",
            title=payload.get("title"),
            timestamp=datetime.now(timezone.utc),
            payload=payload,
        )

    async def _fetch_cern(self) -> ExternalFetchResponse:
        if not settings.cern_live_url:
            return self._load_cached_event("cern_live_sample.json")
        async with httpx.AsyncClient(timeout=settings.external_timeout_seconds) as client:
            response = await client.get(settings.cern_live_url)
            response.raise_for_status()
            payload = response.json()
        title = payload.get("title") if isinstance(payload, dict) else None
        return ExternalFetchResponse(
            source_name="cern",
            event_type="custom_json_feed",
            title=title,
            timestamp=datetime.now(timezone.utc),
            payload=payload if isinstance(payload, dict) else {"data": payload},
        )

    def _load_cached_event(self, filename: str) -> ExternalFetchResponse:
        payload = json.loads((self._datasets_dir / filename).read_text(encoding="utf-8"))
        event_timestamp = payload.get("timestamp")
        return ExternalFetchResponse(
            source_name=payload["source_name"],
            event_type=payload["event_type"],
            title=payload.get("title"),
            timestamp=datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
            if isinstance(event_timestamp, str)
            else datetime.now(timezone.utc),
            payload=payload.get("payload", {}),
        )

    async def _persist_event(self, event: ExternalFetchResponse) -> None:
        async with SessionLocal() as session:
            session.add(
                ExternalDataEvent(
                    source_name=event.source_name,
                    event_type=event.event_type,
                    title=event.title,
                    timestamp=event.timestamp,
                    payload=event.payload,
                )
            )
            await session.commit()


external_data_service = ExternalDataService()
