from fastapi import APIRouter, HTTPException

from app.schemas.external import ExternalEventResponse, ExternalFetchResponse, ExternalSourceDescriptor
from app.services.external_data_service import external_data_service

router = APIRouter(prefix="/external")


@router.get("/sources", response_model=list[ExternalSourceDescriptor])
async def get_sources() -> list[ExternalSourceDescriptor]:
    return await external_data_service.list_sources()


@router.post("/fetch/{source_name}", response_model=ExternalFetchResponse)
async def fetch_source(source_name: str) -> ExternalFetchResponse:
    try:
        return await external_data_service.fetch_now(source_name.lower())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch {source_name}: {exc}") from exc


@router.get("/fetch/{source_name}", response_model=ExternalFetchResponse)
async def fetch_source_get(source_name: str) -> ExternalFetchResponse:
    """Convenience GET handler so endpoint can be tested directly in browser."""
    return await fetch_source(source_name)


@router.get("/events", response_model=list[ExternalEventResponse])
async def get_events(source_name: str | None = None, limit: int = 50) -> list[ExternalEventResponse]:
    return await external_data_service.list_events(source_name=source_name, limit=limit)
