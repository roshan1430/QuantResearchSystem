from datetime import datetime

from pydantic import BaseModel


class ExternalSourceDescriptor(BaseModel):
    source_name: str
    status: str
    description: str


class ExternalFetchResponse(BaseModel):
    source_name: str
    event_type: str
    title: str | None = None
    timestamp: datetime
    payload: dict


class ExternalEventResponse(BaseModel):
    id: int
    source_name: str
    event_type: str
    title: str | None = None
    timestamp: datetime
    payload: dict | None = None
