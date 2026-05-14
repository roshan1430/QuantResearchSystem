from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websocket.manager import websocket_manager

router = APIRouter()


@router.websocket("/ws/live")
async def live_websocket(websocket: WebSocket) -> None:
    await websocket_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
