
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Any
import asyncio
from ..services.detection_store import detection_store
from ..services.camera_service import get_camera, get_last_detection
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["WS"])

@router.websocket("/detections/{camera_id}")
async def ws_detections(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        # refuse if detection not started
        if not detection_store.has_started(camera_id):
            await websocket.send_text(json.dumps({"error":"detection not started"}))
            await websocket.close()
            return
        last = None
        while True:
            await asyncio.sleep(1.0)  # push interval
            det = get_last_detection(camera_id)
            if det is None:
                det = {"camera_id": camera_id, "timestamp": 0.0, "fps": 0, "tracks": []}
            # send only when changed (simple diff by timestamp)
            if last is None or det.get("timestamp") != last.get("timestamp"):
                await websocket.send_text(json.dumps(det))
                last = det
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", camera_id)
    except Exception:
        logger.exception("WebSocket error for %s", camera_id)
        try:
            await websocket.close()
        except Exception:
            pass