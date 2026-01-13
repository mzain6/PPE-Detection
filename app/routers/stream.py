from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional
import asyncio
import cv2

from ..services.frame_store import get_frame, get_seq
from ..services.camera_service import get_camera
from ..config import settings

router = APIRouter(prefix="/stream", tags=["Stream"])

BOUNDARY = b"--frame"
CRLF = b"\r\n"
CONTENT_TYPE = b"Content-Type: image/jpeg"

def _encode_jpeg(frame) -> bytes:
    """
    Synchronous JPEG encoding. This will be executed in a threadpool via run_in_executor.
    """
    ret, buf = cv2.imencode(".jpg", frame)
    if not ret:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()

async def _wait_for_frame(camera_id: str, seq: int, timeout: float) -> tuple:
    """
    Helper to call blocking get_frame in executor.
    Returns (frame, new_seq)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_frame, camera_id, seq, timeout)

async def _encode_jpeg_async(frame) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_jpeg, frame)

async def mjpeg_stream_generator(camera_id: str, request: Request, out_fps: Optional[int]) -> AsyncGenerator[bytes, None]:
    """
    Async generator producing multipart MJPEG frames.
    - Uses frame_store.get_frame (blocking) via run_in_executor to avoid blocking the event loop.
    - Throttles outgoing frames to out_fps (if provided) or camera configured fps.
    - Stops when client disconnects.
    """
    cam = get_camera(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="camera not registered")

    # determine outgoing fps (throttle). Use provided fps or camera fps or default.
    target_fps = int(out_fps) if out_fps and int(out_fps) > 0 else int(cam.get("fps", settings.default_fps))
    interval = max(0.01, 1.0 / float(target_fps))

    # starting sequence
    loop = asyncio.get_running_loop()
    seq = await loop.run_in_executor(None, get_seq, camera_id)

    while True:
        # client disconnected?
        if await request.is_disconnected():
            break

        # wait for a new frame (block in threadpool)
        # use timeout slightly larger than interval to wake periodically
        timeout = max(0.1, interval * 1.5)
        item = await _wait_for_frame(camera_id, seq, timeout)
        if item is None:
            # nothing available, yield nothing and continue
            await asyncio.sleep(0.01)
            continue
        frame, seq = item
        if frame is None:
            # timeout/no new frame - continue and check disconnect
            await asyncio.sleep(0.01)
            continue

        try:
            jpg = await _encode_jpeg_async(frame)
        except Exception:
            # skip this frame on encode error
            await asyncio.sleep(0.01)
            continue

        # build multipart chunk
        part = BOUNDARY + CRLF
        part += CONTENT_TYPE + CRLF + CRLF
        part += jpg + CRLF

        yield part

        # throttle outgoing frames
        await asyncio.sleep(interval)


@router.get("/{camera_id}")
async def stream_camera(camera_id: str, request: Request, fps: Optional[int] = None):
    """
    MJPEG stream endpoint:
    - Query param `fps` throttles outgoing frames (e.g. /stream/cam-1?fps=5).
    - This endpoint does NOT open the camera; it consumes frames produced by the worker.
    - Uses a non-blocking async generator backed by run_in_executor for blocking calls.
    """
    cam = get_camera(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="camera not registered")

    gen = mjpeg_stream_generator(camera_id, request, fps)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")