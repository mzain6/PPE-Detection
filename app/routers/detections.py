
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from ..schemas import DetectionsResponse
from ..services.pipeline_manager import pipeline_manager
from ..services.camera_service import ensure_stream, register_webcam
from ..services.detection_store import detection_store
from ..config import settings
from datetime import datetime
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detections", tags=["Detection", "PPE", "Fall"])

@router.post("/frame/{camera_id}", response_model=DetectionsResponse)
async def detect_frame(camera_id: str, file: UploadFile = File(...)):
    """
    Accept uploaded image and run detection (QA). This starts detection for camera_id.
    Uses async pipeline_manager.infer_and_store.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="invalid image")
    try:
        res = await pipeline_manager.infer_and_store(camera_id, frame)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }

@router.post("/capture_rtsp/{camera_id}", response_model=DetectionsResponse)
async def capture_rtsp(camera_id: str):
    """
    Grab a frame from the registered RTSP camera and run detection.
    Uses async pipeline_manager.infer_and_store.
    """
    try:
        stream = ensure_stream(camera_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="camera not registered")
    frame = stream.read(timeout=5.0)
    if frame is None:
        raise HTTPException(status_code=503, detail="no frame from camera")
    try:
        res = await pipeline_manager.infer_and_store(camera_id, frame)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }

@router.post("/detect/webcam", response_model=DetectionsResponse)
async def detect_webcam(device_index: Optional[int] = 0):
    """
    Start detection on local webcam and process a single frame.
    Registers the webcam if needed, runs inference and persists result.
    Uses async pipeline_manager.infer_and_store.
    """
    cam_id = f"webcam-{device_index}"
    try:
        register_webcam(cam_id, device_index=int(device_index), fps=settings.default_fps)
    except ValueError:
        # already registered — ok
        pass
    try:
        stream = ensure_stream(cam_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="webcam not registered")

    # try to read a frame (give webcam a few attempts)
    frame = None
    for _ in range(3):
        frame = stream.read(timeout=1.0)
        if frame is not None:
            break

    if frame is None:
        logger.warning("No frame from webcam %s", cam_id)
        raise HTTPException(status_code=503, detail="no frame from webcam")

    try:
        res = await pipeline_manager.infer_and_store(cam_id, frame)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }

@router.get("/latest/{camera_id}", response_model=DetectionsResponse)
@router.get("/detect/latest/{camera_id}", response_model=DetectionsResponse)
def latest(camera_id: str):
    """
    Return last detection for camera_id. If detection hasn't been started for this camera, return 404.
    """
    if not detection_store.has_started(camera_id):
        raise HTTPException(status_code=404, detail="detection not started for camera")
    res = detection_store.get(camera_id)
    if res is None:
        # return an empty but valid DetectionsResponse structure (no tracks)
        return {
            "camera_id": camera_id,
            "timestamp": datetime.utcfromtimestamp(datetime.utcnow().timestamp()),
            "fps": settings.default_fps,
            "tracks": []
        }
    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }