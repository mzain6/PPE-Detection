#python app/routers/detections.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from ..schemas import DetectionsResponse
from ..ai.pipeline import AIPipeline
from ..services.camera_service import ensure_stream, register_webcam, save_detection, get_last_detection
from ..services.detection_store import detection_store
from ..config import settings
from datetime import datetime
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detections", tags=["Detection", "PPE", "Fall"])

pipeline = AIPipeline()

@router.post("/frame/{camera_id}", response_model=DetectionsResponse)
async def detect_frame(camera_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="invalid image")
    try:
        res = pipeline.process_frame(frame, camera_id=camera_id, roi=None)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    detection_store.mark_started(camera_id)
    save_detection(camera_id, res)
    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }

@router.post("/capture_rtsp/{camera_id}", response_model=DetectionsResponse)
def capture_rtsp(camera_id: str):
    try:
        stream = ensure_stream(camera_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="camera not registered")
    frame = stream.read(timeout=5.0)
    if frame is None:
        raise HTTPException(status_code=503, detail="no frame from camera")
    try:
        res = pipeline.process_frame(frame, camera_id=camera_id, roi=None)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    detection_store.mark_started(camera_id)
    save_detection(camera_id, res)
    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }

@router.post("/detect/webcam", response_model=DetectionsResponse)
def detect_webcam(device_index: Optional[int] = 0):
    cam_id = f"webcam-{device_index}"
    try:
        register_webcam(cam_id, device_index=int(device_index), fps=settings.default_fps)
    except ValueError:
        pass
    try:
        stream = ensure_stream(cam_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="webcam not registered")
    frame = None
    for _ in range(3):
        frame = stream.read(timeout=1.0)
        if frame is not None:
            break
    if frame is None:
        logger.warning("No frame from webcam %s", cam_id)
        raise HTTPException(status_code=503, detail="no frame from webcam")
    try:
        res = pipeline.process_frame(frame, camera_id=cam_id, roi=None)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    detection_store.mark_started(cam_id)
    save_detection(cam_id, res)
    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }

@router.get("/latest/{camera_id}", response_model=DetectionsResponse)
@router.get("/detect/latest/{camera_id}", response_model=DetectionsResponse)
def latest(camera_id: str):
    # If detection not started return 404 (frontend must start detection explicitly)
    if not detection_store.has_started(camera_id):
        raise HTTPException(status_code=404, detail="detection not started for camera")
    res = get_last_detection(camera_id)
    if res is None:
        # return empty but valid payload (no tracks)
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