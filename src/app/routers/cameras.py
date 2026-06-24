#python app/routers/cameras.py
from fastapi import APIRouter, HTTPException
from typing import Dict
from datetime import datetime
from ..schemas import CameraRegisterRequest, CameraConnectRequest, CameraResponse, HealthResponse
from ..services.camera_service import register_camera, get_camera, health_check, ensure_stream

router = APIRouter(prefix="/cameras", tags=["Cameras"])

@router.post("/register", response_model=CameraResponse)
def register_camera_route(req: CameraRegisterRequest):
    try:
        rec = register_camera(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CameraResponse(camera_id=rec["camera_id"], rtsp_url=rec["rtsp_url"], fps=rec["fps"], registered_at=rec["registered_at"])

@router.post("/connect", response_model=CameraResponse)
def connect_camera_route(req: CameraConnectRequest):
    try:
        # Convert Request to match internal system logic
        register_req = CameraRegisterRequest(
            camera_id=req.camera_id,
            rtsp_url=req.source,
            fps=15
        )
        rec = register_camera(register_req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CameraResponse(camera_id=rec["camera_id"], rtsp_url=rec["rtsp_url"], fps=rec["fps"], registered_at=rec["registered_at"])

@router.get("/{camera_id}/health", response_model=HealthResponse, tags=["Health"])
def camera_health(camera_id: str):
    healthy, last_dt, msg = health_check(camera_id, timeout=2.0)
    return HealthResponse(camera_id=camera_id, healthy=healthy, last_frame_time=last_dt, message=msg)