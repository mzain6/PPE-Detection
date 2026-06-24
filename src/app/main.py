from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.routers import cameras, detections, stream, health, alerts
from app.routers import ws_detections, runner
from app.services.worker import worker
from app.utils.gpu_utils import get_cuda_info, optimize_for_inference
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PPE Detection API",
    description="Enterprise-ready PPE Detection API with YOLOv8 for helmet and vest detection.",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os

# Create evidence directory if it doesn't exist
os.makedirs("evidence", exist_ok=True)

# Mount evidence directory for static file access
app.mount("/evidence", StaticFiles(directory="evidence"), name="evidence")

from app.schemas import CameraRegisterRequest, CameraConnectRequest, CameraResponse
from app.services.camera_service import register_camera

app.include_router(health.router)
app.include_router(cameras.router)

@app.post("/api/camera/connect", response_model=CameraResponse, tags=["Cameras"])
def api_camera_connect(req: CameraConnectRequest):
    try:
        from app.services.camera_service import get_camera, reset_stream
        from app.services.ppe_stream_worker import start_ppe_stream
        # Check if already registered
        existing = get_camera(req.camera_id)
        if existing:
            # Restart the annotated detection stream for this camera
            start_ppe_stream(req.camera_id, req.source, fps=15)
            reg_at = existing["registered_at"]
            return CameraResponse(
                camera_id=existing["camera_id"], 
                rtsp_url=existing["rtsp_url"], 
                fps=existing["fps"], 
                registered_at=reg_at.isoformat() if hasattr(reg_at, 'isoformat') else str(reg_at)
            )
            
        register_req = CameraRegisterRequest(
            camera_id=req.camera_id,
            rtsp_url=req.source,
            fps=15
        )
        rec = register_camera(register_req)
        # Start the annotated detection stream using run_cameras_with_face_tracking.py logic
        start_ppe_stream(req.camera_id, req.source, fps=15)
        reg_at = rec["registered_at"]
        return CameraResponse(camera_id=rec["camera_id"], rtsp_url=rec["rtsp_url"], fps=rec["fps"], registered_at=reg_at.isoformat() if hasattr(reg_at, 'isoformat') else str(reg_at))
    except Exception as e:
        from fastapi import HTTPException
        logger.exception("Failed to connect camera")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stream/{camera_id}", tags=["Stream"])
async def api_stream_camera(camera_id: str, request: Request, fps: int = None):
    from app.routers.stream import stream_camera
    return await stream_camera(camera_id, request, fps)
app.include_router(detections.router)
app.include_router(stream.router)
app.include_router(ws_detections.router)
app.include_router(alerts.router)  # Add alerts router
app.include_router(runner.router)  # Add runner router

@app.on_event("startup")
def startup_event():
    logger.info("Starting PPE Detection API...")
    
    # Log GPU information
    gpu_info = get_cuda_info()
    if gpu_info["cuda_available"]:
        logger.info(f"GPU acceleration enabled with {gpu_info['device_count']} device(s)")
        optimize_for_inference()
    else:
        logger.warning("No GPU detected, using CPU inference")
    
    # Start background worker
    worker.start()
    logger.info("PPE Detection API started successfully")

@app.get("/alerts", response_class=HTMLResponse)
async def read_alerts_dashboard():
    """Serves the alerts dashboard."""
    try:
        with open("templates/alerts.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: templates/alerts.html not found</h1>"

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down PPE Detection API...")
    worker.stop()
    logger.info("PPE Detection API stopped")
