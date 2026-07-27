"""
SafeSite AI — FastAPI Main Entry Point
Includes all existing + new dashboard API routers.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SafeSite AI — PPE Detection API",
    description="Enterprise-ready PPE Detection API with YOLOv8 + Dashboard backend.",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# ── CORS — allow Next.js frontend ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # needed for CSV/PDF downloads
)

# ── Static Files ──────────────────────────────────────────────────────────────
os.makedirs("evidence", exist_ok=True)
os.makedirs("data/personnel_photos", exist_ok=True)

app.mount("/evidence", StaticFiles(directory="evidence"), name="evidence")
app.mount("/data",     StaticFiles(directory="data"),     name="data")

# ── Existing Routers (preserved) ─────────────────────────────────────────────
from app.routers import health, cameras, detections, stream, ws_detections, runner, alerts

app.include_router(health.router)
app.include_router(alerts.router)
app.include_router(runner.router)

# Keep legacy stream endpoint
@app.get("/api/stream/{camera_id}", tags=["Stream (Legacy)"])
async def api_stream_camera(camera_id: str, request: Request, fps: int = None):
    from app.routers.stream import stream_camera
    return await stream_camera(camera_id, request, fps)

app.include_router(stream.router)
app.include_router(ws_detections.router)
app.include_router(detections.router)

# ── NEW Dashboard Routers ─────────────────────────────────────────────────────
from app.routers.auth          import router as auth_router
from app.routers.cameras_v2    import router as cameras_v2_router
from app.routers.violations_v2 import router as violations_router
from app.routers.analytics     import router as analytics_router
from app.routers.personnel     import router as personnel_router, access_router
from app.routers.users         import router as users_router
from app.routers.sites         import router as sites_router

app.include_router(auth_router)
app.include_router(cameras_v2_router)
app.include_router(violations_router)
app.include_router(analytics_router)
app.include_router(personnel_router)
app.include_router(access_router)
app.include_router(users_router)
app.include_router(sites_router)

# Legacy camera connect endpoint
from app.schemas import CameraRegisterRequest, CameraConnectRequest, CameraResponse
from app.services.camera_service import register_camera

@app.post("/api/camera/connect", response_model=CameraResponse, tags=["Cameras (Legacy)"])
def api_camera_connect(req: CameraConnectRequest):
    try:
        from app.services.camera_service import get_camera
        from app.services.ppe_stream_worker import start_ppe_stream
        existing = get_camera(req.camera_id)
        if existing:
            start_ppe_stream(req.camera_id, req.source, fps=15)
            reg_at = existing["registered_at"]
            return CameraResponse(
                camera_id=existing["camera_id"],
                rtsp_url=existing["rtsp_url"],
                fps=existing["fps"],
                registered_at=reg_at.isoformat() if hasattr(reg_at, 'isoformat') else str(reg_at)
            )
        register_req = CameraRegisterRequest(camera_id=req.camera_id, rtsp_url=req.source, fps=15)
        rec = register_camera(register_req)
        start_ppe_stream(req.camera_id, req.source, fps=15)
        reg_at = rec["registered_at"]
        return CameraResponse(
            camera_id=rec["camera_id"], rtsp_url=rec["rtsp_url"], fps=rec["fps"],
            registered_at=reg_at.isoformat() if hasattr(reg_at, 'isoformat') else str(reg_at)
        )
    except Exception as e:
        from fastapi import HTTPException
        logger.exception("Failed to connect camera")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy HTML dashboard
@app.get("/alerts", response_class=HTMLResponse, include_in_schema=False)
async def read_alerts_dashboard():
    try:
        with open("templates/alerts.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: templates/alerts.html not found</h1>"

# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Starting SafeSite AI API v2.0...")

    # Run DB migrations
    try:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations applied successfully")
    except Exception as e:
        logger.warning(f"Alembic migration skipped (DB may not be configured yet): {e}")

    # GPU detection
    try:
        from app.utils.gpu_utils import get_cuda_info, optimize_for_inference
        gpu_info = get_cuda_info()
        if gpu_info["cuda_available"]:
            logger.info(f"GPU acceleration enabled with {gpu_info['device_count']} device(s)")
            optimize_for_inference()
        else:
            logger.warning("No GPU detected, using CPU inference")
    except Exception:
        logger.warning("GPU detection skipped")

    # Background worker
    try:
        from app.services.worker import worker
        worker.start()
    except Exception:
        logger.warning("Background worker not started")

    logger.info("SafeSite AI API started successfully")


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down SafeSite AI API...")
    try:
        from app.services.worker import worker
        worker.stop()
    except Exception:
        pass
    logger.info("SafeSite AI API stopped")
