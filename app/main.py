from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.routers import cameras, detections, stream, health, alerts
from app.routers import ws_detections
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

app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(detections.router)
app.include_router(stream.router)
app.include_router(ws_detections.router)
app.include_router(alerts.router)  # Add alerts router

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
