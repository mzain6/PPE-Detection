"""
Health check router for system monitoring.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import psutil

from ..services.camera_service import list_cameras, health_check
from ..services.pipeline_manager import pipeline_manager
from ..utils.gpu_utils import get_cuda_info, get_gpu_memory_info

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)


@router.get("/", response_model=Dict[str, Any])
async def system_health():
    """
    Get overall system health status.
    
    Returns comprehensive health information including:
    - API status
    - System resources (CPU, memory)
    - GPU status and memory
    - Model status
    - Camera status
    """
    try:
        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU info
        gpu_info = get_cuda_info()
        gpu_memory = None
        if gpu_info["cuda_available"] and gpu_info["device_count"] > 0:
            gpu_memory = get_gpu_memory_info(0)
        
        # Camera status
        cameras = list_cameras()
        camera_status = {}
        for cam_id in cameras:
            healthy, last_frame, msg = health_check(cam_id, timeout=1.0)
            camera_status[cam_id] = {
                "healthy": healthy,
                "last_frame_age": last_frame,
                "message": msg
            }
        
        # Model status
        model_loaded = len(pipeline_manager._detectors) > 0
        
        health_data = {
            "status": "healthy",
            "api": {
                "running": True,
                "version": "0.1.0"
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / 1024**3
            },
            "gpu": {
                "available": gpu_info["cuda_available"],
                "device_count": gpu_info["device_count"],
                "cuda_version": gpu_info["cuda_version"],
                "memory": gpu_memory
            },
            "model": {
                "loaded": model_loaded,
                "active_detectors": len(pipeline_manager._detectors)
            },
            "cameras": {
                "total": len(cameras),
                "status": camera_status
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/gpu", response_model=Dict[str, Any])
async def gpu_health():
    """Get detailed GPU health and memory information."""
    try:
        gpu_info = get_cuda_info()
        
        if not gpu_info["cuda_available"]:
            return {
                "available": False,
                "message": "CUDA not available on this system"
            }
        
        # Get memory info for all devices
        devices_memory = []
        for i in range(gpu_info["device_count"]):
            mem_info = get_gpu_memory_info(i)
            devices_memory.append({
                "device_id": i,
                "name": gpu_info["devices"][i]["name"],
                "memory": mem_info
            })
        
        return {
            "available": True,
            "cuda_version": gpu_info["cuda_version"],
            "device_count": gpu_info["device_count"],
            "devices": devices_memory
        }
        
    except Exception as e:
        logger.error(f"Error getting GPU health: {e}")
        raise HTTPException(status_code=500, detail=f"GPU health check failed: {str(e)}")


@router.get("/cameras/{camera_id}", response_model=Dict[str, Any])
async def camera_health(camera_id: str):
    """Get health status for a specific camera."""
    try:
        healthy, last_frame, msg = health_check(camera_id, timeout=2.0)
        
        return {
            "camera_id": camera_id,
            "healthy": healthy,
            "last_frame_age_seconds": last_frame,
            "message": msg
        }
        
    except Exception as e:
        logger.error(f"Error checking camera health: {e}")
        raise HTTPException(status_code=500, detail=f"Camera health check failed: {str(e)}")
