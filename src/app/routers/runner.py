from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import asyncio
import psutil
import sys

router = APIRouter(prefix="/api/runner", tags=["runner"])

# ── Configuration (must match run_vlc_proxy.py) ───────────────────────────────
VLC_PATH     = r"D:\VLC\vlc.exe"
RTSP_URL     = "rtsp://admin:Test1234@192.168.100.66:554/onvif1"
HTTP_PORT    = 8888
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# ──────────────────────────────────────────────────────────────────────────────

# Globals to track processes
tracker_process = None
proxy_process   = None

class RunnerStatus(BaseModel):
    tracker_running: bool
    proxy_running:   bool
    tracker_pid: Optional[int] = None
    proxy_pid:   Optional[int] = None

def _is_process_running(process):
    if process is None:
        return False
    return process.poll() is None

@router.get("/status", response_model=RunnerStatus)
async def get_status():
    """Check if the tracking and proxy scripts are running."""
    tracker_active = _is_process_running(tracker_process)
    proxy_active   = _is_process_running(proxy_process)
    return RunnerStatus(
        tracker_running=tracker_active,
        proxy_running=proxy_active,
        tracker_pid=tracker_process.pid if tracker_active else None,
        proxy_pid=proxy_process.pid   if proxy_active   else None
    )

@router.post("/start-proxy")
async def start_proxy():
    """Start VLC directly to stream RTSP → MJPEG (no wrapper script)."""
    global proxy_process

    if _is_process_running(proxy_process):
        return {"status": "success", "message": "Proxy is already running."}

    if not os.path.exists(VLC_PATH):
        raise HTTPException(status_code=500, detail=f"VLC not found at {VLC_PATH}")

    try:
        vlc_cmd = [
            VLC_PATH,
            "-I", "dummy",
            "--network-caching=300",
            "--rtsp-tcp",
            "--clock-jitter=0",
            "--clock-synchro=0",
            RTSP_URL,
            f'--sout=#transcode{{vcodec=mjpg,scale=0.5,fps=15,acodec=none}}'
            f':http{{mux=mpjpeg,dst=:{HTTP_PORT}/stream.mjpg}}',
            ":sout-keep"
        ]
        proxy_process = subprocess.Popen(
            vlc_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        return {"status": "success", "message": f"VLC Proxy started (PID {proxy_process.pid})."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-detection")
async def start_detection():
    """Start the main Face Tracking and PPE Detection script."""
    global tracker_process
    
    if _is_process_running(tracker_process):
        return {"status": "success", "message": "Detection script is already running."}
        
    try:
        tracker_process = subprocess.Popen(
            [sys.executable, "run_cameras_with_face_tracking.py"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        return {"status": "success", "message": "Detection script started."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-all")
async def start_all():
    """Starts both the VLC Proxy and the Detection script in the proper order."""
    proxy_res = await start_proxy()
    
    # Wait for VLC to initialize the MJPEG stream before connecting the tracker
    await asyncio.sleep(5)
    
    tracker_res = await start_detection()
    
    return {
        "status": "success", 
        "message": "Both proxy and detection scripts triggered.",
        "details": {
            "proxy": proxy_res["message"],
            "tracker": tracker_res["message"]
        }
    }

def _kill_process_tree(pid):
    """Kills a process and all its children to prevent zombie processes."""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

@router.post("/stop-all")
async def stop_all():
    """Stop all running runner scripts (Proxy and Tracker) and specifically kill VLC."""
    global tracker_process, proxy_process
    killed = []
    
    # 1. Kill Python Tracker
    if _is_process_running(tracker_process):
        _kill_process_tree(tracker_process.pid)
        tracker_process = None
        killed.append("Tracker")
        
    # 2. Kill Python Proxy Runner
    if _is_process_running(proxy_process):
        _kill_process_tree(proxy_process.pid)
        proxy_process = None
        killed.append("Python VLC Proxy Spawner")
        
    # 3. Aggressively kill any orphaned `vlc.exe` instances
    try:
        subprocess.run(["taskkill", "/F", "/IM", "vlc.exe"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        killed.append("Orphaned VLC.exe")
    except Exception:
        pass
        
    if not killed:
        return {"status": "success", "message": "No active scripts to stop."}
        
    return {"status": "success", "message": f"Successfully killed: {', '.join(killed)}"}
