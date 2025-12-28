from typing import Dict, List, Optional
import io
import time
import json
import base64

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from ppedetector import PPEDetector
from camera import CameraSource

# FastAPI app
app = FastAPI(title="PPE Detection API")

# Allow cross-origin for easy integration in dev (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Application-wide singletons will be created on startup
app.state.detector: Optional[PPEDetector] = None
app.state.cameras: Dict[str, CameraSource] = {}


# Utility: get or create CameraSource for a given source string
def _parse_source(source: str):
    """Convert numeric strings to int for USB cameras, otherwise return string."""
    try:
        return int(source)
    except Exception:
        return source


def _get_camera(source_key: str, reconnect_delay: float = 2.0) -> CameraSource:
    """
    Return an existing CameraSource or create one.
    source_key should be the original string representation (e.g., "0" or "rtsp://...")
    """
    cam = app.state.cameras.get(source_key)
    if cam is None:
        src = _parse_source(source_key)
        cam = CameraSource(source=src, reconnect_delay_sec=reconnect_delay)
        app.state.cameras[source_key] = cam
    return cam


def _frame_to_jpeg_bytes(frame) -> bytes:
    """Encode BGR frame to JPEG bytes."""
    ret, buf = cv2.imencode(".jpg", frame)
    if not ret:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buf.tobytes()


@app.on_event("startup")
def startup_event():
    """
    Initialize PPEDetector once on app startup.
    - model_path: change if your weights are elsewhere.
    - device: 'auto' will pick GPU if available, otherwise CPU.
    """
    # Load detector singleton
    app.state.detector = PPEDetector(model_path="best.pt", device="auto", conf_thres=0.25)
    # Cameras dict is already initialized in state
    app.state.cameras = {}


@app.on_event("shutdown")
def shutdown_event():
    # Release camera resources and detector on shutdown
    for cam in app.state.cameras.values():
        try:
            cam.release()
        except Exception:
            pass
    app.state.cameras.clear()
    if app.state.detector:
        try:
            app.state.detector.release()
        except Exception:
            pass
        app.state.detector = None


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.get("/detect")
def detect_once(
    source: str = Query("0", description="Camera source: '0' for USB index 0 or RTSP URL"),
    annotate: bool = Query(False, description="If true, returns base64 annotated_frame in response"),
):
    """
    Capture one frame from the given camera source, run PPE detection, and return structured JSON.
    - annotate=True will attach a base64-encoded annotated JPEG in the response (useful for debugging).
    """
    if app.state.detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    source_key = source
    cam = _get_camera(source_key)
    ok, frame = cam.read(reconnect=True)
    if not ok or frame is None:
        raise HTTPException(status_code=504, detail="Failed to read frame from camera")

    # Run detection (annotate flag controls whether detector returns annotated frame)
    out_frame, detections = app.state.detector.detect(frame, annotate=annotate)

    payload = {
        "timestamp": time.time(),
        "source": source,
        "detections": detections,
    }

    if annotate and out_frame is not None:
        try:
            jpg = _frame_to_jpeg_bytes(out_frame)
            payload["annotated_frame_base64"] = base64.b64encode(jpg).decode("utf-8")
        except Exception:
            # If encoding fails, continue without annotated frame
            pass

    return JSONResponse(content=payload)


@app.get("/stream/mjpeg")
def stream_mjpeg(
    source: str = Query("0", description="Camera source: '0' for USB index 0 or RTSP URL"),
    annotate: bool = Query(True, description="If true, stream annotated frames; else stream raw frames"),
    fps: float = Query(15.0, description="Target frames per second for MJPEG stream (best-effort)"),
):
    """
    MJPEG stream endpoint.
    - Streams frames (annotated or raw) as multipart MJPEG.
    - Runs detection per-frame (keeps person_id stable via tracker).
    - Use this endpoint for simple video viewers or debug dashboards.
    """

    if app.state.detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    source_key = source
    cam = _get_camera(source_key)

    boundary = b"--frame"
    target_delay = 1.0 / max(1.0, fps)

    def mjpeg_generator():
        last_frame_time = 0.0
        frame_idx = 0
        for frame in cam.stream_frames():
            frame_idx += 1
            start = time.time()

            # Run detection - keep annotate flag per request
            out_frame, detections = app.state.detector.detect(frame, annotate=annotate)

            # Choose which frame to send (annotate toggle)
            send_frame = out_frame if annotate and out_frame is not None else frame

            try:
                jpg = _frame_to_jpeg_bytes(send_frame)
            except Exception:
                # Skip this frame if encoding fails
                continue

            # Yield multipart chunk
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

            # Throttle loop to target fps to avoid saturating CPU/GPU
            elapsed = time.time() - start
            to_sleep = target_delay - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# Optional: endpoint to list active camera sources
@app.get("/cameras")
def list_cameras():
    return {"active_sources": list(app.state.cameras.keys())}


# Notes for integration (short):
# - The detector keeps person/helmet/vest logic and tracker-based person_id stable.
# - Plug fall detection, face recognition, permit validation, vehicle detection inside PPEDetector.detect()
#   (there are TODO comments in ppedetector.py indicating these integration points).
# - For high-throughput production, run multiple workers and prefer a dedicated GPU server.
# - To run: `uvicorn app_fastapi:app --host 0.0.0.0 --port 8000`
```

What changed and why (concise)
- Added `app_fastapi.py` implementing a FastAPI wrapper that:
  - Instantiates a singleton `PPEDetector` at startup.
  - Manages camera sources via a lightweight cache so multiple endpoints can reuse streams.
  - Exposes `/detect` for single-frame JSON detections (optionally returns annotated frame as base64).
  - Exposes `/stream/mjpeg` for MJPEG streaming with optional annotation toggle.
  - Keeps the inference and I/O code separated so the detector remains reusable for background tasks or other integrations.
- This file expects your existing `ppedetector.py` and `camera.py` to be present (no model changes).

Would you like a small example showing how to secure these endpoints (API key header) or how to deploy with Gunicorn + Uvicorn workers for production?