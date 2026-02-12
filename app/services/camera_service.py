
# app/services/camera_service.py
import logging
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime
from ..utils.video_stream import VideoStream
from ..schemas import CameraRegisterRequest
from ..config import settings
from .detection_store import detection_store

logger = logging.getLogger(__name__)

_camera_registry: Dict[str, Dict] = {}
_video_streams: Dict[str, VideoStream] = {}

def _is_webcam_source(src: Union[str, int]) -> Optional[int]:
    if isinstance(src, int):
        return int(src)
    if isinstance(src, str):
        s = src.strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
    return None

def register_camera(req: CameraRegisterRequest) -> Dict:
    if req.camera_id in _camera_registry:
        raise ValueError("camera_id already registered")
    fps = req.fps or settings.default_fps
    src = req.rtsp_url
    webcam_idx = _is_webcam_source(src)
    if webcam_idx is not None:
        source_type = "WEBCAM"
        source_val: Union[int, str] = int(webcam_idx)
    else:
        source_type = "RTSP"
        source_val = str(src)

    record = {
        "camera_id": req.camera_id,
        "rtsp_url": req.rtsp_url,
        "source": source_val,
        "source_type": source_type,
        "fps": fps,
        "registered_at": datetime.utcnow(),
        "roi": req.roi,
        "last_frame_time": None
    }
    _camera_registry[req.camera_id] = record

    if source_type == "WEBCAM":
        _video_streams[req.camera_id] = VideoStream.from_webcam(int(source_val), fps=fps)
    else:
        _video_streams[req.camera_id] = VideoStream.from_rtsp(str(source_val), fps=fps)

    logger.info("Registered camera %s source_type=%s fps=%s", req.camera_id, source_type, fps)
    return record

def get_camera(camera_id: str) -> Optional[Dict]:
    return _camera_registry.get(camera_id)

def list_cameras() -> List[Dict]:
    return list(_camera_registry.values())

def get_stream(camera_id: str) -> Optional[VideoStream]:
    return _video_streams.get(camera_id)

def ensure_stream(camera_id: str) -> VideoStream:
    rec = _camera_registry.get(camera_id)
    if not rec:
        raise KeyError("camera not registered")
    stream = _video_streams.get(camera_id)
    if stream is None:
        if rec.get("source_type") == "WEBCAM":
            stream = VideoStream.from_webcam(int(rec["source"]), fps=rec.get("fps", settings.default_fps))
        else:
            stream = VideoStream.from_rtsp(str(rec["source"]), fps=rec.get("fps", settings.default_fps))
        _video_streams[camera_id] = stream
    return stream

def reset_stream(camera_id: str) -> VideoStream:
    rec = _camera_registry.get(camera_id)
    if not rec:
        raise KeyError("camera not registered")
    existing = _video_streams.get(camera_id)
    try:
        if existing:
            existing.release()
    except Exception:
        logger.exception("Error releasing existing stream for %s", camera_id)
    if rec.get("source_type") == "WEBCAM":
        stream = VideoStream.from_webcam(int(rec["source"]), fps=rec.get("fps", settings.default_fps))
    else:
        stream = VideoStream.from_rtsp(str(rec["source"]), fps=rec.get("fps", settings.default_fps))
    _video_streams[camera_id] = stream
    logger.info("Reset video stream for %s source_type=%s", camera_id, rec.get("source_type"))
    return stream

def health_check(camera_id: str, timeout: float = 2.0) -> Tuple[bool, Optional[datetime], Optional[str]]:
    rec = _camera_registry.get(camera_id)
    if not rec:
        return False, None, "camera not registered"
    stream = ensure_stream(camera_id)
    frame = stream.read(timeout=timeout)
    if frame is not None:
        now = datetime.utcnow()
        rec["last_frame_time"] = now
        logger.info("Health OK for %s (source_type=%s)", camera_id, rec.get("source_type"))
        return True, now, None
    logger.warning("Health check failed (no frame) for %s", camera_id)
    return False, None, "no frame read"

def register_webcam(camera_id: str, device_index: int = 0, fps: int = 5) -> Dict:
    if camera_id in _camera_registry:
        raise ValueError("camera_id already registered")
    record = {
        "camera_id": camera_id,
        "rtsp_url": str(device_index),
        "source": int(device_index),
        "source_type": "WEBCAM",
        "fps": fps,
        "registered_at": datetime.utcnow(),
        "roi": None,
        "last_frame_time": None
    }
    _camera_registry[camera_id] = record
    _video_streams[camera_id] = VideoStream.from_webcam(device_index, fps=fps)
    logger.info("Registered webcam %s index=%s fps=%s", camera_id, device_index, fps)
    return record

def save_detection(camera_id: str, detection: Dict) -> None:
    """
    Persist detection via DetectionStore (thread-safe, worker-safe).
    """
    detection_store.save(camera_id, detection)

def get_last_detection(camera_id: str) -> Optional[Dict]:
    return detection_store.get(camera_id)

def update_last_frame_time(camera_id: str, ts: Optional[datetime] = None) -> None:
    rec = _camera_registry.get(camera_id)
    if not rec:
        return
    rec["last_frame_time"] = ts or datetime.utcnow()