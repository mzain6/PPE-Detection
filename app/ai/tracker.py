"""
RTSP helper utilities for MVP.

Provides:
- `frame_to_bgr(frame_bytes)`: convert uploaded image bytes to BGR numpy image.
- `RTSPReader`: lightweight OpenCV-based RTSP reader with simple sampling and health info.

Notes:
- This is an MVP helper. For robust on-prem deployments replace with an ffmpeg/gstreamer-based reader.
- Keep this module DB-agnostic and synchronous; background workers should call `read()` at configured FPS.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import time
import cv2
import numpy as np

def frame_to_bgr(frame_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convert uploaded image bytes (multipart) to a BGR numpy image.
    Returns None if conversion fails.
    """
    if not frame_bytes:
        return None
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

class RTSPReader:
    """
    Simple RTSP reader using OpenCV VideoCapture.

    Usage:
      reader = RTSPReader(rtsp_url, fps=5)
      reader.start()
      frame = reader.read(timeout=5)
      reader.release()
    """
    def __init__(self, rtsp_url: str, fps: int = 5, backend: int = cv2.CAP_FFMPEG):
        self.rtsp_url = rtsp_url
        self.fps = max(1, int(fps) if fps else 1)
        self.backend = backend
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_read: Optional[float] = None

    def start(self) -> "RTSPReader":
        """
        Initialize VideoCapture if not already started.
        """
        if self.cap is None or not getattr(self.cap, "isOpened", lambda: False)():
            # Attempt to open with specified backend; fallback to default if it fails.
            try:
                self.cap = cv2.VideoCapture(self.rtsp_url, self.backend)
            except Exception:
                self.cap = cv2.VideoCapture(self.rtsp_url)
            # try to set a small buffer size where supported (best-effort)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return self

    def read(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Read one frame from the stream.

        - Attempts reads until a frame is returned or timeout elapses.
        - Returns BGR numpy array or None on failure.
        """
        if self.cap is None:
            self.start()
        if self.cap is None or not self.cap.isOpened():
            return None

        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_read = time.time()
                return frame
            # short sleep to avoid busy loop
            time.sleep(max(0.01, 1.0 / (self.fps * 2)))
        return None

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """
        Release capture resources.
        """
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.cap = None
            self.last_read = None

def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0.0 else 0.0

@dataclass
class Track:
    id: int
    bbox: Tuple[float, float, float, float]
    last_seen: float
    missing_frames: int = 0
    stable_count: int = 0
    person_conf: float = 0.0
    ppe: Optional[List[Dict[str, Any]]] = None

class IOUTracker:
    """
    Lightweight IOU tracker for MVP.

    - update(detections, timestamp) expects detections as:
      [{"bbox": (x1,y1,x2,y2), "conf": float, "ppe": [...]}, ...]
    - returns internal tracks dict keyed by track id.
    """
    def __init__(self, iou_thresh: float = 0.3, max_missing: int = 10, stable_frames: int = 5):
        self.iou_thresh = float(iou_thresh)
        self.max_missing = int(max_missing)
        self.stable_frames = int(stable_frames)
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> Dict[int, Track]:
        if timestamp is None:
            timestamp = time.time()

        unmatched = set(self.tracks.keys())

        for det in detections:
            det_bbox = tuple(float(v) for v in det.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            det_conf = float(det.get("conf", 0.0))
            det_ppe = det.get("ppe", []) or []

            best_tid = None
            best_iou = 0.0
            for tid, tr in self.tracks.items():
                ov = iou(det_bbox, tr.bbox)
                if ov > best_iou:
                    best_iou = ov
                    best_tid = tid

            if best_iou >= self.iou_thresh and best_tid is not None:
                tr = self.tracks[best_tid]
                tr.bbox = det_bbox
                tr.last_seen = timestamp
                tr.missing_frames = 0
                tr.person_conf = det_conf
                tr.ppe = det_ppe if det_ppe else (tr.ppe or [])
                tr.stable_count = tr.stable_count + 1
                unmatched.discard(best_tid) 
            else:
                tid = self.next_id
                self.next_id += 1
                tr = Track(
                    id=tid,
                    bbox=det_bbox,
                    last_seen=timestamp,
                    missing_frames=0,
                    stable_count=1,
                    person_conf=det_conf,
                    ppe=det_ppe
                )
                self.tracks[tid] = tr

        # increment missing_frames for unmatched and remove stale tracks
        for tid in list(unmatched):
            tr = self.tracks.get(tid)
            if tr is None:
                continue
            tr.missing_frames += 1
            if tr.missing_frames > self.max_missing:
                del self.tracks[tid]

        return self.tracks

    def as_list(self) -> List[Track]:
        return list(self.tracks.values())