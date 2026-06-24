#python app/services/frame_store.py
"""
Thread-safe per-camera frame store.

- Worker writes frames via `set_frame(camera_id, frame)` (makes an internal copy).
- Stream endpoint reads frames via `get_frame(camera_id, wait_for_seq, timeout)`.
- `get_seq(camera_id)` returns current sequence number so consumers can wait for new frames.

This module uses threading.Condition for efficient wait/notify and avoids opening the camera
from the stream endpoint (frames are produced by the existing worker).
"""
from typing import Optional, Dict, Tuple
import threading
import copy
import time
import numpy as np

class _FrameEntry:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.frame: Optional[np.ndarray] = None
        self.seq: int = 0
        self.last_update: Optional[float] = None

_frames: Dict[str, _FrameEntry] = {}
_frames_lock = threading.Lock()

def _ensure_entry(camera_id: str) -> _FrameEntry:
    with _frames_lock:
        e = _frames.get(camera_id)
        if e is None:
            e = _FrameEntry()
            _frames[camera_id] = e
        return e

def set_frame(camera_id: str, frame: np.ndarray) -> int:
    """
    Store a copy of frame for camera_id and notify waiting consumers.
    Returns new sequence number.
    """
    if frame is None:
        raise ValueError("frame must be a numpy array")
    e = _ensure_entry(camera_id)
    with e.lock:
        # copy frame to avoid shared-memory issues
        e.frame = copy.deepcopy(frame)
        e.seq += 1
        e.last_update = time.time()
        e.cond.notify_all()
        return e.seq

def get_seq(camera_id: str) -> int:
    e = _ensure_entry(camera_id)
    with e.lock:
        return e.seq

def get_frame(camera_id: str, wait_for_seq: Optional[int] = None, timeout: Optional[float] = None) -> Optional[Tuple[Optional[np.ndarray], int]]:
    """
    If wait_for_seq is None: returns latest frame and its seq immediately (may be (None, seq=0)).
    If wait_for_seq is provided: block until seq > wait_for_seq or timeout; returns (frame, seq) or (None, seq)
    """
    e = _ensure_entry(camera_id)
    with e.lock:
        if wait_for_seq is None:
            return (copy.deepcopy(e.frame) if e.frame is not None else None, e.seq)
        # wait until a new seq is available
        if e.seq > wait_for_seq:
            return (copy.deepcopy(e.frame) if e.frame is not None else None, e.seq)
        # wait with timeout
        waited = e.cond.wait_for(lambda: e.seq > wait_for_seq, timeout=timeout)
        return (copy.deepcopy(e.frame) if e.frame is not None else None, e.seq)

def get_last_update(camera_id: str) -> Optional[float]:
    e = _ensure_entry(camera_id)
    with e.lock:
        return e.last_update