#python app/services/detection_store.py
from typing import Dict, Optional, Any
import threading
import time
import copy
import logging

logger = logging.getLogger(__name__)

class DetectionStore:
    """
    Thread-safe store for last detection per camera.
    - mark_started(camera_id): indicates detection lifecycle started
    - save(camera_id, detection): persist last detection (deep copy)
    - get(camera_id): return deep copy of detection or None
    - has_started(camera_id): whether detection has been started
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._started: Dict[str, float] = {}

    def mark_started(self, camera_id: str) -> None:
        with self._lock:
            if camera_id not in self._started:
                self._started[camera_id] = time.time()
                logger.info("Detection started for camera %s", camera_id)

    def save(self, camera_id: str, detection: Dict[str, Any]) -> None:
        with self._lock:
            self._store[camera_id] = copy.deepcopy(detection)
            if camera_id not in self._started:
                self._started[camera_id] = time.time()
            logger.debug("Saved detection for %s (tracks=%d)", camera_id, len(detection.get("tracks", [])))

    def get(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            d = self._store.get(camera_id)
            return copy.deepcopy(d) if d is not None else None

    def has_started(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._started

# singleton instance
detection_store = DetectionStore()