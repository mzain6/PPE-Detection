
"""
DetectionStore: in-memory default, with optional Redis-backed implementation.
Select Redis by setting environment variable REDIS_URL (e.g. redis://localhost:6379/0).
"""
from typing import Dict, Optional, Any
import threading
import time
import copy
import os
import json
import logging

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL")

# Attempt to import redis only if REDIS_URL provided
_redis = None
if REDIS_URL:
    try:
        import redis
        _redis = redis
    except Exception:
        logger.warning("REDIS_URL provided but redis package not installed; falling back to in-memory store")
        _redis = None

class InMemoryDetectionStore:
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

class RedisDetectionStore:
    def __init__(self, url: str):
        # redis.StrictRedis is thread-safe for our usage
        self._client = _redis.from_url(url, decode_responses=True)
        # keys:
        # detection:{camera_id} -> JSON string
        # detection_started:{camera_id} -> timestamp
        self._prefix_det = "detection:"
        self._prefix_started = "detection_started:"

    def mark_started(self, camera_id: str) -> None:
        key = self._prefix_started + camera_id
        try:
            # use setnx to preserve first start time
            now = str(time.time())
            self._client.setnx(key, now)
            logger.info("Redis mark_started for %s", camera_id)
        except Exception:
            logger.exception("Redis mark_started failed for %s", camera_id)

    def save(self, camera_id: str, detection: Dict[str, Any]) -> None:
        key = self._prefix_det + camera_id
        try:
            data = json.dumps(detection)
            self._client.set(key, data)
            # ensure started flag exists
            self.mark_started(camera_id)
            logger.debug("Redis saved detection for %s", camera_id)
        except Exception:
            logger.exception("Redis save failed for %s", camera_id)

    def get(self, camera_id: str) -> Optional[Dict[str, Any]]:
        key = self._prefix_det + camera_id
        try:
            data = self._client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception:
            logger.exception("Redis get failed for %s", camera_id)
            return None

    def has_started(self, camera_id: str) -> bool:
        key = self._prefix_started + camera_id
        try:
            return self._client.exists(key) == 1
        except Exception:
            logger.exception("Redis has_started failed for %s", camera_id)
            return False

# factory: choose redis-backed store if available, else in-memory
if REDIS_URL and _redis is not None:
    detection_store = RedisDetectionStore(REDIS_URL)
    logger.info("Using RedisDetectionStore with %s", REDIS_URL)
else:
    detection_store = InMemoryDetectionStore()
    logger.info("Using InMemoryDetectionStore")