
import threading
import time
import logging
from typing import Optional
from .camera_service import list_cameras, ensure_stream, save_detection, reset_stream, update_last_frame_time
from ..services.pipeline_manager import pipeline_manager
from ..config import settings
from .frame_store import set_frame

logger = logging.getLogger(__name__)

class CameraWorker:
    def __init__(self, poll_interval: float = 1.0):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.poll_interval = poll_interval
        # use pipeline_manager for inference (sync path for worker)
        self.pipeline_manager = pipeline_manager
        self._open_retries = 3

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("CameraWorker started (poll_interval=%s)", self.poll_interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("CameraWorker stopped")

    def _run_loop(self):
        while not self._stop.is_set():
            cams = list_cameras()
            for cam in cams:
                cam_id = cam.get("camera_id")
                if not cam_id:
                    continue
                try:
                    stream = ensure_stream(cam_id)
                    try:
                        stream.start()
                    except Exception:
                        logger.exception("Failed to start stream for %s", cam_id)

                    if not getattr(stream, "is_open", lambda: True)():
                        logger.warning("Stream not open for %s, attempting reset", cam_id)
                        try:
                            stream = reset_stream(cam_id)
                            stream.start()
                        except Exception:
                            logger.exception("Reset failed for %s", cam_id)
                            continue

                    frame = stream.read(timeout=0.5)
                    if frame is None:
                        logger.debug("No frame for %s (source_type=%s)", cam_id, cam.get("source_type"))
                        if cam.get("source_type") == "WEBCAM":
                            for attempt in range(self._open_retries):
                                logger.info("Retrying webcam read for %s attempt=%s", cam_id, attempt+1)
                                try:
                                    stream = reset_stream(cam_id)
                                    stream.start()
                                except Exception:
                                    logger.exception("Error resetting stream for %s", cam_id)
                                    time.sleep(0.2)
                                    continue
                                frame = stream.read(timeout=0.5)
                                if frame is not None:
                                    break
                        if frame is None:
                            continue

                    # publish frame for streaming consumers
                    try:
                        set_frame(cam_id, frame)
                    except Exception:
                        logger.exception("Failed to set frame for %s", cam_id)

                    # run inference synchronously using pipeline_manager.infer_sync
                    try:
                        res = self.pipeline_manager.infer_sync(cam_id, frame)
                    except Exception:
                        logger.exception("Inference failed for %s", cam_id)
                        continue

                    # update last frame time and persist already done by pipeline_manager via detection_store
                    update_last_frame_time(cam_id)
                    logger.info("Processed frame for %s source_type=%s fps=%s tracks=%s",
                                cam_id, cam.get("source_type"), cam.get("fps"), len(res.get("tracks", [])))
                except Exception:
                    logger.exception("Unhandled error processing camera %s", cam.get("camera_id"))
                    continue
            time.sleep(self.poll_interval)

# singleton worker instance
worker = CameraWorker(poll_interval=max(0.5, 1.0 / settings.default_fps))