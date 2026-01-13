# app/services/worker.py
# Modified to publish frames into frame_store so streaming endpoints can reuse them.
import threading
import time
import logging
from typing import Optional
from .camera_service import list_cameras, ensure_stream, save_detection, reset_stream, get_camera, update_last_frame_time
from ..ai.pipeline import AIPipeline
from ..config import settings
from .frame_store import set_frame

logger = logging.getLogger(__name__)

class CameraWorker:
    def __init__(self, poll_interval: float = 1.0):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.poll_interval = poll_interval
        self.pipeline = AIPipeline()
        # read retries for webcam recovery
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
                    # ensure stream exists
                    stream = ensure_stream(cam_id)
                    # explicitly start capture
                    try:
                        stream.start()
                    except Exception:
                        logger.exception("Failed to start stream for %s", cam_id)

                    # if not open, try resetting (useful for webcams)
                    if not getattr(stream, "is_open", lambda: True)():
                        logger.warning("Stream not open for %s, attempting reset", cam_id)
                        try:
                            stream = reset_stream(cam_id)
                            stream.start()
                        except Exception:
                            logger.exception("Reset failed for %s", cam_id)
                            continue

                    # read a frame (short timeout)
                    frame = stream.read(timeout=0.5)
                    if frame is None:
                        logger.debug("No frame for %s (source_type=%s)", cam_id, cam.get("source_type"))
                        # if webcam, attempt a few quick retries and recreate stream if necessary
                        if cam.get("source_type") == "WEBCAM":
                            for attempt in range(self._open_retries):
                                logger.info("Retrying webcam read for %s attempt=%s", cam_id, attempt+1)
                                # recreate stream and try again
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

                    # publish frame to frame_store for streaming consumers
                    try:
                        set_frame(cam_id, frame)
                    except Exception:
                        logger.exception("Failed to set frame for %s", cam_id)

                    # run inference
                    try:
                        res = self.pipeline.process_frame(frame, camera_id=cam_id, roi=cam.get("roi"))
                    except Exception:
                        logger.exception("Inference failed for %s", cam_id)
                        continue

                    # persist last detection and update last frame time
                    save_detection(cam_id, res)
                    update_last_frame_time(cam_id)
                    logger.info("Processed frame for %s source_type=%s fps=%s tracks=%s",
                                cam_id, cam.get("source_type"), cam.get("fps"), len(res.get("tracks", [])))
                except Exception:
                    # swallow per-camera errors; keep worker alive
                    logger.exception("Unhandled error processing camera %s", cam.get("camera_id"))
                    continue
            time.sleep(self.poll_interval)

# singleton worker instance created elsewhere (import worker and call worker.start())
worker = CameraWorker(poll_interval=max(0.5, 1.0 / settings.default_fps))