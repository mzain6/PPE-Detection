
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
        import asyncio
        from app.database import DATABASE_URL
        from app.models import Camera
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        from .camera_service import _camera_registry, _video_streams
        from ..utils.video_stream import VideoStream

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create a dedicated async engine and session pool for this thread
        worker_engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
        WorkerSessionLocal = sessionmaker(worker_engine, class_=AsyncSession, expire_on_commit=False)
        
        while not self._stop.is_set():
            # Fetch active cameras from DB
            async def _fetch_cams():
                async with WorkerSessionLocal() as db:
                    res = await db.execute(select(Camera).where(Camera.is_active == True))
                    return res.scalars().all()
            
            db_cams = loop.run_until_complete(_fetch_cams())
            
            # Sync DB cameras to registry
            for c in db_cams:
                cid = str(c.id)
                if cid not in _camera_registry:
                    src = c.stream_url
                    source_type = "WEBCAM" if src.isdigit() else "RTSP"
                    source_val = int(src) if src.isdigit() else str(src)
                    
                    _camera_registry[cid] = {
                        "camera_id": cid,
                        "rtsp_url": src,
                        "source": source_val,
                        "source_type": source_type,
                        "fps": c.fps_target,
                        "registered_at": c.created_at,
                        "roi": None,
                        "last_frame_time": None
                    }
                    if source_type == "WEBCAM":
                        _video_streams[cid] = VideoStream.from_webcam(source_val, fps=c.fps_target)
                    else:
                        _video_streams[cid] = VideoStream.from_rtsp(source_val, fps=c.fps_target)
            
            cams = list_cameras()
            for cam in cams:
                cam_id = cam.get("camera_id")
                if not cam_id:
                    continue
                
                # Check if it was removed or deactivated
                if not any(str(c.id) == cam_id for c in db_cams):
                    if cam_id in _video_streams:
                        _video_streams[cam_id].release()
                        del _video_streams[cam_id]
                    if cam_id in _camera_registry:
                        del _camera_registry[cam_id]
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

                    # run inference synchronously using pipeline_manager.infer_sync
                    try:
                        res = self.pipeline_manager.infer_sync(cam_id, frame)
                        
                        # Draw bounding boxes on the frame for the MJPEG stream
                        import cv2
                        for track in res.get("tracks", []):
                            # Draw person box
                            pb = track.get("bbox", {})
                            if pb:
                                px1, py1, px2, py2 = int(pb.get("x1",0)), int(pb.get("y1",0)), int(pb.get("x2",0)), int(pb.get("y2",0))
                                color = (0, 0, 255) if track.get("fall") else (0, 255, 0)
                                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                            
                            # Draw PPE boxes
                            for ppe in track.get("ppe", []):
                                ppeb = ppe.get("bbox", [0,0,0,0])
                                if len(ppeb) == 4:
                                    cx1, cy1, cx2, cy2 = int(ppeb[0]), int(ppeb[1]), int(ppeb[2]), int(ppeb[3])
                                    lbl = ppe.get("label", "")
                                    conf = ppe.get("confidence", 0.0)
                                    ppe_color = (0, 255, 0) if lbl == "Hardhat" or lbl == "Safety Vest" else (0, 0, 255)
                                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), ppe_color, 2)
                                    cv2.putText(frame, f"{lbl} ({conf:.0%})", (cx1, cy2 + 16),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ppe_color, 1)

                    except Exception:
                        logger.exception("Inference failed for %s", cam_id)
                        continue

                    # publish annotated frame for streaming consumers
                    try:
                        set_frame(cam_id, frame)
                    except Exception:
                        logger.exception("Failed to set frame for %s", cam_id)

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