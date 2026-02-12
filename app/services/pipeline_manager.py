
"""
Pipeline manager for orchestrating per-camera detectors and inference execution.

- GPU-first device selection.
- ThreadPoolExecutor for blocking detector.infer calls.
- Exposes both async `infer_and_store` and sync `infer_sync` to support sync routes.
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional
import logging

from ..services.detection_store import detection_store
from ..services.camera_service import get_camera
from ..config import settings

# Import detector implementation from app.ai (single-file detector location in this repo)
from ..ai.yolov8_detector import YoloV8Detector

logger = logging.getLogger(__name__)

class PipelineManager:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._detectors: Dict[str, YoloV8Detector] = {}
        self.default_device = self._select_device()

    def _select_device(self) -> Optional[str]:
        try:
            import torch
            if torch.cuda.is_available():
                return "0"  # ultralytics accepts "0" or "cuda:0"
        except Exception:
            pass
        return None

    def _create_detector(self, camera_id: str) -> YoloV8Detector:
        cam = get_camera(camera_id)
        device = self.default_device
        model_path = settings.model_path
        det = YoloV8Detector(model_path=model_path, device=device, camera_id=camera_id)
        return det

    def get_detector(self, camera_id: str) -> YoloV8Detector:
        det = self._detectors.get(camera_id)
        if det is None:
            det = self._create_detector(camera_id)
            self._detectors[camera_id] = det
            logger.info("PipelineManager created detector for %s (device=%s)", camera_id, self.default_device)
        return det

    async def infer_and_store(self, camera_id: str, frame) -> Dict:
        """
        Async wrapper: runs detector.infer in executor and saves to detection_store.
        """
        loop = asyncio.get_running_loop()
        det = self.get_detector(camera_id)
        res = await loop.run_in_executor(self.executor, det.infer, frame)
        if "camera_id" not in res:
            res["camera_id"] = camera_id
        detection_store.mark_started(camera_id)
        detection_store.save(camera_id, res)
        return res

    def infer_sync(self, camera_id: str, frame, timeout: Optional[float] = 15.0) -> Dict:
        """
        Synchronous inference for blocking routes/workers.
        """
        det = self.get_detector(camera_id)
        future: Future = self.executor.submit(det.infer, frame)
        res = future.result(timeout=timeout)
        if "camera_id" not in res:
            res["camera_id"] = camera_id
        detection_store.mark_started(camera_id)
        detection_store.save(camera_id, res)
        return res

    def shutdown(self):
        for det in self._detectors.values():
            try:
                det.close()
            except Exception:
                logger.exception("Error closing detector")
        self.executor.shutdown(wait=False)

# singleton instance
pipeline_manager = PipelineManager(max_workers=max(2, int(getattr(settings, "max_workers", 4))))