#python app/ai/pipeline.py
"""
Compatibility facade: keep `AIPipeline` API for existing code but delegate to PipelineManager.
This allows existing routes and worker code calling AIPipeline().process_frame(...) to continue working.
"""
from typing import Any, Dict, Optional, Tuple
import asyncio
import logging

from ..config import settings
from ..services.pipeline_manager import pipeline_manager

logger = logging.getLogger(__name__)

class AIPipeline:
    def __init__(self):
        # Minimal init; heavy models are managed by PipelineManager per camera
        self.default_fps = int(getattr(settings, "default_fps", 5))

    def process_frame(self, frame, camera_id: str = "unknown", roi: Optional[Tuple[float,float,float,float]] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper that delegates to PipelineManager.infer_sync (blocking).
        Keeps the same signature as previous AIPipeline.process_frame.
        """
        try:
            res = pipeline_manager.infer_sync(camera_id, frame)
            return res
        except Exception as e:
            logger.exception("AIPipeline.process_frame failed for %s: %s", camera_id, e)
            raise

# Backwards compatibility: instantiate when module imported
def get_pipeline():
    return AIPipeline()