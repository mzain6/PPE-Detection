#python app/utils/rtsp.py
"""
Legacy compatibility module — re-export VideoStream utilities from app.utils.video_stream.

DO NOT put routers or HTTP logic here. Use app.services.camera_service for camera management.
"""
from .video_stream import VideoStream, frame_to_bgr

__all__ = ["VideoStream", "frame_to_bgr"]