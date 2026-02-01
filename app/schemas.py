from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime

class BoundingBox(BaseModel):
    # Depending on format, this could be a list or a struct. 
    # Current main.py returns list [x1, y1, x2, y2]
    pass

class DetectionItem(BaseModel):
    person_id: int
    helmet_status: str = Field(..., description="'yes' or 'no'")
    vest_status: str = Field(..., description="'yes' or 'no'")
    bounding_box: List[int] = Field(..., description="[x1, y1, x2, y2]", min_items=4, max_items=4)
    confidence: float

class DetectResponse(BaseModel):
    timestamp: float
    source: str
    detections: List[DetectionItem]
    annotated_frame_base64: Optional[str] = None

class CameraInfo(BaseModel):
    active_sources: List[str]

class StatsInfo(BaseModel):
    frames: int
    last_time: float
    fps: float
    start_time: float
    uptime_sec: Optional[float]

class StatsResponse(BaseModel):
    cameras: Dict[str, StatsInfo]

# Camera registration schemas
class CameraRegisterRequest(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier")
    rtsp_url: str = Field(..., description="RTSP stream URL or webcam index")
    fps: int = Field(default=15, description="Frames per second")
    roi: Optional[List[int]] = Field(default=None, description="Region of interest [x1, y1, x2, y2]")

class CameraResponse(BaseModel):
    camera_id: str
    rtsp_url: str
    fps: int
    registered_at: Optional[str] = None

class HealthResponse(BaseModel):
    camera_id: str
    healthy: bool
    last_frame_time: Optional[float] = None
    message: str

class DetectionsResponse(BaseModel):
    camera_id: str
    timestamp: float
    fps: int
    tracks: List[Dict[str, Any]]

