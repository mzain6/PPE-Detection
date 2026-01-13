from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime

class CameraRegisterRequest(BaseModel):
    camera_id: str = Field(..., example="cam-01")
    rtsp_url: str = Field(..., example="rtsp://user:pass@192.168.1.10:554/stream")
    fps: Optional[int] = Field(None, example=5)
    roi: Optional[Tuple[float, float, float, float]] = Field(
        None, description="Normalized ROI [x1,y1,x2,y2] (values 0..1)"
    )

class CameraResponse(BaseModel):
    camera_id: str
    rtsp_url: str
    fps: int
    registered_at: datetime

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    area: float

class PPEItem(BaseModel):
    label: str
    confidence: float

class TrackDet(BaseModel):
    track_id: int
    bbox: BoundingBox
    person_confidence: float
    ppe: List[PPEItem] = []
    stable: bool = False
    fall: Optional[bool] = None

class DetectionsResponse(BaseModel):
    camera_id: str
    timestamp: datetime
    fps: int
    tracks: List[TrackDet]

class HealthResponse(BaseModel):
    camera_id: str
    healthy: bool
    last_frame_time: Optional[datetime] = None
    message: Optional[str] = None