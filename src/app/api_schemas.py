"""
Pydantic schemas for SafeSite AI API request/response validation.
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr

from app.models import CameraType, ViolationType, UserRole, EventType


# ─── Site ────────────────────────────────────────────────────────────────────

class SiteOut(BaseModel):
    id: UUID
    name: str
    location: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Camera ──────────────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    site_id: UUID
    name: str
    stream_url: str
    type: CameraType = CameraType.rtsp
    fps_target: int = 15
    confidence_threshold: float = 0.5
    is_entrance: bool = False


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    stream_url: Optional[str] = None
    type: Optional[CameraType] = None
    fps_target: Optional[int] = None
    confidence_threshold: Optional[float] = None
    is_active: Optional[bool] = None
    is_entrance: Optional[bool] = None


class CameraOut(BaseModel):
    id: UUID
    site_id: UUID
    name: str
    stream_url: str
    type: CameraType
    fps_target: int
    confidence_threshold: float
    is_active: bool
    is_entrance: bool
    last_seen: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class CameraTestRequest(BaseModel):
    stream_url: str
    type: CameraType = CameraType.rtsp


class CameraTestResponse(BaseModel):
    success: bool
    message: str


# ─── Violation ───────────────────────────────────────────────────────────────

class ViolationOut(BaseModel):
    id: UUID
    camera_id: UUID
    site_id: UUID
    track_id: Optional[int]
    violation_type: ViolationType
    confidence: float
    person_id: Optional[UUID]
    person_name: Optional[str] = None
    camera_name: Optional[str] = None
    screenshot_path: Optional[str]
    evidence_video_path: Optional[str]
    is_reviewed: bool
    timestamp: datetime

    class Config:
        from_attributes = True


class ViolationListResponse(BaseModel):
    items: List[ViolationOut]
    total: int
    page: int
    page_size: int
    total_pages: int


class ViolationCreate(BaseModel):
    camera_id: UUID
    site_id: UUID
    track_id: Optional[int] = None
    violation_type: ViolationType
    confidence: float
    person_id: Optional[UUID] = None
    screenshot_path: Optional[str] = None
    evidence_video_path: Optional[str] = None


# ─── Personnel ───────────────────────────────────────────────────────────────

class PersonnelCreate(BaseModel):
    site_id: UUID
    name: str
    employee_id: str
    department: Optional[str] = None


class PersonnelUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None


class PersonnelOut(BaseModel):
    id: UUID
    site_id: UUID
    name: str
    employee_id: str
    department: Optional[str]
    photo_path: Optional[str]
    is_active: bool
    last_seen: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Access Log ──────────────────────────────────────────────────────────────

class AccessLogOut(BaseModel):
    id: UUID
    person_id: Optional[UUID]
    camera_id: UUID
    event_type: EventType
    confidence: Optional[float]
    face_crop_path: Optional[str]
    timestamp: datetime
    person_name: Optional[str] = None
    camera_name: Optional[str] = None

    class Config:
        from_attributes = True


# ─── Auth ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


# ─── User ────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    role: UserRole = UserRole.viewer
    site_ids: List[UUID] = []


class UserUpdate(BaseModel):
    role: Optional[UserRole] = None
    site_ids: Optional[List[UUID]] = None
    is_active: Optional[bool] = None


class UserOut(BaseModel):
    id: UUID
    name: str
    email: str
    role: UserRole
    site_ids: List[UUID]
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# Update forward ref
LoginResponse.model_rebuild()


# ─── Alert Config ────────────────────────────────────────────────────────────

class AlertConfigUpdate(BaseModel):
    email_recipients: Optional[List[str]] = None
    violation_types: Optional[List[str]] = None
    cooldown_minutes: Optional[int] = None
    is_active: Optional[bool] = None


class AlertConfigOut(BaseModel):
    id: UUID
    site_id: UUID
    email_recipients: List[str]
    violation_types: List[str]
    cooldown_minutes: int
    is_active: bool
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# ─── Analytics ───────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    total_violations_today: int
    active_cameras: int
    total_cameras: int
    unreviewed_incidents: int
    offline_cameras: int
    system_healthy: bool


class ViolationsByTypeItem(BaseModel):
    type: str
    count: int


class ViolationsByCameraItem(BaseModel):
    camera_name: str
    count: int


class TrendItem(BaseModel):
    date: str
    count: int


class PeakHourCell(BaseModel):
    day: int   # 0=Mon … 6=Sun
    hour: int  # 0-23
    count: int
