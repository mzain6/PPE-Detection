"""
SQLAlchemy ORM Models for SafeSite AI
All models use UUID primary keys and PostgreSQL-native types.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, Float, Integer,
    ForeignKey, DateTime, Text, Enum as SAEnum,
    ARRAY
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base
import enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class CameraType(str, enum.Enum):
    rtsp = "rtsp"
    webcam = "webcam"


class ViolationType(str, enum.Enum):
    no_helmet = "no_helmet"
    no_vest = "no_vest"
    no_gloves = "no_gloves"
    no_both = "no_both"


class UserRole(str, enum.Enum):
    super_admin = "super_admin"
    site_admin = "site_admin"
    safety_officer = "safety_officer"
    viewer = "viewer"


class EventType(str, enum.Enum):
    entry = "entry"
    exit = "exit"
    detection = "detection"


# ─── Models ──────────────────────────────────────────────────────────────────

class Site(Base):
    __tablename__ = "sites"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    location = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

    cameras = relationship("Camera", back_populates="site")
    violations = relationship("Violation", back_populates="site")
    personnel = relationship("Personnel", back_populates="site")
    alert_config = relationship("AlertConfig", back_populates="site", uselist=False)


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=False)
    name = Column(String(100), nullable=False)
    stream_url = Column(Text, nullable=False)
    type = Column(SAEnum(CameraType), default=CameraType.rtsp)
    fps_target = Column(Integer, default=15)
    confidence_threshold = Column(Float, default=0.5)
    is_active = Column(Boolean, default=True)
    is_entrance = Column(Boolean, default=False)
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("Site", back_populates="cameras")
    violations = relationship("Violation", back_populates="camera", cascade="all, delete-orphan")
    access_logs = relationship("AccessLog", back_populates="camera", cascade="all, delete-orphan")


class Violation(Base):
    __tablename__ = "violations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=False)
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=False)
    track_id = Column(Integer, nullable=True)
    violation_type = Column(SAEnum(ViolationType), nullable=False)
    confidence = Column(Float, nullable=False)
    person_id = Column(UUID(as_uuid=True), ForeignKey("personnel.id"), nullable=True)
    screenshot_path = Column(Text, nullable=True)
    evidence_video_path = Column(Text, nullable=True)
    is_reviewed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="violations")
    site = relationship("Site", back_populates="violations")
    person = relationship("Personnel", back_populates="violations")


class Personnel(Base):
    __tablename__ = "personnel"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=False)
    name = Column(String(100), nullable=False)
    employee_id = Column(String(50), nullable=False)
    department = Column(String(100), nullable=True)
    face_embedding = Column(Text, nullable=True)  # JSON-serialized float array
    photo_path = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("Site", back_populates="personnel")
    violations = relationship("Violation", back_populates="person")
    access_logs = relationship("AccessLog", back_populates="person")


class AccessLog(Base):
    __tablename__ = "access_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id = Column(UUID(as_uuid=True), ForeignKey("personnel.id"), nullable=True)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=False)
    event_type = Column(SAEnum(EventType), default=EventType.detection)
    confidence = Column(Float, nullable=True)
    face_crop_path = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    person = relationship("Personnel", back_populates="access_logs")
    camera = relationship("Camera", back_populates="access_logs")


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True, nullable=False, index=True)
    hashed_password = Column(Text, nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.viewer)
    site_ids = Column(ARRAY(UUID(as_uuid=True)), default=list)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AlertConfig(Base):
    __tablename__ = "alert_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), unique=True, nullable=False)
    email_recipients = Column(ARRAY(String), default=list)
    violation_types = Column(ARRAY(String), default=list)
    cooldown_minutes = Column(Integer, default=5)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    site = relationship("Site", back_populates="alert_config")
