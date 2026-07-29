"""
SQLAlchemy ORM Models for SafeSite AI
Supports both PostgreSQL and SQLite (for local development without a DB server).
"""
import uuid
import json
from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, Float, Integer,
    ForeignKey, DateTime, Text, Enum as SAEnum,
    TypeDecorator, event
)
from sqlalchemy.orm import relationship
from app.database import Base, DATABASE_URL
import enum


# ─── Cross-DB Compatibility Helpers ──────────────────────────────────────────

def _is_sqlite():
    return "sqlite" in DATABASE_URL.lower()


class UUIDType(TypeDecorator):
    """Stores UUIDs as strings in SQLite, native UUID in PostgreSQL."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return uuid.UUID(value)


class ArrayOfUUID(TypeDecorator):
    """Stores a list of UUIDs as a JSON string (SQLite compatible)."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return "[]"
        return json.dumps([str(v) for v in value])

    def process_result_value(self, value, dialect):
        if not value:
            return []
        return [uuid.UUID(v) for v in json.loads(value)]


class ArrayOfString(TypeDecorator):
    """Stores a list of strings as a JSON string (SQLite compatible)."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return "[]"
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if not value:
            return []
        return json.loads(value)


def _uuid_col(primary_key=False):
    """Return a UUID column compatible with both PG and SQLite."""
    if _is_sqlite():
        return Column(UUIDType, primary_key=primary_key, default=uuid.uuid4)
    else:
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID
        return Column(PG_UUID(as_uuid=True), primary_key=primary_key, default=uuid.uuid4)


def _uuid_fk_col(fk_table, nullable=False):
    if _is_sqlite():
        return Column(UUIDType, ForeignKey(fk_table), nullable=nullable)
    else:
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID
        return Column(PG_UUID(as_uuid=True), ForeignKey(fk_table), nullable=nullable)


def _array_uuid_col():
    if _is_sqlite():
        return Column(ArrayOfUUID, default=list)
    else:
        from sqlalchemy.dialects.postgresql import ARRAY, UUID as PG_UUID
        return Column(ARRAY(PG_UUID(as_uuid=True)), default=list)


def _array_str_col():
    if _is_sqlite():
        return Column(ArrayOfString, default=list)
    else:
        from sqlalchemy.dialects.postgresql import ARRAY
        return Column(ARRAY(String), default=list)


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

    id          = _uuid_col(primary_key=True)
    name        = Column(String(100), nullable=False)
    location    = Column(String(200))
    created_at  = Column(DateTime, default=datetime.utcnow)

    cameras     = relationship("Camera",     back_populates="site")
    violations  = relationship("Violation",  back_populates="site")
    personnel   = relationship("Personnel",  back_populates="site")
    alert_config = relationship("AlertConfig", back_populates="site", uselist=False)


class Camera(Base):
    __tablename__ = "cameras"

    id                   = _uuid_col(primary_key=True)
    site_id              = _uuid_fk_col("sites.id", nullable=False)
    name                 = Column(String(100), nullable=False)
    stream_url           = Column(Text, nullable=False)
    type                 = Column(SAEnum(CameraType), default=CameraType.rtsp)
    fps_target           = Column(Integer, default=15)
    confidence_threshold = Column(Float, default=0.5)
    is_active            = Column(Boolean, default=True)
    is_entrance          = Column(Boolean, default=False)
    last_seen            = Column(DateTime, nullable=True)
    created_at           = Column(DateTime, default=datetime.utcnow)

    site        = relationship("Site",     back_populates="cameras")
    violations  = relationship("Violation", back_populates="camera", cascade="all, delete-orphan")
    access_logs = relationship("AccessLog", back_populates="camera", cascade="all, delete-orphan")


class Violation(Base):
    __tablename__ = "violations"

    id                   = _uuid_col(primary_key=True)
    camera_id            = _uuid_fk_col("cameras.id", nullable=False)
    site_id              = _uuid_fk_col("sites.id", nullable=False)
    track_id             = Column(Integer, nullable=True)
    violation_type       = Column(SAEnum(ViolationType), nullable=False)
    confidence           = Column(Float, nullable=False)
    person_id            = _uuid_fk_col("personnel.id", nullable=True)
    screenshot_path      = Column(Text, nullable=True)
    evidence_video_path  = Column(Text, nullable=True)
    is_reviewed          = Column(Boolean, default=False)
    timestamp            = Column(DateTime, default=datetime.utcnow, index=True)
    created_at           = Column(DateTime, default=datetime.utcnow)

    camera  = relationship("Camera",    back_populates="violations")
    site    = relationship("Site",      back_populates="violations")
    person  = relationship("Personnel", back_populates="violations")


class Personnel(Base):
    __tablename__ = "personnel"

    id              = _uuid_col(primary_key=True)
    site_id         = _uuid_fk_col("sites.id", nullable=False)
    name            = Column(String(100), nullable=False)
    employee_id     = Column(String(50), nullable=False)
    department      = Column(String(100), nullable=True)
    face_embedding  = Column(Text, nullable=True)   # JSON-serialized float array
    photo_path      = Column(Text, nullable=True)
    is_active       = Column(Boolean, default=True)
    last_seen       = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    site        = relationship("Site",      back_populates="personnel")
    violations  = relationship("Violation", back_populates="person")
    access_logs = relationship("AccessLog", back_populates="person")


class AccessLog(Base):
    __tablename__ = "access_log"

    id           = _uuid_col(primary_key=True)
    person_id    = _uuid_fk_col("personnel.id", nullable=True)
    camera_id    = _uuid_fk_col("cameras.id",   nullable=False)
    event_type   = Column(SAEnum(EventType), default=EventType.detection)
    confidence   = Column(Float, nullable=True)
    face_crop_path = Column(Text, nullable=True)
    timestamp    = Column(DateTime, default=datetime.utcnow, index=True)

    person  = relationship("Personnel", back_populates="access_logs")
    camera  = relationship("Camera",    back_populates="access_logs")


class User(Base):
    __tablename__ = "users"

    id              = _uuid_col(primary_key=True)
    name            = Column(String(100), nullable=False)
    email           = Column(String(200), unique=True, nullable=False, index=True)
    hashed_password = Column(Text, nullable=False)
    role            = Column(SAEnum(UserRole), default=UserRole.viewer)
    site_ids        = _array_uuid_col()
    is_active       = Column(Boolean, default=True)
    last_login      = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


class AlertConfig(Base):
    __tablename__ = "alert_configs"

    id               = _uuid_col(primary_key=True)
    site_id          = _uuid_fk_col("sites.id", nullable=False)
    email_recipients = _array_str_col()
    violation_types  = _array_str_col()
    cooldown_minutes = Column(Integer, default=5)
    is_active        = Column(Boolean, default=True)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    site = relationship("Site", back_populates="alert_config")
