from typing import List, Tuple, Dict
from pathlib import Path
import yaml

class Settings:
    # path to yaml config file (relative to repo root)
    config_path: str = "config.yaml"

    # model
    model_path: str = "best.pt"
    input_size: int = 640
    device: str = "auto"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    # Default class IDs - will be overridden by config.yaml
    classes: Dict[str, int] = {"person": 0, "helmet": 1, "vest": 2}

    # tracking
    tracking_iou_threshold: float = 0.5  # Increased for better matching
    max_lost_seconds: float = 5.0  # Increased to keep tracks alive longer
    
    # Additional detector settings
    max_missing_frames: int = 1  # Only allow 1 missing frame before removal (instant disappearance)
    stable_frames: int = 3
    person_class_name: str = "person"
    ppe_class_names: List[str] = ["head_helmet", "vest"]  # Updated to match model class names
    ppe_confidence_threshold: float = 0.25
    min_box_area: int = 100
    fall_detection_enabled: bool = False
    aspect_ratio_threshold: float = 1.5

    
    # camera
    default_fps: int = 15
    ffmpeg_transport: str = "tcp"
    
    # reconnection
    reconnect_initial_delay: float = 2.0
    reconnect_max_delay: float = 30.0
    reconnect_backoff_factor: float = 2.0
    reconnect_max_retries: int = 0

    # api
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["*"]

    # alerts
    alerts_enabled: bool = True
    violation_threshold_seconds: float = 10.0
    alert_endpoint: str = "http://localhost:8000/api/ppe-alert"
    
    # GPU optimization settings
    batch_enabled: bool = True
    batch_size: int = 8
    batch_timeout_ms: int = 100
    gpu_memory_threshold_gb: float = 0.5  # Reserve this much VRAM
    max_model_instances: int = -1  # -1 = auto-calculate
    
    # Face detection settings
    face_detection_enabled: bool = True
    face_model_path: str = "yolov8m-face-lindevs.pt"
    face_confidence_threshold: float = 0.5
    face_min_size: int = 30
    
    # Face tracking settings
    face_similarity_threshold: float = 0.6
    face_max_age_seconds: float = 300.0
    face_min_stable_frames: int = 3
    face_reidentification_enabled: bool = True

    def __init__(self, config_path: str = None):
        if config_path:
            self.config_path = config_path
        self.load()

    def load(self) -> None:
        p = Path(self.config_path)
        if not p.exists():
            return
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        m = data.get("model", {})
        self.model_path = m.get("path", self.model_path)
        self.input_size = m.get("input_size", self.input_size)
        self.device = m.get("device", self.device)
        self.confidence_threshold = m.get("conf_threshold", self.confidence_threshold)
        self.iou_threshold = m.get("iou_threshold", self.iou_threshold)
        self.classes = m.get("classes", self.classes)

        t = data.get("tracking", {})
        self.tracking_iou_threshold = t.get("iou_threshold", self.tracking_iou_threshold)
        self.max_lost_seconds = t.get("max_lost_seconds", self.max_lost_seconds)

        c = data.get("camera", {})
        self.default_fps = c.get("default_fps", self.default_fps)
        self.ffmpeg_transport = c.get("ffmpeg_transport", self.ffmpeg_transport)
        
        rec = c.get("reconnect", {})
        self.reconnect_initial_delay = rec.get("initial_delay", self.reconnect_initial_delay)
        self.reconnect_max_delay = rec.get("max_delay", self.reconnect_max_delay)
        self.reconnect_backoff_factor = rec.get("backoff_factor", self.reconnect_backoff_factor)
        self.reconnect_max_retries = rec.get("max_retries", self.reconnect_max_retries)

        a = data.get("api", {})
        self.api_host = a.get("host", self.api_host)
        self.api_port = a.get("port", self.api_port)
        self.cors_origins = a.get("cors_origins", self.cors_origins)

        # Load alerts configuration
        alerts_cfg = data.get("alerts", {})
        self.alerts_enabled = alerts_cfg.get("enabled", self.alerts_enabled)
        self.violation_threshold_seconds = alerts_cfg.get("violation_threshold_seconds", self.violation_threshold_seconds)
        self.alert_endpoint = alerts_cfg.get("alert_endpoint", self.alert_endpoint)
        
        # Load GPU optimization settings
        mc = data.get("multi_camera", {})
        batch_cfg = mc.get("batch_inference", {})
        self.batch_enabled = batch_cfg.get("enabled", self.batch_enabled)
        self.batch_size = batch_cfg.get("batch_size", self.batch_size)
        self.batch_timeout_ms = batch_cfg.get("timeout_ms", self.batch_timeout_ms)
        
        gpu_cfg = mc.get("gpu", {})
        self.gpu_memory_threshold_gb = gpu_cfg.get("memory_threshold_gb", self.gpu_memory_threshold_gb)
        self.max_model_instances = gpu_cfg.get("max_instances", self.max_model_instances)
        
        # Load face detection settings
        face_det = data.get("face_detection", {})
        self.face_detection_enabled = face_det.get("enabled", self.face_detection_enabled)
        self.face_model_path = face_det.get("model_path", self.face_model_path)
        self.face_confidence_threshold = face_det.get("confidence_threshold", self.face_confidence_threshold)
        self.face_min_size = face_det.get("min_face_size", self.face_min_size)
        
        # Load face tracking settings
        face_track = data.get("face_tracking", {})
        self.face_similarity_threshold = face_track.get("similarity_threshold", self.face_similarity_threshold)
        self.face_max_age_seconds = face_track.get("max_face_age_seconds", self.face_max_age_seconds)
        self.face_min_stable_frames = face_track.get("min_stable_frames", self.face_min_stable_frames)
        self.face_reidentification_enabled = face_track.get("reidentification_enabled", self.face_reidentification_enabled)

# single shared settings instance
settings = Settings()
