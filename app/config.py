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

# single shared settings instance
settings = Settings()
