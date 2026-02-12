from typing import List, Tuple
from pathlib import Path
import yaml

class Settings:
    # path to yaml config file (relative to repo root)
    config_path: str = "config.yaml"

    # model
    model_path: str = "best.pt"
    person_class_name: str = "person"
    ppe_class_names: List[str] = ["helmet", "vest", "mask"]
    confidence_threshold: float = 0.35
    ppe_confidence_threshold: float = 0.25

    # tracking
    iou_threshold: float = 0.3
    max_missing_frames: int = 10
    stable_frames: int = 5

    # camera
    default_fps: int = 5
    min_box_area: int = 900
    roi_enabled: bool = False
    roi: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    # fall detection
    fall_detection_enabled: bool = False
    aspect_ratio_threshold: float = 0.5

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
        self.person_class_name = m.get("person_class_name", self.person_class_name)
        self.ppe_class_names = m.get("ppe_class_names", self.ppe_class_names)
        self.confidence_threshold = m.get("confidence_threshold", self.confidence_threshold)
        self.ppe_confidence_threshold = m.get("ppe_confidence_threshold", self.ppe_confidence_threshold)

        t = data.get("tracking", {})
        self.iou_threshold = t.get("iou_threshold", self.iou_threshold)
        self.max_missing_frames = t.get("max_missing_frames", self.max_missing_frames)
        self.stable_frames = t.get("stable_frames", self.stable_frames)

        c = data.get("camera", {})
        self.default_fps = c.get("default_fps", self.default_fps)
        self.min_box_area = c.get("min_box_area", self.min_box_area)
        self.roi_enabled = c.get("roi_enabled", self.roi_enabled)
        roi_val = c.get("roi", list(self.roi))
        try:
            self.roi = tuple(roi_val)
        except Exception:
            self.roi = (0.0, 0.0, 1.0, 1.0)

        f = data.get("fall_detection", {})
        self.fall_detection_enabled = f.get("enabled", self.fall_detection_enabled)
        self.aspect_ratio_threshold = f.get("aspect_ratio_threshold", self.aspect_ratio_threshold)

# single shared settings instance
settings = Settings()