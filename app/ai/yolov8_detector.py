
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time
import logging
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # graceful degrade in test envs

# import tracker and iou helper from same package (relative import)
try:
    from .tracker import IOUTracker, iou as iou_helper
except Exception:
    # fallback safe definitions if tracker import fails at import-time
    IOUTracker = None
    def iou_helper(a, b):
        return 0.0

from ..config import settings
from .base import Detector

logger = logging.getLogger(__name__)

def _area(bbox: Tuple[float,float,float,float]) -> float:
    x1,y1,x2,y2 = bbox
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def _safe_parse_box(box) -> Optional[Tuple[float,float,float,float,float,int]]:
    """
    Return (x1,y1,x2,y2,conf,cls_idx) or None on failure.
    """
    try:
        xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, "cpu") else (box.xyxy.tolist()[0] if hasattr(box.xyxy, "tolist") else None)
    except Exception:
        try:
            xyxy = list(box.xyxy)
        except Exception:
            return None
    if not xyxy:
        return None
    try:
        x1,y1,x2,y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    except Exception:
        return None
    conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(getattr(box, "conf", 0.0))
    cls_idx = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(getattr(box, "cls", 0))
    return (x1,y1,x2,y2,conf,cls_idx)

class YoloV8Detector(Detector):
    """
    GPU-first YOLOv8 detector with a per-instance IOUTracker.
    - device: '0' or 'cuda:0' or None
    - camera_id optional (fills returned dict)
    """
    def __init__(self, model_path: str, device: Optional[str] = None, camera_id: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self.camera_id = camera_id or "unknown"
        self.model = None
        self._loaded = False

        # tracking and thresholds
        # IOUTracker should be available; if not, initialization will still proceed but tracking will be disabled
        if IOUTracker is not None:
            self.tracker = IOUTracker(iou_thresh=float(settings.iou_threshold),
                                      max_missing=int(settings.max_missing_frames),
                                      stable_frames=int(settings.stable_frames))
        else:
            self.tracker = None
        self.person_label = settings.person_class_name
        self.ppe_labels = set(settings.ppe_class_names or [])
        self.conf_thresh = float(settings.confidence_threshold)
        self.ppe_conf_thresh = float(settings.ppe_confidence_threshold)
        self.min_area = int(settings.min_box_area)
        self.stable_frames = int(settings.stable_frames)
        self.fall_enabled = bool(settings.fall_detection_enabled)
        self.aspect_threshold = float(settings.aspect_ratio_threshold)

        # attempt load
        self.load()

    def load(self) -> None:
        if self._loaded:
            return
        if YOLO is None:
            raise RuntimeError("ultralytics YOLO not available")
        try:
            # Many ultralytics versions accept device parameter or .to()
            # Support 'cuda:0' or '0' semantics
            dev = None
            if self.device:
                # normalize device
                if isinstance(self.device, str) and self.device.isdigit():
                    dev = int(self.device)
                else:
                    dev = self.device
            try:
                self.model = YOLO(self.model_path) if dev is None else YOLO(self.model_path, device=dev)  # type: ignore
            except TypeError:
                # older ultralytics: construct then .to()
                self.model = YOLO(self.model_path)
                try:
                    if dev is not None:
                        self.model.to(dev)  # type: ignore
                except Exception:
                    logger.debug("model.to(device) not supported")
            self._loaded = True
            logger.info("YOLO model loaded for camera=%s device=%s", self.camera_id, self.device)
        except Exception as e:
            logger.exception("Failed to load YOLO model: %s", e)
            raise

    def _parse_results(self, results, frame_shape) -> List[Dict[str, Any]]:
        out = []
        if results is None:
            return out
        res = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results
        if res is None:
            return out
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return out
        names = getattr(res, "names", {}) or {}
        for box in boxes:
            parsed = _safe_parse_box(box)
            if parsed is None:
                continue
            x1,y1,x2,y2,conf,cls_idx = parsed
            label = names.get(cls_idx, str(cls_idx)) if names else str(cls_idx)
            bbox = (x1,y1,x2,y2)
            if _area(bbox) < self.min_area:
                continue
            out.append({"bbox": bbox, "conf": conf, "label": label})
        return out

    def _simple_fall_detect(self, bbox: Tuple[float,float,float,float]) -> bool:
        x1,y1,x2,y2 = bbox
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        ratio = w / h
        return (ratio > 1.0) and (ratio > self.aspect_threshold)

    def infer(self, frame) -> Dict[str, Any]:
        if not self._loaded or self.model is None:
            raise RuntimeError("model not loaded")
        ts = time.time()
        results = self.model(frame, imgsz=settings.input_size, verbose=False)
        parsed = self._parse_results(results, frame.shape)
        persons = []
        ppe_candidates = []
        for d in parsed:
            lbl = d.get("label", "")
            if lbl == self.person_label and d.get("conf", 0.0) >= self.conf_thresh:
                persons.append({"bbox": d["bbox"], "conf": d["conf"], "ppe": []})
            elif lbl in self.ppe_labels and d.get("conf", 0.0) >= self.ppe_conf_thresh:
                ppe_candidates.append(d)
        # attach PPE
        for ppe in ppe_candidates:
            best_idx = None; best_iou = 0.0
            for idx, per in enumerate(persons):
                try:
                    ov = iou_helper(ppe["bbox"], per["bbox"])
                except Exception:
                    ov = 0.0
                if ov > best_iou:
                    best_iou = ov; best_idx = idx
            if best_idx is not None and best_iou > 0.0:
                # Include bbox information for drawing separate boxes
                persons[best_idx].setdefault("ppe", []).append({
                    "label": ppe["label"], 
                    "confidence": float(ppe["conf"]),
                    "bbox": ppe["bbox"]  # Add bbox for separate visualization
                })
        # update tracker (if available)
        if self.tracker is not None:
            tracks_map = self.tracker.update(persons, timestamp=ts)
        else:
            # fallback: create simple track entries without persistent tracking
            tracks_map = {}
            for i, per in enumerate(persons, start=1):
                from dataclasses import dataclass
                @dataclass
                class _T:
                    id: int
                    bbox: Tuple[float,float,float,float]
                    last_seen: float
                    missing_frames: int = 0
                    stable_count: int = 1
                    person_conf: float = 0.0
                    ppe: List[Dict[str, Any]] = None
                tr = _T(id=i, bbox=per["bbox"], last_seen=ts, person_conf=per.get("conf",0.0), ppe=per.get("ppe",[]))
                tracks_map[i] = tr

        tracks_out = []
        for tr in list(tracks_map.values()):
            stable = (getattr(tr, "stable_count", 1) >= self.stable_frames)
            bbox = tr.bbox
            bb = {
                "x1": float(bbox[0]), "y1": float(bbox[1]),
                "x2": float(bbox[2]), "y2": float(bbox[3]),
                "area": float(_area(bbox))
            }
            tobj = {
                "track_id": tr.id,
                "bbox": bb,
                "person_confidence": float(getattr(tr, "person_conf", 0.0)),
                "ppe": getattr(tr, "ppe", []) or [],
                "stable": bool(stable)
            }
            if self.fall_enabled:
                tobj["fall"] = bool(self._simple_fall_detect(bbox))
            tracks_out.append(tobj)
        return {"camera_id": self.camera_id or "unknown", "timestamp": ts, "fps": int(settings.default_fps), "tracks": tracks_out}

    def close(self) -> None:
        try:
            self.model = None
        except Exception:
            logger.exception("Error closing detector for %s", self.camera_id)