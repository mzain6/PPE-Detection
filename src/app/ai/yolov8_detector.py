
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


def _resolve_model_path(path: str) -> str:
    """
    Model format preference chain: OpenVINO IR > ONNX > PyTorch (.pt)

    OpenVINO is Intel-CPU-optimised (2-4x faster than ONNX on i5/i7/i9).
    Falls back gracefully so the system always starts even without exports.
    """
    import os
    if not path or not path.endswith(".pt"):
        return path  # already an optimised format — leave as-is

    base     = path[:-3]
    stem     = os.path.basename(base)
    dir_path = os.path.dirname(path)

    # 1. OpenVINO IR directory
    ov_dir = os.path.join(dir_path, f"{stem}_openvino_model")
    if os.path.isdir(ov_dir):
        logger.info("[OV] Using OpenVINO model: %s", ov_dir)
        return ov_dir

    # 2. ONNX file
    onnx_path = base + ".onnx"
    if os.path.isfile(onnx_path):
        logger.info("[ONNX] Using ONNX model: %s", onnx_path)
        return onnx_path

    # 3. Original PyTorch weights
    logger.debug("[PT] No optimised format found, using .pt: %s", path)
    return path


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
        return None
    if not xyxy:
        return None
    try:
        x1,y1,x2,y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    except Exception:
        return None
    
    # Handle conf as array or scalar
    try:
        conf_val = box.conf.cpu().numpy() if hasattr(box.conf, "cpu") else box.conf
        if hasattr(conf_val, '__len__'):
            conf = float(conf_val[0]) if len(conf_val) > 0 else 0.0
        else:
            conf = float(conf_val)
    except Exception:
        conf = float(getattr(box, "conf", 0.0))
    
    # Handle cls as array or scalar
    try:
        cls_val = box.cls.cpu().numpy() if hasattr(box.cls, "cpu") else box.cls
        if hasattr(cls_val, '__len__'):
            cls_idx = int(cls_val[0]) if len(cls_val) > 0 else 0
        else:
            cls_idx = int(cls_val)
    except Exception:
        cls_idx = int(getattr(box, "cls", 0))
    
    return (x1,y1,x2,y2,conf,cls_idx)

class YoloV8Detector(Detector):
    """
    GPU-first YOLOv8 detector with a per-instance IOUTracker.
    Now supports TWO models:
    - person_model_path: For detecting persons (e.g., yolov8n.pt)
    - ppe_model_path: For detecting helmet/vest on persons (e.g., best.pt)
    - device: '0' or 'cuda:0' or None
    - camera_id optional (fills returned dict)
    """
    def __init__(self, model_path: str, device: Optional[str] = None, camera_id: Optional[str] = None,
                 person_model_path: Optional[str] = None, ppe_model_path: Optional[str] = None):
        # If person_model_path and ppe_model_path are provided, use them
        # Otherwise fallback to single model_path for backward compatibility
        self.person_model_path = person_model_path or model_path
        self.ppe_model_path = ppe_model_path or model_path
        self.use_separate_models = (person_model_path is not None and ppe_model_path is not None)
        
        self.model_path = model_path  # Kept for backward compatibility
        self.device = device
        self.camera_id = camera_id or "unknown"
        
        # Models
        self.person_model = None
        self.ppe_model = None
        self.model = None  # For backward compatibility
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
            # Normalize device
            dev = None
            if self.device:
                if isinstance(self.device, str):
                    if self.device == "auto":
                        import torch
                        dev = "0" if torch.cuda.is_available() else "cpu"
                    elif self.device.isdigit():
                        dev = int(self.device)
                    else:
                        dev = self.device
            
            # Helper to load a single model (resolves .pt → .onnx automatically)
            def _load_model(path):
                resolved = _resolve_model_path(path)
                try:
                    return YOLO(resolved) if dev is None else YOLO(resolved, device=dev)
                except TypeError:
                    model = YOLO(resolved)
                    if dev is not None:
                        try:
                            model.to(dev)
                        except Exception:
                            pass
                    return model
            
            # Load models
            if self.use_separate_models:
                logger.info(f"Loading person model: {self.person_model_path}")
                self.person_model = _load_model(self.person_model_path)
                
                logger.info(f"Loading PPE model: {self.ppe_model_path}")
                self.ppe_model = _load_model(self.ppe_model_path)
                
                self.model = self.person_model  # Backward compatibility
            else:
                self.model = _load_model(self.model_path)
            
            self._loaded = True
            model_info = f"{self.person_model_path} + {self.ppe_model_path}" if self.use_separate_models else self.model_path
            logger.info(f"YOLO loaded for camera={self.camera_id}: {model_info}")
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
        if not self._loaded or (self.model is None and not self.use_separate_models):
            raise RuntimeError("model not loaded")
        ts = time.time()
        
        import torch
        with torch.inference_mode():
            if self.use_separate_models:
                # TWO-MODEL PIPELINE: person detection + PPE detection
                # Step 1: Detect persons using person model
                person_results = self.person_model(frame, imgsz=settings.input_size, verbose=False)
                person_detections = self._parse_results(person_results, frame.shape)
                
                persons = []
                for d in person_detections:
                    lbl = d.get("label", "")
                    # Standard YOLO uses person class
                    if lbl == "person" and d.get("conf", 0.0) >= self.conf_thresh:
                        persons.append({"bbox": d["bbox"], "conf": d["conf"], "ppe": []})
                
                # Step 2: For each person, detect PPE in their crop
                h, w = frame.shape[:2]
                for person in persons:
                    bbox = person["bbox"]
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Bounds check
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))
                    
                    # Extract person crop
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:
                        # Detect PPE in person crop
                        ppe_results = self.ppe_model(person_crop, imgsz=settings.input_size, verbose=False)
                        ppe_detections = self._parse_results(ppe_results, person_crop.shape)
                        
                        # Convert PPE bbox from crop coordinates to frame coordinates
                        for ppe in ppe_detections:
                            ppe_label = ppe.get("label", "")
                            ppe_conf = ppe.get("conf", 0.0)
                            
                            if ppe_label in self.ppe_labels and ppe_conf >= self.ppe_conf_thresh:
                                # Convert crop bbox to frame bbox
                                cx1, cy1, cx2, cy2 = ppe["bbox"]
                                frame_bbox = (x1 + cx1, y1 + cy1, x1 + cx2, y1 + cy2)
                                
                                person["ppe"].append({
                                    "label": ppe_label,
                                    "confidence": float(ppe_conf),
                                    "bbox": frame_bbox
                                })
            else:
                # SINGLE-MODEL MODE (backward compatibility)
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
        
        # Rest of the method continues the same...
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
    
    def infer_batch(self, frames: list) -> list:
        """
        Run inference on a batch of frames simultaneously.
        
        Args:
            frames: List of numpy arrays (frames)
        
        Returns:
            List of detection dictionaries, one per frame
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("model not loaded")
        
        if not frames:
            return []
        
        ts = time.time()
        
        try:
            # Run batch inference - YOLO model handles list of frames
            results = self.model(frames, imgsz=settings.input_size, verbose=False)
            
            # Process each result
            batch_outputs = []
            for idx, (frame, result) in enumerate(zip(frames, results)):
                # Parse results for this frame
                parsed = self._parse_results([result], frame.shape)
                
                # Separate persons and PPE
                persons = []
                ppe_candidates = []
                for d in parsed:
                    lbl = d.get("label", "")
                    if lbl == self.person_label and d.get("conf", 0.0) >= self.conf_thresh:
                        persons.append({"bbox": d["bbox"], "conf": d["conf"], "ppe": []})
                    elif lbl in self.ppe_labels and d.get("conf", 0.0) >= self.ppe_conf_thresh:
                        ppe_candidates.append(d)
                
                # Attach PPE to persons
                for ppe in ppe_candidates:
                    best_idx = None
                    best_iou = 0.0
                    for p_idx, per in enumerate(persons):
                        try:
                            from .tracker import iou as iou_helper
                            ov = iou_helper(ppe["bbox"], per["bbox"])
                        except Exception:
                            ov = 0.0
                        if ov > best_iou:
                            best_iou = ov
                            best_idx = p_idx
                    if best_idx is not None and best_iou > 0.0:
                        persons[best_idx].setdefault("ppe", []).append({
                            "label": ppe["label"],
                            "confidence": float(ppe["conf"]),
                            "bbox": ppe["bbox"]
                        })
                
                # Note: Tracking is NOT done in batch mode
                # Tracking must be done per-camera externally with separate tracker
                # Return raw persons list for external tracking
                batch_outputs.append({
                    "camera_id": self.camera_id or "unknown",
                    "timestamp": ts,
                    "fps": int(settings.default_fps),
                    "persons": persons  # Raw detections without tracking
                })
            
            return batch_outputs
            
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            # Return empty results for all frames
            return [{"camera_id": self.camera_id, "timestamp": ts, "fps": 0, "persons": []} for _ in frames]
    
    def get_model_memory_usage(self) -> float:
        """
        Get current model memory usage in MB.
        
        Returns:
            Memory usage in MB, or 0 if unavailable
        """
        try:
            import torch
            if torch.cuda.is_available() and self._loaded:
                # Get allocated memory for this process
                # Note: This is approximate as it's per-device, not per-model
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                return allocated_mb
            return 0.0
        except Exception as e:
            logger.debug(f"Error getting model memory usage: {e}")
            return 0.0
