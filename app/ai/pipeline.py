"""
AI pipeline (MVP):
- Loads YOLOv8 model from config
- Runs inference on BGR numpy frames
- Filters by configured thresholds and ROI
- Associates PPE detections to nearest person via IOU
- Uses IOUTracker to maintain one bbox per person and stable suppression
- Returns pure JSON-serializable dicts (DB-agnostic)
"""
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import cv2

from ultralytics import YOLO

from ..config import settings
from .tracker import IOUTracker

def _area(bbox: Tuple[float, float, float, float]) -> float:
    x1,y1,x2,y2 = bbox
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    xa1,ya1,xa2,ya2 = a
    xb1,yb1,xb2,yb2 = b
    xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
    inter_w = max(0.0, xi2 - xi1); inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2-xa1) * max(0.0, ya2-ya1)
    area_b = max(0.0, xb2-xb1) * max(0.0, yb2-yb1)
    union = area_a + area_b - inter
    return (inter/union) if union > 0 else 0.0

class AIPipeline:
    def __init__(self):
        # load model lazily - if path invalid an exception will surface on init
        try:
            self.model = YOLO(settings.model_path)
        except Exception as e:
            # keep model None but allow app to start; will raise on inference if missing
            self.model = None
            self._load_err = e
        self.person_label = settings.person_class_name
        self.ppe_labels = set(settings.ppe_class_names or [])
        self.conf_thresh = float(settings.confidence_threshold)
        self.ppe_conf_thresh = float(settings.ppe_confidence_threshold)
        self.min_area = int(settings.min_box_area)
        self.tracker = IOUTracker(iou_thresh=float(settings.iou_threshold),
                                  max_missing=int(settings.max_missing_frames),
                                  stable_frames=int(settings.stable_frames))
        self.stable_frames = int(settings.stable_frames)
        self.fall_enabled = bool(settings.fall_detection_enabled)
        self.aspect_threshold = float(settings.aspect_ratio_threshold)

    def _in_roi(self, bbox: Tuple[float,float,float,float], roi: Optional[Tuple[float,float,float,float]], frame_shape: Tuple[int,int]):
        if not roi or not settings.roi_enabled:
            return True
        h, w = frame_shape[:2]
        x1 = bbox[0] / w; y1 = bbox[1] / h; x2 = bbox[2] / w; y2 = bbox[3] / h
        rx1, ry1, rx2, ry2 = roi
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx >= rx1) and (cx <= rx2) and (cy >= ry1) and (cy <= ry2)

    def _simple_fall_detect(self, bbox: Tuple[float,float,float,float]) -> bool:
        x1,y1,x2,y2 = bbox
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        ratio = w / h
        # heuristic: lying person wider than tall and above aspect threshold
        return (ratio > 1.0) and (ratio > self.aspect_threshold)

    def _parse_results(self, results, frame_shape) -> List[Dict[str, Any]]:
        """
        Convert ultralytics results to a list of detection dicts:
        { bbox: (x1,y1,x2,y2), conf: float, label: str, is_ppe: bool }
        """
        dets = []
        if results is None:
            return dets
        # results can be a Results object or list; use first item
        res = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results
        if res is None:
            return dets
        # attempt to iterate boxes; support several ultralytics versions
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return dets
        names = getattr(res, "names", {})
        for box in boxes:
            # extract xyxy, conf, cls robustly
            try:
                xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, "cpu") else (box.xyxy.tolist()[0] if hasattr(box.xyxy, "tolist") else None)
            except Exception:
                # fallback if xyxy is array-like
                try:
                    xyxy = list(box.xyxy)
                except Exception:
                    continue
            if not xyxy:
                continue
            x1,y1,x2,y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(getattr(box, "conf", 0.0))
            cls_idx = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(getattr(box, "cls", 0))
            label = names.get(cls_idx, str(cls_idx)) if names else str(cls_idx)
            bbox = (x1,y1,x2,y2)
            area = _area(bbox)
            if area < self.min_area:
                continue
            dets.append({"bbox": bbox, "conf": conf, "label": label})
        return dets

    def process_frame(self, frame: np.ndarray, camera_id: str = "unknown", roi: Optional[Tuple[float,float,float,float]] = None) -> Dict[str, Any]:
        """
        Run inference + tracking on a single BGR numpy frame.
        Returns a pure dict ready for JSON serialization:
        { camera_id, timestamp, fps, tracks: [ { track_id, bbox, person_confidence, ppe, stable, (fall) } ] }
        """
        if self.model is None:
            raise RuntimeError(f"YOLO model not loaded: {getattr(self, '_load_err', 'unknown error')}")
        ts = time.time()
        # run inference (ultralytics handles BGR numpy frames)
        results = self.model(frame, imgsz=640, verbose=False)
        parsed = self._parse_results(results, frame.shape)
        # separate persons vs ppe candidates
        persons = []
        ppe_candidates = []
        for d in parsed:
            lbl = d.get("label", "")
            if lbl == self.person_label and d.get("conf", 0.0) >= self.conf_thresh:
                persons.append({"bbox": d["bbox"], "conf": d["conf"], "ppe": []})
            elif lbl in self.ppe_labels and d.get("conf", 0.0) >= self.ppe_conf_thresh:
                ppe_candidates.append(d)
        # attach PPE to nearest person by IOU
        for ppe in ppe_candidates:
            best_idx = None; best_iou = 0.0
            for idx, per in enumerate(persons):
                ov = _iou(ppe["bbox"], per["bbox"])
                if ov > best_iou:
                    best_iou = ov; best_idx = idx
            if best_idx is not None and best_iou > 0.0:
                persons[best_idx].setdefault("ppe", []).append({"label": ppe["label"], "confidence": float(ppe["conf"])})
        # ROI filtering (frame-level check)
        if roi and settings.roi_enabled:
            persons = [p for p in persons if self._in_roi(p["bbox"], roi, frame.shape)]
        # update tracker (IOUTracker ensures one bbox per person)
        tracks_map = self.tracker.update(persons, timestamp=ts)
        # build response
        tracks_out = []
        for tr in list(tracks_map.values()):
            stable = (tr.stable_count >= self.stable_frames)
            bbox = tr.bbox
            bb = {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "area": float(_area(bbox))
            }
            tobj: Dict[str, Any] = {
                "track_id": tr.id,
                "bbox": bb,
                "person_confidence": float(tr.person_conf),
                "ppe": tr.ppe or [],
                "stable": bool(stable)
            }
            if self.fall_enabled:
                tobj["fall"] = bool(self._simple_fall_detect(bbox))
            tracks_out.append(tobj)
        return {
            "camera_id": camera_id,
            "timestamp": ts,
            "fps": int(settings.default_fps),
            "tracks": tracks_out
        }