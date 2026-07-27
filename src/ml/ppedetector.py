"""
ppedetector.py

Core PPE detection service class: PPEDetector.

Responsibilities:
- Load YOLOv8 model (GPU if available, CPU fallback)
- Run inference for classes: person (0), helmet (12), vest (16)
- Apply NMS to person detections to ensure ONE bbox per person
- Associate helmet/vest to persons by IoU
- Provide simple IoU tracker that assigns stable person_id across frames
- Optionally annotate frames (draw boxes and labels)
- Return structured JSON-ready list of detections

Usage:
    detector = PPEDetector(model_path="best.pt")
    annotated_frame, detections = detector.detect(frame, annotate=True)
    # detections: list of dicts with keys: person_id, helmet_status, vest_status, bounding_box, confidence
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import time

# Keep ultralytics use minimal
from ultralytics import YOLO

# Optional: detect torch to decide device
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def _prefer_onnx(pt_path: str) -> str:
    """
    Model format preference chain: OpenVINO IR > ONNX > PyTorch (.pt)
    OpenVINO is Intel-CPU-optimised (2-4x faster than ONNX on i5/i7/i9).
    Falls back silently to .pt so the system is always stable.
    """
    import os
    if not pt_path or not pt_path.endswith(".pt"):
        return pt_path
    base     = pt_path[:-3]
    stem     = os.path.basename(base)
    dir_path = os.path.dirname(pt_path)
    # 1. OpenVINO IR directory
    ov_dir = os.path.join(dir_path, f"{stem}_openvino_model")
    if os.path.isdir(ov_dir):
        return ov_dir
    # 2. ONNX
    onnx_path = base + ".onnx"
    if os.path.isfile(onnx_path):
        return onnx_path
    # 3. Original .pt
    return pt_path

# Alias
_prefer_optimized = _prefer_onnx



def _to_numpy(x):
    """Convert tensor-like to numpy safely."""
    try:
        if _TORCH_AVAILABLE and hasattr(x, "cpu") and isinstance(x, torch.Tensor):
            return x.cpu().numpy()
    except Exception:
        pass
    return np.array(x)


def _xyxy_to_xywh(box):
    # box = [x1,y1,x2,y2]
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def _iou(boxA, boxB) -> float:
    # boxes in xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


class SimpleTracker:
    """
    Lightweight IoU-based tracker.
    - Stores last bbox and last_seen timestamp for each track.
    - Assigns persistent integer IDs.
    - Removes tracks not seen for `max_lost_seconds`.
    """

    def __init__(self, iou_threshold=0.3, max_lost_seconds=2.0):
        self.next_id = 1
        self.tracks = {}  # id -> {bbox, last_seen_time}
        self.iou_threshold = iou_threshold
        self.max_lost_seconds = max_lost_seconds

    def update(self, detections: List[Tuple[List[int], float]]):
        """
        detections: list of (bbox_xyxy, confidence)
        returns: list of (person_id, bbox, confidence)
        """
        now = time.time()
        assigned = {}
        results = []

        # Try greedy matching by IoU (tracks -> detections)
        unmatched_dets = set(range(len(detections)))
        for tid, info in list(self.tracks.items()):
            best_iou = 0.0
            best_det = None
            for di in unmatched_dets:
                det_box, det_conf = detections[di]
                i = _iou(info["bbox"], det_box)
                if i > best_iou:
                    best_iou = i
                    best_det = di
            if best_det is not None and best_iou >= self.iou_threshold:
                det_box, det_conf = detections[best_det]
                self.tracks[tid]["bbox"] = det_box
                self.tracks[tid]["last_seen"] = now
                results.append((tid, det_box, det_conf))
                unmatched_dets.remove(best_det)

        # Remaining unmatched detections become new tracks
        for di in list(unmatched_dets):
            det_box, det_conf = detections[di]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": det_box, "last_seen": now}
            results.append((tid, det_box, det_conf))

        # Cleanup tracks not seen in time
        to_delete = []
        for tid, info in list(self.tracks.items()):
            if now - info["last_seen"] > self.max_lost_seconds:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return results


class PPEDetector:
    """
    PPE Detector service.

    Parameters:
    - model_path: Path to yolov8 weights (do NOT modify weights).
    - device: 'auto'|'cpu'|'cuda'  - auto picks GPU if torch.cuda.is_available()
    - conf_thres: confidence threshold for detections
    - iou_thres: IoU threshold for person NMS & tracker matching
    - imgsz: inference size
    """

    def __init__(self,
                 model_path: str = "best.pt",
                 device: str = "auto",
                 conf_thres: float = 0.25,
                 person_iou_nms: float = 0.5,
                 tracker_iou: float = 0.3,
                 imgsz: int = 640):
        # device resolution
        if device == "auto":
            if _TORCH_AVAILABLE and hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Resolve model path: prefer .onnx if available (faster inference),
        # fall back to .pt if ONNX export hasn't been run yet.
        resolved_path = _prefer_onnx(model_path)
        self.model = YOLO(resolved_path)

        # move model to device if supported
        try:
            if self.device == "cuda":
                self.model.to("cuda")
        except Exception:
            # ignore, ultralytics may manage device automatically
            pass

        self.conf_thres = conf_thres
        self.person_iou_nms = person_iou_nms
        self.tracker = SimpleTracker(iou_threshold=tracker_iou)
        self.imgsz = imgsz

        # class ids used
        self.CLASS_PERSON = 0
        self.CLASS_HELMET = 12
        self.CLASS_VEST = 16

    def _parse_results(self, results) -> Dict[int, List[Dict]]:
        """
        Parse ultralytics results into a dict of lists keyed by class id:
            {cls: [ {bbox: [x1,y1,x2,y2], conf: float}, ... ] }
        """
        parsed = {}
        if len(results) == 0:
            return parsed
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return parsed

        xyxy = _to_numpy(getattr(boxes, "xyxy", []))
        confs = _to_numpy(getattr(boxes, "conf", []))
        clss = _to_numpy(getattr(boxes, "cls", []))

        if xyxy is None or len(xyxy) == 0:
            return parsed

        for i in range(len(xyxy)):
            cls = int(clss[i])
            bbox = [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])]
            conf = float(confs[i])
            if conf < self.conf_thres:
                continue
            parsed.setdefault(cls, []).append({"bbox": bbox, "conf": conf})
        return parsed

    def _nms_boxes(self, boxes: List[List[float]], scores: List[float], iou_threshold: float):
        """
        Simple NMS using OpenCV. boxes expected as [x1,y1,x2,y2]
        Returns indices kept.
        """
        if len(boxes) == 0:
            return []
        xywh = [ _xyxy_to_xywh(b) for b in boxes ]
        # cv2.dnn.NMSBoxes expects int boxes
        try:
            indices = cv2.dnn.NMSBoxes(xywh, scores, self.conf_thres, iou_threshold)
            # cv2 returns list of lists or numpy array depending on platform
            if isinstance(indices, (list, tuple)):
                flat = [int(i) for i in indices]
            else:
                flat = [int(i[0]) if hasattr(i, '__len__') else int(i) for i in indices]
            return flat
        except Exception:
            # Fallback: simple greedy NMS
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            keep = []
            while idxs:
                i = idxs.pop(0)
                keep.append(i)
                rem = []
                for j in idxs:
                    if _iou(boxes[i], boxes[j]) <= iou_threshold:
                        rem.append(j)
                idxs = rem
            return keep

    def detect(self, frame: np.ndarray, annotate: bool = False) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Run detection on a single frame.

        Returns:
            annotated_frame (or original frame if annotate False),
            detections: list of dicts:
                {
                  person_id: int,
                  helmet_status: "yes"/"no",
                  vest_status: "yes"/"no",
                  bounding_box: [x1,y1,x2,y2],
                  confidence: float
                }
        """
        if frame is None:
            return None, []

        # run model: detect person, helmet, vest
        results = self.model(frame, classes=[self.CLASS_PERSON, self.CLASS_HELMET, self.CLASS_VEST],
                             imgsz=self.imgsz, verbose=False)

        parsed = self._parse_results(results)

        persons = parsed.get(self.CLASS_PERSON, [])
        helmets = parsed.get(self.CLASS_HELMET, [])
        vests = parsed.get(self.CLASS_VEST, [])

        # apply NMS for person boxes to ensure one bbox per person
        person_boxes = [p["bbox"] for p in persons]
        person_scores = [p["conf"] for p in persons]
        keep_idx = self._nms_boxes(person_boxes, person_scores, self.person_iou_nms)
        persons_nms = [persons[i] for i in keep_idx]

        # prepare detections for tracker: list of (bbox, conf)
        person_inputs = [ ( [float(b) for b in p["bbox"]], float(p["conf"]) ) for p in persons_nms ]

        # get assigned ids
        tracked = self.tracker.update(person_inputs)  # list of (id, bbox, conf)

        detections = []
        for tid, bbox, conf in tracked:
            # find helmet overlap
            helmet_yes = "no"
            for h in helmets:
                if _iou(bbox, h["bbox"]) > 0.25:
                    helmet_yes = "yes"
                    break
            # find vest overlap
            vest_yes = "no"
            for v in vests:
                if _iou(bbox, v["bbox"]) > 0.25:
                    vest_yes = "yes"
                    break

            det = {
                "person_id": int(tid),
                "helmet_status": helmet_yes,
                "vest_status": vest_yes,
                "bounding_box": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "confidence": float(conf)
            }
            detections.append(det)

        out_frame = frame if not annotate else frame.copy()
        if annotate and out_frame is not None:
            # draw boxes and labels
            for d in detections:
                x1, y1, x2, y2 = d["bounding_box"]
                label = f"ID:{d['person_id']} H:{d['helmet_status']} V:{d['vest_status']} {d['confidence']:.2f}"
                color = (0, 200, 0) if d["helmet_status"] == "yes" and d["vest_status"] == "yes" else (0, 0, 200)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out_frame, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # draw helmets and vests as smaller boxes (optional visibility)
            for h in helmets:
                bx = [int(x) for x in h["bbox"]]
                cv2.rectangle(out_frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 150, 150), 1)
                cv2.putText(out_frame, f"H:{h['conf']:.2f}", (bx[0], bx[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,150,150), 1)
            for v in vests:
                bx = [int(x) for x in v["bbox"]]
                cv2.rectangle(out_frame, (bx[0], bx[1]), (bx[2], bx[3]), (150, 150, 0), 1)
                cv2.putText(out_frame, f"V:{v['conf']:.2f}", (bx[0], bx[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,0), 1)

        # TODO: plug fall detection here using `detections` and frame (e.g. posture analysis, keypoints)
        # TODO: plug face recognition here using cropped face regions per person (requires face detection)
        # TODO: plug permit-to-work validation here (cross-reference person_id with DB/permissions)
        # TODO: plug vehicle detection later by running model with vehicle classes and merging logic

        return out_frame, detections

    def release(self):
        # currently model doesn't require explicit release, but method provided for API symmetry
        try:
            del self.model
        except Exception:
            pass