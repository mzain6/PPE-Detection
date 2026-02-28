"""
PPE Stream Worker
-----------------
Replicates the exact detection + drawing logic of run_cameras_with_face_tracking.py
but feeds annotated frames into frame_store so the MJPEG stream endpoint can serve them.

One PPEStreamThread is created per connected camera.
"""

import threading
import time
import os
import logging
import cv2
import numpy as np
import math

import torch
from ultralytics import YOLO

from .frame_store import set_frame

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Model paths (same as run_cameras_with_face_tracking.py) ───────────────────
HELMET_MODEL_PATH = os.path.join(BASE_DIR, "helmet.pt")
VEST_MODEL_PATH   = os.path.join(BASE_DIR, "yolov8_vest_small.pt")
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")   # fallback if pose not available

# ── Settings (same as run_cameras_with_face_tracking.py) ──────────────────────
BASE_CONF        = 0.10
INFERENCE_IMGSZ  = 640
IOU_THRESHOLD    = 0.5
CAM_SIZE         = (640, 360)

USE_GPU  = torch.cuda.is_available()
USE_FP16 = False
DEVICE   = 0 if USE_GPU else "cpu"

PERSON_MIN_AREA        = 3500
PERSON_MIN_ASPECT_RATIO = 0.2
PERSON_MAX_ASPECT_RATIO = 5.0

# Colours / labels (same as run_cameras_with_face_tracking.py TARGET_CLASSES)
TARGET_CLASSES = {
    "Safety Vest":  {"color": (0, 255, 0),   "label": "SAFE"},
    "Hardhat":       {"color": (0, 255, 0),   "label": "SAFE"},
    "NO-Safety Vest": {"color": (0, 0, 255),  "label": "VIOLATION"},
    "NO-Hardhat":    {"color": (0, 0, 255),   "label": "VIOLATION"},
}

# Class-name remaps (helmet model)
def _remap_helmet(name):
    if name in ("helmet", "hi-viz helmet"):
        return "Hardhat"
    elif name == "head":
        return "NO-Hardhat"
    return name

# Class-name remaps (vest model)
SKIP_VEST_CLASSES = {
    "gloves", "boots", "goggles", "none", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots",
    "no-helmet", "no-gloves", "no-boots", "no-goggle",
}
def _remap_vest(name):
    if name == "vest":
        return "Safety Vest"
    if name in SKIP_VEST_CLASSES:
        return None   # caller must skip
    return name


# ── Shared model singletons (loaded once, reused by all threads) ───────────────
_model_lock   = threading.Lock()
_model_helmet = None
_model_vest   = None
_model_person = None
_helmet_ids   = []
_vest_ids     = []

def _get_target_class_ids(model, labels):
    return [i for i, n in model.names.items() if n in labels]

def _ensure_models():
    global _model_helmet, _model_vest, _model_person, _helmet_ids, _vest_ids
    with _model_lock:
        if _model_helmet is None:
            logger.info("[PPEStreamWorker] Loading YOLO models …")
            _model_helmet = YOLO(HELMET_MODEL_PATH)
            _model_vest   = YOLO(VEST_MODEL_PATH)

            # Try pose model first, fall back to yolov8n
            try:
                _model_person = YOLO(os.path.join(BASE_DIR, "yolov8n-pose.pt"))
            except Exception:
                _model_person = YOLO(PERSON_MODEL_PATH)

            if USE_GPU:
                _model_helmet.to("cuda:0")
                _model_vest.to("cuda:0")
                _model_person.to("cuda:0")

            _helmet_ids = _get_target_class_ids(
                _model_helmet, {"Hardhat", "NO-Hardhat", "head", "helmet", "hi-viz helmet"})
            _vest_ids = _get_target_class_ids(_model_vest, {"vest"})

            logger.info("[PPEStreamWorker] Models ready. GPU=%s", USE_GPU)


# ── Per-camera worker thread ───────────────────────────────────────────────────
class PPEStreamThread(threading.Thread):
    """
    Runs in the background for one camera.
    Mirrors the per-camera inference block inside run_cameras_with_face_tracking.py.
    """

    def __init__(self, camera_id: str, source, fps: int = 15):
        super().__init__(daemon=True, name=f"PPEStreamThread-{camera_id}")
        self.camera_id = camera_id
        self.source    = source   # int (webcam index) or str (RTSP url)
        self.fps       = fps
        self._stop_evt = threading.Event()

        # Object history for fall detection (same as run_cameras_with_face_tracking.py)
        self._object_history = {}
        self._frame_counter  = 0

    def stop(self):
        self._stop_evt.set()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _open_capture(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, int):
            if os.name == "nt":
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)
        elif isinstance(src, str) and src.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _draw_person_box(self, frame, x1, y1, x2, y2, state_str, is_fallen):
        color = (0, 0, 255) if is_fallen else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"State: {state_str}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_ppe_box(self, frame, x1, y1, x2, y2, c_name, conf):
        info = TARGET_CLASSES.get(c_name)
        if info is None:
            color, LBL = (255, 255, 0), c_name
        else:
            color, LBL = info["color"], info["label"]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{c_name} ({conf:.0%})",
                    (int(x1), int(y2) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        _ensure_models()

        cap = self._open_capture()
        if not cap.isOpened():
            logger.error("[PPEStreamThread] Cannot open source for %s", self.camera_id)
            return

        interval = 1.0 / max(1, self.fps)
        logger.info("[PPEStreamThread] Started for %s (fps=%s GPU=%s)",
                    self.camera_id, self.fps, USE_GPU)

        while not self._stop_evt.is_set():
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.warning("[PPEStreamThread] Failed to read frame from %s, retrying …", self.camera_id)
                time.sleep(0.5)
                cap.release()
                cap = self._open_capture()
                continue

            # Resize to CAM_SIZE exactly like run_cameras_with_face_tracking.py
            frame = cv2.resize(frame, CAM_SIZE)

            self._frame_counter += 1
            current_time = time.time()

            try:
                # ── 1. PERSON DETECTION ───────────────────────────────────────
                tracked_persons = {}
                if self.camera_id not in self._object_history:
                    self._object_history[self.camera_id] = {}

                res_person = _model_person.track(
                    frame,
                    conf=0.3,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    imgsz=INFERENCE_IMGSZ,
                    verbose=False,
                    iou=IOU_THRESHOLD,
                    half=USE_FP16,
                    device=DEVICE,
                    max_det=10,
                    agnostic_nms=True,
                )

                if res_person and len(res_person) > 0:
                    result = res_person[0]
                    if result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        ids   = result.boxes.id.cpu().numpy().astype(int)
                        # keypoints optional
                        kpts_array = (result.keypoints.xy.cpu().numpy()
                                      if result.keypoints is not None
                                      else [None] * len(boxes))

                        for b, tid, kp in zip(boxes, ids, kpts_array):
                            x1, y1, x2, y2 = b
                            w = x2 - x1; h = y2 - y1
                            area = w * h
                            ar   = h / w if w > 0 else 0

                            if area < PERSON_MIN_AREA:            continue
                            if ar < PERSON_MIN_ASPECT_RATIO:       continue
                            if ar > PERSON_MAX_ASPECT_RATIO:       continue

                            # Fall detection via keypoints (same logic as source file)
                            is_fallen = False
                            if kp is not None and len(kp) >= 13:
                                ls, rs = kp[5], kp[6]
                                lh, rh = kp[11], kp[12]
                                if all(v[0] > 0 and v[1] > 0 for v in [ls, rs, lh, rh]):
                                    mid_s = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
                                    mid_h = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
                                    dx = mid_h[0] - mid_s[0]
                                    dy = mid_h[1] - mid_s[1]
                                    angle = math.degrees(math.atan2(abs(dx), abs(dy)+1e-6))
                                    is_fallen = angle > 45.0
                                    # Draw skeleton spine
                                    cv2.line(frame,
                                             (int(mid_s[0]), int(mid_s[1])),
                                             (int(mid_h[0]), int(mid_h[1])),
                                             (0, 255, 255), 3)

                            state_str = "Fallen" if is_fallen else "Upright"

                            # Update history
                            hist = self._object_history[self.camera_id]
                            if tid not in hist:
                                hist[tid] = {"state": "Upright", "fall_start_time": None,
                                             "last_seen_time": current_time,
                                             "last_bbox": [x1, y1, x2, y2], "fall_alerted": False}
                            hist[tid]["state"] = state_str
                            hist[tid]["last_seen_time"] = current_time
                            hist[tid]["last_bbox"] = [int(x1), int(y1), int(x2), int(y2)]

                            if is_fallen:
                                if hist[tid]["fall_start_time"] is None:
                                    hist[tid]["fall_start_time"] = current_time
                                time_fallen = current_time - hist[tid]["fall_start_time"]
                                cv2.putText(frame, f"Falling: {time_fallen:.1f}s",
                                            (int(x1), int(y1)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                            else:
                                hist[tid]["fall_start_time"] = None
                                hist[tid]["fall_alerted"]    = False

                            self._draw_person_box(frame, x1, y1, x2, y2, state_str, is_fallen)
                            tracked_persons[tid] = b

                # ── 2. PPE DETECTION ──────────────────────────────────────────
                all_detections = []
                for res, model, remap_fn in [
                    (_model_helmet.track(
                        frame, conf=BASE_CONF, persist=True,
                        classes=_helmet_ids, imgsz=INFERENCE_IMGSZ,
                        verbose=False, iou=IOU_THRESHOLD,
                        half=USE_FP16, device=DEVICE), _model_helmet, _remap_helmet),
                    (_model_vest.track(
                        frame, conf=BASE_CONF, persist=True,
                        classes=_vest_ids, imgsz=INFERENCE_IMGSZ,
                        verbose=False, iou=IOU_THRESHOLD,
                        half=USE_FP16, device=DEVICE), _model_vest, _remap_vest),
                ]:
                    if not res or len(res) == 0:
                        continue
                    result = res[0]
                    if result.boxes.id is not None:
                        rboxes = result.boxes.xyxy.cpu().numpy()
                        rids   = result.boxes.id.cpu().numpy().astype(int)
                        rclss  = result.boxes.cls.cpu().numpy().astype(int)
                        rconfs = result.boxes.conf.cpu().numpy()
                        for rb, rtid, rc, rcf in zip(rboxes, rids, rclss, rconfs):
                            c_name = remap_fn(model.names[rc])
                            if c_name is None:
                                continue
                            all_detections.append((rb, rtid, c_name, rcf))
                    elif result.boxes.xyxy is not None:
                        rboxes = result.boxes.xyxy.cpu().numpy()
                        rclss  = result.boxes.cls.cpu().numpy().astype(int)
                        rconfs = result.boxes.conf.cpu().numpy()
                        for rb, rc, rcf in zip(rboxes, rclss, rconfs):
                            c_name = remap_fn(model.names[rc])
                            if c_name is None:
                                continue
                            all_detections.append((rb, -1, c_name, rcf))

                # Draw PPE boxes
                for (rb, rtid, c_name, rcf) in all_detections:
                    rx1, ry1, rx2, ry2 = rb
                    self._draw_ppe_box(frame, rx1, ry1, rx2, ry2, c_name, rcf)

                # ── 3. LIVE indicator ─────────────────────────────────────────
                cv2.circle(frame, (20, 20), 8, (0, 255, 0), -1)
                cv2.putText(frame, "LIVE", (35, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception:
                logger.exception("[PPEStreamThread] Inference error for %s", self.camera_id)

            # Push annotated frame into frame_store for the MJPEG stream endpoint
            try:
                set_frame(self.camera_id, frame)
            except Exception:
                logger.exception("[PPEStreamThread] set_frame failed for %s", self.camera_id)

            # Throttle to configured FPS
            elapsed = time.time() - t_start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        logger.info("[PPEStreamThread] Stopped for %s", self.camera_id)


# ── Registry of active threads ─────────────────────────────────────────────────
_active_threads: dict[str, PPEStreamThread] = {}
_threads_lock = threading.Lock()


def start_ppe_stream(camera_id: str, source, fps: int = 15):
    """Start (or restart) the annotated detection stream for camera_id."""
    with _threads_lock:
        existing = _active_threads.get(camera_id)
        if existing and existing.is_alive():
            logger.info("[PPEStreamWorker] Thread already running for %s", camera_id)
            return
        t = PPEStreamThread(camera_id=camera_id, source=source, fps=fps)
        _active_threads[camera_id] = t
        t.start()
        logger.info("[PPEStreamWorker] Launched PPEStreamThread for %s", camera_id)


def stop_ppe_stream(camera_id: str):
    with _threads_lock:
        t = _active_threads.pop(camera_id, None)
    if t:
        t.stop()
        t.join(timeout=3.0)
        logger.info("[PPEStreamWorker] Stopped thread for %s", camera_id)


def stop_all():
    with _threads_lock:
        ids = list(_active_threads.keys())
    for cid in ids:
        stop_ppe_stream(cid)
