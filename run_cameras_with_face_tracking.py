"""
Multi-Camera PPE Detection with Face Recognition and Person Tracking
Integrates face recognition at entrance camera with cross-camera person tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import threading
from collections import deque
import sys
import requests
from datetime import datetime
import os
import json

# Import our custom modules
from face_recognition_manager import FaceRecognitionManager
from central_tracking_manager import CentralTrackingManager

# ─── Access Control Manager ───────────────────────────────────────────────────
class AccessControlManager:
    """Loads permissions.json and checks if a GlobalID is allowed on a camera."""
    PERMISSIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'permissions.json')

    def __init__(self):
        self.permissions = {}
        self.camera_rules = {}
        self._load()

    def _load(self):
        try:
            with open(self.PERMISSIONS_PATH, 'r') as f:
                data = json.load(f)
            self.permissions  = data.get('persons', {})
            self.camera_rules = data.get('cameras', {})
            print(f"[AccessControl] Loaded {len(self.permissions)} persons, {len(self.camera_rules)} camera rules")
        except Exception as e:
            print(f"[AccessControl] WARNING: Could not load permissions.json: {e}")

    def is_allowed(self, person_id: str, camera_id: str) -> bool:
        """Returns True if person_id is allowed in camera_id, False otherwise."""
        if not person_id or person_id in ("???", None):
            return False
        cam_rule = self.camera_rules.get(camera_id, {})
        zone_type = cam_rule.get('zone_type', 'allowed_all')
        allowed_ids = cam_rule.get('allowed_ids', [])

        if zone_type == 'allowed_all':
            return True                           # entrance — everyone welcome
        elif zone_type == 'restricted_all':
            return person_id in allowed_ids       # nobody unless explicitly listed
        elif zone_type == 'allowed_whitelist':
            return person_id in allowed_ids       # only listed IDs
        return True

    def get_zone_type(self, camera_id: str) -> str:
        return self.camera_rules.get(camera_id, {}).get('zone_type', 'allowed_all')
# ─────────────────────────────────────────────────────────────────────────────

# --- CONFIGURATION ---
# Camera brand detection:
#   Dahua  → rtsp://user:pass@ip/cam/realmonitor?channel=N&subtype=1  (direct OpenCV)
#   YooSee → requires VLC proxy (MJPEG over HTTP)
CAM_CONFIG = [
    # Cam 1: Entrance camera — webcam, face recognition runs here
    {"id": "Cam 1", "url": 0, "is_entrance": True,  "brand": "webcam"},
    # Cam 2: RTSP Channel 1 — RESTRICTED ZONE (Dahua, direct)
    {"id": "Cam 2",
     "url": "rtsp://admin:ADMIN123@192.168.100.158:554/cam/realmonitor?channel=1&subtype=1",
     "is_entrance": False, "brand": "dahua"},
    # Cam 3: RTSP Channel 5 — ALLOWED ZONE for Person 1 (Dahua, direct)
    {"id": "Cam 3",
     "url": "rtsp://admin:ADMIN123@192.168.100.158:554/cam/realmonitor?channel=5&subtype=1",
     "is_entrance": False, "brand": "dahua"},
]

# Paths
# Paths
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELMET_MODEL_PATH = os.path.join(BASE_DIR, "helmet.pt")
VEST_MODEL_PATH = os.path.join(BASE_DIR, "yolov8_vest_small.pt")
FACE_DB_PATH = os.path.join(BASE_DIR, "face_database.pkl")

# Settings
BASE_CONF = 0.10
GRID_SIZE = (1280, 720)
CAM_SIZE = (640, 360)
INFERENCE_IMGSZ = 640  # Increased from 480 for better GPU utilization
IOU_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.4

# GPU Optimization
import torch
USE_GPU = torch.cuda.is_available()
USE_FP16 = False
DEVICE = 0 if USE_GPU else 'cpu'
print(f"[Device] Using {'GPU (CUDA)' if USE_GPU else 'CPU'}")

# Face recognition settings
FACE_CHECK_INTERVAL = 5  # Check for new frames every N frames
FACE_SIMILARITY_THRESHOLD = 0.3  # Lower threshold = more lenient matching

# Violation & Evidence Settings
VIOLATION_THRESHOLD = 10.0  # seconds
API_ALERT_URL = "http://localhost:8001/api/ppe-alert"
EVIDENCE_DIR = "evidence"
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

# Class-Specific Logic
TARGET_CLASSES = {
    'Safety Vest': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.10, 'min_area': 400, 'max_ratio': 3.0},
    'Hardhat': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.50, 'min_area': 450, 'max_ratio': 3.0},
    'NO-Safety Vest': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.20, 'min_area': 400, 'max_ratio': 3.0},
    'NO-Hardhat': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.15, 'min_area': 300, 'max_ratio': 3.5}  # Lowered for better detection
}

# Person Detection Validation (STRICT)
PERSON_MIN_AREA = 3500  # Increased from 2500 -> 3500 to filter small objects
PERSON_MAX_ASPECT_RATIO = 3.5  # Reduced from 4.0 (people aren't super thin)
PERSON_MIN_ASPECT_RATIO = 1.6  # Increased from 1.2 (people are clearly taller than wide)

# Sub-Area Restricted Zones (Custom Polygons)
# Maps camera_id -> list of polygons (each a numpy array of points for cv2.pointPolygonTest)
# The coordinates here are for a 640x360 frame
RESTRICTED_ZONES = {
    "Cam 2": [
        # Shifted left and down significantly to lay flat on the grass
        np.array([[220, 260], [380, 200], [500, 280], [340, 340]], np.int32)
    ]
}


class ViolationState:
    """Tracks violation duration for a specific person to prevent flickers/duplicates"""
    def __init__(self, person_id):
        self.person_id = person_id
        # {violation_type: start_time}
        self.violation_starts = {} 
        # Active confirmed violations
        self.active_violations = set()
        # Track last alert time for the PERSON (strict cooldown)
        self.last_alert_time = 0
        
    def reset_cooldowns(self):
        """Resets cooldown, allowing immediate re-alerting."""
        self.last_alert_time = 0
        self.active_violations.clear()
        
    def update(self, current_violations):
        """
        Updates status and returns list of NEW confirmed violations to alert.
        current_violations: list of strings (e.g. ['NO_HELMET', 'NO_VEST'])
        """
        alerts = []
        now = datetime.now()
        timestamp = now.timestamp()
        
        # Check for new or continuing violations
        newly_confirmed = []
        
        for v in current_violations:
            if v not in self.violation_starts:
                self.violation_starts[v] = now
            
            # Check duration - drastically shorter for restricted areas
            threshold = 0.5 if v == "RESTRICTED_AREA_INTRUSION" else VIOLATION_THRESHOLD
            duration = (now - self.violation_starts[v]).total_seconds()
            
            if duration >= threshold:
                if v not in self.active_violations:
                    newly_confirmed.append(v)
                    self.active_violations.add(v)

        # Handle Cooldown & Merging
        if newly_confirmed:
            # Check if this is a high-priority alert that ignores the general cooldown
            is_high_priority = "RESTRICTED_AREA_INTRUSION" in newly_confirmed
            
            # Check strict person-level cooldown (10 minutes)
            if is_high_priority or (timestamp - self.last_alert_time) >= 600:
                # valid alert
                if not is_high_priority:
                    self.last_alert_time = timestamp
                else:
                    # For restricted areas, we might want a shorter specific cooldown if needed, but for testing we bypass
                    pass
                
                # Merge into one alert if multiple
                merged_violation = " + ".join(sorted(newly_confirmed))
                alerts.append(merged_violation)
            else:
                print(f"[Cooldown] Suppressing alerts for {self.person_id}: {newly_confirmed}")
        
        # Check for compliance (violation ended)
        active_types = list(self.violation_starts.keys())
        for v in active_types:
            if v not in current_violations:
                # Violation ended
                del self.violation_starts[v]
                if v in self.active_violations:
                    self.active_violations.remove(v)
                    print(f"[Violation] {self.person_id} became compliant for {v}")
                    
        return alerts


class ActiveRecording:
    """Represents an active video recording session for a violation"""
    def __init__(self, start_time, pre_frames, camera_id, person_id, violation_type, track_id):
        self.start_time = start_time
        # Determine strict buffer limits (prevent memory issues)
        self.frames = list(pre_frames)[-75:] # Ensure max 2.5s pre-buffer
        self.camera_id = camera_id
        self.person_id = person_id
        self.violation_type = violation_type
        self.track_id = track_id
        self.done = False
        
    def add_frame(self, frame):
        if self.done: return
        self.frames.append(frame)
        
        # Stop after 3.0 seconds (extended to capture full fall + aftermath)
        if (datetime.now() - self.start_time).total_seconds() >= 3.0:
            self.done = True
            self.save_and_alert()
            
    def save_and_alert(self):
        """Save video to disk and send API alert"""
        threading.Thread(target=self._worker).start()
        
    def _worker(self):
        try:
            timestamp_str = int(datetime.now().timestamp())
            # Use .webm for browser compatibility
            filename = f"violation_{self.track_id}_{self.violation_type}_{timestamp_str}.webm"
            filepath = os.path.join(EVIDENCE_DIR, filename)
            
            if not self.frames:
                print("[Error] No frames to save for violation")
                return
                
            height, width, _ = self.frames[0].shape
            
            # Write Video
            # Using VP80 codec for WebM (supported by Chrome/Edge)
            out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'VP80'), 30.0, (width, height))
            for f in self.frames:
                out.write(f)
            out.release()
            
            print(f"[Evidence] Saved {filepath}")
            
            # Send Alert to API
            video_link = f"http://localhost:8001/evidence/{filename}"
            payload = {
                "track_id": int(self.track_id) if isinstance(self.track_id, (int, float)) else 0,
                "person_id": str(self.person_id), # Send Person Name
                "timestamp": datetime.now().isoformat(),
                "camera_id": str(self.camera_id),
                "violation_type": self.violation_type,
                "video_link": video_link,
                "screenshot_path": "" 
            }
            
            response = requests.post(API_ALERT_URL, json=payload, timeout=5)
            if response.status_code == 200:
                print(f"[Alert] Successfully sent to API: {self.violation_type}")
            else:
                print(f"[Alert] API returned {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[Error] Recording worker failed: {e}")




class CameraStream:
    """Threaded camera capture to prevent blocking."""
    def __init__(self, src, channel_id, buffer_len=75):
        self.channel_id = channel_id
        self.frame_buffer = deque(maxlen=buffer_len) # ~2.5s @ 30fps
        if isinstance(src, int) or str(src).isdigit():
            src = int(src)
            
        # Use DirectShow on Windows to avoid black screen / slow connection for Webcams
        if isinstance(src, int) and os.name == 'nt':
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        elif isinstance(src, str) and src.startswith("rtsp://"):
            # For RTSP, force TCP transport to avoid FFmpeg pthread_frame async_lock assertion crashes
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.capture = cv2.VideoCapture(src)
            
        # Reduce buffer size for real-time RTSP
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status = self.capture.isOpened()
        if not self.status:
            print(f"[Error] Failed to open Camera {channel_id}")
            self.frame = np.zeros((CAM_SIZE[1], CAM_SIZE[0], 3), np.uint8) 
            cv2.putText(self.frame, f"Cam {channel_id} Disconnected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            print(f"[Success] Connected to Camera {channel_id}")
            (self.status, self.frame) = self.capture.read()
            if not self.status:
                self.frame = np.zeros((CAM_SIZE[1], CAM_SIZE[0], 3), np.uint8)

        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped and self.status:
            try:
                (grabbed, frame) = self.capture.read()
                if grabbed:
                    if np.mean(frame) < 1.0:
                         print(f"[Warning] Cam {self.channel_id} produced black frame!")
                    self.frame = cv2.resize(frame, CAM_SIZE)
                    self.frame_buffer.append(self.frame)
                else:
                    self.status = False
            except Exception as e:
                print(f"[Error] Camera {self.channel_id} read failed: {e}")
                self.status = False

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.capture.isOpened():
            try:
                self.capture.release()
            except Exception as e:
                print(f"[Warning] Error releasing Cam {self.channel_id}: {e}")


def get_target_class_ids(model, specific_labels):
    ids = []
    for id, name in model.names.items():
        if name in specific_labels:
            ids.append(id)
    return ids


def draw_restricted_alert(frame, camera_id, person_id, person_name):
    """Draw a red RESTRICTED ZONE alert overlay on the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent red overlay on top strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # Bold alert text
    label = f"🚨 RESTRICTED ZONE — {person_name or person_id} UNAUTHORIZED"
    cv2.putText(frame, label, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    # Red border
    cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 220), 4)
    return frame


def main():
    # Initialize Central Tracking Manager
    print("Initializing Central Tracking Manager...")
    central_manager = CentralTrackingManager()

    # Initialize Access Control Manager
    access_control = AccessControlManager()
    
    # Initialize Face Recognition (only if entrance camera exists)
    face_manager = None
    has_entrance = any(config.get('is_entrance', False) for config in CAM_CONFIG)
    
    if has_entrance:
        try:
            print("Initializing Face Recognition Manager...")
            face_manager = FaceRecognitionManager(face_db_path=FACE_DB_PATH, similarity_threshold=FACE_SIMILARITY_THRESHOLD)
        except Exception as e:
            print(f"[Warning] Face recognition initialization failed: {e}")
            print("[Warning] Continuing without face recognition")
            face_manager = None
    
    # Load PPE Detection Models with GPU optimization
    print("Loading PPE detection models with GPU acceleration...")
    model_helmet = YOLO(HELMET_MODEL_PATH)
    model_vest = YOLO(VEST_MODEL_PATH)
    
    # Load standard YOLO for person detection (class 0 = person)
    print("Loading person detection model...")
    model_person = YOLO('yolov8n.pt')  # Standard model with person class
    
    # Move models to GPU
    if USE_GPU:
        print("🚀 Moving models to GPU (CUDA) for maximum performance...")
        model_helmet = model_helmet.to('cuda:0')
        model_vest = model_vest.to('cuda:0')
        model_person = model_person.to('cuda:0')
        print("✅ Models loaded on GPU!")
        print(f"💾 Using FP{16 if USE_FP16 else 32} precision")
        
        # Monitor GPU memory
        import torch
        if torch.cuda.is_available():
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")

    # Update: Include classes from new helmet.pt model ('head', 'helmet', 'hi-viz helmet')
    helmet_ids = get_target_class_ids(model_helmet, ['Hardhat', 'NO-Hardhat', 'head', 'helmet', 'hi-viz helmet'])
    # yolov8_vest_small.pt class 2 = 'vest' (only vest class needed; skip helmet/gloves/boots etc.)
    vest_ids = get_target_class_ids(model_vest, ['vest'])
    print(f"[VestModel] Using class IDs: {vest_ids} from {model_vest.names}")

    # Start Camera Streams
    cams = []
    print(f"Initializing {len(CAM_CONFIG)} Cameras...")
    
    for i, config in enumerate(CAM_CONFIG):
        print(f"Connecting to {config['id']}...")
        cam = CameraStream(config['url'], config['id'])
        cams.append(cam)

    time.sleep(2.0)

    window_name = 'Face Tracking + PPE Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, GRID_SIZE[0], GRID_SIZE[1])
    
    # State Tracking
    smoothed_boxes = {}
    frame_count = 0
    
    # Track which trackers have been assigned person IDs
    tracker_person_map = {}  # {(cam_idx, tracker_id): person_id}
    
    # Coasting memory for anti-flicker
    # {(cam_idx, tracker_id): {'bbox': bbox, 'person_id': id, 'last_seen': time}}
    # Coasting memory for anti-flicker
    # {(cam_idx, tracker_id): {'bbox': bbox, 'person_id': id, 'last_seen': time}}
    tracker_coasting = {}
    
    # Cache for PPE detections during frame skipping
    # {cam_idx: [(box, track_id, class_name, conf), ...]}
    last_ppe_detections = {}
    
    # Violation Tracking
    # Key: (cam_idx, person_id) -> ViolationState
    violation_states = {}
    active_recordings = []

    # Fall Detection State
    # position_history: {(cam_idx, tracker_id): deque of y_center values}
    # fall_states:      {(cam_idx, tracker_id): time when fall first detected}
    from collections import deque
    position_history = {}  # rolling Y-center buffer
    fall_states      = {}  # time at which person first became horizontal
    FALL_ASPECT_THRESHOLD = 1.5   # width/height > this = horizontal/fallen (stricter for webcams)
    FALL_VELOCITY_THRESHOLD = 20  # pixels dropped in 5 frames = fast drop
    FALL_ALERT_SECONDS = 5.0      # seconds must stay fallen before alert
    fall_alert_cooldowns = {}      # {(cam_idx, tracker_id): last_alert_time}
    fall_recordings = set()        # {(cam_idx, tracker_id)} currently being recorded
    FALL_ALERT_API_COOLDOWN = 30.0 # seconds between repeated API calls for same fall
    person_last_seen   = {}        # {(cam_idx, pid): last_time they appeared in tracked_persons}
    person_last_zone   = {}        # {(cam_idx, pid): 'restricted' or 'safe'}
    FALL_DISAPPEAR_SEC = 3.0       # seconds a known person can vanish before triggering fall alert

    
    # Cooldown Sync
    last_reset_check = time.time()
    last_known_clear_time = 0.0

    print("Starting Multi-Cam Inference with Face Tracking. Press 'Q' to quit.")

    while True:
        # --- SYNC WITH BACKEND (Periodically check if history was cleared) ---
        if time.time() - last_reset_check > 5.0: # Check every 5s
            last_reset_check = time.time()
            try:
                # Use a short timeout to not block main loop
                resp = requests.get(f"{API_ALERT_URL}s/status", timeout=0.5) 
                if resp.status_code == 200:
                    data = resp.json()
                    server_clear_time = data.get("last_cleared", 0.0)
                    
                    if server_clear_time > last_known_clear_time:
                        print(f"[Sync] History cleared on server ({server_clear_time}). Resetting local cooldowns.")
                        last_known_clear_time = server_clear_time
                        # Reset all violation states
                        for vs in violation_states.values():
                            vs.reset_cooldowns()
            except Exception:
                # Ignore connection errors (server might be down/busy)
                pass # print("[Sync] Failed to check status")

        # Grab Frames
        frames = [cam.read() for cam in cams]
        processed_frames = []

        # Process Each Camera Frame
        for cam_idx, frame in enumerate(frames):
            config = CAM_CONFIG[cam_idx]
            is_entrance = config.get('is_entrance', False)
            camera_id = config['id']
            
            # Draw "Active" indicator
            cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)
            
            current_time = time.time()
            tracked_persons = {}
            
            # --- FRAME SKIPPING OPTIMIZATION ---
            # Process every frame on GPU for maximum speed
            DO_INFERENCE = True  # Process all frames with GPU
            
            # Initialize collections
            tracked_persons = {}
            all_detections = []
            
            if DO_INFERENCE:
                # 1. PERSON DETECTION (Optimized: conf=0.60 STRICT)
                res_person = model_person.track(
                    frame, 
                    conf=0.60,  # CRITICAL: Increased to 60% to stop false detections
                    persist=True, 
                    classes=[0], 
                    imgsz=INFERENCE_IMGSZ,
                    verbose=False, 
                    iou=IOU_THRESHOLD,
                    half=USE_FP16,
                    device=0 if USE_GPU else 'cpu',
                    max_det=10,  # Limit max people to prevent ghost detections
                    agnostic_nms=True
                )
                
                if res_person and len(res_person) > 0:
                    result = res_person[0]
                    if result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        ids = result.boxes.id.cpu().numpy().astype(int)
                        
                        # Validate person detections (filter false positives)
                        for b, tid in zip(boxes, ids):
                            x1, y1, x2, y2 = b
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            aspect_ratio = height / width if width > 0 else 0
                            
                            # Validation: STRICT CHECK
                            # 1. Area check
                            if area < PERSON_MIN_AREA: continue
                            
                            # 2. Aspect Ratio check
                            if not (PERSON_MIN_ASPECT_RATIO <= aspect_ratio <= PERSON_MAX_ASPECT_RATIO): continue
                            
                            # Valid person detection
                            tracked_persons[tid] = b
                
                # 2. PPE DETECTION (GPU optimized)
                res_h = model_helmet.track(
                    frame, conf=BASE_CONF, persist=True, classes=helmet_ids, 
                    imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD,
                    half=USE_FP16, device=0 if USE_GPU else 'cpu'
                )
                res_v = model_vest.track(
                    frame, conf=BASE_CONF, persist=True, classes=vest_ids, 
                    imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD,
                    half=USE_FP16, device=0 if USE_GPU else 'cpu'
                )
                
                for res, model in [(res_h, model_helmet), (res_v, model_vest)]:
                    if res and len(res) > 0:
                        result = res[0]
                        if result.boxes.id is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            ids = result.boxes.id.cpu().numpy().astype(int)
                            clss = result.boxes.cls.cpu().numpy().astype(int)
                            confs = result.boxes.conf.cpu().numpy()
                            for b, tid, c, cf in zip(boxes, ids, clss, confs):
                                c_name = model.names[c]
                                # Remap helmet model class names
                                if c_name in ['helmet', 'hi-viz helmet']: c_name = 'Hardhat'
                                elif c_name == 'head': c_name = 'NO-Hardhat'
                                # Remap vest model class names (yolov8_vest_small.pt)
                                elif c_name == 'vest': c_name = 'Safety Vest'
                                # Skip non-vest PPE from vest model (gloves, boots, helmet etc.)
                                elif c_name in ['gloves', 'boots', 'goggles', 'none', 'Person',
                                                'no_helmet', 'no_goggle', 'no_gloves', 'no_boots',
                                                'no-helmet', 'no-gloves', 'no-boots', 'no-goggle']:
                                    continue
                                all_detections.append((b, tid, c_name, cf))
                        elif result.boxes.xyxy is not None:
                            # Untracked detections
                            boxes = result.boxes.xyxy.cpu().numpy()
                            clss = result.boxes.cls.cpu().numpy().astype(int)
                            confs = result.boxes.conf.cpu().numpy()
                            for b, c, cf in zip(boxes, clss, confs):
                                c_name = model.names[c]
                                if c_name in ['helmet', 'hi-viz helmet']: c_name = 'Hardhat'
                                elif c_name == 'head': c_name = 'NO-Hardhat'
                                # Remap vest model class names (yolov8_vest_small.pt)
                                elif c_name == 'vest': c_name = 'Safety Vest'
                                elif c_name in ['gloves', 'boots', 'goggles', 'none', 'Person',
                                                'no_helmet', 'no_goggle', 'no_gloves', 'no_boots',
                                                'no-helmet', 'no-gloves', 'no-boots', 'no-goggle']:
                                    continue
                                all_detections.append((b, -1, c_name, cf))
                
                # 3. PROXY TRACKER CREATION (Fallback for missing people)
                # Ifwe have a PPE detection that is NOT inside a person box, create a proxy person
                for idx, (ppe_box, ppe_tid, ppe_class, ppe_conf) in enumerate(all_detections):
                    # Determine the proxy ID
                    if ppe_tid == -1:
                        # Untracked PPE: use hash-based temporary ID
                        proxy_id = 10000 + hash((cam_idx, tuple(ppe_box), ppe_class)) % 10000
                    else:
                        # Tracked PPE: use fixed offset
                        proxy_id = 10000 + ppe_tid
                    
                    # Check if inside any existing person
                    is_inside = False
                    px1, py1, px2, py2 = ppe_box
                    ppe_center = ((px1+px2)/2, (py1+py2)/2)
                    
                    for person_box in tracked_persons.values():
                        bx1, by1, bx2, by2 = person_box
                        if bx1 < ppe_center[0] < bx2 and by1 < ppe_center[1] < by2:
                            is_inside = True
                            break
                    
                    if not is_inside:
                        # Create Proxy Person Box
                        # Estimate Body Box
                        width = px2 - px1
                        height = py2 - py1
                        
                        if 'Vest' in ppe_class:
                            # Vest is torso -> Extrapolate Head up and Legs down
                            proxy_box = [px1, max(0, py1 - height*0.3), px2, min(CAM_SIZE[1], py2 + height*1.5)]
                        else:
                            # Head/Helmet -> Extrapolate Body down
                            proxy_box = [max(0, px1 - width*0.5), py1, min(CAM_SIZE[0], px2 + width*0.5), min(CAM_SIZE[1], py2 + height*4.0)]
                            
                        tracked_persons[proxy_id] = np.array(proxy_box)
                        print(f"[Proxy] Created proxy {proxy_id} from {ppe_class} (tid={ppe_tid}) on {camera_id}")

                # 4. UPDATE COASTING with fresh data
                for tid, bbox in tracked_persons.items():
                    pid = tracker_person_map.get((cam_idx, tid))
                    tracker_coasting[(cam_idx, tid)] = {
                        'bbox': bbox,
                        'person_id': pid,
                        'last_seen': current_time
                    }
                
                # 5. CACHE PPE
                last_ppe_detections[cam_idx] = all_detections

            else:
                # SKIP INFERENCE case
                all_detections = last_ppe_detections.get(cam_idx, [])
                # tracked_persons remains empty, will be filled by coasting restoration below

            # --- COASTING RESTORATION (Anti-Flicker & Frame Skipping Fill) ---
            # Restore tracks if:
            # 1. We skipped inference (fill from memory)
            # 2. We ran inference but lost the track temporarily (anti-flicker)
            
            cam_coasters = [k for k in tracker_coasting.keys() if k[0] == cam_idx]
            
            for k in cam_coasters:
                tid = k[1]
                data = tracker_coasting[k]
                
                # If identifier not in current detections (either because skipped or lost)
                if tid not in tracked_persons:
                    # Determine coasting timeout based on ID confirmation
                    # Confirmed IDs (with face recognition) get longer coasting time
                    person_id = data.get('person_id')
                    if person_id and not person_id.startswith('U') and person_id != "???":
                        # Confirmed authorized person - longer coasting (3 seconds)
                        coasting_timeout = 3.0
                    else:
                        # Unconfirmed or unauthorized - standard coasting (1 second)
                        coasting_timeout = 1.0
                    
                    # If within memory window
                    if current_time - data['last_seen'] < coasting_timeout:
                        tracked_persons[tid] = data['bbox']
                        # Ensure ID map has it
                        if data['person_id'] and (cam_idx, tid) not in tracker_person_map:
                             tracker_person_map[(cam_idx, tid)] = data['person_id']

            # --- FACE RECOGNITION (Entrance Camera Only) ---
            face_boxes_to_draw = []  # Store face boxes to draw separately
            
            if is_entrance and face_manager:
                # Detect faces in entrance camera EVERY FRAME for better accuracy
                try:
                    face_results = face_manager.process_frame_for_faces(frame)
                    
                    if len(face_results) > 0:
                        print(f"[FaceDetection] Found {len(face_results)} face(s)")
                    
                    trackers_with_faces = set()
                    for face_result in face_results:
                        person_id = face_result['person_id']
                        face_bbox = face_result['bbox']
                        similarity = face_result['similarity']
                        authorized = face_result['authorized']
                        
                        print(f"[FaceDetection] Face detected at {face_bbox}, Person ID: {person_id}, Similarity: {similarity:.2f}")
                        
                        # Store face box for drawing later
                        face_boxes_to_draw.append({
                            'bbox': face_bbox,
                            'person_id': person_id,
                            'similarity': similarity,
                            'authorized': authorized
                        })
                        
                        # Find closest tracked person to this face
                        best_tracker_id = None
                        best_distance = float('inf')
                        
                        # FALLBACK: If no tracked persons exist, create a proxy from the face
                        if len(tracked_persons) == 0 and person_id:
                            # Create proxy person box from face (estimate body as ~4x face height below)
                            fx1, fy1, fx2, fy2 = face_bbox
                            face_height = fy2 - fy1
                            face_width = fx2 - fx1
                            
                            # Estimate full body: face is roughly top 1/7 of body
                            proxy_person_box = np.array([
                                max(0, fx1 - face_width * 0.2),  # Slightly wider
                                fy1,  # Start at face top
                                min(CAM_SIZE[0], fx2 + face_width * 0.2),
                                min(CAM_SIZE[1], fy2 + face_height * 5.0)  # Extend down
                            ])
                            
                            # Use a special proxy ID (50000 range for face proxies)
                            proxy_tracker_id = 50000 + hash(tuple(face_bbox)) % 1000
                            tracked_persons[proxy_tracker_id] = proxy_person_box
                            best_tracker_id = proxy_tracker_id
                            best_distance = 0  # Perfect match since we created it from the face
                            print(f"[FaceProxy] Created proxy person {proxy_tracker_id} from face detection")
                        else:
                            # Normal case: find closest existing tracked person
                            for tracker_id, person_bbox in tracked_persons.items():
                                # Calculate distance between face center and person center
                                face_center_x = (face_bbox[0] + face_bbox[2]) / 2
                                face_center_y = (face_bbox[1] + face_bbox[3]) / 2
                                person_center_x = (person_bbox[0] + person_bbox[2]) / 2
                                person_center_y = (person_bbox[1] + person_bbox[3]) / 2
                                
                                distance = np.sqrt((face_center_x - person_center_x)**2 + (face_center_y - person_center_y)**2)
                                
                                if distance < best_distance:
                                    best_distance = distance
                                    best_tracker_id = tracker_id
                        
                        print(f"[FaceLink] Best tracker: {best_tracker_id}, distance: {best_distance:.1f}")
                        
                        # Link face to tracker if close enough
                        if best_tracker_id is not None and best_distance < 500:
                            if person_id:  # Authorized person detected
                                # Register person immediately
                                central_manager.register_person(
                                    person_id=person_id,
                                    tracker_id=best_tracker_id,
                                    camera_id=camera_id,
                                    face_embedding=None,
                                    bbox=tracked_persons[best_tracker_id]
                                )
                                tracker_person_map[(cam_idx, best_tracker_id)] = person_id
                                print(f"[Entrance] ✅ Immediate Assignment: Person {person_id} to tracker {best_tracker_id}")
                                trackers_with_faces.add(best_tracker_id)
                                        
                            else:  # Unauthorized person
                                # CRITICAL: Do not overwrite confirmed Person IDs!
                                # Check if this tracker already has a confirmed ID
                                if (cam_idx, best_tracker_id) in tracker_person_map:
                                    existing_id = tracker_person_map[(cam_idx, best_tracker_id)]
                                    # If already has a Person ID (01, 02, etc.), keep it
                                    if existing_id and not existing_id.startswith('U'):
                                        print(f"[Entrance] ℹ Tracker {best_tracker_id} already has confirmed ID {existing_id}, keeping it")
                                        trackers_with_faces.add(best_tracker_id)
                                        continue  # Skip unauthorized registration
                                
                                # Check if already registered as unauthorized
                                existing_person_id = central_manager.get_person_by_tracker(best_tracker_id, camera_id)
                                if not existing_person_id:
                                    person_id = central_manager.register_unauthorized_person(
                                        tracker_id=best_tracker_id,
                                        camera_id=camera_id,
                                        bbox=tracked_persons[best_tracker_id]
                                    )
                                    tracker_person_map[(cam_idx, best_tracker_id)] = person_id
                                    print(f"[Entrance] ✗ Unauthorized person assigned: {person_id}")
                                    trackers_with_faces.add(best_tracker_id)
                        else:
                            if best_tracker_id is None:
                                print(f"[FaceDetection] No tracked persons found to link face")
                            else:
                                print(f"[FaceDetection] Face too far from tracker (distance: {best_distance:.1f})")
                    
                except Exception as e:
                    print(f"[FaceDetection] Error: {e}")
                    import traceback
                    traceback.print_exc()

            # --- PERSON TRACKING (All Cameras) ---
            # For each tracked person, assign or retrieve person ID
            for tracker_id in tracked_persons.keys():
                # Check if we already have a person ID for this tracker
                person_id = None
                if (cam_idx, tracker_id) in tracker_person_map:
                    person_id = tracker_person_map[(cam_idx, tracker_id)]
                
                # If ID is missing OR it is "???", try to resolve it
                if not person_id or person_id == "???":
                    # Query central manager first
                    if not person_id:
                        person_id = central_manager.get_person_by_tracker(tracker_id, camera_id)
                    
                    if not person_id:
                        # Try spatial-temporal matching
                        person_id = central_manager.identify_new_tracker(
                            tracker_id=tracker_id,
                            camera_id=camera_id,
                            bbox=tracked_persons[tracker_id]
                        )
                    
                    if person_id:
                        tracker_person_map[(cam_idx, tracker_id)] = person_id
                    else:
                        # Mark as unidentified for now (will be resolved in batch below)
                        tracker_person_map[(cam_idx, tracker_id)] = "???"
            
            # --- BATCH FORCE ID ASSIGNMENT (Side Cameras Only) ---
            # After individual matching, assign remaining "???" trackers to available entrance IDs
            if not is_entrance:
                # Find all unidentified trackers
                unidentified_trackers = [(tid, bbox) for tid, bbox in tracked_persons.items() 
                                        if tracker_person_map.get((cam_idx, tid)) == "???"]
                
                if unidentified_trackers:
                    # Get already-assigned IDs from CURRENTLY ACTIVE trackers only
                    # CRITICAL FIX: Include LOCKED IDs from other cameras to prevent duplication
                    current_cam_assigned_ids = set()
                    for tid in tracked_persons.keys():
                        pid = tracker_person_map.get((cam_idx, tid))
                        if pid and pid != "???":
                            current_cam_assigned_ids.add(pid)
                    
                    # Get all recent persons WITH POSITIONS
                    # Filter out IDs that are already assigned on THIS camera
                    recent_persons_with_pos = central_manager.get_all_recent_ids_with_positions(exclude_ids=current_cam_assigned_ids)
                    
                    if recent_persons_with_pos:
                        # SPATIAL MATCHING: Assign IDs based on position similarity
                        assignments = []  # List of (tracker_id, person_id, bbox, distance) tuples
                        
                        # Strict Distance Threshold for Spatial Matching (reduced to prevent jumps)
                        MAX_SPATIAL_MATCH_DIST = 150  # pixels
                        
                        for tracker_id, bbox in unidentified_trackers:
                            # Calculate tracker center
                            tracker_cx = (bbox[0] + bbox[2]) / 2
                            tracker_cy = (bbox[1] + bbox[3]) / 2
                            
                            best_person_id = None
                            min_distance = float('inf')
                            
                            for person_id, person_cx, person_cy in recent_persons_with_pos:
                                # Calculate Euclidean distance between centers
                                distance = np.sqrt((tracker_cx - person_cx)**2 + (tracker_cy - person_cy)**2)
                                
                                if distance < min_distance and distance < MAX_SPATIAL_MATCH_DIST:
                                    min_distance = distance
                                    best_person_id = person_id
                            
                            if best_person_id:
                                assignments.append((tracker_id, best_person_id, bbox, min_distance))
                        
                        # Sort by distance to prioritize closer matches first
                        assignments.sort(key=lambda x: x[3])
                        
                        # Assign IDs, ensuring no duplicates
                        used_ids = set()
                        for tracker_id, person_id, bbox, distance in assignments:
                            if person_id not in used_ids:
                                print(f"[SideCam] Spatial-match {person_id} to tracker {tracker_id} (dist={distance:.1f}px)")
                                
                                tracker_person_map[(cam_idx, tracker_id)] = person_id
                                used_ids.add(person_id)
                                
                                # Register this new location so it sticks
                                central_manager.register_person(
                                    person_id=person_id,
                                    tracker_id=tracker_id,
                                    camera_id=camera_id,
                                    face_embedding=None,
                                    bbox=bbox
                                )

            # --- VIOLATION LOGIC START ---
            # --- VIOLATION LOGIC START ---
            # 1. Map ALL PPE (Positive & Negative) to Person IDs
            person_ppe_map = {} # {person_id: set([Safety Vest, NO-Safety Vest, ...])}
            
            for (box, track_id, class_name, conf) in all_detections:
                # Find owner person
                owner_pid = None
                
                if track_id != -1 and (cam_idx, track_id) in tracker_person_map:
                    owner_pid = tracker_person_map[(cam_idx, track_id)]
                else:
                    # Spatial check against tracked persons
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    
                    for tid, pbox in tracked_persons.items():
                        px1, py1, px2, py2 = pbox
                        if px1 < cx < px2 and py1 < cy < py2:
                            # Overlap found
                            owner_pid = tracker_person_map.get((cam_idx, tid))
                            break
                
                if not owner_pid or owner_pid == "???":
                    # Determine tid to use for unknown person
                    matched_tid = track_id if track_id != -1 else None
                    if matched_tid is None:
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        for t, pbox in tracked_persons.items():
                            if pbox[0] < cx < pbox[2] and pbox[1] < cy < pbox[3]:
                                matched_tid = t
                                break
                    if matched_tid is None:
                        continue # Still no tracker, skip
                    owner_pid = f"Unknown_T{matched_tid}"
                    
                if owner_pid not in person_ppe_map:
                    person_ppe_map[owner_pid] = set()
                person_ppe_map[owner_pid].add(class_name)

            # 2. Resolve Conflicts & Determine Violations
            current_frame_violations = {} # {person_id: [confimed_violations]}
            
            for pid, items in person_ppe_map.items():
                # Conflict Resolution: Positive overrides Negative
                if 'Safety Vest' in items and 'NO-Safety Vest' in items:
                    items.remove('NO-Safety Vest')
                if 'Hardhat' in items and 'NO-Hardhat' in items:
                    items.remove('NO-Hardhat')
                    
                # Collect remaining violations
                violations = [x for x in items if 'NO-' in x]
                if violations:
                    current_frame_violations[pid] = violations

            # 1.5 Sub-Area Intrusion Detection (Polygon Test)
            zones = RESTRICTED_ZONES.get(camera_id, [])
            if zones:
                for tid, bbox in tracked_persons.items():
                    pid = tracker_person_map.get((cam_idx, tid))
                    if not pid or pid == "???":
                        pid = f"Unknown_T{tid}"
                    
                    x1, y1, x2, y2 = bbox
                    # Calculate 'foot' position (bottom center)
                    foot_x = (x1 + x2) / 2
                    foot_y = y2
                    
                    in_zone = False
                    for polygon in zones:
                        # Check if foot is inside polygon
                        # returns > 0 if inside, 0 if on edge, < 0 if outside
                        pt_test = cv2.pointPolygonTest(polygon, (foot_x, foot_y), False)
                        if cam_idx == 1: # Cam 2
                            print(f"[ZoneDebug] {pid} feet at ({foot_x:.1f}, {foot_y:.1f}) -> Polygon Test Result: {pt_test}")
                        
                        if pt_test >= 0:
                            if pid not in current_frame_violations:
                                current_frame_violations[pid] = []
                            current_frame_violations[pid].append("RESTRICTED_AREA_INTRUSION")
                            in_zone = True
                            break # Once inside any zone on this camera, breaking to next person
                            
                    if in_zone:
                        if person_last_zone.get((cam_idx, pid)) != 'restricted':
                            print(f"[RestrictedZone] Person {pid} ENTERED the restricted zone on Cam {cam_idx+1}")
                        person_last_zone[(cam_idx, pid)] = 'restricted'
                    else:
                        person_last_zone[(cam_idx, pid)] = 'safe'
            else:
                for tid in tracked_persons.keys():
                    pid = tracker_person_map.get((cam_idx, tid))
                    if not pid or pid == "???":
                        pid = f"Unknown_T{tid}"
                    person_last_zone[(cam_idx, pid)] = 'safe'
            
            # 2. Update Violation States
            # Iterate all confirmed people on this camera
            active_pids = set()
            for tid in tracked_persons:
                pid = tracker_person_map.get((cam_idx, tid))
                if not pid or pid == "???":
                    pid = f"Unknown_T{tid}"
                active_pids.add(pid)
                
            for pid in active_pids:
                # get state or create new
                state_key = (cam_idx, pid)
                if state_key not in violation_states:
                    violation_states[state_key] = ViolationState(pid)
                
                violations = list(current_frame_violations.get(pid, []))
                
                # Update state
                new_alerts = violation_states[state_key].update(violations)
                
                # 3. Trigger Recordings
                for alert_v in new_alerts:
                    print(f"🚨 TRIGGER VIOLATION: {pid} - {alert_v} (Cam {cam_idx})")
                    
                    # Find track_id for filename
                    pid_tid = "unknown"
                    for tid, p in tracker_person_map.items():
                         if p == pid and tid[0] == cam_idx:
                             pid_tid = tid[1]
                             break
                    
                    # Fix: Use cams[cam_idx] instead of 'cam' variable from outer scope
                    current_cam = cams[cam_idx]
                    
                    rec = ActiveRecording(
                        start_time=datetime.now(),
                        pre_frames=current_cam.frame_buffer,
                        camera_id=camera_id,
                        person_id=pid,
                        violation_type=alert_v,
                        track_id=pid_tid
                    )
                    active_recordings.append(rec)

            # 4. Feed Frames to Recorders
            active_recordings = [r for r in active_recordings if not r.done]
            for rec in active_recordings:
                if str(rec.camera_id) == str(camera_id):
                    rec.add_frame(frame)
            
            # --- FILTER VISUAL DETECTIONS (Remove Conflicts) ---
            # Apply same logic to prevent drawing conflicting boxes
            filtered_detections = []
            person_has_positive = {} # {person_id: set(['Hardhat', 'Safety Vest'])}
            
            # First pass: identify positive detections per person
            for (box, track_id, class_name, conf) in all_detections:
                if class_name in ['Safety Vest', 'Hardhat']:
                    # Find owner
                    owner_pid = None
                    if track_id != -1 and (cam_idx, track_id) in tracker_person_map:
                        owner_pid = tracker_person_map[(cam_idx, track_id)]
                    else:
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        for tid, pbox in tracked_persons.items():
                            px1, py1, px2, py2 = pbox
                            if px1 < cx < px2 and py1 < cy < py2:
                                owner_pid = tracker_person_map.get((cam_idx, tid))
                                break
                    
                    if owner_pid and owner_pid != "???":
                        if owner_pid not in person_has_positive:
                            person_has_positive[owner_pid] = set()
                        person_has_positive[owner_pid].add(class_name)
            
            # Second pass: filter out conflicting negatives
            for detection in all_detections:
                box, track_id, class_name, conf = detection
                should_keep = True
                
                if class_name in ['NO-Safety Vest', 'NO-Hardhat']:
                    # Find owner
                    owner_pid = None
                    if track_id != -1 and (cam_idx, track_id) in tracker_person_map:
                        owner_pid = tracker_person_map[(cam_idx, track_id)]
                    else:
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        for tid, pbox in tracked_persons.items():
                            px1, py1, px2, py2 = pbox
                            if px1 < cx < px2 and py1 < cy < py2:
                                owner_pid = tracker_person_map.get((cam_idx, tid))
                                break
                    
                    if owner_pid and owner_pid in person_has_positive:
                        # Check for conflict
                        if class_name == 'NO-Safety Vest' and 'Safety Vest' in person_has_positive[owner_pid]:
                            should_keep = False
                        elif class_name == 'NO-Hardhat' and 'Hardhat' in person_has_positive[owner_pid]:
                            should_keep = False
                
                if should_keep:
                    filtered_detections.append(detection)
            
            # Use filtered list for drawing
            all_detections = filtered_detections
            
            # --- VIOLATION LOGIC END ---

            # ─── RESTRICTED AREA CHECK ──────────────────────────────────────────
            # Only check on non-entrance cameras that have a restriction
            zone_type = access_control.get_zone_type(camera_id)
            if not is_entrance and zone_type in ('restricted_all', 'allowed_whitelist'):
                for tid, p_bbox in tracked_persons.items():
                    pid = tracker_person_map.get((cam_idx, tid))
                    if not pid or pid == "???": continue

                    if not access_control.is_allowed(pid, camera_id):
                        # Person is NOT allowed here — draw alert
                        person_name = None
                        if pid in access_control.permissions:
                            person_name = access_control.permissions[pid].get('name', pid)
                        frame = draw_restricted_alert(frame, camera_id, pid, person_name)
                        # Red box around the specific person
                        rx1, ry1, rx2, ry2 = [int(v) for v in p_bbox]
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 220), 3)
                        cv2.putText(frame, f"UNAUTHORIZED: {pid}", (rx1, ry1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
                        print(f"[AccessControl] 🚨 ALERT: {pid} in RESTRICTED {camera_id}")
                    else:
                        # Person is allowed — show green ACCESS OK tag
                        rx1, ry1, rx2, ry2 = [int(v) for v in p_bbox]
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 200, 0), 2)
                        cv2.putText(frame, f"AUTHORIZED: {pid}", (rx1, ry1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            # ────────────────────────────────────────────────────────────────────

            # --- DRAW PERSISTENT PERSON ID BOXES ---
            # Draw Person ID box at HEAD LEVEL for each tracked person (like a face box)
            # print(f"[Debug] Tracked persons: {len(tracked_persons)}, Person map: {tracker_person_map}")  # Debug
            for tracker_id, person_bbox in tracked_persons.items():
                person_id = tracker_person_map.get((cam_idx, tracker_id), None)
                # print(f"[Debug] Tracker {tracker_id}: Person ID = {person_id}")  # Debug
                
                # Skip if this person already has a face box drawn in this frame
                if tracker_id in trackers_with_faces:
                    continue
                
                # Only draw if person has a confirmed ID (not "???")
                # MODIFIED: Draw "???" as Yellow for debugging visibility
                if person_id:
                    px1, py1, px2, py2 = map(int, person_bbox)
                    
                    # Calculate head-level box (top 30% of person bbox)
                    person_height = py2 - py1
                    person_width = px2 - px1
                    
                    # Create face-like box at head level
                    face_height = int(person_height * 0.35)
                    face_width = int(person_width * 0.7)
                    
                    # Center the box horizontally
                    center_x = (px1 + px2) // 2
                    fx1 = center_x - face_width // 2
                    fx2 = center_x + face_width // 2
                    fy1 = py1
                    fy2 = py1 + face_height
                    
                    # Determine color and label based on authorization
                    if person_id == "???":
                        id_color = (0, 255, 255) # Yellow for unknown
                        id_label = "Unidentified"
                    elif person_id.startswith('U'):
                        # Unauthorized
                        id_color = (0, 0, 255)  # Red
                        id_label = f"Unauthorized {person_id}"
                    else:
                        # Authorized
                        id_color = (0, 255, 0)  # Green
                        id_label = f"Person {person_id}"
                    
                    # print(f"[Debug] Drawing persistent box for {id_label}")  # Debug
                    # Draw "Face Box" at head level
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), id_color, 3)
                    
                    # Draw Person ID label at bottom of face box
                    (pid_w, pid_h), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (fx1, fy2 + 5), (fx1 + pid_w + 10, fy2 + pid_h + 15), id_color, -1)
                    cv2.putText(frame, id_label, (fx1 + 5, fy2 + pid_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- NO VEST: Draw red box on persons with no vest detected ---
            for tracker_id, person_bbox in tracked_persons.items():
                pid = tracker_person_map.get((cam_idx, tracker_id), None)
                if not pid:
                    continue
                # Check if this person had a vest detected this frame
                has_vest = pid in person_has_positive and 'Safety Vest' in person_has_positive.get(pid, set())
                if not has_vest:
                    px1, py1, px2, py2 = map(int, person_bbox)
                    person_height = py2 - py1
                    # Torso box: from ~35% to ~75% of person height
                    torso_y1 = py1 + int(person_height * 0.35)
                    torso_y2 = py1 + int(person_height * 0.75)
                    # Draw red torso box
                    cv2.rectangle(frame, (px1, torso_y1), (px2, torso_y2), (0, 0, 255), 2)
                    # Draw label background + text
                    label = "No Vest"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (px1, torso_y1 - lh - 8), (px1 + lw + 8, torso_y1), (0, 0, 200), -1)
                    cv2.putText(frame, label, (px1 + 4, torso_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ─── FALL DETECTION CHECK ───────────────────────────────────────────
            now = time.time()
            for tracker_id, person_bbox in tracked_persons.items():
                pid = tracker_person_map.get((cam_idx, tracker_id), None)
                key = (cam_idx, tracker_id)
                
                # Suppress fall detection completely if the person is in a restricted zone
                if pid and person_last_zone.get((cam_idx, pid)) == 'restricted':
                    # Ensure any existing fall countdown is immediately canceled
                    if key in fall_states:
                        del fall_states[key]
                    fall_recordings.discard(key)
                    continue

                px1, py1, px2, py2 = person_bbox
                width  = px2 - px1
                height = py2 - py1
                if height < 10:  # skip degenerate boxes
                    continue
                y_center = (py1 + py2) / 2.0

                # Update position history
                if key not in position_history:
                    position_history[key] = deque(maxlen=10)
                position_history[key].append(y_center)

                # --- Aspect ratio: width/height > threshold means person is horizontal ---
                aspect_ratio = width / max(height, 1)
                is_horizontal = aspect_ratio > FALL_ASPECT_THRESHOLD

                # Debug: print aspect ratio every 60 frames for Cam 1
                if frame_count % 60 == 0 and cam_idx == 0:
                    print(f"[FallDebug] Tracker {tracker_id}: w={width:.0f} h={height:.0f} ratio={aspect_ratio:.2f} horizontal={is_horizontal}")

                # --- Velocity: did bounding-box center drop fast in last 5 frames? ---
                hist = position_history[key]
                vertical_drop = (hist[-1] - hist[-5]) if len(hist) >= 5 else 0
                is_fast_drop  = vertical_drop > FALL_VELOCITY_THRESHOLD

                # --- Determine fall state ---
                # Use OR: either sudden fast drop OR sustained horizontal posture counts as fall
                if is_horizontal or is_fast_drop:
                    if key not in fall_states:
                        # Record when they first went horizontal
                        if is_fast_drop:
                            # Sudden fall — mark start immediately
                            fall_states[key] = now
                        else:
                            # Gradual / we weren't sure — start timer anyway
                            fall_states[key] = now

                    # How long have they been on the ground?
                    fallen_duration = now - fall_states[key]

                    if fallen_duration >= FALL_ALERT_SECONDS:
                        # ====== FALL ALERT ======
                        ix1, iy1, ix2, iy2 = map(int, person_bbox)
                        # Semi-transparent orange overlay on the person
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (0, 140, 255), -1)
                        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                        # Bold red border around person
                        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 50, 255), 3)
                        # Banner at top
                        banner_text = f"FALL DETECTED ({int(fallen_duration)}s)"
                        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (ix1, iy1 - bh - 12), (ix1 + bw + 10, iy1), (0, 50, 255), -1)
                        cv2.putText(frame, banner_text, (ix1 + 5, iy1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        pid = tracker_person_map.get(key, "???")
                        print(f"[FallDetection] \U0001f6a8 FALL ALERT: Person {pid} fallen for {fallen_duration:.1f}s on {camera_id}")

                        # --- Trigger ActiveRecording (saves video + sends API with video_link) ---
                        last_fall_api = fall_alert_cooldowns.get(key, 0)
                        if now - last_fall_api >= FALL_ALERT_API_COOLDOWN and key not in fall_recordings:
                            fall_alert_cooldowns[key] = now
                            fall_recordings.add(key)
                            current_cam = cams[cam_idx]
                            rec = ActiveRecording(
                                start_time=datetime.now(),
                                pre_frames=current_cam.frame_buffer,
                                camera_id=camera_id,
                                person_id=pid,
                                violation_type="fall_detected",
                                track_id=tracker_id
                            )
                            active_recordings.append(rec)
                            print(f"[FallDetection] \U0001f4f9 Recording fall evidence for Person {pid}...")
                    else:
                        # Fallen but not long enough yet — show countdown
                        ix1, iy1, ix2, iy2 = map(int, person_bbox)
                        remaining = FALL_ALERT_SECONDS - fallen_duration
                        cv2.putText(frame, f"Fall? ({remaining:.1f}s)", (ix1, iy1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 130, 255), 2)
                else:
                    # Person is upright — clear fall state and recording lock
                    if key in fall_states:
                        del fall_states[key]
                    fall_recordings.discard(key)
            # ────────────────────────────────────────────────────────────────────

            # ────────────────────────────────────────────────────────────────────

            # --- DRAWING ---
            
            # Draw Restricted Sub-Area Zones
            if zones:
                for polygon in zones:
                    # Draw solid red border
                    cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)
                    
                    # Add Label at the top-left of the polygon
                    min_x = np.min(polygon[:, 0])
                    min_y = np.min(polygon[:, 1])
                    cv2.putText(frame, "RESTRICTED AREA", (min_x, max(20, min_y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for box, track_id, class_name, conf in all_detections:
                unique_id = f"C{cam_idx+1}_{class_name}_{track_id}"
                
                config_class = TARGET_CLASSES.get(class_name)
                if not config_class:
                    continue
                
                if conf < config_class['conf']: 
                    continue
                
                # Geometric Filter
                x1, y1, x2, y2 = map(int, box)
                w, h = x2-x1, y2-y1
                if w*h < config_class['min_area']: 
                    continue
                if h/w > config_class['max_ratio'] if w > 0 else 0: 
                    continue

                # Smoothing
                if unique_id in smoothed_boxes:
                    smooth_box = SMOOTHING_FACTOR * box + (1 - SMOOTHING_FACTOR) * smoothed_boxes[unique_id]
                else:
                    smooth_box = box
                smoothed_boxes[unique_id] = smooth_box

                # Adjust box position based on class type
                display_box = smooth_box.copy()
                
                # For hardhat detections, only show top portion (where helmet sits)
                if class_name in ['Hardhat', 'NO-Hardhat']:
                    box_height = smooth_box[3] - smooth_box[1]
                    helmet_height_ratio = 0.4  # Show 40% height of detection
                    
                    # Shift box upward to cover hair and forehead (reduced for lower position)
                    upward_shift = box_height * 0.15  # Move up by 15% of box height (lowered from 30%)
                    
                    # Apply shift and height adjustment
                    display_box[1] = smooth_box[1] - upward_shift  # Move top up
                    display_box[3] = display_box[1] + (box_height * helmet_height_ratio)  # Set bottom
                
                # Draw PPE box
                sx1, sy1, sx2, sy2 = map(int, display_box)
                color = config_class['color']
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
                
                # PPE label - ONLY show class name (no Person ID)
                label = class_name
                
                # Label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (sx1, sy1 - label_h - 10), (sx1 + label_w, sy1), color, -1)
                cv2.putText(frame, label, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- DRAW FACE BOXES SEPARATELY ---
            for face_data in face_boxes_to_draw:
                face_bbox = face_data['bbox']
                person_id = face_data['person_id']
                similarity = face_data['similarity']
                authorized = face_data['authorized']
                
                # Draw face bounding box
                fx1, fy1, fx2, fy2 = map(int, face_bbox)
                
                # Determine color based on authorization
                if person_id:  # Authorized
                    face_color = (0, 255, 0)  # Green
                    face_label = f"Person {person_id} (Face)"
                else:  # Unauthorized
                    face_color = (0, 0, 255)  # Red
                    face_label = f"Unauthorized (Face)"
                
                # Draw face box
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
                
                # Draw label with similarity score
                label_with_sim = f"{face_label} [{similarity:.2f}]"
                (label_w, label_h), _ = cv2.getTextSize(label_with_sim, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Label background
                cv2.rectangle(frame, (fx1, fy2 + 5), (fx1 + label_w + 10, fy2 + label_h + 15), face_color, -1)
                cv2.putText(frame, label_with_sim, (fx1 + 5, fy2 + label_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add Camera Label
            label_text = f"{camera_id}"
            if is_entrance:
                label_text += " (ENTRANCE)"
            cv2.putText(frame, label_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show tracking stats
            stats = central_manager.get_stats()
            stats_text = f"Active: {stats['total_persons']} | Auth: {stats['authorized']} | Unauth: {stats['unauthorized']}"
            print(f"[Stats] {stats_text}")  # Debug print
            cv2.putText(frame, stats_text, (10, CAM_SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            processed_frames.append(frame)

        # Stitch into Grid
        count = len(processed_frames)
        if count == 0:
            continue
            
        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        
        total_slots = cols * rows
        while len(processed_frames) < total_slots:
            processed_frames.append(np.zeros((CAM_SIZE[1], CAM_SIZE[0], 3), dtype=np.uint8))
            
        grid_rows = []
        for r in range(rows):
            row_frames = processed_frames[r*cols : (r+1)*cols]
            grid_rows.append(np.hstack(row_frames))
            
        grid = np.vstack(grid_rows)
        grid = cv2.resize(grid, GRID_SIZE)

        cv2.imshow(window_name, grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Periodic cleanup
        if frame_count % 300 == 0:  # Every 10 seconds at 30fps
            central_manager.cleanup_old_persons(timeout=30)

    for cam in cams:
        cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
