
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import threading
from collections import deque, Counter

# --- CONFIGURATION ---
# Add as many cameras as you want. 
# You can mix different IPs (DVRs), Users, and Passwords.
CAM_CONFIG = [
    # DVR 1 (Found IP)
    {"id": "Cam 1", "url": "rtsp://admin:admin@2021@192.168.100.49:554/cam/realmonitor?channel=1&subtype=1"},
    {"id": "Cam 2", "url": "rtsp://admin:admin@2021@192.168.100.49:554/cam/realmonitor?channel=2&subtype=1"},
    {"id": "Cam 3", "url": "rtsp://admin:admin@2021@192.168.100.49:554/cam/realmonitor?channel=3&subtype=1"},
    {"id": "Cam 4", "url": "rtsp://admin:admin@2021@192.168.100.49:554/cam/realmonitor?channel=4&subtype=1"},
    
    # EXAMPLE: Local Webcam
    # {"id": "My Webcam", "url": 0}, 
    
    # EXAMPLE: DVR 2
    # {"id": "Office Cam", "url": "rtsp://admin:password@192.168.1.50:554/cam/realmonitor?channel=1&subtype=1"},
]

# Paths
HELMET_MODEL_PATH = r"C:\Users\hashi\Downloads\PPE-Detection-2\PPE-Detection-2\best1.pt"
VEST_MODEL_PATH = r"C:\Users\hashi\Downloads\PPE-Detection-2\PPE-Detection-2\self.pt"

# Settings
BASE_CONF = 0.10
GRID_SIZE = (1280, 720) # Total Window Size (2x2 grid means each cam is 640x360)
CAM_SIZE = (640, 360)   # Resize input to this for display stitching
INFERENCE_IMGSZ = 640   # Smaller inference size for speed on 4 cams
IOU_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
SMOOTHING_FACTOR = 0.4
HISTORY_LEN = 8
STABILITY_THRESHOLD = 0.5

# Class-Specific Logic
TARGET_CLASSES = {
    'Safety Vest': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.10, 'min_area': 400, 'max_ratio': 3.0}, # Keep vest low as it's harder to see
    'Hardhat':     {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.50, 'min_area': 450, 'max_ratio': 3.0}, # Keep hardhat high to avoid false positives
    'NO-Safety Vest': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.25, 'min_area': 500, 'max_ratio': 2.5}, # Lowered conf & area for better violation detection
    'NO-Hardhat':  {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.25, 'min_area': 500, 'max_ratio': 2.5}  # Lowered conf & area for better violation detection
}

class CameraStream:
    """Threaded camera capture to prevent blocking."""
    def __init__(self, src, channel_id):
        self.channel_id = channel_id
        # Handle Integer (Webcam) or String (RTSP)
        if isinstance(src, int) or str(src).isdigit():
            src = int(src)
            
        self.capture = cv2.VideoCapture(src)
        self.status = self.capture.isOpened()
        if not self.status:
            print(f"[Error] Failed to open Camera {channel_id}")
            self.frame = np.zeros((CAM_SIZE[1], CAM_SIZE[0], 3), np.uint8) 
            cv2.putText(self.frame, f"Cam {channel_id} Disconnected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            print(f"[Success] Connected to Camera {channel_id}")
            # Read first frame
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
                    # Resize immediately to save memory/bandwidth if stream is huge
                    self.frame = cv2.resize(frame, CAM_SIZE)
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

def apply_nms(boxes, confs, iou_thresh):
    indices = cv2.dnn.NMSBoxes(boxes, confs, BASE_CONF, iou_thresh)
    if len(indices) > 0:
        return indices.flatten()
    return []

def get_target_class_ids(model, specific_labels):
    ids = []
    for id, name in model.names.items():
        if name in specific_labels:
            ids.append(id)
    return ids

def main():
    # Load Models
    print("Loading models...")
    model_helmet = YOLO(HELMET_MODEL_PATH)
    model_vest = YOLO(VEST_MODEL_PATH)

    helmet_ids = get_target_class_ids(model_helmet, ['Hardhat', 'NO-Hardhat'])
    vest_ids = get_target_class_ids(model_vest, ['Safety Vest', 'NO-Safety Vest'])

    # Start Streams
    cams = []
    print(f"Initializing {len(CAM_CONFIG)} Cameras...")
    
    for i, config in enumerate(CAM_CONFIG):
        print(f"Contenting to {config['id']}...")
        cam = CameraStream(config['url'], config['id'])
        cams.append(cam)

    # Wait a sec for streams to stabilize
    time.sleep(2.0)

    window_name = '4-Camera Aggregated View'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, GRID_SIZE[0], GRID_SIZE[1])
    
    # State Tracking
    smoothed_boxes = {}      
    class_history = {}       
    stable_classes = {}    

    print("Starting Multi-Cam Inference. Press 'Q' to quit.")

    while True:
        # 1. Grab Frames
        frames = [cam.read() for cam in cams]
        processed_frames = []

        # 2. Process Each Frame
        for i, frame in enumerate(frames):
            
            # Draw "Active" dot
            cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1) 
            
            # --- INFERENCE ---
            # Using smaller img size for speed (INFERENCE_IMGSZ=640)
            res_h = model_helmet.track(frame, conf=BASE_CONF, persist=True, classes=helmet_ids, imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            res_v = model_vest.track(frame, conf=BASE_CONF, persist=True, classes=vest_ids, imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            
            all_detections = []
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
                             # DEBUG PRINT
                             print(f"[DEBUG] Raw Detection: {c_name} Conf: {cf:.2f}") 
                             all_detections.append((b, tid, c_name, cf))
                    # Handle cases with no ID (still detect even if tracking fails temporarily)
                    elif result.boxes.xyxy is not None:
                         boxes = result.boxes.xyxy.cpu().numpy()
                         clss = result.boxes.cls.cpu().numpy().astype(int)
                         confs = result.boxes.conf.cpu().numpy()
                         for b, c, cf in zip(boxes, clss, confs):
                             c_name = model.names[c]
                             print(f"[DEBUG] Raw Detection (No Track): {c_name} Conf: {cf:.2f}")
                             # Assign dummy ID -1
                             all_detections.append((b, -1, c_name, cf))

            # --- DRAWING ---
            for box, track_id, class_name, conf in all_detections:
                 # Unique ID for Cam + Track
                 unique_id = f"C{i+1}_{class_name}_{track_id}"
                 
                 config = TARGET_CLASSES.get(class_name)
                 if not config:
                     print(f"[DEBUG] Ignored Class: {class_name} (Not in TARGET_CLASSES)")
                     continue
                 
                 if conf < config['conf']: 
                     print(f"[DEBUG] Low Conf: {class_name} {conf:.2f} < {config['conf']}")
                     continue
                 
                 # Geometric Filter
                 x1, y1, x2, y2 = map(int, box)
                 w, h = x2-x1, y2-y1
                 if w*h < config['min_area']: 
                     print(f"[DEBUG] Too Small: {class_name} Area {w*h} < {config['min_area']}")
                     continue
                 if h/w > config['max_ratio'] if w > 0 else 0: continue

                 # Debug Print (Once per tracking ID to avoid spam)
                 if unique_id not in smoothed_boxes:
                     print(f"[Cam {i+1}] Detected: {class_name} ({conf:.2f})")

                 # Smoothing (Simplified)
                 if unique_id in smoothed_boxes:
                     smooth_box = SMOOTHING_FACTOR * box + (1 - SMOOTHING_FACTOR) * smoothed_boxes[unique_id]
                 else:
                     smooth_box = box
                 smoothed_boxes[unique_id] = smooth_box

                 # Draw
                 sx1, sy1, sx2, sy2 = map(int, smooth_box)
                 color = config['color']
                 cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
                 # Label
                 label = f"{class_name}"
                 cv2.putText(frame, label, (sx1, sy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Add Cam Label
            cv2.putText(frame, f"{cams[i].channel_id}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            processed_frames.append(frame)

        # 3. Stitch into Grid
        # Dynamic Grid Logic
        count = len(processed_frames)
        if count == 0:
            continue
            
        # Basic auto-grid calculation
        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        
        # Pad with black frames if needed
        total_slots = cols * rows
        while len(processed_frames) < total_slots:
            processed_frames.append(np.zeros((CAM_SIZE[1], CAM_SIZE[0], 3), dtype=np.uint8))
            
        # Create Rows
        grid_rows = []
        for r in range(rows):
            row_frames = processed_frames[r*cols : (r+1)*cols]
            grid_rows.append(np.hstack(row_frames))
            
        # Combine Rows
        grid = np.vstack(grid_rows)
        
        # Resize to fit window if needed (optional)
        grid = cv2.resize(grid, GRID_SIZE)

        cv2.imshow(window_name, grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cams:
        cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
