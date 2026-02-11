"""
Multi-Camera PPE Detection with Face Recognition and Person Tracking
Integrates face recognition at entrance camera with cross-camera person tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from collections import deque, Counter
import sys

# Import our custom modules
from face_recognition_manager import FaceRecognitionManager
from central_tracking_manager import CentralTrackingManager

# --- CONFIGURATION ---
CAM_CONFIG = [
    # Cam 1: Entrance camera with face recognition (WEBCAM)
    {"id": "Cam 1", "url": 0, "is_entrance": True},
    # Cam 2: Disabled for testing
    # {"id": "Cam 2", "url": "rtsp://admin:ADMIN123@192.168.100.157:554/cam/realmonitor?channel=5&subtype=1", "is_entrance": False},
]

# Paths
HELMET_MODEL_PATH = r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\helmet.pt"
VEST_MODEL_PATH = r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\vest.pt"
FACE_DB_PATH = r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\face_database.pkl"

# Settings
BASE_CONF = 0.10
GRID_SIZE = (1280, 720)
CAM_SIZE = (640, 360)
INFERENCE_IMGSZ = 640
IOU_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.4

# Face recognition settings
FACE_CHECK_INTERVAL = 5  # Check for new faces every N frames
FACE_SIMILARITY_THRESHOLD = 0.3  # Lower threshold = more lenient matching

# Class-Specific Logic
TARGET_CLASSES = {
    'Safety Vest': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.10, 'min_area': 400, 'max_ratio': 3.0},
    'Hardhat': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.50, 'min_area': 450, 'max_ratio': 3.0},
    'NO-Safety Vest': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.20, 'min_area': 400, 'max_ratio': 3.0},
    'NO-Hardhat': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.15, 'min_area': 300, 'max_ratio': 3.5}  # Lowered for better detection
}


class CameraStream:
    """Threaded camera capture to prevent blocking."""
    def __init__(self, src, channel_id):
        self.channel_id = channel_id
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


def get_target_class_ids(model, specific_labels):
    ids = []
    for id, name in model.names.items():
        if name in specific_labels:
            ids.append(id)
    return ids


def main():
    # Initialize Central Tracking Manager
    print("Initializing Central Tracking Manager...")
    central_manager = CentralTrackingManager()
    
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
    
    # Load PPE Detection Models
    print("Loading PPE detection models...")
    model_helmet = YOLO(HELMET_MODEL_PATH)
    model_vest = YOLO(VEST_MODEL_PATH)
    
    # Load standard YOLO for person detection (class 0 = person)
    print("Loading person detection model...")
    model_person = YOLO('yolov8n.pt')  # Standard model with person class

    # Update: Include classes from new helmet.pt model ('head', 'helmet', 'hi-viz helmet')
    helmet_ids = get_target_class_ids(model_helmet, ['Hardhat', 'NO-Hardhat', 'head', 'helmet', 'hi-viz helmet'])
    vest_ids = get_target_class_ids(model_vest, ['Safety Vest', 'NO-Safety Vest'])

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
    
    # Face recognition confirmation tracking (2-second window)
    face_confirmation = {}  # {tracker_id: {'person_id': id, 'start_time': timestamp, 'confirmed': bool}}
    CONFIRMATION_DURATION = 2.0  # Seconds required for confirmation

    print("Starting Multi-Cam Inference with Face Tracking. Press 'Q' to quit.")

    while True:
        frame_count += 1
        
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
            
            # --- PERSON DETECTION (For Tracking) ---
            # Use standard YOLO person class (class 0) to detect people from all angles
            res_person = model_person.track(frame, conf=0.25, persist=True, classes=[0], imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            
            # Track person bounding boxes
            tracked_persons = {}  # {tracker_id: bbox}
            if res_person and len(res_person) > 0:
                result = res_person[0]
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    for b, tid in zip(boxes, ids):
                        tracked_persons[tid] = b
                        print(f"[PersonDetect] Person detected: tracker {tid}")
            
            # --- PPE DETECTION (All Cameras) ---
            res_h = model_helmet.track(frame, conf=BASE_CONF, persist=True, classes=helmet_ids, imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            res_v = model_vest.track(frame, conf=BASE_CONF, persist=True, classes=vest_ids, imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            
            # Collect all detections
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
                            
                            # MAP NEW MODEL CLASSES TO TARGET CLASSES
                            if c_name in ['helmet', 'hi-viz helmet']:
                                c_name = 'Hardhat'
                            elif c_name == 'head':
                                c_name = 'NO-Hardhat'
                                
                            all_detections.append((b, tid, c_name, cf))
                    elif result.boxes.xyxy is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        clss = result.boxes.cls.cpu().numpy().astype(int)
                        confs = result.boxes.conf.cpu().numpy()
                        for b, c, cf in zip(boxes, clss, confs):
                            c_name = model.names[c]

                            # MAP NEW MODEL CLASSES TO TARGET CLASSES
                            if c_name in ['helmet', 'hi-viz helmet']:
                                c_name = 'Hardhat'
                            elif c_name == 'head':
                                c_name = 'NO-Hardhat'

                            all_detections.append((b, -1, c_name, cf))

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
                if (cam_idx, tracker_id) in tracker_person_map:
                    person_id = tracker_person_map[(cam_idx, tracker_id)]
                else:
                    # Query central manager
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
                        # Unknown person (not yet recognized)
                        tracker_person_map[(cam_idx, tracker_id)] = "???"

            # --- DRAW PERSISTENT PERSON ID BOXES ---
            # Draw Person ID box at HEAD LEVEL for each tracked person (like a face box)
            print(f"[Debug] Tracked persons: {len(tracked_persons)}, Person map: {tracker_person_map}")  # Debug
            for tracker_id, person_bbox in tracked_persons.items():
                person_id = tracker_person_map.get((cam_idx, tracker_id), None)
                print(f"[Debug] Tracker {tracker_id}: Person ID = {person_id}")  # Debug
                
                # Skip if this person already has a face box drawn in this frame
                if tracker_id in trackers_with_faces:
                    continue
                
                # Only draw if person has a confirmed ID (not "???")
                if person_id and person_id != "???":
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
                    if person_id.startswith('U'):
                        # Unauthorized
                        id_color = (0, 0, 255)  # Red
                        id_label = f"Unauthorized {person_id}"
                    else:
                        # Authorized
                        id_color = (0, 255, 0)  # Green
                        id_label = f"Person {person_id}"
                    
                    print(f"[Debug] Drawing persistent box for {id_label}")  # Debug
                    # Draw "Face Box" at head level
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), id_color, 3)
                    
                    # Draw Person ID label at bottom of face box
                    (pid_w, pid_h), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (fx1, fy2 + 5), (fx1 + pid_w + 10, fy2 + pid_h + 15), id_color, -1)
                    cv2.putText(frame, id_label, (fx1 + 5, fy2 + pid_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- DRAWING ---
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
