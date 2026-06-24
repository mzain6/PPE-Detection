"""
Dual Video Processor for PPE Detection System
Processes camera1.mp4 (Entrance) and camera2.mp4 (Side) concurrently.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from tqdm import tqdm

# Import our custom modules
from face_recognition_manager import FaceRecognitionManager
from central_tracking_manager import CentralTrackingManager

# --- CONFIGURATION ---
VIDEO_DIR = r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\videos"
OUTPUT_PATH = os.path.join(VIDEO_DIR, "dual_cam_output.mp4")

CAM_CONFIG = [
    # Cam 1: Entrance camera with face recognition
    {"id": "Cam 1", "path": os.path.join(VIDEO_DIR, "camera1.mp4"), "is_entrance": True},
    # Cam 2: Side camera
    {"id": "Cam 2", "path": os.path.join(VIDEO_DIR, "camera2.mp4"), "is_entrance": False},
]

# Paths
HELMET_MODEL_PATH = r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\helmet.pt"
VEST_MODEL_PATH = r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\vest.pt"
FACE_DB_PATH = r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\face_database.pkl"

# Settings
BASE_CONF = 0.25
GRID_SIZE = (1280, 720)
CAM_SIZE = (640, 360)
INFERENCE_IMGSZ = 640
IOU_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.4
FACE_SIMILARITY_THRESHOLD = 0.3

# Class-Specific Logic
TARGET_CLASSES = {
    'Safety Vest': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.10, 'min_area': 400, 'max_ratio': 3.0},
    'Hardhat': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.50, 'min_area': 450, 'max_ratio': 3.0},
    'NO-Safety Vest': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.20, 'min_area': 400, 'max_ratio': 3.0},
    'NO-Hardhat': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.15, 'min_area': 300, 'max_ratio': 3.5}
}

def get_target_class_ids(model, specific_labels):
    ids = []
    for id, name in model.names.items():
        if name in specific_labels:
            ids.append(id)
    return ids

def main():
    print("="*60)
    print("Dual Video PPE Processing")
    print("="*60)

    # 1. Initialize Managers
    print("Initializing Central Tracking Manager...")
    central_manager = CentralTrackingManager()
    
    print("Initializing Face Recognition Manager...")
    face_manager = None
    try:
        face_manager = FaceRecognitionManager(face_db_path=FACE_DB_PATH, similarity_threshold=FACE_SIMILARITY_THRESHOLD)
    except Exception as e:
        print(f"[Warning] Face recognition init failed: {e}")

    # 2. Load Models
    print("Loading models (this may take a moment)...")
    model_helmet = YOLO(HELMET_MODEL_PATH)
    model_vest = YOLO(VEST_MODEL_PATH)
    model_person = YOLO('yolov8n.pt')
    
    helmet_ids = get_target_class_ids(model_helmet, ['Hardhat', 'NO-Hardhat', 'head', 'helmet', 'hi-viz helmet'])
    vest_ids = get_target_class_ids(model_vest, ['Safety Vest', 'NO-Safety Vest'])

    # 3. Open Video Streams
    caps = []
    total_frames = 0
    
    for config in CAM_CONFIG:
        path = config['path']
        if not os.path.exists(path):
            print(f"Error: Video not found at {path}")
            return
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            return
            
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Opened {config['id']}: {frames} frames @ {fps:.1f} fps")
        
        # We'll use the shorter video length to determine total processing length
        if total_frames == 0 or frames < total_frames:
            total_frames = frames
            
        caps.append(cap)
        
    # 4. Setup Video Writer
    msg = f"Processing {total_frames} frames..."
    print(msg)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, GRID_SIZE) # Fixed 30fps for output

    # State Tracking
    smoothed_boxes = {}
    tracker_person_map = {}
    trackers_with_faces = set()

    # Process Loop
    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        current_frames = []
        
        # Read frame from each camera
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                current_frames.append(None)
            else:
                frame = cv2.resize(frame, CAM_SIZE)
                current_frames.append(frame)
        
        if any(f is None for f in current_frames):
            print("End of video stream reached.")
            break
            
        processed_frames = []
        
        # Process each camera's frame
        for cam_idx, frame in enumerate(current_frames):
            config = CAM_CONFIG[cam_idx]
            is_entrance = config['is_entrance']
            camera_id = config['id']
            
            # Draw "Active" indicator (simulated live feed)
            cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)
            
            # --- PERSON DETECTION & TRACKING ---
            res_person = model_person.track(frame, conf=0.25, persist=True, classes=[0], imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            
            tracked_persons = {} 
            if res_person and len(res_person) > 0:
                result = res_person[0]
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    for b, tid in zip(boxes, ids):
                        tracked_persons[tid] = b

            # --- PPE DETECTION ---
            res_h = model_helmet.track(frame, conf=BASE_CONF, persist=True, classes=helmet_ids, imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            res_v = model_vest.track(frame, conf=BASE_CONF, persist=True, classes=vest_ids, imgsz=INFERENCE_IMGSZ, verbose=False, iou=IOU_THRESHOLD)
            
            all_detections = []
             # Collect all detections
            for res, model in [(res_h, model_helmet), (res_v, model_vest)]:
                if res and len(res) > 0:
                    result = res[0]
                    # Handle tracked detections
                    if result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        ids = result.boxes.id.cpu().numpy().astype(int)
                        clss = result.boxes.cls.cpu().numpy().astype(int)
                        confs = result.boxes.conf.cpu().numpy()
                        for b, tid, c, cf in zip(boxes, ids, clss, confs):
                            c_name = model.names[c]
                            # Map class names
                            if c_name in ['helmet', 'hi-viz helmet']: c_name = 'Hardhat'
                            elif c_name == 'head': c_name = 'NO-Hardhat'
                            all_detections.append((b, tid, c_name, cf))
                    # Handle untracked detections
                    elif result.boxes.xyxy is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        clss = result.boxes.cls.cpu().numpy().astype(int)
                        confs = result.boxes.conf.cpu().numpy()
                        for b, c, cf in zip(boxes, clss, confs):
                            c_name = model.names[c]
                            if c_name in ['helmet', 'hi-viz helmet']: c_name = 'Hardhat'
                            elif c_name == 'head': c_name = 'NO-Hardhat'
                            all_detections.append((b, -1, c_name, cf))

            # --- FACE RECOGNITION (Entrance Only) ---
            face_boxes_to_draw = []
            trackers_with_faces = set()
            
            if is_entrance and face_manager:
                try:
                    # Only process faces every 3rd frame to speed up processing
                    # But for now, let's do every frame for accuracy in this demo
                    face_results = face_manager.process_frame_for_faces(frame)
                    
                    for face_result in face_results:
                        person_id = face_result['person_id']
                        face_bbox = face_result['bbox']
                        similarity = face_result['similarity']
                        authorized = face_result['authorized']
                        
                        face_boxes_to_draw.append(face_result)
                        
                        # Find closest tracked person
                        best_tracker_id = None
                        best_distance = float('inf')
                        
                        for tracker_id, person_bbox in tracked_persons.items():
                            fx = (face_bbox[0] + face_bbox[2]) / 2
                            fy = (face_bbox[1] + face_bbox[3]) / 2
                            px = (person_bbox[0] + person_bbox[2]) / 2
                            py = (person_bbox[1] + person_bbox[3]) / 2
                            dist = np.sqrt((fx-px)**2 + (fy-py)**2)
                            
                            if dist < best_distance:
                                best_distance = dist
                                best_tracker_id = tracker_id
                        
                        if best_tracker_id is not None and best_distance < 100: # Pixel distance threshold
                             if person_id:
                                central_manager.register_person(
                                    person_id=person_id,
                                    tracker_id=best_tracker_id,
                                    camera_id=camera_id,
                                    bbox=tracked_persons[best_tracker_id]
                                )
                                tracker_person_map[(cam_idx, best_tracker_id)] = person_id
                                trackers_with_faces.add(best_tracker_id)
                             else:
                                # Unauthorized
                                # Check if already has ID
                                if (cam_idx, best_tracker_id) not in tracker_person_map:
                                    u_id = central_manager.register_unauthorized_person(
                                        tracker_id=best_tracker_id,
                                        camera_id=camera_id, 
                                        bbox=tracked_persons[best_tracker_id]
                                    )
                                    tracker_person_map[(cam_idx, best_tracker_id)] = u_id
                                trackers_with_faces.add(best_tracker_id)

                except Exception as e:
                    pass

            # --- ID ASSIGNMENT AND DISPLAY ---
            # Update person locations in central manager
            for tracker_id, bbox in tracked_persons.items():
                if (cam_idx, tracker_id) in tracker_person_map:
                    person_id = tracker_person_map[(cam_idx, tracker_id)]
                    central_manager.update_tracker_bbox(tracker_id, camera_id, bbox)
                else:
                    # Look up or identify
                    pid = central_manager.get_person_by_tracker(tracker_id, camera_id)
                    if not pid:
                        pid = central_manager.identify_new_tracker(tracker_id, camera_id, bbox)
                    
                    if pid:
                        tracker_person_map[(cam_idx, tracker_id)] = pid
                    else:
                        tracker_person_map[(cam_idx, tracker_id)] = "???"

            # Draw Person Boxes + IDs
            for tracker_id, person_bbox in tracked_persons.items():
                person_id = tracker_person_map.get((cam_idx, tracker_id), "???")
                
                # Draw main person box
                px1, py1, px2, py2 = map(int, person_bbox)
                
                # Only draw ID box if not already drawing a face match box
                # or if we want to show the tracked person context
                color = (0, 255, 0)
                if person_id and person_id.startswith('U'):
                    color = (0, 0, 255)
                elif person_id == "???":
                    color = (200, 200, 200)

                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                
                # Label
                label = f"ID: {person_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (px1, py1-h-10), (px1+w, py1), color, -1)
                cv2.putText(frame, label, (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Draw PPE Detections
            for box, track_id, class_name, conf in all_detections:
                 unique_id = f"C{cam_idx}_{class_name}_{track_id}"
                 config_class = TARGET_CLASSES.get(class_name)
                 
                 if not config_class or conf < config_class['conf']: continue
                 
                 # Basic smoothing
                 if unique_id in smoothed_boxes:
                     box = SMOOTHING_FACTOR * box + (1 - SMOOTHING_FACTOR) * smoothed_boxes[unique_id]
                 smoothed_boxes[unique_id] = box
                 
                 bx1, by1, bx2, by2 = map(int, box)
                 color = config_class['color']
                 
                 cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                 cv2.putText(frame, class_name, (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add Camera Label
            label_text = f"{camera_id}"
            if is_entrance: label_text += " (ENTRANCE)"
            cv2.putText(frame, label_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            processed_frames.append(frame)

        # Create Grid
        if len(processed_frames) == 2:
            # Stack horizontally
            grid = np.hstack(processed_frames)
            # Resize to output size
            grid = cv2.resize(grid, GRID_SIZE)
            out.write(grid)

    # Cleanup
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"\nProcessing Complete!")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
