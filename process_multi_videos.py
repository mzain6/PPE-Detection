"""
Multi-Camera Video Processor (3 Streams) for PPE Detection
Processes camera0.mp4 (Entrance), camera1.mp4 (Side), and camera2.mp4 (Side) concurrently.
Ensures ID persistence across cameras and strict 1x playback speed.
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
OUTPUT_PATH = os.path.join(VIDEO_DIR, "multi_cam_output.mp4")

CAM_CONFIG = [
    # Cam 0: Entrance camera w/ Face Rec
    {"id": "Cam 0", "path": os.path.join(VIDEO_DIR, "camera0.mp4"), "is_entrance": True},
    # Cam 1: Side camera
    {"id": "Cam 1", "path": os.path.join(VIDEO_DIR, "camera1.mp4"), "is_entrance": False},
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
OUTPUT_FPS = 30.0

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
    print("Multi-Camera (3) PPE Processing - 1x Speed Sync")
    print("="*60)

    # 1. Initialize Managers
    print("Initializing Managers...")
    central_manager = CentralTrackingManager()
    face_manager = None
    try:
        face_manager = FaceRecognitionManager(face_db_path=FACE_DB_PATH, similarity_threshold=FACE_SIMILARITY_THRESHOLD)
    except Exception as e:
        print(f"[Warning] Face usage limited: {e}")

    # 2. Load Models
    print("Loading models...")
    model_helmet = YOLO(HELMET_MODEL_PATH)
    model_vest = YOLO(VEST_MODEL_PATH)
    model_person = YOLO('yolov8n.pt')
    
    helmet_ids = get_target_class_ids(model_helmet, ['Hardhat', 'NO-Hardhat', 'head', 'helmet', 'hi-viz helmet'])
    vest_ids = get_target_class_ids(model_vest, ['Safety Vest', 'NO-Safety Vest'])

    # 3. Open Video Streams & Analyzers
    caps = []
    cam_fps = []
    cam_frames = []
    
    max_duration = 0
    
    for config in CAM_CONFIG:
        path = config['path']
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            return
            
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frames / fps if fps > 0 else 0
        
        print(f"Opened {config['id']}: {frames} frames @ {fps:.2f} fps ({duration:.1f}s)")
        
        if duration > max_duration:
            max_duration = duration
            
        caps.append(cap)
        cam_fps.append(fps)
        cam_frames.append(None) # Store current frame
        
    # 4. Setup Video Writer
    total_output_frames = int(max_duration * OUTPUT_FPS)
    print(f"Generating {total_output_frames} frames ({max_duration:.1f}s) at {OUTPUT_FPS} FPS...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, OUTPUT_FPS, GRID_SIZE)

    # State Tracking
    smoothed_boxes = {}
    tracker_person_map = {}
    
    # Process Loop
    for out_frame_idx in tqdm(range(total_output_frames), desc="Processing"):
        current_time_sec = out_frame_idx / OUTPUT_FPS
        
        processed_frames = []
        
        # Determine strict frame index for each camera at this timestamp
        
        for i, cap in enumerate(caps):
            fps = cam_fps[i]
            target_frame_idx = int(current_time_sec * fps)
            
            # Grab frame if we are behind
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            frame = cam_frames[i] # Default to previous frame (hold)
            
            if current_pos <= target_frame_idx:
                # Need to read forward
                ret = True
                while current_pos <= target_frame_idx and ret:
                    ret, new_frame = cap.read()
                    if ret:
                        cam_frames[i] = new_frame
                        frame = new_frame
                    current_pos += 1
            
            if frame is None:
                # Video ended or not started, show black
                frame = np.zeros((CAM_SIZE[1], CAM_SIZE[0], 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, CAM_SIZE)
                
            # --- PROCESS THIS FRAME ---
            # We process every output frame to ensure tracking smoothness on the visualization
            # identifying/tracking logic works best on continuous frames
            
            config = CAM_CONFIG[i]
            is_entrance = config['is_entrance']
            camera_id = config['id']
            
            # Copy frame for drawing
            draw_frame = frame.copy()
            
            # Draw "Active" indicator
            cv2.circle(draw_frame, (30, 30), 10, (0, 255, 0), -1)
            
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
                            if c_name in ['helmet', 'hi-viz helmet']: c_name = 'Hardhat'
                            elif c_name == 'head': c_name = 'NO-Hardhat'
                            all_detections.append((b, tid, c_name, cf))
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
            if is_entrance and face_manager:
                try:
                    # Run face detection
                    face_results = face_manager.process_frame_for_faces(frame)
                    
                    for face_result in face_results:
                        person_id = face_result['person_id']
                        face_bbox = face_result['bbox']
                        
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
                        
                        if best_tracker_id is not None and best_distance < 100:
                             if person_id:
                                central_manager.register_person(
                                    person_id=person_id,
                                    tracker_id=best_tracker_id,
                                    camera_id=camera_id,
                                    bbox=tracked_persons[best_tracker_id]
                                )
                                tracker_person_map[(i, best_tracker_id)] = person_id
                             else:
                                if (i, best_tracker_id) not in tracker_person_map:
                                    u_id = central_manager.register_unauthorized_person(
                                        tracker_id=best_tracker_id,
                                        camera_id=camera_id, 
                                        bbox=tracked_persons[best_tracker_id]
                                    )
                                    tracker_person_map[(i, best_tracker_id)] = u_id

                except Exception as e:
                    pass

            # --- ID ASSIGNMENT AND DISPLAY ---
            for tracker_id, bbox in tracked_persons.items():
                if (i, tracker_id) in tracker_person_map:
                    person_id = tracker_person_map[(i, tracker_id)]
                    central_manager.update_tracker_bbox(tracker_id, camera_id, bbox)
                else:
                    pid = central_manager.get_person_by_tracker(tracker_id, camera_id)
                    if not pid:
                        pid = central_manager.identify_new_tracker(tracker_id, camera_id, bbox)
                    
                    if pid:
                        tracker_person_map[(i, tracker_id)] = pid
                    else:
                        tracker_person_map[(i, tracker_id)] = "???"

            # Draw Person Boxes + IDs
            for tracker_id, person_bbox in tracked_persons.items():
                person_id = tracker_person_map.get((i, tracker_id), "???")
                
                px1, py1, px2, py2 = map(int, person_bbox)
                color = (0, 255, 0)
                if person_id and person_id.startswith('U'): color = (0, 0, 255)
                elif person_id == "???": color = (200, 200, 200)

                cv2.rectangle(draw_frame, (px1, py1), (px2, py2), color, 2)
                
                label = f"ID: {person_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(draw_frame, (px1, py1-h-10), (px1+w, py1), color, -1)
                cv2.putText(draw_frame, label, (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Draw PPE Detections
            for box, track_id, class_name, conf in all_detections:
                 config_class = TARGET_CLASSES.get(class_name)
                 if not config_class or conf < config_class['conf']: continue
                 
                 bx1, by1, bx2, by2 = map(int, box)
                 color = config_class['color']
                 cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2), color, 2)
                 cv2.putText(draw_frame, class_name, (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add Camera Label
            label_text = f"{camera_id}"
            if is_entrance: label_text += " (ENTRANCE)"
            cv2.putText(draw_frame, label_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            processed_frames.append(draw_frame)

        # Stitch into Grid (2x2)
        if len(processed_frames) > 0:
            cols = 2
            rows = 2
            total_slots = cols * rows
            
            while len(processed_frames) < total_slots:
                processed_frames.append(np.zeros((CAM_SIZE[1], CAM_SIZE[0], 3), dtype=np.uint8))
            
            grid_rows = []
            for r in range(rows):
                row_frames = processed_frames[r*cols : (r+1)*cols]
                grid_rows.append(np.hstack(row_frames))
                
            grid = np.vstack(grid_rows)
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
