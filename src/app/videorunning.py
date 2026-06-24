"""
Video PPE Detection - Safety Vest and Helmet Detection
Processes video file and displays/saves results with PPE detection (no face tracking)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "Check Video.mp4")
OUTPUT_DIR = "output_videos"

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELMET_MODEL_PATH = os.path.join(BASE_DIR, "helmet.pt")
VEST_MODEL_PATH = os.path.join(BASE_DIR, "vest.pt")

# Settings
BASE_CONF = 0.001  # LOWEST POSSIBLE - detect anything
INFERENCE_IMGSZ = 1280  # Increased from 640 for better distant detection (4x slower but more accurate)
IOU_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.4

# GPU Optimization
USE_GPU = True
USE_FP16 = False

TARGET_CLASSES = {
    'Safety Vest': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.001, 'min_area': 1, 'max_ratio': 10.0},
    'Hardhat': {'color': (0, 255, 0), 'label': 'SAFE', 'conf': 0.20, 'min_area': 100, 'max_ratio': 8.0},
    'NO-Safety Vest': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.80, 'min_area': 100, 'max_ratio': 5.0},
    'NO-Safety Vest (Inferred)': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.0, 'min_area': 1, 'max_ratio': 20.0},
    'NO-Hardhat': {'color': (0, 0, 255), 'label': 'VIOLATION', 'conf': 0.50, 'min_area': 150, 'max_ratio': 5.0}
}


def get_target_class_ids(model, specific_labels):
    ids = []
    for id, name in model.names.items():
        if name in specific_labels:
            ids.append(id)
    return ids


def main():
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"[Error] Video file not found: {VIDEO_PATH}")
        return
    
    # Load PPE Detection Models
    print("Loading PPE detection models...")
    model_helmet = YOLO(HELMET_MODEL_PATH)
    model_vest = YOLO(VEST_MODEL_PATH)
    
    # Load person detection model
    print("Loading person detection model...")
    model_person = YOLO('yolov8n.pt')
    
    # Move models to GPU
    if USE_GPU:
        print("🚀 Moving models to GPU (CUDA) for maximum performance...")
        model_helmet = model_helmet.to('cuda:0')
        model_vest = model_vest.to('cuda:0')
        model_person = model_person.to('cuda:0')
        print("✅ Models loaded on GPU!")
        
        # Monitor GPU memory
        import torch
        if torch.cuda.is_available():
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
    
    # Get class IDs
    helmet_ids = get_target_class_ids(model_helmet, ['Hardhat', 'NO-Hardhat', 'head', 'helmet', 'hi-viz helmet'])
    vest_ids = get_target_class_ids(model_vest, ['Safety Vest', 'NO-Safety Vest'])
    
    # Open video file
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("[Error] Failed to open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")
    
    # Create output video writer
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"ppe_detection_{timestamp_str}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Output will be saved to: {output_path}")
    print("Processing video... Press 'Q' to quit early.")
    
    # State for smoothing
    smoothed_boxes = {}
    frame_count = 0
    
    # Create window
    window_name = 'PPE Detection - Video Processing'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        frame_count += 1
        
        # Show progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Person detection - LOWERED to catch distant workers
        res_person = model_person.track(
            frame,
            conf=0.40,  # Lowered from 0.60 to detect distant persons
            persist=True,
            classes=[0],
            imgsz=INFERENCE_IMGSZ,
            verbose=False,
            iou=IOU_THRESHOLD,
            half=USE_FP16,
            device=0 if USE_GPU else 'cpu',
            max_det=15,  # Increased from 10
            agnostic_nms=True
        )
        
        tracked_persons = {}
        
        if res_person and len(res_person) > 0:
            result = res_person[0]
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                
                for b, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = b
                    width_p = x2 - x1
                    height_p = y2 - y1
                    area = width_p * height_p
                    aspect_ratio = height_p / width_p if width_p > 0 else 0
                    
                    # Validation - LOWERED for distant workers
                    if area < 1500:  # Reduced from 3500 to catch distant persons
                        continue
                    if not (1.3 <= aspect_ratio <= 4.5):  # More permissive
                        continue
                    
                    tracked_persons[tid] = b
        
        # PPE Detection
        all_detections = []
        
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
                        if c_name in ['helmet', 'hi-viz helmet']:
                            c_name = 'Hardhat'
                        elif c_name == 'head':
                            c_name = 'NO-Hardhat'
                        all_detections.append((b, -1, c_name, cf))
        
        # --- CONFLICT RESOLUTION: Prioritize Positive over Negative ---
        # Group detections by person/location to resolve conflicts
        person_detections = {}  # {person_id: list of detections}
        
        for detection in all_detections:
            box, track_id, class_name, conf = detection
            
            # Find which person this detection belongs to
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            # Match to tracked person by overlap
            matched_person = None
            for tid, pbox in tracked_persons.items():
                px1, py1, px2, py2 = pbox
                if px1 < cx < px2 and py1 < cy < py2:
                    matched_person = tid
                    break
            
            # Use tracked person ID or create a spatial key
            person_key = matched_person if matched_person is not None else f"untracked_{int(cx)}_{int(cy)}"
            
            if person_key not in person_detections:
                person_detections[person_key] = []
            person_detections[person_key].append(detection)
        
        # Filter out conflicting detections
        filtered_detections = []
        for person_key, detections in person_detections.items():
            # Check what this person has
            has_safety_vest = any(d[2] == 'Safety Vest' for d in detections)
            has_hardhat = any(d[2] == 'Hardhat' for d in detections)
            
            for detection in detections:
                box, track_id, class_name, conf = detection
                
                # Skip negative detections if positive exists
                if class_name == 'NO-Safety Vest' and has_safety_vest:
                    continue  # Don't draw NO-Safety Vest if Safety Vest detected
                if class_name == 'NO-Hardhat' and has_hardhat:
                    continue  # Don't draw NO-Hardhat if Hardhat detected
                
                filtered_detections.append(detection)
        
        # Use filtered detections for drawing
        all_detections = filtered_detections
        
        # --- FALLBACK: Mark persons WITHOUT vest detection as violations ---
        # If person detected but NO vest info (neither positive nor negative), assume violation
        persons_with_vest_detection = set()
        for detection in all_detections:
            box, track_id, class_name, conf = detection
            if 'Vest' in class_name:
                # Find which person this belongs to
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                for tid, pbox in tracked_persons.items():
                    px1, py1, px2, py2 = pbox
                    if px1 < cx < px2 and py1 < cy < py2:
                        persons_with_vest_detection.add(tid)
                        break
        
        # Add red person boxes for those without vest detection
        for tid, person_bbox in tracked_persons.items():
            if tid not in persons_with_vest_detection:
                # No vest detection at all - mark as violation
                # Add a person-level violation box
                all_detections.append((person_bbox, tid, 'NO-Safety Vest (Inferred)', 0.99))
        
        # Draw person boxes (optional, just for visualization)
        for tid, person_bbox in tracked_persons.items():
            px1, py1, px2, py2 = map(int, person_bbox)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (200, 200, 200), 1)
            cv2.putText(frame, f"Person {tid}", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw PPE detections
        for box, track_id, class_name, conf in all_detections:
            unique_id = f"{class_name}_{track_id}"
            
            config_class = TARGET_CLASSES.get(class_name)
            if not config_class:
                continue
            
            if conf < config_class['conf']:
                continue
            
            # Geometric Filter
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w * h < config_class['min_area']:
                continue
            if h / w > config_class['max_ratio'] if w > 0 else 0:
                continue
            
            # Smoothing
            if unique_id in smoothed_boxes:
                smooth_box = SMOOTHING_FACTOR * box + (1 - SMOOTHING_FACTOR) * smoothed_boxes[unique_id]
            else:
                smooth_box = box
            smoothed_boxes[unique_id] = smooth_box
            
            # Adjust box for helmet
            display_box = smooth_box.copy()
            if class_name in ['Hardhat', 'NO-Hardhat']:
                box_height = smooth_box[3] - smooth_box[1]
                helmet_height_ratio = 0.4
                upward_shift = box_height * 0.15
                display_box[1] = smooth_box[1] - upward_shift
                display_box[3] = display_box[1] + (box_height * helmet_height_ratio)
            
            # Draw box
            sx1, sy1, sx2, sy2 = map(int, display_box)
            color = config_class['color']
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
            
            # Label
            label = f"{class_name} ({conf:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (sx1, sy1 - label_h - 10), (sx1 + label_w, sy1), color, -1)
            cv2.putText(frame, label, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(all_detections)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output
        out.write(frame)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested quit")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Processing complete!")
    print(f"📹 Output saved to: {output_path}")
    print(f"📊 Processed {frame_count} frames")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
