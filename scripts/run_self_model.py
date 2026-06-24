
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
from collections import deque, Counter

# Paths
HELMET_MODEL_PATH = r"C:\Users\hashi\Downloads\PPE-Detection-2\PPE-Detection-2\best1.pt"
VEST_MODEL_PATH = r"C:\Users\hashi\Downloads\PPE-Detection-2\PPE-Detection-2\check.pt" # Updated to new model
VIDEO_PATH = r"C:\Users\hashi\Downloads\PPE-Detection-2\PPE-Detection-2\PPE Test 2.mp4"

# Global Detection Settings
BASE_CONF = 0.10        # Minimum confidence for tracker initialization
IMG_SIZE = 640          # LOWERED from 1280 to 640 for speed/smoothness
IOU_THRESHOLD = 0.5     # Tracking IOU
NMS_THRESHOLD = 0.3     # Overlap threshold
SMOOTHING_FACTOR = 0.4  # Box position smoothing

# Temporal Stability
HISTORY_LEN = 8         
STABILITY_THRESHOLD = 0.5 

# Class-Specific Logic
TARGET_CLASSES = {
    'Safety Vest': {
        'color': (0, 255, 0),   
        'label': 'SAFE',      
        'conf': 0.10,       
        'min_area': 800,    
        'max_ratio': 3.0    
    },
    'Hardhat': {
        'color': (0, 255, 0),   
        'label': 'SAFE',      
        'conf': 0.25,       
        'min_area': 900,
        'max_ratio': 3.0
    },
    'NO-Safety Vest': {
        'color': (0, 0, 255),   
        'label': 'VIOLATION', 
        'conf': 0.50,       
        'min_area': 2000,   
        'max_ratio': 2.5    
    },
    'NO-Hardhat': {
        'color': (0, 0, 255),   
        'label': 'VIOLATION', 
        'conf': 0.50,       
        'min_area': 2000,
        'max_ratio': 2.5
    }
}

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
    # GPU Check
    device = 'cpu'
    if torch.cuda.is_available():
        device = '0'
        print(f"CUDA Available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA NOT Available. Using CPU (might be slow with imgsz=1280).")

    # Load Models
    print(f"Loading HELMET model from {HELMET_MODEL_PATH}...")
    model_helmet = YOLO(HELMET_MODEL_PATH)
    
    print(f"Loading VEST model from {VEST_MODEL_PATH}...")
    model_vest = YOLO(VEST_MODEL_PATH)

    # Define Responsibility
    # best1.pt -> Hardhats
    # self.pt -> Vests
    helmet_labels = ['Hardhat', 'NO-Hardhat']
    vest_labels = ['Safety Vest', 'NO-Safety Vest']

    helmet_ids = get_target_class_ids(model_helmet, helmet_labels)
    vest_ids = get_target_class_ids(model_vest, vest_labels)
    
    print(f"Helmet Model IDs to track: {helmet_ids} ({helmet_labels})")
    print(f"Vest Model IDs to track: {vest_ids} ({vest_labels})")

    # Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video file: {VIDEO_PATH}")
        return

    # Video Writer
    OUTPUT_PATH = r"C:\Users\hashi\Downloads\PPE-Detection-2\PPE-Detection-2\output_check_model.mp4"
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    print(f"Saving output to: {OUTPUT_PATH}")

    # Window
    window_name = 'Live Detection (Multi-Model: Best1 + Self)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # State Tracking
    smoothed_boxes = {}      
    class_history = {}       
    stable_classes = {}      

    print(f"Starting Multi-Model Inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Run HELMET Inference
        results_helmet = model_helmet.track(
            frame, conf=BASE_CONF, persist=True, classes=helmet_ids, 
            imgsz=IMG_SIZE, device=device, iou=IOU_THRESHOLD, verbose=False
        )
        
        # 2. Run VEST Inference
        results_vest = model_vest.track(
            frame, conf=BASE_CONF, persist=True, classes=vest_ids, 
            imgsz=IMG_SIZE, device=device, iou=IOU_THRESHOLD, verbose=False
        )

        # 3. Combine Detections
        all_detections = [] # list of (box, track_id, class_name, conf)
        
        for results, source_model in [(results_helmet, model_helmet), (results_vest, model_vest)]:
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes.id is not None:
                    current_boxes = boxes.xyxy.cpu().numpy()
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()

                    for box, track_id, cls_id, conf in zip(current_boxes, track_ids, class_ids, confs):
                        # Ensure track_id is unique across models? 
                        # Ideally yes, but here they track totally different objects usually.
                        # To be safe, we can prefix track_id based on class type if needed, 
                        # but "Vest" and "Helmet" are spatially distinct enough usually.
                        # Simple hack: Offset VEST IDs by 10000 to avoid ID collision if they happen to use same numbers?
                        # Actually model.track persist might handle own IDs internally.
                        # Let's use a composite key for our smoothing map: f"{class_type}_{track_id}"
                        
                        class_name = source_model.names[cls_id]
                        all_detections.append((box, track_id, class_name, conf))

        # 4. Process & Visualize
        # Need to re-format for NMS?
        # If we trust the models' internal NMS/Tracking, we just need to draw.
        # But we still apply our Geometric Logic
        
        xywh_boxes_for_nms = []
        detection_metadata = []

        for det in all_detections:
            box, track_id, class_name, conf = det
            x1, y1, x2, y2 = box
            xywh_boxes_for_nms.append([x1, y1, x2-x1, y2-y1])
            detection_metadata.append(det)

        # Global NMS (Optional, but good if models overlap strangely)
        confs_list = [d[3] for d in detection_metadata]
        keep_indices = apply_nms(xywh_boxes_for_nms, confs_list, NMS_THRESHOLD)

        for idx in keep_indices:
            box, track_id, class_name, conf = detection_metadata[idx]
            
            # UNIQUE ID for Smoothing Map (Class + ID) to prevent collision between models
            unique_track_id = f"{class_name}_{track_id}"

            config = TARGET_CLASSES.get(class_name)
            if not config: continue

            # --- FILTERING ---
            if conf < config['conf']: continue
            
            x1, y1, x2, y2 = map(int, box)
            w_box = x2 - x1
            h_box = y2 - y1
            area = w_box * h_box
            ratio = h_box / w_box if w_box > 0 else 0
            
            if area < config['min_area']: continue
            if ratio > config['max_ratio']: continue
            # -----------------

            # --- STABILITY ---
            if unique_track_id not in class_history:
                class_history[unique_track_id] = deque(maxlen=HISTORY_LEN)
                stable_classes[unique_track_id] = class_name 
            
            class_history[unique_track_id].append(class_name)
            counts = Counter(class_history[unique_track_id])
            most_common_class, count = counts.most_common(1)[0]
            
            if count / len(class_history[unique_track_id]) > STABILITY_THRESHOLD:
                stable_classes[unique_track_id] = most_common_class
            
            display_class = stable_classes[unique_track_id]
            display_config = TARGET_CLASSES.get(display_class, config)
            # -----------------

            # --- SMOOTHING ---
            if unique_track_id in smoothed_boxes:
                prev_box = smoothed_boxes[unique_track_id]
                smooth_box = SMOOTHING_FACTOR * box + (1 - SMOOTHING_FACTOR) * prev_box
            else:
                smooth_box = box
            smoothed_boxes[unique_track_id] = smooth_box
            # -----------------

            # --- DRAW ---
            x1, y1, x2, y2 = map(int, smooth_box)
            color = display_config['color']
            label_text = display_config['label'] 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3) 
            
            full_label = f"{display_class}: {label_text} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (w, h), _ = cv2.getTextSize(full_label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w + 10, y1), color, -1)
            cv2.putText(frame, full_label, (x1 + 5, y1 - 10), font, font_scale, (255, 255, 255), thickness)

        # Resize for display (Fix "Zoomed" look)
        # Resize to fit a standard laptop screen (e.g. height 720) while keeping aspect ratio
        display_height = 720
        h, w = frame.shape[:2]
        scale = display_height / h
        display_width = int(w * scale)
        display_frame = cv2.resize(frame, (display_width, display_height))

        cv2.imshow(window_name, display_frame)
        
        # Write to file
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
