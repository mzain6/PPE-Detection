"""
PPE Detection with best1.pt Model - Optimized for Distance Detection
Runs on PPE Test 2.mp4 with optimized parameters for detecting workers at a distance.

Key optimizations:
- Lower confidence threshold (0.25) to catch smaller/distant objects
- IoU threshold (0.45) to handle overlapping workers
- Higher image size (1280) for better detection at distance
- Saves annotated output as output_test.mp4
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from pathlib import Path

# File paths
CURRENT_DIR = Path(__file__).parent
MODEL_PATH = CURRENT_DIR / "best1.pt"
INPUT_VIDEO = CURRENT_DIR / "PPE Test 2.mp4"
OUTPUT_VIDEO = CURRENT_DIR / "output_test.mp4"

print("=" * 60)
print("PPE Detection - best1.pt Model (Distance Optimized)")
print("=" * 60)

# Verify files exist
if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    exit(1)

if not INPUT_VIDEO.exists():
    print(f"ERROR: Input video not found: {INPUT_VIDEO}")
    exit(1)

print(f"\nInput: {INPUT_VIDEO}")
print(f"Output: {OUTPUT_VIDEO}")
print(f"Model: {MODEL_PATH}")

# Load YOLOv8 model
print("\nLoading YOLOv8 model...")
model = YOLO(str(MODEL_PATH))
print("✓ Model loaded successfully!")

# Distance detection parameters - OPTIMIZED
CONF_THRESHOLD = 0.25   # Lower confidence to catch distant/small objects
IOU_THRESHOLD = 0.45    # Handle overlapping workers better
IMG_SIZE = 1280         # Higher resolution for better distance detection (try 640 if too slow)

print(f"\nDetection Parameters (Optimized for Distance):")
print(f"  Confidence Threshold: {CONF_THRESHOLD}")
print(f"  IoU Threshold: {IOU_THRESHOLD}")
print(f"  Image Size: {IMG_SIZE}")

# Open video
cap = cv2.VideoCapture(str(INPUT_VIDEO))
if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo Properties:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total Frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.1f} seconds")

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

print(f"\nProcessing video...")
print(f"This may take a while depending on video length and hardware...")

frame_n = 0
start_time = time.time()
detection_stats = {
    "total_persons": 0,
    "total_helmets": 0,
    "total_vests": 0,
    "frames_with_detections": 0
}

# Get class names from model
class_names = model.names if hasattr(model, 'names') else {}
print(f"\nModel Classes: {class_names}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_n += 1
    
    # Progress update every 30 frames
    if frame_n % 30 == 0 or frame_n == 1:
        elapsed = time.time() - start_time
        fps_actual = frame_n / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_n) / fps_actual if fps_actual > 0 else 0
        print(f"  Frame {frame_n}/{total_frames} ({100*frame_n/total_frames:.1f}%) - "
              f"{fps_actual:.1f} fps - ETA: {eta:.0f}s")
    
    # Run detection with optimized parameters
    results = model(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )
    
    # Process detections
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        persons_count = 0
        helmets_count = 0
        vests_count = 0
        
        if boxes is not None and len(boxes) > 0:
            detection_stats["frames_with_detections"] += 1
            
            for box in boxes:
                # Get class ID and name
                cls_id = int(box.cls.cpu().numpy()[0])
                class_name = class_names.get(cls_id, f"class_{cls_id}").lower()
                
                # Get confidence
                conf = float(box.conf.cpu().numpy()[0])
                
                # Get bounding box
                x1, y1, x2, y2 = [int(v) for v in box.xyxy.cpu().numpy()[0]]
                
                # Count detections by type
                if "person" in class_name or "worker" in class_name:
                    persons_count += 1
                    color = (0, 255, 0)  # Green for person
                    label_prefix = "PERSON"
                elif "helmet" in class_name or "head" in class_name or "hard" in class_name:
                    helmets_count += 1
                    color = (255, 200, 0)  # Cyan for helmet
                    label_prefix = "HELMET"
                elif "vest" in class_name or "jacket" in class_name or "safety" in class_name:
                    vests_count += 1
                    color = (0, 165, 255)  # Orange for vest
                    label_prefix = "VEST"
                else:
                    color = (255, 255, 255)  # White for other
                    label_prefix = class_name.upper()
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{label_prefix} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Update statistics
        detection_stats["total_persons"] += persons_count
        detection_stats["total_helmets"] += helmets_count
        detection_stats["total_vests"] += vests_count
        
        # Draw frame info overlay
        info_text = f"Frame: {frame_n} | Persons: {persons_count} | Helmets: {helmets_count} | Vests: {vests_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Write annotated frame
    out.write(frame)

# Cleanup
cap.release()
out.release()

# Final statistics
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"✓ Processing Complete!")
print(f"{'='*60}")
print(f"\nPerformance:")
print(f"  Total Frames: {frame_n}")
print(f"  Processing Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  Average FPS: {frame_n/elapsed:.1f}")

print(f"\nDetection Statistics:")
print(f"  Frames with detections: {detection_stats['frames_with_detections']} "
      f"({100*detection_stats['frames_with_detections']/frame_n:.1f}%)")
print(f"  Total person detections: {detection_stats['total_persons']}")
print(f"  Total helmet detections: {detection_stats['total_helmets']}")
print(f"  Total vest detections: {detection_stats['total_vests']}")
print(f"  Avg persons/frame: {detection_stats['total_persons']/frame_n:.2f}")

print(f"\nOutput saved to: {OUTPUT_VIDEO}")
print(f"File size: {os.path.getsize(OUTPUT_VIDEO) / (1024*1024):.1f} MB")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("\nIf detection quality is not satisfactory, try:")
print("  1. If too slow: Change IMG_SIZE from 1280 to 640 (line 33)")
print("  2. If missing detections: Lower CONF_THRESHOLD to 0.2 (line 31)")
print("  3. If too many false positives: Raise CONF_THRESHOLD to 0.3 (line 31)")
print("  4. If workers are merging: Lower IOU_THRESHOLD to 0.3 (line 32)")
print("\nFor the best balance between speed and accuracy:")
print("  - Use IMG_SIZE=640 for real-time processing")
print("  - Use IMG_SIZE=1280 for maximum detection quality (slower)")
print("="*60)
