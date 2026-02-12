"""
PPE Detection with best1.pt Model - LIVE DISPLAY VERSION
Runs on PPE Test 2.mp4 with optimized parameters for detecting workers at a distance.
Shows live processing window so you can see detections in real-time.

Key optimizations:
- Lower confidence threshold (0.25) to catch smaller/distant objects
- IoU threshold (0.45) to handle overlapping workers
- Higher image size (1280) for better detection at distance
- LIVE DISPLAY - Press 'q' to stop, 's' to skip ahead, SPACE to pause
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
print("LIVE DISPLAY MODE - You'll see detections in real-time!")
print("=" * 60)

# Verify files exist
if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    exit(1)

if not INPUT_VIDEO.exists():
    print(f"ERROR: Input video not found: {INPUT_VIDEO}")
    exit(1)

print(f"\nInput: {INPUT_VIDEO.name}")
print(f"Output: {OUTPUT_VIDEO.name}")
print(f"Model: {MODEL_PATH.name}")

# Load YOLOv8 model
print("\nLoading YOLOv8 model...")
model = YOLO(str(MODEL_PATH))
print("✓ Model loaded successfully!")

# Distance detection parameters - OPTIMIZED
CONF_THRESHOLD = 0.25   # Lower confidence to catch distant/small objects
IOU_THRESHOLD = 0.45    # Handle overlapping workers better
IMG_SIZE = 1280         # Higher resolution for better distance detection

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

print(f"\n{'='*60}")
print("LIVE PROCESSING STARTED")
print("='*60}")
print("\nControls:")
print("  'q' - Quit and save")
print("  's' - Skip forward 10 seconds")
print("  SPACE - Pause/Resume")
print("  'f' - Fast mode (skip display for speed)")
print(f"\n{'='*60}\n")

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
print(f"Model Classes: {class_names}\n")

paused = False
fast_mode = False
window_name = "PPE Detection - Live (Press 'q' to quit, SPACE to pause)"

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_n += 1
        
        # Progress update
        elapsed = time.time() - start_time
        fps_actual = frame_n / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_n) / fps_actual if fps_actual > 0 else 0
        
        # Run detection with optimized parameters
        results = model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )
        
        # Process detections
        display_frame = frame.copy()
        persons_count = 0
        helmets_count = 0
        vests_count = 0
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
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
                    
                    # Count detections by type and set colors
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
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    label = f"{label_prefix} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Update statistics
            detection_stats["total_persons"] += persons_count
            detection_stats["total_helmets"] += helmets_count
            detection_stats["total_vests"] += vests_count
        
        # Draw info overlay
        info_y = 30
        # Status bar background
        cv2.rectangle(display_frame, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (0, 0), (width, 100), (255, 255, 255), 2)
        
        # Frame info
        info_text = f"Frame: {frame_n}/{total_frames} ({100*frame_n/total_frames:.1f}%)"
        cv2.putText(display_frame, info_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection counts
        info_y += 25
        det_text = f"Persons: {persons_count} | Helmets: {helmets_count} | Vests: {vests_count}"
        cv2.putText(display_frame, det_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Performance
        info_y += 25
        perf_text = f"Speed: {fps_actual:.1f} fps | ETA: {eta:.0f}s"
        cv2.putText(display_frame, perf_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        # Progress bar
        bar_width = width - 20
        bar_height = 10
        bar_y = height - 30
        progress = frame_n / total_frames
        cv2.rectangle(display_frame, (10, bar_y), (10 + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(display_frame, (10, bar_y), (10 + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(display_frame, (10, bar_y), (10 + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Write to output file
        out.write(display_frame)
        
        # Show live display unless in fast mode
        if not fast_mode:
            cv2.imshow(window_name, display_frame)
        
        # Print progress to console every 30 frames
        if frame_n % 30 == 0:
            print(f"  Frame {frame_n}/{total_frames} ({100*frame_n/total_frames:.1f}%) - "
                  f"{fps_actual:.1f} fps - ETA: {eta:.0f}s - "
                  f"P:{persons_count} H:{helmets_count} V:{vests_count}")
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n[User stopped processing]")
        break
    elif key == ord(' '):
        paused = not paused
        status = "PAUSED" if paused else "RESUMED"
        print(f"\n[{status}]")
    elif key == ord('s'):
        # Skip forward 10 seconds
        skip_frames = 10 * fps
        new_pos = min(frame_n + skip_frames, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"\n[Skipped to frame {frame_n}]")
    elif key == ord('f'):
        fast_mode = not fast_mode
        status = "ENABLED" if fast_mode else "DISABLED"
        print(f"\n[Fast mode {status}]")
        if fast_mode:
            cv2.destroyAllWindows()

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Final statistics
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"✓ Processing Complete!")
print(f"{'='*60}")
print(f"\nPerformance:")
print(f"  Total Frames Processed: {frame_n}")
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
if os.path.exists(OUTPUT_VIDEO):
    print(f"File size: {os.path.getsize(OUTPUT_VIDEO) / (1024*1024):.1f} MB")

print("\n" + "="*60)
