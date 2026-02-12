"""
Debug script to check what the model is actually detecting (all classes, all confidences).
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Load model
model_path = "best.pt"
print(f"Loading model: {model_path}")
model = YOLO(model_path)

# Check model info
print("\n" + "="*60)
print("MODEL INFORMATION")
print("="*60)
print(f"Model: {model_path}")
print(f"Class names: {model.names}")
print(f"Number of classes: {len(model.names)}")
print()

# Load image
image_path = r"C:\Users\hashi\Downloads\PPE2\PPE2\PPE-Detection\person.jpeg"
print(f"Loading image: {image_path}")
frame = cv2.imread(image_path)
print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

# Run detection with VERY LOW threshold to see everything
print("\n" + "="*60)
print("RUNNING DETECTION (conf=0.01 to see all detections)")
print("="*60)

results = model(frame, conf=0.01, verbose=False)

# Parse all detections
result = results[0]
boxes = result.boxes

print(f"\nTotal detections found: {len(boxes)}")
print()

if len(boxes) == 0:
    print("⚠️  NO DETECTIONS AT ALL!")
else:
    # Group by class
    detections_by_class = {}
    
    for box in boxes:
        cls_id = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())
        xyxy = box.xyxy.cpu().numpy()[0]
        
        class_name = model.names.get(cls_id, f"class_{cls_id}")
        
        if class_name not in detections_by_class:
            detections_by_class[class_name] = []
        
        detections_by_class[class_name].append({
            'confidence': conf,
            'bbox': xyxy
        })
    
    # Display results
    for class_name in sorted(detections_by_class.keys()):
        dets = detections_by_class[class_name]
        print(f"\n{class_name.upper()}: {len(dets)} detection(s)")
        for i, det in enumerate(dets, 1):
            conf = det['confidence']
            bbox = det['bbox']
            print(f"  #{i}: {conf:.2%} confidence at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Check for helmet specifically
helmet_found = False
for class_name in detections_by_class.keys():
    if 'helmet' in class_name.lower() or 'head' in class_name.lower():
        helmet_found = True
        print(f"✅ Helmet class '{class_name}' WAS detected!")
        break

if not helmet_found:
    print("❌ NO helmet-related class detected at all")
    print("\nPossible reasons:")
    print("1. No helmet visible in the image")
    print("2. Helmet is too small or occluded")
    print("3. Model not trained to detect helmets")
    print("4. Class name mismatch (check model.names above)")

print("\n" + "="*60)
