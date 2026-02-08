"""
Quick debug to check what the model actually detects
"""
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    # Run detection
    results = model(frame, verbose=True)
    
    print("\n" + "="*60)
    print("MODEL DETECTION DEBUG")
    print("="*60)
    
    # Get class names from model
    print(f"\nModel class names: {results[0].names}")
    
    # Show all detections
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        print(f"\nDetections found: {len(boxes)}")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            label = results[0].names.get(cls_id, str(cls_id))
            print(f"  [{i}] Class ID: {cls_id}, Label: '{label}', Confidence: {conf:.3f}")
    else:
        print("\nNO DETECTIONS FOUND!")
    
    print("="*60)
else:
    print("Failed to capture frame from webcam")
