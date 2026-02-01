"""
Simple script to run detection and display results with detailed output.
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ai.yolov8_detector import YoloV8Detector
from app.utils.gpu_utils import select_device

# Initialize detector
print("Initializing detector...")
device = select_device(settings.device)
detector = YoloV8Detector(
    model_path=settings.model_path,
    device=device,
    camera_id="test"
)

# Load image
image_path = r"C:\Users\hashi\Downloads\PPE2\PPE2\PPE-Detection\person.jpeg"
print(f"\nLoading image: {image_path}")
frame = cv2.imread(image_path)

if frame is None:
    print("ERROR: Could not load image!")
    sys.exit(1)

print(f"Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels")

# Run detection
print("\nRunning detection...")
results = detector.infer(frame)

# Display results
print("\n" + "="*60)
print("DETECTION RESULTS")
print("="*60)
print(f"Camera ID: {results.get('camera_id')}")
print(f"Timestamp: {results.get('timestamp')}")
print(f"Total Tracks: {len(results.get('tracks', []))}")
print()

tracks = results.get('tracks', [])
if not tracks:
    print("⚠️  NO PERSONS DETECTED")
else:
    for i, track in enumerate(tracks, 1):
        print(f"Person {i} (Track ID: {track.get('track_id')})")
        print(f"  Confidence: {track.get('person_confidence', 0):.2%}")
        
        bbox = track.get('bbox', {})
        print(f"  Bounding Box: ({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}) to ({bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f})")
        
        ppe_items = track.get('ppe', [])
        if ppe_items:
            print(f"  PPE Detected: {len(ppe_items)} items")
            for ppe in ppe_items:
                print(f"    - {ppe.get('label')}: {ppe.get('confidence', 0):.2%}")
        else:
            print(f"  PPE Detected: ❌ NONE - UNSAFE!")
        
        # Check specific PPE
        has_helmet = any(p['label'] in ['helmet', 'head_helmet'] for p in ppe_items)
        has_vest = any(p['label'] == 'vest' for p in ppe_items)
        
        if has_helmet and has_vest:
            status = "✅ SAFE (Full PPE)"
        elif has_helmet or has_vest:
            status = "⚠️  PARTIAL PPE"
        else:
            status = "❌ UNSAFE (No PPE)"
        
        print(f"  Status: {status}")
        print()

print("="*60)

# Draw and save annotated image
print("\nCreating annotated image...")
annotated = frame.copy()

for track in tracks:
    bbox = track.get('bbox', {})
    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
    
    ppe_items = track.get('ppe', [])
    has_helmet = any(p['label'] in ['helmet', 'head_helmet'] for p in ppe_items)
    has_vest = any(p['label'] == 'vest' for p in ppe_items)
    
    # Color coding
    if has_helmet and has_vest:
        color = (0, 255, 0)  # Green
        status = "SAFE"
    elif has_helmet or has_vest:
        color = (0, 255, 255)  # Yellow
        status = "PARTIAL"
    else:
        color = (0, 0, 255)  # Red
        status = "UNSAFE"
    
    # Draw box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    
    # Draw label
    label = f"ID:{track.get('track_id')} {status}"
    if has_helmet:
        label += " +HELMET"
    if has_vest:
        label += " +VEST"
    
    # Background for text
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Save
output_path = "person_detected_detailed.jpeg"
cv2.imwrite(output_path, annotated)
print(f"Saved annotated image to: {output_path}")

# Try to display
try:
    cv2.imshow("PPE Detection Results", annotated)
    print("\n✅ Image window opened! Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"\n⚠️  Could not display image window: {e}")
    print(f"But the annotated image was saved to: {output_path}")
