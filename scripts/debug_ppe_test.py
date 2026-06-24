"""
Simple PPE Detection Test - Debug boxes issue
"""
import cv2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.ai.yolov8_detector import YoloV8Detector
from app.config import settings

def main():
    print("=== PPE Detection Debug Test ===")
    print(f"Model: {settings.model_path}")
    print(f"Device: {settings.device}")
    
    # Load detector
    detector = YoloV8Detector(
        model_path=settings.model_path,
        device=settings.device,
        camera_id="debug_test"
    )
    print("Detector loaded!")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam!")
        return
    
    print("Webcam opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        result = detector.infer(frame)
        
        # Get persons - this is the RAW detection output
        persons = result.get('persons', [])
        
        # DEBUG: Print what we got
        if len(persons) > 0:
            print(f"\n=== DETECTED {len(persons)} PERSON(S) ===")
            for i, person in enumerate(persons):
                bbox = person.get('bbox', {})
                conf = person.get('conf', 0)
                ppe = person.get('ppe', [])
                print(f"Person {i}: bbox={bbox}, conf={conf:.2f}, ppe_count={len(ppe)}")
                
                # Draw DIRECTLY on frame
                if isinstance(bbox, dict):
                    x1 = int(bbox.get('x1', 0))
                    y1 = int(bbox.get('y1', 0))
                    x2 = int(bbox.get('x2', 0))
                    y2 = int(bbox.get('y2', 0))
                    
                    print(f"  Bbox coords: ({x1},{y1}) to ({x2},{y2})")
                    
                    # Draw person box - RED for no PPE
                    if len(ppe) == 0:
                        color = (0, 0, 255)  # Red
                        label = f"Person {i} - NO PPE"
                    else:
                        color = (0, 255, 0)  # Green
                        label = f"Person {i} - PPE OK"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw PPE items
                    for ppe_item in ppe:
                        ppe_bbox = ppe_item.get('bbox', {})
                        ppe_label = ppe_item.get('label', '')
                        ppe_conf = ppe_item.get('conf', 0)
                        
                        if isinstance(ppe_bbox, dict):
                            px1 = int(ppe_bbox.get('x1', 0))
                            py1 = int(ppe_bbox.get('y1', 0))
                            px2 = int(ppe_bbox.get('x2', 0))
                            py2 = int(ppe_bbox.get('y2', 0))
                            
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{ppe_label} {ppe_conf:.2f}", (px1, py1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            print(f"  PPE: {ppe_label} at ({px1},{py1})-({px2},{py2})")
                    
                    # Draw NO HELMET / NO VEST indicators if missing
                    helmet_found = any(p.get('label') in ['helmet', 'head_helmet'] for p in ppe)
                    vest_found = any(p.get('label') == 'vest' for p in ppe)
                    
                    if not helmet_found:
                        hx1 = x1 + (x2-x1)//4
                        hy1 = y1
                        hx2 = x2 - (x2-x1)//4
                        hy2 = y1 + (y2-y1)//4
                        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                        cv2.putText(frame, "NO HELMET", (hx1, hy1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if not vest_found:
                        vx1 = x1 + (x2-x1)//6
                        vy1 = y1 + (y2-y1)//4
                        vx2 = x2 - (x2-x1)//6
                        vy2 = y1 + (y2-y1)*3//5
                        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
                        cv2.putText(frame, "NO VEST", (vx1, vy1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add info text
        cv2.putText(frame, f"Persons: {len(persons)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("PPE Debug Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
