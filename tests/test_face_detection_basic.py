"""
Alternative face detection test WITHOUT face_recognition library.
Uses only YOLOv8-face for detection (no embeddings/re-identification).
This tests basic face detection capability.
"""

import cv2
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ai.yolov8_detector import YoloV8Detector
from app.config import settings
from app.ai.tracker import IOUTracker

# Import face detector only
try:
    from app.ai.face_detector import FaceDetector
    FACE_DETECTOR_AVAILABLE = True
    print("✅ Face detector module imported successfully")
except Exception as e:
    FACE_DETECTOR_AVAILABLE = False
    print(f"❌ Face detector not available: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*70)
    print("  Basic Face Detection Test (Without Re-identification)")
    print("="*70)
    print("This test checks if face detection works (without persistent IDs)")
    print("="*70)
    
    # Initialize components
    print("\n[1/3] Loading PPE detector...")
    try:
        ppe_detector = YoloV8Detector(
            model_path=settings.model_path,
            device=settings.device,
            camera_id="webcam_test"
        )
        print("✅ PPE detector loaded")
    except Exception as e:
        print(f"❌ Failed to load PPE detector: {e}")
        return
    
    # Initialize face detector
    print("\n[2/3] Loading face detector...")
    try:
        face_detector = FaceDetector(
            model_path=settings.face_model_path,
            confidence_threshold=settings.face_confidence_threshold,
            device=settings.device
        )
        print("✅ Face detector loaded")
    except Exception as e:
        print(f"❌ Failed to load face detector: {e}")
        return
    
    # Initialize IOU tracker (fallback without face embeddings)
    print("\n[3/3] Creating IOU tracker...")
    tracker = IOUTracker(
        iou_thresh=float(settings.iou_threshold),
        max_missing=int(settings.max_missing_frames),
        stable_frames=int(settings.stable_frames)
    )
    print("✅ IOU tracker created")
    
    # Open webcam
    print("\n[*] Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return
    
    print("✅ Webcam opened")
    print("\n" + "="*70)
    print("  INSTRUCTIONS:")
    print("="*70)
    print("1. Show your face to the camera - green box if face detected")
    print("2. Face detection will be shown with bounding boxes")
    print("3. Person IDs use IOU tracking (won't persist after leaving)")
    print("4. Press 'q' to quit")
    print("="*70 + "\n")
    
    frame_count = 0
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0
    faces_detected = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()
            
            # Run PPE detection
            detection_result = ppe_detector.infer(frame)
            persons = detection_result.get('persons', [])
            
            # Detect faces in person crops
            persons_with_faces = []
            for person in persons:
                bbox = person.get('bbox', {})
                
                try:
                    # Crop person
                    x1 = int(bbox.get('x1', 0))
                    y1 = int(bbox.get('y1', 0))
                    x2 = int(bbox.get('x2', 0))
                    y2 = int(bbox.get('y2', 0))
                    
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))
                    
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:
                        # Detect face
                        face_result = face_detector.detect_face(person_crop)
                        
                        if face_result is not None:
                            person['has_face'] = True
                            person['face_conf'] = face_result['confidence']
                            faces_detected += 1
                            
                            # Convert face bbox to frame coordinates
                            face_bbox = face_result['bbox']
                            person['face_bbox_abs'] = [
                                x1 + face_bbox[0],
                                y1 + face_bbox[1],
                                x1 + face_bbox[2],
                                y1 + face_bbox[3]
                            ]
                        else:
                            person['has_face'] = False
                    else:
                        person['has_face'] = False
                        
                except Exception as e:
                    person['has_face'] = False
                
                persons_with_faces.append(person)
            
            # Update IOU tracker
            timestamp = time.time()
            tracked_persons = tracker.update(persons_with_faces, timestamp=timestamp)
            
            # Draw annotations
            annotated_frame = frame.copy()
            
            for track in tracked_persons.values():
                track_id = track.id
                bbox = track.bbox
                has_face = getattr(track, 'has_face', False)
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Color based on face detection
                if has_face:
                    color = (0, 255, 0)  # Green - face detected
                    label_suffix = " 😊"
                else:
                    color = (0, 165, 255) # Orange - no face
                    label_suffix = ""
                
                # Draw person bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID label
                label = f"ID {track_id}{label_suffix}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw face bbox if detected
                if has_face and hasattr(track, 'face_bbox_abs'):
                    fx1, fy1, fx2, fy2 = map(int, track.face_bbox_abs)
                    cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 1)
            
            # Draw info
            info_y = 30
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            info_y += 30
            cv2.putText(annotated_frame, f"Persons: {len(tracked_persons)}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            info_y += 30
            cv2.putText(annotated_frame, f"Faces Detected: {sum(1 for t in tracked_persons.values() if getattr(t, 'has_face', False))}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            info_y += 40
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Basic Face Detection Test', annotated_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[*] Quitting...")
                break
    
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final stats
        print("\n" + "="*70)
        print("  TEST RESULTS")
        print("="*70)
        print(f"Total frames processed: {frame_count}")
        print(f"Total faces detected: {faces_detected}")
        print(f"Average FPS: {current_fps:.1f}")
        print("="*70)
        
        print("\n✅ Basic face detection test completed!")
        print("\nNote: This test uses IOU tracking (bbox overlap), not face re-identification.")
        print("To enable persistent IDs, install: pip install face-recognition")

if __name__ == "__main__":
    main()
