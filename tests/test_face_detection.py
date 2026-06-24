"""
Test face detection and re-identification with webcam.
This script verifies that persistent person IDs work correctly.
"""

import cv2
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ai.yolov8_detector import YoloV8Detector
from app.config import settings

# Import face detection components
try:
    from app.ai.face_detector import FaceDetector
    from app.ai.face_embedder import FaceEmbedder
    from app.ai.face_tracker import FaceTracker
    FACE_AVAILABLE = True
    print("✅ Face detection modules imported successfully")
except Exception as e:
    FACE_AVAILABLE = False
    print(f"❌ Face detection not available: {e}")
    print("   Install: pip install face-recognition")
    sys.exit(1)

def main():
    print("\n" + "="*70)
    print("  Face Detection & Re-identification Test")
    print("="*70)
    print(f"Face detection enabled: {settings.face_detection_enabled}")
    print(f"Face model: {settings.face_model_path}")
    print(f"Similarity threshold: {settings.face_similarity_threshold}")
    print("="*70)
    
    # Initialize components
    print("\n[1/4] Loading PPE detector...")
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
    print("\n[2/4] Loading face detector...")
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
    
    # Initialize face embedder
    print("\n[3/4] Loading face embedder...")
    try:
        face_embedder = FaceEmbedder(model="large")
        print("✅ Face embedder loaded")
    except Exception as e:
        print(f"❌ Failed to load face embedder: {e}")
        print("   Make sure face_recognition is installed:")
        print("   pip install face-recognition")
        return
    
    # Initialize face tracker
    print("\n[4/4] Creating face tracker...")
    face_tracker = FaceTracker(
        similarity_threshold=settings.face_similarity_threshold,
        max_face_age_seconds=settings.face_max_age_seconds,
        min_stable_frames=settings.face_min_stable_frames
    )
    print("✅ Face tracker created")
    
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
    print("1. Show your face to the camera - you'll get an ID (e.g., ID 1)")
    print("2. Move OUT of frame completely")
    print("3. Move BACK into frame - you should get the SAME ID!")
    print("4. Press 'q' to quit")
    print("="*70 + "\n")
    
    frame_count = 0
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0
    
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
            
            # Extract face embeddings
            face_embeddings = []
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
                    
                    if person_crop.size == 0:
                        face_embeddings.append(None)
                        continue
                    
                    # Detect face
                    face_result = face_detector.detect_face(person_crop)
                    
                    if face_result is None:
                        face_embeddings.append(None)
                        continue
                    
                    # Extract embedding
                    face_crop = face_result['face_crop']
                    embedding = face_embedder.extract_embedding(face_crop, num_jitters=1)
                    face_embeddings.append(embedding)
                    
                except Exception as e:
                    face_embeddings.append(None)
            
            # Update face tracker
            timestamp = time.time()
            tracked_persons = face_tracker.update(persons, face_embeddings, timestamp)
            
            # Draw annotations
            annotated_frame = frame.copy()
            
            for person in tracked_persons:
                track_id = person.get('track_id', -1)
                bbox = person.get('bbox', {})
                has_face = person.get('has_face', False)
                stable = person.get('stable', False)
                
                x1 = int(bbox.get('x1', 0))
                y1 = int(bbox.get('y1', 0))
                x2 = int(bbox.get('x2', 0))
                y2 = int(bbox.get('y2', 0))
                
                # Color based on stability and face detection
                if has_face and stable:
                    color = (0, 255, 0)  # Green - stable with face
                    label_suffix = " ✓"
                elif has_face:
                    color = (0, 255, 255)  # Yellow - face detected but not stable
                    label_suffix = " ⏳"
                else:
                    color = (0, 0, 255)  # Red - no face
                    label_suffix = " ✗"
                
                # Draw bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID label
                label = f"ID {track_id}{label_suffix}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw info
            info_y = 30
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            info_y += 30
            cv2.putText(annotated_frame, f"Persons: {len(tracked_persons)}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            info_y += 30
            stats = face_tracker.get_stats()
            known_faces = stats.get('known_faces', 0)
            cv2.putText(annotated_frame, f"Known Faces: {known_faces}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            info_y += 40
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Face Detection Test', annotated_frame)
            
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
        stats = face_tracker.get_stats()
        print(f"Total frames processed: {frame_count}")
        print(f"Known faces in database: {stats.get('known_faces', 0)}")
        print(f"Next person ID: {stats.get('next_id', 0)}")
        print("\nKnown person IDs:", face_tracker.get_known_person_ids())
        print("="*70)
        
        print("\n✅ Test completed!")
        print("\nDid your person ID stay consistent when you left and returned?")
        print("If yes, face re-identification is working! 🎉")

if __name__ == "__main__":
    main()
