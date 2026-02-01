"""
Simple webcam PPE detection viewer - Optimized for performance
Usage: python webcam_ppe_viewer.py
Press 'q' to quit
"""
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.ai.yolov8_detector import YoloV8Detector
from app.services.violation_monitor import ViolationMonitor
import time

def main():
    print("Initializing PPE detector...")
    print(f"Model input size: {settings.input_size}px (optimized for CPU)")
    
    # Initialize detector with CPU
    detector = YoloV8Detector(
        model_path=settings.model_path,
        device=None,  # Force CPU
        camera_id="webcam"
    )
    print("✓ Detector loaded")
    
    # Initialize violation monitor
    violation_monitor = ViolationMonitor(
        alert_url=settings.alert_endpoint,
        violation_threshold=settings.violation_threshold_seconds,
        enabled=settings.alerts_enabled
    )
    if settings.alerts_enabled:
        print(f"✓ Violation monitoring enabled (threshold: {settings.violation_threshold_seconds}s)")
    else:
        print("⚠ Violation monitoring disabled")
    
    
    # Open webcam with optimized settings
    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend for Windows
    
    # Optimize webcam settings for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        print("Try: python webcam_ppe_viewer.py")
        return
    
    print("✓ Webcam opened")
    print("\n" + "="*50)
    print("PPE Detection Active - Press 'q' to quit")
    print("="*50 + "\n")
    
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run detection every frame
            detections = detector.infer(frame)
            tracks = detections.get("tracks", [])
            
            # Calculate FPS every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_display = 10 / elapsed
                start_time = time.time()
            
            # Draw detections
            for track in tracks:
                bbox = track.get("bbox", {})
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                track_id = track.get("track_id", -1)
                ppe_items = track.get("ppe", [])
                
                # Check PPE status
                has_helmet = any(p["label"] in ["helmet", "head_helmet"] for p in ppe_items)
                has_vest = any(p["label"] == "vest" for p in ppe_items)
                
                # Update violation monitor
                violation_monitor.update(track_id, has_helmet, has_vest, time.time())
                
                # 1. Draw PERSON box with ID (always draw this)
                # Always green when person is detected
                person_color = (0, 255, 0)  # Green - Person detected
                
                # Draw person bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 3)
                
                # Person label with ID only
                person_label = f"Person ID: {track_id}"
                (label_w, label_h), _ = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), person_color, -1)
                cv2.putText(frame, person_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 2. Draw separate boxes for detected PPE items (Helmet and Vest)
                for ppe in ppe_items:
                    ppe_label = ppe.get("label", "")
                    ppe_bbox = ppe.get("bbox")
                    
                    if ppe_bbox:
                        # Get PPE bounding box coordinates
                        px1, py1, px2, py2 = int(ppe_bbox[0]), int(ppe_bbox[1]), int(ppe_bbox[2]), int(ppe_bbox[3])
                        conf = ppe.get("confidence", 0.0)
                        
                        # Set color and label based on PPE type
                        if ppe_label in ["helmet", "head_helmet"]:
                            # Green if confidence > 0.2, otherwise yellow
                            if conf > 0.2:
                                ppe_color = (0, 255, 0)  # Green for good confidence
                            else:
                                ppe_color = (0, 255, 255)  # Yellow for low confidence
                            full_label = f"Helmet ({conf:.2f})"
                        elif ppe_label == "vest":
                            ppe_color = (0, 255, 0)  # Green for vest
                            full_label = f"Safety Vest ({conf:.2f})"
                        else:
                            ppe_color = (255, 0, 255)  # Magenta for other
                            full_label = f"{ppe_label.capitalize()} ({conf:.2f})"
                        
                        # Draw PPE bounding box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), ppe_color, 2)
                        
                        # Draw PPE label above box
                        (pw, ph), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (px1, py1 - ph - 10), (px1 + pw, py1), ppe_color, -1)
                        cv2.putText(frame, full_label, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 3. Draw "No Helmet" or "No Vest" indicators if missing
                # Estimate head position (top 1/3 of person box) for "No Helmet"
                if not has_helmet:
                    head_y1 = y1
                    head_y2 = y1 + int((y2 - y1) * 0.3)
                    head_x1 = x1 + int((x2 - x1) * 0.2)
                    head_x2 = x2 - int((x2 - x1) * 0.2)
                    
                    # Red box for "No Helmet"
                    no_helmet_color = (0, 0, 255)  # Red
                    cv2.rectangle(frame, (head_x1, head_y1), (head_x2, head_y2), no_helmet_color, 2)
                    
                    no_helmet_label = "No Helmet"
                    (nh_w, nh_h), _ = cv2.getTextSize(no_helmet_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (head_x1, head_y1 - nh_h - 10), (head_x1 + nh_w, head_y1), no_helmet_color, -1)
                    cv2.putText(frame, no_helmet_label, (head_x1, head_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Estimate torso position (middle 1/2 of person box) for "No Vest"
                if not has_vest:
                    torso_y1 = y1 + int((y2 - y1) * 0.25)
                    torso_y2 = y1 + int((y2 - y1) * 0.75)
                    torso_x1 = x1 + int((x2 - x1) * 0.1)
                    torso_x2 = x2 - int((x2 - x1) * 0.1)
                    
                    # Red box for "No Vest"
                    no_vest_color = (0, 0, 255)  # Red
                    cv2.rectangle(frame, (torso_x1, torso_y1), (torso_x2, torso_y2), no_vest_color, 2)
                    
                    no_vest_label = "No Vest"
                    (nv_w, nv_h), _ = cv2.getTextSize(no_vest_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (torso_x1, torso_y1 - nv_h - 10), (torso_x1 + nv_w, torso_y1), no_vest_color, -1)
                    cv2.putText(frame, no_vest_label, (torso_x1, torso_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Check for violations after processing all tracks (pass frame for screenshot)
            violation_monitor.check_violations(camera_id="webcam", frame=frame)
            
            # Display FPS and stats
            stats_text = f"FPS: {fps_display:.1f} | Tracks: {len(tracks)}"
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("PPE Detection - Press 'q' to quit", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print("Webcam closed")

if __name__ == "__main__":
    main()
