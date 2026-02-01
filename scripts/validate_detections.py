"""
Detection validation script for manual testing and accuracy assessment.

Usage:
    python scripts/validate_detections.py --source 0  # Webcam
    python scripts/validate_detections.py --source "rtsp://example.com/stream"  # RTSP
    python scripts/validate_detections.py --source "video.mp4"  # Video file
    python scripts/validate_detections.py --source "image.jpg"  # Single image
"""
import argparse
import cv2
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ai.yolov8_detector import YoloV8Detector
from app.utils.gpu_utils import get_cuda_info, select_device


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()
    
    tracks = detections.get("tracks", [])
    
    for track in tracks:
        bbox = track.get("bbox", {})
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        track_id = track.get("track_id", -1)
        person_conf = track.get("person_confidence", 0.0)
        ppe_items = track.get("ppe", [])
        
        # Determine PPE status
        has_helmet = any(p["label"] in ["helmet", "head_helmet"] for p in ppe_items)
        has_vest = any(p["label"] == "vest" for p in ppe_items)
        
        # Color coding: Green = full PPE, Yellow = partial, Red = no PPE
        if has_helmet and has_vest:
            color = (0, 255, 0)  # Green
            status = "SAFE"
        elif has_helmet or has_vest:
            color = (0, 255, 255)  # Yellow
            status = "PARTIAL"
        else:
            color = (0, 0, 255)  # Red
            status = "UNSAFE"
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"ID:{track_id} {status}"
        if has_helmet:
            label += " +H"
        if has_vest:
            label += " +V"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw PPE details
        y_offset = y2 + 20
        for ppe in ppe_items:
            ppe_label = f"{ppe['label']}: {ppe['confidence']:.2f}"
            cv2.putText(annotated, ppe_label, (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 15
    
    # Draw stats
    stats = f"Tracks: {len(tracks)} | FPS: {detections.get('fps', 0)}"
    cv2.putText(annotated, stats, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated


def main():
    parser = argparse.ArgumentParser(description="PPE Detection Validation")
    parser.add_argument("--source", type=str, default="0", 
                       help="Video source: webcam index, RTSP URL, video file, or image")
    parser.add_argument("--display", action="store_true", 
                       help="Display annotated frames (requires GUI)")
    parser.add_argument("--save", type=str, 
                       help="Save annotated video to file")
    parser.add_argument("--max-frames", type=int, default=0,
                       help="Maximum frames to process (0 = unlimited)")
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Initializing PPE detector...")
    gpu_info = get_cuda_info()
    device = select_device(settings.device)
    
    print(f"GPU Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"Using device: {device}")
    else:
        print("Using CPU")
    
    detector = YoloV8Detector(
        model_path=settings.model_path,
        device=device,
        camera_id="validation"
    )
    
    # Open video source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    print(f"Opening source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return
    
    # Video writer setup
    writer = None
    if args.save:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving to: {args.save}")
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    
    print("Processing frames... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream")
                break
            
            frame_count += 1
            
            # Run detection
            detections = detector.infer(frame)
            
            # Annotate frame
            annotated = draw_detections(frame, detections)
            
            # Display
            if args.display:
                cv2.imshow("PPE Detection Validation", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Save
            if writer:
                writer.write(annotated)
            
            # Stats
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Processed {frame_count} frames | FPS: {fps:.2f} | Tracks: {len(detections.get('tracks', []))}")
            
            # Max frames check
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Final stats
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nProcessing complete:")
        print(f"  Total frames: {frame_count}")
        print(f"  Time elapsed: {elapsed:.2f}s")
        print(f"  Average FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
