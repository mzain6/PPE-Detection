"""
test_camera_manager.py

Test script for multi-camera manager.
Tests parallel processing with 2-3 cameras.
"""

import sys
import time
import cv2
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.camera_manager import CameraManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_detection_callback(camera_id: str, frame):
    """
    Simple callback for testing - just annotates frame with camera ID and timestamp
    In production, this will call the PPE detector
    """
    import cv2
    import time
    
    # Simulate processing time (remove in production with actual detector)
    time.sleep(0.01)
    
    # Annotate frame
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Add camera ID
    cv2.putText(annotated, f"Camera: {camera_id}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(annotated, timestamp, 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mock detections
    detections = {
        'camera_id': camera_id,
        'timestamp': time.time(),
        'persons': []  # Will be populated by actual detector
    }
    
    return detections, annotated


def main():
    """Test camera manager with multiple cameras"""
    
    # Create camera manager
    manager = CameraManager(max_cameras=10)
    
    # Add cameras (modify sources as needed)
    cameras_to_add = [
        ('camera_1', 0),  # USB webcam
        # ('camera_2', 'rtsp://example.com/stream1'),  # RTSP stream example
        # ('camera_3', 'rtsp://example.com/stream2'),
    ]
    
    print("\n=== Adding Cameras ===")
    for camera_id, source in cameras_to_add:
        success = manager.add_camera(
            camera_id=camera_id,
            source=source,
            detection_callback=simple_detection_callback,
            reconnect_delay=2.0
        )
        if success:
            print(f"✓ {camera_id} added")
        else:
            print(f"✗ {camera_id} failed to add")
    
    # Display windows for each camera
    camera_list = manager.get_camera_list()
    print(f"\nActive cameras: {camera_list}")
    
    if not camera_list:
        print("No cameras available. Exiting.")
        return
    
    print("\n=== Starting Display ===")
    print("Press 'q' to quit, 's' for stats")
    
    try:
        while True:
            for camera_id in camera_list:
                processor = manager.get_processor(camera_id)
                if processor:
                    frame, detections, annotated = processor.get_latest()
                    
                    if annotated is not None:
                        # Resize for display if needed
                        display_frame = cv2.resize(annotated, (640, 480))
                        cv2.imshow(f"Camera {camera_id}", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Show statistics
                stats = manager.get_system_stats()
                print("\n=== System Statistics ===")
                print(f"Total Cameras: {stats['total_cameras']}")
                print(f"Average FPS: {stats['average_fps']}")
                print("\nPer-Camera Stats:")
                for cam_stat in stats['cameras']:
                    print(f"  {cam_stat['camera_id']}:")
                    print(f"    Connected: {cam_stat['connected']}")
                    print(f"    FPS: {cam_stat.get('processing_fps', 0)}")
                    print(f"    Frames: {cam_stat['frame_count']}")
                    print(f"    Dropped: {cam_stat['dropped_frames']}")
            
            time.sleep(0.01)  # Small delay to prevent CPU spinning
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\n=== Cleaning Up ===")
        cv2.destroyAllWindows()
        manager.shutdown()
        print("Shutdown complete")


if __name__ == "__main__":
    main()
