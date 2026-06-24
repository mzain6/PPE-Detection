"""
multi_camera_ppe_demo.py

Demo script for multi-camera PPE detection.
Integrates CameraManager with PPE detection.
"""

import sys
import time
import cv2
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.camera_manager import CameraManager
from app.services.ppe_camera_integration import PPECameraIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Multi-camera PPE detection demo"""
    
    print("\n" + "="*60)
    print(" Multi-Camera PPE Detection System")
    print("="*60)
    
    # Initialize components
    print("\n[1/3] Initializing Camera Manager...")
    camera_manager = CameraManager(max_cameras=10)
    
    print("[2/3] Initializing PPE Detection...")
    ppe_integration = PPECameraIntegration(
        model_path="best.pt",  # or ppe_model.pt
        device="auto"  # Will use GPU if available
    )
    
    # Configure cameras to add
    cameras_config = [
        {'id': 'camera_1', 'source': 0},  # USB webcam
        # Add more cameras as needed:
        # {'id': 'camera_2', 'source': 'rtsp://192.168.1.100/stream'},
        # {'id': 'camera_3', 'source': 'rtsp://192.168.1.101/stream'},
    ]
    
    print(f"[3/3] Adding {len(cameras_config)} camera(s)...")
    
    # Add cameras with PPE detection
    for cam_config in cameras_config:
        camera_id = cam_config['id']
        source = cam_config['source']
        
        # Create detection callback for this camera
        detection_callback = ppe_integration.create_detection_callback(camera_id)
        
        # Add camera to manager
        success = camera_manager.add_camera(
            camera_id=camera_id,
            source=source,
            detection_callback=detection_callback,
            reconnect_delay=2.0
        )
        
        if success:
            print(f"  ✓ {camera_id} added")
        else:
            print(f"  ✗ {camera_id} failed")
    
    # Get active cameras
    camera_list = camera_manager.get_camera_list()
    
    if not camera_list:
        print("\n❌ No cameras available. Exiting.")
        return
    
    print(f"\n✓ Active cameras: {camera_list}")
    print("\n" + "="*60)
    print(" Controls:")
    print("   'q' - Quit")
    print("   's' - Show statistics")
    print("   'h' - Help")
    print("="*60 + "\n")
    
    # Main display loop
    try:
        fps_start = time.time()
        frame_count = 0
        
        while True:
            display_frames = []
            
            # Get latest frames from all cameras
            for camera_id in camera_list:
                processor = camera_manager.get_processor(camera_id)
                if processor:
                    _, _, annotated = processor.get_latest()
                    
                    if annotated is not None:
                        # Resize for display
                        display_frame = cv2.resize(annotated, (640, 480))
                        
                        # Add FPS overlay
                        elapsed = time.time() - fps_start
                        fps = frame_count / max(elapsed, 1)
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                                   (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 255), 2)
                        
                        display_frames.append((camera_id, display_frame))
            
            # Display frames
            for camera_id, frame in display_frames:
                cv2.imshow(f"PPE Detection - {camera_id}", frame)
                frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Shutting down...")
                break
                
            elif key == ord('s'):
                # Show statistics
                print("\n" + "="*60)
                print(" SYSTEM STATISTICS")
                print("="*60)
                
                # Camera manager stats
                cam_stats = camera_manager.get_system_stats()
                print(f"\n📊 Camera Manager:")
                print(f"   Total Cameras: {cam_stats['total_cameras']}")
                print(f"   Average FPS: {cam_stats['average_fps']}")
                
                print(f"\n📹 Per-Camera Stats:")
                for cam in cam_stats['cameras']:
                    print(f"\n   {cam['camera_id']}:")
                    print(f"     Connected: {cam['connected']}")
                    print(f"     Processing FPS: {cam.get('processing_fps', 0)}")
                    print(f"     Frames Captured: {cam['frame_count']}")
                    print(f"     Frames Dropped: {cam['dropped_frames']}")
                    print(f"     Queue Size: {cam['queue_size']}")
                
                # PPE detection stats
                ppe_stats = ppe_integration.get_stats()
                print(f"\n🔍 PPE Detection:")
                for camera_id, stats in ppe_stats.items():
                    print(f"\n   {camera_id}:")
                    print(f"     Frames Processed: {stats['total_frames']}")
                    print(f"     Avg Detection Time: {stats['avg_detection_time_ms']} ms")
                    print(f"     Inference FPS: {stats['inference_fps']}")
                
                print("\n" + "="*60 + "\n")
                
            elif key == ord('h'):
                print("\n" + "="*60)
                print(" HELP")
                print("="*60)
                print("   'q' - Quit application")
                print("   's' - Show detailed statistics")
                print("   'h' - Show this help message")
                print("="*60 + "\n")
            
            time.sleep(0.01)  # Prevent CPU spinning
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cv2.destroyAllWindows()
        
        print("   - Shutting down camera manager...")
        camera_manager.shutdown()
        
        print("   - Shutting down PPE integration...")
        ppe_integration.shutdown()
        
        print("✓ Shutdown complete\n")


if __name__ == "__main__":
    main()
