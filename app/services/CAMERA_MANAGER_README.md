# Camera Manager - Multi-Camera Parallel Processing

## Overview

The Camera Manager enables parallel processing of multiple cameras (up to 10) with independent threads and automatic reconnection.

## Architecture

```
CameraManager
 ├─ CameraStream (per camera)
 │   ├─ Background thread for frame reading
 │   ├─ Frame queue (max 5 frames)
 │   └─ Auto-reconnect on failure
 │
 └─ CameraProcessor (per camera)
     ├─ Processing thread
     ├─ Detection callback
     └─ Latest results storage
```

## Key Features

### CameraStream
- **Thread-safe frame reading**: Dedicated thread per camera
- **Auto-reconnection**: Automatically reconnects on failure with configurable delay
- **Frame queue**: Buffers up to 5 frames, drops oldest when full (prevents blocking)
- **Statistics**: Tracks FPS, frame count, dropped frames

### CameraProcessor
- **Independent processing**: Each camera has its own processing pipeline
- **Detection callback**: Pluggable detection function
- **Latest results**: Stores most recent frame, detections, and annotated output

### CameraManager
- **Centralized orchestration**: Manages up to 10 cameras
- **Dynamic add/remove**: Add or remove cameras at runtime
- **System monitoring**: Aggregated statistics across all cameras
- **Clean shutdown**: Properly releases all resources

## Usage

### Basic Example

```python
from app.services.camera_manager import CameraManager

def my_detection_callback(camera_id, frame):
    # Your detection logic here
    detections = []
    annotated_frame = frame.copy()
    return detections, annotated_frame

# Create manager
manager = CameraManager(max_cameras=10)

# Add cameras
manager.add_camera('camera_1', 0, detection_callback=my_detection_callback)  # USB
manager.add_camera('camera_2', 'rtsp://192.168.1.100/stream', detection_callback=my_detection_callback)  # RTSP

# Get latest results
processor = manager.get_processor('camera_1')
frame, detections, annotated = processor.get_latest()

# Get statistics
stats = manager.get_system_stats()
print(f"Average FPS: {stats['average_fps']}")

# Cleanup
manager.shutdown()
```

### Testing

Run the test script with your webcam:

```bash
python test_camera_manager.py
```

**Controls:**
- `q`: Quit
- `s`: Show statistics

### Configuration

Update `config.yaml`:

```yaml
multi_camera:
  max_cameras: 10
  frame_queue_size: 5
  reconnect_delay: 3.0
```

## Next Steps

1. ✅ Camera Manager implemented
2. ⏳ Integrate with PPE Detector
3. ⏳ Add batch inference
4. ⏳ Create FastAPI endpoints
5. ⏳ Add face detection
