"""
camera_manager.py

Multi-camera management system for parallel processing.
Orchestrates multiple camera streams with independent processing threads.
"""

import threading
import logging
import time
from typing import Dict, Optional, Callable
from queue import Queue
import cv2

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Thread-safe wrapper for a single camera source.
    Handles frame reading in a separate thread with auto-reconnection.
    """
    
    def __init__(self, camera_id: str, source, reconnect_delay: float = 3.0):
        """
        Args:
            camera_id: Unique identifier for this camera
            source: Camera source (int for USB, str for RTSP URL)
            reconnect_delay: Seconds to wait before reconnection attempts
        """
        self.camera_id = camera_id
        self.source = source
        self.reconnect_delay = reconnect_delay
        
        self.cap = None
        self.is_running = False
        self.connected = False
        self.frame_queue = Queue(maxsize=5)
        
        self.read_thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = None
        
    def connect(self) -> bool:
        """Establish connection to camera source"""
        try:
            if isinstance(self.source, str):
                # RTSP/HTTP stream
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
            else:
                # USB camera
                self.cap = cv2.VideoCapture(int(self.source))
            
            if self.cap.isOpened():
                self.connected = True
                logger.info(f"Camera {self.camera_id} connected to {self.source}")
                return True
            else:
                logger.error(f"Camera {self.camera_id} failed to open")
                return False
                
        except Exception as e:
            logger.error(f"Camera {self.camera_id} connection error: {e}")
            return False
    
    def start(self):
        """Start the frame reading thread"""
        if self.is_running:
            logger.warning(f"Camera {self.camera_id} already running")
            return
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        logger.info(f"Camera {self.camera_id} thread started")
    
    def _read_loop(self):
        """Background thread that continuously reads frames"""
        while self.is_running:
            if not self.connected:
                # Attempt reconnection
                logger.info(f"Camera {self.camera_id} attempting reconnection...")
                if self.connect():
                    time.sleep(0.5)  # Brief stabilization
                else:
                    time.sleep(self.reconnect_delay)
                    continue
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"Camera {self.camera_id} frame read failed")
                    self.connected = False
                    self._release_cap()
                    continue
                
                # Update statistics
                with self.lock:
                    self.frame_count += 1
                    self.last_frame_time = time.time()
                
                # Add frame to queue (drop oldest if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Drop oldest
                        self.dropped_frames += 1
                    except:
                        pass
                
                self.frame_queue.put(frame, block=False)
                
            except Exception as e:
                logger.error(f"Camera {self.camera_id} read error: {e}")
                self.connected = False
                self._release_cap()
                time.sleep(self.reconnect_delay)
    
    def get_frame(self) -> Optional[tuple]:
        """
        Get the latest frame from queue (non-blocking)
        
        Returns:
            (frame, timestamp) or (None, None) if no frame available
        """
        try:
            frame = self.frame_queue.get_nowait()
            timestamp = time.time()
            return frame, timestamp
        except:
            return None, None
    
    def _release_cap(self):
        """Release the video capture safely"""
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        finally:
            self.cap = None
    
    def stop(self):
        """Stop the reading thread and release resources"""
        logger.info(f"Stopping camera {self.camera_id}")
        self.is_running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        
        self._release_cap()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
    
    def get_stats(self) -> dict:
        """Get camera statistics"""
        with self.lock:
            fps = 0
            if self.last_frame_time:
                elapsed = time.time() - self.last_frame_time
                if elapsed < 5.0:  # Only calculate if recent
                    fps = self.frame_count / max(elapsed, 1)
            
            return {
                'camera_id': self.camera_id,
                'connected': self.connected,
                'frame_count': self.frame_count,
                'dropped_frames': self.dropped_frames,
                'fps': round(fps, 2),
                'queue_size': self.frame_queue.qsize()
            }


class CameraProcessor:
    """
    Independent processor for a single camera.
    Manages detection, tracking, and output for one camera stream.
    """
    
    def __init__(self, camera_id: str, camera_stream: CameraStream, 
                 detection_callback: Optional[Callable] = None):
        """
        Args:
            camera_id: Camera identifier
            camera_stream: CameraStream instance
            detection_callback: Function to call with detections (camera_id, frame, detections)
        """
        self.camera_id = camera_id
        self.camera_stream = camera_stream
        self.detection_callback = detection_callback
        
        self.is_running = False
        self.process_thread = None
        
        # Latest processed frame and results
        self.latest_frame = None
        self.latest_detections = None
        self.latest_annotated = None
        self.lock = threading.Lock()
        
        # Processing statistics
        self.processed_count = 0
        self.process_start_time = time.time()
    
    def start(self):
        """Start the processing thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        logger.info(f"Processor for camera {self.camera_id} started")
    
    def _process_loop(self):
        """Main processing loop - gets frames and runs detection"""
        while self.is_running:
            frame, timestamp = self.camera_stream.get_frame()
            
            if frame is None:
                time.sleep(0.01)  # Brief sleep if no frame
                continue
            
            # Call detection callback if provided
            if self.detection_callback:
                try:
                    detections, annotated_frame = self.detection_callback(self.camera_id, frame)
                    
                    # Store latest results
                    with self.lock:
                        self.latest_frame = frame
                        self.latest_detections = detections
                        self.latest_annotated = annotated_frame
                        self.processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Detection callback error for camera {self.camera_id}: {e}")
    
    def get_latest(self) -> tuple:
        """
        Get latest processed frame and detections
        
        Returns:
            (frame, detections, annotated_frame)
        """
        with self.lock:
            return self.latest_frame, self.latest_detections, self.latest_annotated
    
    def stop(self):
        """Stop the processing thread"""
        logger.info(f"Stopping processor for camera {self.camera_id}")
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        elapsed = time.time() - self.process_start_time
        fps = self.processed_count / max(elapsed, 1)
        
        return {
            'camera_id': self.camera_id,
            'processed_frames': self.processed_count,
            'processing_fps': round(fps, 2)
        }


class CameraManager:
    """
    Central orchestrator for multiple camera streams.
    Manages camera lifecycle, processing, and monitoring.
    """
    
    def __init__(self, max_cameras: int = 10):
        """
        Args:
            max_cameras: Maximum number of concurrent cameras
        """
        self.max_cameras = max_cameras
        
        self.cameras: Dict[str, CameraStream] = {}
        self.processors: Dict[str, CameraProcessor] = {}
        
        self.lock = threading.Lock()
        
        logger.info(f"CameraManager initialized (max cameras: {max_cameras})")
    
    def add_camera(self, camera_id: str, source, 
                   detection_callback: Optional[Callable] = None,
                   reconnect_delay: float = 3.0) -> bool:
        """
        Add a new camera to the manager
        
        Args:
            camera_id: Unique identifier for the camera
            source: Camera source (int for USB, str for RTSP)
            detection_callback: Function to process frames
            reconnect_delay: Reconnection delay in seconds
            
        Returns:
            True if camera added successfully, False otherwise
        """
        with self.lock:
            if len(self.cameras) >= self.max_cameras:
                logger.error(f"Maximum camera limit ({self.max_cameras}) reached")
                return False
            
            if camera_id in self.cameras:
                logger.error(f"Camera {camera_id} already exists")
                return False
            
            try:
                # Create camera stream
                camera_stream = CameraStream(camera_id, source, reconnect_delay)
                camera_stream.start()
                
                # Create processor
                processor = CameraProcessor(camera_id, camera_stream, detection_callback)
                processor.start()
                
                self.cameras[camera_id] = camera_stream
                self.processors[camera_id] = processor
                
                logger.info(f"Camera {camera_id} added successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add camera {camera_id}: {e}")
                return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera from the manager
        
        Args:
            camera_id: Camera to remove
            
        Returns:
            True if removed successfully
        """
        with self.lock:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False
            
            try:
                # Stop processor
                if camera_id in self.processors:
                    self.processors[camera_id].stop()
                    del self.processors[camera_id]
                
                # Stop camera stream
                self.cameras[camera_id].stop()
                del self.cameras[camera_id]
                
                logger.info(f"Camera {camera_id} removed")
                return True
                
            except Exception as e:
                logger.error(f"Error removing camera {camera_id}: {e}")
                return False
    
    def get_camera_list(self) -> list:
        """Get list of all active cameras"""
        with self.lock:
            return list(self.cameras.keys())
    
    def get_camera_stream(self, camera_id: str) -> Optional[CameraStream]:
        """Get camera stream by ID"""
        return self.cameras.get(camera_id)
    
    def get_processor(self, camera_id: str) -> Optional[CameraProcessor]:
        """Get processor by camera ID"""
        return self.processors.get(camera_id)
    
    def get_system_stats(self) -> dict:
        """Get overall system statistics"""
        with self.lock:
            camera_stats = []
            total_fps = 0
            
            for camera_id in self.cameras:
                cam_stats = self.cameras[camera_id].get_stats()
                proc_stats = self.processors[camera_id].get_stats()
                
                combined = {**cam_stats, **proc_stats}
                camera_stats.append(combined)
                total_fps += combined.get('processing_fps', 0)
            
            return {
                'total_cameras': len(self.cameras),
                'max_cameras': self.max_cameras,
                'average_fps': round(total_fps / max(len(self.cameras), 1), 2),
                'cameras': camera_stats
            }
    
    def shutdown(self):
        """Shutdown all cameras and cleanup"""
        logger.info("Shutting down CameraManager")
        
        with self.lock:
            camera_ids = list(self.cameras.keys())
        
        for camera_id in camera_ids:
            self.remove_camera(camera_id)
        
        logger.info("CameraManager shutdown complete")
