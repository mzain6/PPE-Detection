"""
batch_coordinator.py

Coordinates frame batching across multiple cameras for efficient GPU processing.
Collects frames from multiple cameras and processes them in batches.
"""

import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from queue import Queue, Empty
import numpy as np

logger = logging.getLogger(__name__)


class BatchCoordinator:
    """
    Coordinates frame collection and batch processing across cameras.
    
    Features:
    - Collects frames from multiple cameras
    - Creates batches up to configured size
    - Timeout mechanism to avoid waiting too long for full batch
    - Thread-safe operations
    - Performance metrics tracking
    """
    
    def __init__(self, batch_size: int = 8, timeout_ms: int = 100):
        """
        Args:
            batch_size: Maximum frames per batch
            timeout_ms: Max wait time to fill batch (milliseconds)
        """
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.timeout_sec = timeout_ms / 1000.0
        
        # Frame queue: (camera_id, frame, timestamp)
        self.frame_queue = Queue()
        
        # Statistics
        self.total_batches = 0
        self.total_frames = 0
        self.batch_sizes = []
        self.lock = threading.Lock()
        
        logger.info(f"BatchCoordinator initialized (batch_size={batch_size}, timeout={timeout_ms}ms)")
    
    def submit_frame(self, camera_id: str, frame: np.ndarray) -> bool:
        """
        Submit a frame for batch processing.
        
        Args:
            camera_id: Camera identifier
            frame: Frame to process
        
        Returns:
            True if submitted successfully
        """
        try:
            timestamp = time.time()
            self.frame_queue.put((camera_id, frame, timestamp), block=False)
            return True
        except Exception as e:
            logger.error(f"Error submitting frame from {camera_id}: {e}")
            return False
    
    def get_batch(self, timeout_sec: Optional[float] = None) -> Tuple[List[str], List[np.ndarray], List[float]]:
        """
        Collect frames into a batch.
        
        Blocks until:
        - Batch size is reached, OR
        - Timeout expires, OR
        - At least one frame is available and additional wait yields no more frames
        
        Args:
            timeout_sec: Override default timeout
        
        Returns:
            Tuple of (camera_ids, frames, timestamps)
        """
        timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        start_time = time.time()
        
        camera_ids = []
        frames = []
        timestamps = []
        
        # Try to get first frame (blocking with timeout)
        try:
            camera_id, frame, ts = self.frame_queue.get(timeout=timeout)
            camera_ids.append(camera_id)
            frames.append(frame)
            timestamps.append(ts)
        except Empty:
            # No frames available within timeout
            return camera_ids, frames, timestamps
        
        # Collect additional frames until batch full or timeout
        while len(frames) < self.batch_size:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                break
            
            try:
                # Try to get more frames (non-blocking, then short wait)
                camera_id, frame, ts = self.frame_queue.get(timeout=min(0.01, remaining))
                camera_ids.append(camera_id)
                frames.append(frame)
                timestamps.append(ts)
            except Empty:
                # No more frames immediately available
                # If we have at least one frame and timeout approaching, process what we have
                if len(frames) > 0 and (time.time() - start_time) > (timeout * 0.5):
                    break
        
        # Update statistics
        if len(frames) > 0:
            with self.lock:
                self.total_batches += 1
                self.total_frames += len(frames)
                self.batch_sizes.append(len(frames))
                # Keep only recent batch sizes
                if len(self.batch_sizes) > 100:
                    self.batch_sizes.pop(0)
        
        return camera_ids, frames, timestamps
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch processing statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            if len(self.batch_sizes) == 0:
                avg_batch_size = 0
            else:
                avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
            
            return {
                "total_batches": self.total_batches,
                "total_frames_processed": self.total_frames,
                "average_batch_size": round(avg_batch_size, 2),
                "max_batch_size": self.batch_size,
                "current_queue_size": self.frame_queue.qsize(),
                "batch_utilization_pct": round((avg_batch_size / self.batch_size) * 100, 1) if self.batch_size > 0 else 0
            }
    
    def clear_queue(self):
        """Clear all pending frames from queue."""
        cleared = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                cleared += 1
            except Empty:
                break
        
        logger.info(f"Cleared {cleared} pending frames from batch queue")
    
    def reset_stats(self):
        """Reset statistics counters."""
        with self.lock:
            self.total_batches = 0
            self.total_frames = 0
            self.batch_sizes = []
        logger.info("Batch coordinator statistics reset")
