"""
Test batch coordinator functionality.
"""
import pytest
import time
import numpy as np
from app.services.batch_coordinator import BatchCoordinator


def test_batch_coordinator_init():
    """Test BatchCoordinator initialization."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    assert coordinator.batch_size == 4
    assert coordinator.timeout_ms == 50
    assert coordinator.timeout_sec == 0.05
    assert coordinator.total_batches == 0
    assert coordinator.total_frames == 0


def test_submit_single_frame():
    """Test submitting a single frame."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = coordinator.submit_frame("camera_1", frame)
    
    assert result is True
    assert coordinator.frame_queue.qsize() == 1


def test_submit_multiple_frames():
    """Test submitting multiple frames from different cameras."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    for i in range(3):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = coordinator.submit_frame(f"camera_{i}", frame)
        assert result is True
    
    assert coordinator.frame_queue.qsize() == 3


def test_get_batch_single_frame():
    """Test getting a batch with a single frame."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    coordinator.submit_frame("camera_1", frame)
    
    camera_ids, frames, timestamps = coordinator.get_batch(timeout_sec=0.1)
    
    assert len(camera_ids) == 1
    assert len(frames) == 1
    assert len(timestamps) == 1
    assert camera_ids[0] == "camera_1"
    assert frames[0].shape == (480, 640, 3)


def test_get_batch_multiple_frames():
    """Test getting a batch with multiple frames."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=100)
    
    # Submit 3 frames
    for i in range(3):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coordinator.submit_frame(f"camera_{i}", frame)
    
    camera_ids, frames, timestamps = coordinator.get_batch(timeout_sec=0.2)
    
    assert len(camera_ids) == 3
    assert len(frames) == 3
    assert len(timestamps) == 3


def test_get_batch_timeout():
    """Test batch timeout behavior."""
    coordinator = BatchCoordinator(batch_size=10, timeout_ms=50)
    
    # Submit only 2 frames (less than batch size)
    for i in range(2):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coordinator.submit_frame(f"camera_{i}", frame)
    
    # Should return after timeout even though batch not full
    start_time =time.time()
    camera_ids, frames, timestamps = coordinator.get_batch()
    elapsed = time.time() - start_time
    
    assert len(camera_ids) == 2
    assert elapsed < 0.2  # Should timeout quickly


def test_get_batch_empty():
    """Test getting batch when no frames available."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    start_time = time.time()
    camera_ids, frames, timestamps = coordinator.get_batch(timeout_sec=0.1)
    elapsed = time.time() - start_time
    
    assert len(camera_ids) == 0
    assert len(frames) == 0
    assert elapsed >= 0.09  # Should wait for timeout


def test_batch_size_limit():
    """Test that batch size is limited to max."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=100)
    
    # Submit more frames than batch size
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coordinator.submit_frame(f"camera_{i}", frame)
    
    camera_ids, frames, timestamps = coordinator.get_batch()
    
    # Should return max batch size
    assert len(camera_ids) <= 4
    assert len(frames) <= 4


def test_get_stats():
    """Test statistics tracking."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    # Process a batch
    for i in range(3):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coordinator.submit_frame(f"camera_{i}", frame)
    
    camera_ids, frames, timestamps = coordinator.get_batch()
    
    stats = coordinator.get_stats()
    
    assert stats["total_batches"] == 1
    assert stats["total_frames_processed"] == 3
    assert stats["max_batch_size"] == 4
    assert stats["average_batch_size"] == 3.0


def test_clear_queue():
    """Test clearing the frame queue."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    # Submit several frames
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coordinator.submit_frame(f"camera_{i}", frame)
    
    assert coordinator.frame_queue.qsize() == 5
    
    coordinator.clear_queue()
    
    assert coordinator.frame_queue.qsize() == 0


def test_reset_stats():
    """Test resetting statistics."""
    coordinator = BatchCoordinator(batch_size=4, timeout_ms=50)
    
    # Process a batch
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    coordinator.submit_frame("camera_1", frame)
    coordinator.get_batch()
    
    assert coordinator.total_batches == 1
    
    coordinator.reset_stats()
    
    assert coordinator.total_batches == 0
    assert coordinator.total_frames == 0
    assert len(coordinator.batch_sizes) == 0


def test_batch_utilization():
    """Test batch utilization percentage."""
    coordinator = BatchCoordinator(batch_size=8, timeout_ms=50)
    
    # Submit 4 frames (50% utilization)
    for i in range(4):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coordinator.submit_frame(f"camera_{i}", frame)
    
    coordinator.get_batch()
    
    stats = coordinator.get_stats()
    assert stats["batch_utilization_pct"] == 50.0
