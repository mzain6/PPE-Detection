"""
benchmark_batch_inference.py

Benchmark script to compare single-frame vs batch inference performance.
Simulates multiple cameras and measures FPS improvements.
"""

import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.yolov8_detector import YoloV8Detector
from app.services.batch_coordinator import BatchCoordinator
from app.utils.gpu_utils import get_gpu_memory_info, get_cuda_info
from app.config import settings

def generate_test_frame(width=640, height=480):
    """Generate a random test frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def benchmark_single_frame(detector, num_frames=100):
    """
    Benchmark single-frame processing.
    
    Args:
        detector: YoloV8Detector instance
        num_frames: Number of frames to process
    
    Returns:
        Average FPS
    """
    print(f"\n[Single-Frame Mode] Processing {num_frames} frames...")
    
    frames_processed = 0
    start_time = time.time()
    
    for i in range(num_frames):
        frame = generate_test_frame()
        result = detector.infer(frame)
        frames_processed += 1
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            current_fps = frames_processed / elapsed
            print(f"  Progress: {i+1}/{num_frames} frames, Current FPS: {current_fps:.2f}")
    
    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time
    
    print(f"  ✓ Completed: {frames_processed} frames in {total_time:.2f}s")
    print(f"  ✓ Average FPS: {avg_fps:.2f}")
    
    return avg_fps


def benchmark_batch_processing(detector, batch_size=8, num_frames=100):
    """
    Benchmark batch processing.
    
    Args:
        detector: YoloV8Detector instance
        batch_size: Frames per batch
        num_frames: Total frames to process
    
    Returns:
        Average FPS
    """
    print(f"\n[Batch Mode] Processing {num_frames} frames (batch size: {batch_size})...")
    
    frames_processed = 0
    start_time = time.time()
    
    # Process in batches
    num_batches = (num_frames + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        # Generate batch of frames
        current_batch_size = min(batch_size, num_frames - frames_processed)
        batch_frames = [generate_test_frame() for _ in range(current_batch_size)]
        
        # Run batch inference
        results = detector.infer_batch(batch_frames)
        frames_processed += len(batch_frames)
        
        if (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            current_fps = frames_processed / elapsed
            print(f"  Progress: Batch {batch_idx+1}/{num_batches}, {frames_processed}/{num_frames} frames, Current FPS: {current_fps:.2f}")
    
    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time
    
    print(f"  ✓ Completed: {frames_processed} frames in {total_time:.2f}s")
    print(f"  ✓ Average FPS: {avg_fps:.2f}")
    
    return avg_fps


def multi_camera_simulation(num_cameras=4, frames_per_camera=50, batch_size=8):
    """
    Simulate multiple cameras with batch processing.
    
    Args:
        num_cameras: Number of cameras to simulate
        frames_per_camera: Frames to process per camera
        batch_size: Batch size for processing
    
    Returns:
        Statistics dictionary
    """
    print(f"\n[Multi-Camera Simulation] {num_cameras} cameras, {frames_per_camera} frames each...")
    
    detector = YoloV8Detector(
        model_path=settings.model_path,
        device=settings.device,
        camera_id="batch_benchmark"
    )
    
    coordinator = BatchCoordinator(batch_size=batch_size, timeout_ms=50)
    
    # Submit frames from all cameras
    total_frames = 0
    start_time = time.time()
    
    for frame_idx in range(frames_per_camera):
        for cam_idx in range(num_cameras):
            frame = generate_test_frame()
            coordinator.submit_frame(f"camera_{cam_idx}", frame)
            total_frames += 1
    
    print(f"  Submitted {total_frames} frames from {num_cameras} cameras")
    
    # Process all batches
    frames_processed = 0
    batches_processed = 0
    
    while frames_processed < total_frames:
        camera_ids, frames, timestamps = coordinator.get_batch(timeout_sec=0.1)
        
        if not frames:
            break  # No more frames
        
        # Run batch inference
        results = detector.infer_batch(frames)
        frames_processed += len(frames)
        batches_processed += 1
    
    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time
    
    batch_stats = coordinator.get_stats()
    
    print(f"  ✓ Processed: {frames_processed}/{total_frames} frames in {batches_processed} batches")
    print(f"  ✓ Total time: {total_time:.2f}s")
    print(f"  ✓ Average FPS: {avg_fps:.2f}")
    print(f"  ✓ Average batch size: {batch_stats['average_batch_size']:.2f}")
    print(f"  ✓ Batch utilization: {batch_stats['batch_utilization_pct']:.1f}%")
    
    detector.close()
    
    return {
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "batch_stats": batch_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch inference performance")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--cameras", type=int, default=4, help="Number of cameras for multi-camera simulation (default: 4)")
    parser.add_argument("--skip-single", action="store_true", help="Skip single-frame benchmark")
    parser.add_argument("--skip-batch", action="store_true", help="Skip batch benchmark")
    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-camera simulation")
    
    args = parser.parse_args()
    
    print("="*70)
    print(" Batch Inference Performance Benchmark")
    print("="*70)
    
    # Display system info
    cuda_info = get_cuda_info()
    print(f"\n📊 System Information:")
    print(f"   CUDA Available: {cuda_info['cuda_available']}")
    if cuda_info['cuda_available']:
        print(f"   CUDA Version: {cuda_info['cuda_version']}")
        print(f"   GPU Count: {cuda_info['device_count']}")
        for device in cuda_info['devices']:
            print(f"   GPU {device['id']}: {device['name']} ({device['memory_total'] / 1024**3:.2f} GB)")
        
        mem_info = get_gpu_memory_info(0)
        print(f"\n💾 GPU Memory (Device 0):")
        print(f"   Total: {mem_info['total_gb']:.2f} GB")
        print(f"   Allocated: {mem_info['allocated_gb']:.2f} GB")
        print(f"   Free: {mem_info['free_gb']:.2f} GB")
    
    print(f"\n⚙️  Benchmark Configuration:")
    print(f"   Model: {settings.model_path}")
    print(f"   Device: {settings.device}")
    print(f"   Input Size: {settings.input_size}")
    print(f"   Frames to process: {args.frames}")
    print(f"   Batch size: {args.batch_size}")
    
    # Initialize detector
    print(f"\n🔧 Initializing detector...")
    detector = YoloV8Detector(
        model_path=settings.model_path,
        device=settings.device,
        camera_id="benchmark"
    )
    print(f"   ✓ Detector initialized")
    
    results = {}
    
    # Single-frame benchmark
    if not args.skip_single:
        single_fps = benchmark_single_frame(detector, args.frames)
        results['single_frame_fps'] = single_fps
    
    # Batch benchmark
    if not args.skip_batch:
        batch_fps = benchmark_batch_processing(detector, args.batch_size, args.frames)
        results['batch_fps'] = batch_fps
    
    # Multi-camera simulation
    if not args.skip_multi:
        multi_stats = multi_camera_simulation(args.cameras, args.frames // args.cameras, args.batch_size)
        results['multi_camera'] = multi_stats
    
    # Summary
    print("\n" + "="*70)
    print(" BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    if 'single_frame_fps' in results and 'batch_fps' in results:
        improvement = ((results['batch_fps'] - results['single_frame_fps']) / results['single_frame_fps']) * 100
        print(f"\n📈 Performance Comparison:")
        print(f"   Single-frame FPS: {results['single_frame_fps']:.2f}")
        print(f"   Batch FPS: {results['batch_fps']:.2f}")
        print(f"   Improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"   ✅ Batch processing is {improvement:.1f}% FASTER!")
        else:
            print(f"   ⚠️  Batch processing is {abs(improvement):.1f}% slower")
    
    if 'multi_camera' in results:
        print(f"\n 🎥 Multi-Camera Performance:")
        print(f"   {args.cameras} cameras @ {results['multi_camera']['avg_fps']:.2f} FPS")
        print(f"   Batch utilization: {results['multi_camera']['batch_stats']['batch_utilization_pct']:.1f}%")
    
    # Final GPU memory
    if cuda_info['cuda_available']:
        mem_info = get_gpu_memory_info(0)
        print(f"\n💾 Final GPU Memory:")
        print(f"   Allocated: {mem_info['allocated_gb']:.2f} GB")
        print(f"   Free: {mem_info['free_gb']:.2f} GB")
    
    print("\n" + "="*70)
    print("✓ Benchmark complete!")
    print("="*70 + "\n")
    
    detector.close()


if __name__ == "__main__":
    main()
