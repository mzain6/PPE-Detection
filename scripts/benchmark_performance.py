"""
Performance benchmarking script for PPE detection system.

Measures:
- Inference FPS
- GPU utilization and memory
- CPU and system memory usage
- Detection latency
"""
import argparse
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ai.yolov8_detector import YoloV8Detector
from app.utils.gpu_utils import get_cuda_info, get_gpu_memory_info, select_device
import psutil


def benchmark_detector(detector, num_frames=100, resolution=(1920, 1080)):
    """Run benchmark on detector."""
    print(f"\nRunning benchmark with {num_frames} frames at {resolution}...")
    
    # Generate dummy frames
    frames = [np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8) 
              for _ in range(min(10, num_frames))]
    
    latencies = []
    start_time = time.time()
    
    for i in range(num_frames):
        frame = frames[i % len(frames)]
        
        frame_start = time.time()
        detections = detector.infer(frame)
        frame_end = time.time()
        
        latencies.append(frame_end - frame_start)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        "total_time": total_time,
        "total_frames": num_frames,
        "fps": num_frames / total_time,
        "avg_latency": np.mean(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "std_latency": np.std(latencies)
    }


def main():
    parser = argparse.ArgumentParser(description="PPE Detection Performance Benchmark")
    parser.add_argument("--frames", type=int, default=100, 
                       help="Number of frames to process")
    parser.add_argument("--resolution", type=str, default="1920x1080",
                       help="Frame resolution (WxH)")
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    print("=" * 60)
    print("PPE Detection Performance Benchmark")
    print("=" * 60)
    
    # System info
    print("\n[System Information]")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    
    # GPU info
    gpu_info = get_cuda_info()
    print(f"\n[GPU Information]")
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print(f"GPU Count: {gpu_info['device_count']}")
        for dev in gpu_info['devices']:
            print(f"  - {dev['name']}: {dev['memory_total'] / 1024**3:.2f} GB")
    
    # Model info
    print(f"\n[Model Configuration]")
    print(f"Model: {settings.model_path}")
    print(f"Input Size: {settings.input_size}")
    print(f"Confidence Threshold: {settings.confidence_threshold}")
    
    # Initialize detector
    print(f"\n[Initializing Detector]")
    device = select_device(settings.device)
    print(f"Device: {device or 'CPU'}")
    
    detector = YoloV8Detector(
        model_path=settings.model_path,
        device=device,
        camera_id="benchmark"
    )
    
    # Get initial GPU memory
    if gpu_info['cuda_available']:
        initial_gpu_mem = get_gpu_memory_info(0)
        print(f"Initial GPU Memory: {initial_gpu_mem['allocated_gb']:.2f} GB allocated")
    
    # Run benchmark
    results = benchmark_detector(detector, args.frames, resolution)
    
    # Display results
    print("\n" + "=" * 60)
    print("[Benchmark Results]")
    print("=" * 60)
    print(f"Total Frames: {results['total_frames']}")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"Average FPS: {results['fps']:.2f}")
    print(f"Average Latency: {results['avg_latency']*1000:.2f}ms")
    print(f"Min Latency: {results['min_latency']*1000:.2f}ms")
    print(f"Max Latency: {results['max_latency']*1000:.2f}ms")
    print(f"Std Latency: {results['std_latency']*1000:.2f}ms")
    
    # Final GPU memory
    if gpu_info['cuda_available']:
        final_gpu_mem = get_gpu_memory_info(0)
        print(f"\n[GPU Memory Usage]")
        print(f"Allocated: {final_gpu_mem['allocated_gb']:.2f} GB")
        print(f"Reserved: {final_gpu_mem['reserved_gb']:.2f} GB")
        print(f"Free: {final_gpu_mem['free_gb']:.2f} GB")
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    print(f"\n[System Resources]")
    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"Memory Usage: {mem.percent:.1f}% ({mem.used / 1024**3:.2f} GB / {mem.total / 1024**3:.2f} GB)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
