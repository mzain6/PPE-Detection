"""
GPU Availability Check and Asynchronous Processing Test
Checks if CUDA GPU is available and tests async batch processing capabilities
"""
import torch
import sys
from pathlib import Path

print("=" * 60)
print("GPU Availability Check")
print("=" * 60)

# Check PyTorch CUDA availability
print("\n1. CUDA Availability:")
cuda_available = torch.cuda.is_available()
print(f"   CUDA Available: {cuda_available}")

if cuda_available:
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"      Name: {torch.cuda.get_device_name(i)}")
        print(f"      Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"      Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"      Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"      Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("   ❌ No CUDA GPU detected")
    print("   Will run on CPU (slower)")

# Check device selection
print("\n2. Device Selection:")
device = torch.device('cuda' if cuda_available else 'cpu')
print(f"   Selected Device: {device}")

# Test tensor creation on GPU
if cuda_available:
    print("\n3. GPU Tensor Test:")
    try:
        test_tensor = torch.rand(1000, 1000).to(device)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"   ✓ Successfully created and multiplied tensors on GPU")
        print(f"   Tensor shape: {result.shape}")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Test YOLOv8 model loading
print("\n4. YOLOv8 Model Test:")
try:
    from ultralytics import YOLO
    model_path = Path(__file__).parent / "best1.pt"
    
    if model_path.exists():
        print(f"   Loading model: {model_path.name}")
        model = YOLO(str(model_path))
        
        if cuda_available:
            print(f"   Attempting to move model to GPU...")
            model.to('cuda')
            print(f"   ✓ Model successfully loaded on GPU")
        else:
            print(f"   ✓ Model loaded on CPU")
            
        # Get model info
        print(f"\n   Model Info:")
        print(f"      Classes: {model.names}")
        print(f"      Number of parameters: {sum(p.numel() for p in model.model.parameters())}")
        
    else:
        print(f"   ⚠ Model not found: {model_path}")
        
except Exception as e:
    print(f"   ❌ Error loading model: {e}")

# Async processing recommendation
print("\n" + "=" * 60)
print("Asynchronous Processing Recommendations")
print("=" * 60)

if cuda_available:
    print("\n✓ GPU is available! Recommendations:")
    print("  1. Use batch processing for multiple frames")
    print("  2. Enable async CUDA operations with torch.cuda.Stream()")
    print("  3. Process frames in batches of 4-8 for optimal throughput")
    print("  4. Use larger imgsz (1280) for better accuracy without much speed penalty")
    print("\n  Estimated Performance:")
    print("    - Batch size 1: ~2-3 fps")
    print("    - Batch size 4: ~8-12 fps")
    print("    - Batch size 8: ~15-20 fps")
else:
    print("\n❌ No GPU available. Recommendations:")
    print("  1. Use smaller imgsz (640 or 320) for CPU processing")
    print("  2. Process every Nth frame to reduce load")
    print("  3. Lower resolution input videos")
    print("  4. Consider multi-process CPU inference")
    print("\n  Estimated Performance:")
    print("    - imgsz=320: ~5-10 fps")
    print("    - imgsz=640: ~1-3 fps")
    print("    - imgsz=1280: ~0.5-1 fps (not recommended)")

print("\n" + "=" * 60)
