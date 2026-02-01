"""
GPU utilities for device detection, monitoring, and optimization.
"""
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_cuda_info() -> Dict[str, Any]:
    """
    Get CUDA availability and device information.
    
    Returns:
        Dictionary with CUDA information including availability, version, and device count.
    """
    info = {
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": []
    }
    
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            
            for i in range(info["device_count"]):
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
                }
                info["devices"].append(device_info)
                
        logger.info(f"CUDA Available: {info['cuda_available']}")
        if info["cuda_available"]:
            logger.info(f"CUDA Version: {info['cuda_version']}")
            logger.info(f"GPU Count: {info['device_count']}")
            for device in info["devices"]:
                logger.info(f"GPU {device['id']}: {device['name']} ({device['memory_total'] / 1024**3:.2f} GB)")
    except ImportError:
        logger.warning("PyTorch not available, GPU support disabled")
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {e}")
    
    return info


def select_device(device: str = "auto") -> Optional[str]:
    """
    Select the best available device for inference.
    
    Args:
        device: Device preference - "auto", "cpu", "cuda", or specific device like "cuda:0"
    
    Returns:
        Device string to use, or None for CPU
    """
    if device == "cpu":
        logger.info("Using CPU (explicitly requested)")
        return None
    
    if device == "auto":
        cuda_info = get_cuda_info()
        if cuda_info["cuda_available"] and cuda_info["device_count"] > 0:
            logger.info("Auto-selected GPU device")
            return "0"  # Use first GPU
        else:
            logger.info("No GPU available, using CPU")
            return None
    
    # Specific device requested (e.g., "cuda", "cuda:0", "0")
    cuda_info = get_cuda_info()
    if cuda_info["cuda_available"]:
        # Normalize device string
        if device == "cuda":
            return "0"
        elif device.startswith("cuda:"):
            return device.split(":")[1]
        else:
            return device
    else:
        logger.warning(f"GPU device '{device}' requested but CUDA not available, falling back to CPU")
        return None


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """
    Get GPU memory usage information.
    
    Args:
        device_id: GPU device ID
    
    Returns:
        Dictionary with memory information in GB
    """
    info = {
        "total_gb": 0.0,
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "free_gb": 0.0
    }
    
    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            
            info["total_gb"] = total / 1024**3
            info["allocated_gb"] = allocated / 1024**3
            info["reserved_gb"] = reserved / 1024**3
            info["free_gb"] = (total - reserved) / 1024**3
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
    
    return info


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU cache: {e}")


def optimize_for_inference():
    """Apply optimizations for inference performance."""
    try:
        import torch
        if torch.cuda.is_available():
            # Enable cudnn benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmarking for inference optimization")
    except Exception as e:
        logger.error(f"Error applying inference optimizations: {e}")
