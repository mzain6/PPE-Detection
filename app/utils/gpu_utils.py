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


def get_gpu_utilization(device_id: int = 0) -> float:
    """
    Get GPU compute utilization percentage.
    
    Args:
        device_id: GPU device ID
    
    Returns:
        Utilization percentage (0-100), or 0 if unavailable
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(utilization.gpu)
    except ImportError:
        logger.debug("pynvml not available, GPU utilization unavailable")
        return 0.0
    except Exception as e:
        logger.debug(f"Error getting GPU utilization: {e}")
        return 0.0


def estimate_model_memory_mb(model_path: str) -> float:
    """
    Estimate memory usage of a model file.
    
    Args:
        model_path: Path to model weights file
    
    Returns:
        Estimated memory in MB
    """
    import os
    try:
        if os.path.exists(model_path):
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            # Model in memory is typically 2-3x the file size (weights + activations + gradients)
            # For inference only, estimate 2.5x
            estimated_mb = file_size_mb * 2.5
            logger.debug(f"Model {model_path}: {file_size_mb:.1f}MB file, estimated {estimated_mb:.1f}MB in memory")
            return estimated_mb
        else:
            logger.warning(f"Model file not found: {model_path}")
            return 0.0
    except Exception as e:
        logger.error(f"Error estimating model memory: {e}")
        return 0.0


def calculate_max_model_instances(model_path: str, device_id: int = 0, reserved_gb: float = 0.5) -> int:
    """
    Calculate maximum number of model instances that can fit in GPU memory.
    
    Args:
        model_path: Path to model weights
        device_id: GPU device ID
        reserved_gb: Amount of VRAM to keep free (GB)
    
    Returns:
        Maximum number of instances, or -1 if GPU unavailable
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("GPU not available, unlimited CPU instances")
            return -1  # Unlimited for CPU
        
        mem_info = get_gpu_memory_info(device_id)
        total_gb = mem_info.get("total_gb", 0)
        
        if total_gb == 0:
            return -1
        
        # Calculate available memory
        available_gb = total_gb - reserved_gb
        
        # Estimate model memory
        model_memory_mb = estimate_model_memory_mb(model_path)
        model_memory_gb = model_memory_mb / 1024
        
        if model_memory_gb == 0:
            logger.warning("Could not estimate model memory, defaulting to 1 instance")
            return 1
        
        # Calculate max instances
        max_instances = int(available_gb / model_memory_gb)
        max_instances = max(1, max_instances)  # At least 1
        
        logger.info(f"GPU memory: {total_gb:.2f}GB total, {available_gb:.2f}GB available")
        logger.info(f"Model memory: {model_memory_gb:.2f}GB per instance")
        logger.info(f"Max instances: {max_instances}")
        
        return max_instances
        
    except Exception as e:
        logger.error(f"Error calculating max instances: {e}")
        return 1  # Safe default


class GPUMemoryMonitor:
    """
    Context manager for monitoring GPU memory usage during operations.
    
    Usage:
        with GPUMemoryMonitor(device_id=0) as monitor:
            # Your GPU operation
            model.load()
        
        print(f"Memory used: {monitor.memory_used_mb}MB")
    """
    
    def __init__(self, device_id: int = 0, name: str = "Operation"):
        self.device_id = device_id
        self.name = name
        self.memory_before = 0.0
        self.memory_after = 0.0
        self.memory_used_mb = 0.0
    
    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device_id)
                self.memory_before = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
        except Exception as e:
            logger.debug(f"Error in GPUMemoryMonitor enter: {e}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device_id)
                self.memory_after = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
                self.memory_used_mb = self.memory_after - self.memory_before
                logger.info(f"{self.name}: Used {self.memory_used_mb:.1f}MB GPU memory")
        except Exception as e:
            logger.debug(f"Error in GPUMemoryMonitor exit: {e}")
        return False
