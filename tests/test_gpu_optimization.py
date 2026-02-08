"""
GPU optimization tests - requires NVIDIA GPU with CUDA.
"""
import pytest
from app.utils.gpu_utils import (
    get_cuda_info,
    select_device,
    get_gpu_memory_info,
    clear_gpu_cache,
    optimize_for_inference
)


def test_get_cuda_info():
    """Test CUDA information retrieval."""
    info = get_cuda_info()
    
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "cuda_version" in info
    assert "device_count" in info
    assert "devices" in info
    
    if info["cuda_available"]:
        assert info["device_count"] > 0
        assert len(info["devices"]) == info["device_count"]


def test_select_device_cpu():
    """Test CPU device selection."""
    device = select_device("cpu")
    assert device is None


def test_select_device_auto():
    """Test automatic device selection."""
    device = select_device("auto")
    # Should return either None (CPU) or a device string (GPU)
    assert device is None or isinstance(device, str)


@pytest.mark.gpu
def test_gpu_memory_info():
    """Test GPU memory information retrieval (requires GPU)."""
    cuda_info = get_cuda_info()
    
    if not cuda_info["cuda_available"]:
        pytest.skip("CUDA not available")
    
    mem_info = get_gpu_memory_info(0)
    
    assert isinstance(mem_info, dict)
    assert "total_gb" in mem_info
    assert "allocated_gb" in mem_info
    assert "reserved_gb" in mem_info
    assert "free_gb" in mem_info
    
    assert mem_info["total_gb"] > 0


@pytest.mark.gpu
def test_clear_gpu_cache():
    """Test GPU cache clearing (requires GPU)."""
    cuda_info = get_cuda_info()
    
    if not cuda_info["cuda_available"]:
        pytest.skip("CUDA not available")
    
    # Should not raise any exceptions
    clear_gpu_cache()


@pytest.mark.gpu
def test_optimize_for_inference():
    """Test inference optimization (requires GPU)."""
    cuda_info = get_cuda_info()
    
    if not cuda_info["cuda_available"]:
        pytest.skip("CUDA not available")
    
    # Should not raise any exceptions
    optimize_for_inference()


def test_estimate_model_memory():
    """Test model memory estimation."""
    from app.utils.gpu_utils import estimate_model_memory_mb
    
    # Test with non-existent file
    memory = estimate_model_memory_mb("nonexistent.pt")
    assert memory == 0.0
    
    # Test with real model file if exists
    import os
    if os.path.exists("best.pt"):
        memory = estimate_model_memory_mb("best.pt")
        assert memory > 0


def test_calculate_max_model_instances():
    """Test max model instances calculation."""
    from app.utils.gpu_utils import calculate_max_model_instances
    
    # Test with non-existent model
    max_instances = calculate_max_model_instances("nonexistent.pt")
    assert max_instances >= -1  # -1 for unlimited (CPU) or positive number
    
    # If GPU available, should return positive number
    cuda_info = get_cuda_info()
    if cuda_info["cuda_available"]:
        # Test with real model if exists
        import os
        if os.path.exists("best.pt"):
            max_instances = calculate_max_model_instances("best.pt", reserved_gb=0.5)
            assert max_instances >= 1


def test_get_gpu_utilization():
    """Test GPU utilization retrieval."""
    from app.utils.gpu_utils import get_gpu_utilization
    
    utilization = get_gpu_utilization(0)
    assert isinstance(utilization, float)
    assert utilization >= 0


@pytest.mark.gpu
def test_gpu_memory_monitor():
    """Test GPUMemoryMonitor context manager."""
    from app.utils.gpu_utils import GPUMemoryMonitor
    import torch
    
    cuda_info = get_cuda_info()
    if not cuda_info["cuda_available"]:
        pytest.skip("CUDA not available")
    
    with GPUMemoryMonitor(device_id=0, name="Test Operation") as monitor:
        # Allocate some memory
        tensor = torch.zeros((1000, 1000), device='cuda')
    
    # Memory usage should be recorded
    assert isinstance(monitor.memory_used_mb, float)
    
    # Clean up
    del tensor
    torch.cuda.empty_cache()


def test_gpu_memory_monitor_no_gpu():
    """Test GPUMemoryMonitor when GPU not available."""
    from app.utils.gpu_utils import GPUMemoryMonitor
    
    # Should not raise even without GPU
    with GPUMemoryMonitor(device_id=0, name="Test") as monitor:
        pass
    
    # Should have zero memory usage
    assert monitor.memory_used_mb == 0.0
