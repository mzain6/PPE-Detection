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
