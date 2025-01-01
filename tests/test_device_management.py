"""Tests for the device management module.

This module contains tests for device management functionality, including
multi-GPU support, distributed training, and profiling capabilities.
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import List, Tuple

from pytorch_basics_library.device_management import DeviceManager, device_manager

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def device_mgr() -> DeviceManager:
    """Create a fresh DeviceManager instance for testing."""
    return DeviceManager()

@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for testing."""
    return SimpleModel()

@pytest.fixture
def tensor_2d() -> torch.Tensor:
    """Create a 2D tensor for testing."""
    return torch.randn(10, 10)

def test_get_available_devices(device_mgr: DeviceManager):
    """Test getting list of available devices."""
    devices = device_mgr.get_available_devices()
    assert len(devices) >= 1  # At least CPU should be available
    assert devices[0] == torch.device('cpu')
    
    if torch.cuda.is_available():
        assert len(devices) == torch.cuda.device_count() + 1
        assert all(d.type == 'cuda' for d in devices[1:])

def test_get_device_with_id(device_mgr: DeviceManager):
    """Test getting specific GPU device."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            device_mgr.get_device(0)
    else:
        device = device_mgr.get_device(0)
        assert device.type == 'cuda'
        assert device.index == 0
        
        # Test invalid device ID
        with pytest.raises(ValueError, match="GPU .* not found"):
            device_mgr.get_device(torch.cuda.device_count())

def test_to_device_with_id(
    device_mgr: DeviceManager,
    tensor_2d: torch.Tensor,
    simple_model: nn.Module
):
    """Test moving objects to specific device."""
    if torch.cuda.is_available():
        # Test tensor
        tensor = device_mgr.to_device(tensor_2d, device_id=0)
        assert tensor.device.type == 'cuda'
        assert tensor.device.index == 0
        
        # Test model
        model = device_mgr.to_device(simple_model, device_id=0)
        assert next(model.parameters()).device.type == 'cuda'
        assert next(model.parameters()).device.index == 0

def test_synchronize(device_mgr: DeviceManager):
    """Test CUDA synchronization."""
    if torch.cuda.is_available():
        # Create some GPU operations
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = a @ b
        
        # Synchronize should complete without error
        device_mgr.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_stats_per_device(device_mgr: DeviceManager):
    """Test memory statistics for specific device."""
    # Get stats for first GPU
    stats = device_mgr.get_memory_stats(0)
    assert 'allocated' in stats
    assert 'cached' in stats
    assert 'max_allocated' in stats
    
    # Allocate some memory and check stats
    tensor = torch.randn(1000, 1000, device='cuda:0')
    new_stats = device_mgr.get_memory_stats(0)
    assert new_stats['allocated'] > stats['allocated']

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profiler(device_mgr: DeviceManager):
    """Test CUDA profiler functionality."""
    # Start profiler
    device_mgr.start_profiler(wait=0, warmup=0, active=1)
    
    # Do some GPU operations
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    c = a @ b
    
    # Stop profiler
    device_mgr.stop_profiler()
    
    # Profile specific section
    def matrix_multiply():
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        return a @ b
    
    result, stats = device_mgr.profile_section("matrix_multiply", matrix_multiply)
    assert isinstance(result, torch.Tensor)
    assert 'cpu_time' in stats
    assert 'cuda_time' in stats
    assert 'memory_allocated' in stats

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clear_memory_per_device(device_mgr: DeviceManager):
    """Test memory clearing for specific device."""
    # Allocate some memory
    tensor = torch.randn(1000, 1000, device='cuda:0')
    initial_stats = device_mgr.get_memory_stats(0)
    
    # Clear memory
    device_mgr.clear_memory(0)
    
    # Check memory was cleared
    final_stats = device_mgr.get_memory_stats(0)
    assert final_stats['allocated'] <= initial_stats['allocated']

def test_device_properties_per_device(device_mgr: DeviceManager):
    """Test getting device properties for specific device."""
    if torch.cuda.is_available():
        props = device_mgr.get_device_properties(0)
        assert 'name' in props
        assert 'total_memory' in props
        assert 'compute_capability' in props
        assert 'multi_processor_count' in props
        assert 'max_threads_per_block' in props
        assert 'warp_size' in props
    else:
        props = device_mgr.get_device_properties()
        assert props == {"device": "cpu"}

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_distributed_setup(device_mgr: DeviceManager):
    """Test distributed setup functionality.
    
    Note: This test is more of a syntax check as actual distributed
    setup requires multiple processes.
    """
    with pytest.raises(RuntimeError):
        # Should fail without proper environment setup
        device_mgr.setup_distributed(0, 2)
    
    # Test cleanup
    device_mgr.cleanup_distributed()
    assert not device_mgr.is_distributed
    assert device_mgr.world_size == 1
    assert device_mgr.rank == 0

def test_to_device_distributed(
    device_mgr: DeviceManager,
    simple_model: nn.Module
):
    """Test distributed model wrapping.
    
    Note: This test checks the error case as actual distributed
    setup requires multiple processes.
    """
    with pytest.raises(RuntimeError, match="Must call setup_distributed first"):
        device_mgr.to_device(simple_model, distributed=True)

def test_invalid_inputs(device_mgr: DeviceManager):
    """Test error handling for invalid inputs."""
    # Test invalid object type
    with pytest.raises(TypeError, match="must be a torch.Tensor or torch.nn.Module"):
        device_mgr.to_device([1, 2, 3])
    
    # Test invalid device ID
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            device_mgr.get_device(999)
    
    # Test profiler without CUDA
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA profiler requires CUDA"):
            device_mgr.start_profiler()

def test_global_instance():
    """Test that global device_manager instance works correctly."""
    assert isinstance(device_manager, DeviceManager)
    assert device_manager.device is not None 