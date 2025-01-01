import pytest
import torch
import torch.nn as nn
from ..device_management import DeviceManager
from ..tensor_utils import TensorOps

@pytest.fixture(scope="session")
def device_manager():
    """Session-wide DeviceManager instance."""
    return DeviceManager()

@pytest.fixture(scope="session")
def tensor_ops():
    """Session-wide TensorOps instance."""
    return TensorOps()

@pytest.fixture
def sample_tensor():
    """Creates a sample tensor for testing."""
    return torch.randn(10, 5)

@pytest.fixture
def sample_model():
    """Creates a sample model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    return SimpleModel()

@pytest.fixture
def cuda_required():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available") 