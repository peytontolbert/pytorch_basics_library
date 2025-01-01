"""Tests for the initializers module.

This module contains comprehensive tests for the weight initialization strategies
provided by the initializers module.
"""

import pytest
import torch
import torch.nn as nn
from typing import Type, Tuple

from pytorch_basics_library.initializers import (
    Initializer,
    XavierInitializer,
    KaimingInitializer,
    UniformInitializer,
    NormalInitializer,
    initialize_model,
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    uniform,
    normal
)

# Test fixtures
@pytest.fixture
def tensor_2d() -> torch.Tensor:
    """Create a 2D tensor for testing."""
    return torch.empty(10, 20)

@pytest.fixture
def tensor_4d() -> torch.Tensor:
    """Create a 4D tensor for testing convolutions."""
    return torch.empty(64, 32, 3, 3)

@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

# Base Initializer tests
def test_initializer_validate_parameters():
    """Test parameter validation in base Initializer."""
    init = Initializer()
    
    # Test invalid tensor type
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        init._validate_parameters([1, 2, 3])
    
    # Test invalid tensor dtype
    with pytest.raises(ValueError, match="floating point tensors"):
        init._validate_parameters(torch.tensor([1, 2, 3], dtype=torch.long))

def test_initializer_initialize_not_implemented():
    """Test NotImplementedError in base Initializer."""
    init = Initializer()
    tensor = torch.empty(10)
    with pytest.raises(NotImplementedError, match="must implement initialize"):
        init.initialize(tensor)

# Xavier Initializer tests
class TestXavierInitializer:
    """Tests for XavierInitializer."""
    
    def test_invalid_gain(self):
        """Test initialization with invalid gain."""
        with pytest.raises(ValueError, match="Gain must be positive"):
            XavierInitializer(gain=-1.0)
    
    def test_invalid_distribution(self):
        """Test initialization with invalid distribution."""
        with pytest.raises(ValueError, match="must be 'uniform' or 'normal'"):
            XavierInitializer(distribution='invalid')
    
    def test_initialize_2d(self, tensor_2d: torch.Tensor):
        """Test initialization of 2D tensor."""
        init = XavierInitializer(gain=1.0)
        init.initialize(tensor_2d)
        
        fan_in, fan_out = tensor_2d.size(1), tensor_2d.size(0)
        std = 1.0 * (2.0 / (fan_in + fan_out)) ** 0.5
        bound = std * 3 ** 0.5
        
        assert tensor_2d.min() >= -bound
        assert tensor_2d.max() <= bound
    
    def test_initialize_4d(self, tensor_4d: torch.Tensor):
        """Test initialization of 4D tensor."""
        init = XavierInitializer(distribution='normal')
        init.initialize(tensor_4d)
        
        fan_in = tensor_4d.size(1) * tensor_4d.size(2) * tensor_4d.size(3)
        fan_out = tensor_4d.size(0) * tensor_4d.size(2) * tensor_4d.size(3)
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        
        assert abs(tensor_4d.std() - std) < 0.1

# Kaiming Initializer tests
class TestKaimingInitializer:
    """Tests for KaimingInitializer."""
    
    def test_invalid_mode(self):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError, match="must be 'fan_in' or 'fan_out'"):
            KaimingInitializer(mode='invalid')
    
    def test_invalid_nonlinearity(self):
        """Test initialization with invalid nonlinearity."""
        with pytest.raises(ValueError, match="Invalid nonlinearity"):
            KaimingInitializer(nonlinearity='invalid')
    
    def test_initialize_fan_in(self, tensor_2d: torch.Tensor):
        """Test initialization with fan_in mode."""
        init = KaimingInitializer(mode='fan_in')
        init.initialize(tensor_2d)
        
        fan_in = tensor_2d.size(1)
        std = (2.0 / fan_in) ** 0.5  # gain=sqrt(2) for ReLU
        
        assert abs(tensor_2d.std() - std) < 0.1
    
    def test_initialize_fan_out(self, tensor_2d: torch.Tensor):
        """Test initialization with fan_out mode."""
        init = KaimingInitializer(mode='fan_out', distribution='uniform')
        init.initialize(tensor_2d)
        
        fan_out = tensor_2d.size(0)
        std = (2.0 / fan_out) ** 0.5
        bound = std * 3 ** 0.5
        
        assert tensor_2d.min() >= -bound
        assert tensor_2d.max() <= bound

# Uniform Initializer tests
class TestUniformInitializer:
    """Tests for UniformInitializer."""
    
    def test_invalid_bounds(self):
        """Test initialization with invalid bounds."""
        with pytest.raises(ValueError, match="must be greater than lower bound"):
            UniformInitializer(a=1.0, b=0.0)
    
    def test_initialize(self, tensor_2d: torch.Tensor):
        """Test uniform initialization."""
        a, b = -0.5, 0.5
        init = UniformInitializer(a=a, b=b)
        init.initialize(tensor_2d)
        
        assert tensor_2d.min() >= a
        assert tensor_2d.max() <= b
        # Check uniform distribution properties
        assert abs(tensor_2d.mean()) < 0.1
        assert abs(tensor_2d.std() - (b - a) / 3.464) < 0.1  # std of uniform dist

# Normal Initializer tests
class TestNormalInitializer:
    """Tests for NormalInitializer."""
    
    def test_invalid_std(self):
        """Test initialization with invalid standard deviation."""
        with pytest.raises(ValueError, match="must be positive"):
            NormalInitializer(std=0.0)
    
    def test_initialize(self, tensor_2d: torch.Tensor):
        """Test normal initialization."""
        mean, std = 0.0, 0.01
        init = NormalInitializer(mean=mean, std=std)
        init.initialize(tensor_2d)
        
        assert abs(tensor_2d.mean() - mean) < 0.01
        assert abs(tensor_2d.std() - std) < 0.01

# Model initialization tests
class TestInitializeModel:
    """Tests for initialize_model function."""
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(TypeError, match="must be an instance of nn.Module"):
            initialize_model([1, 2, 3], 'xavier')
    
    def test_invalid_initializer_name(self, simple_model: nn.Module):
        """Test initialization with invalid initializer name."""
        with pytest.raises(ValueError, match="Unknown initializer"):
            initialize_model(simple_model, 'invalid')
    
    def test_initialize_with_string(self, simple_model: nn.Module):
        """Test initialization using string name."""
        model = initialize_model(simple_model, 'xavier', {'gain': 2.0})
        
        # Check that weights are initialized and biases are zero
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert param.std() > 0
            elif 'bias' in name:
                assert torch.all(param == 0)
    
    def test_initialize_with_instance(self, simple_model: nn.Module):
        """Test initialization using initializer instance."""
        init = KaimingInitializer(mode='fan_out')
        model = initialize_model(simple_model, init)
        
        # Check that weights are initialized and biases are zero
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert param.std() > 0
            elif 'bias' in name:
                assert torch.all(param == 0)

# Test pre-configured instances
def test_preconfigured_instances(tensor_2d: torch.Tensor):
    """Test all pre-configured initializer instances."""
    instances = [
        xavier_uniform,
        xavier_normal,
        kaiming_uniform,
        kaiming_normal,
        uniform,
        normal
    ]
    
    for init in instances:
        # Verify that each instance can initialize without errors
        tensor = tensor_2d.clone()
        init.initialize(tensor)
        assert not torch.all(tensor == 0)  # Verify initialization happened