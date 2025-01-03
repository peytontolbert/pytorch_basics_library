"""
PyTorch Basics Library
=====================

A lightweight, high-performance library providing essential utilities for PyTorch development,
focusing on device management, tensor operations, and weight initialization.

Modules:
    - device_management: Tools for device selection and memory management
    - tensor_utils: Efficient tensor operations and manipulations
    - initializers: Weight initialization methods for neural networks

Example:
    >>> from pytorch_basics_library import device_manager, tensor_ops
    >>> from pytorch_basics_library.initializers import initialize_model
    >>> 
    >>> # Get the best available device
    >>> device = device_manager.get_device()
    >>> 
    >>> # Initialize a model
    >>> model = initialize_model(your_model, 'kaiming')
"""

from typing import List, Union, Callable, Optional, Any

from .device_management import device_manager
from .tensor_utils import tensor_ops
from .initializers import (
    # Classes
    Initializer, XavierInitializer, KaimingInitializer,
    UniformInitializer, NormalInitializer, OrthogonalInitializer,
    
    # Functions
    xavier_uniform_, xavier_normal_,
    kaiming_uniform_, kaiming_normal_,
    orthogonal_, uniform_, normal_,
    
    # Default instances
    xavier_uniform, xavier_normal,
    kaiming_uniform, kaiming_normal,
    uniform, normal, orthogonal,
    
    # Utility functions
    initialize_model
)

__version__ = '0.1.0'
__author__ = 'Peyton Tolbert'
__email__ = 'email@peytontolbert.com'

__all__: List[str] = [
    # Classes
    'Initializer', 'XavierInitializer', 'KaimingInitializer',
    'UniformInitializer', 'NormalInitializer', 'OrthogonalInitializer',
    
    # Functions
    'xavier_uniform_', 'xavier_normal_',
    'kaiming_uniform_', 'kaiming_normal_',
    'orthogonal_', 'uniform_', 'normal_',
    
    # Default instances
    'xavier_uniform', 'xavier_normal',
    'kaiming_uniform', 'kaiming_normal',
    'uniform', 'normal', 'orthogonal',
    
    # Utility functions
    'initialize_model',
    
    # Managers
    'device_manager',
    'tensor_ops'
] 