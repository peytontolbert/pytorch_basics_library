"""
PyTorch Basics Library
=====================

A lightweight, high-performance library providing essential utilities for PyTorch development.
"""

from .device_management import device_manager
from .tensor_utils import tensor_ops
from .initializers import (
    initialize_model,
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    uniform,
    normal
)

__version__ = '0.1.0'

__all__ = [
    'device_manager',
    'tensor_ops',
    'initialize_model',
    'xavier_uniform',
    'xavier_normal',
    'kaiming_uniform',
    'kaiming_normal',
    'uniform',
    'normal'
] 