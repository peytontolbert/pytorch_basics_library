"""
Device Management Examples
========================

This script demonstrates the usage of the DeviceManager class for handling
device-related operations in PyTorch.
"""

import torch
import torch.nn as nn
from pytorch_basics_library.device_management import device_manager

def basic_device_operations():
    """Demonstrate basic device operations."""
    print("\n1. Basic Device Operations")
    print("--------------------------")
    
    # Get current device
    device = device_manager.get_device()
    print(f"Current device: {device}")
    
    # Create a sample tensor and move it to device
    tensor = torch.randn(5, 3)
    tensor = device_manager.to_device(tensor)
    print(f"Tensor device: {tensor.device}")

def model_device_management():
    """Demonstrate model device management."""
    print("\n2. Moving Models to Device")
    print("-------------------------")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    model = device_manager.to_device(model)
    print(f"Model device: {next(model.parameters()).device}")

def memory_management():
    """Demonstrate memory management features."""
    print("\n3. Memory Management")
    print("-------------------")
    
    # Get memory statistics
    memory_stats = device_manager.get_memory_stats()
    print("Memory statistics:")
    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f} GB")
        else:
            print(f"{key}: {value}")
    
    # Clear memory
    device_manager.clear_memory()
    print("\nMemory cleared")

def device_properties():
    """Display device properties."""
    print("\n4. Device Properties")
    print("-------------------")
    
    properties = device_manager.get_device_properties()
    print("Device properties:")
    for key, value in properties.items():
        print(f"{key}: {value}")

def memory_profiling():
    """Demonstrate memory profiling."""
    print("\n5. Memory Profiling")
    print("------------------")
    
    def matrix_operations():
        x = torch.randn(1000, 1000, device=device_manager.device)
        y = x @ x.t()
        return y.sum()
    
    stats, result = device_manager.profile_memory_usage(matrix_operations)
    print("Memory profiling results:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f} GB")
        else:
            print(f"{key}: {value}")
    print(f"Operation result: {result}")

def main():
    """Run all examples."""
    print("Device Management Examples")
    print("=========================")
    
    basic_device_operations()
    model_device_management()
    memory_management()
    device_properties()
    memory_profiling()

if __name__ == "__main__":
    main() 