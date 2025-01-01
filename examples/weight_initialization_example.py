"""
Weight Initialization Examples
============================

This script demonstrates the usage of various weight initialization strategies
available in the PyTorch Basics Library.
"""

import torch
import torch.nn as nn
from pytorch_basics_library.initializers import (
    xavier_uniform, xavier_normal,
    kaiming_uniform, kaiming_normal,
    uniform, normal,
    initialize_model
)

def create_sample_model():
    """Create a sample model for demonstration."""
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.features(x)
    
    return SampleModel()

def demonstrate_xavier_initialization():
    """Demonstrate Xavier/Glorot initialization."""
    print("\n1. Xavier/Glorot Initialization")
    print("------------------------------")
    
    model = create_sample_model()
    
    # Xavier uniform
    print("\nXavier Uniform:")
    initialized_model = initialize_model(model, xavier_uniform)
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")
    
    # Xavier normal
    print("\nXavier Normal:")
    initialized_model = initialize_model(model, xavier_normal)
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")

def demonstrate_kaiming_initialization():
    """Demonstrate Kaiming/He initialization."""
    print("\n2. Kaiming/He Initialization")
    print("---------------------------")
    
    model = create_sample_model()
    
    # Kaiming uniform
    print("\nKaiming Uniform:")
    initialized_model = initialize_model(model, kaiming_uniform)
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")
    
    # Kaiming normal
    print("\nKaiming Normal:")
    initialized_model = initialize_model(model, kaiming_normal)
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")

def demonstrate_basic_distributions():
    """Demonstrate basic distribution initializations."""
    print("\n3. Basic Distribution Initialization")
    print("----------------------------------")
    
    model = create_sample_model()
    
    # Uniform distribution
    print("\nUniform Distribution:")
    initialized_model = initialize_model(model, uniform)
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Min:  {param.min().item():.4f}")
            print(f"  Max:  {param.max().item():.4f}")
            print(f"  Mean: {param.mean().item():.4f}")
    
    # Normal distribution
    print("\nNormal Distribution:")
    initialized_model = initialize_model(model, normal)
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")

def demonstrate_custom_initialization():
    """Demonstrate custom initialization configurations."""
    print("\n4. Custom Initialization")
    print("----------------------")
    
    model = create_sample_model()
    
    # Custom Xavier initialization
    print("\nCustom Xavier (gain=2.0):")
    initialized_model = initialize_model(
        model,
        'xavier',
        {'gain': 2.0, 'distribution': 'uniform'}
    )
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")
    
    # Custom Kaiming initialization
    print("\nCustom Kaiming (mode='fan_out'):")
    initialized_model = initialize_model(
        model,
        'kaiming',
        {'mode': 'fan_out', 'nonlinearity': 'relu'}
    )
    for name, param in initialized_model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")

def main():
    """Run all examples."""
    print("Weight Initialization Examples")
    print("=============================")
    
    demonstrate_xavier_initialization()
    demonstrate_kaiming_initialization()
    demonstrate_basic_distributions()
    demonstrate_custom_initialization()

if __name__ == "__main__":
    main() 