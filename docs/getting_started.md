# Getting Started with PyTorch Basics Library

This guide will help you get started with the PyTorch Basics Library, covering installation and basic usage of core features.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pytorch_basics_library.git
cd pytorch_basics_library
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Device Management

The device manager handles device selection and memory management:

```python
from pytorch_basics_library.device_management import device_manager

# Get the best available device
device = device_manager.get_device()
print(f"Using device: {device}")

# Move tensors or models to device
tensor = torch.randn(10, 10)
tensor = device_manager.to_device(tensor)

# Check memory usage (if CUDA is available)
memory_stats = device_manager.get_memory_stats()
print(f"Memory usage: {memory_stats}")
```

### 2. Tensor Operations

The tensor operations module provides common tensor manipulations:

```python
from pytorch_basics_library.tensor_utils import tensor_ops

# Create a tensor
data = [1, 2, 3, 4]
tensor = tensor_ops.create_tensor(data, dtype=torch.float32)

# Normalize the tensor
normalized = tensor_ops.normalize(tensor)

# Perform batch operations
batch = torch.randn(32, 10)
splits = tensor_ops.split_batch(batch, batch_size=8)
```

### 3. Weight Initialization

Initialize model weights using various strategies:

```python
from pytorch_basics_library.initializers import initialize_model
import torch.nn as nn

# Create a model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

model = SimpleModel()

# Initialize with Xavier/Glorot
model = initialize_model(model, 'xavier')

# Initialize with Kaiming/He
model = initialize_model(model, 'kaiming', {'nonlinearity': 'relu'})
```

## Next Steps

- Check out the [examples](../examples/) directory for more detailed usage examples
- Read the specific documentation for each module:
  - [Device Management](device_management.md)
  - [Tensor Operations](tensor_operations.md)
  - [Weight Initialization](weight_initialization.md)
- Run the test suite to verify your installation:
  ```bash
  pytest
  ```