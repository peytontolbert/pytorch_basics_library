# PyTorch Basics Library

A lightweight, high-performance library providing essential utilities for PyTorch development, focusing on device management, tensor operations, and weight initialization.

## Features

### Device Management
- Automatic device selection (CPU/CUDA)
- Memory management and profiling
- Device property inspection
- Easy model/tensor device movement

```python
from pytorch_basics_library.device_management import device_manager

# Automatically select best device
device = device_manager.get_device()

# Move model to device
model = device_manager.to_device(model)

# Monitor memory usage
stats = device_manager.get_memory_stats()
```

### Tensor Operations
- Efficient tensor creation and manipulation
- Common operations (batch_dot, normalize)
- Shape manipulation utilities
- Type conversion helpers
- Advanced operations (gather, masked_fill)

```python
from pytorch_basics_library.tensor_utils import tensor_ops

# Create tensor from various sources
tensor = tensor_ops.create_tensor([1, 2, 3], dtype=torch.float32)

# Perform operations
normalized = tensor_ops.normalize(tensor)
```

### Weight Initialization
- Xavier/Glorot initialization
- Kaiming/He initialization
- Basic distribution initializers
- Custom initialization support

```python
from pytorch_basics_library.initializers import initialize_model, kaiming_normal

# Initialize model weights
model = initialize_model(model, 'xavier')

# Use pre-configured initializers
model = initialize_model(model, kaiming_normal)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/peytontolbert/pytorch_basics_library.git

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from pytorch_basics_library import device_manager, tensor_ops, initialize_model

# Create and move model to best available device
model = YourModel()
model = device_manager.to_device(model)

# Initialize weights
model = initialize_model(model, 'kaiming')

# Create and manipulate tensors
data = torch.randn(32, 10)
data = tensor_ops.normalize(data)
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [Device Management](docs/device_management.md)
- [Tensor Operations](docs/tensor_operations.md)
- [Weight Initialization](docs/weight_initialization.md)
- [Examples](examples/README.md)

## Examples

Check out our [examples](examples/) directory for detailed usage examples:
- Device management examples
- Tensor operations examples
- Weight initialization examples

## Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_device_management.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.21+