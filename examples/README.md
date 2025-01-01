# PyTorch Basics Library Examples

This directory contains example scripts demonstrating the usage of various components of the PyTorch Basics Library.

## Available Examples

### 1. Device Management (`device_management_example.py`)
Demonstrates the usage of the `DeviceManager` class, including:
- Basic device operations
- Moving models to different devices
- Memory management
- Device properties inspection
- Memory profiling

### 2. Tensor Operations (`tensor_ops_example.py`)
Shows how to use the `TensorOps` class for:
- Tensor creation from different sources
- Basic tensor operations
- Shape manipulation
- Advanced operations (gather, masked fill)
- Type conversion

## Running the Examples

1. Make sure you have installed the library and its dependencies:
```bash
pip install -r ../requirements.txt
```

2. Run an example script:
```bash
# For device management examples
python device_management_example.py

# For tensor operations examples
python tensor_ops_example.py
```

## Expected Output

Each example script provides detailed output showing the results of various operations. The output is formatted to be easily readable and includes explanations of what each operation does.

## Notes

- Some examples require CUDA-capable GPU to demonstrate all features
- Memory management examples might show different results based on your hardware
- All examples include error handling and informative messages 