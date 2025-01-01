# Device Management

The device management module provides utilities for handling device selection, memory management, and device-related operations in PyTorch.

## Overview

The `DeviceManager` class is designed to:
- Automatically select the best available device (CUDA/CPU)
- Move tensors and models between devices
- Monitor and manage GPU memory
- Profile memory usage
- Inspect device properties

## Basic Usage

```python
from pytorch_basics_library.device_management import device_manager

# Get the best available device
device = device_manager.get_device()

# Move tensors to device
tensor = torch.randn(10, 10)
tensor = device_manager.to_device(tensor)

# Move models to device
model = YourModel()
model = device_manager.to_device(model)
```

## Memory Management

### Getting Memory Statistics

```python
# Get current memory usage
stats = device_manager.get_memory_stats()
print(f"Allocated memory: {stats['allocated']} GB")
print(f"Cached memory: {stats['cached']} GB")
print(f"Max allocated: {stats['max_allocated']} GB")
```

### Clearing Memory

```python
# Clear unused memory
device_manager.clear_memory()
```

## Device Properties

```python
# Get device properties
props = device_manager.get_device_properties()
print(f"Device name: {props['name']}")
print(f"Total memory: {props['total_memory']}")
print(f"Compute capability: {props['compute_capability']}")
```

## Memory Profiling

Profile memory usage of specific operations:

```python
def my_operation():
    x = torch.randn(1000, 1000, device=device_manager.device)
    y = x @ x.t()
    return y.sum()

# Profile the operation
stats, result = device_manager.profile_memory_usage(my_operation)
print(f"Memory used: {stats['memory_used']} GB")
print(f"Peak memory: {stats['peak_memory']} GB")
```

## API Reference

### `DeviceManager`

#### `get_device() -> torch.device`
Returns the best available device (CUDA if available, else CPU).

#### `to_device(obj: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]`
Moves a tensor or model to the current device.

#### `get_memory_stats() -> Dict[str, float]`
Returns current GPU memory statistics in GB.

#### `clear_memory() -> None`
Clears unused memory caches and runs garbage collection.

#### `get_device_properties() -> Dict[str, str]`
Returns properties of the current device.

#### `profile_memory_usage(func) -> Tuple[Dict[str, float], Any]`
Profiles memory usage of a function and returns memory statistics and function result.

## Best Practices

1. **Device Selection**
   - Let the DeviceManager handle device selection instead of hardcoding devices
   - Use the same device manager instance throughout your application

2. **Memory Management**
   - Clear memory when switching between large operations
   - Monitor memory usage during training
   - Profile memory-intensive operations

3. **Error Handling**
   - Handle cases where CUDA is not available
   - Check device compatibility when loading models/tensors

## Examples

Check out the [device management example](../examples/device_management_example.py) for more detailed usage examples. 