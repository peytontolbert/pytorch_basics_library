# Tensor Operations

The tensor operations module provides a comprehensive set of utilities for tensor manipulation and mathematical operations in PyTorch.

## Overview

The `TensorOps` class provides:
- Tensor creation from various data types
- Common mathematical operations
- Shape manipulation utilities
- Type conversion helpers
- Advanced tensor operations

## Basic Usage

```python
from pytorch_basics_library.tensor_utils import tensor_ops

# Create tensors from different sources
list_tensor = tensor_ops.create_tensor([1, 2, 3, 4])
numpy_tensor = tensor_ops.create_tensor(numpy_array)
float_tensor = tensor_ops.create_tensor(data, dtype=torch.float32)
```

## Core Operations

### Batch Operations

```python
# Batch dot product
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[2., 3.], [4., 5.]])
result = tensor_ops.batch_dot(a, b)

# Normalize batches
tensor = torch.tensor([[3., 4.], [6., 8.]])
normalized = tensor_ops.normalize(tensor)
```

### Shape Manipulation

```python
# Reshape maintaining batch dimension
tensor = torch.randn(6, 4)
reshaped = tensor_ops.reshape_batch(tensor, batch_size=2, shape=[2, 6])

# Concatenate tensors
t1 = torch.ones(2, 3)
t2 = torch.zeros(2, 3)
concatenated = tensor_ops.concatenate([t1, t2])

# Split into batches
splits = tensor_ops.split_batch(tensor, batch_size=2)
```

### Advanced Operations

```python
# Gather values along dimension
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
indices = torch.tensor([[0], [2]])
gathered = tensor_ops.gather_along_dim(tensor, indices, dim=1)

# Masked fill
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
mask = torch.tensor([[True, False, True], [False, True, False]])
filled = tensor_ops.masked_fill(tensor, mask, value=0)
```

## API Reference

### `TensorOps`

#### `create_tensor(data: Union[List, np.ndarray, torch.Tensor], dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`
Creates a tensor from various input types.

#### `batch_dot(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor`
Performs batch-wise dot product.

#### `normalize(tensor: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor`
Normalizes the tensor along the specified dimension.

#### `reshape_batch(tensor: torch.Tensor, batch_size: int, shape: Union[Tuple, List]) -> torch.Tensor`
Reshapes tensor while preserving batch dimension.

#### `concatenate(tensor_list: List[torch.Tensor], dim: int = 0) -> torch.Tensor`
Concatenates a list of tensors along the specified dimension.

#### `split_batch(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]`
Splits a tensor into batches.

#### `type_convert(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor`
Safely converts tensor to specified dtype.

#### `gather_along_dim(tensor: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor`
Gathers values along a dimension using indices.

#### `masked_fill(tensor: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor`
Fills elements of tensor with value where mask is True.

## Best Practices

1. **Type Safety**
   - Always specify dtype when creating tensors for critical operations
   - Use type_convert for safe type conversions
   - Handle type mismatches gracefully

2. **Shape Management**
   - Verify tensor shapes before operations
   - Use reshape_batch for maintaining batch dimensions
   - Handle dynamic batch sizes appropriately

3. **Memory Efficiency**
   - Use in-place operations when possible
   - Be mindful of creating unnecessary copies
   - Clean up large temporary tensors

4. **Numerical Stability**
   - Use eps parameter in normalize for stability
   - Handle edge cases in mathematical operations
   - Validate input ranges for critical operations

## Examples

Check out the [tensor operations example](../examples/tensor_ops_example.py) for more detailed usage examples. 