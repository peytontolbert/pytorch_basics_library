"""
Tensor Operations Examples
========================

This script demonstrates the usage of the TensorOps class for handling
tensor operations in PyTorch.
"""

import torch
import numpy as np
from pytorch_basics_library.tensor_utils import tensor_ops

def tensor_creation():
    """Demonstrate tensor creation from different sources."""
    print("\n1. Tensor Creation")
    print("-----------------")
    
    # Create from list
    list_data = [1, 2, 3, 4]
    list_tensor = tensor_ops.create_tensor(list_data)
    print(f"From list: {list_tensor}")
    
    # Create from numpy array
    numpy_data = np.array([[1, 2], [3, 4]])
    numpy_tensor = tensor_ops.create_tensor(numpy_data)
    print(f"From numpy: {numpy_tensor}")
    
    # Create with specific dtype
    float_tensor = tensor_ops.create_tensor(list_data, dtype=torch.float32)
    print(f"With dtype: {float_tensor} (dtype: {float_tensor.dtype})")

def basic_operations():
    """Demonstrate basic tensor operations."""
    print("\n2. Basic Operations")
    print("------------------")
    
    # Batch dot product
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[2., 3.], [4., 5.]])
    dot_product = tensor_ops.batch_dot(a, b)
    print(f"Batch dot product: {dot_product}")
    
    # Normalization
    tensor = torch.tensor([[3., 4.], [6., 8.]])
    normalized = tensor_ops.normalize(tensor)
    print(f"Normalized tensor: {normalized}")
    print(f"Norms: {torch.norm(normalized, dim=1)}")  # Should be close to 1

def shape_manipulation():
    """Demonstrate shape manipulation operations."""
    print("\n3. Shape Manipulation")
    print("--------------------")
    
    # Reshape batch
    tensor = torch.randn(6, 4)
    print(f"Original shape: {tensor.shape}")
    reshaped = tensor_ops.reshape_batch(tensor, batch_size=2, shape=[2, 6])
    print(f"Reshaped: {reshaped.shape}")
    
    # Concatenation
    t1 = torch.ones(2, 3)
    t2 = torch.zeros(2, 3)
    concatenated = tensor_ops.concatenate([t1, t2])
    print(f"Concatenated shape: {concatenated.shape}")
    print(f"Concatenated tensor:\n{concatenated}")
    
    # Splitting
    splits = tensor_ops.split_batch(concatenated, batch_size=2)
    print(f"Split into {len(splits)} tensors of shape {splits[0].shape}")

def advanced_operations():
    """Demonstrate advanced tensor operations."""
    print("\n4. Advanced Operations")
    print("--------------------")
    
    # Gather along dimension
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    indices = torch.tensor([[0], [2]])
    gathered = tensor_ops.gather_along_dim(tensor, indices, dim=1)
    print(f"Gathered values:\n{gathered}")
    
    # Masked fill
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    filled = tensor_ops.masked_fill(tensor, mask, value=0)
    print(f"Masked fill result:\n{filled}")

def type_conversion():
    """Demonstrate type conversion operations."""
    print("\n5. Type Conversion")
    print("-----------------")
    
    # Integer to float conversion
    int_tensor = torch.tensor([1, 2, 3])
    float_tensor = tensor_ops.type_convert(int_tensor, torch.float32)
    print(f"Original dtype: {int_tensor.dtype}")
    print(f"Converted dtype: {float_tensor.dtype}")

def main():
    """Run all examples."""
    print("Tensor Operations Examples")
    print("=========================")
    
    tensor_creation()
    basic_operations()
    shape_manipulation()
    advanced_operations()
    type_conversion()

if __name__ == "__main__":
    main() 