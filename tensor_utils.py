"""Tensor Operations Module.

This module provides a comprehensive set of utilities for tensor manipulation
and mathematical operations in PyTorch.

Example:
    >>> from pytorch_basics_library.tensor_utils import tensor_ops
    >>> data = [1, 2, 3, 4]
    >>> tensor = tensor_ops.create_tensor(data, dtype=torch.float32)
    >>> normalized = tensor_ops.normalize(tensor)
"""

from typing import List, Union, Tuple, Optional, Sequence, TypeVar, overload
import numpy as np
import torch

# Type variables
T = TypeVar('T', bound=torch.Tensor)
TensorLike = Union[List, np.ndarray, torch.Tensor]
Shape = Union[Tuple[int, ...], List[int]]

class TensorOps:
    """A class containing tensor manipulation utilities.
    
    This class provides a comprehensive set of operations for creating,
    manipulating, and transforming PyTorch tensors. It includes utilities
    for common operations like normalization, shape manipulation, and
    type conversion.
    """
    
    @staticmethod
    def create_tensor(
        data: TensorLike,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Create a tensor from various input types.
        
        Args:
            data: Input data (list, numpy array, or tensor)
            dtype: Desired tensor dtype
            device: Device to place tensor on
            
        Returns:
            PyTorch tensor
            
        Raises:
            TypeError: If data is not a list, numpy array, or tensor
            ValueError: If data is empty

        Example:
            >>> data = [1, 2, 3]
            >>> tensor = tensor_ops.create_tensor(data, dtype=torch.float32)
            >>> print(tensor.dtype)  # torch.float32
        """
        if not isinstance(data, (list, np.ndarray, torch.Tensor)):
            raise TypeError("Data must be a list, numpy array, or tensor")
        if isinstance(data, (list, np.ndarray)) and len(data) == 0:
            raise ValueError("Data cannot be empty")
            
        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)
            
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
            
        return tensor
    
    @staticmethod
    def batch_dot(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        """Perform batch-wise dot product.
        
        Args:
            tensor_a: Tensor of shape (batch_size, n)
            tensor_b: Tensor of shape (batch_size, n)
            
        Returns:
            Tensor of shape (batch_size,)
            
        Raises:
            ValueError: If tensor shapes don't match or aren't 2D

        Example:
            >>> a = torch.tensor([[1., 2.], [3., 4.]])
            >>> b = torch.tensor([[2., 3.], [4., 5.]])
            >>> result = tensor_ops.batch_dot(a, b)
            >>> print(result)  # tensor([ 8., 32.])
        """
        if tensor_a.dim() != 2 or tensor_b.dim() != 2:
            raise ValueError("Tensors must be 2D (batch_size, n)")
        if tensor_a.shape != tensor_b.shape:
            raise ValueError("Tensor shapes must match")
            
        return torch.sum(tensor_a * tensor_b, dim=1)
    
    @staticmethod
    def normalize(
        tensor: torch.Tensor,
        dim: int = 1,
        eps: float = 1e-12
    ) -> torch.Tensor:
        """Normalize the tensor along the specified dimension.
        
        Args:
            tensor: Input tensor
            dim: Dimension to normalize
            eps: Small value to avoid division by zero
            
        Returns:
            Normalized tensor
            
        Raises:
            ValueError: If dimension is invalid

        Example:
            >>> tensor = torch.tensor([[3., 4.], [6., 8.]])
            >>> normalized = tensor_ops.normalize(tensor)
            >>> print(torch.norm(normalized, dim=1))  # tensor([1., 1.])
        """
        if dim >= tensor.dim():
            raise ValueError(f"Invalid dimension {dim} for tensor of dim {tensor.dim()}")
            
        return tensor / (tensor.norm(dim=dim, keepdim=True) + eps)
    
    @staticmethod
    def reshape_batch(
        tensor: torch.Tensor,
        batch_size: int,
        shape: Shape
    ) -> torch.Tensor:
        """Reshape tensor while preserving batch dimension.
        
        Args:
            tensor: Input tensor
            batch_size: Batch size
            shape: Desired shape (excluding batch dimension)
            
        Returns:
            Reshaped tensor
            
        Raises:
            ValueError: If tensor size doesn't match target shape

        Example:
            >>> tensor = torch.randn(6, 4)
            >>> reshaped = tensor_ops.reshape_batch(tensor, 2, [3, 4])
            >>> print(reshaped.shape)  # torch.Size([2, 3, 4])
        """
        total_size = batch_size * np.prod(shape)
        if tensor.numel() != total_size:
            raise ValueError(f"Tensor size {tensor.numel()} doesn't match target size {total_size}")
            
        return tensor.view(batch_size, *shape)
    
    @staticmethod
    def concatenate(
        tensor_list: List[torch.Tensor],
        dim: int = 0
    ) -> torch.Tensor:
        """Concatenate a list of tensors along the specified dimension.
        
        Args:
            tensor_list: List of tensors
            dim: Dimension to concatenate
            
        Returns:
            Concatenated tensor
            
        Raises:
            ValueError: If tensor_list is empty or tensors have incompatible shapes

        Example:
            >>> t1 = torch.ones(2, 3)
            >>> t2 = torch.zeros(2, 3)
            >>> result = tensor_ops.concatenate([t1, t2])
            >>> print(result.shape)  # torch.Size([4, 3])
        """
        if not tensor_list:
            raise ValueError("Tensor list cannot be empty")
            
        return torch.cat(tensor_list, dim=dim)
    
    @staticmethod
    def split_batch(
        tensor: torch.Tensor,
        batch_size: int
    ) -> List[torch.Tensor]:
        """Split a tensor into batches.
        
        Args:
            tensor: Input tensor
            batch_size: Size of each batch
            
        Returns:
            List of tensors
            
        Raises:
            ValueError: If batch_size is invalid

        Example:
            >>> tensor = torch.randn(6, 3)
            >>> splits = tensor_ops.split_batch(tensor, 2)
            >>> print(len(splits))  # 3
            >>> print(splits[0].shape)  # torch.Size([2, 3])
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if batch_size > tensor.size(0):
            raise ValueError("Batch size cannot be larger than tensor size")
            
        return torch.split(tensor, batch_size, dim=0)
    
    @staticmethod
    def type_convert(
        tensor: torch.Tensor,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Safely convert tensor to specified dtype.
        
        Args:
            tensor: Input tensor
            dtype: Target dtype
            
        Returns:
            Converted tensor

        Example:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> float_tensor = tensor_ops.type_convert(tensor, torch.float32)
            >>> print(float_tensor.dtype)  # torch.float32
        """
        return tensor.to(dtype)
    
    @staticmethod
    def gather_along_dim(
        tensor: torch.Tensor,
        indices: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        """Gather values along a dimension using indices.
        
        Args:
            tensor: Input tensor
            indices: Indices to gather
            dim: Dimension to gather along
            
        Returns:
            Gathered tensor
            
        Raises:
            ValueError: If dimension is invalid

        Example:
            >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> indices = torch.tensor([[0], [2]])
            >>> result = tensor_ops.gather_along_dim(tensor, indices, dim=1)
            >>> print(result)  # tensor([[1], [6]])
        """
        if dim >= tensor.dim():
            raise ValueError(f"Invalid dimension {dim} for tensor of dim {tensor.dim()}")
            
        return torch.gather(tensor, dim, indices)
    
    @staticmethod
    def masked_fill(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        value: float
    ) -> torch.Tensor:
        """Fill elements of tensor with value where mask is True.
        
        Args:
            tensor: Input tensor
            mask: Boolean mask
            value: Fill value
            
        Returns:
            Filled tensor
            
        Raises:
            ValueError: If tensor and mask shapes don't match

        Example:
            >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> mask = torch.tensor([[True, False, True], [False, True, False]])
            >>> result = tensor_ops.masked_fill(tensor, mask, 0)
            >>> print(result)  # tensor([[0, 2, 0], [4, 0, 6]])
        """
        if tensor.shape != mask.shape:
            raise ValueError("Tensor and mask shapes must match")
            
        return tensor.masked_fill(mask, value)

# Create a global instance
tensor_ops = TensorOps() 