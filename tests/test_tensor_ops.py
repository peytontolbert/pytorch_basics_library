import pytest
import torch
import numpy as np
from ..tensor_utils import TensorOps

@pytest.fixture
def tensor_ops():
    return TensorOps()

def test_create_tensor_from_list(tensor_ops):
    data = [1, 2, 3, 4]
    tensor = tensor_ops.create_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == data

def test_create_tensor_from_numpy(tensor_ops):
    data = np.array([1, 2, 3, 4])
    tensor = tensor_ops.create_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    np.testing.assert_array_equal(tensor.numpy(), data)

def test_create_tensor_with_dtype(tensor_ops):
    data = [1, 2, 3, 4]
    tensor = tensor_ops.create_tensor(data, dtype=torch.float32)
    assert tensor.dtype == torch.float32

def test_batch_dot(tensor_ops):
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[2., 3.], [4., 5.]])
    result = tensor_ops.batch_dot(a, b)
    expected = torch.tensor([8., 32.])  # (1*2 + 2*3), (3*4 + 4*5)
    assert torch.allclose(result, expected)

def test_normalize(tensor_ops):
    tensor = torch.tensor([[3., 4.], [6., 8.]])
    normalized = tensor_ops.normalize(tensor)
    # Check that the norm of each row is approximately 1
    norms = torch.norm(normalized, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))

def test_reshape_batch(tensor_ops):
    tensor = torch.randn(6, 4)
    reshaped = tensor_ops.reshape_batch(tensor, batch_size=2, shape=[2, 6])
    assert reshaped.shape == (2, 2, 6)

def test_concatenate(tensor_ops):
    t1 = torch.ones(2, 3)
    t2 = torch.zeros(2, 3)
    result = tensor_ops.concatenate([t1, t2])
    assert result.shape == (4, 3)
    assert torch.equal(result[:2], t1)
    assert torch.equal(result[2:], t2)

def test_split_batch(tensor_ops):
    tensor = torch.randn(6, 3)
    splits = tensor_ops.split_batch(tensor, batch_size=2)
    assert len(splits) == 3
    assert all(s.shape == (2, 3) for s in splits)

def test_type_convert(tensor_ops):
    tensor = torch.tensor([1, 2, 3])
    float_tensor = tensor_ops.type_convert(tensor, torch.float32)
    assert float_tensor.dtype == torch.float32

def test_gather_along_dim(tensor_ops):
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    indices = torch.tensor([[0], [2]])
    result = tensor_ops.gather_along_dim(tensor, indices, dim=1)
    expected = torch.tensor([[1], [6]])
    assert torch.equal(result, expected)

def test_masked_fill(tensor_ops):
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    filled = tensor_ops.masked_fill(tensor, mask, value=0)
    expected = torch.tensor([[0, 2, 0], [4, 0, 6]])
    assert torch.equal(filled, expected)

def test_error_handling():
    ops = TensorOps()
    with pytest.raises(TypeError):
        ops.create_tensor({"invalid": "input"})
    
    with pytest.raises((RuntimeError, ValueError)):
        # Test incompatible shapes for batch_dot
        a = torch.randn(2, 3)
        b = torch.randn(2, 4)
        ops.batch_dot(a, b) 