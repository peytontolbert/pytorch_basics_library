"""Multi-GPU and Distributed Training Example.

This script demonstrates how to use the device management module for multi-GPU
and distributed training scenarios.
"""

import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from pytorch_basics_library.device_management import device_manager

class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def create_dummy_data(
    n_samples: int = 1000,
    input_size: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for training.
    
    Args:
        n_samples: Number of samples
        input_size: Input dimension
        
    Returns:
        Tuple of (inputs, targets)
    """
    X = torch.randn(n_samples, input_size)
    w = torch.randn(input_size, 1)
    y = X @ w + 0.1 * torch.randn(n_samples, 1)
    return X, y

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer
) -> float:
    """Train for one epoch.
    
    Args:
        model: Neural network
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = device_manager.to_device(data)
        target = device_manager.to_device(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

def train_distributed(
    rank: int,
    world_size: int,
    model: nn.Module,
    dataset: TensorDataset,
    epochs: int = 10
):
    """Training function for distributed setup.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        model: Neural network
        dataset: Training dataset
        epochs: Number of training epochs
    """
    # Setup distributed environment
    device_manager.setup_distributed(rank, world_size)
    
    # Create distributed sampler and dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=train_sampler
    )
    
    # Move model to device and wrap in DDP
    model = device_manager.to_device(model, distributed=True)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        loss = train_epoch(model, train_loader, criterion, optimizer)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    # Cleanup
    device_manager.cleanup_distributed()

def profile_training(
    model: nn.Module,
    train_loader: DataLoader,
    n_steps: int = 100
):
    """Profile the training process.
    
    Args:
        model: Neural network
        train_loader: DataLoader for training data
        n_steps: Number of steps to profile
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Start profiler
    device_manager.start_profiler(
        wait=10,
        warmup=10,
        active=20,
        record_shapes=True
    )
    
    # Training steps
    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i >= n_steps:
            break
            
        data = device_manager.to_device(data)
        target = device_manager.to_device(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        device_manager.synchronize()
    
    # Stop profiler
    device_manager.stop_profiler()

def main_single_gpu():
    """Example of single-GPU training with profiling."""
    print("\nRunning single-GPU example...")
    
    # Create model and data
    model = SimpleModel()
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Move model to GPU
    model = device_manager.to_device(model, device_id=0)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Profile memory usage during training
    def training_step():
        data, target = next(iter(train_loader))
        data = device_manager.to_device(data)
        target = device_manager.to_device(target)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        return loss
    
    print("\nProfiling single training step:")
    result, stats = device_manager.profile_section(
        "training_step",
        training_step
    )
    print(f"CPU Time: {stats['cpu_time']:.2f} ms")
    print(f"GPU Time: {stats['cuda_time']:.2f} ms")
    print(f"Memory Used: {stats['memory_allocated']:.2f} GB")
    
    # Train for a few steps with full profiling
    print("\nProfiling full training:")
    profile_training(model, train_loader, n_steps=50)

def main_multi_gpu():
    """Example of multi-GPU distributed training."""
    print("\nRunning multi-GPU example...")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping multi-GPU example.")
        return
    
    if torch.cuda.device_count() < 2:
        print("Less than 2 GPUs available. Skipping multi-GPU example.")
        return
    
    # Create model and data
    model = SimpleModel()
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    
    # Number of processes = number of GPUs
    world_size = torch.cuda.device_count()
    
    # Spawn processes
    mp.spawn(
        train_distributed,
        args=(world_size, model, dataset),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = device_manager.get_device_properties(i)
            print(f"\nGPU {i}: {props['name']}")
            print(f"Memory: {props['total_memory']}")
            print(f"Compute Capability: {props['compute_capability']}")
    else:
        print("No GPUs available, using CPU")
    
    try:
        main_single_gpu()
        main_multi_gpu()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        if torch.cuda.is_available():
            device_manager.cleanup_distributed()  # Cleanup in case of interruption 