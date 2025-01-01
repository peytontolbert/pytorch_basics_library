"""Device Management Module.

This module provides utilities for handling device selection, memory management,
and device-related operations in PyTorch, including multi-GPU support and
advanced profiling capabilities.

Example:
    >>> from pytorch_basics_library.device_management import device_manager
    >>> device = device_manager.get_device()
    >>> model = device_manager.to_device(model, device_id=0)  # Specific GPU
    >>> stats = device_manager.get_memory_stats()
"""

import gc
from typing import Union, Dict, TypeVar, Tuple, Any, Callable, Optional, List
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.profiler as profiler
from torch.nn.parallel import DistributedDataParallel as DDP

# Type variables for generic types
T = TypeVar('T', torch.Tensor, nn.Module)
FuncType = Callable[[], Any]

class DeviceManager:
    """A class to handle device management and memory utilities in PyTorch.
    
    This class provides a centralized way to manage device selection, memory
    management, and device-related operations. It supports multi-GPU setups,
    distributed training, and advanced profiling.

    Attributes:
        device (torch.device): The current device being used.
        world_size (int): Number of processes in distributed training.
        rank (int): Current process rank in distributed training.
        is_distributed (bool): Whether distributed training is enabled.
    """
    
    def __init__(self) -> None:
        """Initialize the DeviceManager with the best available device."""
        self.device = self.get_device()
        self.world_size = 1
        self.rank = 0
        self.is_distributed = False
        self._profiler = None
        
    def setup_distributed(
        self,
        rank: int,
        world_size: int,
        backend: str = 'nccl'
    ) -> None:
        """Setup distributed training environment.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: Distributed backend ('nccl' or 'gloo')
            
        Example:
            >>> # In each process
            >>> device_manager.setup_distributed(rank, world_size)
            >>> model = DDP(model)
        """
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
            
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        self.world_size = world_size
        self.rank = rank
        self.is_distributed = True
        
        # Set device for this process
        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed training environment."""
        if self.is_distributed:
            dist.destroy_process_group()
            self.is_distributed = False
            self.world_size = 1
            self.rank = 0
    
    @staticmethod
    def get_available_devices() -> List[torch.device]:
        """Get list of all available devices.
        
        Returns:
            List of available devices (CPU and GPUs)
            
        Example:
            >>> devices = device_manager.get_available_devices()
            >>> print(f"Available devices: {devices}")
        """
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.extend([
                torch.device(f'cuda:{i}')
                for i in range(torch.cuda.device_count())
            ])
        return devices
    
    def get_device(self, device_id: Optional[int] = None) -> torch.device:
        """Get the best available device or a specific GPU.
        
        Args:
            device_id: Specific GPU ID to use (None for auto-select)
            
        Returns:
            Selected device
            
        Example:
            >>> device = device_manager.get_device(0)  # Use first GPU
            >>> device = device_manager.get_device()   # Auto-select
        """
        if device_id is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {device_id} not found")
            return torch.device(f'cuda:{device_id}')
            
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def to_device(
        self,
        obj: T,
        device_id: Optional[int] = None,
        distributed: bool = False
    ) -> T:
        """Move a tensor or model to a device and optionally wrap in DDP.
        
        Args:
            obj: PyTorch tensor or model to move
            device_id: Specific GPU ID (None for current device)
            distributed: Whether to wrap model in DistributedDataParallel
            
        Returns:
            The object moved to device (and wrapped in DDP if specified)
            
        Example:
            >>> model = device_manager.to_device(model, device_id=0)
            >>> model = device_manager.to_device(model, distributed=True)
        """
        if not isinstance(obj, (torch.Tensor, nn.Module)):
            raise TypeError("Object must be a torch.Tensor or torch.nn.Module")
            
        device = self.get_device(device_id) if device_id is not None else self.device
        obj = obj.to(device)
        
        if distributed and isinstance(obj, nn.Module):
            if not self.is_distributed:
                raise RuntimeError("Must call setup_distributed first")
            obj = DDP(obj, device_ids=[self.rank])
            
        return obj
    
    def synchronize(self) -> None:
        """Synchronize all CUDA devices.
        
        Example:
            >>> # After computation on multiple GPUs
            >>> device_manager.synchronize()
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def get_memory_stats(self, device_id: Optional[int] = None) -> Dict[str, Union[float, str]]:
        """Get current GPU memory statistics.
        
        Args:
            device_id: Specific GPU ID (None for current device)
            
        Returns:
            Dictionary containing memory statistics in GB
            
        Example:
            >>> stats = device_manager.get_memory_stats(0)
            >>> print(f"GPU 0 memory used: {stats['allocated']:.2f} GB")
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        device = device_id if device_id is not None else self.device.index
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1e9,
            "cached": torch.cuda.memory_reserved(device) / 1e9,
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1e9
        }
    
    def start_profiler(
        self,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        record_shapes: bool = True
    ) -> None:
        """Start CUDA profiler for performance analysis.
        
        Args:
            wait: Number of iterations to wait before profiling
            warmup: Number of iterations for warmup
            active: Number of iterations to profile
            record_shapes: Whether to record tensor shapes
            
        Example:
            >>> device_manager.start_profiler()
            >>> # Run operations to profile
            >>> device_manager.stop_profiler()
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiler requires CUDA")
            
        self._profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=wait,
                warmup=warmup,
                active=active
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=record_shapes,
            with_stack=True
        )
        self._profiler.start()
    
    def stop_profiler(self) -> None:
        """Stop CUDA profiler and save results.
        
        Example:
            >>> device_manager.start_profiler()
            >>> # Run operations to profile
            >>> device_manager.stop_profiler()
        """
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None
    
    def profile_section(
        self,
        name: str,
        func: FuncType,
        record_shapes: bool = True
    ) -> Tuple[Any, Dict[str, float]]:
        """Profile a specific section of code.
        
        Args:
            name: Name of the section to profile
            func: Function to profile
            record_shapes: Whether to record tensor shapes
            
        Returns:
            Tuple of (function result, profiling stats)
            
        Example:
            >>> def heavy_computation():
            ...     return torch.randn(1000, 1000) @ torch.randn(1000, 1000)
            >>> result, stats = device_manager.profile_section(
            ...     "matrix_multiply",
            ...     heavy_computation
            ... )
        """
        if not torch.cuda.is_available():
            result = func()
            return result, {"error": "CUDA profiler not available"}
            
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=record_shapes
        ) as prof:
            result = func()
            
        stats = {
            "cpu_time": prof.self_cpu_time_total / 1000,  # ms
            "cuda_time": prof.self_cuda_time_total / 1000 if hasattr(prof, 'self_cuda_time_total') else 0,  # ms
            "memory_allocated": torch.cuda.max_memory_allocated() / 1e9  # GB
        }
        
        return result, stats
    
    def clear_memory(self, device_id: Optional[int] = None) -> None:
        """Clear unused memory caches and run garbage collection.
        
        Args:
            device_id: Specific GPU ID (None for all devices)
            
        Example:
            >>> # After some memory-intensive operations
            >>> device_manager.clear_memory()
        """
        gc.collect()
        if torch.cuda.is_available():
            if device_id is not None:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device_id)
            else:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
    
    def get_device_properties(self, device_id: Optional[int] = None) -> Dict[str, str]:
        """Get properties of a specific device.
        
        Args:
            device_id: Specific GPU ID (None for current device)
            
        Returns:
            Dictionary containing device properties
            
        Example:
            >>> props = device_manager.get_device_properties(0)
            >>> print(f"GPU 0 name: {props['name']}")
        """
        if torch.cuda.is_available():
            device = device_id if device_id is not None else self.device.index
            props = torch.cuda.get_device_properties(device)
            return {
                "name": props.name,
                "total_memory": f"{props.total_memory / 1e9:.2f} GB",
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": str(props.multi_processor_count),
                "max_threads_per_block": str(props.max_threads_per_block),
                "max_threads_per_multiprocessor": str(props.max_threads_per_multiprocessor),
                "warp_size": str(props.warp_size)
            }
        return {"device": "cpu"}

# Create a global instance
device_manager = DeviceManager()