# Distributed Training

This guide covers distributed training capabilities in PyTorch Basics Library.

## Basic Setup

```python
from pytorch_basics_library.distributed import setup_distributed, cleanup_distributed
import torch.multiprocessing as mp

def train_worker(rank, world_size):
    # Setup distributed
    setup_distributed(rank, world_size)
    
    # Your training code here
    model = YourModel()
    model = convert_to_distributed(model, rank)
    
    # Training loop...
    
    # Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True)
```

## Memory Management

```python
from pytorch_basics_library.memory import (
    get_memory_stats,
    optimize_memory_usage,
    enable_gradient_checkpointing
)

# Monitor memory usage
print(get_memory_stats())

# Optimize model memory usage
model = optimize_memory_usage(model)

# Enable gradient checkpointing for large models
model = enable_gradient_checkpointing(model)
```

## Checkpointing

```python
from pytorch_basics_library.serialization import save_checkpoint, load_checkpoint

# Save training state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
save_checkpoint(checkpoint, 'checkpoint.pth', is_best=is_best)

# Load training state
checkpoint = load_checkpoint('checkpoint.pth', model, optimizer)
start_epoch = checkpoint['epoch']
``` 