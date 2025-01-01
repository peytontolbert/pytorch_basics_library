# Advanced Usage Tutorial

This tutorial covers advanced features and best practices when using PyTorch Basics Library.

## Custom Model Integration

```python
import torch.nn as nn
from pytorch_basics_library import init_weights, to_device, get_device

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Initialize weights
        init_weights(self.features, method='kaiming')
    
    def forward(self, x):
        return self.features(x)

# Usage
device = get_device()
model = CustomModel()
model = to_device(model, device)
```

## Batch Processing Utilities

```python
from pytorch_basics_library import split_batch, normalize

def process_large_dataset(large_tensor, batch_size=32):
    # Split into batches
    batches = split_batch(large_tensor, batch_size)
    
    processed_batches = []
    for batch in batches:
        # Normalize each batch
        normalized_batch = normalize(batch)
        # Process batch
        processed = model(normalized_batch)
        processed_batches.append(processed)
    
    # Combine results
    return torch.cat(processed_batches, dim=0)
```

## Custom Training Loops

```python
from pytorch_basics_library import (
    get_device,
    to_device,
    compute_loss,
    clip_grad_norm_,
    accumulate_metric
)

class TrainingManager:
    def __init__(self, model, optimizer, scheduler):
        self.device = get_device()
        self.model = to_device(model, self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_step(self, inputs, targets):
        inputs = to_device(inputs, self.device)
        targets = to_device(targets, self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = compute_loss(outputs, targets)
        loss.backward()
        
        clip_grad_norm_(self.model, max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
```

## Performance Optimization

```python
import torch
from pytorch_basics_library import to_float32, concatenate

def optimize_memory_usage(tensor_list):
    # Convert to float32 for better numerical stability
    converted = [to_float32(t) for t in tensor_list]
    
    # Efficient concatenation
    batch = concatenate(converted, dim=0)
    
    # Use torch.cuda.empty_cache() if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return batch
``` 