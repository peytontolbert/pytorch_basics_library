# Training Utilities Tutorial

This guide covers the training-related utilities provided by PyTorch Basics Library.

## Learning Rate Scheduling

```python
import torch.optim as optim
from pytorch_basics_library import get_scheduler

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create learning rate scheduler
scheduler = get_scheduler(
    optimizer,
    'steplr',
    step_size=10,
    gamma=0.1
)

# Usage in training loop
for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()
```

## Gradient Clipping

```python
from pytorch_basics_library import clip_grad_norm_

def training_step(model, optimizer, inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Clip gradients to prevent explosion
    clip_grad_norm_(model, max_norm=1.0)
    
    optimizer.step()
```

## Metric Tracking

```python
from pytorch_basics_library import accumulate_metric

# Initialize metric tracker
metric_tracker = {
    'loss': [],
    'accuracy': []
}

# During training
loss = compute_loss(outputs, targets)
accuracy = compute_accuracy(outputs, targets)

# Accumulate metrics
accumulate_metric(metric_tracker, 'loss', loss.item())
accumulate_metric(metric_tracker, 'accuracy', accuracy)
```

## Complete Training Example

```python
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    metric_tracker = {'loss': [], 'accuracy': []}
    
    for inputs, targets in dataloader:
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(outputs, targets)
        loss.backward()
        
        clip_grad_norm_(model, max_norm=1.0)
        optimizer.step()
        
        accumulate_metric(metric_tracker, 'loss', loss.item())
    
    scheduler.step()
    return metric_tracker
``` 