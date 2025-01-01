# Custom Extensions Tutorial

Learn how to extend PyTorch Basics Library with custom functionality.

## Creating Custom Initializers

```python
import torch.nn as nn
from pytorch_basics_library import init_weights

def custom_init(module):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -0.1, 0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# Register custom initializer
def register_custom_init():
    init_weights._initializers['custom'] = custom_init

# Usage
register_custom_init()
model = nn.Linear(10, 5)
init_weights(model, method='custom')
```

## Custom Metric Tracking

```python
from pytorch_basics_library import accumulate_metric

class MetricTracker:
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name, initial_value=None):
        self.metrics[name] = [] if initial_value is None else [initial_value]
    
    def update(self, name, value):
        if name not in self.metrics:
            self.add_metric(name)
        accumulate_metric(self.metrics, name, value)
    
    def get_average(self, name):
        return sum(self.metrics[name]) / len(self.metrics[name])

# Usage
tracker = MetricTracker()
tracker.add_metric('loss')
tracker.add_metric('accuracy')

# During training
tracker.update('loss', loss.item())
tracker.update('accuracy', acc)
```

## Custom Device Management

```python
from pytorch_basics_library import get_device, to_device

class DeviceManager:
    def __init__(self, force_cpu=False):
        self.device = torch.device('cpu') if force_cpu else get_device()
        
    def to_device(self, *args):
        return tuple(to_device(arg, self.device) for arg in args)
    
    def clear_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

# Usage
device_mgr = DeviceManager()
inputs, targets = device_mgr.to_device(inputs, targets)
```

## Custom Training Callbacks

```python
class TrainingCallback:
    def on_epoch_start(self, epoch, model):
        pass
    
    def on_epoch_end(self, epoch, model, metrics):
        pass
    
    def on_batch_start(self, batch_idx):
        pass
    
    def on_batch_end(self, batch_idx, loss):
        pass

class MetricLoggingCallback(TrainingCallback):
    def __init__(self):
        self.epoch_metrics = {}
    
    def on_epoch_end(self, epoch, model, metrics):
        self.epoch_metrics[epoch] = metrics
        print(f"Epoch {epoch} metrics:", metrics)

# Usage with training loop
callback = MetricLoggingCallback()

for epoch in range(num_epochs):
    callback.on_epoch_start(epoch, model)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        callback.on_batch_start(batch_idx)
        # Training step
        callback.on_batch_end(batch_idx, loss.item())
    
    callback.on_epoch_end(epoch, model, metric_tracker)
``` 