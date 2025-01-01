# Weight Initialization

The weight initialization module provides various strategies for initializing neural network weights in PyTorch.

## Overview

The module provides:
- Xavier/Glorot initialization
- Kaiming/He initialization
- Basic distribution initializers (uniform, normal)
- Custom initialization support
- Pre-configured instances for common use cases

## Basic Usage

```python
from pytorch_basics_library.initializers import initialize_model
import torch.nn as nn

# Create a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Initialize with string identifier
model = initialize_model(model, 'xavier')

# Initialize with custom configuration
model = initialize_model(
    model,
    'kaiming',
    {'mode': 'fan_out', 'nonlinearity': 'relu'}
)
```

## Initialization Strategies

### Xavier/Glorot Initialization

```python
from pytorch_basics_library.initializers import xavier_uniform, xavier_normal

# Using pre-configured instances
model = initialize_model(model, xavier_uniform)
model = initialize_model(model, xavier_normal)

# Custom configuration
model = initialize_model(
    model,
    'xavier',
    {'gain': 2.0, 'distribution': 'uniform'}
)
```

### Kaiming/He Initialization

```python
from pytorch_basics_library.initializers import kaiming_uniform, kaiming_normal

# Using pre-configured instances
model = initialize_model(model, kaiming_uniform)
model = initialize_model(model, kaiming_normal)

# Custom configuration
model = initialize_model(
    model,
    'kaiming',
    {
        'mode': 'fan_out',
        'nonlinearity': 'relu',
        'distribution': 'normal'
    }
)
```

### Basic Distributions

```python
from pytorch_basics_library.initializers import uniform, normal

# Uniform distribution
model = initialize_model(model, uniform)
model = initialize_model(model, 'uniform', {'a': -0.5, 'b': 0.5})

# Normal distribution
model = initialize_model(model, normal)
model = initialize_model(model, 'normal', {'mean': 0.0, 'std': 0.02})
```

## API Reference

### Initializer Classes

#### `XavierInitializer`
```python
XavierInitializer(gain: float = 1.0, distribution: str = 'uniform')
```
- `gain`: Scaling factor for the weights
- `distribution`: Either 'uniform' or 'normal'

#### `KaimingInitializer`
```python
KaimingInitializer(
    mode: str = 'fan_in',
    nonlinearity: str = 'relu',
    distribution: str = 'normal'
)
```
- `mode`: Either 'fan_in' or 'fan_out'
- `nonlinearity`: The non-linear function (e.g., 'relu', 'tanh')
- `distribution`: Either 'normal' or 'uniform'

#### `UniformInitializer`
```python
UniformInitializer(a: float = 0.0, b: float = 1.0)
```
- `a`: Lower bound
- `b`: Upper bound

#### `NormalInitializer`
```python
NormalInitializer(mean: float = 0.0, std: float = 1.0)
```
- `mean`: Mean of the distribution
- `std`: Standard deviation

### Utility Functions

#### `initialize_model`
```python
initialize_model(
    model: nn.Module,
    initializer: Union[str, Initializer],
    initializer_config: Optional[Dict] = None
) -> nn.Module
```
- `model`: PyTorch model to initialize
- `initializer`: String identifier or Initializer instance
- `initializer_config`: Configuration dictionary for string initializers

## Best Practices

1. **Initialization Selection**
   - Use Xavier/Glorot for layers without non-linearities
   - Use Kaiming/He for ReLU networks
   - Consider the network architecture when choosing initialization

2. **Configuration**
   - Adjust gain/scale based on network depth
   - Match initialization to activation functions
   - Use appropriate fan mode for different layer types

3. **Validation**
   - Check weight statistics after initialization
   - Verify gradient flow in deep networks
   - Monitor training stability

## Common Issues and Solutions

1. **Vanishing/Exploding Gradients**
   - Use Kaiming initialization for deep ReLU networks
   - Adjust gain for very deep networks
   - Consider layer normalization

2. **Training Instability**
   - Verify initialization statistics
   - Check for appropriate scaling
   - Monitor gradient norms during training

3. **Type Mismatches**
   - Ensure consistent dtype across layers
   - Handle non-floating point tensors
   - Validate input parameters

## Examples

Check out the [weight initialization example](../examples/weight_initialization_example.py) for more detailed usage examples. 