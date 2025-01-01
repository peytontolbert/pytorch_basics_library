# Models Documentation

This document covers the various model implementations available in PyTorch Basics Library.

## Base Model

The `BaseModel` class provides common functionality for all models:

```python
from pytorch_basics_library.models import BaseModel

class MyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

# Usage
model = MyModel()
model.to_device()  # Automatically moves to GPU if available
print(f"Trainable parameters: {model.count_parameters()}")
```

## Transformer Model

Implementation of the standard Transformer architecture:

```python
from pytorch_basics_library.models import TransformerModel

# Create a transformer model
transformer = TransformerModel(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# Generate sample data
src = torch.randn(10, 32, 512)  # (seq_len, batch_size, d_model)
tgt = torch.randn(20, 32, 512)

# Forward pass
output = transformer(src, tgt)
```

## Attention Mechanisms

The library provides various attention implementations:

```python
from pytorch_basics_library.models import MultiHeadAttention, SelfAttention

# Multi-head attention
mha = MultiHeadAttention(embed_dim=512, num_heads=8)
output = mha(query, key, value)

# Self-attention
self_attn = SelfAttention(embed_dim=512)
output = self_attn(x)
``` 