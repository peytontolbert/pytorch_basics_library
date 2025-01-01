"""Basic Usage Examples for Initializers.

This script demonstrates basic usage patterns for the initializers module,
including different initialization strategies and their effects on model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from pytorch_basics_library.initializers import (
    initialize_model,
    xavier_uniform,
    kaiming_normal,
    UniformInitializer,
    NormalInitializer
)

def create_synthetic_data(n_samples: int = 1000) -> tuple:
    """Create synthetic regression data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X, y) tensors
    """
    X = torch.randn(n_samples, 10)
    w = torch.randn(10, 1)
    y = X @ w + 0.1 * torch.randn(n_samples, 1)
    return X, y

class SimpleNet(nn.Module):
    """Simple neural network for regression."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    n_epochs: int = 10
) -> list:
    """Train the model and return training losses.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        n_epochs: Number of training epochs
        
    Returns:
        List of training losses per epoch
    """
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

def compare_initializations():
    """Compare different initialization strategies."""
    # Create synthetic dataset
    X, y = create_synthetic_data()
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize models with different strategies
    models = {
        'Xavier Uniform': initialize_model(SimpleNet(), 'xavier'),
        'Kaiming Normal': initialize_model(SimpleNet(), 'kaiming'),
        'Uniform(-0.1, 0.1)': initialize_model(
            SimpleNet(),
            UniformInitializer(a=-0.1, b=0.1)
        ),
        'Normal(0, 0.01)': initialize_model(
            SimpleNet(),
            NormalInitializer(mean=0, std=0.01)
        )
    }
    
    # Train models and collect losses
    all_losses = {}
    criterion = nn.MSELoss()
    
    for name, model in models.items():
        print(f"\nTraining with {name} initialization:")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        losses = train_model(model, train_loader, criterion, optimizer)
        all_losses[name] = losses
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, losses in all_losses.items():
        plt.plot(losses, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss with Different Initializations')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('initialization_comparison.png')
    plt.close()

def demonstrate_custom_initialization():
    """Demonstrate custom initialization strategy."""
    # Create a model with custom initialization per layer
    model = SimpleNet()
    
    # Initialize different layers with different strategies
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'net.0' in name:  # First layer
                xavier_uniform.initialize(module.weight)
            elif 'net.2' in name:  # Second layer
                kaiming_normal.initialize(module.weight)
            elif 'net.4' in name:  # Output layer
                nn.init.normal_(module.weight, mean=0, std=0.01)
            
            # Initialize all biases to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    # Print initialization statistics
    print("\nCustom Initialization Statistics:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}:")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std:  {param.std().item():.4f}")

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("Comparing different initialization strategies...")
    compare_initializations()
    
    print("\nDemonstrating custom initialization...")
    demonstrate_custom_initialization() 