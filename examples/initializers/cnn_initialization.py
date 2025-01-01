"""CNN Initialization Examples.

This script demonstrates initialization strategies specifically designed for
convolutional neural networks, including proper initialization of different
layer types and visualization of initialization effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from pytorch_basics_library.initializers import (
    initialize_model,
    XavierInitializer,
    KaimingInitializer
)

class ConvNet(nn.Module):
    """Simple CNN architecture for demonstration."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def analyze_initialization(
    model: nn.Module,
    layer_types: Tuple[type] = (nn.Conv2d, nn.Linear)
) -> Dict[str, Dict[str, float]]:
    """Analyze initialization statistics for model layers.
    
    Args:
        model: The neural network model
        layer_types: Tuple of layer types to analyze
        
    Returns:
        Dictionary containing statistics for each layer
    """
    stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            weight = module.weight.data
            stats[name] = {
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'min': weight.min().item(),
                'max': weight.max().item(),
                'shape': tuple(weight.shape)
            }
    
    return stats

def plot_weight_distributions(
    models: Dict[str, nn.Module],
    save_path: str = 'weight_distributions.png'
):
    """Plot weight distributions for different initialization strategies.
    
    Args:
        models: Dictionary of models with different initializations
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    n_models = len(models)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    for idx, (name, model) in enumerate(models.items(), 1):
        plt.subplot(n_rows, n_cols, idx)
        
        # Collect all weights
        weights = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights.append(module.weight.data.view(-1).cpu().numpy())
        
        # Plot distribution
        sns.histplot(torch.cat([torch.from_numpy(w) for w in weights]), bins=50)
        plt.title(f'{name} Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_conv_filters(
    model: nn.Module,
    layer_name: str = 'features.0',
    save_path: str = 'conv_filters.png'
):
    """Visualize convolutional filters after initialization.
    
    Args:
        model: The neural network model
        layer_name: Name of the convolutional layer to visualize
        save_path: Path to save the visualization
    """
    # Get the specified conv layer
    conv_layer = dict(model.named_modules())[layer_name]
    if not isinstance(conv_layer, nn.Conv2d):
        raise ValueError(f"Layer {layer_name} is not a Conv2d layer")
    
    # Get weights and normalize them for visualization
    weights = conv_layer.weight.data.cpu()
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Plot filters
    n_filters = min(32, weights.size(0))
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    plt.figure(figsize=(2*n_cols, 2*n_rows))
    for i in range(n_filters):
        plt.subplot(n_rows, n_cols, i + 1)
        # For RGB filters, convert to grayscale by averaging channels
        if weights.size(1) == 3:
            plt.imshow(weights[i].mean(0), cmap='gray')
        else:
            plt.imshow(weights[i][0], cmap='gray')
        plt.axis('off')
    
    plt.suptitle(f'Filters from {layer_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_cnn_initializations():
    """Compare different initialization strategies for CNNs."""
    # Create models with different initialization strategies
    models = {
        'Xavier (fan_in)': ConvNet(),
        'Xavier (fan_out)': ConvNet(),
        'Kaiming (fan_in)': ConvNet(),
        'Kaiming (fan_out)': ConvNet()
    }
    
    # Initialize models
    xavier_in = XavierInitializer(distribution='normal')
    xavier_out = XavierInitializer(distribution='normal')
    kaiming_in = KaimingInitializer(mode='fan_in', nonlinearity='relu')
    kaiming_out = KaimingInitializer(mode='fan_out', nonlinearity='relu')
    
    initializers = {
        'Xavier (fan_in)': xavier_in,
        'Xavier (fan_out)': xavier_out,
        'Kaiming (fan_in)': kaiming_in,
        'Kaiming (fan_out)': kaiming_out
    }
    
    # Apply initializations and analyze
    for name, model in models.items():
        initialize_model(model, initializers[name])
        
        print(f"\nAnalyzing {name} initialization:")
        stats = analyze_initialization(model)
        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name}:")
            for stat_name, value in layer_stats.items():
                if stat_name != 'shape':
                    print(f"  {stat_name}: {value:.4f}")
                else:
                    print(f"  {stat_name}: {value}")
    
    # Plot weight distributions
    plot_weight_distributions(models)
    
    # Visualize conv filters for each initialization
    for name, model in models.items():
        visualize_conv_filters(
            model,
            save_path=f'conv_filters_{name.lower().replace(" ", "_")}.png'
        )

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("Comparing CNN initialization strategies...")
    compare_cnn_initializations() 