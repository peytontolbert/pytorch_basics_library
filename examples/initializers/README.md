# Initializers Examples

This directory contains example scripts demonstrating various usage patterns of the initializers module.

## Basic Usage (`basic_usage.py`)

This script demonstrates fundamental usage patterns of the initializers module on a simple regression task. It includes:

- Comparison of different initialization strategies
- Training convergence visualization
- Custom initialization per layer
- Pre-configured initializer usage

To run:
```bash
python basic_usage.py
```

The script will:
1. Train models with different initialization strategies
2. Generate a plot comparing training convergence (`initialization_comparison.png`)
3. Print statistics about custom initialization

## CNN Initialization (`cnn_initialization.py`)

This script focuses on initialization strategies for convolutional neural networks. It includes:

- Visualization of convolutional filters
- Analysis of weight distributions
- Comparison of fan-in vs fan-out modes
- Layer-wise statistics

To run:
```bash
python cnn_initialization.py
```

The script will generate several visualizations:
- `weight_distributions.png`: Weight distributions for different initialization strategies
- `conv_filters_*.png`: Visualizations of convolutional filters for each initialization method

## Requirements

These examples require additional dependencies:
```bash
pip install matplotlib seaborn
```

## Expected Output

### Basic Usage
The script will output training loss curves showing how different initialization strategies affect model convergence. Example output:

```
Training with Xavier Uniform initialization:
Epoch 1/10, Loss: 0.8245
Epoch 2/10, Loss: 0.3421
...

Training with Kaiming Normal initialization:
Epoch 1/10, Loss: 0.7123
Epoch 2/10, Loss: 0.2987
...
```

### CNN Initialization
The script will output detailed statistics for each layer and initialization strategy:

```
Analyzing Xavier (fan_in) initialization:
features.0 (Conv2d):
  mean: 0.0012
  std: 0.1234
  shape: (32, 3, 3, 3)
...

Analyzing Kaiming (fan_out) initialization:
features.0 (Conv2d):
  mean: -0.0003
  std: 0.1567
  shape: (32, 3, 3, 3)
...
```

## Tips for Best Results

1. **Reproducibility**: Both scripts use `torch.manual_seed(42)` for reproducible results. Change this value to experiment with different random initializations.

2. **Visualization**: The generated plots are saved as PNG files. For interactive visualization, you can modify the scripts to use `plt.show()` instead of `plt.savefig()`.

3. **Custom Models**: The examples use simple models for demonstration. You can modify the model architectures or create your own to experiment with different network structures.

4. **Hardware Acceleration**: The examples will automatically use CUDA if available. No code changes are needed.

## Further Reading

- [Understanding Xavier Initialization](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- [Delving Deep into Rectifiers (Kaiming/He Initialization)](https://arxiv.org/abs/1502.01852)
- [PyTorch nn.init Documentation](https://pytorch.org/docs/stable/nn.init.html) 