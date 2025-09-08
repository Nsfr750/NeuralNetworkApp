# Neural Network Application

A flexible and modular deep learning framework built with PyTorch, designed for rapid experimentation with neural network architectures, training workflows, and visualization tools.

## Features

### Network Building

- **NetworkBuilder**: Intuitive API for building neural networks layer by layer
- **Layer Types**: Support for Conv2D, Linear, BatchNorm, Dropout, Pooling, and more
- **Residual Connections**: Easy addition of skip connections and residual blocks
- **Shape Inference**: Automatic calculation of layer shapes

### Training & Optimization

- **Training Loop**: Flexible training loop with progress tracking
- **Learning Rate Scheduling**: Built-in support for various schedulers
- **Mixed Precision Training**: For faster training on supported hardware
- **Gradient Clipping**: For more stable training

### Visualization

- **Training Metrics**: Real-time plotting of loss and metrics
- **Model Architecture**: Visualization of network architecture
- **Embedding Visualization**: t-SNE and UMAP for high-dimensional data
- **Attention Maps**: Visualization of attention mechanisms
- **TensorBoard Integration**: For advanced experiment tracking

### Transfer Learning

- **Pre-trained Models**: Easy loading of pre-trained models
- **Fine-tuning**: Tools for fine-tuning on custom datasets
- **Model Export**: Export to ONNX and TFLite formats

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9.0+
- CUDA (optional, for GPU acceleration)

### Install from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/Nsfr750/NeuralNetworkApp.git
   cd NeuralNetworkApp
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. Install the package in development mode with all dependencies:

   ```bash
   pip install -e ".[all]"
   ```

   Or install with specific features:

   ```bash
   # Core functionality only
   pip install -e .
   
   # With visualization tools
   pip install -e ".[visualization]"
   
   # For development (includes tests and docs)
   pip install -e ".[dev]"
   ```

## Quick Start

### Building a Simple CNN

```python
from neuralnetworkapp import NetworkBuilder, LayerType
import torch

# Create a network builder
builder = NetworkBuilder(input_shape=(3, 32, 32))  # CIFAR-10 input shape

# Add layers
builder.add_conv2d(out_channels=32, kernel_size=3, padding=1)
builder.add_activation('relu')
builder.add_maxpool2d(kernel_size=2, stride=2)

builder.add_conv2d(out_channels=64, kernel_size=3, padding=1)
builder.add_activation('relu')
builder.add_maxpool2d(kernel_size=2, stride=2)

builder.add_flatten()
builder.add_linear(128)
builder.add_activation('relu')
builder.add_dropout(0.5)
builder.add_linear(10)  # 10 output classes

# Build the model
model = builder.build()
print(model)
```

### Training with Visualization

```python
from neuralnetworkapp import TrainingVisualizer
import torch.optim as optim
import torch.nn as nn

# Create a visualizer
visualizer = TrainingVisualizer(log_dir="logs")

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Log metrics
        visualizer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / len(val_loader.dataset)
    
    # Log validation metrics
    visualizer.add_scalar('val/loss', val_loss, epoch)
    visualizer.add_scalar('val/accuracy', val_acc, epoch)

# Save figures
visualizer.save_figures()
```

## Project Structure

```text
NeuralNetworkApp/
├── src/
│   └── neuralnetworkapp/
│       ├── __init__.py         # Package initialization
│       ├── version.py          # Version information
│       ├── network_builder/    # Network building utilities
│       ├── visualization/      # Visualization tools
│       ├── data/               # Data loading and processing
│       └── utils/              # Utility functions
├── tests/                     # Unit tests
├── examples/                  # Example scripts
├── docs/                      # Documentation
├── pyproject.toml             # Build system configuration
└── README.md                  # This file
```

## Examples

Check the `examples/` directory for complete examples, including:

- Image classification with CNNs
- Transfer learning with pre-trained models
- Custom training loops with visualization
- Model export to ONNX and TFLite

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

GNU General Public License v3 (GPL-3.0) © 2025 Nsfr750

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/Nsfr750/NeuralNetworkApp/issues).
