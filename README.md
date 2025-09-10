# Neural Network Creator

**Neural Network Creator** - Making deep learning accessible through intuitive GUI design.

## Features

### 🎨 Modern GUI Interface

- **PySide6 Interface**: Clean, modern graphical user interface
- **Intuitive Design**: User-friendly layout for easy navigation
- **Professional Styling**: Custom application icon and themed interface
- **Multi-language Support**: English and Italian language options

### 🧠 Neural Network Creation

- **Visual Network Builder**: Create neural networks through an intuitive GUI
- **Layer Management**: Add, configure, and manage network layers easily
- **Real-time Preview**: See network architecture as you build it
- **Model Configuration**: Flexible input/output size and layer configuration

### 🚀 Training & Evaluation

- **Training Dashboard**: Monitor training progress in real-time
- **Performance Metrics**: Track loss, accuracy, and other metrics
- **Visualization Tools**: Interactive plots for training history
- **Model Evaluation**: Comprehensive evaluation tools and metrics

### 🛠️ Tools & Utilities

- **Tools Menu**: Centralized access to application utilities
- **Log Viewer**: View and analyze application logs with filtering options
- **Model Management**: Load, save, and manage trained models
- **Data Handling**: Built-in data loading and preprocessing tools

### 🌐 Advanced Features

- **Transfer Learning**: Support for pre-trained models and fine-tuning
- **Model Export**: Export models to various formats
- **Batch Processing**: Process multiple datasets efficiently
- **GPU Acceleration**: CUDA support for faster training

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9.0+
- CUDA (optional, for GPU acceleration)

### Install from Source

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Nsfr750/NeuralNetworkApp.git
   cd NeuralNetworkApp
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install the package in development mode with all dependencies**:

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

### Launching the Application

1. **Start the GUI Application**:

   ```bash
   python main.py
   ```

2. **Using the Interface**:

   - **Network Creation**: Use the visual interface to build neural networks
   - **Training**: Configure and start training with real-time monitoring
   - **Tools Menu**: Access utilities like the Log Viewer
   - **Language**: Switch between English and Italian from the menu

### Building Your First Neural Network

1. **Launch the application** and navigate to the network creation section

2. **Configure your network**:

   - Set input size (e.g., 784 for MNIST)
   - Add hidden layers (e.g., 128, 64)
   - Set output size (e.g., 10 for MNIST classes)
   - Choose activation function (e.g., ReLU)
   - Configure dropout rate if needed

3. **Create the model** and proceed to training

### Training Your Model

1. **Load your dataset** (e.g., MNIST, CIFAR-10)

2. **Configure training parameters**:

   - Number of epochs
   - Learning rate
   - Optimizer (Adam, SGD, etc.)
   - Loss function
   - Batch size

3. **Start training** and monitor progress in real-time

4. **Evaluate performance** using the built-in metrics and visualizations

### Using the Tools Menu

- **View Logs**: Access application logs with filtering options
- **Model Management**: Load, save, and manage trained models
- **Settings**: Configure application preferences

## Examples

Check the `examples/` directory for complete examples, including:

- **MNIST Classification**: Handwritten digit recognition with CNNs
- **CIFAR-10 Training**: Image classification on colored images
- **Network Builder Demo**: How to use the network builder programmatically
- **Optimization Examples**: Various optimization techniques and hyperparameter tuning

### Main Application Window
- Clean, modern interface with intuitive navigation
- Real-time network architecture visualization
- Interactive training dashboard

### Tools Menu
- Centralized access to utilities
- Log viewer with filtering capabilities
- Model management tools

### About Dialog
- Professional branding with company logo
- Version and license information
- Styled interface elements

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

GNU General Public License v3 (GPL-3.0) © 2025 Nsfr750

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/Nsfr750/NeuralNetworkApp/issues).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes and updates in each version.

---
© Copyright 2025 Nsfr750. All rights reserved.