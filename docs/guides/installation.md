# Installation Guide

This guide will help you install NeuralNetworkApp and its dependencies on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for development installation)

## Installation Options

### Using pip (Recommended)

```bash
# Install the latest stable release
pip install neuralnetworkapp

# Or install with visualization dependencies
pip install "neuralnetworkapp[vis]"

# For Windows users
pip install -r requirements-windows.txt
```

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/Nsfr750/NeuralNetworkApp.git
   cd NeuralNetworkApp
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Verification

To verify your installation, run:

```python
import neuralnetworkapp as nna
print(f"NeuralNetworkApp version: {nna.__version__}")
```

## Common Issues

### Missing Dependencies
If you encounter missing dependencies, install them using:
```bash
pip install -r requirements.txt
```

### CUDA Support
For GPU acceleration with CUDA, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- cuDNN installed

### Windows-Specific Notes
- Use Anaconda for easier dependency management
- Install Visual C++ Build Tools if you encounter compilation errors

---
© Copyright 2025 Nsfr750. All Rights Reserved.
