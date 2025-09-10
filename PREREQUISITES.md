# Prerequisites

This document outlines the system and software requirements needed to run, develop, and contribute to the Neural Network App.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Processor**: 64-bit processor with 4+ cores (CPU with AVX support recommended)
- **Memory**: 8GB RAM
- **Storage**: 2GB available space
- **Display**: 1280x720 resolution
- **GPU**: Optional but recommended for training (NVIDIA GPU with CUDA support)

### Recommended Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Processor**: 64-bit quad-core processor or higher
- **Memory**: 16GB RAM or more
- **Storage**: 5GB available space (SSD recommended)
- **Display**: 1920x1080 resolution or higher
- **GPU**: NVIDIA GPU with 4GB+ VRAM and CUDA support

## Software Dependencies

### Runtime Dependencies
- **Python**: 3.8 or higher
- **Pip**: Latest version
- **Git**: For version control
- **CUDA Toolkit**: 11.8+ (optional, for GPU acceleration)
- **cuDNN**: 8.6+ (optional, for GPU acceleration)

### Python Packages
Core dependencies (automatically installed with the application):
```
# Core scientific computing
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.2.0,<2.0.0
scipy>=1.10.0,<2.0.0

# Image processing
wand>=0.6.13,<1.0.0
opencv-python-headless>=4.5.0,<5.0.0
Pillow>=8.0.0

# PyTorch ecosystem
torch>=2.0.0,<3.0.0
torchvision>=0.15.0,<1.0.0
torchaudio>=2.0.0,<3.0.0
pytorch-lightning>=1.12.0,<3.0.0
torch-optimizer>=0.3.0
torch-lr-finder>=0.2.1
torch-amp>=0.1.0

# GUI Framework
PySide6>=6.8.3,<7.0.0
PySide6-Qt6>=6.8.3,<7.0.0

# Configuration and utilities
pyyaml>=6.0.2,<7.0.0
pydantic>=2.5.2,<3.0.0
tqdm>=4.65.0,<5.0.0

# Visualization
seaborn>=0.12.0,<1.0.0
plotly>=5.0.0,<6.0.0
```

### Development Dependencies
Additional dependencies for development:
```bash
pytest>=7.0.0,<8.0.0
pytest-cov>=4.0.0,<5.0.0
black>=23.0.0,<24.0.0
isort>=5.12.0,<6.0.0
flake8>=6.0.0,<7.0.0
mypy>=1.0.0,<2.0.0
sphinx>=4.2.0
sphinx-rtd-theme>=1.0.0
```

## Development Environment Setup

### Windows Setup
1. Install Python 3.8+ from [python.org](https://www.python.org/downloads/windows/)
2. Install Git from [git-scm.com](https://git-scm.com/download/win)
3. Install CUDA Toolkit 11.8+ (optional, for GPU acceleration) from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
4. Install cuDNN 8.6+ (optional, for GPU acceleration) from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
5. Clone the repository:
   ```bash
   git clone https://github.com/Nsfr750/NeuralNetworkApp.git
   cd NeuralNetworkApp
   ```
6. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
7. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Building from Source
1. Install build tools:
   ```bash
   pip install build
   ```
2. Build the package:
   ```bash
   python -m build
   ```
3. Install the built package:
   ```bash
   pip install dist/neuralnetworkapp-*.whl
   ```

## Configuration

### Environment Variables
- `NEURALNETWORKAPP_CONFIG_PATH`: Path to custom configuration file
- `NEURALNETWORKAPP_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `NEURALNETWORKAPP_DATA_DIR`: Directory to store datasets
- `NEURALNETWORKAPP_CHECKPOINTS_DIR`: Directory to store model checkpoints
- `NEURALNETWORKAPP_LOGS_DIR`: Directory to store training logs

### Configuration Files
- `config/config.json`: Main configuration file
- `config/logging.conf`: Logging configuration
- `data/`: Dataset storage directory
- `checkpoints/`: Model checkpoint storage
- `logs/`: Training and application logs

## Troubleshooting

### Common Issues

#### 1. **Missing Dependencies**
**Problem**: Import errors for PyTorch or other packages
**Solution**: Ensure all dependencies are installed correctly:
```bash
pip install -r requirements.txt
```
For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. **GPU Not Detected**
**Problem**: CUDA device not available even with NVIDIA GPU
**Solution**: 
- Verify CUDA Toolkit and cuDNN installation
- Check GPU driver compatibility
- Test CUDA availability:
  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.device_count())
  ```

#### 3. **Memory Issues During Training**
**Problem**: Out of memory errors during model training
**Solution**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Clear GPU cache between runs

#### 4. **Import Errors for Custom Modules**
**Problem**: Cannot import neuralnetworkapp modules
**Solution**:
- Ensure you're running from the project root directory
- Add the `src` directory to Python path:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  ```

#### 5. **GUI Not Starting**
**Problem**: Application fails to start GUI
**Solution**:
- Check PySide6 installation
- Verify display settings
- Run with debug mode:
  ```bash
  python main.py --debug
  ```

## Getting Help
If you encounter any issues, please:
1. Check the [documentation](docs/)
2. Review the [examples](examples/)
3. [Open an issue](https://github.com/Nsfr750/NeuralNetworkApp/issues) on GitHub
4. Join our [Discord server](https://discord.gg/ryqNeuRYjD)
5. Visit our [GitHub repository](https://github.com/Nsfr750/NeuralNetworkApp)
