# Examples Directory

This directory contains example scripts demonstrating how to use the Neural Network App for various machine learning tasks.

## Overview

The examples directory provides practical implementations of neural network concepts, training workflows, and advanced techniques. Each example is designed to be educational and easily adaptable for your own projects.

## Available Examples

### 1. CNN Example (`cnn_example.py`)
**Purpose**: Basic Convolutional Neural Network implementation
**Difficulty**: Beginner
**Dataset**: MNIST
**Key Features**:
- CNN architecture with convolutional and pooling layers
- Training loop with loss tracking
- Model evaluation and visualization
- Basic data preprocessing

**Usage**:
```bash
python examples/cnn_example.py
```

### 2. Network Builder Demo (`network_builder_demo.py`)
**Purpose**: Demonstrate the network builder functionality
**Difficulty**: Intermediate
**Dataset**: MNIST
**Key Features**:
- Dynamic network architecture creation
- Layer-by-layer network construction
- Custom layer configurations
- Network visualization and analysis

**Usage**:
```bash
python examples/network_builder_demo.py
```

### 3. CIFAR-10 Training (`cifar10_training.py`)
**Purpose**: Advanced image classification with CIFAR-10
**Difficulty**: Intermediate
**Dataset**: CIFAR-10
**Key Features**:
- Complex CNN architecture for color images
- Data augmentation techniques
- Advanced training strategies
- Performance optimization

**Usage**:
```bash
python examples/cifar10_training.py
```

### 4. Optimization Example (`optimization_example.py`)
**Purpose**: Demonstrate various optimization techniques
**Difficulty**: Advanced
**Dataset**: MNIST
**Key Features**:
- Different optimizers comparison (SGD, Adam, RMSprop)
- Learning rate scheduling
- Gradient clipping
- Convergence analysis

**Usage**:
```bash
python examples/optimization_example.py
```

### 5. Transfer Learning Example (`transfer_learning_example.py`)
**Purpose**: Transfer learning with pre-trained models
**Difficulty**: Advanced
**Dataset**: Custom dataset
**Key Features**:
- Pre-trained model loading and fine-tuning
- Feature extraction techniques
- Domain adaptation strategies
- Performance comparison with scratch training

**Usage**:
```bash
python examples/transfer_learning_example.py
```

## Example Structure

Each example follows a consistent structure:

```python
"""
Example: [Example Name]
Description: [Brief description of the example]
Author: Neural Network App
Date: [Creation date]
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Configuration
class Config:
    """Configuration parameters for the example."""
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 10
        # ... other parameters

# Model Definition
class Model(nn.Module):
    """Neural network model for this example."""
    def __init__(self):
        super().__init__()
        # Model architecture
    
    def forward(self, x):
        # Forward pass

# Training Function
def train_model(model, train_loader, criterion, optimizer, config):
    """Training loop for the model."""
    # Training implementation

# Evaluation Function
def evaluate_model(model, test_loader):
    """Model evaluation and metrics calculation."""
    # Evaluation implementation

# Main Function
def main():
    """Main execution function."""
    # Setup, training, and evaluation

if __name__ == "__main__":
    main()
```

## Running Examples

### Prerequisites
Before running any example, ensure you have:
- Python 3.8 or higher
- Required packages installed (see `requirements.txt`)
- Datasets downloaded and placed in the `data/` directory
- Sufficient disk space for models and checkpoints

### Basic Usage
```bash
# Run a specific example
python examples/cnn_example.py

# Run with custom parameters
python examples/cnn_example.py --epochs 20 --batch_size 64

# Run with debug mode
python examples/cnn_example.py --debug
```

### Command Line Arguments
Most examples support common command line arguments:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimizer
- `--debug`: Enable debug mode with verbose output
- `--save_model`: Save trained model to checkpoints
- `--plot_results`: Generate training plots

## Customizing Examples

### Modifying Architecture
To modify the neural network architecture:

```python
# In the Model class
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Modify channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Add more layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjust dimensions
        self.fc2 = nn.Linear(128, 10)
```

### Changing Hyperparameters
Modify the configuration class:

```python
class Config:
    def __init__(self):
        self.batch_size = 64          # Increase batch size
        self.learning_rate = 0.0005   # Decrease learning rate
        self.epochs = 50              # Increase training epochs
        self.momentum = 0.9           # Add momentum
```

### Adding New Features
To add new features like data augmentation:

```python
# Add to imports
from torchvision import transforms

# Modify data loading
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Add augmentation
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## Learning Path

### Beginner Level
1. **Start with `cnn_example.py`**: Learn basic CNN concepts
2. **Understand the training loop**: Grasp the fundamental training process
3. **Experiment with hyperparameters**: See how they affect performance

### Intermediate Level
1. **Try `network_builder_demo.py`**: Learn dynamic network creation
2. **Work with `cifar10_training.py`**: Handle more complex datasets
3. **Implement custom layers**: Extend the basic architectures

### Advanced Level
1. **Explore `optimization_example.py`**: Master optimization techniques
2. **Use `transfer_learning_example.py`**: Apply advanced ML concepts
3. **Create your own examples**: Build custom solutions

## Best Practices

### Code Organization
- Follow the established example structure
- Use clear and descriptive variable names
- Add comprehensive comments and docstrings
- Separate configuration, model, and training logic

### Experimentation
- Keep track of experiments and results
- Use version control for your modifications
- Document your changes and findings
- Share successful modifications with the community

### Performance Optimization
- Monitor GPU memory usage
- Use appropriate batch sizes
- Implement early stopping when needed
- Profile code to identify bottlenecks

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model architectures
   - Enable gradient checkpointing

2. **Slow Training**
   - Check GPU utilization
   - Optimize data loading
   - Use mixed precision training

3. **Poor Model Performance**
   - Verify data preprocessing
   - Check learning rate and optimizer settings
   - Ensure proper data splitting

### Debug Mode
Enable debug mode for detailed output:
```bash
python examples/cnn_example.py --debug
```

This provides:
- Detailed training progress
- Model architecture summary
- Data loading statistics
- Memory usage information

## Creating New Examples

### Example Template
```python
"""
Example: [Your Example Name]
Description: [Brief description]
Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 10

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your architecture
    
    def forward(self, x):
        # Implement forward pass

def train_model(model, train_loader, criterion, optimizer, config):
    # Implement training loop
    pass

def evaluate_model(model, test_loader):
    # Implement evaluation
    pass

def main():
    # Main execution
    pass

if __name__ == "__main__":
    main()
```

### Submission Guidelines
When contributing new examples:
1. Follow the established structure and conventions
2. Include comprehensive documentation
3. Test the example with different configurations
4. Provide usage instructions and expected outputs
5. Add appropriate error handling

## Integration with Neural Network App

Examples can be integrated with the Neural Network App:

1. **Using App Components**: Import and use app modules
   ```python
   from src.neuralnetworkapp.builder import NetworkBuilder
   from src.neuralnetworkapp.data import DataLoader
   ```

2. **Saving to Checkpoints**: Save models to the checkpoints directory
   ```python
   torch.save(model.state_dict(), "checkpoints/custom_models/my_model.pth")
   ```

3. **Loading App Config**: Use app configuration
   ```python
   from src.neuralnetworkapp.config import get_config
   config = get_config()
   ```

## Performance Benchmarks

### Expected Results
| Example | Dataset | Accuracy | Training Time | Model Size |
|---------|---------|----------|---------------|------------|
| CNN Example | MNIST | ~98% | 5-10 min | ~2MB |
| CIFAR-10 | CIFAR-10 | ~75% | 30-60 min | ~10MB |
| Transfer Learning | Custom | Varies | Varies | Varies |

### Hardware Requirements
- **GPU**: Recommended for faster training (NVIDIA GPU with CUDA)
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: Minimum 5GB free space
- **CPU**: Multi-core processor recommended

## Contributing

We welcome contributions to the examples directory! To contribute:

1. **Fork the repository**
2. **Create your example** following the guidelines
3. **Test thoroughly** with different configurations
4. **Submit a pull request** with detailed description
5. **Respond to feedback** and make necessary changes

## Support and Community

- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Join community discussions about examples
- **Documentation**: Check the main documentation for additional resources
- **Tutorials**: Look for video tutorials and blog posts

---

*Last updated: September 2025*
