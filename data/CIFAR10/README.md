# CIFAR-10 Dataset

This directory contains the CIFAR-10 dataset for the Neural Network App.

## Overview

CIFAR-10 is a widely used dataset for machine learning and computer vision tasks. It contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Dataset Information

### Basic Statistics
- **Total Images**: 60,000
- **Training Images**: 50,000
- **Test Images**: 10,000
- **Image Size**: 32x32 pixels
- **Color Channels**: 3 (RGB)
- **Number of Classes**: 10
- **Images per Class**: 6,000 (5,000 training + 1,000 test)

### Classes
The dataset contains the following 10 classes:

1. **Airplane** - Various aircraft and planes
2. **Automobile** - Cars, trucks, and other vehicles
3. **Bird** - Different species of birds
4. **Cat** - Various cat breeds and poses
5. **Deer** - Deer and similar animals
6. **Dog** - Different dog breeds
7. **Frog** - Frogs and amphibians
8. **Horse** - Horses and equine animals
9. **Ship** - Ships and boats
10. **Truck** - Trucks and heavy vehicles

## Directory Structure

```
data/CIFAR10/
├── README.md                          # This file
├── raw/                               # Raw dataset files
│   ├── cifar-10-batches-bin/          # Binary batch files
│   │   ├── data_batch_1.bin           # Training batch 1
│   │   ├── data_batch_2.bin           # Training batch 2
│   │   ├── data_batch_3.bin           # Training batch 3
│   │   ├── data_batch_4.bin           # Training batch 4
│   │   ├── data_batch_5.bin           # Training batch 5
│   │   └── test_batch.bin             # Test batch
│   ├── cifar-10-batches-py/           # Python pickle files
│   │   ├── data_batch_1               # Training batch 1
│   │   ├── data_batch_2               # Training batch 2
│   │   ├── data_batch_3               # Training batch 3
│   │   ├── data_batch_4               # Training batch 4
│   │   ├── data_batch_5               # Training batch 5
│   │   └── test_batch                 # Test batch
│   ├── batches.meta                   # Metadata file
│   └── README.txt                     # Original CIFAR-10 documentation
└── processed/                         # Processed data files
    ├── train/                         # Training data
    │   ├── images/                    # Training images (PNG format)
    │   │   ├── airplane_00001.png
    │   │   ├── airplane_00002.png
    │   │   └── ...                    # All training images
    │   └── labels/                    # Training labels
    │       ├── train_labels.txt       # Text format
    │       ├── train_labels.json      # JSON format
    │       └── train_labels.pt        # PyTorch format
    ├── test/                          # Test data
    │   ├── images/                    # Test images (PNG format)
    │   │   ├── airplane_00001.png
    │   │   ├── airplane_00002.png
    │   │   └── ...                    # All test images
    │   └── labels/                    # Test labels
    │       ├── test_labels.txt        # Text format
    │       ├── test_labels.json       # JSON format
    │       └── test_labels.pt         # PyTorch format
    ├── validation/                    # Validation data (optional)
    │   ├── images/                    # Validation images
    │   └── labels/                    # Validation labels
    ├── metadata/                      # Dataset metadata
    │   ├── class_names.json           # Class names mapping
    │   ├── dataset_stats.json         # Dataset statistics
    │   └── preprocessing_config.json  # Preprocessing configuration
    └── tensors/                       # PyTorch tensor files
        ├── train_images.pt            # Training images tensor
        ├── train_labels.pt            # Training labels tensor
        ├── test_images.pt             # Test images tensor
        └── test_labels.pt             # Test labels tensor
```

## Data Formats

### Raw Data Format
The CIFAR-10 dataset is available in multiple formats:

#### Binary Format (.bin)
- Each batch file contains 10,000 images
- Each image is stored as 3073 bytes (1 byte label + 3072 bytes image data)
- Image data: 1024 bytes red channel + 1024 bytes green channel + 1024 bytes blue channel
- Label: Single byte (0-9)

#### Python Pickle Format (.pkl)
- Dictionary format with keys: 'data', 'labels', 'batch_label', 'filenames'
- 'data': numpy array of shape (10000, 3072)
- 'labels': list of 10000 integers (0-9)
- 'batch_label': string describing the batch
- 'filenames': list of 10000 filenames

### Processed Data Format
After processing, the data is available in multiple formats:

#### Image Files (PNG)
- Individual 32x32 PNG images
- Organized by class and dataset split
- Easy to inspect and use with standard image libraries

#### Label Files
- **Text format**: One label per line
- **JSON format**: Structured data with metadata
- **PyTorch format**: Tensor format for direct loading

#### Tensor Files (.pt)
- PyTorch tensor format for efficient loading
- Combined images and labels in single files
- Optimized for neural network training

## Download and Setup

### Automatic Download
The Neural Network App can automatically download the CIFAR-10 dataset:

```python
from torchvision import datasets, transforms

# Download CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root='data/CIFAR10/raw',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.CIFAR10(
    root='data/CIFAR10/raw',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
```

### Manual Download
To download manually:

1. Visit the official CIFAR-10 website: https://www.cs.toronto.edu/~kriz/cifar.html
2. Download the CIFAR-10 binary version
3. Extract the files to `data/CIFAR10/raw/`
4. Run the preprocessing script to convert to processed format

### Preprocessing Script
Use the included preprocessing script to convert raw data to processed format:

```python
# examples/cifar10_preprocessing.py
python examples/cifar10_preprocessing.py --input_dir data/CIFAR10/raw --output_dir data/CIFAR10/processed
```

## Usage Examples

### Basic Usage with PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = datasets.CIFAR10(
    root='data/CIFAR10/processed',
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='data/CIFAR10/processed',
    train=False,
    download=False,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

### Custom Data Loading
```python
import json
import torch
from PIL import Image
from pathlib import Path

def load_cifar10_custom(data_dir):
    """Load CIFAR-10 dataset from processed files."""
    data_dir = Path(data_dir)
    
    # Load class names
    with open(data_dir / 'metadata' / 'class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Load training data
    train_images = []
    train_labels = []
    
    train_dir = data_dir / 'train'
    for class_idx, class_name in enumerate(class_names):
        class_dir = train_dir / 'images' / class_name
        for img_path in class_dir.glob('*.png'):
            image = Image.open(img_path).convert('RGB')
            train_images.append(image)
            train_labels.append(class_idx)
    
    # Load test data
    test_images = []
    test_labels = []
    
    test_dir = data_dir / 'test'
    for class_idx, class_name in enumerate(class_names):
        class_dir = test_dir / 'images' / class_name
        for img_path in class_dir.glob('*.png'):
            image = Image.open(img_path).convert('RGB')
            test_images.append(image)
            test_labels.append(class_idx)
    
    return (train_images, train_labels), (test_images, test_labels), class_names
```

### Data Augmentation
```python
from torchvision import transforms

# Advanced data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Test transform (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

## Data Preprocessing

### Normalization
CIFAR-10 images should be normalized using the dataset statistics:

```python
# CIFAR-10 normalization parameters
mean = [0.4914, 0.4822, 0.4465]  # Channel-wise mean
std = [0.2023, 0.1994, 0.2010]   # Channel-wise standard deviation

# Apply normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### Data Splitting
```python
from torch.utils.data import random_split

# Split training data into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

## Performance Benchmarks

### Expected Model Performance
| Model Architecture | Accuracy | Training Time | Parameters |
|-------------------|----------|---------------|------------|
| Simple CNN | ~70-75% | 30-60 min | ~100K |
| ResNet-18 | ~85-90% | 2-4 hours | ~11M |
| VGG-16 | ~88-92% | 3-5 hours | ~138M |
| EfficientNet-B0 | ~90-94% | 1-2 hours | ~5M |

### Class-wise Performance
Different classes have varying difficulty levels:

| Class | Typical Accuracy | Notes |
|-------|------------------|-------|
| Automobile | ~95% | Easy to distinguish |
| Ship | ~92% | Distinctive features |
| Truck | ~90% | Similar to automobile |
| Airplane | ~88% | Variable shapes |
| Frog | ~85% | Complex textures |
| Horse | ~83% | Pose variations |
| Bird | ~80% | High diversity |
| Cat | ~78% | Similar to other animals |
| Dog | ~75% | High intra-class variation |
| Deer | ~72% | Similar background colors |

## Best Practices

### Data Loading
- Use multiple workers for faster data loading
- Pin memory for GPU training
- Use appropriate batch sizes (32-128)
- Implement proper data shuffling

### Training Tips
- Start with simple models before complex architectures
- Use data augmentation for better generalization
- Monitor training and validation loss
- Implement early stopping to prevent overfitting

### Memory Management
- Use smaller batch sizes if memory is limited
- Implement gradient accumulation for large effective batch sizes
- Use mixed precision training for faster training
- Clear cache between experiments

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
   
   # Enable gradient checkpointing
   from torch.utils.checkpoint import checkpoint
   ```

2. **Slow Training**
   ```python
   # Use multiple workers
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
   
   # Pin memory for GPU
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
   ```

3. **Poor Performance**
   ```python
   # Check data normalization
   print(f"Data mean: {train_data.mean()}, std: {train_data.std()}")
   
   # Verify data augmentation
   # Visualize augmented images
   ```

### Data Validation
```python
def validate_cifar10_data(data_dir):
    """Validate CIFAR-10 dataset integrity."""
    data_dir = Path(data_dir)
    
    # Check required files
    required_files = [
        'train/images', 'train/labels',
        'test/images', 'test/labels',
        'metadata/class_names.json'
    ]
    
    for file_path in required_files:
        if not (data_dir / file_path).exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")
    
    # Check image count
    train_count = len(list((data_dir / 'train' / 'images').rglob('*.png')))
    test_count = len(list((data_dir / 'test' / 'images').rglob('*.png')))
    
    if train_count != 50000:
        raise ValueError(f"Expected 50,000 training images, found {train_count}")
    
    if test_count != 10000:
        raise ValueError(f"Expected 10,000 test images, found {test_count}")
    
    print("✓ CIFAR-10 dataset validation passed")
```

## Integration with Neural Network App

The CIFAR-10 dataset integrates seamlessly with the Neural Network App:

### Using in the App
```python
# Load CIFAR-10 in the app
from src.neuralnetworkapp.data import load_cifar10

train_loader, test_loader, class_names = load_cifar10(
    data_dir='data/CIFAR10/processed',
    batch_size=32,
    augment=True
)
```

### App Configuration
```python
# Add to app configuration
config = {
    'dataset': {
        'name': 'CIFAR10',
        'path': 'data/CIFAR10/processed',
        'input_size': (3, 32, 32),
        'num_classes': 10,
        'normalization': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        }
    }
}
```

## References and Resources

### Official Sources
- [CIFAR-10 Official Website](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-10 Paper](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [PyTorch CIFAR-10 Documentation](https://pytorch.org/vision/stable/datasets.html#cifar)

### Tutorials and Examples
- [PyTorch CIFAR-10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [TensorFlow CIFAR-10 Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Kaggle CIFAR-10 Competitions](https://www.kaggle.com/c/cifar-10)

### Research Papers
- "Learning Multiple Layers of Features from Tiny Images" (Krizhevsky, 2009)
- "Network in Network" (Lin et al., 2013)
- "Deep Residual Learning for Image Recognition" (He et al., 2015)

---

*Last updated: September 2025*
