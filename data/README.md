# Data Directory

This directory contains datasets and data processing resources for the Neural Network App.

## Overview

The data directory is organized to store various datasets used for training, validation, and testing neural network models. It provides a structured approach to managing different types of data and their preprocessing.

## Directory Structure

```
data/
├── README.md                           # This file
├── MNIST/                              # MNIST handwritten digit dataset
│   ├── raw/                           # Raw dataset files
│   │   ├── t10k-images-idx3-ubyte      # Test images
│   │   ├── t10k-labels-idx1-ubyte      # Test labels
│   │   ├── train-images-idx3-ubyte     # Training images
│   │   └── train-labels-idx1-ubyte     # Training labels
│   ├── processed/                      # Processed MNIST data
│   │   ├── train/                      # Training data (images/labels)
│   │   ├── test/                       # Test data (images/labels)
│   │   └── validation/                 # Validation data (images/labels)
│   └── README.md                       # MNIST-specific documentation
├── CIFAR10/                            # CIFAR-10 image dataset
│   ├── raw/                           # Raw CIFAR-10 files
│   ├── processed/                      # Processed CIFAR-10 data
│   └── README.md                       # CIFAR-10 documentation
├── custom_datasets/                    # User-defined datasets
│   └── [dataset_name]/
│       ├── raw/                        # Raw data files
│       ├── processed/                  # Processed data
│       ├── metadata.json               # Dataset metadata
│       └── README.md                   # Dataset-specific info
└── temp/                               # Temporary data files
    └── downloads/                      # Downloaded datasets
```

## Dataset Categories

### MNIST Dataset
- **Location**: `data/MNIST/`
- **Description**: Handwritten digit recognition dataset
- **Size**: 60,000 training + 10,000 test images
- **Format**: 28x28 grayscale images
- **Labels**: 0-9 digits

### CIFAR-10 Dataset
- **Location**: `data/CIFAR10/`
- **Description**: 10-class image classification dataset
- **Size**: 50,000 training + 10,000 test images
- **Format**: 32x32 color images
- **Classes**: Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Custom Datasets
- **Location**: `data/custom_datasets/`
- **Description**: User-defined datasets for specific applications
- **Format**: Variable (depends on user requirements)
- **Structure**: Follows the standard raw/processed pattern

## Data Processing Pipeline

### 1. Raw Data Storage
- Store original, unprocessed datasets in the `raw/` subdirectories
- Maintain original file formats and naming conventions
- Include any accompanying metadata files

### 2. Data Processing
- Process raw data into formats suitable for neural network training
- Apply necessary preprocessing (normalization, resizing, etc.)
- Split data into train/validation/test sets
- Store processed data in the `processed/` subdirectories

### 3. Data Organization
```
[dataset]/
├── raw/
│   ├── [original_files]               # Original dataset files
│   └── [metadata_files]               # Dataset metadata
└── processed/
    ├── train/
    │   ├── images/                    # Training images
    │   └── labels/                    # Training labels
    ├── validation/
    │   ├── images/                    # Validation images
    │   └── labels/                    # Validation labels
    └── test/
        ├── images/                    # Test images
        └── labels/                    # Test labels
```

## File Naming Conventions

### Raw Data Files
- Keep original filenames when possible
- Use descriptive names for custom datasets
- Include version numbers if multiple versions exist

### Processed Data Files
- Images: `image_[index].[extension]` (e.g., `image_00001.png`)
- Labels: `label_[index].txt` or `label_[index].json`
- Batches: `batch_[batch_number].pth` or `batch_[batch_number].npz`

### Metadata Files
- `dataset_info.json` - Basic dataset information
- `preprocessing_config.json` - Preprocessing parameters
- `data_stats.json` - Dataset statistics

## Data Management Best Practices

### 1. Dataset Organization
- Use consistent directory structures
- Separate raw and processed data
- Include documentation for each dataset
- Maintain data integrity and versioning

### 2. File Management
- Use appropriate file formats for different data types
- Compress large datasets when possible
- Include checksums for data integrity verification
- Backup important datasets

### 3. Documentation
- Create README files for each dataset
- Document data sources and preprocessing steps
- Include dataset statistics and characteristics
- Provide usage examples

## Data Loading Examples

### Loading MNIST Data
```python
import torch
from torchvision import datasets, transforms
from pathlib import Path

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
data_path = Path("data/MNIST/processed")
train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Loading Custom Data
```python
import json
from pathlib import Path

def load_custom_dataset(dataset_name):
    dataset_path = Path(f"data/custom_datasets/{dataset_name}")
    
    # Load metadata
    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load processed data
    train_data = torch.load(dataset_path / "processed" / "train" / "data.pth")
    test_data = torch.load(dataset_path / "processed" / "test" / "data.pth")
    
    return train_data, test_data, metadata
```

## Data Preprocessing Guidelines

### Image Data
- Resize images to consistent dimensions
- Normalize pixel values (typically 0-1 or -1 to 1)
- Apply data augmentation for training data
- Convert to appropriate tensor formats

### Text Data
- Tokenize text sequences
- Create vocabulary mappings
- Pad sequences to consistent lengths
- Convert to numerical representations

### Numerical Data
- Normalize or standardize features
- Handle missing values appropriately
- Encode categorical variables
- Split into features and targets

## Dataset Statistics

### MNIST Dataset
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28x28 pixels
- **Color channels**: 1 (grayscale)
- **Classes**: 10 (digits 0-9)
- **File size**: ~50 MB (compressed)

### CIFAR-10 Dataset
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Image size**: 32x32 pixels
- **Color channels**: 3 (RGB)
- **Classes**: 10
- **File size**: ~163 MB (compressed)

## Adding New Datasets

### Steps to Add a New Dataset
1. Create dataset directory: `data/[dataset_name]/`
2. Create `raw/` and `processed/` subdirectories
3. Download or copy raw data files to `raw/`
4. Process the data and save to `processed/`
5. Create metadata files
6. Write dataset-specific README
7. Update the main data README if needed

### Required Files
- `README.md` - Dataset documentation
- `metadata.json` - Dataset information
- Processed data files in appropriate format
- Any preprocessing scripts or configuration

## Integration with Neural Network App

The Neural Network App automatically detects datasets in this directory. To use datasets in the app:

1. Place your dataset in the appropriate subdirectory
2. Follow the recommended directory structure
3. Include necessary metadata files
4. Ensure data is in compatible formats
5. Restart the app to detect new datasets

## Troubleshooting

### Common Issues
1. **Missing Files**: Ensure all required files are present
2. **Format Errors**: Verify data formats match expected types
3. **Path Issues**: Check file paths and permissions
4. **Memory Issues**: Consider data loading strategies for large datasets

### Data Validation
```python
def validate_dataset(dataset_path):
    """Validate dataset structure and files."""
    required_files = ["metadata.json", "README.md"]
    required_dirs = ["raw", "processed"]
    
    for file in required_files:
        if not (dataset_path / file).exists():
            raise FileNotFoundError(f"Missing required file: {file}")
    
    for dir in required_dirs:
        if not (dataset_path / dir).exists():
            raise FileNotFoundError(f"Missing required directory: {dir}")
    
    return True
```

## Storage Considerations

- **Disk Space**: Monitor available disk space for large datasets
- **Compression**: Use appropriate compression for large files
- **Backup Strategy**: Maintain backups of important datasets
- **Cloud Storage**: Consider cloud storage for very large datasets

## Contributing

When adding new datasets:

1. Follow the established directory structure
2. Include comprehensive documentation
3. Provide data loading examples
4. Test dataset compatibility with the app
5. Update documentation as needed

---

*Last updated: September 2025*
