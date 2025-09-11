# Checkpoints Directory

This directory contains saved model checkpoints for the Neural Network App.

## Overview

Model checkpoints are saved states of neural network models during or after training. They allow you to:

- Resume training from a specific point
- Load pre-trained models for inference
- Compare different model versions
- Backup important model states

## Directory Structure

```
checkpoints/
├── README.md                          # This file
├── mnist_cnn/                         # MNIST CNN model checkpoints
│   ├── best_model.pth                 # Best performing model
│   ├── final_model.pth                # Final trained model
│   └── checkpoint_epoch_XX.pth        # Intermediate checkpoints
├── custom_models/                     # Custom neural network models
│   └── [model_name]/
│       ├── config.json                # Model configuration
│       └── weights.pth                # Model weights
└── experiments/                       # Experimental models
    └── [experiment_name]/
        ├── model.pth                   # Model file
        └── training_log.txt           # Training log
```

## File Naming Conventions

### Model Files
- `best_model.pth` - Best performing model based on validation metrics
- `final_model.pth` - Model at the end of training
- `checkpoint_epoch_XX.pth` - Checkpoint at specific epoch (XX = epoch number)
- `[model_name]_v[version].pth` - Versioned model files

### Configuration Files
- `config.json` - Model architecture and training configuration
- `hyperparameters.json` - Training hyperparameters
- `training_log.txt` - Detailed training progress log

## Checkpoint Management

### Saving Checkpoints

When training models, checkpoints should be saved with meaningful names and include:

1. **Model weights** - The trained parameters
2. **Model configuration** - Architecture details
3. **Training state** - Optimizer state, current epoch, loss values
4. **Metadata** - Training date, dataset used, performance metrics

### Loading Checkpoints

To load a checkpoint in your code:

```python
import torch
from pathlib import Path

# Load model checkpoint
checkpoint_path = Path("checkpoints/mnist_cnn/best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load model state
model.load_state_dict(checkpoint['model_state_dict'])

# Load optimizer state (for resuming training)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Get training metadata
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### Best Practices

1. **Regular Backups**: Save checkpoints at regular intervals during training
2. **Version Control**: Use clear versioning for different model iterations
3. **Documentation**: Include README files in subdirectories explaining the model purpose
4. **Cleanup**: Remove old or unnecessary checkpoints to save disk space
5. **Validation**: Always validate checkpoint files before deployment

## Storage Guidelines

- **Large Models**: Consider compressing large checkpoint files
- **Cloud Storage**: For very large models, consider using cloud storage solutions
- **Backup Strategy**: Maintain backups of important checkpoints
- **Disk Space**: Monitor disk space usage and clean up old checkpoints

## Model Categories

### MNIST Models
- Location: `checkpoints/mnist_cnn/`
- Purpose: Handwritten digit recognition models
- Format: PyTorch `.pth` files

### Custom Models
- Location: `checkpoints/custom_models/`
- Purpose: User-defined neural network architectures
- Format: Model weights + configuration files

### Experimental Models
- Location: `checkpoints/experiments/`
- Purpose: Research and experimental models
- Format: Model files + training logs

## Troubleshooting

### Common Issues

1. **Loading Errors**: Ensure PyTorch version compatibility
2. **Missing Files**: Check if all required files are present
3. **Corrupted Files**: Verify file integrity and retrain if necessary
4. **Memory Issues**: Use `map_location='cpu'` when loading on CPU

### Recovery Steps

1. Check file integrity and permissions
2. Verify PyTorch version compatibility
3. Load with error handling:
   ```python
   try:
       checkpoint = torch.load(checkpoint_path)
   except Exception as e:
       print(f"Error loading checkpoint: {e}")
   ```

## Integration with Neural Network App

The Neural Network App automatically detects and loads checkpoints from this directory. To use checkpoints in the app:

1. Place your checkpoint files in the appropriate subdirectory
2. Ensure the checkpoint format matches the expected model architecture
3. Add configuration files if needed
4. Restart the app to detect new checkpoints

## Contributing

When adding new checkpoints:

1. Follow the directory structure
2. Use the naming conventions
3. Include necessary documentation
4. Test checkpoint loading functionality
5. Update the app's checkpoint detection if needed

---

*Last updated: September 2025*
