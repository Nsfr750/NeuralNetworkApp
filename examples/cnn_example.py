"""
CNN Example with Batch Normalization and Dropout

This script demonstrates how to use the CNN models with batch normalization and dropout.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import our custom modules
from neuralnetworkapp.models.cnn_modelsimport create_cnn
from neuralnetworkapp.trainingimport Trainer
from neuralnetworkapp.optimizationimport get_optimizer, get_scheduler

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_cifar10(batch_size=128):
    """Load CIFAR-10 dataset with data augmentation."""
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Just normalization for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader

def train_cnn():
    """Train a CNN model on CIFAR-10."""
    # Load data
    train_loader, test_loader = load_cifar10(batch_size=128)
    
    # Create model
    input_shape = (3, 32, 32)  # CIFAR-10 images are 32x32 with 3 color channels
    num_classes = 10  # CIFAR-10 has 10 classes
    
    # You can choose between 'simple_cnn', 'vgg_like', or 'resnet_like'
    model = create_cnn(
        model_name='vgg_like',
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_fc=0.5  # Dropout for fully connected layers
    ).to(device)
    
    print(f"Model architecture:")
    print(model)
    
    # Create optimizer
    optimizer = get_optimizer(
        name='adam',
        model_params=model.parameters(),
        custom_params={
            'lr': 0.001,
            'weight_decay': 1e-4
        }
    )
    
    # Create learning rate scheduler
    scheduler = get_scheduler(
        name='reduce_on_plateau',
        optimizer=optimizer,
        custom_params={
            'mode': 'min',
            'factor': 0.1,
            'patience': 5,
            'verbose': True
        }
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        metrics=['accuracy'],
        use_amp=True  # Enable mixed precision training if available
    )
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=20,
        verbose=1
    )
    
    return history

if __name__ == '__main__':
    train_cnn()
