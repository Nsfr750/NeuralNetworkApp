"""
Example usage of the Neural Network Application

This module demonstrates how to use the various components of the neural network application,
including data loading, model creation, training, and evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from neuralnetworkapp.models import create_model, ModelType, ActivationType
from neuralnetworkapp.data.utils import get_image_dataloaders, get_tabular_dataloaders, TabularDataset
from neuralnetworkapp.metrics import get_metrics_for_task
from neuralnetworkapp.training import Trainer, EarlyStopping, ModelCheckpoint
from neuralnetworkapp.data.augmentation import get_default_image_augmentations


def train_mnist():
    """Train a CNN model on the MNIST dataset."""
    print("Training CNN on MNIST dataset...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 10
    
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simpler model that's guaranteed to work with MNIST dimensions
    class SimpleMNISTCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleMNISTCNN, self).__init__()
            self.features = nn.Sequential(
                # Input: 1x28x28
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32x28x28
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32x14x14
                nn.BatchNorm2d(32),
                nn.Dropout2d(0.25),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x14x14
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64x7x7
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.25)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.25),
                nn.Linear(128, num_classes)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = SimpleMNISTCNN(num_classes=num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Define metrics
    metrics = get_metrics_for_task(
        task_type='multiclass',
        num_classes=num_classes,
        average='macro'
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=5, verbose=True),
        ModelCheckpoint(
            filepath='checkpoints/mnist_cnn_best.pth',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=True
        )
    ]
    
    # Create trainer using the second Trainer class implementation
    trainer = Trainer(
        model=model,
        device=device,
        optimizer='adam',
        learning_rate=0.001,
        weight_decay=1e-5,  # L2 regularization
        loss_fn='cross_entropy',
        metrics=['accuracy']
    )
    
    # Set up the scheduler with the correct parameter structure for the optimization module
    trainer.set_scheduler(
        'steplr',  # Scheduler name
        custom_params={
            'step_size': 5,  # Step size in number of epochs
            'gamma': 0.1     # Multiplicative factor for learning rate decay
        }
    )
    # Remove the original scheduler since we're using the trainer's internal one
    scheduler = None
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for this example
        epochs=num_epochs  # Using 'epochs' parameter instead of 'num_epochs'
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, prefix='test')
    print(f"Test metrics: {test_metrics}")
    
    return history


def train_tabular():
    """Train a model on a tabular dataset."""
    print("Training MLP on tabular dataset...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Example: Using the Boston Housing dataset
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler
    
    # Load dataset
    data = load_boston()
    X = data.data
    y = data.target
    
    # Convert to regression task (predicting house prices)
    # For classification, you would use a different dataset and adjust the output layer
    
    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = X_train.shape[1]
    hidden_sizes = [64, 32]
    output_size = 1  # Regression task
    
    model = create_model(
        model_type=ModelType.MLP,
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=ActivationType.RELU,
        dropout=0.2,
        batch_norm=True,
        use_skip=True
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define metrics for regression
    from neuralnetworkapp.metrics import MSELoss, MAELoss, RMSELoss, R2Score
    metrics = {
        'mse': MSELoss(),
        'mae': MAELoss(),
        'rmse': RMSELoss(),
        'r2': R2Score()
    }
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=True),
        ModelCheckpoint(
            filepath='checkpoints/tabular_mlp_best.pth',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=True
        )
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        metrics=metrics,
        callbacks=callbacks,
        log_dir='logs/tabular_mlp',
        verbose=1
    )
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for this example
        num_epochs=50
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, prefix='test')
    print(f"Test metrics: {test_metrics}")
    
    return history


def train_with_custom_data():
    """Example of training with custom data using our data utilities."""
    print("Training with custom data...")
    
    # Example: Using CIFAR-10 dataset with our data utilities
    from torchvision.datasets import CIFAR10
    
    # Define transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    train_dataset = CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transforms
    )
    
    test_dataset = CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transforms
    )
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        model_type=ModelType.CNN,
        input_shape=(3, 32, 32),  # CIFAR-10 images are 32x32 with 3 channels
        conv_layers=[
            (32, 3, 1, 1, 2),  # (out_channels, kernel_size, stride, padding, pool_size)
            (64, 3, 1, 1, 2),
            (128, 3, 1, 1, 2),
        ],
        dense_sizes=[256, 128],
        output_size=10,  # 10 classes in CIFAR-10
        activation=ActivationType.RELU,
        dropout=0.3,
        batch_norm=True
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Define metrics
    metrics = get_metrics_for_task(
        task_type='multiclass',
        num_classes=10,
        average='macro'
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=True),
        ModelCheckpoint(
            filepath='checkpoints/cifar10_cnn_best.pth',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=True
        )
    ]
    
    # Create trainer with mixed precision training
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        callbacks=callbacks,
        log_dir='logs/cifar10_cnn',
        use_amp=True,
        clip_grad_norm=1.0,
        gradient_accumulation_steps=2,
        verbose=1
    )
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for this example
        num_epochs=50
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, prefix='test')
    print(f"Test metrics: {test_metrics}")
    
    return history


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run examples
    print("=== MNIST Example ===")
    mnist_history = train_mnist()
    
    print("\n=== Tabular Data Example ===")
    tabular_history = train_tabular()
    
    print("\n=== Custom Data Example ===")
    custom_history = train_with_custom_data()
    
    print("\nAll examples completed successfully!")
