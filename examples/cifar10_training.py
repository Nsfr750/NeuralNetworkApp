"""
CIFAR-10 Training Example

This example demonstrates how to use the TrainerApp to train a CNN on the CIFAR-10 dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Import our trainer
from src.neuralnetworkapp.gui.trainer_app import ModelTrainer

# Set random seed for reproducibility
torch.manual_seed(42)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x16x16
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x8x8
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x4x4
            nn.Dropout(0.4),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def load_cifar10(batch_size=128, val_split=0.1):
    """
    Load CIFAR-10 dataset and create data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader, num_classes, classes
    """
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Just normalization for validation and test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    train_val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # Split into training and validation sets
    if val_split > 0:
        val_size = int(len(train_val_dataset) * val_split)
        train_size = len(train_val_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size]
        )
        
        # Apply test transform to validation set
        val_dataset.dataset.transform = transform_test
    else:
        train_dataset = train_val_dataset
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
    else:
        val_loader = None
    
    # Load test set
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Get class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, val_loader, test_loader, len(classes), classes

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader, num_classes, classes = load_cifar10(
        batch_size=128, val_split=0.1
    )
    
    # Create model
    print("Creating model...")
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        project_name="cifar10_classification",
        log_dir="runs"
    )
    
    # Train the model
    print("Starting training...")
    trainer.train(num_epochs=50, save_best=True)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate()
    
    # Plot training history
    print("\nGenerating training plots...")
    trainer.plot_training_history()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
