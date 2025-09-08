"""
Network Builder Demo

This script demonstrates how to use the NetworkBuilder and TrainingVisualizer
classes to create, train, and visualize a neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import our custom modules
from neuralnetworkapp.builderimport NetworkBuilder, LayerType
from neuralnetworkapp.visualizationimport TrainingVisualizer
from help_docs import help_system

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_dummy_data(num_samples=1000, input_shape=(3, 32, 32), num_classes=10):
    """Create dummy training and validation data."""
    # Generate random data
    x_train = torch.randn(num_samples, *input_shape)
    y_train = torch.randint(0, num_classes, (num_samples,))
    
    # Create a validation set
    x_val = torch.randn(num_samples // 5, *input_shape)
    y_val = torch.randint(0, num_classes, (num_samples // 5,))
    
    # Create datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train a model with the given data loaders."""
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create visualizer
    visualizer = TrainingVisualizer(log_dir='runs/network_builder_demo')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log metrics
            global_step = epoch * len(train_loader) + batch_idx
            visualizer.add_scalar('train/loss', loss.item(), global_step)
            visualizer.add_scalar('train/accuracy', 100. * correct / total, global_step)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | '
                      f'Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100. * correct / total:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Log validation metrics
        val_accuracy = 100. * correct / total
        visualizer.add_scalar('val/loss', val_loss / len(val_loader), (epoch + 1) * len(train_loader))
        visualizer.add_scalar('val/accuracy', val_accuracy, (epoch + 1) * len(train_loader))
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training Loss: {train_loss / len(train_loader):.4f} | '
              f'Validation Loss: {val_loss / len(val_loader):.4f} | '
              f'Validation Acc: {val_accuracy:.2f}%\n')
        
        # Update plots
        visualizer.plot_metrics()
    
    # Close the visualizer
    visualizer.close()
    
    return model


def main():
    """Main function to demonstrate the NetworkBuilder and TrainingVisualizer."""
    print("Network Builder Demo")
    print("=" * 80)
    
    # Create a network using NetworkBuilder
    print("\nCreating a CNN with NetworkBuilder...")
    builder = NetworkBuilder(input_shape=(3, 32, 32))
    
    # Add layers to the network
    (builder
     .add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)
     .add_layer(LayerType.RELU)
     .add_layer(LayerType.MAXPOOL2D, kernel_size=2, stride=2)
     .add_layer(LayerType.CONV2D, out_channels=64, kernel_size=3, padding=1)
     .add_layer(LayerType.RELU)
     .add_layer(LayerType.MAXPOOL2D, kernel_size=2, stride=2)
     .add_layer(LayerType.FLATTEN)
     .add_layer(LayerType.LINEAR, out_features=128)
     .add_layer(LayerType.RELU)
     .add_layer(LayerType.LINEAR, out_features=10))
    
    # Print network summary
    print("\nNetwork Summary:")
    print("-" * 80)
    builder.summary()
    
    # Visualize the network architecture
    print("\nGenerating network visualization...")
    builder.visualize("network_architecture", format="png", show_shapes=True)
    
    # Build the PyTorch model
    model = builder.build().to(device)
    
    # Create dummy data
    print("\nCreating dummy training data...")
    train_loader, val_loader = create_dummy_data()
    
    # Train the model
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, num_epochs=5)
    
    print("\nTraining complete!")
    
    # Show help for NetworkBuilder
    print("\nTo learn more about NetworkBuilder, run:")
    print("  from help_docs import help")
    print("  help('NetworkBuilder')")


if __name__ == "__main__":
    main()
