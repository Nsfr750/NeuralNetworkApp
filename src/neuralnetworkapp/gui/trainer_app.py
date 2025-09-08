"""
Neural Network Trainer Application

A user-friendly interface for training and evaluating neural network models
with real-time visualization of metrics and predictions.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# Import our visualization components
from ..visualization import TrainingVisualizer

class TrainerApp:
    """
    A user-friendly interface for training and evaluating neural network models.
    
    This class provides a high-level API for common training workflows with
    built-in visualization of metrics, model architecture, and predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "runs/experiment",
        project_name: str = "neural_network_project"
    ) -> None:
        """
        Initialize the TrainerApp.
        
        Args:
            model: The neural network model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for test data
            criterion: Loss function
            optimizer: Optimization algorithm
            scheduler: Learning rate scheduler
            device: Device to run the model on ('cuda' or 'cpu')
            log_dir: Directory to save logs and checkpoints
            project_name: Name of the project for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
        # Set up loss function, optimizer, and scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler
        
        # Set up logging and visualization
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.visualizer = TrainingVisualizer(log_dir=str(self.log_dir / "visualization"))
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Save configuration
        self._save_config(project_name)
    
    def _save_config(self, project_name: str) -> None:
        """Save the training configuration to a JSON file."""
        config = {
            'project_name': project_name,
            'model': str(self.model),
            'criterion': str(self.criterion),
            'optimizer': str(self.optimizer),
            'scheduler': str(self.scheduler) if self.scheduler else None,
            'device': str(self.device),
            'batch_size': self.train_loader.batch_size if self.train_loader else None,
            'num_epochs': 0,  # Will be updated during training
            'best_val_loss': self.best_val_loss,
            'log_dir': str(self.log_dir.absolute())
        }
        
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_one_epoch(self) -> Tuple[float, float]:
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item()
            
            # Log metrics
            step = self.epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), step)
            self.visualizer.add_scalar('train/loss', loss.item(), step)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: {self.epoch + 1}, Batch: {batch_idx + 1}, '
                      f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validate the model on the given data loader."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def fit(self, num_epochs: int = 10, save_best: bool = True) -> None:
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            save_best: Whether to save the best model based on validation loss
        """
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_loss, train_acc = self.train_one_epoch()
            
            # Validate
            if self.val_loader is not None:
                val_loss, val_acc = self.validate(self.val_loader)
                
                # Update learning rate scheduler if available
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Save best model
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                
                # Log metrics
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/accuracy', val_acc, epoch)
                self.visualizer.add_scalar('val/loss', val_loss, epoch)
                self.visualizer.add_scalar('val/accuracy', val_acc, epoch)
                
                # Update history
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Log training metrics
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('train/accuracy', train_acc, epoch)
            self.visualizer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.visualizer.add_scalar('train/accuracy', train_acc, epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Print epoch summary
            if self.val_loader is not None:
                print(f'Epoch: {epoch + 1}/{num_epochs}, '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            else:
                print(f'Epoch: {epoch + 1}/{num_epochs}, '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Save checkpoint
            self.save_checkpoint('latest_checkpoint.pth')
            
            # Plot metrics
            self.plot_metrics()
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on the given data loader.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        loss, accuracy = self.validate(loader)
        
        # Additional metrics can be added here
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'error_rate': 100 - accuracy
        }
        
        # Log metrics
        self.writer.add_scalar('test/loss', loss, self.epoch)
        self.writer.add_scalar('test/accuracy', accuracy, self.epoch)
        self.visualizer.add_scalar('test/loss', loss, self.epoch)
        self.visualizer.add_scalar('test/accuracy', accuracy, self.epoch)
        
        # Print evaluation results
        print(f'Test Loss: {loss:.4f}, Test Acc: {accuracy:.2f}%')
        
        return metrics
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on the given input tensor.
        
        Args:
            inputs: Input tensor of shape (batch_size, *input_shape)
            
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            return predicted.cpu()
    
    def plot_metrics(self) -> None:
        """Plot training and validation metrics."""
        # Plot loss
        self.visualizer.plot_metrics(
            metrics=['train/loss', 'val/loss'],
            title='Training and Validation Loss',
            ylabel='Loss'
        )
        
        # Plot accuracy
        self.visualizer.plot_metrics(
            metrics=['train/accuracy', 'val/accuracy'],
            title='Training and Validation Accuracy',
            ylabel='Accuracy (%)'
        )
        
        # Save figures
        self.visualizer.save_figures()
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, self.log_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f'Loaded checkpoint from epoch {self.epoch}')
        print(f'Best validation loss: {self.best_val_loss:.4f}')


class ModelTrainer:
    """
    A simplified interface for training and evaluating models with visualization.
    
    This class provides a more user-friendly API on top of TrainerApp,
    with sensible defaults and automatic visualization setup.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        project_name: str = "neural_network_project",
        log_dir: str = "runs"
    ) -> None:
        """
        Initialize the ModelTrainer.
        
        Args:
            model: The neural network model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for test data
            criterion: Loss function (default: CrossEntropyLoss for classification)
            optimizer: Optimization algorithm (default: Adam)
            scheduler: Learning rate scheduler
            device: Device to run the model on ('cuda' or 'cpu')
            project_name: Name of the project for logging
            log_dir: Base directory for saving logs and checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        
        # Set up loss function, optimizer, and scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler
        
        # Set up logging directory
        self.project_name = project_name
        self.log_dir = Path(log_dir) / project_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the trainer
        self.trainer = TrainerApp(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            log_dir=str(self.log_dir),
            project_name=project_name
        )
        
        print(f"Training initialized. Logs will be saved to: {self.log_dir}")
    
    def train(self, num_epochs: int = 10, save_best: bool = True) -> None:
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            save_best: Whether to save the best model based on validation loss
        """
        print(f"Starting training for {num_epochs} epochs...")
        self.trainer.fit(num_epochs=num_epochs, save_best=save_best)
        print("Training completed!")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.test_loader is None:
            raise ValueError("No test loader provided for evaluation.")
        
        print("Evaluating on test set...")
        metrics = self.trainer.evaluate(self.test_loader)
        return metrics
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on the given input tensor.
        
        Args:
            inputs: Input tensor of shape (batch_size, *input_shape)
            
        Returns:
            Model predictions
        """
        return self.trainer.predict(inputs)
    
    def save_model(self, filename: str = "model.pth") -> None:
        """
        Save the model to a file.
        
        Args:
            filename: Name of the file to save the model to
        """
        save_path = self.log_dir / filename
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model file
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self) -> None:
        """Plot the training and validation metrics."""
        self.trainer.plot_metrics()
        
        # Save the figures to the log directory
        self.trainer.visualizer.save_figures()
        print(f"Training history plots saved to {self.log_dir / 'visualization'}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    project_name: str = "neural_network_project",
    log_dir: str = "runs"
) -> ModelTrainer:
    """
    Create a ModelTrainer instance with the given configuration.
    
    This is a convenience function for quickly setting up a trainer with sensible defaults.
    """
    return ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        project_name=project_name,
        log_dir=log_dir
    )
