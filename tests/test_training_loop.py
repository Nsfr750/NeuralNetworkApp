"""
Tests for training loop and metrics.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from neuralnetworkapp.builderimport NetworkBuilder, LayerType
from neuralnetworkapp.visualizationimport TrainingVisualizer

class TestTrainingLoop(unittest.TestCase):
    """Test cases for training loop and metrics."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create a simple dataset
        torch.manual_seed(42)
        self.x_train = torch.randn(100, 3, 32, 32)
        self.y_train = torch.randint(0, 10, (100,))
        self.x_val = torch.randn(20, 3, 32, 32)
        self.y_val = torch.randint(0, 10, (20,))
        
        # Create data loaders
        self.train_loader = DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=16,
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(self.x_val, self.y_val),
            batch_size=16,
            shuffle=False
        )
        
        # Create a simple model
        self.model = NetworkBuilder(input_shape=(3, 32, 32)) \
            .add_layer(LayerType.CONV2D, out_channels=16, kernel_size=3, padding=1) \
            .add_layer(LayerType.RELU) \
            .add_layer(LayerType.FLATTEN) \
            .add_layer(LayerType.LINEAR, out_features=10) \
            .build()
        
        # Set up loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def test_training_step(self):
        """Test a single training step."""
        self.model.train()
        x_batch = torch.randn(8, 3, 32, 32)
        y_batch = torch.randint(0, 10, (8,))
        
        # Forward pass
        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Verify outputs
        self.assertEqual(outputs.shape, (8, 10))
        self.assertIsInstance(loss.item(), float)
    
    def test_validation_loop(self):
        """Test the validation loop."""
        self.model.eval()
        total = 0
        correct = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_loss /= len(self.val_loader.dataset)
        val_acc = 100.0 * correct / total
        
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 100.0)
        self.assertIsInstance(val_loss, float)
    
    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        visualizer = TrainingVisualizer(use_tensorboard=False)
        
        # Simulate training for 2 epochs
        for epoch in range(2):
            self.model.train()
            for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
                # Forward pass
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y_batch).sum().item()
                acc = 100.0 * correct / y_batch.size(0)
                
                # Log metrics
                global_step = epoch * len(self.train_loader) + batch_idx
                visualizer.add_scalar('train/loss', loss.item(), global_step)
                visualizer.add_scalar('train/accuracy', acc, global_step)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    outputs = self.model(x_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            val_loss /= len(self.val_loader.dataset)
            val_acc = 100.0 * correct / total
            
            # Log validation metrics
            visualizer.add_scalar('val/loss', val_loss, (epoch + 1) * len(self.train_loader))
            visualizer.add_scalar('val/accuracy', val_acc, (epoch + 1) * len(self.train_loader))
        
        # Verify metrics were tracked
        self.assertIn('train/loss', visualizer.metrics)
        self.assertIn('train/accuracy', visualizer.metrics)
        self.assertIn('val/loss', visualizer.metrics)
        self.assertIn('val/accuracy', visualizer.metrics)
        
        # Clean up
        visualizer.close()
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduling."""
        # Set up scheduler
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        
        # Initial learning rate
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Train for one epoch
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            outputs = self.model(x_batch)
            loss = self.criterion(outputs, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Step the scheduler
        scheduler.step()
        
        # Check if learning rate was reduced
        new_lr = self.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(new_lr, initial_lr * 0.1)


if __name__ == "__main__":
    unittest.main()
