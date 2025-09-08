"""
Tests for the visualization module.
"""

import os
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import the module to test
from src.visualization import TrainingVisualizer


class TestTrainingVisualizer(unittest.TestCase):
    """Test cases for the TrainingVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name) / "test_logs"
        
        # Create a simple model and optimizer for testing
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create a visualizer
        self.visualizer = TrainingVisualizer(
            log_dir=str(self.log_dir),
            use_tensorboard=False  # Don't test TensorBoard in unit tests
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.visualizer.close()
        self.temp_dir.cleanup()
    
    def test_add_scalar(self):
        """Test adding scalar values to the visualizer."""
        # Add some scalar values
        for i in range(5):
            self.visualizer.add_scalar("train/loss", 1.0 / (i + 1), i)
            self.visualizer.add_scalar("val/accuracy", 0.8 + 0.05 * i, i)
        
        # Check that metrics were recorded
        self.assertIn("train/loss", self.visualizer.metrics)
        self.assertIn("val/accuracy", self.visualizer.metrics)
        self.assertEqual(len(self.visualizer.metrics["train/loss"]), 5)
        self.assertEqual(len(self.visualizer.metrics["val/accuracy"]), 5)
    
    def test_plot_metrics(self):
        """Test plotting metrics."""
        # Add some metrics
        for i in range(5):
            self.visualizer.add_scalar("train/loss", 1.0 / (i + 1), i)
            self.visualizer.add_scalar("val/loss", 0.8 / (i + 1), i)
        
        # Plot the metrics
        fig = self.visualizer.plot_metrics(["train/loss", "val/loss"], 
                                         title="Test Loss Curves")
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertIn("metrics", self.visualizer.figures)
    
    def test_plot_confusion_matrix(self):
        """Test plotting a confusion matrix."""
        # Create a dummy confusion matrix
        confusion_matrix = np.array([
            [5, 0, 0],
            [0, 3, 1],
            [0, 0, 6]
        ])
        
        # Plot the confusion matrix
        fig = self.visualizer.plot_confusion_matrix(
            confusion_matrix,
            class_names=["Class 0", "Class 1", "Class 2"],
            title="Test Confusion Matrix"
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertIn("confusion_matrix", self.visualizer.figures)
    
    def test_plot_learning_rate_schedule(self):
        """Test plotting a learning rate schedule."""
        # Create a simple learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=2,
            gamma=0.1
        )
        
        # Plot the learning rate schedule
        fig = self.visualizer.plot_learning_rate_schedule(
            self.optimizer,
            num_iterations=10,
            title="Test Learning Rate Schedule"
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertIn("learning_rate_schedule", self.visualizer.figures)
    
    def test_plot_embeddings(self):
        """Test plotting embeddings with t-SNE."""
        # Skip this test if scikit-learn is not available
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            self.skipTest("scikit-learn is required for this test")
        
        # Create some dummy embeddings and labels
        np.random.seed(42)
        embeddings = np.random.randn(100, 32)  # 100 samples, 32D embeddings
        labels = np.random.randint(0, 5, size=100)  # 5 classes
        
        # Plot the embeddings
        fig = self.visualizer.plot_embeddings(
            embeddings,
            labels,
            title="Test Embedding Visualization"
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertIn("embeddings", self.visualizer.figures)
    
    def test_plot_attention(self):
        """Test plotting an attention map."""
        # Create a dummy image and attention map
        image = np.random.rand(64, 64, 3)  # RGB image
        attention_map = np.random.rand(16, 16)  # Smaller attention map
        
        # Plot the attention map
        fig = self.visualizer.plot_attention(
            image,
            attention_map,
            title="Test Attention Map"
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertIn("attention_map", self.visualizer.figures)
    
    def test_save_figures(self):
        """Test saving figures to disk."""
        # Create a figure
        self.visualizer.plot_metrics(["dummy_metric"], title="Dummy Figure")
        
        # Save the figures
        output_dir = Path(self.temp_dir.name) / "saved_figures"
        self.visualizer.save_figures(output_dir=output_dir)
        
        # Check that the file was created
        self.assertTrue((output_dir / "metrics.png").exists())
    
    def test_context_manager(self):
        """Test using the visualizer as a context manager."""
        with TrainingVisualizer(log_dir=str(self.log_dir)) as viz:
            # Add a metric
            viz.add_scalar("test/metric", 0.5, 0)
            
            # Check that the metric was added
            self.assertIn("test/metric", viz.metrics)
        
        # The visualizer should be closed now
        self.assertTrue(True)  # Just check that we got here without errors


if __name__ == "__main__":
    unittest.main()
