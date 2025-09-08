"""
Neural Network App - A flexible neural network builder and trainer with visualization capabilities.

This package provides tools for creating, training, and evaluating neural networks with a focus on
simplicity and flexibility. It includes modules for data loading, model building, training, and visualization.

Main components:
    - models: Neural network model definitions and architectures
    - data: Data loading and preprocessing utilities
    - training: Training loops and optimization
    - utils: Helper functions and utilities
    - visualization: Tools for visualizing models and training progress

Example usage:
    >>> from neuralnetworkapp import NeuralNetwork, Trainer, load_tabular_data
    >>> 
    >>> # Load data
    >>> X, y = load_tabular_data('data.csv')
    >>> 
    >>> # Create model
    >>> model = NeuralNetwork(input_size=X.shape[1], hidden_sizes=[128, 64], output_size=10)
    >>> 
    >>> # Create trainer and train
    >>> trainer = Trainer(model, optimizer='adam', learning_rate=0.001)
    >>> history = trainer.fit(X, y, epochs=10, batch_size=32)
"""

# Import main classes and functions to make them available at the package level
from .models import NeuralNetwork, create_model
from .training import Trainer
from .data import load_tabular_data, create_data_loaders, TabularDataset
from .utils import (
    save_model, load_model, plot_training_history,
    save_config, load_config, count_parameters, set_seed
)

# Version information
from .version import __version__, version_info

__all__ = [
    # Models
    'NeuralNetwork', 'create_model',
    
    # Training
    'Trainer',
    
    # Data
    'load_tabular_data', 'create_data_loaders', 'TabularDataset',
    
    # Utils
    'save_model', 'load_model', 'plot_training_history',
    'save_config', 'load_config', 'count_parameters', 'set_seed',
    
    # Version
    '__version__', 'version_info'
]
