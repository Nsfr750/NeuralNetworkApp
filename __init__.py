"""
Neural Network Creator - A PySide6 application for creating and managing neural networks
© Copyright 2025 Nsfr750 - All rights reserved
"""

__version__ = '0.1.0'

# Import key components to make them easily accessible
from .model import NeuralNetwork, create_model
from .trainer import Trainer
from .data import TabularDataset, load_tabular_data, create_data_loaders, load_image_data
from .utils import (
    save_model, load_model, plot_training_history,
    save_config, load_config, count_parameters, set_seed
)

__all__ = [
    'NeuralNetwork',
    'create_model',
    'Trainer',
    'TabularDataset',
    'load_tabular_data',
    'create_data_loaders',
    'load_image_data',
    'save_model',
    'load_model',
    'plot_training_history',
    'save_config',
    'load_config',
    'count_parameters',
    'set_seed'
]
