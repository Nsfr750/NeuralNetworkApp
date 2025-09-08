"""
Neural Network Application

A flexible framework for building, training, and evaluating neural networks with PyTorch.
"""

__version__ = "0.1.0"
__author__ = "Nsfr750"
__email__ = "nsfr750@yandex.com"
__license__ = "GPL-3.0-or-later"
__copyright__ = "© Copyright 2025 Nsfr750 - All rights reserved"

# Import key components for easier access
from .network_builder import NetworkBuilder, LayerType

__all__ = [
    'NetworkBuilder',
    'LayerType',
]
