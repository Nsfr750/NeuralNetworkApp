"""
Optimization Module

This module provides various optimization algorithms and learning rate schedulers
for training neural networks.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, 
    ReduceLROnPlateau, CosineAnnealingLR
)
from typing import Dict, Any, Optional, Union, List, Type

# Supported optimizers
OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta,
}

# Supported learning rate schedulers
SCHEDULERS = {
    'steplr': StepLR,
    'multisteplr': MultiStepLR,
    'exponential': ExponentialLR,
    'reduce_on_plateau': ReduceLROnPlateau,
    'cosine': CosineAnnealingLR,
}

# Default optimizer parameters
DEFAULT_OPTIMIZER_PARAMS = {
    'sgd': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
    'adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0},
    'adamw': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01},
    'rmsprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0},
    'adagrad': {'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0},
    'adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 0},
}

# Default scheduler parameters
DEFAULT_SCHEDULER_PARAMS = {
    'steplr': {'step_size': 30, 'gamma': 0.1},
    'multisteplr': {'milestones': [30, 80], 'gamma': 0.1},
    'exponential': {'gamma': 0.95},
    'reduce_on_plateau': {
        'mode': 'min', 'factor': 0.1, 'patience': 10, 
        'verbose': True, 'threshold': 0.0001, 'threshold_mode': 'rel',
        'cooldown': 0, 'min_lr': 0, 'eps': 1e-8
    },
    'cosine': {'T_max': 50, 'eta_min': 0, 'last_epoch': -1},
}


def get_optimizer(
    name: str, 
    model_params, 
    custom_params: Optional[Dict[str, Any]] = None
) -> torch.optim.Optimizer:
    """
    Get an optimizer instance.
    
    Args:
        name: Name of the optimizer (e.g., 'adam', 'sgd')
        model_params: Model parameters to optimize
        custom_params: Custom parameters for the optimizer
        
    Returns:
        Initialized optimizer instance
    """
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {name}. Available: {list(OPTIMIZERS.keys())}")
    
    # Get default params and update with custom ones
    params = DEFAULT_OPTIMIZER_PARAMS.get(name, {}).copy()
    if custom_params:
        params.update(custom_params)
    
    return OPTIMIZERS[name](model_params, **params)


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    custom_params: Optional[Dict[str, Any]] = None
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get a learning rate scheduler instance.
    
    Args:
        name: Name of the scheduler (e.g., 'steplr', 'reduce_on_plateau')
        optimizer: The optimizer whose learning rate to schedule
        custom_params: Custom parameters for the scheduler
        
    Returns:
        Initialized scheduler instance or None if name is empty/None
    """
    if not name:
        return None
        
    name = name.lower()
    if name not in SCHEDULERS:
        raise ValueError(f"Unsupported scheduler: {name}. Available: {list(SCHEDULERS.keys())}")
    
    # Get default params and update with custom ones
    params = DEFAULT_SCHEDULER_PARAMS.get(name, {}).copy()
    if custom_params:
        params.update(custom_params)
    
    return SCHEDULERS[name](optimizer, **params)
