"""
Loss Functions Module

This module provides various loss functions for training neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, Tuple

# Supported loss functions
LOSS_FUNCTIONS = {
    # Classification
    'cross_entropy': nn.CrossEntropyLoss,
    'bce': nn.BCELoss,
    'bce_with_logits': nn.BCEWithLogitsLoss,
    'nll': nn.NLLLoss,
    'poisson_nll': nn.PoissonNLLLoss,
    'kl_div': nn.KLDivLoss,
    'hinge_embedding': nn.HingeEmbeddingLoss,
    'multi_margin': nn.MultiMarginLoss,
    'multi_label_margin': nn.MultiLabelMarginLoss,
    'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
    'soft_margin': nn.SoftMarginLoss,
    
    # Regression
    'mse': nn.MSELoss,
    'l1': nn.L1Loss,
    'smooth_l1': nn.SmoothL1Loss,
    'huber': nn.HuberLoss,
    
    # Specialized
    'ctc': nn.CTCLoss,
    'margin_ranking': nn.MarginRankingLoss,
    'triplet_margin': nn.TripletMarginLoss,
    'triplet_margin_with_distance': nn.TripletMarginWithDistanceLoss,
}

# Default parameters for each loss function
DEFAULT_LOSS_PARAMS = {
    'cross_entropy': {'weight': None, 'ignore_index': -100, 'reduction': 'mean'},
    'bce': {'weight': None, 'reduction': 'mean'},
    'bce_with_logits': {'weight': None, 'reduction': 'mean', 'pos_weight': None},
    'nll': {'weight': None, 'ignore_index': -100, 'reduction': 'mean'},
    'poisson_nll': {'log_input': True, 'full': False, 'eps': 1e-8, 'reduction': 'mean'},
    'kl_div': {'reduction': 'mean'},
    'mse': {'reduction': 'mean'},
    'l1': {'reduction': 'mean'},
    'smooth_l1': {'reduction': 'mean', 'beta': 1.0},
    'huber': {'reduction': 'mean', 'delta': 1.0},
    'hinge_embedding': {'margin': 1.0, 'reduction': 'mean'},
    'multi_margin': {'p': 1, 'margin': 1.0, 'weight': None, 'reduction': 'mean'},
    'multi_label_margin': {'reduction': 'mean'},
    'soft_margin': {'reduction': 'mean'},
    'multi_label_soft_margin': {'weight': None, 'reduction': 'mean'},
    'margin_ranking': {'margin': 0.0, 'reduction': 'mean'},
    'triplet_margin': {'margin': 1.0, 'p': 2.0, 'eps': 1e-6, 'swap': False, 'reduction': 'mean'},
    'triplet_margin_with_distance': {
        'distance_function': None, 
        'margin': 1.0, 
        'swap': False, 
        'reduction': 'mean'
    },
    'ctc': {
        'blank': 0, 
        'reduction': 'mean', 
        'zero_infinity': False
    },
}


def get_loss_function(
    name: str, 
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """
    Get a loss function instance.
    
    Args:
        name: Name of the loss function (e.g., 'mse', 'cross_entropy')
        device: Device to move the loss function to
        **kwargs: Additional arguments to pass to the loss function
        
    Returns:
        Initialized loss function instance
    """
    name = name.lower()
    if name not in LOSS_FUNCTIONS:
        available = ', '.join(f"'{k}'" for k in LOSS_FUNCTIONS.keys())
        raise ValueError(f"Unsupported loss function: '{name}'. Available: {available}")
    
    # Get default params and update with custom ones
    params = DEFAULT_LOSS_PARAMS.get(name, {}).copy()
    params.update(kwargs)
    
    # Handle special cases
    if name == 'cross_entropy' and 'weight' in params and params['weight'] is not None:
        params['weight'] = torch.tensor(params['weight'], device=device)
    
    # Create loss function
    loss_fn = LOSS_FUNCTIONS[name](**params)
    
    # Move to device if specified
    if device is not None:
        loss_fn = loss_fn.to(device)
    
    return loss_fn


class CompositeLoss(nn.Module):
    """
    A composite loss function that combines multiple loss functions.
    """
    def __init__(
        self, 
        losses: Dict[str, Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize composite loss.
        
        Args:
            losses: Dictionary mapping loss names to their configurations
                   Example: {'mse': {'reduction': 'mean'}, 'l1': {'reduction': 'sum'}}
            weights: Dictionary mapping loss names to their weights
                    If None, all losses will have equal weight
            device: Device to move the loss functions to
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = weights or {}
        
        for name, config in losses.items():
            if not isinstance(config, dict):
                config = {}
            self.losses[name] = get_loss_function(name, device=device, **config)
            if name not in self.weights:
                self.weights[name] = 1.0
    
    def forward(self, *inputs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the composite loss.
        
        Args:
            *inputs: Inputs to pass to each loss function
            
        Returns:
            Tuple of (total_loss, individual_losses)
        """
        individual_losses = {}
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss = loss_fn(*inputs)
            weighted_loss = loss * self.weights[name]
            individual_losses[name] = loss.detach()
            total_loss = total_loss + weighted_loss
        
        return total_loss, individual_losses
