"""
Model Utilities

This module provides utility functions for working with PyTorch models,
including visualization and analysis tools.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.hooks import RemovableHandle
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import matplotlib.pyplot as plt
from PIL import Image
import io


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count the number of trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


def model_summary(
    model: nn.Module, 
    input_size: Tuple[int, ...],
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtypes: Optional[List[torch.dtype]] = None
) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        device: Device to run the model on
        dtypes: List of dtypes to test (useful for mixed precision training)
    """
    from torchsummary import summary
    
    if dtypes is None:
        dtypes = [torch.float32]
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    
    for dtype in dtypes:
        print(f"\n{'='*50}")
        print(f"Summary (dtype={dtype}):")
        print(f"{'='*50}")
        
        # Create a temporary model for the summary
        temp_model = model.to(device).to(dtype)
        
        # Get parameter counts
        trainable_params, total_params = count_parameters(temp_model)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Print model summary
        try:
            summary(
                temp_model, 
                input_size=input_size[1:],  # Remove batch size for summary
                device=device,
                dtypes=[dtype]
            )
        except Exception as e:
            print(f"Could not generate full summary: {e}")
            print("Falling back to simple parameter count...")
            print(temp_model)


def visualize_feature_maps(
    model: nn.Module, 
    input_tensor: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
    max_maps: int = 16,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Visualize feature maps from convolutional layers.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor of shape (1, C, H, W)
        layer_indices: Indices of layers to visualize (None for all conv layers)
        max_maps: Maximum number of feature maps to display per layer
        figsize: Figure size
    """
    # Ensure model is in evaluation mode
    was_training = model.training
    model.eval()
    
    # Register hooks to capture feature maps
    activations: Dict[str, torch.Tensor] = {}
    handles: List[RemovableHandle] = []
    
    def get_activation(name: str) -> Callable:
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for all convolutional layers or specified layers
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Conv2d):
            if layer_indices is None or i in layer_indices:
                handles.append(layer.register_forward_hook(get_activation(f"{i}_{name}")))
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor.unsqueeze(0).to(next(model.parameters()).device))
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Restore model's original training mode
    model.train(was_training)
    
    # Visualize feature maps
    for name, activation in activations.items():
        # Get the feature maps
        feature_maps = activation[0]  # Get first item in batch
        num_maps = min(feature_maps.size(0), max_maps)
        
        # Create a grid of feature maps
        rows = int(np.ceil(np.sqrt(num_maps)))
        cols = int(np.ceil(num_maps / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f"Feature maps: {name}", fontsize=12)
        
        for i in range(rows * cols):
            if i < num_maps:
                ax = axes.flat[i]
                ax.imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
                ax.axis('off')
            else:
                axes.flat[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def plot_training_history(history: Dict[str, List[float]], figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training history
        figsize: Figure size
    """
    # Extract metrics
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
    
    # Plot accuracy if available
    if 'train_accuracy' in history and 'val_accuracy' in history:
        ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()


def log_model_to_tensorboard(
    writer: SummaryWriter,
    model: nn.Module,
    input_tensor: torch.Tensor,
    global_step: int = 0,
    tag: str = "model"
) -> None:
    """
    Log model graph and parameters to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        input_tensor: Example input tensor
        global_step: Global step for logging
        tag: Tag for the model in TensorBoard
    """
    # Log model graph
    writer.add_graph(model, input_tensor, verbose=False)
    
    # Log model parameters
    for name, param in model.named_parameters():
        writer.add_histogram(f"{tag}/{name}", param.data, global_step)
        if param.grad is not None:
            writer.add_histogram(f"{tag}/{name}_grad", param.grad, global_step)


def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to save the checkpoint
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        metrics: Dictionary of metrics to save
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics or {},
        'is_best': is_best
    }
    
    torch.save(checkpoint, path)
    if is_best:
        best_path = path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to the checkpoint
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load the model on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Prepare return dictionary
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'is_best': checkpoint.get('is_best', False)
    }
    
    return model, checkpoint_info
