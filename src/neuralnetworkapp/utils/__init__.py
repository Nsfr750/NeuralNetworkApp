import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

def save_model(
    model: torch.nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    create_dir: bool = True
) -> None:
    """
    Save a PyTorch model along with its configuration and training state.
    
    Args:
        model: The PyTorch model to save
        path: Path to save the model to (should end with .pth or .pt)
        optimizer: Optimizer state to save
        epoch: Current training epoch
        val_loss: Validation loss at this epoch
        metrics: Dictionary of metrics to save
        config: Model configuration dictionary
        create_dir: Whether to create the directory if it doesn't exist
    """
    path = Path(path)
    
    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config or {},
        'val_loss': val_loss,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save model
    torch.save(checkpoint, path)
    
    # Save config separately as JSON
    if config is not None:
        config_path = path.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

def load_model(
    path: Union[str, Path],
    model_class: Optional[torch.nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a PyTorch model from a checkpoint.
    
    Args:
        path: Path to the saved model checkpoint
        model_class: The model class to instantiate
        config: Model configuration (if not saved in checkpoint)
        device: Device to load the model to
        strict: Whether to strictly enforce that the keys in state_dict match
                the keys returned by the model's state_dict() function
                
    Returns:
        Dictionary containing:
            - 'model': The loaded model
            - 'optimizer': The optimizer state (if available)
            - 'epoch': The epoch at which the model was saved
            - 'val_loss': Validation loss at the time of saving
            - 'metrics': Metrics at the time of saving
            - 'config': Model configuration
    """
    path = Path(path)
    
    # Load checkpoint
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    checkpoint = torch.load(path, map_location=device)
    
    # Get model config (either from checkpoint or provided)
    model_config = checkpoint.get('config', config)
    if model_config is None:
        raise ValueError("Model configuration not found in checkpoint and not provided")
    
    # Create model if class is provided
    model = None
    if model_class is not None:
        if hasattr(model_class, 'from_config'):
            model = model_class.from_config(model_config)
        else:
            model = model_class(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model = model.to(device)
    
    # Prepare result
    result = {
        'model': model,
        'optimizer': checkpoint.get('optimizer_state_dict'),
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss'),
        'metrics': checkpoint.get('metrics', {}),
        'config': model_config,
        'timestamp': checkpoint.get('timestamp')
    }
    
    return result

def plot_training_history(
    history: Dict[str, Any],
    metrics: Optional[list] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot (if None, plot all available)
        save_path: Path to save the plot to (optional)
        show: Whether to display the plot
        figsize: Figure size
    """
    if not history:
        print("No history to plot")
        return
    
    if metrics is None:
        # Get all metrics that start with 'val_'
        metrics = [m[4:] for m in history.keys() if m.startswith('val_') and m != 'val_loss']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    for metric in metrics:
        train_metric = f'train_{metric}'
        val_metric = f'val_{metric}'
        
        if train_metric in history:
            ax2.plot(epochs, history[train_metric], '--', label=f'Training {metric}')
        if val_metric in history:
            ax2.plot(epochs, history[val_metric], '--', label=f'Validation {metric}')
    
    if metrics:
        ax2.set_title('Training and Validation Metrics')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metric Value')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
