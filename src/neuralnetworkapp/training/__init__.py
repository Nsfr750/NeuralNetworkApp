"""
Neural Network Trainer Module

This module provides a flexible and extensible training loop for PyTorch models,
with support for various training strategies, callbacks, and evaluation metrics.
"""

import os
import json
import time
import copy
import logging
from pathlib import Path
from datetime import datetime
from typing import (
    Dict, List, Tuple, Optional, Callable, Any, Union, 
    Type, TypeVar, Generic, Sequence, cast
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

# Import our custom modules
from neuralnetworkapp.metrics import Metric, MetricCollection, get_metrics_for_task
from neuralnetworkapp.data.utils import get_cross_validation_splits

# Type variables for type hints
T = TypeVar('T', bound=nn.Module)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve after a given patience."""
    
    def __init__(
        self, 
        patience: int = 7, 
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping after the metric stops improving
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the epoch with the best value
                                 of the monitored quantity
            verbose: If True, prints a message for each validation loss improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = -1
    
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model: Model to track
            epoch: Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        score = -val_loss  # We want to maximize the negative loss (minimize the loss)
        
        if self.best_score is None:
            self.best_score = score
            self._save_weights(model, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        logger.info(f'Restoring model weights from epoch {self.best_epoch}')
                    model.load_state_dict(self.best_weights)
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self._save_weights(model, epoch)
            self.counter = 0
            
        return self.early_stop
    
    def _save_weights(self, model: nn.Module, epoch: int) -> None:
        """Save model weights and update best epoch."""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        if self.verbose:
            logger.info(f'Validation loss improved. Best score: {self.best_score:.6f}')


class ModelCheckpoint:
    """Save the model after every epoch."""
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            filepath: Path to save the model file. Can contain named formatting options
                     that will be filled with values from the training state.
                     Example: 'checkpoints/model_epoch_{epoch:03d}_loss_{val_loss:.4f}.pth'
            monitor: Quantity to monitor (e.g., 'val_loss', 'val_accuracy')
            mode: One of {'min', 'max'}. In 'min' mode, the callback will save the model
                  when the monitored quantity decreases, and in 'max' mode it will save
                  when the monitored quantity increases.
            save_best_only: If True, only save when the monitored quantity improves
            save_weights_only: If True, only the model's weights will be saved
            verbose: Verbosity mode
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = -1
        
        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, model: nn.Module, epoch: int, logs: Dict[str, float]) -> None:
        """Save the model if conditions are met."""
        if self.monitor not in logs:
            logger.warning(f"Can save best model only with {self.monitor} available, skipping.")
            return
            
        current = logs[self.monitor]
        
        # Check if we should save the model
        if (self.mode == 'min' and current < self.best) or \
           (self.mode == 'max' and current > self.best) or \
           not self.save_best_only:
            
            if self.save_best_only:
                self.best = current
                self.best_epoch = epoch
                
                if self.verbose:
                    logger.info(f"Epoch {epoch:03d}: {self.monitor} improved to {current:.6f}, saving model...")
            
            # Format the filepath with the current metrics
            filepath = str(self.filepath).format(epoch=epoch, **logs)
            
            # Save the model
            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                torch.save(model, filepath)
                
            if self.verbose and not self.save_best_only:
                logger.info(f"Epoch {epoch:03d}: saving model to {filepath}")


class Trainer:
    """A flexible trainer for PyTorch models with support for various training strategies."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[Union[nn.Module, Callable]] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        callbacks: Optional[List[Callable]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        use_amp: bool = False,
        clip_grad_norm: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
        verbose: int = 1
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer (default: Adam with lr=1e-3)
            scheduler: Learning rate scheduler
            device: Device to train on ('cuda', 'mps', 'cpu' or None for auto-detection)
            metrics: Dictionary of metrics to track during training
            callbacks: List of callback functions to call during training
            log_dir: Directory to save logs and checkpoints
            use_amp: Whether to use Automatic Mixed Precision (AMP)
            clip_grad_norm: If not None, clips gradient norm to this value
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
        """
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.device = self._get_device(device)
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.log_dir = Path(log_dir) if log_dir else Path('logs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.use_amp = use_amp
        self.clip_grad_norm = clip_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.verbose = verbose
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Set up logging
        self._setup_logging()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')  # For tracking best model
    
    def _get_device(self, device: Optional[Union[str, torch.device]] = None) -> torch.device:
        """Get the device to use for training."""
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _setup_logging(self) -> None:
        """Set up file logging."""
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicate logs
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        if self.verbose > 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics as a string."""
        return ' - '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    
    def _train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int,
        max_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        
        # Initialize metrics
        metrics = {f'train_{name}': 0.0 for name in self.metrics}
        metrics['train_loss'] = 0.0
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        # Training loop
        total_steps = min(len(train_loader), max_steps) if max_steps else len(train_loader)
        processed_samples = 0
        
        # Progress bar
        if self.verbose == 1:
            from tqdm import tqdm
            pbar = tqdm(total=total_steps, desc=f'Epoch {epoch + 1}', unit='batch')
        
        for step, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / self.gradient_accumulation_steps
            
            # Backward pass and optimize
            self.scaler.scale(loss).backward()
            
            # Update metrics
            metrics['train_loss'] += loss.item() * self.gradient_accumulation_steps
            
            for name, metric in self.metrics.items():
                metric.update(outputs.detach(), targets.detach())
                metrics[f'train_{name}'] = metric.compute()
            
            # Gradient accumulation: step only after accumulating enough gradients
            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None and not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                # Update global step
                self.global_step += 1
            
            # Update progress bar
            processed_samples += inputs.size(0)
            
            if self.verbose == 1:
                pbar.update(1)
                pbar.set_postfix(loss=metrics['train_loss'] / (step + 1))
            
            # Early stopping if max_steps is reached
            if max_steps is not None and (step + 1) >= max_steps:
                break
        
        # Close progress bar
        if self.verbose == 1:
            pbar.close()
        
        # Calculate average metrics
        metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        
        # Log metrics
        if self.verbose > 1:
            logger.info(f'Epoch {epoch + 1} - Train: {self._format_metrics(metrics)}')
        
        return metrics
    
    def _evaluate(
        self, 
        eval_loader: DataLoader,
        prefix: str = 'val'
    ) -> Dict[str, float]:
        """Evaluate the model on the given data loader."""
        self.model.eval()
        
        # Initialize metrics
        metrics = {f'{prefix}_loss': 0.0}
        for name in self.metrics:
            metrics[f'{prefix}_{name}'] = 0.0
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        # Evaluation loop
        with torch.no_grad():
            for inputs, targets in eval_loader:
                # Move data to device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Update metrics
                metrics[f'{prefix}_loss'] += loss.item() * inputs.size(0)
                
                for name, metric in self.metrics.items():
                    metric.update(outputs, targets)
                    metrics[f'{prefix}_{name}'] = metric.compute() * inputs.size(0)
        
        # Calculate average metrics
        metrics = {k: v / len(eval_loader.dataset) for k, v in metrics.items()}
        
        # Log metrics
        if self.verbose > 1:
            logger.info(f'Evaluation - {prefix}: {self._format_metrics(metrics)}')
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        max_steps: Optional[int] = None,
        initial_epoch: int = 0,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train for
            max_steps: Maximum number of training steps per epoch
            initial_epoch: Epoch at which to start training
            callbacks: List of callbacks to call during training
            
        Returns:
            Dictionary with training history
        """
        # Combine callbacks
        all_callbacks = self.callbacks + (callbacks or [])
        
        # Initialize history
        history = {'epoch': []}
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(initial_epoch, initial_epoch + num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, epoch, max_steps)
            
            # Evaluate on validation set
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader, prefix='val')
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                else:
                    self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics['epoch'] = epoch
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            # Update history
            for key, value in metrics.items():
                if key not in history:
                    history[key] = []
                history[key].append(value)
            
            # Log metrics
            if self.verbose > 0:
                epoch_time = time.time() - epoch_start_time
                log_str = f'Epoch {epoch + 1}/{initial_epoch + num_epochs} - {epoch_time:.1f}s - '
                log_str += f'lr: {metrics["lr"]:.2e} - '
                log_str += self._format_metrics(metrics)
                logger.info(log_str)
            
            # Write to tensorboard
            for key, value in metrics.items():
                if key != 'epoch':
                    self.writer.add_scalar(key, value, epoch)
            
            # Call callbacks
            for callback in all_callbacks:
                if hasattr(callback, '__call__'):
                    callback(self.model, epoch, metrics)
            
            # Early stopping
            if hasattr(self, 'early_stop') and self.early_stop:
                logger.info('Early stopping triggered')
                break
        
        # Close tensorboard writer
        self.writer.close()
        
        # Log total training time
        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s')
        
        return history
    
    def evaluate(
        self, 
        test_loader: DataLoader, 
        prefix: str = 'test',
        return_predictions: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]:
        """
        Evaluate the model on the given data loader.
        
        Args:
            test_loader: DataLoader for test data
            prefix: Prefix for metric names
            return_predictions: If True, also return model predictions and targets
            
        Returns:
            Dictionary with evaluation metrics, and optionally predictions and targets
        """
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        # Initialize metrics
        metrics = {f'{prefix}_loss': 0.0}
        for name in self.metrics:
            metrics[f'{prefix}_{name}'] = 0.0
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        # Evaluation loop
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move data to device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Store outputs and targets
                if return_predictions:
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())
                
                # Update metrics
                metrics[f'{prefix}_loss'] += loss.item() * inputs.size(0)
                
                for name, metric in self.metrics.items():
                    metric.update(outputs, targets)
                    metrics[f'{prefix}_{name}'] = metric.compute() * inputs.size(0)
        
        # Calculate average metrics
        metrics = {k: v / len(test_loader.dataset) for k, v in metrics.items()}
        
        # Log metrics
        logger.info(f'Evaluation - {prefix}: {self._format_metrics(metrics)}')
        
        if return_predictions:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            return metrics, all_outputs, all_targets
        
        return metrics
    
    def predict(
        self, 
        data_loader: DataLoader,
        return_targets: bool = False,
        return_probs: bool = False,
        to_numpy: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate predictions for the input data.
        
        Args:
            data_loader: DataLoader for input data
            return_targets: If True, also return the target values
            return_probs: If True, return raw probabilities/logits instead of class predictions
            to_numpy: If True, convert outputs to numpy arrays
            
        Returns:
            Model predictions, and optionally targets
        """
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                # Move data to device
                inputs = inputs.to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                
                # Store outputs and targets
                all_outputs.append(outputs.cpu())
                if return_targets:
                    all_targets.append(targets.cpu())
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # Convert to class predictions if needed
        if not return_probs:
            if all_outputs.ndim > 1 and all_outputs.size(1) > 1:
                all_outputs = torch.argmax(all_outputs, dim=1)
            else:
                all_outputs = (all_outputs > 0).long().squeeze()
        
        # Convert to numpy if requested
        if to_numpy:
            all_outputs = all_outputs.numpy()
            if return_targets:
                all_targets = torch.cat(all_targets, dim=0).numpy()
        
        if return_targets:
            return all_outputs, all_targets
        
        return all_outputs
    
    def save_checkpoint(
        self, 
        filepath: Union[str, Path],
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        **kwargs
    ) -> None:
        """
        Save the model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            save_optimizer: Whether to save the optimizer state
            save_scheduler: Whether to save the scheduler state
            **kwargs: Additional items to save in the checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if save_optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if save_scheduler and self.scheduler is not None else None,
            'best_metric': self.best_metric,
            **kwargs
        }
        
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f'Checkpoint saved to {filepath}')
    
    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        **kwargs
    ) -> 'Trainer':
        """
        Load a model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
            model: Model to load the weights into
            optimizer: Optimizer to load the state into
            scheduler: Scheduler to load the state into
            **kwargs: Additional arguments to pass to the Trainer constructor
            
        Returns:
            Trainer instance with the loaded state
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create trainer
        trainer = cls(model=model, optimizer=optimizer, scheduler=scheduler, **kwargs)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        trainer.epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_metric = checkpoint.get('best_metric', float('inf'))
        
        logger.info(f'Checkpoint loaded from {filepath} (epoch {checkpoint["epoch"]})')
        
        return trainer


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    num_epochs: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    metrics: Optional[Dict[str, Metric]] = None,
    callbacks: Optional[List[Callable]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    use_amp: bool = False,
    clip_grad_norm: Optional[float] = None,
    gradient_accumulation_steps: int = 1,
    verbose: int = 1
) -> Dict[str, List[float]]:
    """
    Train a PyTorch model with a simple interface.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        criterion: Loss function (default: CrossEntropyLoss for classification, MSELoss for regression)
        optimizer: Optimizer (default: Adam with lr=1e-3)
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        device: Device to train on ('cuda', 'mps', 'cpu' or None for auto-detection)
        metrics: Dictionary of metrics to track during training
        callbacks: List of callback functions to call during training
        log_dir: Directory to save logs and checkpoints
        use_amp: Whether to use Automatic Mixed Precision (AMP)
        clip_grad_norm: If not None, clips gradient norm to this value
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights
        verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
        
    Returns:
        Dictionary with training history
    """
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        metrics=metrics,
        callbacks=callbacks,
        log_dir=log_dir,
        use_amp=use_amp,
        clip_grad_norm=clip_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        verbose=verbose
    )
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )
    
    return history
class Trainer:
    """
    Handles training and evaluation of a PyTorch model.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        loss_fn: str = 'cross_entropy',
        metrics: List[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The PyTorch model to train
            device: Device to train on (default: cuda if available, else cpu)
            optimizer: Name of optimizer to use (adam, sgd, rmsprop)
            learning_rate: Learning rate for the optimizer
            weight_decay: L2 regularization parameter
            loss_fn: Loss function to use (cross_entropy, mse, l1)
            metrics: List of metrics to track during training/validation
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer_name = optimizer.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = self._get_optimizer(self.optimizer_name, learning_rate, weight_decay)
        
        # Set up loss function
        self.loss_fn_name = loss_fn.lower()
        self.criterion = self._get_loss_fn(self.loss_fn_name)
        
        # Learning rate scheduler
        self.scheduler = None
        self.scheduler_params = {}
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Set up metrics
        self.metrics = metrics or ['accuracy']
        self.metric_fns = {name: self._get_metric_fn(name) for name in self.metrics}
        
        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {name: [] for name in self.metrics},
            'val_metrics': {name: [] for name in self.metrics}
        }
    
    def set_learning_rate(self, lr: float):
        """Set learning rate for the optimizer."""
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def set_scheduler(self, scheduler_name: str, **scheduler_params):
        """
        Set up a learning rate scheduler.
        
        Args:
            scheduler_name: Name of the scheduler ('step', 'multistep', 'plateau', 'cosine')
            **scheduler_params: Parameters for the scheduler
        """
        from neuralnetworkapp.optimization import get_scheduler
        
        # Save scheduler params for checkpointing
        self.scheduler_params = {'name': scheduler_name, **scheduler_params}
        
        # Create the scheduler
        self.scheduler = get_scheduler(
            scheduler_name,
            self.optimizer,
            **scheduler_params
        )
    
    def step_scheduler(self, metrics=None):
        """
        Step the learning rate scheduler.
        
        Args:
            metrics: Metrics for schedulers that require them (e.g., ReduceLROnPlateau)
        """
        if self.scheduler is None:
            return
            
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metrics is None:
                raise ValueError("Metric value must be provided for ReduceLROnPlateau scheduler")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_learning_rate(self) -> float:
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_current_optimizer_state(self) -> dict:
        """Get the current state of the optimizer."""
        return {
            'optimizer': self.optimizer_name,
            'learning_rate': self.get_learning_rate(),
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler_params if hasattr(self, 'scheduler_params') else None
        }
    
    def _get_optimizer(self, name: str, lr: float, weight_decay: float):
        """Get optimizer by name with support for more optimizers."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if name == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=False)
        elif name == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=False)
        elif name == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif name == 'rmsprop':
            return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif name == 'adagrad':
            return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        elif name == 'adadelta':
            return optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
        elif name == 'adamax':
            return optim.Adamax(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _get_loss_fn(self, name: str):
        """Get loss function by name with support for more loss functions."""
        from neuralnetworkapp.losses import get_loss_function
        
        # Map common names to loss function names
        loss_mapping = {
            'cross_entropy': 'cross_entropy',
            'bce': 'bce',
            'bce_logits': 'bce_with_logits',
            'mse': 'mse',
            'l1': 'l1',
            'smooth_l1': 'smooth_l1',
            'huber': 'huber',
            'kl_div': 'kl_div',
            'nll': 'nll',
            'poisson_nll': 'poisson_nll',
            'hinge_embedding': 'hinge_embedding',
            'multi_margin': 'multi_margin',
            'multi_label_margin': 'multi_label_margin',
            'soft_margin': 'soft_margin',
            'triplet_margin': 'triplet_margin',
            'ctc': 'ctc',
            'margin_ranking': 'margin_ranking'
        }
        
        # Get the canonical name
        canonical_name = loss_mapping.get(name.lower(), name.lower())
        
        try:
            return get_loss_function(canonical_name, device=self.device)
        except ValueError as e:
            raise ValueError(f"Error initializing loss function '{name}': {str(e)}")
    
    def _get_metric_fn(self, name: str) -> Callable:
        """Get metric function by name."""
        def accuracy(output, target):
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum()
            return correct.float() / len(target)
            
        metrics = {
            'accuracy': accuracy
        }
        return metrics.get(name.lower(), lambda x, y: torch.tensor(0.0, device=self.device))
    
    def train_epoch(self, train_loader: DataLoader):
        """
        Train the model for one epoch with support for mixed precision training.
        """
        self.model.train()
        running_loss = 0.0
        
        # Initialize metrics
        metric_values = {name: 0.0 for name in self.metrics}
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.scaler is not None):
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize with gradient scaling for mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (only if clip_grad_norm is defined)
                if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping (only if clip_grad_norm is defined)
                if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.clip_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update running loss and metrics
            running_loss += loss.item() * inputs.size(0)
            
            # Update metrics
            with torch.no_grad():
                for name, metric_fn in self.metric_fns.items():
                    metric_values[name] += metric_fn(outputs, targets).item() * inputs.size(0)
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                current_lr = self.get_learning_rate()
                print(f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {current_lr:.6f}", end='\r')
        
        # Calculate average loss and metrics for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Format metrics with 'train_' prefix to be consistent with evaluate
        epoch_metrics = {'train_loss': epoch_loss}
        for name in self.metrics:
            epoch_metrics[name] = metric_values[name] / len(train_loader.dataset)
        
        # Step the learning rate scheduler if one is defined
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()
        
        return epoch_metrics['train_loss'], {k: v for k, v in epoch_metrics.items() if k != 'train_loss'}
    
    def evaluate(self, data_loader: DataLoader, prefix: str = 'val') -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on the given data loader.
        
        Args:
            data_loader: DataLoader for evaluation data
            prefix: Prefix for metric names (e.g., 'val', 'test')
            
        Returns:
            Tuple of (average_loss, metrics_dict) where metrics_dict includes all metrics with the given prefix
        """
        self.model.eval()
        running_loss = 0.0
        running_metrics = {name: 0.0 for name in self.metrics}
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                
                # Update metrics
                for name in self.metrics:
                    running_metrics[name] += self.metric_fns[name](outputs, targets).item() * inputs.size(0)
        
        # Calculate evaluation statistics
        avg_loss = running_loss / len(data_loader.dataset)
        
        # Calculate average metrics
        avg_metrics = {}
        for name in self.metrics:
            avg_metrics[name] = running_metrics[name] / len(data_loader.dataset)
        
        # Create metrics dict with prefix for logging
        prefixed_metrics = {f'{prefix}_loss': avg_loss}
        for name, value in avg_metrics.items():
            prefixed_metrics[f'{prefix}_{name}'] = value
        
        return avg_loss, avg_metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        patience: int = None,
        save_path: str = None,
        save_best: bool = True,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Maximum number of epochs to train for
            patience: Number of epochs to wait before early stopping
            save_path: Path to save the best model
            save_best: Whether to save the best model based on validation loss
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
            
        Returns:
            Dictionary containing training history
        """
        if patience is not None:
            best_val_loss = float('inf')
            epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Evaluate on validation set if provided
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader, prefix='val')
            else:
                val_loss = None
                val_metrics = {name: 0.0 for name in self.metrics}
            
            # Update history
            self.history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
            
            # Update metrics history
            for name in self.metrics:
                # Train metrics are already in the correct format from train_epoch
                self.history['train_metrics'][name].append(train_metrics.get(name, 0.0))
                # Val metrics are in the metrics dict
                self.history['val_metrics'][name].append(val_metrics.get(name, 0.0))
            
            # Print progress
            if verbose > 0:
                log = f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}'
                for name in self.metrics:
                    log += f' - train_{name}: {train_metrics.get(name, 0.0):.4f}'
                
                if val_loss is not None:
                    log += f' - val_loss: {val_loss:.4f}'
                    for name in self.metrics:
                        log += f' - val_{name}: {val_metrics.get(f"val_{name}", 0.0):.4f}'
                
                print(log)
            
            # Early stopping
            if patience is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    
                    # Save best model
                    if save_path is not None and save_best:
                        self.save_checkpoint(save_path, epoch, val_loss)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        if verbose > 0:
                            print(f'Early stopping after {epoch+1} epochs')
                        break
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'model_config': self.model.get_config() if hasattr(self.model, 'get_config') else {}
        }
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        # Save model config separately
        config_path = str(Path(path).with_suffix('.config.json'))
        with open(config_path, 'w') as f:
            json.dump(checkpoint['model_config'], f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, path: str, model: nn.Module = None, device: str = None):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        # Create model if not provided
        if model is None and 'model_config' in checkpoint:
            from neuralnetworkapp.models import create_model
            model = create_model(**checkpoint['model_config'])
        
        # Create trainer
        trainer = cls(model=model, device=device)
        
        # Load state dicts
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history
        if 'history' in checkpoint:
            trainer.history = checkpoint['history']
        
        return trainer, checkpoint.get('epoch', 0)
