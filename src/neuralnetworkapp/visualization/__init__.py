"""
Training Visualization and Monitoring

This module provides utilities for real-time visualization of training metrics,
model architecture visualization, and interactive model analysis.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output, HTML
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# For 3D visualization
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class TrainingVisualizer:
    """
    A class for real-time visualization of training metrics.
    
    Features:
    - Live updating plots of loss and metrics
    - Support for multiple metrics and datasets (train/val/test)
    - TensorBoard integration
    - Model architecture visualization
    - Feature space visualization
    """
    
    def __init__(self, log_dir: str = 'runs', use_tensorboard: bool = True):
        """
        Initialize the training visualizer.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            use_tensorboard: Whether to use TensorBoard for visualization
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.writer = None
        self.metrics = {}
        self.figures = {}
        self.axes = {}
        
        # Set up TensorBoard if enabled
        if self.use_tensorboard:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Set up matplotlib style
        plt.style.use('seaborn')
        sns.set_style("whitegrid")
        
        # For Jupyter notebook display
        self.is_notebook = self._is_notebook()
        
    def _is_notebook(self) -> bool:
        """Check if running in a Jupyter notebook."""
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:  # type: ignore
                return False
        except (ImportError, AttributeError):
            return False
        return True
    
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Add a scalar value to the visualizer.
        
        Args:
            tag: Data identifier
            value: Value to record
            step: Global step value to record
        """
        if tag not in self.metrics:
            self.metrics[tag] = {'steps': [], 'values': []}
        
        self.metrics[tag]['steps'].append(step)
        self.metrics[tag]['values'].append(value)
        
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """
        Add multiple scalar values to the visualizer.
        
        Args:
            main_tag: Parent name for the tags
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Global step value to record
        """
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}" if main_tag else tag
            self.add_scalar(full_tag, value, step)
    
    def add_figure(self, tag: str, figure: Any, step: int) -> None:
        """
        Add a figure to the visualizer.
        
        Args:
            tag: Data identifier
            figure: Figure or a matplotlib figure
            step: Global step value to record
        """
        if self.writer is not None:
            self.writer.add_figure(tag, figure, step)
    
    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """
        Add a histogram to the visualizer.
        
        Args:
            tag: Data identifier
            values: Values to build histogram
            step: Global step value to record
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def plot_metrics(self, metrics: List[str] = None, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot training metrics.
        
        Args:
            metrics: List of metric tags to plot (None for all)
            figsize: Figure size (width, height)
        """
        if not self.metrics:
            print("No metrics to plot")
            return
        
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        # Group metrics by type (e.g., 'loss', 'accuracy')
        metric_groups = {}
        for metric in metrics:
            if '/' in metric:
                group, name = metric.split('/', 1)
                if group not in metric_groups:
                    metric_groups[group] = []
                metric_groups[group].append(metric)
            else:
                if 'other' not in metric_groups:
                    metric_groups['other'] = []
                metric_groups['other'].append(metric)
        
        # Create subplots for each metric group
        num_plots = len(metric_groups)
        if num_plots == 0:
            return
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots))
        
        if num_plots == 1:
            axes = [axes]
        
        for ax, (group, group_metrics) in zip(axes, metric_groups.items()):
            for metric in group_metrics:
                if metric in self.metrics:
                    steps = self.metrics[metric]['steps']
                    values = self.metrics[metric]['values']
                    ax.plot(steps, values, label=metric)
            
            ax.set_title(f"{group.capitalize()} Metrics")
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if self.is_notebook:
            clear_output(wait=True)
            display(plt.gcf())
        else:
            plt.show()
        
        return fig
    
    def visualize_embedding(self, embeddings: torch.Tensor, labels: torch.Tensor = None, 
                          method: str = 'pca', title: str = 'Embedding Visualization',
                          figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Visualize embeddings in 2D or 3D space.
        
        Args:
            embeddings: Tensor of shape (n_samples, n_features)
            labels: Optional tensor of shape (n_samples,) for coloring points
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
            title: Plot title
            figsize: Figure size (width, height)
        """
        if len(embeddings) == 0:
            print("No embeddings to visualize")
            return
        
        # Convert to numpy if needed
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if labels is not None and torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # Reduce dimensionality
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(embeddings)
            x_label, y_label = 'PC1', 'PC2'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
            coords = reducer.fit_transform(embeddings)
            x_label, y_label = 't-SNE 1', 't-SNE 2'
        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2)
                coords = reducer.fit_transform(embeddings)
                x_label, y_label = 'UMAP 1', 'UMAP 2'
            except ImportError:
                print("UMAP not installed. Using PCA instead.")
                return self.visualize_embedding(embeddings, labels, 'pca', title, figsize)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Create plot
        plt.figure(figsize=figsize)
        
        if labels is not None:
            scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
        
        plt.title(f"{title} ({method.upper()})")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        
        if self.is_notebook:
            clear_output(wait=True)
            display(plt.gcf())
        else:
            plt.show()
    
    def visualize_attention(self, image: torch.Tensor, attention_map: torch.Tensor, 
                          alpha: float = 0.5, cmap: str = 'viridis',
                          figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Visualize attention maps over an image.
        
        Args:
            image: Input image tensor (C, H, W)
            attention_map: Attention map tensor (H', W')
            alpha: Transparency for the attention overlay
            cmap: Colormap for the attention map
            figsize: Figure size (width, height)
        """
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        if torch.is_tensor(attention_map):
            attention_map = attention_map.squeeze().cpu().numpy()
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Transpose image from (C, H, W) to (H, W, C)
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Resize attention map to match image size if needed
        if attention_map.shape != image.shape[:2]:
            import cv2
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot attention map
        im = ax2.imshow(attention_map, cmap=cmap)
        ax2.set_title('Attention Map')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Plot overlay
        ax3.imshow(image)
        ax3.imshow(attention_map, cmap=cmap, alpha=alpha)
        ax3.set_title('Attention Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        
        if self.is_notebook:
            clear_output(wait=True)
            display(fig)
        else:
            plt.show()
    
    def close(self) -> None:
        """Close the visualizer and release resources."""
        if self.writer is not None:
            self.writer.close()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


class RealTimePlot:
    """
    A simple real-time plot for monitoring training progress.
    
    This is useful for non-Jupyter environments where you still want live updates.
    """
    
    def __init__(self, max_points: int = 100, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the real-time plot.
        
        Args:
            max_points: Maximum number of points to display
            figsize: Figure size (width, height)
        """
        self.max_points = max_points
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.lines = {}
        self.data = {}
        
        plt.ion()  # Enable interactive mode
        plt.show()
    
    def update(self, values: Dict[str, float], step: int) -> None:
        """
        Update the plot with new values.
        
        Args:
            values: Dictionary of {metric_name: value}
            step: Current step/x-value
        """
        for name, value in values.items():
            if name not in self.data:
                self.data[name] = {'x': [], 'y': []}
                line, = self.ax.plot([], [], 'o-', label=name)
                self.lines[name] = line
            
            self.data[name]['x'].append(step)
            self.data[name]['y'].append(value)
            
            # Limit the number of points
            if len(self.data[name]['x']) > self.max_points:
                self.data[name]['x'] = self.data[name]['x'][-self.max_points:]
                self.data[name]['y'] = self.data[name]['y'][-self.max_points:]
            
            # Update the line data
            self.lines[name].set_data(
                self.data[name]['x'],
                self.data[name]['y']
            )
        
        # Update the plot
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend(loc='upper left')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self) -> None:
        """Close the plot."""
        plt.close(self.fig)
    
    def __del__(self):
        """Destructor to ensure the plot is closed."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Create a visualizer
    visualizer = TrainingVisualizer(log_dir='runs/example')
    
    # Simulate training
    epochs = 10
    steps_per_epoch = 100
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            
            # Simulate metrics
            train_loss = 1.0 / (0.1 * global_step + 1) + np.random.normal(0, 0.01)
            train_acc = 1.0 - 0.5 * np.exp(-0.1 * global_step / steps_per_epoch) + np.random.normal(0, 0.01)
            
            # Add metrics to visualizer
            visualizer.add_scalar('train/loss', train_loss, global_step)
            visualizer.add_scalar('train/accuracy', train_acc, global_step)
            
            # Validation every 20 steps
            if step % 20 == 0:
                val_loss = 0.8 * train_loss + np.random.normal(0, 0.005)
                val_acc = train_acc * 0.95 + np.random.normal(0, 0.005)
                
                visualizer.add_scalar('val/loss', val_loss, global_step)
                visualizer.add_scalar('val/accuracy', val_acc, global_step)
            
            # Update plots every 10 steps
            if step % 10 == 0:
                visualizer.plot_metrics()
    
    # Close the visualizer
    visualizer.close()
