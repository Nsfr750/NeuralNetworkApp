"""
Visualization utilities for neural network training and analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

# Set the style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class TrainingVisualizer:
    """
    A class to visualize training progress, metrics, and model performance.
    
    This class provides methods to track and visualize various metrics during training,
    including loss curves, accuracy, learning rate schedules, and more.
    """
    
    def __init__(self, log_dir: str = "logs", use_tensorboard: bool = True):
        """
        Initialize the TrainingVisualizer.
        
        Args:
            log_dir: Directory to save logs and visualizations
            use_tensorboard: Whether to use TensorBoard for visualization
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.metrics: Dict[str, List[float]] = {}
        self.figures: Dict[str, Figure] = {}
        self.writer = None
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer if enabled
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Add a scalar value to the metrics.
        
        Args:
            tag: Tag for the scalar (e.g., 'train/loss', 'val/accuracy')
            value: Scalar value to log
            step: Step or epoch number
        """
        if tag not in self.metrics:
            self.metrics[tag] = []
        
        # Add to metrics dictionary
        while len(self.metrics[tag]) <= step:
            self.metrics[tag].append(None)
        self.metrics[tag][step] = value
        
        # Log to TensorBoard if enabled
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def add_figure(self, tag: str, figure: Figure, step: Optional[int] = None) -> None:
        """
        Add a matplotlib figure to the visualizer.
        
        Args:
            tag: Tag for the figure
            figure: Matplotlib figure to add
            step: Optional step or epoch number
        """
        self.figures[tag] = figure
        
        # Log to TensorBoard if enabled
        if self.use_tensorboard and self.writer is not None and step is not None:
            self.writer.add_figure(tag, figure, step)
    
    def plot_metrics(self, metrics: Union[str, List[str]] = None, 
                    title: str = "Training Metrics") -> Figure:
        """
        Plot training metrics over time.
        
        Args:
            metrics: List of metric tags to plot (or single tag as string).
                    If None, all metrics will be plotted.
            title: Title for the plot
            
        Returns:
            Matplotlib figure containing the plot
        """
        if not self.metrics:
            raise ValueError("No metrics to plot. Call add_scalar() first.")
        
        # Handle single metric case
        if isinstance(metrics, str):
            metrics = [metrics]
        
        # If no metrics specified, use all available
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each metric
        for metric in metrics:
            if metric not in self.metrics:
                print(f"Warning: Metric '{metric}' not found. Skipping...")
                continue
                
            values = self.metrics[metric]
            steps = range(len(values))
            
            # Filter out None values for plotting
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            if not valid_indices:
                continue
                
            x = [steps[i] for i in valid_indices]
            y = [values[i] for i in valid_indices]
            
            ax.plot(x, y, 'o-', label=metric)
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        
        # Save the figure
        self.add_figure("metrics", fig)
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: List[str] = None,
                            title: str = "Confusion Matrix") -> Figure:
        """
        Plot a confusion matrix.
        
        Args:
            confusion_matrix: 2D array of shape (n_classes, n_classes)
            class_names: List of class names
            title: Title for the plot
            
        Returns:
            Matplotlib figure containing the confusion matrix
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize the confusion matrix
        cm_normalized = confusion_matrix.astype('float') / (
            confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8
        )
        
        # Plot the confusion matrix
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        # Save the figure
        self.add_figure("confusion_matrix", fig)
        
        return fig
    
    def plot_learning_rate_schedule(self, optimizer: torch.optim.Optimizer, 
                                  num_iterations: int, 
                                  title: str = "Learning Rate Schedule") -> Figure:
        """
        Plot the learning rate schedule.
        
        Args:
            optimizer: PyTorch optimizer
            num_iterations: Number of iterations to plot
            title: Title for the plot
            
        Returns:
            Matplotlib figure containing the learning rate schedule
        """
        lrs = []
        
        # Simulate the learning rate schedule
        for i in range(num_iterations):
            # Create a dummy loss for the optimizer
            optimizer.zero_grad()
            
            # Get the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            # Update the learning rate
            dummy_loss = torch.tensor(0.0, requires_grad=True)
            dummy_loss.backward()
            optimizer.step()
        
        # Plot the learning rate schedule
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(num_iterations), lrs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning Rate")
        ax.set_title(title)
        ax.grid(True)
        
        # Save the figure
        self.add_figure("learning_rate_schedule", fig)
        
        return fig
    
    def plot_embeddings(self, embeddings: np.ndarray, labels: np.ndarray,
                       title: str = "Embedding Visualization") -> Figure:
        """
        Visualize embeddings using t-SNE.
        
        Args:
            embeddings: 2D array of shape (n_samples, embedding_dim)
            labels: 1D array of shape (n_samples,) containing class labels
            title: Title for the plot
            
        Returns:
            Matplotlib figure containing the embedding visualization
        """
        try:
            from sklearn.manifold import TSNE
            
            # Reduce dimensionality using t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Plot the embeddings
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.7
            )
            
            # Add colorbar if we have class labels
            if len(np.unique(labels)) > 1:
                plt.colorbar(scatter, label="Class")
            
            # Customize plot
            ax.set_title(title)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True)
            
            # Save the figure
            self.add_figure("embeddings", fig)
            
            return fig
            
        except ImportError:
            print("scikit-learn is required for t-SNE visualization. "
                  "Install with: pip install scikit-learn")
            return None
    
    def plot_attention(self, image: np.ndarray, attention_map: np.ndarray,
                      alpha: float = 0.5,
                      title: str = "Attention Map") -> Figure:
        """
        Plot an attention map over an image.
        
        Args:
            image: Input image as a numpy array (H, W, C)
            attention_map: Attention map as a numpy array (H, W)
            alpha: Transparency for the attention overlay
            title: Title for the plot
            
        Returns:
            Matplotlib figure containing the attention visualization
        """
        # Normalize the attention map
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min() + 1e-8
        )
        
        # Resize attention map to match image dimensions if needed
        if attention_map.shape != image.shape[:2]:
            import cv2
            attention_map = cv2.resize(
                attention_map, 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Plot attention map
        im = ax2.imshow(attention_map, cmap='viridis')
        ax2.set_title("Attention Map")
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Plot overlay
        ax3.imshow(image)
        ax3.imshow(attention_map, alpha=alpha, cmap='viridis')
        ax3.set_title("Attention Overlay")
        ax3.axis('off')
        
        # Add main title
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        self.add_figure("attention_map", fig)
        
        return fig
    
    def save_figures(self, output_dir: str = None) -> None:
        """
        Save all figures to files.
        
        Args:
            output_dir: Directory to save the figures. If None, uses the log directory.
        """
        if output_dir is None:
            output_dir = os.path.join(self.log_dir, "figures")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            # Replace any invalid characters in the filename
            safe_name = "".join(c if c.isalnum() else "_" for c in name).rstrip('_')
            filepath = os.path.join(output_dir, f"{safe_name}.png")
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
    
    def close(self) -> None:
        """Close any open resources."""
        if self.writer is not None:
            self.writer.close()
        plt.close('all')
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Example of using the TrainingVisualizer
    import numpy as np
    
    # Create a visualizer
    visualizer = TrainingVisualizer(log_dir="example_logs")
    
    # Simulate training metrics
    for epoch in range(10):
        train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
        val_loss = 0.8 / (epoch + 1) + np.random.normal(0, 0.01)
        train_acc = 1.0 - 0.5 / (epoch + 1) + np.random.normal(0, 0.01)
        val_acc = 0.9 - 0.4 / (epoch + 1) + np.random.normal(0, 0.01)
        
        # Add metrics
        visualizer.add_scalar("train/loss", train_loss, epoch)
        visualizer.add_scalar("val/loss", val_loss, epoch)
        visualizer.add_scalar("train/accuracy", train_acc, epoch)
        visualizer.add_scalar("val/accuracy", val_acc, epoch)
    
    # Plot metrics
    visualizer.plot_metrics(["train/loss", "val/loss"], title="Training and Validation Loss")
    visualizer.plot_metrics(["train/accuracy", "val/accuracy"], title="Training and Validation Accuracy")
    
    # Example confusion matrix
    confusion = np.random.randint(0, 100, size=(10, 10))
    np.fill_diagonal(confusion, np.random.randint(100, 200, size=10))
    visualizer.plot_confusion_matrix(
        confusion,
        class_names=[f"Class {i}" for i in range(10)],
        title="Example Confusion Matrix"
    )
    
    # Save all figures
    visualizer.save_figures()
    
    # Close the visualizer
    visualizer.close()
    
    print("Example visualizations saved to 'example_logs/figures/'")
