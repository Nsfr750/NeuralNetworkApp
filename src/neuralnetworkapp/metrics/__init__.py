"""
Metrics module for model evaluation.

This module provides various metrics for evaluating model performance,
including classification, regression, and custom metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from enum import Enum, auto
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, mean_squared_error,
    mean_absolute_error, r2_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, cohen_kappa_score
)


class MetricType(Enum):
    """Supported metric types."""
    # Classification metrics
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    F1 = auto()
    ROC_AUC = auto()
    PR_AUC = auto()
    CONFUSION_MATRIX = auto()
    CLASSIFICATION_REPORT = auto()
    COHEN_KAPPA = auto()
    
    # Regression metrics
    MSE = auto()
    RMSE = auto()
    MAE = auto()
    R2 = auto()
    
    # Custom metrics
    CUSTOM = auto()


class Metric:
    """Base class for all metrics."""
    def __init__(self, name: str, metric_type: MetricType):
        self.name = name
        self.metric_type = metric_type
        self.reset()
    
    def reset(self) -> None:
        """Reset the metric state."""
        self.values = []
        
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update the metric with a new batch of predictions and targets.
        
        Args:
            y_pred: Model predictions
            y_true: Ground truth values
        """
        raise NotImplementedError("Subclasses must implement update method")
    
    def compute(self) -> float:
        """Compute the metric value."""
        if not self.values:
            return 0.0
        return float(np.mean(self.values))
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute the metric for the given predictions and targets."""
        self.update(y_pred, y_true)
        return self.compute()
    
    def __str__(self) -> str:
        return f"{self.name}: {self.compute():.4f}"


class Accuracy(Metric):
    """Accuracy metric for classification tasks."""
    def __init__(self, threshold: float = 0.5, name: str = "accuracy"):
        super().__init__(name, MetricType.ACCURACY)
        self.threshold = threshold
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        # Convert logits to probabilities if needed
        if y_pred.dim() > 1 and y_pred.size(1) > 1:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
        else:
            y_pred = (y_pred > self.threshold).long()
        
        y_true = y_true.long()
        correct = (y_pred == y_true).float().sum().item()
        total = y_true.numel()
        self.values.append(correct / total if total > 0 else 0.0)


class Precision(Metric):
    """Precision metric for classification tasks."""
    def __init__(self, average: str = 'binary', name: str = "precision"):
        super().__init__(name, MetricType.PRECISION)
        self.average = average
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        # Handle multi-class classification
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)
        
        try:
            precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)
            self.values.append(precision)
        except ValueError:
            self.values.append(0.0)


class Recall(Metric):
    """Recall metric for classification tasks."""
    def __init__(self, average: str = 'binary', name: str = "recall"):
        super().__init__(name, MetricType.RECALL)
        self.average = average
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)
        
        try:
            recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)
            self.values.append(recall)
        except ValueError:
            self.values.append(0.0)


class F1Score(Metric):
    """F1 score metric for classification tasks."""
    def __init__(self, average: str = 'binary', name: str = "f1"):
        super().__init__(name, MetricType.F1)
        self.average = average
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)
        
        try:
            f1 = f1_score(y_true, y_pred, average=self.average, zero_division=0)
            self.values.append(f1)
        except ValueError:
            self.values.append(0.0)


class ROCAUC(Metric):
    """ROC AUC score for binary and multi-class classification."""
    def __init__(self, average: str = 'macro', multi_class: str = 'ovr', name: str = "roc_auc"):
        super().__init__(name, MetricType.ROC_AUC)
        self.average = average
        self.multi_class = multi_class
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        # For binary classification
        if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
            if len(np.unique(y_true)) > 1:  # At least two classes needed
                try:
                    auc_score = roc_auc_score(y_true, y_pred)
                    self.values.append(auc_score)
                except ValueError:
                    self.values.append(0.0)
        # For multi-class classification
        elif y_pred.ndim == 2 and y_pred.shape[1] > 1:
            try:
                auc_score = roc_auc_score(
                    y_true, y_pred, 
                    multi_class=self.multi_class,
                    average=self.average
                )
                self.values.append(auc_score)
            except (ValueError, IndexError):
                self.values.append(0.0)


class PRAUC(Metric):
    """Precision-Recall AUC score for binary and multi-class classification."""
    def __init__(self, average: str = 'macro', name: str = "pr_auc"):
        super().__init__(name, MetricType.PR_AUC)
        self.average = average
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        # For binary classification
        if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
            if len(np.unique(y_true)) > 1:  # At least two classes needed
                try:
                    precision, recall, _ = precision_recall_curve(y_true, y_pred)
                    auc_score = auc(recall, precision)
                    self.values.append(auc_score)
                except ValueError:
                    self.values.append(0.0)
        # For multi-class classification
        elif y_pred.ndim == 2 and y_pred.shape[1] > 1:
            try:
                # One-vs-rest approach for multi-class
                n_classes = y_pred.shape[1]
                precision = dict()
                recall = dict()
                auc_scores = []
                
                for i in range(n_classes):
                    precision[i], recall[i], _ = precision_recall_curve(
                        (y_true == i).astype(int), 
                        y_pred[:, i]
                    )
                    auc_scores.append(auc(recall[i], precision[i]))
                
                if self.average == 'macro':
                    self.values.append(np.mean(auc_scores))
                else:
                    self.values.extend(auc_scores)
            except (ValueError, IndexError):
                self.values.append(0.0)


class MSELoss(Metric):
    """Mean Squared Error for regression tasks."""
    def __init__(self, name: str = "mse"):
        super().__init__(name, MetricType.MSE)
        self.mse = nn.MSELoss()
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        mse = self.mse(y_pred, y_true).item()
        self.values.append(mse)


class MAELoss(Metric):
    """Mean Absolute Error for regression tasks."""
    def __init__(self, name: str = "mae"):
        super().__init__(name, MetricType.MAE)
        self.mae = nn.L1Loss()
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        mae = self.mae(y_pred, y_true).item()
        self.values.append(mae)


class RMSELoss(Metric):
    """Root Mean Squared Error for regression tasks."""
    def __init__(self, name: str = "rmse"):
        super().__init__(name, MetricType.RMSE)
        self.mse = nn.MSELoss()
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        mse = self.mse(y_pred, y_true).item()
        self.values.append(np.sqrt(mse))


class R2Score(Metric):
    """R² score for regression tasks."""
    def __init__(self, name: str = "r2"):
        super().__init__(name, MetricType.R2)
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        try:
            r2 = r2_score(y_true, y_pred)
            self.values.append(r2)
        except ValueError:
            self.values.append(0.0)


class ConfusionMatrix(Metric):
    """Confusion matrix for classification tasks."""
    def __init__(self, num_classes: int, name: str = "confusion_matrix"):
        self.num_classes = num_classes  # Set num_classes before calling parent's __init__
        super().__init__(name, MetricType.CONFUSION_MATRIX)
        self.reset()
    
    def reset(self) -> None:
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)
        
        # Update confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        self.conf_matrix += cm
    
    def compute(self) -> np.ndarray:
        return self.conf_matrix
    
    def __str__(self) -> str:
        return f"{self.name}:\n{self.conf_matrix}"


class ClassificationReport(Metric):
    """Classification report with precision, recall, f1-score and support."""
    def __init__(self, target_names: Optional[List[str]] = None, name: str = "classification_report"):
        super().__init__(name, MetricType.CLASSIFICATION_REPORT)
        self.target_names = target_names
        self.reset()
    
    def reset(self) -> None:
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)
        
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
    
    def compute(self) -> str:
        return classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.target_names,
            zero_division=0
        )
    
    def __str__(self) -> str:
        return f"{self.name}:\n{self.compute()}"


class CohenKappa(Metric):
    """Cohen's kappa score for classification tasks."""
    def __init__(self, weights: Optional[str] = None, name: str = "cohen_kappa"):
        super().__init__(name, MetricType.COHEN_KAPPA)
        self.weights = weights
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)
        
        try:
            kappa = cohen_kappa_score(y_true, y_pred, weights=self.weights)
            self.values.append(kappa)
        except ValueError:
            self.values.append(0.0)


class MetricCollection:
    """A collection of metrics that can be updated and computed together."""
    def __init__(self, metrics: Optional[Dict[str, Metric]] = None):
        self.metrics = metrics or {}
    
    def add_metric(self, name: str, metric: Metric) -> None:
        """Add a metric to the collection."""
        self.metrics[name] = metric
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update all metrics in the collection."""
        for metric in self.metrics.values():
            metric.update(y_pred, y_true)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics in the collection."""
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset(self) -> None:
        """Reset all metrics in the collection."""
        for metric in self.metrics.values():
            metric.reset()
    
    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.metrics
    
    def __str__(self) -> str:
        return "\n".join([f"{name}: {metric.compute():.4f}" for name, metric in self.metrics.items()])


def get_metrics_for_task(
    task_type: str,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    average: str = 'macro',
    target_names: Optional[List[str]] = None
) -> MetricCollection:
    """
    Get a collection of metrics suitable for a specific task.
    
    Args:
        task_type: Type of task ('binary', 'multiclass', 'regression')
        num_classes: Number of classes (for classification tasks)
        threshold: Threshold for binary classification
        average: Averaging strategy for multi-class metrics
        target_names: Names of the target classes
        
    Returns:
        MetricCollection with appropriate metrics for the task
    """
    metrics = {}
    
    if task_type == 'binary':
        metrics.update({
            'accuracy': Accuracy(threshold=threshold),
            'precision': Precision(average=average),
            'recall': Recall(average=average),
            'f1': F1Score(average=average),
            'roc_auc': ROCAUC(average=average),
            'pr_auc': PRAUC(average=average)
        })
        
    elif task_type == 'multiclass':
        if num_classes is None:
            raise ValueError("num_classes must be provided for multiclass tasks")
            
        metrics.update({
            'accuracy': Accuracy(),
            'precision': Precision(average=average),
            'recall': Recall(average=average),
            'f1': F1Score(average=average),
            'roc_auc': ROCAUC(average=average, multi_class='ovr'),
            'pr_auc': PRAUC(average=average),
            'confusion_matrix': ConfusionMatrix(num_classes=num_classes),
            'classification_report': ClassificationReport(target_names=target_names),
            'cohen_kappa': CohenKappa()
        })
        
    elif task_type == 'regression':
        metrics.update({
            'mse': MSELoss(),
            'rmse': RMSELoss(),
            'mae': MAELoss(),
            'r2': R2Score()
        })
        
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return MetricCollection(metrics)
