# Model Evaluation Guide

This guide covers the evaluation of neural network models, including metrics, visualization, and interpretation techniques.

## Table of Contents
- [Evaluation Metrics](#evaluation-metrics)
- [Confusion Matrix](#confusion-matrix)
- [ROC and AUC](#roc-and-auc)
- [Precision-Recall Curves](#precision-recall-curves)
- [Custom Metrics](#custom-metrics)
- [Model Interpretation](#model-interpretation)
- [Fairness and Bias](#fairness-and-bias)
- [Best Practices](#best-practices)

## Evaluation Metrics

### Classification Metrics
```python
from neuralnetworkapp.metrics import ClassificationMetrics

# Initialize with true and predicted labels
metrics = ClassificationMetrics(y_true, y_pred)

# Get various metrics
accuracy = metrics.accuracy()
precision = metrics.precision(average='weighted')
recall = metrics.recall(average='weighted')
f1 = metrics.f1_score(average='weighted')

# Class-wise metrics
class_report = metrics.classification_report()
print(class_report)
```

### Regression Metrics
```python
from neuralnetworkapp.metrics import RegressionMetrics

metrics = RegressionMetrics(y_true, y_pred)

mse = metrics.mean_squared_error()
mae = metrics.mean_absolute_error()
r2 = metrics.r2_score()
```

## Confusion Matrix

### Generating a Confusion Matrix
```python
from neuralnetworkapp.visualization import plot_confusion_matrix
import matplotlib.pyplot as plt

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(
    y_true, 
    y_pred, 
    classes=class_names,
    normalize=True,
    title='Normalized Confusion Matrix'
)
plt.show()
```

## ROC and AUC

### Plotting ROC Curves
```python
from neuralnetworkapp.visualization import plot_roc_curve

plt.figure(figsize=(10, 8))
plot_roc_curve(
    y_true,
    y_scores,  # Prediction probabilities
    n_classes=10,
    title='ROC Curves for Multi-class Classification'
)
plt.show()
```

## Precision-Recall Curves

### Plotting Precision-Recall Curves
```python
from neuralnetworkapp.visualization import plot_precision_recall_curve

plt.figure(figsize=(10, 8))
plot_precision_recall_curve(
    y_true,
    y_scores,
    n_classes=10,
    title='Precision-Recall Curves'
)
plt.show()
```

## Custom Metrics

### Implementing Custom Metrics
```python
import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-6))
        
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Usage
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[F1Score()]
)
```

## Model Interpretation

### Feature Importance
```python
from neuralnetworkapp.interpretation import compute_feature_importance

# For tree-based models
importance = compute_feature_importance(
    model,
    X_train,
    method='permutation',  # or 'shap', 'lime', etc.
    n_repeats=10
)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

### SHAP Values
```python
import shap

# Create explainer
explainer = shap.DeepExplainer(model, X_train[:100])

# Calculate SHAP values
shap_values = explainer.shap_values(X_test[:10])

# Plot SHAP summary
shap.summary_plot(shap_values, X_test[:10], class_names=class_names)
```

## Fairness and Bias

### Fairness Metrics
```python
from neuralnetworkapp.fairness import compute_fairness_metrics

# Compute fairness metrics
metrics = compute_fairness_metrics(
    y_true,
    y_pred,
    sensitive_features=df[['gender', 'age_group']],
    privileged_groups={'gender': ['male'], 'age_group': ['25-40']}
)

# Print fairness report
print(metrics.report())
```

## Best Practices

1. **Evaluation Strategy**
   - Always use a held-out test set
   - Use cross-validation for small datasets
   - Consider time-based splits for time series

2. **Metric Selection**
   - Choose metrics aligned with business objectives
   - Consider multiple metrics for a complete picture
   - Use class weights for imbalanced datasets

3. **Error Analysis**
   - Analyze misclassified examples
   - Look for patterns in errors
   - Identify edge cases

4. **Model Comparison**
   - Use statistical tests for significance
   - Compare multiple models
   - Consider model complexity vs. performance

5. **Documentation**
   - Document all metrics used
   - Save evaluation results
   - Track model versions and their performance

## Common Issues and Solutions

### Overly Optimistic Metrics
- Check for data leakage
- Ensure proper train/test split
- Use cross-validation

### High Variance in Metrics
- Increase dataset size
- Use more regularization
- Try simpler models

### Model Performs Differently in Production
- Check for data drift
- Monitor input distributions
- Implement shadow deployments

---
© Copyright 2025 Nsfr750. All Rights Reserved.
