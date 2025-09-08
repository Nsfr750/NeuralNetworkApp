# Troubleshooting Guide

This guide provides solutions to common issues you might encounter while using NeuralNetworkApp.

## Installation Issues

### 1. Package Not Found
**Error**: `ModuleNotFoundError: No module named 'neuralnetworkapp'`

**Solution**:
```bash
# Install the package in development mode
pip install -e .

# Or install from PyPI
pip install neuralnetworkapp
```

### 2. Dependency Conflicts
**Error**: `pkg_resources.VersionConflict: (package version) (...) is required but (...) is installed`

**Solution**:
```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with exact dependencies
pip install -r requirements.txt
```

## Training Issues

### 1. CUDA Out of Memory
**Error**: `ResourceExhaustedError: OOM when allocating tensor`

**Solutions**:
- Reduce batch size
- Use a smaller model
- Enable mixed precision training
- Clear GPU memory:
  ```python
  import tensorflow as tf
  tf.keras.backend.clear_session()
  ```

### 2. Training is Slow
**Issue**: Training takes too long

**Solutions**:
- Use a GPU if available
- Increase batch size (if memory allows)
- Use mixed precision training:
  ```python
  from tensorflow.keras.mixed_precision import set_global_policy
  set_global_policy('mixed_float16')
  ```
- Profile your code to find bottlenecks

## Model Issues

### 1. Model Not Learning
**Issue**: Training accuracy is not improving

**Solutions**:
- Check learning rate (try values between 1e-1 and 1e-5)
- Normalize input data
- Try different optimizers (Adam, SGD, etc.)
- Add batch normalization layers
- Check for vanishing/exploding gradients

### 2. Overfitting
**Issue**: Large gap between training and validation accuracy

**Solutions**:
- Add dropout layers
- Use data augmentation
- Add L1/L2 regularization
- Get more training data
- Use early stopping

## Data Loading Issues

### 1. Shape Mismatch
**Error**: `ValueError: Input 0 of layer sequential is incompatible with the layer`

**Solutions**:
- Verify input shape matches the first layer's expected input
- Check for extra dimensions using `np.squeeze()`
- Ensure data type matches expected type

### 2. Memory Errors with Large Datasets
**Error**: `MemoryError: Unable to allocate array with shape (...)`

**Solutions**:
- Use data generators
- Reduce batch size
- Use `tf.data.Dataset` for better memory management
- Process data in smaller chunks

## Common Error Messages

### 1. `TypeError: 'NoneType' object is not callable`
**Cause**: Usually occurs when a function is called on `None`

**Solution**:
- Check if all required parameters are provided
- Verify function returns a value
- Add print statements to debug the flow

### 2. `ValueError: No gradients provided for any variable`
**Cause**: The loss function might not be connected to the model's computation graph

**Solution**:
- Ensure your loss function uses model outputs
- Check that all operations are differentiable
- Verify that the model is compiled before training

## Performance Optimization

### 1. Enable XLA Compilation
```python
# Enable XLA compilation
tf.config.optimizer.set_jit(True)
```

### 2. Use Mixed Precision
```python
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
```

### 3. Optimize Data Pipeline
```python
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(buffer_size=1024) \
    .batch(32) \
    .prefetch(tf.data.AUTOTUNE)
```

## Getting Help

If you encounter an issue not covered here:
1. Check the [GitHub Issues](https://github.com/Nsfr750/NeuralNetworkApp/issues)
2. Search the documentation
3. Create a new issue with:
   - Error message
   - Code to reproduce
   - Environment details
   - Expected vs actual behavior

---
© Copyright 2025 Nsfr750. All Rights Reserved.
