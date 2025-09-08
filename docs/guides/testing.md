# Testing Guide

This guide covers testing practices and patterns for the NeuralNetworkApp project.

## Table of Contents
- [Testing Philosophy](#testing-philosophy)
- [Test Organization](#test-organization)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Model Testing](#model-testing)
- [Fixtures and Mocks](#fixtures-and-mocks)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)

## Testing Philosophy

### Testing Principles
1. **Test First**: Write tests before or alongside implementation
2. **Isolation**: Test one thing at a time
3. **Determinism**: Tests should be deterministic and repeatable
4. **Readability**: Tests should be clear and serve as documentation
5. **Maintainability**: Keep tests DRY and well-organized

### Testing Pyramid
```
        /\\
       /  \\
      /    \\
     / Unit \\
    /        \\
   /----------\\
  /Integration\\
 /             \\
/---------------\\
|    E2E       |
|              |
----------------
```

## Test Organization

### Directory Structure
```
tests/
├── __init__.py
├── conftest.py            # Shared fixtures
├── unit/                  # Unit tests
│   ├── test_layers.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/           # Integration tests
│   ├── test_training.py
│   └── test_pipeline.py
└── data/                  # Test data
    └── fixtures/
```

### Test File Structure
```python
"""Tests for neuralnetworkapp/layers/attention.py."""
import pytest
import numpy as np
import tensorflow as tf

from neuralnetworkapp.layers.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention layer."""

    @pytest.fixture
    def layer(self):
        """Return a configured MultiHeadAttention layer."""
        return MultiHeadAttention(num_heads=4, key_dim=64)

    def test_initialization(self, layer):
        """Test layer initialization."""
        assert layer.num_heads == 4
        assert layer.key_dim == 64
        assert layer.supports_masking

    def test_call_shape(self, layer, random_inputs):
        """Test output shape of the layer."""
        query = tf.random.normal((32, 10, 128))
        value = tf.random.normal((32, 10, 128))
        output = layer(query=query, value=value)
        assert output.shape == (32, 10, 128)

    # More test methods...
```

## Unit Testing

### Testing Layers
```python
def test_conv2d_layer():
    """Test Conv2D layer with valid padding."""
    layer = tf.keras.layers.Conv2D(
        filters=32, 
        kernel_size=3, 
        padding='valid'
    )
    input_tensor = tf.random.normal((1, 28, 28, 3))
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (1, 26, 26, 32)
```

### Testing Custom Losses
```python
def test_custom_loss():
    """Test custom loss function."""
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    y_pred = tf.constant([[0.9, 0.1], [0.3, 0.7]])
    
    loss = custom_loss(y_true, y_pred)
    assert loss.shape == (2,)  # Per-sample loss
    assert tf.reduce_all(loss >= 0.0)  # Non-negative
```

## Integration Testing

### Testing Training Loop
```python
def test_training_loop(tf_dataset, model):
    """Test complete training loop."""
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Run one training step
    for x_batch, y_batch in tf_dataset.take(1):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # Verify model updated
    assert not tf.math.is_nan(loss)
```

### Testing Callbacks
```python
def test_model_checkpoint(tmp_path):
    """Test model checkpoint callback."""
    model = create_model()
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )
    
    # Train for one epoch
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=[model_checkpoint]
    )
    
    # Verify checkpoint was created
    assert checkpoint_path.exists()
```

## Model Testing

### Testing Model Architecture
```python
def test_model_architecture():
    """Test model architecture and output shapes."""
    model = create_model()
    
    # Test input/output shapes
    assert len(model.inputs) == 1
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
    
    # Test layer count
    assert len(model.layers) == 8
    
    # Test specific layer types
    assert isinstance(model.layers[1], tf.keras.layers.Conv2D)
    assert isinstance(model.layers[-1], tf.keras.layers.Dense)
```

### Testing Model Saving/Loading
```python
def test_model_save_load(tmp_path):
    """Test model saving and loading."""
    model = create_model()
    test_input = tf.random.normal((1, 28, 28, 1))
    
    # Get predictions before saving
    original_output = model.predict(test_input)
    
    # Save and load model
    model_path = tmp_path / "saved_model"
    model.save(model_path)
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Verify predictions match
    loaded_output = loaded_model.predict(test_input)
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-6)
```

## Fixtures and Mocks

### Common Fixtures
```python
# conftest.py
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture
def random_inputs():
    """Generate random input tensors for testing."""
    np.random.seed(42)
    tf.random.set_seed(42)
    return tf.random.normal((32, 224, 224, 3))

@pytest.fixture
def tf_dataset():
    """Create a simple tf.data.Dataset for testing."""
    x = np.random.rand(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(100, 1))
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
```

### Using Mocks
```python
from unittest.mock import patch, MagicMock

def test_training_with_mock():
    """Test training with a mocked model."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.trainable_variables = [tf.Variable(1.0)]
    
    # Mock the fit method
    with patch.object(mock_model, 'fit') as mock_fit:
        train_model(mock_model)
        mock_fit.assert_called_once()
```

## Test Coverage

### Measuring Coverage
```bash
# Install coverage
pip install pytest-cov

# Run tests with coverage
pytest --cov=neuralnetworkapp --cov-report=term-missing

# Generate HTML report
pytest --cov=neuralnetworkapp --cov-report=html
```

### Coverage Configuration
```ini
# .coveragerc
[run]
source = neuralnetworkapp
omit = 
    */tests/*
    */__init__.py
    */version.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
```

## Continuous Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=neuralnetworkapp --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## Best Practices

### Test Naming
- Be descriptive and specific
- Follow pattern: `test_<method>_when_<condition>_then_<expected_behavior>`
- Example: `test_predict_when_input_is_empty_then_raises_value_error`

### Test Organization
- Group related tests in classes
- Use fixtures for common setup
- Keep tests independent and isolated

### Performance
- Use `@pytest.mark.slow` for slow tests
- Run fast tests frequently during development
- Use `pytest-xdist` for parallel test execution

## Common Issues and Solutions

### Flaky Tests
- Use fixed random seeds
- Avoid time-based tests
- Use `pytest-rerunfailures` for known flaky tests

### Slow Tests
- Use smaller test datasets
- Mock external services
- Use `@pytest.mark.slow` and run separately

### Debugging Tests
- Use `pytest --pdb` to enter debugger on failure
- Add `print()` statements or use logging
- Use `pytest -v` for verbose output

---
© Copyright 2025 Nsfr750. All Rights Reserved.
