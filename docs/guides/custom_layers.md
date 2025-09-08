# Custom Layers Guide

This guide explains how to create and use custom layers in NeuralNetworkApp.

## Table of Contents
- [Creating Custom Layers](#creating-custom-layers)
- [Layer with Trainable Weights](#layer-with-trainable-weights)
- [Custom Activation Functions](#custom-activation-functions)
- [Custom Regularizers](#custom-regularizers)
- [Custom Initializers](#custom-initializers)
- [Best Practices](#best-practices)

## Creating Custom Layers

### Basic Custom Layer
```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class SimpleDense(Layer):
    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Create weights when the layer is first called
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
            trainable=True,
            name='w'
        )
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True,
            name='b'
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Usage
layer = SimpleDense(64)
output = layer(tf.ones((1, 10)))  # Input shape (batch_size, 10)
```

## Layer with Trainable Weights

### Advanced Custom Layer with Config
```python
class CustomDense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config
```

## Custom Activation Functions

### Simple Activation Function
```python
@tf.function
def custom_activation(x):
    return tf.math.tanh(x) ** 2

# Usage in a model
model.add(tf.keras.layers.Dense(64, activation=custom_activation))
```

### Custom Activation Layer
```python
class CustomActivation(Layer):
    def __init__(self, alpha=0.5, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.alpha = alpha
        
    def call(self, inputs):
        return tf.maximum(self.alpha * inputs, inputs)  # LeakyReLU variant
        
    def get_config(self):
        return {'alpha': self.alpha}
```

## Custom Regularizers

### Custom L1 Regularizer
```python
class CustomL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=0.01):
        self.l1 = l1
    
    def __call__(self, x):
        return self.l1 * tf.reduce_sum(tf.abs(x))
    
    def get_config(self):
        return {'l1': float(self.l1)}

# Usage
model.add(tf.keras.layers.Dense(
    64, 
    kernel_regularizer=CustomL1Regularizer(l1=0.01)
))
```

## Custom Initializers

### Custom Weight Initializer
```python
class CustomInitializer(tf.keras.initializers.Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev
    
    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(
            shape, 
            mean=self.mean, 
            stddev=self.stddev, 
            dtype=dtype or tf.keras.backend.floatx()
        )
    
    def get_config(self):
        return {'mean': self.mean, 'stddev': self.stddev}

# Usage
model.add(tf.keras.layers.Dense(
    64, 
    kernel_initializer=CustomInitializer(mean=0.0, stddev=0.02)
))
```

## Best Practices

1. **Performance**
   - Use `@tf.function` for performance-critical operations
   - Prefer built-in operations over custom Python loops
   - Use `tf.vectorized_map` for element-wise operations

2. **Serialization**
   - Always implement `get_config()` for custom layers
   - Register custom objects for model saving/loading
   ```python
   tf.keras.utils.get_custom_objects().update({
       'CustomDense': CustomDense,
       'CustomActivation': CustomActivation
   })
   ```

3. **Debugging**
   - Add `@tf.function(experimental_compile=True)` for XLA compilation
   - Use `tf.print()` for debugging
   - Check shapes with `tf.shape()`

4. **Testing**
   - Test with different input shapes
   - Verify gradient computation
   - Check serialization/deserialization

## Common Issues and Solutions

### Shape Mismatch
- Verify input shapes in `build()`
- Use `tf.keras.backend.int_shape()` for debugging
- Check layer compatibility in sequential models

### Non-Trainable Weights
- Ensure weights are created in `build()`
- Mark weights as trainable with `trainable=True`
- Check `layer.trainable_weights`

### Serialization Errors
- Implement `get_config()`
- Register custom objects
- Use `tf.keras.utils.register_keras_serializable`

---
© Copyright 2025 Nsfr750. All Rights Reserved.
