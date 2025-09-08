# Network Builder API Reference

The `NetworkBuilder` class provides a fluent interface for constructing neural network models.

## Class: NetworkBuilder

### `__init__(self, input_shape, num_classes=None, name='model')`
Initialize a new NetworkBuilder instance.

**Parameters:**
- `input_shape`: Tuple - Shape of input data (e.g., (28, 28, 1) for MNIST)
- `num_classes`: int, optional - Number of output classes for classification
- `name`: str, optional - Name of the model

### Methods

#### `add_conv2d(filters, kernel_size, strides=(1, 1), padding='same', activation='relu', **kwargs)`
Add a 2D convolutional layer.

**Parameters:**
- `filters`: int - Number of output filters
- `kernel_size`: int or tuple - Size of the convolution window
- `strides`: tuple - Strides of the convolution
- `padding`: str - 'same' or 'valid'
- `activation`: str - Activation function to use
- `**kwargs`: Additional layer arguments

**Returns:**
`self` for method chaining

#### `add_maxpool2d(pool_size=(2, 2), strides=None, padding='valid')`
Add a 2D max pooling layer.

**Parameters:**
- `pool_size`: tuple - Size of the max pooling window
- `strides`: tuple, optional - Strides for pooling
- `padding`: str - 'same' or 'valid'

**Returns:**
`self` for method chaining

#### `add_dense(units, activation=None, **kwargs)`
Add a fully connected layer.

**Parameters:**
- `units`: int - Dimensionality of the output space
- `activation`: str - Activation function to use
- `**kwargs`: Additional layer arguments

**Returns:**
`self` for method chaining

#### `add_dropout(rate)`
Add a dropout layer.

**Parameters:**
- `rate`: float - Fraction of the input units to drop

**Returns:**
`self` for method chaining

#### `add_batchnorm()`
Add a batch normalization layer.

**Returns:**
`self` for method chaining

#### `build()`
Build and return the Keras model.

**Returns:**
A compiled Keras model instance

## Example Usage

```python
from neuralnetworkapp import NetworkBuilder

# Create a simple CNN
builder = NetworkBuilder(input_shape=(28, 28, 1), num_classes=10)

model = builder \
    .add_conv2d(32, 3, activation='relu') \
    .add_maxpool2d() \
    .add_conv2d(64, 3, activation='relu') \
    .add_maxpool2d() \
    .add_flatten() \
    .add_dense(128, activation='relu') \
    .add_dropout(0.5) \
    .add_dense(10, activation='softmax') \
    .build()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## Advanced Usage

### Custom Layers
You can add custom layers using the `add_layer` method:

```python
from tensorflow.keras.layers import Layer

class CustomLayer(Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer='zeros', trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Add custom layer to the network
builder.add_layer(CustomLayer(64))
```

## Common Issues

### Shape Mismatch
Ensure that the input shape of each layer matches the output shape of the previous layer.

### Model Not Learning
- Try different activation functions
- Adjust learning rate
- Add batch normalization layers
- Try different weight initializers

### Overfitting
- Add dropout layers
- Use L1/L2 regularization
- Increase training data or use data augmentation

---
© Copyright 2025 Nsfr750. All Rights Reserved.
