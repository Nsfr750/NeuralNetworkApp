# Quick Start Guide

This guide will help you build, train, and evaluate your first neural network model using NeuralNetworkApp in just a few minutes.

## Your First Neural Network

### 1. Import Required Modules

```python
import numpy as np
from neuralnetworkapp import NetworkBuilder
from neuralnetworkapp.data import get_mnist_data
from neuralnetworkapp.training import Trainer
```

### 2. Load and Prepare Data

```python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = get_mnist_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension for CNN
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
```

### 3. Build a Simple CNN Model

```python
# Initialize network builder
builder = NetworkBuilder(input_shape=(28, 28, 1), num_classes=10)

# Build a simple CNN
model = builder \
    .add_conv2d(filters=32, kernel_size=3, activation='relu') \
    .add_maxpool2d() \
    .add_flatten() \
    .add_dense(128, activation='relu') \
    .add_dense(10, activation='softmax') \
    .build()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 4. Train the Model

```python
# Initialize trainer
trainer = Trainer(
    model=model,
    train_data=(x_train, y_train),
    val_data=(x_test, y_test),
    batch_size=32,
    epochs=5
)

# Start training
history = trainer.train()
```

### 5. Evaluate the Model

```python
# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')

# Make predictions
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
print(f'Predicted classes: {predicted_classes}')
print(f'True classes: {y_test[:5]}')
```

## Next Steps

- Explore more complex architectures in the [Examples](examples/index.md)
- Learn about [custom training loops](guides/advanced_training.md)
- Dive into [model evaluation](guides/evaluation.md) techniques

## Common Issues

### Shape Mismatch
Ensure your input data shape matches the expected input shape of the model.

### Low Accuracy
- Try increasing the number of training epochs
- Adjust the learning rate
- Add more layers or increase the number of filters/neurons
- Add regularization techniques like dropout

### Memory Issues
- Reduce batch size
- Use data generators for large datasets
- Enable mixed-precision training

---
© Copyright 2025 Nsfr750. All Rights Reserved.
