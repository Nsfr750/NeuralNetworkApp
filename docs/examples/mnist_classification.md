# MNIST Handwritten Digit Classification

This example demonstrates how to build, train, and evaluate a simple Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset.

## Overview

- **Dataset**: MNIST (60,000 training and 10,000 test grayscale images of handwritten digits 0-9)
- **Model**: Simple CNN with 2 convolutional layers and 2 dense layers
- **Training**: 5 epochs with Adam optimizer
- **Accuracy**: ~99% on test set

## Prerequisites

- Python 3.8+
- NeuralNetworkApp
- NumPy
- Matplotlib (for visualization)

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from neuralnetworkapp import NetworkBuilder
from neuralnetworkapp.data import get_mnist_data
from neuralnetworkapp.training import Trainer
from neuralnetworkapp.visualization import plot_training_history

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and prepare data
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = get_mnist_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension for CNN
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# 2. Build the model
print("\nBuilding the model...")
model = NetworkBuilder(input_shape=(28, 28, 1), num_classes=10) \
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
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# 3. Train the model
print("\nTraining the model...")
trainer = Trainer(
    model=model,
    train_data=(x_train, y_train),
    val_data=(x_test, y_test),
    batch_size=128,
    epochs=5,
    verbose=1
)

history = trainer.train()

# 4. Evaluate the model
print("\nEvaluating the model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')

# 5. Make predictions
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)

print('\nSample predictions:')
for i in range(5):
    print(f'Predicted: {predicted_classes[i]}, True: {y_test[i]}')

# 6. Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## Expected Output

```
Loading MNIST dataset...
Training data shape: (60000, 28, 28, 1)
Test data shape: (10000, 28, 28, 1)

Building the model...
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320       
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496     
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
flatten (Flatten)            (None, 3136)              0         
dense (Dense)                (None, 128)               401536    
dropout (Dropout)            (None, 128)               0         
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 421,642
Trainable params: 421,642
Non-trainable params: 0

Training the model...
Epoch 1/5
469/469 [==============================] - 15s 30ms/step - loss: 0.2504 - accuracy: 0.9258 - val_loss: 0.0559 - val_accuracy: 0.9816
Epoch 2/5
469/469 [==============================] - 14s 30ms/step - loss: 0.0878 - accuracy: 0.9734 - val_loss: 0.0409 - val_accuracy: 0.9868
Epoch 3/5
469/469 [==============================] - 14s 30ms/step - loss: 0.0657 - accuracy: 0.9804 - val_loss: 0.0350 - val_accuracy: 0.9882
Epoch 4/5
469/469 [==============================] - 14s 30ms/step - loss: 0.0541 - accuracy: 0.9835 - val_loss: 0.0316 - val_accuracy: 0.9897
Epoch 5/5
469/469 [==============================] - 14s 30ms/step - loss: 0.0463 - accuracy: 0.9859 - val_loss: 0.0282 - val_accuracy: 0.9908

Evaluating the model...
Test accuracy: 0.9908

Sample predictions:
Predicted: 7, True: 7
Predicted: 2, True: 2
Predicted: 1, True: 1
Predicted: 0, True: 0
Predicted: 4, True: 4
```

## Explanation

1. **Data Loading**: The MNIST dataset is loaded and preprocessed
2. **Model Architecture**: A CNN with two convolutional layers followed by max pooling, and two dense layers
3. **Training**: The model is trained for 5 epochs with a batch size of 128
4. **Evaluation**: The model achieves ~99% accuracy on the test set
5. **Visualization**: Training history is plotted to monitor model performance

## Common Issues

### Low Accuracy
- Ensure input data is properly normalized
- Try increasing the number of training epochs
- Adjust the learning rate

### Memory Issues
- Reduce batch size if you encounter memory errors
- Clear session if running multiple experiments: `import tensorflow as tf; tf.keras.backend.clear_session()`

### Slow Training
- Use a GPU if available
- Consider using a smaller model or batch size
- Try mixed precision training for potential speedup

## Next Steps

- Experiment with different model architectures
- Try data augmentation to improve generalization
- Explore hyperparameter tuning

---
© Copyright 2025 Nsfr750. All Rights Reserved.
