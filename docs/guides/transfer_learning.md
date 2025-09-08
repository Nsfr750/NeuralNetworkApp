# Transfer Learning Guide

This guide explains how to leverage transfer learning in NeuralNetworkApp to adapt pre-trained models to new tasks.

## Table of Contents
- [Introduction to Transfer Learning](#introduction-to-transfer-learning)
- [Using Pre-trained Models](#using-pre-trained-models)
- [Feature Extraction](#feature-extraction)
- [Fine-tuning](#fine-tuning)
- [Custom Head Architecture](#custom-head-architecture)
- [Learning Rate Scheduling](#learning-rate-scheduling)
- [Best Practices](#best-practices)

## Introduction to Transfer Learning

Transfer learning involves taking a pre-trained model and adapting it to a new, similar task. This is particularly useful when you have limited training data.

### When to Use Transfer Learning
- Limited training data available
- Task is similar to the original model's training task
- Computational resources are limited

## Using Pre-trained Models

### Loading Pre-trained Models
```python
from tensorflow.keras.applications import (
    ResNet50, 
    EfficientNetB0,
    MobileNetV2
)

# Load model with pre-trained weights
base_model = ResNet50(
    weights='imagenet',
    include_top=False,  # Exclude final classification layer
    input_shape=(224, 224, 3)
)

# Freeze the base model
base_model.trainable = False
```

## Feature Extraction

### Using Pre-trained Models as Feature Extractors
```python
# Create a new model on top
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # Important: set training=False
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Fine-tuning

### Fine-tuning a Pre-trained Model
```python
# Unfreeze the top layers of the model
base_model.trainable = True

# Freeze the bottom N layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Recompile the model for the modifications to take effect
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Custom Head Architecture

### Building a Custom Classifier Head
```python
def build_custom_head(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Add custom layers
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Final classification layer
    outputs = tf.keras.layers.Dense(
        num_classes, 
        activation='softmax'
    )(x)
    
    return tf.keras.Model(inputs, outputs, name='custom_head')

# Create the complete model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = build_custom_head(x.shape[1:], num_classes=10)(x)

model = tf.keras.Model(inputs, outputs)
```

## Learning Rate Scheduling

### Custom Learning Rate Schedule
```python
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
```

## Best Practices

### Data Augmentation
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

# Include data augmentation in the model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
# ... rest of the model
```

### Callbacks
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]
```

## Common Issues and Solutions

### Overfitting
- Use more aggressive data augmentation
- Add more dropout
- Reduce model capacity
- Use L1/L2 regularization

### Underfitting
- Unfreeze more layers
- Increase model capacity
- Train for more epochs
- Reduce regularization

### Slow Training
- Use a smaller base model
- Increase batch size
- Use mixed precision training
- Utilize GPU/TPU acceleration

---
© Copyright 2025 Nsfr750. All Rights Reserved.
