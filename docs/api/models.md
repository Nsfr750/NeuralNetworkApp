# Models API Reference

This document provides a comprehensive reference for the model architectures and building blocks in NeuralNetworkApp.

## Table of Contents
- [Predefined Models](#predefined-models)
- [Custom Layers](#custom-layers)
- [Model Utilities](#model-utilities)
- [Model Training](#model-training)
- [Transfer Learning](#transfer-learning)

## Predefined Models

### `create_cnn`
Create a Convolutional Neural Network.

```python
from neuralnetworkapp.models import create_cnn

model = create_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    filters=[32, 64, 128],
    kernel_sizes=[3, 3, 3],
    dense_units=[512],
    dropout_rate=0.5,
    activation='relu',
    output_activation='softmax'
)
```

### `create_mlp`
Create a Multi-Layer Perceptron.

```python
from neuralnetworkapp.models import create_mlp

model = create_mlp(
    input_dim=784,
    num_classes=10,
    hidden_units=[512, 256],
    dropout_rates=[0.2, 0.2],
    activation='relu',
    output_activation='softmax'
)
```

### `create_resnet`
Create a ResNet model.

```python
from neuralnetworkapp.models import create_resnet

model = create_resnet(
    input_shape=(224, 224, 3),
    num_classes=1000,
    version=50,  # 18, 34, 50, 101, or 152
    include_top=True,
    weights='imagenet',
    pooling=None
)
```

## Custom Layers

### `AttentionLayer`
Self-attention layer for sequence data.

```python
from neuralnetworkapp.layers import AttentionLayer

# In your model
inputs = tf.keras.layers.Input(shape=(None, 128))
attention_output, attention_weights = AttentionLayer(return_attention_scores=True)(inputs)
```

### `Time2Vector`
Time embedding layer for time series data.

```python
from neuralnetworkapp.layers import Time2Vector

# In your model
inputs = tf.keras.layers.Input(shape=(24, 5))  # 24 timesteps, 5 features
time_embedding = Time2Vector(24)(inputs)
```

## Model Utilities

### `get_model_summary`
Get a formatted model summary as a string.

```python
from neuralnetworkapp.models.utils import get_model_summary

summary = get_model_summary(model, line_length=100, print_fn=None)
print(summary)
```

### `count_params`
Count the number of trainable and non-trainable parameters.

```python
from neuralnetworkapp.models.utils import count_params

trainable, non_trainable = count_params(model)
print(f"Trainable params: {trainable:,}")
print(f"Non-trainable params: {non_trainable:,}")
```

### `plot_model_architecture`
Generate a visualization of the model architecture.

```python
from neuralnetworkapp.models.visualization import plot_model_architecture

plot_model_architecture(
    model,
    to_file='model.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96
)
```

## Model Training

### `compile_model`
Helper function to compile a model with common configurations.

```python
from neuralnetworkapp.models.training import compile_model

compile_model(
    model,
    optimizer='adam',
    learning_rate=0.001,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    clipnorm=1.0,
    clipvalue=0.5
)
```

### `train_model`
Train a model with common callbacks and configurations.

```python
from neuralnetworkapp.models.training import train_model

history = train_model(
    model,
    train_data,
    validation_data=None,
    epochs=100,
    batch_size=32,
    callbacks=[],
    early_stopping_patience=10,
    reduce_lr_patience=5,
    checkpoint_path='checkpoints/model.ckpt',
    tensorboard_logdir='logs',
    verbose=1
)
```

## Transfer Learning

### `create_transfer_model`
Create a model using transfer learning.

```python
from neuralnetworkapp.models.transfer import create_transfer_model

base_model_name = 'EfficientNetB0'  # or any from tf.keras.applications
input_shape = (224, 224, 3)
num_classes = 10

model = create_transfer_model(
    base_model_name=base_model_name,
    input_shape=input_shape,
    num_classes=num_classes,
    freeze_base=True,
    custom_top=None,
    weights='imagenet',
    pooling='avg'
)
```

### `fine_tune_model`
Fine-tune a pre-trained model.

```python
from neuralnetworkapp.models.transfer import fine_tune_model

# Unfreeze the top N layers of the base model
model = fine_tune_model(
    model,
    base_model_name='EfficientNetB0',
    num_layers_to_unfreeze=10,
    learning_rate=1e-5
)
```

## Advanced Usage

### Custom Training Loop
```python
@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

# Training loop
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss_value = train_step(model, x_batch, y_batch, optimizer, loss_fn)
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss_value:.4f}")
```

## Best Practices

1. **Model Architecture**
   - Start with a simple model
   - Gradually increase complexity
   - Use model subclassing for custom architectures

2. **Training**
   - Use learning rate scheduling
   - Monitor training with callbacks
   - Save checkpoints regularly

3. **Optimization**
   - Use mixed precision training
   - Enable XLA compilation
   - Profile model performance

## Common Issues and Solutions

### Overfitting
- Add more data or use data augmentation
- Increase dropout rate
- Add L1/L2 regularization
- Use early stopping

### Underfitting
- Increase model capacity
- Train for more epochs
- Decrease regularization
- Use a different architecture

### Training Instability
- Normalize input data
- Use gradient clipping
- Try a different optimizer
- Adjust learning rate

---
© Copyright 2025 Nsfr750. All Rights Reserved.
