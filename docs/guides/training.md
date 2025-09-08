# Training Guide

This guide covers the training process, optimization techniques, and best practices for training neural networks with NeuralNetworkApp.

## Table of Contents
- [Basic Training](#basic-training)
- [Custom Training Loops](#custom-training-loops)
- [Learning Rate Scheduling](#learning-rate-scheduling)
- [Regularization](#regularization)
- [Mixed Precision Training](#mixed-precision-training)
- [Distributed Training](#distributed-training)
- [Checkpointing](#checkpointing)
- [Best Practices](#best-practices)

## Basic Training

### Using the Trainer Class
```python
from neuralnetworkapp.training import Trainer

# Initialize trainer
trainer = Trainer(
    model=model,
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    batch_size=32,
    epochs=10,
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Start training
history = trainer.train()
```

### Training with Callbacks
```python
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(factor=0.1, patience=3)
]

trainer = Trainer(
    # ... other parameters ...
    callbacks=callbacks
)
```

## Custom Training Loops

### Basic Training Loop
```python
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss_value = loss_fn(y_batch, logits)
    
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # Update metrics
    train_acc_metric.update_state(y_batch, logits)
    return loss_value

# Training loop
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss_value = train_step(x_batch, y_batch)
        
        # Log every 100 batches
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss_value:.4f}")
    
    # Reset metrics at the end of each epoch
    train_acc = train_acc_metric.result()
    print(f"Training accuracy: {train_acc:.4f}")
    train_acc_metric.reset_states()
```

## Learning Rate Scheduling

### Built-in Schedulers
```python
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay,
    PiecewiseConstantDecay,
    CosineDecay
)

# Exponential decay
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9
)

# Piecewise constant decay
lr_schedule = PiecewiseConstantDecay(
    boundaries=[1000, 2000],
    values=[1e-3, 1e-4, 1e-5]
)

# Cosine decay
lr_schedule = CosineDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

## Regularization

### L1/L2 Regularization
```python
from tensorflow.keras import regularizers

# Add L2 regularization to a layer
model.add(tf.keras.layers.Dense(
    64,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l1(0.01)
))
```

### Dropout
```python
model.add(tf.keras.layers.Dropout(0.5))
```

## Mixed Precision Training

### Enabling Mixed Precision
```python
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')

# Now build and compile your model
model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

## Distributed Training

### Multi-GPU Training
```python
# Create a distribution strategy
distribute = tf.distribute.MirroredStrategy()

# Open a strategy scope
with distribute.scope():
    # Everything that creates variables should be under the strategy scope
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(train_dataset, epochs=10)
```

## Checkpointing

### Saving Checkpoints
```python
checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    model=model,
    epoch=tf.Variable(0)
)

# Save checkpoints
checkpoint.save('training_checkpoints/ckpt')

# Restore from checkpoint
checkpoint.restore(tf.train.latest_checkpoint('training_checkpoints'))
```

## Best Practices

1. **Monitoring**
   - Use TensorBoard for visualization
   - Track multiple metrics
   - Monitor hardware utilization

2. **Hyperparameter Tuning**
   - Use learning rate finder
   - Try different batch sizes
   - Experiment with different optimizers

3. **Debugging**
   - Check for NaNs/Infs
   - Monitor gradient flow
   - Use gradient clipping for stability

4. **Reproducibility**
   - Set random seeds
   - Document all hyperparameters
   - Save model configurations

5. **Performance**
   - Use data prefetching
   - Enable XLA compilation
   - Profile training loop

## Common Issues and Solutions

### Training is Slow
- Enable mixed precision training
- Increase batch size
- Use a more powerful GPU
- Profile with TensorBoard

### Model Not Converging
- Check learning rate
- Normalize input data
- Try different weight initializations
- Add batch normalization

### Overfitting
- Add more training data
- Use data augmentation
- Add regularization
- Use early stopping

---
© Copyright 2025 Nsfr750. All Rights Reserved.
