# Training API Reference

This document provides a comprehensive reference for the training utilities and workflows in NeuralNetworkApp.

## Table of Contents
- [Training Loops](#training-loops)
- [Optimizers](#optimizers)
- [Loss Functions](#loss-functions)
- [Metrics](#metrics)
- [Callbacks](#callbacks)
- [Distributed Training](#distributed-training)

## Training Loops

### `train_model`
High-level training function with built-in callbacks.

```python
from neuralnetworkapp.training import train_model

history = train_model(
    model,
    train_data,
    validation_data=None,
    epochs=100,
    batch_size=32,
    callbacks=[],
    verbose=1
)
```

### `CustomTrainingLoop`
Flexible training loop with fine-grained control.

```python
from neuralnetworkapp.training import CustomTrainingLoop

# Initialize training loop
training_loop = CustomTrainingLoop(
    model=model,
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Run training
history = training_loop.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[]
)
```

## Optimizers

### `create_optimizer`
Create an optimizer with common configurations.

```python
from neuralnetworkapp.training.optimizers import create_optimizer

optimizer = create_optimizer(
    name='adam',
    learning_rate=0.001,
    decay=1e-6,
    clipnorm=1.0,
    clipvalue=0.5,
    **kwargs
)
```

### Learning Rate Schedulers

#### `CosineDecayWithWarmup`
Cosine decay schedule with warmup.

```python
from neuralnetworkapp.training.schedules import CosineDecayWithWarmup

# Total training steps
total_steps = 1000
warmup_steps = 100
initial_learning_rate = 0.001

lr_schedule = CosineDecayWithWarmup(
    initial_learning_rate=initial_learning_rate,
    decay_steps=total_steps - warmup_steps,
    warmup_steps=warmup_steps,
    alpha=0.1  # Minimum learning rate as a fraction of initial_learning_rate
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

## Loss Functions

### Custom Losses
```python
from neuralnetworkapp.training.losses import (
    FocalLoss,
    DiceLoss,
    JaccardLoss,
    CombinedLoss
)

# Focal Loss for class imbalance
loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

# Dice Loss for segmentation
dice_loss = DiceLoss(smooth=1e-5)

# Combine multiple losses
combined_loss = CombinedLoss({
    'dice': (DiceLoss(), 0.7),
    'bce': (tf.keras.losses.BinaryCrossentropy(), 0.3)
})
```

### Regularization
```python
from neuralnetworkapp.training.regularizers import (
    add_weight_regularization,
    add_activity_regularization
)

# Add L2 regularization to all dense layers
model = add_weight_regularization(
    model,
    regularizer=tf.keras.regularizers.l2(0.01),
    layer_patterns=['dense', 'conv2d']
)

# Add activity regularization
model = add_activity_regularization(
    model,
    l1=0.01,
    l2=0.01
)
```

## Metrics

### Custom Metrics
```python
from neuralnetworkapp.training.metrics import (
    F1Score,
    Precision,
    Recall,
    BinaryIoU,
    MeanIoU
)

# For binary classification
metrics = [
    'accuracy',
    Precision(threshold=0.5),
    Recall(threshold=0.5),
    F1Score(threshold=0.5)
]

# For segmentation
iou_metric = MeanIoU(num_classes=num_classes)
```

## Callbacks

### Custom Callbacks
```python
from neuralnetworkapp.training.callbacks import (
    LearningRateLogger,
    GradientStatistics,
    ModelCheckpointWithEpoch,
    CSVLoggerWithLR
)

callbacks = [
    # Log learning rate at the end of each epoch
    LearningRateLogger(),
    
    # Log gradient statistics
    GradientStatistics(log_freq=100),
    
    # Save model checkpoints with epoch number
    ModelCheckpointWithEpoch(
        filepath='checkpoints/model_{epoch:03d}.h5',
        save_best_only=True,
        monitor='val_loss'
    ),
    
    # Log metrics to CSV with learning rate
    CSVLoggerWithLR('training_log.csv')
]
```

### Early Stopping with Warmup
```python
from neuralnetworkapp.training.callbacks import EarlyStoppingWithWarmup

early_stopping = EarlyStoppingWithEpochs(
    monitor='val_loss',
    min_epochs=10,  # Minimum number of epochs to train
    patience=5,     # Number of epochs with no improvement before stopping
    verbose=1,
    restore_best_weights=True
)
```

## Distributed Training

### Multi-GPU Training
```python
from neuralnetworkapp.training.distributed import (
    get_strategy,
    prepare_dataset_for_distributed_training
)

# Get distribution strategy
strategy = get_strategy()

with strategy.scope():
    # Create and compile model
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare dataset
train_dataset = prepare_dataset_for_distributed_training(train_dataset, strategy)

# Train the model
model.fit(train_dataset, epochs=10)
```

### TPU Training
```python
from neuralnetworkapp.training.distributed import setup_tpu

tpu, tpu_strategy = setup_tpu()

with tpu_strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_dataset, epochs=10)
```

## Hyperparameter Tuning

### Random Search
```python
from neuralnetworkapp.training.tuning import RandomSearchTuner

param_grid = {
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'batch_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'units': [64, 128, 256]
}

tuner = RandomSearchTuner(
    model_builder=create_model,
    param_grid=param_grid,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='tuning',
    project_name='my_model_tuning'
)

tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
```

## Best Practices

1. **Reproducibility**
   - Set random seeds
   - Use deterministic operations
   - Save training configurations

2. **Monitoring**
   - Use TensorBoard for visualization
   - Log important metrics
   - Monitor hardware utilization

3. **Checkpointing**
   - Save model checkpoints regularly
   - Save optimizer state
   - Include epoch number in filenames

## Common Issues and Solutions

### Training is Too Slow
- Increase batch size
- Use mixed precision training
- Profile data loading
- Enable XLA compilation

### Model Not Learning
- Check learning rate
- Verify data preprocessing
- Inspect gradient flow
- Try overfitting a small batch

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Clear session between runs
- Use mixed precision

---
© Copyright 2025 Nsfr750. All Rights Reserved.
