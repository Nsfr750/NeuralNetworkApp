# Model Optimization Guide

This guide covers techniques for optimizing neural network models in NeuralNetworkApp, including optimizers, loss functions, and advanced optimization strategies.

## Table of Contents

1. [Optimization Techniques](#optimization-techniques)
   - [Pruning](#pruning)
   - [Quantization](#quantization)
   - [Knowledge Distillation](#knowledge-distillation)
2. [Optimizers](#optimizers)
3. [Loss Functions](#loss-functions)
4. [Learning Rate Schedulers](#learning-rate-schedulers)
5. [Regularization](#regularization)
6. [Mixed Precision Training](#mixed-precision-training)
7. [Gradient Clipping](#gradient-clipping)
8. [Best Practices](#best-practices)

## Optimization Techniques

### Pruning

Pruning removes unnecessary weights from a model to reduce its size and improve inference speed.

```python
import tensorflow_model_optimization as tfmot

# Prune all layers
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Define pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=0.80,
        begin_step=0,
        end_step=2000
    )
}

# Apply pruning to the model
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Train with pruning callbacks
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
]

model_for_pruning.fit(..., callbacks=callbacks)
```

### Quantization

Quantization reduces model size and improves inference speed by using lower precision (e.g., 8-bit integers) for weights and activations.

**Post-training quantization**:
```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save
quantized_tflite_model = converter.convert()
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
```

**Quantization-aware training**:
```python
import tensorflow_model_optimization as tfmot

# Apply quantization to the entire model
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

# Train the quantized model
q_aware_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
q_aware_model.fit(train_images, train_labels, epochs=5)
```

### Knowledge Distillation

Train a smaller student model to mimic a larger teacher model's behavior.

```python
# Teacher model (larger, more accurate)
teacher = create_teacher_model()
teacher.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
teacher.fit(train_images, train_labels, epochs=5)

# Student model (smaller, to be trained with teacher's knowledge)
student = create_student_model()

# Custom loss function combining teacher and true labels
def distillation_loss(y_true, y_pred, teacher_logits, temp=2.0):
    return tf.keras.losses.kl_divergence(
        tf.nn.softmax(teacher_logits/temp, axis=1),
        tf.nn.softmax(y_pred/temp, axis=1)
    ) * (temp ** 2)

# Train student with combined loss
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # Get teacher predictions
        teacher_logits = teacher(images, training=False)
        # Get student predictions
        student_logits = student(images, training=True)
        
        # Calculate losses
        task_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, student_logits)
        distill_loss = distillation_loss(labels, student_logits, teacher_logits)
        total_loss = 0.7 * task_loss + 0.3 * distill_loss
    
    # Apply gradients
    grads = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return total_loss
```

## Optimizers

NeuralNetworkApp supports various optimizers for training deep learning models:

- **SGD** - Stochastic Gradient Descent with momentum
  ```python
  tf.keras.optimizers.SGD(
      learning_rate=0.01,
      momentum=0.9,
      nesterov=True
  )
  ```

- **Adam** - Adaptive Moment Estimation
  ```python
  tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False
  )
  ```

- **AdamW** - Adam with weight decay fix
  ```python
  tfa.optimizers.AdamW(
      learning_rate=0.001,
      weight_decay=0.004,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07
  )
  ```

- **RMSprop** - Root Mean Square Propagation
  ```python
  tf.keras.optimizers.RMSprop(
      learning_rate=0.001,
      rho=0.9,
      momentum=0.0,
      epsilon=1e-07,
      centered=False
  )
  ```

## Loss Functions

### Common Loss Functions

- **Categorical Crossentropy**
  ```python
  tf.keras.losses.CategoricalCrossentropy(
      from_logits=False,
      label_smoothing=0.0,
      reduction='auto',
      name='categorical_crossentropy'
  )
  ```

- **Sparse Categorical Crossentropy**
  ```python
  tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=False,
      reduction='auto',
      name='sparse_categorical_crossentropy'
  )
  ```

- **Binary Crossentropy**
  ```python
  tf.keras.losses.BinaryCrossentropy(
      from_logits=False,
      label_smoothing=0.0,
      reduction='auto',
      name='binary_crossentropy'
  )
  ```

### Custom Loss Functions

```python
def custom_huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, squared_loss, linear_loss)
```

## Learning Rate Schedulers

### Built-in Schedulers

```python
# Cosine Decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    alpha=0.01  # Final learning rate will be `initial_learning_rate * alpha`
)

# Exponential Decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Piecewise Constant Decay
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values
)
```

### Custom Learning Rate Schedule

```python
class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps=4000):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # Linear warmup
        if step < self.warmup_steps:
            return self.initial_learning_rate * (step / self.warmup_steps)
        # Cosine decay
        progress = (step - self.warmup_steps) / (100000 - self.warmup_steps)
        return 0.5 * (1 + tf.math.cos(np.pi * progress)) * self.initial_learning_rate
```

## Regularization

### L1/L2 Regularization

```python
# Add L2 regularization to a layer
layer = tf.keras.layers.Dense(
    64,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
    bias_regularizer=tf.keras.regularizers.l2(0.01)
)
```

### Dropout

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # 50% dropout
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Batch Normalization

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    # ... more layers
])
```

## Mixed Precision Training

Mixed precision uses both 16-bit and 32-bit floating-point types to make training faster and use less memory.

```python
# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Build and compile your model
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(train_images, train_labels, epochs=5)
```

## Gradient Clipping

Gradient clipping helps prevent exploding gradients in deep neural networks.

### Global Norm Clipping

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0  # Clip gradients by norm
)
```

### Value Clipping

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipvalue=0.5  # Clip gradients by value
)
```

## Best Practices

1. **Start Simple**
   - Begin with a small model and simple optimizer (e.g., SGD)
   - Gradually increase complexity as needed

2. **Learning Rate**
   - Use learning rate scheduling
   - Try different initial learning rates
   - Consider learning rate warmup for transformer models

3. **Regularization**
   - Start with small regularization values
   - Use early stopping to prevent overfitting
   - Try different dropout rates (0.2-0.5)

4. **Batch Size**
   - Larger batch sizes can speed up training
   - Smaller batch sizes can lead to better generalization
   - Adjust learning rate when changing batch size

5. **Monitoring**
   - Track training and validation metrics
   - Use TensorBoard for visualization
   - Monitor gradient statistics

## Common Issues and Solutions

### Training is Unstable
- Reduce learning rate
- Use gradient clipping
- Try a different optimizer
- Normalize input data

### Model is Overfitting
- Add more training data
- Increase regularization
- Use data augmentation
- Try dropout or weight decay

### Training is Too Slow
- Use mixed precision training
- Increase batch size
- Profile your training loop
- Use a larger learning rate with learning rate scheduling

### Poor Validation Performance
- Check for data leakage
- Ensure proper train/validation split
- Try different model architectures
- Consider transfer learning

---
© Copyright 2025 Nsfr750. All Rights Reserved.
- **Adagrad** - Adaptive Gradient Algorithm
- **Adadelta** - An Adaptive Learning Rate Method
- **Adamax** - A variant of Adam based on infinity norm

## Available Loss Functions

### Classification Losses
- CrossEntropyLoss
- BCELoss
- BCEWithLogitsLoss
- NLLLoss
- PoissonNLLLoss
- KLDivLoss
- HingeEmbeddingLoss
- MultiMarginLoss
- MultiLabelMarginLoss
- SoftMarginLoss
- MultiLabelSoftMarginLoss

### Regression Losses
- MSELoss
- L1Loss
- SmoothL1Loss
- HuberLoss

### Specialized Losses
- CTCLoss (Connectionist Temporal Classification)
- MarginRankingLoss
- TripletMarginLoss
- TripletMarginWithDistanceLoss

## Learning Rate Schedulers

The following learning rate schedulers are available:

- **StepLR** - Decays the learning rate by gamma every step_size epochs
- **MultiStepLR** - Decays the learning rate by gamma when the number of epoch reaches one of the milestones
- **ExponentialLR** - Decays the learning rate by gamma every epoch
- **ReduceLROnPlateau** - Reduces learning rate when a metric has stopped improving
- **CosineAnnealingLR** - Cosine annealing learning rate scheduler

## Usage Examples

### Basic Training

```python
from trainer import Trainer
from optimization import get_optimizer, get_scheduler
from losses import get_loss_function

# Create model
model = YourModel().to(device)

# Get optimizer
optimizer = get_optimizer(
    name='adam',
    model_params=model.parameters(),
    custom_params={'lr': 0.001, 'weight_decay': 1e-4}
)

# Get loss function
criterion = get_loss_function('cross_entropy', device=device)

# Get scheduler (optional)
scheduler = get_scheduler(
    name='reduce_on_plateau',
    optimizer=optimizer,
    custom_params={'mode': 'min', 'factor': 0.1, 'patience': 5}
)

# Initialize trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    metrics=['accuracy'],
    use_amp=True  # Enable mixed precision training
)

# Train the model
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    verbose=1
)
```

### Using Composite Loss

```python
from losses import CompositeLoss

# Define a composite loss with multiple components
criterion = CompositeLoss(
    losses={
        'cross_entropy': {'reduction': 'mean'},
        'l1': {'reduction': 'mean'}
    },
    weights={
        'cross_entropy': 1.0,
        'l1': 0.01  # L1 regularization
    },
    device=device
)
```

## Advanced Topics

### Custom Optimizers

You can create custom optimizers by extending `torch.optim.Optimizer` and registering them in the `OPTIMIZERS` dictionary in `optimization.py`.

### Custom Loss Functions

To add a custom loss function:

1. Implement it as a PyTorch `nn.Module`
2. Add it to the `LOSS_FUNCTIONS` dictionary in `losses.py`
3. Add default parameters to `DEFAULT_LOSS_PARAMS` if needed

### Custom Schedulers

To add a custom scheduler:

1. Implement it following PyTorch's `_LRScheduler` interface
2. Add it to the `SCHEDULERS` dictionary in `optimization.py`
3. Add default parameters to `DEFAULT_SCHEDULER_PARAMS` if needed

## Best Practices

1. **Learning Rate Scheduling**: Always use a learning rate scheduler for better convergence
2. **Gradient Clipping**: Use gradient clipping (especially for RNNs) to prevent exploding gradients
3. **Mixed Precision**: Enable mixed precision training for faster training and reduced memory usage
4. **Gradient Accumulation**: Use gradient accumulation for larger effective batch sizes
5. **Loss Scaling**: When using mixed precision, ensure proper loss scaling is applied

## Troubleshooting

- **NaN/Inf Loss**: Try reducing the learning rate or using gradient clipping
- **Slow Convergence**: Try a different optimizer or adjust learning rate
- **OOM Errors**: Reduce batch size or enable gradient accumulation
- **Training Instability**: Try gradient clipping or a different optimizer

For more examples, see the `examples/optimization_example.py` script.

---
© Copyright 2025 Nsfr750. All Rights Reserved.
