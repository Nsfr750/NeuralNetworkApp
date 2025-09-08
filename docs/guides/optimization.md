# Model Optimization Guide

This guide covers techniques for optimizing neural network models for better performance and efficiency in NeuralNetworkApp.

## Table of Contents
- [Pruning](#pruning)
- [Quantization](#quantization)
- [Knowledge Distillation](#knowledge-distillation)
- [Architecture Search](#architecture-search)
- [Gradient Clipping](#gradient-clipping)
- [Mixed Precision Training](#mixed-precision-training)
- [Best Practices](#best-practices)

## Pruning

### Magnitude-based Weight Pruning
```python
import tensorflow_model_optimization as tfmot

# Define model
model = build_your_model()

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

# Compile the model
model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with pruning
model_for_pruning.fit(
    train_dataset,
    epochs=10,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
)
```

## Quantization

### Post-training Quantization
```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to quantized version
quantized_tflite_model = converter.convert()

# Save the quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
```

### Quantization-aware Training
```python
import tensorflow_model_optimization as tfmot

# Apply quantization to the model
quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for quantization-aware
q_aware_model = quantize_model(model)

# Compile the model
q_aware_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
q_aware_model.fit(train_dataset, epochs=5)
```

## Knowledge Distillation

### Teacher-Student Training
```python
# Teacher model (larger, more accurate)
teacher = build_teacher_model()
teacher.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train teacher model
teacher.fit(train_dataset, epochs=10)

# Student model (smaller, to be trained with teacher's knowledge)
student = build_student_model()

# Knowledge distillation loss
def distillation_loss(y_true, y_pred, teacher_logits, temp=2.0):
    return tf.keras.losses.kl_divergence(
        tf.nn.softmax(teacher_logits/temp, axis=1),
        tf.nn.softmax(y_pred/temp, axis=1)
    ) * (temp ** 2)

# Compile student with both losses
student.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train student with teacher's predictions
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Get teacher predictions
        teacher_pred = teacher(x, training=False)
        # Get student predictions
        student_pred = student(x, training=True)
        # Calculate losses
        task_loss = tf.keras.losses.sparse_categorical_crossentropy(y, student_pred)
        distill_loss = distillation_loss(y, student_pred, teacher_pred)
        # Combine losses
        loss = 0.5 * task_loss + 0.5 * distill_loss
    
    # Apply gradients
    grads = tape.gradient(loss, student.trainable_variables)
    student.optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return loss
```

## Architecture Search

### Neural Architecture Search (NAS)
```python
import autokeras as ak

# Initialize the model
input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ConvBlock()(output_node)
output_node = ak.ClassificationHead()(output_node)

# Initialize the searcher
clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    max_trials=10,  # Number of different models to try
    overwrite=True,
    objective='val_accuracy'
)

# Search for the best model
clf.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10
)

# Get the best model
best_model = clf.export_model()
```

## Gradient Clipping

### Global Norm Clipping
```python
# In model.compile()
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0  # Clip gradients by norm
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Value Clipping
```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipvalue=0.5  # Clip gradients by value
)
```

## Mixed Precision Training

### Enabling Mixed Precision
```python
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')

# Now build and compile your model
model = build_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Best Practices

### 1. Start Simple
- Begin with a baseline model
- Gradually add complexity
- Profile performance at each step

### 2. Monitor Key Metrics
- Training/validation loss
- Memory usage
- Training time per batch/epoch
- GPU/TPU utilization

### 3. Optimization Workflow
1. Profile the model
2. Identify bottlenecks
3. Apply optimizations
4. Validate accuracy
5. Measure speedup
6. Repeat if necessary

### 4. Common Pitfalls
- Over-optimizing too early
- Ignoring accuracy impact
- Not measuring baseline performance
- Forgetting about deployment constraints

## Common Issues and Solutions

### Model Accuracy Drops After Optimization
- Try different optimization techniques
- Fine-tune hyperparameters
- Use a smaller compression ratio
- Apply optimizations gradually

### Optimization Doesn't Improve Performance
- Check for I/O bottlenecks
- Profile with different batch sizes
- Verify hardware acceleration
- Update to latest framework version

### Model Size Not Reduced as Expected
- Check which layers are being optimized
- Try different optimization parameters
- Consider model architecture changes
- Check for non-trainable weights

---
© Copyright 2025 Nsfr750. All Rights Reserved.
