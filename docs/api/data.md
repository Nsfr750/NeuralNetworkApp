# Data API Reference

This document provides a comprehensive reference for the data handling and preprocessing utilities in NeuralNetworkApp.

## Table of Contents
- [Data Generators](#data-generators)
- [Data Augmentation](#data-augmentation)
- [Data Loading](#data-loading)
- [Preprocessing](#preprocessing)

## Data Generators

### `ImageDataGenerator`
A flexible image data generator with real-time data augmentation.

```python
from neuralnetworkapp.data import ImageDataGenerator

# Create data generator with augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Flow from directory
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

### `CSVDataGenerator`
Generator for handling tabular data from CSV files.

```python
from neuralnetworkapp.data import CSVDataGenerator

generator = CSVDataGenerator(
    'data.csv',
    x_cols=['feature1', 'feature2', 'feature3'],
    y_col='target',
    batch_size=32,
    shuffle=True
)
```

## Data Augmentation

### Image Augmentation
```python
from neuralnetworkapp.data.augmentation import (
    random_rotation,
    random_shift,
    random_zoom,
    random_flip,
    random_brightness
)

# Apply augmentation to a batch of images
augmented_images = random_rotation(images, max_angle=20)
augmented_images = random_flip(augmented_images, mode='horizontal')
augmented_images = random_brightness(augmented_images, max_delta=0.2)
```

### Audio Augmentation
```python
from neuralnetworkapp.data.audio import (
    add_noise,
    time_stretch,
    pitch_shift,
    shift_time
)

# Apply audio augmentation
augmented_audio = add_noise(audio, noise_level=0.005)
augmented_audio = time_stretch(augmented_audio, rate=1.1)
```

## Data Loading

### `load_image_dataset`
Load and preprocess an image dataset from a directory.

```python
from neuralnetworkapp.data import load_image_dataset

# Load dataset
train_ds, val_ds = load_image_dataset(
    'data/train',
    validation_split=0.2,
    image_size=(224, 224),
    batch_size=32,
    seed=42
)
```

### `load_tabular_data`
Load and preprocess tabular data from CSV or pandas DataFrame.

```python
from neuralnetworkapp.data import load_tabular_data

# Load from CSV
X, y = load_tabular_data(
    'data.csv',
    target='label',
    test_size=0.2,
    random_state=42
)

# Or from pandas DataFrame
import pandas as pd
df = pd.read_csv('data.csv')
X, y = load_tabular_data(
    df,
    target='label',
    test_size=0.2
)
```

## Preprocessing

### Image Preprocessing
```python
from neuralnetworkapp.data.preprocessing import (
    normalize_images,
    standardize_images,
    resize_images,
    rgb_to_grayscale
)

# Example usage
normalized = normalize_images(images, min_val=0, max_val=1)
resized = resize_images(images, target_size=(224, 224))
gray = rgb_to_grayscale(images)
```

### Text Preprocessing
```python
from neuralnetworkapp.data.text import (
    Tokenizer,
    pad_sequences,
    text_to_sequences,
    create_embedding_matrix
)

# Initialize tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

# Convert text to sequences
sequences = text_to_sequences(tokenizer, texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Create embedding matrix
embedding_matrix = create_embedding_matrix(
    tokenizer.word_index,
    'glove.6B.100d.txt',
    embedding_dim=100
)
```

### Time Series Preprocessing
```python
from neuralnetworkapp.data.timeseries import (
    create_sequences,
    normalize_series,
    split_sequence,
    create_lookback_dataset
)

# Create sequences for time series prediction
X, y = create_sequences(series, n_steps=10)

# Normalize time series data
normalized_series = normalize_series(series)

# Create lookback dataset
X, y = create_lookback_dataset(series, look_back=5, forecast_horizon=1)
```

## Best Practices

1. **Data Pipeline**
   - Use `tf.data` for efficient data loading
   - Cache datasets in memory when possible
   - Prefetch data to avoid I/O bottlenecks

2. **Reproducibility**
   - Set random seeds
   - Use deterministic operations
   - Save preprocessing parameters

3. **Performance**
   - Use vectorized operations
   - Enable parallel processing
   - Profile data loading pipeline

## Common Issues and Solutions

### Memory Errors
- Reduce batch size
- Use generators instead of loading all data at once
- Enable memory growth for GPU

### Slow Data Loading
- Use `tf.data.Dataset.cache()`
- Enable prefetching
- Use `tf.data.Dataset.interleave()` for I/O parallelism

---
© Copyright 2025 Nsfr750. All Rights Reserved.
