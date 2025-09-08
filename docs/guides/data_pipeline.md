# Data Pipeline Guide

This guide covers the data loading, preprocessing, and augmentation pipeline in NeuralNetworkApp.

## Table of Contents
- [Data Loading](#data-loading)
- [Data Preprocessing](#data-preprocessing)
- [Data Augmentation](#data-augmentation)
- [Creating Custom Datasets](#creating-custom-datasets)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

## Data Loading

### Built-in Datasets
```python
from neuralnetworkapp.data import get_mnist_data, get_cifar10_data, get_imdb_data

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = get_mnist_data()

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = get_cifar10_data()

# Load IMDB sentiment analysis dataset
(x_train, y_train), (x_test, y_test) = get_imdb_data()
```

### Custom Data Loading
```python
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_custom_data(data_path):
    # Load and preprocess your data here
    # Return (x_train, y_train), (x_test, y_test)
    pass
```

## Data Preprocessing

### Image Preprocessing
```python
from neuralnetworkapp.data import preprocess_image

# Basic preprocessing
image = preprocess_image(
    image,
    normalize=True,      # Normalize to [0,1]
    resize=(224, 224),   # Resize image
    grayscale=False      # Convert to grayscale
)
```

### Text Preprocessing
```python
from neuralnetworkapp.data import preprocess_text

text = "This is a sample text."
processed_text = preprocess_text(
    text,
    lower=True,         # Convert to lowercase
    remove_punct=True,  # Remove punctuation
    remove_stopwords=True  # Remove stopwords
)
```

## Data Augmentation

### Image Augmentation
```python
from neuralnetworkapp.data import ImageAugmenter

# Create augmenter with desired transformations
augmenter = ImageAugmenter(
    rotation_range=20,      # Random rotation between -20 and 20 degrees
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2, # Random vertical shift
    horizontal_flip=True,   # Random horizontal flip
    zoom_range=0.2          # Random zoom
)

# Apply augmentation
augmented_image = augmenter.augment(image)
```

### Text Augmentation
```python
from neuralnetworkapp.data import TextAugmenter

augmenter = TextAugmenter(
    synonym_replacement=True,  # Replace words with synonyms
    random_deletion=True,      # Randomly delete words
    random_swap=True,          # Randomly swap words
    random_insertion=True      # Randomly insert synonyms
)

augmented_text = augmenter.augment("This is a sample text.")
```

## Creating Custom Datasets

### Using tf.data.Dataset
```python
import tensorflow as tf

def create_dataset(images, labels, batch_size=32, augment=False):
    # Create dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # Apply augmentation if needed
    if augment:
        dataset = dataset.map(
            lambda x, y: (augmenter.augment(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Prefetch for better performance
    return dataset.prefetch(tf.data.AUTOTUNE)
```

## Performance Optimization

### Caching
```python
# Cache after loading
dataset = dataset.cache()
```

### Parallel Processing
```python
# Parallelize data loading
dataset = dataset.map(
    preprocess_function,
    num_parallel_calls=tf.data.AUTOTUNE
)
```

### Prefetching
```python
# Prefetch data to GPU/CPU
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## Best Practices

1. **Data Splitting**
   - Always split data into train/validation/test sets
   - Use stratified sampling for imbalanced datasets

2. **Memory Management**
   - Use generators for large datasets
   - Clear unused variables with `del`
   - Use `gc.collect()` when needed

3. **Reproducibility**
   - Set random seeds
   - Document all preprocessing steps
   - Save preprocessed data for future use

4. **Data Versioning**
   - Version your datasets
   - Keep track of preprocessing steps
   - Document data sources and licenses

5. **Monitoring**
   - Log data statistics
   - Visualize samples
   - Monitor class distribution

## Common Issues and Solutions

### Out of Memory Errors
- Reduce batch size
- Use data generators
- Enable memory growth

### Slow Data Loading
- Use `tf.data` prefetching
- Enable parallel loading
- Consider using TFRecord format

### Data Leakage
- Ensure proper data splitting
- Avoid test set contamination
- Use time-based splits for time series

---
© Copyright 2025 Nsfr750. All Rights Reserved.
