# Model Deployment Guide

This guide covers various methods for deploying NeuralNetworkApp models in production environments.

## Table of Contents
- [Model Export](#model-export)
- [TensorFlow Serving](#tensorflow-serving)
- [REST API with FastAPI](#rest-api-with-fastapi)
- [Docker Deployment](#docker-deployment)
- [Edge Deployment](#edge-deployment)
- [Monitoring and Logging](#monitoring-and-logging)
- [Best Practices](#best-practices)

## Model Export

### Save Model in SavedModel Format
```python
# Save the entire model
model.save('path_to_saved_model')

# Load the model
loaded_model = tf.keras.models.load_model('path_to_saved_model')

# Save only the weights
model.save_weights('model_weights.h5')

# Save model architecture as JSON
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)
```

### Convert to TensorFlow Lite (for mobile/edge)
```python
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# For quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
```

## TensorFlow Serving

### Install TensorFlow Serving
```bash
# For Ubuntu/Debian
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
```

### Serve a Model
```bash
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=your_model \
  --model_base_path=/path/to/saved_model
```

### Make Predictions
```python
import requests
import json

# Prepare data
data = {
    "instances": x_test[:3].tolist()
}

# Make prediction
response = requests.post(
    'http://localhost:8501/v1/models/your_model:predict',
    data=json.dumps(data)
)

# Get predictions
predictions = json.loads(response.text)['predictions']
```

## REST API with FastAPI

### FastAPI Server
```python
from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model('path_to_saved_model')

# Preprocessing function
def preprocess_image(image):
    image = np.array(Image.open(io.BytesIO(image)))
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image.numpy()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image = await file.read()
    image = preprocess_image(image)
    
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    
    # Make prediction
    predictions = model.predict(image)
    predicted_class = int(tf.argmax(predictions[0]))
    
    return {"class_id": predicted_class, "confidence": float(predictions[0][predicted_class])}

# Run with: uvicorn main:app --reload
```

## Docker Deployment

### Dockerfile for FastAPI
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

### Build and Run
```bash
# Build the Docker image
docker build -t neuralnetworkapp .

# Run the container
docker run -p 80:80 neuralnetworkapp
```

## Edge Deployment

### Convert to TensorFlow.js
```bash
# Install the converter
pip install tensorflowjs

# Convert the model
tensorflowjs_converter \
    --input_format=keras \
    path/to/your/model.h5 \
    path/to/output/folder
```

### Use in Browser
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.0.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.0.4"></script>

<script>
  // Load the model
  async function loadModel() {
    const model = await tf.loadLayersModel('model.json');
    return model;
  }
  
  // Make prediction
  async function predict() {
    const model = await loadModel();
    const img = document.getElementById('image');
    const tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();
      
    const prediction = await model.predict(tensor).data();
    console.log(prediction);
  }
</script>
```

## Monitoring and Logging

### TensorBoard Integration
```python
# In your training code
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    update_freq='batch'
)

# Start TensorBoard
tensorboard --logdir=logs/fit
```

### Prometheus Metrics
```python
from prometheus_client import start_http_server, Summary, Counter
import random
import time

# Create metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions')

@REQUEST_TIME.time()
def process_request(t):
    # Simulate prediction
    time.sleep(t)
    PREDICTION_COUNTER.inc()

# Start up the server to expose the metrics.
start_http_server(8000)

# Generate some requests
while True:
    process_request(random.random())
```

## Best Practices

### 1. Model Optimization
- Quantize models for faster inference
- Use TensorRT for NVIDIA GPUs
- Optimize input pipeline

### 2. Security
- Validate all inputs
- Rate limit API endpoints
- Use HTTPS
- Implement authentication

### 3. Scalability
- Use Kubernetes for orchestration
- Implement auto-scaling
- Use message queues for batch processing

### 4. Monitoring
- Track model performance
- Monitor resource usage
- Set up alerts for anomalies
- Log predictions for analysis

## Common Issues and Solutions

### High Latency
- Optimize model architecture
- Use a faster hardware accelerator
- Implement caching for frequent requests
- Use batch processing

### Memory Issues
- Reduce batch size
- Use model quantization
- Implement memory profiling
- Use streaming for large inputs

### Versioning
- Keep track of model versions
- Implement A/B testing
- Maintain backward compatibility
- Document model changes

---
© Copyright 2025 Nsfr750. All Rights Reserved.
