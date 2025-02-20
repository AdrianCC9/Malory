import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os

# Load MobileNetV2 model
model_path = os.path.expanduser("/Users/adrian/models/MobileNetV2")
mobilenet_model = hub.load(model_path)

# Load and preprocess the image
image_path = os.path.expanduser("/Users/adrian/models/MobileNetV2/test_image.jpg")
image = cv2.imread(image_path)

# Ensure image exists
if image is None:
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Resize to MobileNetV2 input size (224x224 pixels)
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Run inference
predictions = mobilenet_model(image)

# Load ImageNet labels
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)

import json
with open(labels_path, "r") as f:
    labels_dict = json.load(f)
labels = np.array([labels_dict[str(i)][1] for i in range(len(labels_dict))])

# Get top 5 predictions
top5_i = np.argsort(predictions.numpy()[0])[-5:][::-1]

# Print results
print("\nTop 5 Predictions:")
for i in top5_i:
    print(f"{labels[i]} ({predictions.numpy()[0][i]:.3f})")
