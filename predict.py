import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.models import load_model

# Load trained model
MODEL_PATH = "model.h5"
IMAGE_PATH = r"D:\git\Handwritten-Character-Recognition1\test-images\2.JPG"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    exit()

model = load_model(MODEL_PATH)

# A-Z character mapping
word_dict = {i: chr(65 + i) for i in range(26)}  # {0: 'A', 1: 'B', ..., 25: 'Z'}

# Load and preprocess image
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image file not found: {IMAGE_PATH}")
    exit()

img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Failed to load image. Check the file format.")
    exit()

# Preprocess the image
img_gray = cv2.cvtColor(cv2.GaussianBlur(img, (7,7), 0), cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
img_final = cv2.resize(img_thresh, (28,28)).reshape(1,28,28,1)

# Make prediction
prediction_index = np.argmax(model.predict(img_final))
prediction_character = word_dict[prediction_index]

# Display the image with prediction
plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
plt.title(f"Prediction: {prediction_character}", fontsize=14)
plt.axis("off")  # Hide axes
plt.show()

print(f"✅ Predicted Character: {prediction_character}")
