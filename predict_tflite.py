import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Paths
TFLITE_MODEL_PATH = "model.tflite"
IMAGE_PATH = r"D:\git\Handwritten-Character-Recognition1\test-images\1.JPG"

# A-Z character mapping
word_dict = {i: chr(65 + i) for i in range(26)}  # {0: 'A', 1: 'B', ..., 25: 'Z'}

# Check if TFLite model exists
if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"❌ TFLite model file not found: {TFLITE_MODEL_PATH}")
    exit()

# Check if image exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image file not found: {IMAGE_PATH}")
    exit()

# Load and preprocess the image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Failed to load image. Check the file format.")
    exit()

img_gray = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
img_resized = cv2.resize(img_thresh, (28, 28)).reshape(1, 28, 28, 1).astype(np.float32)

# Save the processed images locally
cv2.imwrite("original_loaded.jpg", img)  # original image
cv2.imwrite("gray_blurred.jpg", img_gray)  # grayscale and blurred
cv2.imwrite("thresholded.jpg", img_thresh)  # thresholded image

# To visualize the resized one, scale it back to 255 and remove batch & channel dims
resized_for_display = (img_resized[0] * 255).astype(np.uint8).reshape(28, 28)
cv2.imwrite("resized_28x28.jpg", resized_for_display)  # final input to model

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], img_resized)

# Run inference
interpreter.invoke()

# Get prediction
output = interpreter.get_tensor(output_details[0]['index'])
prediction_index = np.argmax(output)
prediction_character = word_dict[prediction_index]

# Display result
plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {prediction_character}", fontsize=14)
plt.axis("off")
plt.show()

print(f"✅ Predicted Character: {prediction_character}")
