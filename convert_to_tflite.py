import tensorflow as tf
import os

# Define file paths
MODEL_PATH = "model.h5"
TFLITE_MODEL_PATH = "model.tflite"

# Check if the Keras model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    exit()

# Load the trained Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ Model successfully converted to TensorFlow Lite: {TFLITE_MODEL_PATH}")
