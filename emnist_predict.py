import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, model_from_json

# Load the model
def load_trained_model():
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('best_model.h5')
    print("Model successfully loaded")
    return model

# Define character set (as per EMNIST)
characters = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

# Segment and classify characters
def segment_and_classify(image_path, model):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    height, width, _ = image.shape

    # Resize image for better segmentation
    image = cv2.resize(image, dsize=(width * 5, height * 4), interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary inverse thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilation
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Gaussian blur
    blurred = cv2.GaussianBlur(dilated, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    results = []
    cropped_images = []

    for ctr in sorted_contours:
        x, y, w, h = cv2.boundingRect(ctr)
        roi = image[y - 10:y + h + 10, x - 10:x + w + 10]
        roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        roi = roi.astype("float32") / 255.0
        roi = 1 - roi  # Invert
        roi = roi.reshape(1, 784)

        prediction = model.predict(roi)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_char = characters[predicted_class]

        cropped_images.append(roi.reshape(28, 28))
        results.append(predicted_char)

    # Display results
    fig, axs = plt.subplots(nrows=len(cropped_images), figsize=(2, len(cropped_images)))
    if len(cropped_images) == 1:
        axs = [axs]  # Make it iterable
    for i, img in enumerate(cropped_images):
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f'Predicted: {results[i]}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

    predicted_string = ''.join(results)
    print("Predicted String:", predicted_string)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Segment and classify characters from an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image (e.g., example.png)')
    args = parser.parse_args()

    model = load_trained_model()
    segment_and_classify(args.image_path, model)
