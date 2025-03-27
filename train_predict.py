import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_data():
    data_path = r"D:\git\Handwritten-Character-Recognition1\archive\A_Z Handwritten_Data.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ CSV file not found at: {data_path}")
        return None
    
    data = pd.read_csv(data_path).astype('float32')
    print(data.head(10))
    
    X = data.drop('0', axis=1)
    y = data['0'].astype(int)  # Convert to integers

    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_images(train_x, test_x):
    train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28, 1))
    test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28, 1))
    return train_x, test_x

def preprocess_labels(train_y, test_y):
    return to_categorical(train_y, num_classes=26), to_categorical(test_y, num_classes=26)

def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='valid'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(26, activation="softmax")
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model(model, train_X, train_yOHE, test_X, test_yOHE):
    # checkpoint_path = "training/cp.ckpt"
    checkpoint_path = "training/cp.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    
    model.fit(train_X, train_yOHE, epochs=1, validation_data=(test_X, test_yOHE), callbacks=[checkpoint_callback])
    model.save('model.h5')

def load_and_evaluate_model(test_X, test_yOHE):
    if not os.path.exists('model.h5'):
        print("❌ Model file 'model.h5' not found.")
        return None
    
    model_saved = keras.models.load_model('model.h5')
    loss, acc = model_saved.evaluate(test_X, test_yOHE, verbose=2)
    print(f'✅ Model Accuracy: {100 * acc:.2f}%')

    return model_saved

def predict_from_image(model_saved, word_dict):
    img_path = r"D:\git\Handwritten-Character-Recognition1\test-images\1.JPG"
    
    if not os.path.exists(img_path):
        print(f"❌ Image file not found at: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image. Check file format.")
        return
    
    img_gray = cv2.cvtColor(cv2.GaussianBlur(img, (7,7), 0), cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    img_final = cv2.resize(img_thresh, (28,28)).reshape(1,28,28,1)
    
    prediction = word_dict[np.argmax(model_saved.predict(img_final))]
    
    cv2.putText(img, "Prediction: " + prediction, (20,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,30), 2)
    cv2.imshow('Handwritten Character Recognition', img)
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to close
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    word_dict = {i: chr(65 + i) for i in range(26)}  # A-Z dictionary

    train_x, test_x, train_y, test_y = load_data()
    
    if train_x is not None:
        train_X, test_X = preprocess_images(train_x, test_x)
        train_yOHE, test_yOHE = preprocess_labels(train_y, test_y)
        
        model = build_model()
        train_model(model, train_X, train_yOHE, test_X, test_yOHE)
        
        model_saved = load_and_evaluate_model(test_X, test_yOHE)
        
        if model_saved:
            predict_from_image(model_saved, word_dict)
