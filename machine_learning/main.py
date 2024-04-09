from typing import Dict, Optional, Tuple
import flwr as fl

import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing import image
import tensorflow as tf

# Define constants
classes = ["head", "hardhat"]
class_labels = {classes[i]: i for i in range(len(classes))}
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20

# Load MobileNetV2 base model without top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Add custom classifier on top of the base model
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(classes), activation='softmax')(x)

# Combine base model with custom classifier
model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def load_dataset():
    directory = "machine_learning/datasets"
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for label in classes:
        train_path = os.path.join(directory, "train", label)
        test_path = os.path.join(directory, "test", label)
        class_index = class_labels[label]

        # Load train images and labels
        for file in os.listdir(train_path):
            img_path = os.path.join(train_path, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            train_images.append(image)
            train_labels.append(class_index)

        # Load test images and labels
        for file in os.listdir(test_path):
            img_path = os.path.join(test_path, file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            test_images.append(image)
            test_labels.append(class_index)

    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    test_images = np.array(test_images, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int32)

    return (train_images, train_labels), (test_images, test_labels)

def main() -> None:
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    # Train model
    model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

    # Evaluate model
    loss, accuracy = model.evaluate(test_images, test_labels)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

if __name__ == "__main__":
    main()
