import os
import cv2
import PIL
import logging
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Suppress TF logs
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# GLOBAL CONSTANTS
# 100-200, adjust per use-case
EPOCHS = 100
TRAINING_WIDTH = 100
TRAINING_HEIGHT = 100
TRAINING_PATH = "datasets/training"
VALIDATION_PATH = "datasets/validation"


def main():
    """
    Initialize and preprocess the dataset.
    """
    global EPOCHS
    global TRAINING_WIDTH
    global TRAINING_HEIGHT
    global TRAINING_PATH

    # Rotations: 90, 180, 270
    augmented_labels = []
    rotations = [
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ]

    print("\nPreprocessing the dataset...")
    training_images = []
    fruits = sorted(os.listdir(TRAINING_PATH))
    for label, fruit in enumerate(fruits):

        folder = f"{TRAINING_PATH}/{fruit}"

        for filename in os.listdir(folder):
            filepath = f"{folder}/{filename}"
            if not os.path.isfile(filepath) or filename.split(".")[-1] != "jpg":
                continue

            image = img_to_array(load_img(filepath, target_size=(TRAINING_HEIGHT, TRAINING_WIDTH)))

            # Normalize the BGR pixels (0-1)
            image = image / 255.0
            training_images.append(image)
            augmented_labels.append(label)

            # for rotation in rotations:
            #     training_images.append(cv2.rotate(image, rotation))
            #    augmented_labels.append(label)

    classes = len(fruits)
    training_images = np.array(training_images)
    training_dataset_size = len(training_images)
    training_labels = to_categorical(np.array(augmented_labels), num_classes=classes)
    augmented_labels = None
    
    print("\nLoading the model...")
    if not os.path.exists("models"):
        os.mkdir("models")

    # Format model name based on preprocessing data.
    model_file_path = f"models/{classes}_{training_dataset_size}_{TRAINING_WIDTH}x{TRAINING_HEIGHT}_{EPOCHS}.h5"
    model = load_model_from_file(model_file_path)
    if model is None:
        model = train_new_model(
            classes, training_images, training_labels, model_file_path
        )

    training_images = None
    print("\nPredicting unseen images...")
    model_predict(model, fruits)


def load_model_from_file(model_file_path):
    """
    Return the specified model if it exists.
    """
    if os.path.exists(model_file_path):
        return load_model(model_file_path)


def train_new_model(classes, training_images, training_labels, model_file_path):
    """
    Prepare and train new model based on updated dataset.
    """
    global EPOCHS
    global TRAINING_WIDTH
    global TRAINING_HEIGHT

    model = Sequential()
    
    # In a Convolutional Model, a NxM kernel slides to each possible location in the 2D input.
    # The kernel (of weights) performs an element-wise multiplication and summing all values into one.
    # The N-kernels will generate N-maps with unique features extracted from the input image.
    # Kernel Sizes: 3x3 (common), 5x5 (suitable for small features), 7x7 or 9x9 (appropriate for larger features)

    # Rectified Linear Unit ReLU(x) = max(0, x)
    # Any negative value becomes zero, addressing the gradients/derivatives
    #   from becoming very small and providing less effective learning
    # ReLU sets all negative values in the feature maps to zero
    #   introducing non-linearity to help in learning complex patterns and relationships
    
    # Batch Normalization helps accelerate and stabilize the training process
    #   by normalizing the activation after the Convolutional Layer.
    # Each feature map is independently normalized.
    
    # Max Pooling is used to downsample and reducing spatial dimensiosn of feature maps.
    # It divides the feature map into non-overlapping regions and chooses the maximum value for each.
    # Simply, it looks for the most important parts and reduces the data size for improved processing.

    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(TRAINING_HEIGHT, TRAINING_WIDTH, 3),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # The complex feature maps must be flattened
    #   before feeding to the Dense layers, since it only accepts 1D arrays.
    model.add(Flatten())
    
    # Dense is a layer where each neuron is fully connected to the previous layer.
    # It means that each neuron accepts the full output of the previous layer.

    # Dropout is typically applied after the fully connected layers.
    # Value range from 0.2-0.5, with 0.5 as ideal to avoid overfitting in smaller datasets.

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output Layer
    # SoftMax softmax(z_i) = exp(z_i) / sum(exp(z_j)) for all j
    # It is used in the output layer for multi-class classifications,
    #   transforming raw scores of the previous layer into a probability distribution
    model.add(Dense(classes, activation="softmax"))

    # AdaM / Adaptive Moment Estimation
    # AdaM tends to reach an optimal solution faster.
    # Categorical Cross-Entropy is used to minimize the difference
    #   between the predicted probabilities and the encoded labels
    #   to make more accurate predictions and assign higher probabilities to the correct classes.
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    training_images = np.array(training_images)
    model.fit(
        training_images,
        training_labels,
        epochs=EPOCHS,
        batch_size=64,
        validation_split=0.2,
    )
    model.save(model_file_path)

    return model


def model_predict(model, fruits):
    """
    Generate model prediction on validation images.
    """

    global VALIDATION_PATH
    global TRAINING_WIDTH
    global TRAINING_HEIGHT

    results = []
    for label, fruit in enumerate(fruits):
        folder = f"{VALIDATION_PATH}/{fruit}"
        for filename in os.listdir(folder):
            filepath = f"{folder}/{filename}"
            if not os.path.isfile(filepath) or filename.split(".")[-1].lower() != "jpg":
                # skip non-jpeg files
                continue

            image = img_to_array(
                load_img(filepath, target_size=(TRAINING_WIDTH, TRAINING_HEIGHT))
            )
            
            reshaped = np.reshape([image], (1, TRAINING_HEIGHT, TRAINING_WIDTH, 3))
            prediction = model.predict(reshaped)[0]
            predicted = fruits[np.argmax(prediction)]
            
            results.append(f" * {fruit} = {predicted}")

    print("\n[RESULTS]")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()