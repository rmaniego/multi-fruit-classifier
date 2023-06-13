import os
import cv2
import logging
import numpy as np
from PIL import Image

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
from tensorflow.keras.callbacks import Callback 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical

"""
RESOURCES:
 - https://github.com/daveluo/coconuts
 - https://github.com/EvidenceN/multi-fruit-classification/blob/master/fruit_model.ipynb
 - https://medium.com/@canerkilinc/hands-on-tensorflow-2-0-multi-label-classifications-with-mlp-88fc97d6a7e6
 - https://www.tensorflow.org/tutorials/keras/classification
"""

# GLOBAL CONSTANTS
# EPOCHS: 100-200, adjust per use-case (TESTING=5; MAX=5000)
EPOCHS = 30
TRAINING_WIDTH = 150
TRAINING_HEIGHT = 150
TRAINING_PATH = "datasets/training"
VALIDATION_PATH = "datasets/validation"
VALIDATION_ACCURACY = 0.98


def main():
    """
    Initialize and preprocess the dataset.
    """
    global EPOCHS
    global TRAINING_WIDTH
    global TRAINING_HEIGHT
    global TRAINING_PATH

    training_labels = []

    print("\nPreprocessing the dataset...")
    target_size = (TRAINING_HEIGHT, TRAINING_WIDTH)
    
    data_generator = ImageDataGenerator(rescale = 1./255, validation_split=0.2, preprocessing_function=RGB2HSV)
    dataset1 = data_generator.flow_from_directory(TRAINING_PATH, target_size=target_size, shuffle=True, subset="training", class_mode="categorical")
    dataset2 = data_generator.flow_from_directory(TRAINING_PATH, target_size=target_size, shuffle=True, subset="validation", class_mode="categorical")
    
    fruits = list(os.listdir(TRAINING_PATH))
    dataset_size = len(dataset1)
    classes = len(fruits)
    

    print("\nLoading the model...")
    if not os.path.exists("models"):
        os.mkdir("models")

    # Format model name based on preprocessing data.
    model_filepath = f"models/{classes}_{dataset_size}_{TRAINING_WIDTH}x{TRAINING_HEIGHT}_{EPOCHS}.h5"
    model = load_model_from_file(model_filepath)
    if model is None:
        model = train_new_model(
            classes, dataset1, dataset2, model_filepath
        )

    dataset1 = None
    dataset2 = None
    print("\nPredicting unseen images...")
    model_predict(model, fruits)


def load_model_from_file(model_filepath):
    """
    Return the specified model if it exists.
    """
    if os.path.exists(model_filepath):
        return load_model(model_filepath)


def train_new_model(classes, dataset1, dataset2, model_filepath):
    """
    Prepare and train new model based on updated dataset.
    """
    global EPOCHS
    global TRAINING_WIDTH
    global TRAINING_HEIGHT

    model = Sequential()

    # In a Convolutional Layer, a NxM kernel slides to each possible location in the 2D input.
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

    # Max Pooling is used to downsample and reducing spatial dimensions of feature maps.
    # It divides the feature map into non-overlapping regions and chooses the maximum value for each.
    # Simply, it looks for the most important parts and reduces the data size for improved processing.
    # Larger pool size may lead to less detailed feature maps.
    # Pool Size: 2x2 and 3x3 (most common)

    model.add(
        Conv2D(
            64,
            kernel_size=(3,3),
            activation="relu",
            input_shape=(TRAINING_HEIGHT, TRAINING_WIDTH, 3),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
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

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))

    # Output Layer
    # SoftMax softmax(z_i) = exp(z_i) / sum(exp(z_j)) for all j
    # It is used in the output layer for multi-class classifications,
    #   transforming raw scores of the previous layer into a probability distribution
    model.add(Dense(classes, activation="softmax"))

    # Categorical Cross-Entropy is used to minimize the difference
    #   between the predicted probabilities and the encoded labels
    #   to make more accurate predictions and assign higher probabilities to the correct classes.
    # AdaM vs RMSprop
    #  - AdaM / Adaptive Moment Estimation, tends to reach an optimal solution faster, slower vs RMSprop.
    #  - Root Mean Squared Propagation (RMSprop) with momentum reaches much further before it changes direction.
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.summary()
    
    callback = StopEarly()

    model.fit(
        dataset1,
        epochs=EPOCHS,
        validation_data=dataset2,
        batch_size=64,
        callbacks = [callback],
        workers=10
    )
    model.save(model_filepath)

    # If the training accuracy is high, but valdiation accuracy is very low:
    #   a.) The model might be too complex and is too specialized to the training data.
    #   b.) The training data is insufficient to see diverse patterns for learning.
    #   c.) Dropout value is too low, resulting to poor generalizations.
    #   d.) Data might be siginificantly different from the training data.
    #   e.) Hyperparameters need more fine tuning for more optimal values.

    return model

def model_predict(model, fruits):
    """
    Generate model prediction on validation images.
    """

    global VALIDATION_PATH
    global TRAINING_WIDTH
    global TRAINING_HEIGHT

    results = []
    target_size = (TRAINING_WIDTH, TRAINING_HEIGHT)
    for label, fruit in enumerate(fruits):
        folder = f"{VALIDATION_PATH}/{fruit}"
        for filename in os.listdir(folder):
            filepath = f"{folder}/{filename}"
            if not os.path.isfile(filepath) or filename.split(".")[-1].lower() != "jpg":
                # skip non-jpeg files
                continue

            image = RGB2HSV(img_to_array(load_img(filepath, target_size=target_size)))

            # Reshape the input image appropriately for the prediction model
            reshaped = np.vstack([np.expand_dims(image, axis=0)])
            prediction = model.predict(reshaped)[0]

            # ArgMax, gets the index of the highest probability in the prediction array.
            predicted = fruits[np.argmax(prediction)]
            results.append(f" * {fruit} = {predicted}")

    print("\n[RESULTS]")
    for result in results:
        print(result)

""" UTILS """

class StopEarly(Callback):
    
    global VALIDATION_ACCURACY
    
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("val_acc", 0) >= VALIDATION_ACCURACY):
            print("Accuracy reached upper threshold, stopping model training...")
            self.model.stop_training=True


def RGB2HSV(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)


if __name__ == "__main__":
    main()