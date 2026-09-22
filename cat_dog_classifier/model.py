"""
cat_dog_classifier / model.py
------------------------------
This module defines the Convolutional Neural Network (CNN) architecture from scratch.

WHAT IS A CNN?
A Convolutional Neural Network (CNN) is a deep learning architecture designed 
to automatically learn spatial hierarchies of features from grid-structured 
data such as images.

WHY NO TRANSFER LEARNING / PRETRAINED MODELS?
We construct this network completely from scratch using basic Keras layers 
(Conv2D, MaxPooling2D, Flatten, Dense, Dropout) so that every operation, 
tensor shape transformation, and hyperparameter can be explained during a B.Tech viva.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_cat_dog_cnn(input_shape=(128, 128, 3), learning_rate=0.001):
    """
    Builds and compiles a simple CNN architecture from scratch for binary image classification.

    Parameters:
    -----------
    input_shape : tuple of int
        Dimensions of the input image (Height, Width, Channels). Default is (128, 128, 3).
        3 represents the RGB color channels (Red, Green, Blue).
    learning_rate : float
        Step size used by the Adam optimizer during backpropagation. Default is 0.001.

    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras Sequential model ready for training.
    """

    model = models.Sequential(name="Custom_Cat_vs_Dog_CNN")

    # =========================================================================
    # CONVOLUTIONAL BLOCK 1
    # =========================================================================
    # Conv2D Layer:
    # Applies 32 learnable filters (kernels) of size (3x3) over the 128x128x3 input image.
    # Why Convolution? Instead of treating pixel locations independently, convolution
    # preserves spatial relationships and extracts low-level features like edges, corners, and lines.
    # Output shape: (126, 126, 32) if padding='valid' -> (128, 128, 32) if padding='same'.
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        input_shape=input_shape,
        name="conv2d_1"
    ))

    # ReLU Activation (Rectified Linear Unit):
    # Formula: f(x) = max(0, x)
    # Why ReLU? Introduces non-linearity into the network, enabling it to learn complex
    # non-linear patterns. It is computationally efficient and helps solve the vanishing gradient problem.
    model.add(layers.Activation('relu', name="relu_1"))

    # MaxPooling2D Layer:
    # Downsamples feature maps by taking the maximum value in each (2x2) window.
    # Why Pooling? Reduces spatial size (height & width by 50%), cuts down computational cost,
    # and provides translation invariance (slight shifts in feature position won't alter recognition).
    # Input: (128, 128, 32) -> Output: (64, 64, 32)
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpooling_1"))

    # =========================================================================
    # CONVOLUTIONAL BLOCK 2
    # =========================================================================
    # Increases filters to 64 to learn mid-level features (e.g., textures, cat/dog whiskers, fur patterns).
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        name="conv2d_2"
    ))
    model.add(layers.Activation('relu', name="relu_2"))
    # Input: (64, 64, 64) -> Output: (32, 32, 64)
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpooling_2"))

    # =========================================================================
    # CONVOLUTIONAL BLOCK 3
    # =========================================================================
    # Increases filters to 128 to learn high-level semantic features (e.g., ear shapes, snouts, eyes).
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        name="conv2d_3"
    ))
    model.add(layers.Activation('relu', name="relu_3"))
    # Input: (32, 32, 128) -> Output: (16, 16, 128)
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpooling_3"))

    # =========================================================================
    # FLATTEN LAYER
    # =========================================================================
    # Converts 3D feature maps (16 x 16 x 128 = 32,768 elements) into a 1D vector.
    # Why Flatten? Dense (Fully Connected) layers require a 1D vector as input.
    model.add(layers.Flatten(name="flatten"))

    # =========================================================================
    # DENSE (FULLY CONNECTED) HIDDEN LAYER
    # =========================================================================
    # Dense Layer with 128 neurons. Combines all spatial features extracted by CNN blocks
    # to form higher-level representations for classification.
    model.add(layers.Dense(units=128, activation='relu', name="dense_1"))

    # Dropout Layer (Rate = 0.5):
    # Randomly sets 50% of input units to 0 during each training update.
    # Why Dropout? Prevents overfitting by forcing neurons to learn redundant, robust features
    # instead of relying heavily on specific co-adapted weights.
    model.add(layers.Dropout(rate=0.5, name="dropout_1"))

    # =========================================================================
    # OUTPUT LAYER (BINARY CLASSIFICATION)
    # =========================================================================
    # Single neuron with Sigmoid activation function.
    # Formula: Sigmoid(z) = 1 / (1 + e^(-z))
    # Why Sigmoid? Produces a probability score strictly bounded between 0 and 1.
    # Thresholding: Score < 0.5 -> Class 0 (Cat 🐱), Score >= 0.5 -> Class 1 (Dog 🐶).
    model.add(layers.Dense(units=1, activation='sigmoid', name="output_layer"))

    # =========================================================================
    # COMPILATION
    # =========================================================================
    # Optimizer: Adam (Adaptive Moment Estimation) adjusts learning rate dynamically per weight.
    # Loss: Binary Cross-Entropy (Log Loss) measures difference between target (0/1) and predicted probability.
    # Metrics: Accuracy (percentage of correct predictions).
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_data_augmentation_layer():
    """
    Returns a Keras Sequential layer for real-time data augmentation.
    Data augmentation increases dataset diversity by applying random geometric transformations.
    This helps reduce overfitting without altering validation data.
    """
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),  # Random rotation by +/- 10% (around 36 degrees)
        layers.RandomZoom(0.1),      # Random zoom in/out by 10%
    ], name="data_augmentation")


if __name__ == "__main__":
    # Test model instantiation and print summary
    custom_cnn = build_cat_dog_cnn(input_shape=(128, 128, 3), learning_rate=0.001)
    custom_cnn.summary()
