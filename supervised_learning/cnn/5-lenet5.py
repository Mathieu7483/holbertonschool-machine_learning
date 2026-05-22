#!/usr/bin/env python3
"""
Write a function that builds a modified version of
the LeNet-5 architecture using keras:
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds the LeNet-5 architecture using Keras

    Args:
        X is a K.Input of shape (m, 28, 28, 1) containing
        the input images for the network
        m is the number of images
        The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
        All layers requiring initialization should initialize
        their kernels with the he_normal initialization method
        The seed for the he_normal initializer should be set
        to zero for each layer to ensure reproducibility.
        All hidden layers requiring activation should
        use the relu activation function
        you may from tensorflow import keras as K
    Returns: a K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    he_init = K.initializers.VarianceScaling(scale=2.0, seed=0)

    # Convolutional Layer 1
    C1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                         kernel_initializer=he_init, activation='relu')(X)

    # Pooling Layer 1
    S2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C1)

    # Convolutional Layer 2
    C3 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                         kernel_initializer=he_init, activation='relu')(S2)

    # Pooling Layer 2
    S4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C3)

    # Flatten layer
    S4_flat = K.layers.Flatten()(S4)

    # Fully Connected Layer 1
    C5 = K.layers.Dense(units=120, kernel_initializer=he_init,
                        activation='relu')(S4_flat)

    # Fully Connected Layer 2
    F6 = K.layers.Dense(units=84, kernel_initializer=he_init,
                        activation='relu')(C5)

    # Output Layer
    y_pred = K.layers.Dense(units=10, kernel_initializer=he_init,
                            activation='softmax')(F6)

    # Model
    model = K.Model(inputs=X, outputs=y_pred)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
