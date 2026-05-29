#!/usr/bin/env python3
"""Write a function that builds an identity block as
described in Deep Residual Learning for Image Recognition (2015)."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in Deep Residual Learning for
    Image Recognition (2015):
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
      F11 is the number of filters in the first 1x1 convolution
      F3 is the number of filters in the 3x3 convolution
      F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters

    # First component of the main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of the main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of the main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
