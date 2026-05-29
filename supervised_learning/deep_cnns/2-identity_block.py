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
    initializer = K.initializers.he_normal(seed=0)

    conv_1x1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)
    batch_norm_1 = K.layers.BatchNormalization(axis=3)(conv_1x1)

    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(batch_norm_1)
    batch_norm_2 = K.layers.BatchNormalization(axis=3)(conv_3x3)

    conv_1x1_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(batch_norm_2)
    batch_norm_3 = K.layers.BatchNormalization(axis=3)(conv_1x1_2)

    add = K.layers.Add()([batch_norm_3, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
