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
    All convolutions inside the inception block use relu activation.
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters

    conv_1x1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        activation='relu'
    )(conv_1x1)

    conv_1x1_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(conv_3x3)

    add = K.layers.Add()([conv_1x1_2, A_prev])
    out = K.layers.Activation('relu')(add)

    return out
