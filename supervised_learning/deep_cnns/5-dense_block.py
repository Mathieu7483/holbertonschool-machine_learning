#!/usr/bin/env python3
"""Write a function that builds a dense block as
described in Densely Connected Convolutional Networks:"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional
    Networks.

    Args:
        X (keras.Input): the input to the dense block.
        nb_filters (int): the number of filters in X.
        growth_rate (int): the growth rate for the dense block.
        layers (int): the number of layers in the dense block.

    Returns:
        tuple: a tuple containing:
            - Y (keras.Tensor): the output of the dense block.
            - nb_filters (int): the number of filters in Y.
    """
    for i in range(layers):
        # Batch Normalization
        Y = K.layers.BatchNormalization(axis=3)(X)
        # ReLU activation
        Y = K.layers.Activation('relu')(Y)
        # 1x1 Convolution
        Y = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                            kernel_initializer=K.initializers.he_normal(seed=0)
                            )(Y)
        # Batch Normalization
        Y = K.layers.BatchNormalization(axis=3)(Y)
        # ReLU activation
        Y = K.layers.Activation('relu')(Y)
        # 3x3 Convolution
        Y = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                            kernel_initializer=K.initializers.he_normal(seed=0)
                            )(Y)
        # Concatenate input and output
        X = K.layers.Concatenate()([X, Y])
        nb_filters += growth_rate

    return X, nb_filters
