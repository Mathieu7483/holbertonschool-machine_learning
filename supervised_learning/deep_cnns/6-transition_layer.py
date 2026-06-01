#!/usr/bin/env python3
"""Write a function that builds a transition layer
as described in Densely Connected Convolutional Networks"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional
    Networks.

    Args:
        X (keras.Input): the input to the transition layer.
        nb_filters (int): the number of filters in X.
        compression (float): the compression factor for the transition layer.

    Returns:
        tuple: a tuple containing:
            - Y (keras.Tensor): the output of the transition layer.
            - nb_filters (int): the number of filters in Y.
    """
    # Batch Normalization
    Y = K.layers.BatchNormalization(axis=3)(X)
    # ReLU activation
    Y = K.layers.Activation('relu')(Y)
    # 1x1 Convolution
    Y = K.layers.Conv2D(int(nb_filters * compression), (1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(Y)
    # Average Pooling
    Y = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(Y)

    nb_filters = int(nb_filters * compression)

    return Y, nb_filters
